# universe.py
#
# Build a dynamic trading universe from Kraken.
# Rules:
#  - USD-quoted spot pairs (…USD)
#  - Exclude obvious non-spot/indices/stables
#  - Rank by 24h USD volume (last price * base_volume_24h)
#  - Keep top N (configurable)
#  - Cache results in SQLite so we don’t slam APIs

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import httpx
from sqlmodel import SQLModel, Field, Session, select

from settings import (
    MIN_VOLUME_USD,  # e.g. 20_000_000
    UNIVERSE_TOP_N,  # e.g. 25
    UNIVERSE_EXCLUDE,  # list[str]
    UNIVERSE_INCLUDE,  # list[str]
    UNIVERSE_CACHE_MINUTES,  # e.g. 60
)

# ---------- Storage table (cached universe) ----------
class UniversePair(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str  # e.g. BTC
    pair: str    # e.g. XXBTZUSD or XBTUSD
    base: str
    quote: str
    price: float
    base_vol_24h: float
    usd_vol_24h: float
    updated_at: datetime

# ---------- HTTP client ----------
_client: Optional[httpx.AsyncClient] = None

async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=20.0,
            headers={
                "User-Agent": "AutoPicklr/1.0",
                "Accept": "application/json",
            },
        )
    return _client

# ---------- Kraken helpers ----------
async def _kraken_asset_pairs() -> Dict:
    cli = await _get_client()
    r = await cli.get("https://api.kraken.com/0/public/AssetPairs")
    r.raise_for_status()
    return r.json()

async def _kraken_ticker(pairs: List[str]) -> Dict:
    """Call Kraken Ticker in chunks (Kraken allows long comma-joined lists)."""
    cli = await _get_client()
    out_result = {}
    CHUNK = 25
    for i in range(0, len(pairs), CHUNK):
        chunk = pairs[i:i+CHUNK]
        r = await cli.get("https://api.kraken.com/0/public/Ticker", params={"pair": ",".join(chunk)})
        r.raise_for_status()
        j = r.json()
        if j.get("error"):
            continue
        out_result.update(j.get("result", {}))
    return {"result": out_result}

from settings import ALLOWED_QUOTES

def _is_usd_pair(pair_info: Dict) -> bool:
    # Prefer wsname (e.g., "ADA/USDT", "SOL/USD")
    ws = pair_info.get("wsname")
    if isinstance(ws, str) and "/" in ws:
        quote = ws.split("/")[-1].upper()
        return quote in ALLOWED_QUOTES
    # Fallback: use Kraken raw "quote" like "ZUSD" or "USDT"
    quote_raw = (pair_info.get("quote") or "").upper()
    if quote_raw in ("ZUSD", "USD") and "USD" in ALLOWED_QUOTES:
        return True
    if quote_raw == "USDT" and "USDT" in ALLOWED_QUOTES:
        return True
    return False

def _extract_base_symbol(pair_info: Dict) -> Optional[str]:
    # Use wsname if present (e.g. "BTC/USD" -> "BTC")
    ws = pair_info.get("wsname")
    if isinstance(ws, str) and "/" in ws:
        base = ws.split("/")[0]
        # Normalize Kraken aliases (XBT -> BTC)
        if base.upper() == "XBT":
            return "BTC"
        return base.upper()
    # fallback: "base" is like "XXBT"; normalize
    base = pair_info.get("base")
    if isinstance(base, str):
        if base.upper() in ("XXBT", "XBT"):
            return "BTC"
        return base.replace("X", "").replace("Z", "").upper()  # crude normalization
    return None

def _extract_pair_key(pair_key: str, pair_info: Dict) -> str:
    """Return the ticker key we must use for /Ticker response mapping."""
    # Kraken’s Ticker result uses the pair key as returned by AssetPairs keys (e.g. "XXBTZUSD", "XETHZUSD")
    return pair_key

def _clean_exclusions(sym: str) -> bool:
    # exclude obvious non-spot tokens & stables by symbol
    stables = {"USD", "USDT", "USDC", "DAI", "EUR", "GBP"}
    if sym in stables:
        return False
    # user exclusions
    if sym in (x.upper() for x in UNIVERSE_EXCLUDE):
        return False
    return True

# ---------- Public: refresh & get universe ----------
async def refresh_universe(session: Session) -> List[UniversePair]:
    """
    Build list from Kraken:
      1) AssetPairs -> pick USD-quoted
      2) Ticker -> gather last price + 24h volume
      3) Rank by USD volume; keep top N, apply MIN_VOLUME_USD
      4) Cache to DB
    Returns the final list of UniversePair rows.
    """
    # 1) AssetPairs
    pairs_json = await _kraken_asset_pairs()
    if pairs_json.get("error"):
        print("[universe] Kraken AssetPairs error:", pairs_json["error"])
        return []

    pairs_raw = pairs_json.get("result", {})
    usd_pairs: Dict[str, Dict] = {k: v for k, v in pairs_raw.items() if _is_usd_pair(v)}
    print(f"[universe] USD/USDT candidate pairs: {len(usd_pairs)}")   # <-- move here

    # Build mapping for ticker call and base symbols
    pair_keys = list(usd_pairs.keys())
    base_map: Dict[str, Tuple[str, Dict]] = {}  # base_sym -> (pair_key, pair_info)
    for pk in pair_keys:
        info = usd_pairs[pk]
        base = _extract_base_symbol(info)
        if not base:
            continue
        if not _clean_exclusions(base):
            continue
        # prefer first occurrence; later we could dedup (e.g. "BTC/USD" vs "BTC/USD.m")
        if base not in base_map:
            base_map[base] = (pk, info)

    # Always include user-specified tickers if present in USD pairs
    for inc in (x.upper() for x in UNIVERSE_INCLUDE):
        # find any pair that matches this base
        for pk, info in usd_pairs.items():
            base = _extract_base_symbol(info)
            if base == inc:
                base_map[inc] = (pk, info)
                break
            print(f"[universe] USD/USDT candidate pairs: {len(usd_pairs)}")

    # 2) Ticker for all selected pair keys
    ticker_keys = [pk for pk, _ in base_map.values()]
    if not ticker_keys:
        print("[universe] No USD pairs found on Kraken.")
        return []

    tick_json = await _kraken_ticker(ticker_keys)
    tick = tick_json.get("result", {})

    rows: List[UniversePair] = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    for base_sym, (pair_key, info) in base_map.items():
        tk_key = _extract_pair_key(pair_key, info)
        tk = tick.get(tk_key)
        if not tk:
            # sometimes Kraken returns a slightly different key; try a heuristic scan
            alt = next((v for k, v in tick.items() if k.endswith("USD") and base_sym in k), None)
            if not alt:
                continue
            tk = alt

        # Kraken Ticker fields:
        #  c: [close, lot_volume]
        #  v: [today_volume_base, last24h_volume_base]
        #  p: [vwap_today, vwap_24h]  (not used)
        #  h/l: high/low today & last24h (not used)
        close = float(tk["c"][0]) if tk.get("c") else None
        vol24 = float(tk["v"][1]) if tk.get("v") else 0.0
        if not close or close <= 0.0:
            continue

        usd_vol24 = close * vol24
        if usd_vol24 < MIN_VOLUME_USD:
            continue

        # Detect quote from wsname when present
        ws = info.get("wsname")
        quote = "USD"
        if isinstance(ws, str) and "/" in ws:
            quote = ws.split("/")[-1].upper()

        # If quote is USDT, we’ll treat USDT≈USD for ranking; that's fine for filtering.
        # (If you later add EUR, we’d convert using a spot FX.)

        rows.append(UniversePair(
            symbol=base_sym,
            pair=pair_key,
            base=base_sym,
            quote=quote,  # now USD or USDT
            price=close,
            base_vol_24h=vol24,
            usd_vol_24h=usd_vol24,  # for USDT this is ~USD
            updated_at=now,
        ))
        print(f"[universe] Passed volume filter (min ${MIN_VOLUME_USD:,.0f}): {len(rows)}")  # <-- move here

    # 3) Rank & cap
    rows.sort(key=lambda r: r.usd_vol_24h, reverse=True)
    if UNIVERSE_TOP_N > 0 and len(rows) > UNIVERSE_TOP_N:
        rows = rows[:UNIVERSE_TOP_N]

    # 4) Save to DB (replace all)
    # Clear old
    old = session.exec(select(UniversePair)).all()
    for r in old:
        session.delete(r)
    for r in rows:
        session.add(r)
    session.commit()

    print(f"[universe] Refreshed {len(rows)} USD pairs from Kraken (min ${MIN_VOLUME_USD:,.0f}, top {UNIVERSE_TOP_N}).")
    return rows

def get_active_universe(session: Session) -> List[str]:
    """
    Return cached universe if fresh; otherwise an empty list (caller may trigger refresh).
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = session.exec(select(UniversePair).order_by(UniversePair.usd_vol_24h.desc())).all()
    if not rows:
        return []
    age = now - rows[0].updated_at
    if age > timedelta(minutes=UNIVERSE_CACHE_MINUTES):
        # stale -> caller can decide to refresh
        return []
    return [r.symbol for r in rows]
