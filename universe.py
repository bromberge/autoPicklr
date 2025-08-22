# universe.py
#
# Build a dynamic trading universe from Kraken.
# Rules:
#  - USD / USDT-quoted spot pairs
#  - Exclude obvious non-spot/indices/stables
#  - Rank by 24h USD volume (last price * base_volume_24h)
#  - Keep top N (configurable)
#  - Cache results in SQLite so we don’t slam APIs

from __future__ import annotations
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import httpx
from sqlmodel import SQLModel, Field, Session, select

from settings import (
    MIN_VOLUME_USD,          # e.g. 1_000_000
    UNIVERSE_TOP_N,          # e.g. 150
    UNIVERSE_EXCLUDE,        # list[str]
    UNIVERSE_INCLUDE,        # list[str]
    UNIVERSE_CACHE_MINUTES,  # e.g. 60
)
# We import ALLOWED_QUOTES here to keep quoting logic in one place.
from settings import ALLOWED_QUOTES

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

# ---------- Pair helpers ----------
def _is_usd_pair(pair_info: dict) -> bool:
    ws = pair_info.get("wsname")
    if isinstance(ws, str) and "/" in ws:
        quote = ws.split("/")[-1].upper()
        return quote in ALLOWED_QUOTES
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
    # Kraken’s Ticker result uses the AssetPairs key (e.g. "XXBTZUSD", "XETHZUSD")
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

# ---------- Universe staleness helper ----------
def universe_stale(session: Session) -> bool:
    """
    Returns True if the cached universe is empty or older than UNIVERSE_CACHE_MINUTES.
    """
    row = session.exec(
        select(UniversePair).order_by(UniversePair.updated_at.desc())
    ).first()
    if not row:
        return True
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    age = now - row.updated_at
    return age >= timedelta(minutes=UNIVERSE_CACHE_MINUTES)

# ---------- “Last known good” fallback ----------
_LAST_GOOD: List[str] = []

# --- Ensure mapping for specific symbols (force-include) ---
from typing import List

async def ensure_pairs_for(session: Session, symbols: List[str]) -> List[str]:
    """
    Ensure UniversePair rows exist for the given base symbols (e.g., ['API3','XMR']).
    - Looks up USD/USDT Kraken pairs for those bases
    - Fetches a ticker for price/volume
    - Inserts rows for any that were missing (ignores min-volume/top-N caps)
    Returns the list of symbols that were added.
    """
    if not symbols:
        return []

    # Which symbols are missing right now?
    existing = session.exec(select(UniversePair).where(UniversePair.symbol.in_(symbols))).all()
    have = {r.symbol for r in existing}
    need = [s for s in symbols if s not in have]
    if not need:
        return []

    # Scan Kraken asset pairs, keep only USD/USDT-quoted
    pairs_json = await _kraken_asset_pairs()
    pairs_raw = pairs_json.get("result", {})
    wanted = {}  # base -> (pair_key, pair_info)
    for pk, info in pairs_raw.items():
        if not _is_usd_pair(info):
            continue
        base = _extract_base_symbol(info)
        if base in need and base not in wanted:
            wanted[base] = (pk, info)

    if not wanted:
        return []

    # Fetch ticker for the wanted pairs
    ticker_keys = [pk for pk, _ in wanted.values()]
    tick_json = await _kraken_ticker(ticker_keys)
    tick = tick_json.get("result", {})

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    added: List[str] = []

    for base_sym, (pair_key, info) in wanted.items():
        tk = tick.get(_extract_pair_key(pair_key, info))
        if not tk:
            continue
        close = float(tk["c"][0]) if tk.get("c") else None
        v_today = float(tk["v"][0]) if tk.get("v") else 0.0
        v_24h   = float(tk["v"][1]) if tk.get("v") else 0.0
        vol24 = v_24h if v_24h > 0 else v_today
        if not close or close <= 0:
            continue

        ws = info.get("wsname")
        quote = "USD"
        if isinstance(ws, str) and "/" in ws:
            quote = ws.split("/")[-1].upper()

        session.add(UniversePair(
            symbol=base_sym,
            pair=pair_key,
            base=base_sym,
            quote=quote,
            price=close,
            base_vol_24h=vol24,
            usd_vol_24h=close * vol24,
            updated_at=now,
        ))
        added.append(base_sym)

    if added:
        session.commit()
        print(f"[universe] Force-included pairs for open positions: {added}")

    return added

# ---------- Public: refresh & get universe ----------
async def refresh_universe(session: Session) -> List[UniversePair]:
    """
    Build list from Kraken:
      1) AssetPairs -> pick USD/USDT quoted
      2) Ticker -> last price + 24h volume (fallback to today's volume if 24h is zero)
      3) Rank by USD volume; keep top N, apply MIN_VOLUME_USD
      4) Cache to DB
    """
    # 1) AssetPairs
    pairs_json = await _kraken_asset_pairs()
    if pairs_json.get("error"):
        print("[universe] Kraken AssetPairs error:", pairs_json["error"])
        return []

    pairs_raw = pairs_json.get("result", {})
    usd_pairs: Dict[str, Dict] = {k: v for k, v in pairs_raw.items() if _is_usd_pair(v)}

    # DEBUG: show candidate count + the settings actually in use
    print(f"[universe] USD/USDT candidate pairs: {len(usd_pairs)}")
    print(f"[universe] Using MIN_VOLUME_USD={MIN_VOLUME_USD:,.0f}, ALLOWED_QUOTES={ALLOWED_QUOTES}, TOP_N={UNIVERSE_TOP_N}")

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
        # prefer first occurrence
        if base not in base_map:
            base_map[base] = (pk, info)

    # Always include user-specified tickers if present in USD/USDT pairs
    for inc in (x.upper() for x in UNIVERSE_INCLUDE):
        for pk, info in usd_pairs.items():
            base = _extract_base_symbol(info)
            if base == inc:
                base_map[inc] = (pk, info)
                break

    # 2) Ticker for all selected pair keys
    ticker_keys = [pk for pk, _ in base_map.values()]
    if not ticker_keys:
        print("[universe] No USD/USDT pairs found on Kraken.")
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
            alt = next((v for k, v in tick.items() if (k.endswith("USD") or k.endswith("USDT")) and base_sym in k), None)
            if not alt:
                continue
            tk = alt

        # Kraken Ticker fields:
        #  c: [close, lot_volume]
        #  v: [today_volume_base, last24h_volume_base]
        close = float(tk["c"][0]) if tk.get("c") else None
        vol24 = 0.0
        if tk.get("v"):
            # prefer 24h base volume; fallback to today's base volume if 24h is zero
            v_today = float(tk["v"][0])
            v_24h   = float(tk["v"][1])
            vol24 = v_24h if v_24h > 0 else v_today

        if not close or close <= 0.0:
            continue

        usd_vol24 = close * vol24
        if usd_vol24 < MIN_VOLUME_USD:
            continue

        # Detect quote from wsname when present (USD or USDT)
        ws = info.get("wsname")
        quote = "USD"
        if isinstance(ws, str) and "/" in ws:
            quote = ws.split("/")[-1].upper()

        rows.append(UniversePair(
            symbol=base_sym,
            pair=pair_key,
            base=base_sym,
            quote=quote,  # USD or USDT
            price=close,
            base_vol_24h=vol24,
            usd_vol_24h=usd_vol24,  # for USDT this ~USD
            updated_at=now,
        ))

    # DEBUG: print AFTER building the list (not inside the loop)
    print(f"[universe] Passed volume filter (min ${MIN_VOLUME_USD:,.0f}): {len(rows)}")

    # 3) Rank & cap
    rows.sort(key=lambda r: r.usd_vol_24h, reverse=True)
    if UNIVERSE_TOP_N > 0 and len(rows) > UNIVERSE_TOP_N:
        rows = rows[:UNIVERSE_TOP_N]

    # 4) Save to DB (replace all)
    old = session.exec(select(UniversePair)).all()
    for r in old:
        session.delete(r)
    for r in rows:
        session.add(r)
    session.commit()

    print(f"[universe] Refreshed {len(rows)} USD/USDT pairs from Kraken (min ${MIN_VOLUME_USD:,.0f}, top {UNIVERSE_TOP_N}).")

    # remember latest good symbols
    global _LAST_GOOD
    _LAST_GOOD = [r.symbol for r in rows]
    return rows

def get_active_universe(session: Session) -> List[str]:
    """
    Return cached universe. If the cache is stale, we still return the current
    DB rows (so we don't collapse to the static UNIVERSE). Fall back to the last
    good list, then to settings.UNIVERSE only if the table is empty.
    """
    global _LAST_GOOD  # <-- declare at the very top of the function
    from settings import UNIVERSE  # local import to avoid surprises

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = session.exec(select(UniversePair).order_by(UniversePair.usd_vol_24h.desc())).all()

    if not rows:
        # DB empty: try LAST_GOOD, then settings.UNIVERSE
        if _LAST_GOOD:
            print(f"[universe] fallback to LAST_GOOD ({len(_LAST_GOOD)})")
            return _LAST_GOOD[:]
        print(f"[universe] fallback to settings.UNIVERSE {UNIVERSE}")
        return UNIVERSE[:]

    # even if stale, return rows (do NOT return [])
    age = now - rows[0].updated_at
    if age > timedelta(minutes=UNIVERSE_CACHE_MINUTES):
        print(f"[universe] cache is stale by {age}, returning STALE rows (not empty)")
        syms = [r.symbol for r in rows]
        _LAST_GOOD = syms[:]
        return syms

    syms = [r.symbol for r in rows]
    _LAST_GOOD = syms[:]
    return syms

# ---------- Optional: debug snapshot ----------
def universe_debug_snapshot(session: Session) -> dict:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=UNIVERSE_CACHE_MINUTES)
    all_rows = session.exec(select(UniversePair)).all()
    fresh = [r for r in all_rows if r.updated_at >= cutoff]
    quote_ok = [r for r in all_rows if (r.quote or "").upper() in ALLOWED_QUOTES]
    vol_ok = [r for r in quote_ok if float(r.usd_vol_24h or 0.0) >= float(MIN_VOLUME_USD)]
    return {
        "rows_total": len(all_rows),
        "rows_fresh": len(fresh),
        "quote_ok": len(quote_ok),
        "vol_ok": len(vol_ok),
        "top_n": UNIVERSE_TOP_N,
        "cache_minutes": UNIVERSE_CACHE_MINUTES,
        "last_good": len(_LAST_GOOD),
    }
