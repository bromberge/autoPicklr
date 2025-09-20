# data.py

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import httpx
from sqlmodel import Session, select

from models import Candle
from settings import (
    UNIVERSE, HISTORY_MINUTES, USE_COINGECKO,
    COINGECKO_API_KEY, DATA_FETCH_INTERVAL_SECONDS
)
from universe import UniversePair  # so we can read the cached pair for each symbol

# Fallback pair map if the universe cache doesn’t have an entry for a symbol yet.
# (These are common Kraken USD pairs; safe to extend later.)
KRAKEN_FALLBACK_PAIRS = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
    "XDG": "XDGUSD",   # Kraken’s DOGE ticker is XDG
    "DOGE": "XDGUSD",
    "XRP": "XRPUSD",
    "LINK": "LINKUSD",
    "ADA": "ADAUSD",
    "LTC": "LTCUSD",
    "DOT": "DOTUSD",
    "AVAX": "AVAXUSD",
    "MATIC": "MATICUSD",
}


# ----- HTTP client & timing -----
_client: Optional[httpx.AsyncClient] = None
_last_fetch_at: Dict[str, float] = {}  # per-symbol throttle (epoch seconds)

def _now_s() -> float:
    return datetime.now(tz=timezone.utc).timestamp()

def _headers() -> Dict[str, str]:
    h = {
        "User-Agent": "AutoPicklr/1.0",
        "Accept": "application/json",
    }
    # CoinGecko key is only used for /simple (spot) helper below
    if COINGECKO_API_KEY:
        h["x-cg-pro-api-key"] = COINGECKO_API_KEY
        h["x-cg-demo-api-key"] = COINGECKO_API_KEY
    return h

async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=20.0, headers=_headers())
    return _client

# ----- DB helpers -----
def _last_candle_ts(session: Session, symbol: str) -> Optional[datetime]:
    row = session.exec(
        select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.desc())
    ).first()
    return row.ts if row else None

def _kraken_pick_key(pair: str, result: Dict[str, list]) -> Optional[str]:
    """
    Kraken's OHLC result keys can be 'XBTUSD', 'XXBTZUSD', 'XETHZUSD', etc.
    Pick the right key for a requested pair.
    """
    # Exact match first
    if pair in result:
        return pair

    # Known aliases
    aliases = {
        "XBTUSD": ["XXBTZUSD", "XBTUSD"],
        "ETHUSD": ["XETHZUSD", "ETHUSD"],
        "SOLUSD": ["SOLUSD"],  # usually exact
    }
    for k in aliases.get(pair, []):
        if k in result:
            return k

    # Heuristic: any key that ends with USD and contains base (XBT/ETH/SOL)
    base = pair[:-3]  # 'XBT', 'ETH', 'SOL'
    for k in result.keys():
        if k.endswith("USD") and base in k:
            return k

    # Fallback: first non-"last" key
    for k in result.keys():
        if k != "last":
            return k

    return None

def _pair_for_symbol(session: Session, symbol: str) -> Optional[str]:
    """Prefer the pair cached by the dynamic universe; fall back to a small static map."""
    sym = symbol.upper()
    row = session.exec(select(UniversePair).where(UniversePair.symbol == sym)).first()
    if row and row.pair:
        return row.pair
    return KRAKEN_FALLBACK_PAIRS.get(sym)


# ----- Kraken 1m OHLCV (no key) -----
# Kraken uses XBTUSD for BTC/USD
KRAKEN_PAIRS: Dict[str, str] = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
    # add more as you expand UNIVERSE, e.g. "ADA": "ADAUSD"
}
# Kraken intervals: 1,5,15,30,60,...
KRAKEN_INTERVAL = 1  # minute

async def _kraken_fetch_ohlc(pair: str, interval: int = 1, since_s: Optional[int] = None) -> Dict:
    """
    GET https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=60&since=UNIX_SECONDS
    Returns {"error":[],"result":{"PAIR":[...],"last": <cursor>}}
    """
    cli = await _get_client()
    params = {"pair": pair, "interval": str(interval)}
    if since_s is not None:
        params["since"] = str(int(since_s))
    r = await cli.get("https://api.kraken.com/0/public/OHLC", params=params)
    r.raise_for_status()
    return r.json()

async def _upsert_minute_candles(session: Session, symbol: str) -> int:
    # Per-symbol throttle
    now = _now_s()
    last = _last_fetch_at.get(symbol, 0.0)
    if now - last < DATA_FETCH_INTERVAL_SECONDS:
        return 0
    _last_fetch_at[symbol] = now

    pair = KRAKEN_PAIRS.get(symbol.upper())
    if not pair:
        print(f"[data] No Kraken pair mapped for {symbol}, skip.")
        return 0

    try:
        data = await _kraken_fetch_ohlc(pair, interval=KRAKEN_INTERVAL)
    except Exception as e:
        print(f"[data] Kraken OHLC error for {symbol} ({pair}): {e}")
        return 0

    if data.get("error"):
        print(f"[data] Kraken returned error for {symbol}: {data['error']}")
        return 0

    # Kraken nests result under the exact pair key (sometimes with suffix)
    result = data.get("result", {})
    key = _kraken_pick_key(pair, result)
    if not key:
        print(f"[data] Kraken missing pair key for {symbol} ({pair}) in result. Keys: {list(result.keys())}")
        return 0


    ohlc = result.get(key, [])
    if not ohlc:
        print(f"[data] Kraken returned empty OHLC for {symbol} ({pair})")
        return 0

    last_ts = _last_candle_ts(session, symbol)
    min_allowed = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(minutes=HISTORY_MINUTES)
    min_allowed = min_allowed.replace(tzinfo=None)

    rows_added = 0
    for row in ohlc:
        # row => [time, open, high, low, close, vwap, volume, count]
        ts = datetime.fromtimestamp(int(row[0]), tz=timezone.utc).replace(tzinfo=None)
        open_px = float(row[1]); high_px = float(row[2]); low_px = float(row[3]); close_px = float(row[4])
        vol = float(row[6])

        if last_ts and ts <= last_ts:
            continue
        if ts < min_allowed:
            continue

        exists = session.exec(
            select(Candle).where(Candle.symbol == symbol, Candle.ts == ts)
        ).first()
        if exists:
            continue

        session.add(Candle(
            symbol=symbol,
            ts=ts,
            open=open_px,
            high=high_px,
            low=low_px,
            close=close_px,
            volume=vol,
        ))
        rows_added += 1

    if rows_added:
        session.commit()
        last_row = ohlc[-1]
        last_ts_str = datetime.fromtimestamp(int(last_row[0]), tz=timezone.utc).replace(tzinfo=None).isoformat()
        last_close = float(last_row[4])
        print(f"[data] {symbol}: +{rows_added} bars from Kraken (last {last_ts_str} close {last_close:.4f})")

    return rows_added

async def _upsert_ohlc_rows(session: Session, symbol: str, rows: list) -> int:
    """
    Insert OHLC rows into Candle table if missing.
    Each Kraken row: [time, open, high, low, close, vwap, volume, count]
    We store ts at the bar-close time (UTC, naive), and keep volume (col 6).
    """
    if not rows:
        return 0

    from datetime import timezone
    from sqlmodel import select

    rows_added = 0
    for row in rows:
        ts = datetime.fromtimestamp(int(row[0]), tz=timezone.utc).replace(tzinfo=None)
        open_px = float(row[1]); high_px = float(row[2]); low_px = float(row[3]); close_px = float(row[4])
        vol = float(row[6])

        exists = session.exec(
            select(Candle).where(Candle.symbol == symbol, Candle.ts == ts)
        ).first()
        if exists:
            continue

        session.add(Candle(
            symbol=symbol,
            ts=ts,
            open=open_px,
            high=high_px,
            low=low_px,
            close=close_px,
            volume=vol,
        ))
        rows_added += 1

    if rows_added:
        session.commit()
    return rows_added


async def backfill_hourly_symbol(session: Session, symbol: str, pair: str, days: int = 365) -> int:
    """
    Paginate Kraken hourly OHLC (interval=60) using the 'since' cursor until we reach 'now'.
    Returns total new bars inserted for this symbol.
    """
    from time import sleep

    INTERVAL = 60  # hourly bars
    # start from (now - days)
    start_dt = datetime.utcnow() - timedelta(days=int(days))
    start_s = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
    end_s = int(datetime.utcnow().replace(tzinfo=timezone.utc).timestamp())

    total_added = 0
    since_s = start_s
    SAFETY_MAX_LOOPS = 2000  # plenty for 365 days hourly
    loops = 0

    while since_s < end_s and loops < SAFETY_MAX_LOOPS:
        loops += 1
        try:
            data = await _kraken_fetch_ohlc(pair, interval=INTERVAL, since_s=since_s)
        except Exception as e:
            print(f"[backfill] OHLC error {symbol}/{pair} since={since_s}: {e}")
            break

        if data.get("error"):
            print(f"[backfill] Kraken error {symbol}: {data['error']}")
            break

        result = data.get("result", {})
        key = _kraken_pick_key(pair, result)
        if not key:
            # sometimes Kraken uses an alternate key; try heuristic
            key = next((k for k in result.keys() if k != "last"), None)
            if not key:
                print(f"[backfill] missing pair key for {symbol} ({pair}); keys={list(result.keys())}")
                break

        ohlc = result.get(key, [])
        last_cursor = int(result.get("last", 0))  # seconds since epoch (Kraken doc)

        added = await _upsert_ohlc_rows(session, symbol, ohlc)
        total_added += added

        # Advance the cursor. If Kraken gave no progress, stop.
        if not last_cursor or last_cursor <= since_s:
            break
        since_s = last_cursor

        # Gentle pacing to avoid hammering (Kraken is tolerant, but be nice)
        sleep(0.25)

    if total_added:
        print(f"[backfill] {symbol}: +{total_added} hourly bars ({days}d window)")
    else:
        print(f"[backfill] {symbol}: no new hourly bars ({days}d window)")
    return total_added


async def backfill_hourly(session: Session, symbols: Optional[list] = None, days: int = 365) -> dict:
    """
    Backfill hourly OHLC for many symbols (uses UniversePair mapping when available).
    If symbols is None, uses the cached universe (fast) or your static UNIVERSE.
    """
    # Decide the working list
    if not symbols:
        # prefer cached universe so we know Kraken pair names
        rows = session.exec(select(UniversePair)).all()
        if rows:
            syms = [r.symbol for r in rows]
            sym_to_pair = {r.symbol: r.pair for r in rows if r.pair}
        else:
            from settings import UNIVERSE
            syms = UNIVERSE[:]
            sym_to_pair = {}
    else:
        syms = [s.upper().strip() for s in symbols]
        # try read pairs for those
        rows = session.exec(select(UniversePair).where(UniversePair.symbol.in_(syms))).all()
        sym_to_pair = {r.symbol: r.pair for r in rows if r.pair}

    # Fallback for any symbol missing a mapped pair
    def _pair(sym: str) -> Optional[str]:
        p = sym_to_pair.get(sym)
        if p:
            return p
        p = _pair_for_symbol(session, sym)  # uses fallback map if not cached
        return p

    total = 0
    errors = 0
    for sym in syms:
        pair = _pair(sym)
        if not pair:
            print(f"[backfill] skip {sym}: no Kraken pair")
            errors += 1
            continue
        try:
            total += await backfill_hourly_symbol(session, sym, pair, days=days)
        except Exception as e:
            print(f"[backfill] error {sym}: {e}")
            errors += 1

    return {"symbols": len(syms), "bars": total, "errors": errors}


async def _upsert_minute_candles_pair(session: Session, symbol: str, pair: str) -> int:
    # Per-symbol throttle
    now = _now_s()
    last = _last_fetch_at.get(symbol, 0.0)
    if now - last < DATA_FETCH_INTERVAL_SECONDS:
        return 0
    _last_fetch_at[symbol] = now

    try:
        data = await _kraken_fetch_ohlc(pair, interval=KRAKEN_INTERVAL)
    except Exception as e:
        print(f"[data] Kraken OHLC error for {symbol} ({pair}): {e}")
        return 0

    if data.get("error"):
        print(f"[data] Kraken returned error for {symbol}: {data['error']}")
        return 0

    result = data.get("result", {})
    key = _kraken_pick_key(pair, result)  # always use the helper
    if not key:
        # accept USD or USDT and match the base (e.g., XBT/ETH/…)
        base = pair[:-3]
        key = next(
            (k for k in result.keys()
             if (k.endswith("USD") or k.endswith("USDT")) and base in k and k != "last"),
            None
        )

    if not key:
        print(f"[data] Kraken missing pair key for {symbol} ({pair}) in result. Keys: {list(result.keys())}")
        return 0

    ohlc = result.get(key, [])
    if not ohlc:
        print(f"[data] Kraken returned empty OHLC for {symbol} ({pair})")
        return 0

    last_ts = _last_candle_ts(session, symbol)
    min_allowed = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(minutes=HISTORY_MINUTES)
    min_allowed = min_allowed.replace(tzinfo=None)

    rows_added = 0
    for row in ohlc:
        ts = datetime.fromtimestamp(int(row[0]), tz=timezone.utc).replace(tzinfo=None)
        open_px = float(row[1]); high_px = float(row[2]); low_px = float(row[3]); close_px = float(row[4])
        vol = float(row[6])

        if last_ts and ts <= last_ts:
            continue
        if ts < min_allowed:
            continue

        exists = session.exec(
            select(Candle).where(Candle.symbol == symbol, Candle.ts == ts)
        ).first()
        if exists:
            continue

        session.add(Candle(
            symbol=symbol,
            ts=ts,
            open=open_px,
            high=high_px,
            low=low_px,
            close=close_px,
            volume=vol,
        ))
        rows_added += 1

    if rows_added > 0:
        session.commit()
        last_row = ohlc[-1]
        last_ts_str = datetime.fromtimestamp(int(last_row[0]), tz=timezone.utc).replace(tzinfo=None).isoformat()
        last_close = float(last_row[4])
        print(f"[data] {symbol}: +{rows_added} bars from Kraken (last {last_ts_str} close {last_close:.4f})")
    else:
        kr_last_ts = datetime.fromtimestamp(int(ohlc[-1][0]), tz=timezone.utc).replace(tzinfo=None)
        print(f"[data] {symbol}: 0 new bars (db last={last_ts}, kraken last={kr_last_ts})")

    return rows_added


# ----- Public entry used by your loop (/admin/tick etc.) -----

async def update_candles(session: Session):
    """Fetch 1m OHLCV for each symbol (Kraken)."""
    tasks = [asyncio.create_task(_upsert_minute_candles(session, sym)) for sym in UNIVERSE]
    await asyncio.sleep(0.15)  # small stagger
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for sym, res in zip(UNIVERSE, results):
        if isinstance(res, Exception):
            print(f"[data] exception for {sym}: {res}")

async def update_candles_for(session: Session, symbols: List[str]):
    """Update candles only for the given list of symbols using their Kraken pairs."""
    tasks = []
    for sym in symbols:
        pair = _pair_for_symbol(session, sym)
        if not pair:
            print(f"[data] No Kraken pair for {sym}; skipping.")
            continue
        tasks.append(asyncio.create_task(_upsert_minute_candles_pair(session, sym, pair)))
    if not tasks:
        return
    await asyncio.gather(*tasks, return_exceptions=True)


# ----- Optional: CoinGecko spot for quick comparison -----

CG_IDS: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}
_id_cache_checked: Dict[str, bool] = {}

async def _cg_lookup_id(symbol: str) -> Optional[str]:
    sym = symbol.upper()
    if sym in CG_IDS:
        return CG_IDS[sym]
    if _id_cache_checked.get(sym):
        return CG_IDS.get(sym)

    if not USE_COINGECKO:
        _id_cache_checked[sym] = True
        return None

    cli = await _get_client()
    try:
        r = await cli.get("https://api.coingecko.com/api/v3/search", params={"query": sym})
        r.raise_for_status()
        data = r.json()
        for coin in data.get("coins", []):
            if coin.get("symbol", "").upper() == sym and coin.get("id"):
                CG_IDS[sym] = coin["id"]
                _id_cache_checked[sym] = True
                return coin["id"]
        if data.get("coins"):
            CG_IDS[sym] = data["coins"][0]["id"]
    except Exception as e:
        print(f"[data] CoinGecko /search error for {sym}: {e}")

    _id_cache_checked[sym] = True
    return CG_IDS.get(sym)

async def cg_simple_prices(symbols: List[str]) -> Dict[str, float]:
    if not USE_COINGECKO:
        return {}
    cli = await _get_client()
    ids = []
    sym_to_id = {}
    for s in symbols:
        cid = await _cg_lookup_id(s)
        sym_to_id[s] = cid
        if cid:
            ids.append(cid)

    print(f"[data] cg_simple lookup ids: {sym_to_id}")

    if not ids:
        return {}

    try:
        r = await cli.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(ids), "vs_currencies": "usd"},
        )
        print("[data] /simple/price status", r.status_code)
        j = r.json()
        out = {}
        id_to_sym = {v: k for k, v in CG_IDS.items()}
        for cid, obj in j.items():
            sym = id_to_sym.get(cid, cid).upper()
            out[sym] = float(obj.get("usd", 0.0))
        return out
    except Exception as e:
        print(f"[data] simple/price error: {e}")
        return {}

# ----- Debug helpers exposed to routes -----

def last_n_candles(session: Session, symbol: str, n: int = 5) -> List[Dict]:
    rows = session.exec(
        select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.desc())
    ).all()[:n]
    return [
        {
            "ts": r.ts.isoformat(),
            "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume
        } for r in rows
    ]

# =======================
# Kraken HOURLY backfill
# =======================
from settings import (
    BACKFILL_DAYS_DEFAULT, BACKFILL_CONCURRENCY, BACKFILL_PAUSE_MS, BACKFILL_MAX_PAIRS,
    ALLOWED_QUOTES,
)
import math
import asyncio

async def _kraken_ohlc_page(pair: str, interval: int = 60, since: Optional[int] = None) -> Dict:
    """
    One page of OHLC from Kraken public OHLC.
    Params:
      - pair: e.g. 'XBTUSD' / 'XXBTZUSD' / 'ETHUSD'
      - interval: 60 = hourly
      - since: unix seconds (optional). Kraken returns a 'last' cursor for the next call.
    """
    cli = await _get_client()
    params = {"pair": pair, "interval": str(interval)}
    if since is not None:
        params["since"] = str(int(since))
    r = await cli.get("https://api.kraken.com/0/public/OHLC", params=params)
    r.raise_for_status()
    return r.json()

def _pick_result_key_for_pair(pair: str, result: Dict[str, list]) -> Optional[str]:
    # Exact hit
    if pair in result:
        return pair
    # Common aliases (Kraken can vary keys)
    aliases = {
        "XBTUSD": ["XXBTZUSD", "XBTUSD"],
        "ETHUSD": ["XETHZUSD", "ETHUSD"],
    }
    for k in aliases.get(pair, []):
        if k in result:
            return k
    # Heuristic fallback
    base = pair[:-3]
    for k in result.keys():
        if k != "last" and (k.endswith("USD") or k.endswith("USDT")) and base in k:
            return k
    # Final fallback: first non-'last'
    for k in result.keys():
        if k != "last":
            return k
    return None

def _unix_from_dt(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

async def _upsert_ohlc_rows(session: Session, symbol: str, rows: list) -> int:
    """
    Insert OHLC rows into Candle table if missing (ts = hour close).
    rows: [[ts, open, high, low, close, vwap, volume, count], ...]
    """
    if not rows:
        return 0
    # find DB last ts once
    last_ts = _last_candle_ts(session, symbol)

    added = 0
    for row in rows:
        ts = datetime.fromtimestamp(int(row[0]), tz=timezone.utc).replace(tzinfo=None)
        if last_ts and ts <= last_ts:
            continue
        # skip obviously bad bars
        try:
            o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4]); v = float(row[6])
        except Exception:
            continue
        if c <= 0:
            continue

        exists = session.exec(
            select(Candle).where(Candle.symbol == symbol, Candle.ts == ts)
        ).first()
        if exists:
            continue

        session.add(Candle(
            symbol=symbol, ts=ts, open=o, high=h, low=l, close=c, volume=v
        ))
        added += 1

    if added:
        session.commit()
    return added

async def _discover_usd_pairs() -> Dict[str, str]:
    """
    Return a mapping {BASE: PAIR_KEY} for ALL Kraken USD/USDT spot pairs.
    Uses Kraken public /AssetPairs and prefers USD then USDT when duplicates exist.
    """
    cli = await _get_client()
    r = await cli.get("https://api.kraken.com/0/public/AssetPairs")
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        print("[backfill] AssetPairs error:", j["error"])
        return {}

    result = j.get("result", {})
    by_base = {}
    for pk, info in result.items():
        ws = info.get("wsname") or ""
        if "/" not in ws:
            continue
        base, quote = ws.split("/")
        base = base.upper()
        quote = quote.upper()
        if quote not in ALLOWED_QUOTES:
            continue
        # Normalize XBT->BTC
        if base in ("XBT", "XXBT"):
            base = "BTC"
        # prefer USD over USDT if both exist
        if base not in by_base:
            by_base[base] = (quote, pk)
        else:
            have_q, _ = by_base[base]
            if have_q != "USD" and quote == "USD":
                by_base[base] = (quote, pk)
    # return BASE -> PAIR_KEY
    return {b: pk for b, (q, pk) in by_base.items()}

async def backfill_hourly(
    session: Session,
    days: int = BACKFILL_DAYS_DEFAULT,
    symbols: Optional[List[str]] = None,
) -> Dict:
    """
    Backfill HOURLY OHLC for many symbols into Candle.
    - Auto-discovers Kraken USD/USDT markets when symbols=None
    - Dedup/skip existing rows
    - Paginates with 'since' cursor
    """
    # 1) pick symbols -> pairs
    base_to_pair = await _discover_usd_pairs()
    if symbols:
        sym_set = {s.upper() for s in symbols}
        base_to_pair = {b: p for b, p in base_to_pair.items() if b in sym_set}

    if BACKFILL_MAX_PAIRS > 0:
        # cap total symbols if configured
        base_to_pair = dict(list(base_to_pair.items())[:BACKFILL_MAX_PAIRS])

    bases = sorted(base_to_pair.keys())
    print(f"[backfill] symbols={len(bases)} days={days}")

    # 2) time window
    start_unix = _unix_from_dt(datetime.utcnow() - timedelta(days=days))
    interval = 60  # hourly

    sem = asyncio.Semaphore(BACKFILL_CONCURRENCY)
    totals = {"symbols": 0, "bars": 0, "errors": 0}

    async def _worker(base: str, pair: str):
        added_total = 0
        since = start_unix
        try:
            async with sem:
                while True:
                    page = await _kraken_ohlc_page(pair, interval=interval, since=since)
                    if page.get("error"):
                        print(f"[backfill] error {base}: {page['error']}")
                        break
                    res = page.get("result", {})
                    key = _pick_result_key_for_pair(pair, res)
                    if not key:
                        break
                    rows = res.get(key, [])
                    if not rows:
                        break

                    # upsert this page
                    added = await _upsert_ohlc_rows(session, base, rows)
                    added_total += added

                    # next cursor
                    nxt = res.get("last")
                    if not nxt:
                        break
                    # If Kraken repeats the same 'last', stop to avoid infinite loop
                    if nxt == since:
                        break
                    since = nxt

                    # tiny pause to be polite
                    await asyncio.sleep(BACKFILL_PAUSE_MS / 1000.0)
        except Exception as e:
            print(f"[backfill] {base} failed: {e}")
            totals["errors"] += 1
            return
        totals["symbols"] += 1
        totals["bars"] += added_total
        if added_total:
            print(f"[backfill] {base}: +{added_total} hourly bars")

    tasks = [asyncio.create_task(_worker(b, base_to_pair[b])) for b in bases]
    if not tasks:
        return {"ok": True, "symbols": 0, "bars": 0}
    await asyncio.gather(*tasks)
    return {"ok": True, **totals}
