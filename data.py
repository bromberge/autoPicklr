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

async def _kraken_fetch_ohlc(pair: str, interval: int = 1) -> Dict:
    """
    GET https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1
    Response: {"error":[],"result":{"XBTUSD":[[ts,open,high,low,close,vwap,volume,count],...]}}
    """
    cli = await _get_client()
    r = await cli.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": pair, "interval": str(interval)}
    )
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
        print(f"[data] {symbol}: +{rows_added} bars from Kraken (last {last_ts_str} close {last_close:.2f})")

    return rows_added

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
        # fallback: accept USD or USDT and match the base (XBT/ETH/...)
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
        print(f"[data] {symbol}: +{rows_added} bars from Kraken (last {last_ts_str} close {last_close:.2f})")
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
