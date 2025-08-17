# data.py

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import httpx
from sqlmodel import Session, select

from models import Candle
from settings import UNIVERSE, HISTORY_MINUTES, USE_COINGECKO

# --- CoinGecko id cache ---
CG_IDS: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}
_id_cache_checked: Dict[str, bool] = {}

_client: Optional[httpx.AsyncClient] = None


def _utc(dt_ms: int) -> datetime:
    return datetime.fromtimestamp(dt_ms / 1000, tz=timezone.utc).replace(tzinfo=None)


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=20.0,
            headers={
                "User-Agent": "AutoPicklr/1.0 (+https://example.com)",
                "Accept": "application/json",
            },
        )
    return _client


async def _cg_lookup_id(symbol: str) -> Optional[str]:
    sym = symbol.upper()
    if sym in CG_IDS:
        return CG_IDS[sym]
    if _id_cache_checked.get(sym):
        return CG_IDS.get(sym)

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


async def _cg_fetch_1m_prices(cid: str, days: int = 1) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    cli = await _get_client()
    r = await cli.get(
        f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart",
        params={"vs_currency": "usd", "days": str(days), "interval": "1m"},
    )
    r.raise_for_status()
    j = r.json()
    prices = j.get("prices", [])
    volumes = j.get("total_volumes", [])
    return prices, volumes


def _last_candle_ts(session: Session, symbol: str) -> Optional[datetime]:
    row = session.exec(
        select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.desc())
    ).first()
    return row.ts if row else None


async def _upsert_minute_candles(session: Session, symbol: str) -> int:
    cid = await _cg_lookup_id(symbol)
    if not cid:
        print(f"[data] No CoinGecko id for {symbol}")
        return 0

    try:
        prices, volumes = await _cg_fetch_1m_prices(cid, days=1)
    except Exception as e:
        print(f"[data] market_chart error for {symbol} ({cid}): {e}")
        return 0

    if not prices:
        print(f"[data] No prices returned for {symbol}")
        return 0

    vol_map = {ts: v for ts, v in volumes} if volumes else {}
    rows_added = 0

    last_ts = _last_candle_ts(session, symbol)
    min_allowed = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(minutes=HISTORY_MINUTES)
    min_allowed = min_allowed.replace(tzinfo=None)

    for ts_ms, close in prices:
        ts = _utc(ts_ms)
        if last_ts and ts <= last_ts:
            continue
        if ts < min_allowed:
            continue

        vol = vol_map.get(ts_ms, 0.0)
        cd = Candle(
            symbol=symbol,
            ts=ts,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=vol,
        )
        exists = session.exec(
            select(Candle).where(Candle.symbol == symbol, Candle.ts == ts)
        ).first()
        if not exists:
            session.add(cd)
            rows_added += 1

    if rows_added:
        session.commit()
        # log the last stored for visibility
        last_close = prices[-1][1]
        last_ts_str = _utc(prices[-1][0]).isoformat()
        print(f"[data] {symbol}: +{rows_added} bars (last {last_ts_str} close {last_close:.2f})")

    return rows_added


async def update_candles(session: Session):
    if not USE_COINGECKO:
        return
    tasks = [asyncio.create_task(_upsert_minute_candles(session, sym)) for sym in UNIVERSE]
    await asyncio.sleep(0.2)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for sym, res in zip(UNIVERSE, results):
        if isinstance(res, Exception):
            print(f"[data] Error for {sym}: {res}")
    # print any exceptions explicitly
    for sym, res in zip(UNIVERSE, results):
        if isinstance(res, Exception):
            print(f"[data] exception for {sym}: {res}")


# --------- Helpers exposed to routes for debugging ---------

async def cg_simple_prices(symbols: List[str]) -> Dict[str, float]:
    """Hit CoinGecko simple/price to show current spot for quick comparison."""
    cli = await _get_client()
    ids = []
    for s in symbols:
        cid = await _cg_lookup_id(s)
        if cid:
            ids.append(cid)
    if not ids:
        return {}
    try:
        r = await cli.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": ",".join(ids), "vs_currencies": "usd"},
        )
        r.raise_for_status()
        j = r.json()
        out = {}
        # map back id -> symbol
        id_to_sym = {v: k for k, v in CG_IDS.items()}
        for cid, obj in j.items():
            sym = id_to_sym.get(cid, cid).upper()
            out[sym] = float(obj.get("usd", 0.0))
        return out
    except Exception as e:
        print(f"[data] simple/price error: {e}")
        return {}


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
