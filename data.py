# data.py
import asyncio
import random
from datetime import datetime, timezone
from typing import Dict, Optional
import httpx
from sqlmodel import Session, select
from models import Candle
from settings import UNIVERSE, USE_COINGECKO

CG_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}

def _utc_floor_minute(dt: Optional[datetime] = None) -> datetime:
    dt = dt or datetime.now(timezone.utc)
    return dt.replace(second=0, microsecond=0, tzinfo=None)

def _seed_price(symbol: str) -> float:
    return {"BTC": 60000.0, "ETH": 3000.0, "SOL": 150.0}.get(symbol, 100.0)

def _last_close(session: Session, symbol: str) -> float:
    c = session.exec(
        select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.desc())
    ).first()
    return c.close if c else _seed_price(symbol)

async def _fetch_cg_last_minute(symbol: str) -> Optional[Dict]:
    cid = CG_IDS.get(symbol)
    if not cid:
        return None
    url = f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart"
    params = {"vs_currency": "usd", "days": "1", "interval": "1m"}
    # NOTE: Free tier can be rate-limited; we handle failures by returning None.
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    prices = data.get("prices", [])
    vols = data.get("total_volumes", [])
    if not prices:
        return None
    ts_ms, close = prices[-1]
    _, vol = vols[-1] if vols else (ts_ms, 0.0)
    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).replace(second=0, microsecond=0, tzinfo=None)
    return {
        "symbol": symbol,
        "ts": ts,
        "open": float(close),
        "high": float(close),
        "low":  float(close),
        "close": float(close),
        "volume": float(vol or 0.0),
    }

def _mock_one(symbol: str) -> Dict:
    prev = _last_close(NoneSession, symbol)  # filled below; placeholder
    drift = 1.0 + random.uniform(-0.015, 0.015)  # ±1.5% to trigger exits faster
    close = round(prev * drift, 6)
    o = prev
    h = max(o, close) * (1 + random.uniform(0.0, 0.001))
    l = min(o, close) * (1 - random.uniform(0.0, 0.001))
    v = round(random.uniform(100.0, 1000.0), 3)
    return {
        "symbol": symbol,
        "ts": _utc_floor_minute(),
        "open": float(o), "high": float(h), "low": float(l),
        "close": float(close), "volume": float(v),
    }

async def update_candles(session: Session):
    """
    If USE_COINGECKO=True, try to fetch last minute from CoinGecko.
    On any failure (rate limit/network/etc), write a mock candle so the loop keeps working.
    """
    global NoneSession
    NoneSession = session  # for the small mock helper above

    for sym in UNIVERSE:
        row = None
        if USE_COINGECKO:
            try:
                row = await _fetch_cg_last_minute(sym)
            except Exception:
                row = None
        if row is None:
            row = _mock_one(sym)

        exists = session.exec(
            select(Candle).where(Candle.symbol == sym, Candle.ts == row["ts"])
        ).first()
        if not exists:
            session.add(Candle(**row))

    session.commit()
    await asyncio.sleep(0)
