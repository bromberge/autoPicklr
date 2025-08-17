#data.py

import httpx, asyncio
from datetime import datetime, timezone
from typing import List, Dict
from sqlmodel import Session, select
from models import Candle
from settings import UNIVERSE, HISTORY_MINUTES

# Simple mapping; adjust to your pairs
CG_IDS = {"BTC":"bitcoin","ETH":"ethereum","SOL":"solana"}

async def fetch_last_minute(symbol: str) -> Dict:
    # CoinGecko has minute data via /market_chart?interval=1m (recent)
    cid = CG_IDS[symbol]
    url = f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart"
    params = {"vs_currency":"usd","days":"1","interval":"1m"}
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    # take the last item
    prices = data.get("prices", [])
    vols = data.get("total_volumes", [])
    if not prices:
        return {}
    ts_ms, close = prices[-1]
    _, vol = vols[-1] if vols else (ts_ms, 0.0)
    # fabricate OHLC from last few points if needed (fast path)
    # For SIM we'll store as close-only minute; open/high/low ~= close
    return {
        "symbol": symbol,
        "ts": datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).replace(tzinfo=None),
        "open": close, "high": close, "low": close, "close": close,
        "volume": vol
    }

async def update_candles(session: Session):
    # Minimal: only last minute bar to keep load down
    for sym in UNIVERSE:
        try:
            cd = await fetch_last_minute(sym)
            if not cd: 
                continue
            exists = session.exec(select(Candle).where(Candle.symbol==sym, Candle.ts==cd["ts"])).first()
            if not exists:
                session.add(Candle(**cd))
        except Exception:
            continue
    session.commit()
