#signal_engine.py

from typing import List, Tuple
from sqlmodel import Session, select
from models import Candle, Signal
from settings import DET_EMA_SHORT, DET_EMA_LONG, MIN_BREAKOUT_PCT, MIN_VOLUME_USD, CHOOSER_THRESHOLD

def ema(arr: List[float], span: int) -> List[float]:
    k = 2/(span+1)
    out = []
    s = None
    for x in arr:
        s = x if s is None else (x - s)*k + s
        out.append(s)
    return out

def compute_signals(session: Session, symbol: str) -> List[Signal]:
    candles = session.exec(
        select(Candle).where(Candle.symbol==symbol).order_by(Candle.ts.asc())
    ).all()
    if len(candles) < max(DET_EMA_LONG+5, 40):
        return []
    closes = [c.close for c in candles]
    vols   = [c.volume for c in candles]
    e1 = ema(closes, DET_EMA_SHORT)
    e2 = ema(closes, DET_EMA_LONG)

    last = candles[-1]
    last_e1, last_e2 = e1[-1], e2[-1]
    breakout = (last.close - max(closes[-20:]))/max(closes[-20:]) if max(closes[-20:])>0 else 0
    vol_ok = (sum(vols[-20:]) / 20) * last.close >= MIN_VOLUME_USD  # crude $ volume proxy
    momentum_ok = last_e1 > last_e2
    breakout_ok = breakout >= MIN_BREAKOUT_PCT

    score = 0.0
    if momentum_ok: score += 0.4
    if breakout_ok: score += 0.3
    if vol_ok:      score += 0.3

    if score < CHOOSER_THRESHOLD:
        return []

    entry = last.close
    stop  = entry * (1 - 0.02)   # 2% stop
    target= entry * (1 + 0.085)  # 8.5% target

    sig = Signal(
        symbol=symbol, ts=last.ts, score=round(score,3),
        entry=entry, stop=stop, target=target,
        reason=f"EMA{DET_EMA_SHORT}>{DET_EMA_LONG}, breakout>= {MIN_BREAKOUT_PCT:.2%}"
    )
    return [sig]
