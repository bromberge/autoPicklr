# signal_engine.py

from typing import List, Optional
from sqlmodel import Session, select
from models import Candle, Signal
from settings import (
    DET_EMA_SHORT, DET_EMA_LONG,
    MIN_BREAKOUT_PCT, MIN_VOLUME_USD, CHOOSER_THRESHOLD,
    BREAKOUT_LOOKBACK, STOP_PCT, TARGET_PCT,
    MIN_EMA_SPREAD, ENABLE_DEBUG_SIGNALS
)

def ema(arr: List[float], span: int) -> List[float]:
    k = 2 / (span + 1)
    out: List[float] = []
    s = None
    for x in arr:
        s = x if s is None else (x - s) * k + s
        out.append(s)
    return out

def _last_candle(session: Session, symbol: str) -> Optional[Candle]:
    return session.exec(
        select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.desc())
    ).first()

from typing import List
from sqlmodel import Session, select
from models import Candle, Signal
from settings import DET_EMA_SHORT, DET_EMA_LONG, MIN_BREAKOUT_PCT, CHOOSER_THRESHOLD

def ema(arr: List[float], span: int) -> List[float]:
    k = 2 / (span + 1)
    out = []
    s = None
    for x in arr:
        s = x if s is None else (x - s) * k + s
        out.append(s)
    return out

def compute_signals(session: Session, symbol: str) -> List[Signal]:
    # Pull full candle history for this symbol (already filled by update_candles_for)
    candles = session.exec(
        select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.asc())
    ).all()

    # Need enough history to compute EMAs and a small breakout window
    min_needed = max(DET_EMA_LONG + 5, 40)
    if len(candles) < min_needed:
        return []

    closes = [c.close for c in candles]
    e1 = ema(closes, DET_EMA_SHORT)
    e2 = ema(closes, DET_EMA_LONG)

    last = candles[-1]
    last_e1, last_e2 = e1[-1], e2[-1]

    # 20-bar breakout (you can parameterize if you want)
    lookback = 20
    recent_high = max(closes[-lookback:])
    breakout = 0.0
    if recent_high > 0:
        breakout = (last.close - recent_high) / recent_high

    # --- scoring ---
    momentum_ok = last_e1 > last_e2
    breakout_ok = breakout >= MIN_BREAKOUT_PCT

    # IMPORTANT: volume gate removed here — universe filter already enforces 24h liquidity
    vol_ok = True

    score = 0.0
    if momentum_ok: score += 0.4
    if breakout_ok: score += 0.3
    if vol_ok:      score += 0.3

    if score < CHOOSER_THRESHOLD:
        return []

    # Risk/targets (keep consistent with your settings or entries)
    entry  = last.close
    stop   = entry * (1 - 0.02)   # 2% stop  (feel free to pull from settings)
    target = entry * (1 + 0.06)   # 6% target (same here)

    sig = Signal(
        symbol=symbol,
        ts=last.ts,
        score=round(score, 3),
        entry=entry,
        stop=stop,
        target=target,
        reason=f"EMA{DET_EMA_SHORT}>{DET_EMA_LONG}, breakout≥{MIN_BREAKOUT_PCT:.2%}"
    )
    print(f"[signal] {symbol}: mom={momentum_ok} brk={breakout_ok:.4f} score={score:.2f}")
    return [sig]

