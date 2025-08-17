# signal_engine.py

from typing import List, Optional
from sqlmodel import Session, select
from models import Candle, Signal
from settings import (
    DET_EMA_SHORT, DET_EMA_LONG, MIN_BREAKOUT_PCT,
    MIN_VOLUME_USD, CHOOSER_THRESHOLD
)

# Turn this on to force simple test signals
DEBUG_FORCE_SIGNALS = True  # set to False later


# ------ helpers ------

def ema(arr: List[float], span: int) -> List[float]:
    k = 2 / (span + 1)
    out: List[float] = []
    s = None
    for x in arr:
        s = x if s is None else (x - s) * k + s
        out.append(s)
    return out


def get_last_candle(session: Session, symbol: str) -> Optional[Candle]:
    """Return the most recent candle for a symbol, or None."""
    return session.exec(
        select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.desc())
    ).first()


def last_close_for_symbol(session: Session, symbol: str) -> Optional[float]:
    c = get_last_candle(session, symbol)
    return c.close if c else None


# ------ main signal function ------

def compute_signals(session: Session, symbol: str) -> List[Signal]:
    signals: List[Signal] = []

    # DEBUG MODE: create a very simple breakout-style signal so you can test end-to-end
    if DEBUG_FORCE_SIGNALS:
        last = get_last_candle(session, symbol)
        if last:
            px = last.close
            entry  = round(px * 1.002, 6)   # 0.2% above
            stop   = round(px * 0.99, 6)    # 1% risk
            target = round(px * 1.01, 6)    # 1% reward

            # Build a Signal using your models.Signal shape
            sig = Signal(
                symbol=symbol,
                ts=last.ts,
                score=1.0,                  # high score so it passes any chooser
                entry=entry,
                stop=stop,
                target=target,
                reason="DEBUG breakout"
            )
            signals.append(sig)
        return signals

    # REAL LOGIC BELOW (runs when DEBUG_FORCE_SIGNALS = False)
    candles = session.exec(
        select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.asc())
    ).all()
    if len(candles) < max(DET_EMA_LONG + 5, 40):
        return []

    closes = [c.close for c in candles]
    vols   = [c.volume for c in candles]
    e1 = ema(closes, DET_EMA_SHORT)
    e2 = ema(closes, DET_EMA_LONG)

    last = candles[-1]
    last_e1, last_e2 = e1[-1], e2[-1]
    hi20 = max(closes[-20:])
    breakout = (last.close - hi20) / hi20 if hi20 > 0 else 0

    # crude $ volume proxy: avg vol * price
    vol_ok = (sum(vols[-20:]) / 20.0) * last.close >= MIN_VOLUME_USD
    momentum_ok = last_e1 > last_e2
    breakout_ok = breakout >= MIN_BREAKOUT_PCT

    score = 0.0
    if momentum_ok: score += 0.4
    if breakout_ok: score += 0.3
    if vol_ok:      score += 0.3

    if score < CHOOSER_THRESHOLD:
        return []

    entry  = last.close
    stop   = entry * (1 - 0.02)   # 2% stop
    target = entry * (1 + 0.085)  # 8.5% target

    sig = Signal(
        symbol=symbol,
        ts=last.ts,
        score=round(score, 3),
        entry=entry,
        stop=stop,
        target=target,
        reason=f"EMA{DET_EMA_SHORT}>{DET_EMA_LONG}, breakout≥{MIN_BREAKOUT_PCT:.2%}"
    )
    return [sig]
