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

def compute_signals(session: Session, symbol: str) -> List[Signal]:
    signals: List[Signal] = []

    # DEBUG mode: easy signals to exercise the pipeline
    if ENABLE_DEBUG_SIGNALS:
        last = _last_candle(session, symbol)
        if not last:
            return signals
        px = last.close
        entry  = round(px * 1.002, 6)  # +0.2%
        stop   = round(px * 0.99,  6)  # -1.0%
        target = round(px * 1.01,  6)  # +1.0%
        sig = Signal(
            symbol=symbol, ts=last.ts, score=1.0,
            entry=entry, stop=stop, target=target,
            reason="DEBUG breakout"
        )
        signals.append(sig)
        return signals

    # REAL STRATEGY
    candles = session.exec(
        select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.asc())
    ).all()
    if len(candles) < max(DET_EMA_LONG + 5, BREAKOUT_LOOKBACK + 5, 40):
        return signals

    closes = [c.close for c in candles]
    vols   = [c.volume for c in candles]
    e1 = ema(closes, DET_EMA_SHORT)
    e2 = ema(closes, DET_EMA_LONG)

    last = candles[-1]
    last_e1, last_e2 = e1[-1], e2[-1]

    # Breakout above N-bar high
    window_hi = max(closes[-BREAKOUT_LOOKBACK:])
    breakout = (last.close - window_hi) / window_hi if window_hi > 0 else 0.0

    # Volume filter ($ volume proxy)
    avg_vol = sum(vols[-20:]) / 20.0 if len(vols) >= 20 else sum(vols) / max(len(vols), 1)
    vol_ok = (avg_vol * last.close) >= MIN_VOLUME_USD

    # Momentum filters
    momentum_ok = last_e1 > last_e2

    # NEW: EMA spread filter (as a % of price)
    # Example: (EMA12 - EMA26) / price >= MIN_EMA_SPREAD
    ema_spread = (last_e1 - last_e2) / last.close if last.close > 0 else 0.0
    ema_spread_ok = ema_spread >= MIN_EMA_SPREAD

    breakout_ok = breakout >= MIN_BREAKOUT_PCT

    # Score (you can tune the weights)
    score = 0.0
    if momentum_ok:    score += 0.35
    if ema_spread_ok:  score += 0.25
    if breakout_ok:    score += 0.20
    if vol_ok:         score += 0.20

    if score < CHOOSER_THRESHOLD:
        return signals

    entry  = float(last.close)
    stop   = entry * (1 - STOP_PCT)
    target = entry * (1 + TARGET_PCT)

    sig = Signal(
        symbol=symbol, ts=last.ts, score=round(score, 3),
        entry=entry, stop=stop, target=target,
        reason=(
            f"EMA{DET_EMA_SHORT}>{DET_EMA_LONG}, "
            f"spread≥{MIN_EMA_SPREAD:.2%}, "
            f"{BREAKOUT_LOOKBACK}-bar breakout≥{MIN_BREAKOUT_PCT:.2%}, "
            f"$vol≥{MIN_VOLUME_USD:,.0f}"
        )
    )
    signals.append(sig)
    return signals
