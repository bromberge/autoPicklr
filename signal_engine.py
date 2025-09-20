# signal_engine.py
from typing import List
from sqlmodel import Session, select

from models import Candle, Signal
from ml_runtime import model_predict_for_symbol, hourly_atr_from_db
from settings import (
    USE_MODEL, MIN_BREAKOUT_PCT,
    DET_EMA_SHORT, DET_EMA_LONG,
    BREAKOUT_LOOKBACK, EMA_SLOPE_LOOKBACK,
    STOP_PCT, TARGET_PCT,
    USE_ATR_STOPS, ATR_STOP_MULT, ATR_TARGET_MULT,
)

def _ema(arr: List[float], span: int) -> List[float]:
    k = 2 / (span + 1)
    out, s = [], None
    for x in arr:
        s = x if s is None else (x - s) * k + s
        out.append(s)
    return out

def compute_signals(session: Session, symbol: str) -> List[Signal]:
    candles = session.exec(
        select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.asc())
    ).all()
    min_needed = max(DET_EMA_LONG + EMA_SLOPE_LOOKBACK + 5, 40)
    if len(candles) < min_needed:
        return []

    closes = [float(c.close) for c in candles]
    e_short = _ema(closes, DET_EMA_SHORT)
    e_long  = _ema(closes, DET_EMA_LONG)

    last = candles[-1]
    px   = float(last.close)
    eS   = float(e_short[-1])
    eL   = float(e_long[-1])

    # breakout vs recent high (exclude current if possible)
    lb = BREAKOUT_LOOKBACK
    recent_high = max(closes[-(lb+1):-1]) if len(closes) > lb else max(closes)
    breakout = (px - recent_high) / recent_high if recent_high > 0 else 0.0

    # EMA(long) slope
    if len(e_long) > EMA_SLOPE_LOOKBACK:
        eL_ago = float(e_long[-(EMA_SLOPE_LOOKBACK + 1)])
    else:
        eL_ago = eL
    ema_long_slope = (eL - eL_ago) / max(1, EMA_SLOPE_LOOKBACK)

    # ATR stops/targets (fallback to %)
    if USE_ATR_STOPS:
        atr = hourly_atr_from_db(session, symbol)
        if atr and atr > 0:
            stop   = px - ATR_STOP_MULT   * atr
            target = px + ATR_TARGET_MULT * atr
        else:
            stop   = px * (1.0 - STOP_PCT)
            target = px * (1.0 + TARGET_PCT)

    from settings import FEE_PCT, SLIPPAGE_PCT
    fee_buf = FEE_PCT * 2.0
    slip_buf = SLIPPAGE_PCT * 2.0
    MIN_STOP_GAP_PCT = max(0.006, 3.0 * (fee_buf + slip_buf))

    min_stop = px * (1.0 - MIN_STOP_GAP_PCT)
    if stop > min_stop:
        stop = min_stop
    
    else:
        stop   = px * (1.0 - STOP_PCT)
        target = px * (1.0 + TARGET_PCT)

    # Score: model first, else heuristic
    score = 0.0
    if USE_MODEL:
        try:
            p = model_predict_for_symbol(session, symbol)
            if p is not None:
                score = float(p)
        except Exception as e:
            print(f"[model] scoring failed for {symbol}: {e}")

    if (not USE_MODEL) or score == 0.0:
        score = 0.0
        if eS > eL:               # momentum
            score += 0.5
        if breakout >= MIN_BREAKOUT_PCT:
            score += 0.5

    # package
    sig = Signal(
        symbol=symbol,
        ts=last.ts,
        score=round(score, 4),
        entry=px,
        stop=float(stop),
        target=float(target),
        reason=f"EMA{DET_EMA_SHORT}>{DET_EMA_LONG}, breakout≥{MIN_BREAKOUT_PCT:.2%}"
    )

    # extras for chooser / debug panels
    try:
        sig.breakout_pct    = float(breakout)
        sig.ema_spread      = (eS - eL) / px if px > 0 else 0.0
        sig.ema_long        = eL
        sig.ema_long_ago    = eL_ago
        sig.ema_long_slope  = ema_long_slope
        sig.extension_pct   = (px / eL - 1.0) if eL > 0 else 0.0
        rr_den = max(px - stop, 1e-12)
        sig.rr = (target - px) / rr_den
    except Exception:
        pass

    return [sig]
