# ml_runtime.py

from __future__ import annotations

import math
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlmodel import Session, select

from models import Candle
from settings import (
    USE_MODEL, MODEL_PATH,
    DET_EMA_SHORT, DET_EMA_LONG,
    BREAKOUT_LOOKBACK, EMA_SLOPE_LOOKBACK,
)

# The exact feature order the model was trained with (from your training logs)
FEATURE_ORDER: List[str] = [
    "ret_1h", "ret_4h", "ret_24h",
    "ema_spread", "ema26_slope",
    "breakout_pct", "atr_pct", "volume",
]

_MODEL: Optional[xgb.Booster] = None


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Plain EMA to avoid external deps; returns array same length as input."""
    if len(arr) == 0:
        return arr
    k = 2.0 / (span + 1.0)
    out = np.empty_like(arr, dtype=float)
    s = float(arr[0])
    out[0] = s
    for i in range(1, len(arr)):
        x = float(arr[i])
        s = (x - s) * k + s
        out[i] = s
    return out


def _to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute bars to hourly bars.
    Expects columns: ['ts','open','high','low','close','volume'].
    """
    if df.empty:
        return df
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    # Use 'h' (lowercase) to avoid the FutureWarning
    hourly = df.resample("h").agg(agg).dropna().reset_index()
    return hourly


def _atr_pct(hourly: pd.DataFrame, n: int = 14) -> float:
    """
    Simple ATR% over last n hourly bars: ATR / last_close.
    """
    if len(hourly) < n + 1:
        return 0.0
    h = hourly["high"].values
    l = hourly["low"].values
    c = hourly["close"].values
    trs = []
    for i in range(1, len(hourly)):
        tr = max(
            h[i] - l[i],
            abs(h[i] - c[i - 1]),
            abs(l[i] - c[i - 1]),
        )
        trs.append(tr)
    if len(trs) < n:
        return 0.0
    atr = float(np.mean(trs[-n:]))
    last_close = float(c[-1]) if c[-1] else 0.0
    return (atr / last_close) if last_close > 0 else 0.0


def _build_now_features_from_hourly(hourly: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Compute the 8 features used in training, from hourly bars.
    Returns dict or None if not enough data.
    """
    # Need enough history for EMA26, slope lookback, and 24h return.
    need = max(DET_EMA_LONG + EMA_SLOPE_LOOKBACK + 2, 30)
    if len(hourly) < need:
        return None

    close = hourly["close"].values.astype(float)
    high = hourly["high"].values.astype(float)
    volume = hourly["volume"].values.astype(float)

    last = float(close[-1])
    if last <= 0:
        return None

    # Returns
    def _ret(n: int) -> float:
        if len(close) <= n or close[-(n + 1)] == 0:
            return 0.0
        return (last / float(close[-(n + 1)]) - 1.0)

    ret_1h = _ret(1)
    ret_4h = _ret(4)
    ret_24h = _ret(24)

    # EMAs and spread
    ema_short = _ema(close, DET_EMA_SHORT)
    ema_long = _ema(close, DET_EMA_LONG)
    ema_spread = (float(ema_short[-1]) - float(ema_long[-1])) / last

    # EMA26 slope over lookback
    back = EMA_SLOPE_LOOKBACK
    ema26_slope = float(ema_long[-1]) - float(ema_long[-(back + 1)])

    # Breakout% over recent highs (exclude the current bar)
    lb = max(2, BREAKOUT_LOOKBACK)
    past_hi = float(np.max(high[-(lb + 1):-1])) if len(high) >= (lb + 1) else float(np.max(high[:-1]))
    breakout_pct = ((last - past_hi) / past_hi) if past_hi > 0 else 0.0

    # ATR%
    atr_pct = _atr_pct(hourly, n=14)

    # Last hour volume (same aggregation as training)
    vol_last = float(volume[-1]) if len(volume) else 0.0

    feats = {
        "ret_1h": ret_1h,
        "ret_4h": ret_4h,
        "ret_24h": ret_24h,
        "ema_spread": ema_spread,
        "ema26_slope": ema26_slope,
        "breakout_pct": breakout_pct,
        "atr_pct": atr_pct,
        "volume": vol_last,
    }
    return feats


def load_model() -> Optional[xgb.Booster]:
    """Load the XGBoost model once. Returns None if disabled or missing."""
    global _MODEL
    if not USE_MODEL:
        return None
    if _MODEL is not None:
        return _MODEL
    try:
        bst = xgb.Booster()
        bst.load_model(MODEL_PATH)
        _MODEL = bst
        print(f"[model] Loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"[model] Could not load model at {MODEL_PATH}: {e}")
        _MODEL = None
    return _MODEL


def _fetch_candles(session: Session, symbol: str) -> pd.DataFrame:
    rows = session.exec(
        select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.asc())
    ).all()
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    data = {
        "ts": [r.ts for r in rows],
        "open": [float(r.open) for r in rows],
        "high": [float(r.high) for r in rows],
        "low": [float(r.low) for r in rows],
        "close": [float(r.close) for r in rows],
        "volume": [float(getattr(r, "volume", 0.0) or 0.0) for r in rows],
    }
    return pd.DataFrame(data)


def model_predict_for_symbol(session: Session, symbol: str) -> Optional[float]:
    """
    Build features NOW for `symbol` from DB candles (hourly aggregation),
    and return model probability in [0,1]. Returns None if disabled/unavailable.
    """
    if not USE_MODEL:
        return None

    bst = load_model()
    if bst is None:
        return None

    try:
        raw = _fetch_candles(session, symbol)
        hourly = _to_hourly(raw)
        feats = _build_now_features_from_hourly(hourly)
        if not feats:
            return None

        # Build DMatrix with EXACT feature names used in training
        X = np.array([[feats[k] for k in FEATURE_ORDER]], dtype=float)
        dmat = xgb.DMatrix(X, feature_names=FEATURE_ORDER)
        pred = bst.predict(dmat)
        if isinstance(pred, np.ndarray) and len(pred) > 0:
            p = float(pred[0])
            # Clip to [0,1] just to be safe
            return max(0.0, min(1.0, p))
        return None
    except Exception as e:
        print(f"[model] predict error for {symbol}: {e}")
        return None
        
from typing import Optional
from sqlmodel import Session

def hourly_atr_from_db(session: Session, symbol: str, n: int = 14) -> float:
    """
    Return ATR (absolute, not %) computed on hourly bars over the last n periods.
    If there isn't enough data, returns 0.0.
    """
    raw = _fetch_candles(session, symbol)
    hourly = _to_hourly(raw)
    if len(hourly) < n + 1:
        return 0.0

    h = hourly["high"].values
    l = hourly["low"].values
    c = hourly["close"].values

    trs = []
    for i in range(1, len(hourly)):
        tr = max(
            h[i] - l[i],
            abs(h[i] - c[i - 1]),
            abs(l[i] - c[i - 1]),
        )
        trs.append(float(tr))

    if len(trs) < n:
        return 0.0

    atr = float(np.mean(trs[-n:]))
    return atr
