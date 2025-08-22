# ml_features.py
# Builds features for training and for "now" at runtime.
# IMPORTANT: Keep the feature names EXACTLY aligned with training:
# ['ret_1h','ret_4h','ret_24h','ema_spread','ema26_slope','breakout_pct','atr_pct','volume']

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from models import Candle
from settings import (
    DET_EMA_SHORT,
    DET_EMA_LONG,
    BREAKOUT_LOOKBACK,
    EMA_SLOPE_LOOKBACK,
)

# Single source of truth for feature names and order
FEATURE_COLS = [
    "ret_1h",
    "ret_4h",
    "ret_24h",
    "ema_spread",
    "ema26_slope",
    "breakout_pct",
    "atr_pct",
    "volume",
]


def _to_df(candles: List[Candle]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        [
            {
                "ts": c.ts,
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume or 0.0),
            }
            for c in candles
        ]
    )
    df = df.sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")
    return df


def load_recent_minutes(session: Session, symbol: str, days: int = 45) -> pd.DataFrame:
    """Load minute bars for a symbol from the DB for the past `days` days."""
    since = datetime.utcnow() - timedelta(days=days)
    rows = session.exec(
        select(Candle)
        .where(Candle.symbol == symbol, Candle.ts >= since)
        .order_by(Candle.ts.asc())
    ).all()
    return _to_df(rows)


def to_hourly(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute bars to hourly bars.
    Uses lower-case 'h' to avoid deprecation warnings.
    """
    if df_minute.empty:
        return df_minute

    rule = "h"
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    hourly = df_minute.resample(rule).agg(agg).dropna().reset_index()
    hourly = hourly.set_index("ts")
    return hourly


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr_like(df_h: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Simple ATR-like measure using range (high-low).
    We express it as a fraction of price (close).
    """
    rng = (df_h["high"] - df_h["low"]).abs()
    atr = rng.rolling(window=window, min_periods=1).mean()
    atr_pct = atr / df_h["close"].replace(0, np.nan)
    return atr_pct.fillna(0.0)


def _ema26_slope(df_h: pd.DataFrame) -> pd.Series:
    """
    Slope of EMA(long) over EMA_SLOPE_LOOKBACK hours (difference, not percent).
    """
    ema26 = _ema(df_h["close"], DET_EMA_LONG)
    shifted = ema26.shift(EMA_SLOPE_LOOKBACK)
    slope = ema26 - shifted
    return slope.fillna(0.0)


def compute_features_hourly(df_h: pd.DataFrame) -> pd.DataFrame:
    """
    Given hourly OHLCV, compute the feature columns and return a new DataFrame
    whose columns are exactly FEATURE_COLS (plus 'close' kept for labeling in training).
    """
    if df_h.empty or len(df_h) < max(DET_EMA_LONG + EMA_SLOPE_LOOKBACK + 1, 30):
        return pd.DataFrame(columns=FEATURE_COLS)

    out = pd.DataFrame(index=df_h.index.copy())

    # Returns
    out["ret_1h"] = df_h["close"].pct_change(1)
    out["ret_4h"] = df_h["close"].pct_change(4)
    out["ret_24h"] = df_h["close"].pct_change(24)

    # EMA spread (short vs long) as a fraction of price
    ema_short = _ema(df_h["close"], DET_EMA_SHORT)
    ema_long = _ema(df_h["close"], DET_EMA_LONG)
    out["ema_spread"] = (ema_short - ema_long) / df_h["close"].replace(0, np.nan)

    # EMA(long) slope (difference over lookback)
    out["ema26_slope"] = _ema26_slope(df_h)

    # Breakout vs recent high
    recent_high = df_h["high"].rolling(window=BREAKOUT_LOOKBACK, min_periods=1).max()
    out["breakout_pct"] = (df_h["close"] / recent_high.replace(0, np.nan)) - 1.0

    # ATR-like percent
    out["atr_pct"] = _atr_like(df_h, window=14)

    # Volume (raw hourly volume)
    out["volume"] = df_h["volume"].fillna(0.0)

    # Clean up
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Keep only the columns we declare, in the exact order
    out = out[FEATURE_COLS]

    # Attach close (useful for labeling in training; harmless otherwise)
    out["close"] = df_h["close"]
    return out


def features_now(session: Session, symbol: str) -> Optional[pd.DataFrame]:
    """
    Build a single-row DataFrame with EXACTLY FEATURE_COLS for the latest hour.
    Returns None if we cannot compute features.
    """
    df_min = load_recent_minutes(session, symbol, days=45)
    if df_min.empty:
        return None

    df_h = to_hourly(df_min)
    feats = compute_features_hourly(df_h)
    if feats.empty:
        return None

    last = feats.iloc[[-1]].copy()  # keep as DataFrame
    # Drop helper columns not used by the model
    if "close" in last.columns:
        last = last.drop(columns=["close"], errors="ignore")

    # Ensure exact columns and order
    for col in FEATURE_COLS:
        if col not in last.columns:
            last[col] = 0.0
    last = last[FEATURE_COLS]

    # Final cleanup
    last = last.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return last


# ---- Training helpers (used by ml_train.py) ---------------------------------

def build_training_matrix(
    session: Session,
    symbols: List[str],
    horizon_hours: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) for training.
    X columns match FEATURE_COLS.
    y is 1 if target is hit before stop within horizon_hours, else 0.
    """
    from settings import TARGET_PCT, STOP_PCT

    X_rows = []
    y_rows = []

    for sym in symbols:
        df_min = load_recent_minutes(session, sym, days=45)
        if df_min.empty:
            continue
        df_h = to_hourly(df_min)
        if df_h.empty:
            continue

        feats = compute_features_hourly(df_h)
        if feats.empty:
            continue

        # We need a clean hourly series to check future highs/lows
        # Merge features with hourly OHLC
        merged = feats.join(df_h[["open", "high", "low", "close"]], how="inner")

        # Walk forward over time to build labels
        # We avoid the very last 'horizon' hours since we can't look into the future
        end_idx = len(merged) - horizon_hours - 1
        if end_idx <= 0:
            continue

        for i in range(end_idx):
            row = merged.iloc[i]
            entry = float(row["close"])
            if entry <= 0:
                continue

            tgt = entry * (1.0 + TARGET_PCT)
            stp = entry * (1.0 - STOP_PCT)

            # look ahead window
            look = merged.iloc[i + 1 : i + 1 + horizon_hours]
            if look.empty:
                continue

            hit_target = (look["high"] >= tgt).idxmax() if (look["high"] >= tgt).any() else None
            hit_stop = (look["low"] <= stp).idxmax() if (look["low"] <= stp).any() else None

            label = 0
            if hit_target is not None and hit_stop is not None:
                # whichever index appears first in time
                label = 1 if hit_target < hit_stop else 0
            elif hit_target is not None:
                label = 1
            elif hit_stop is not None:
                label = 0
            else:
                # neither was hit; treat as 0 (conservative)
                label = 0

            X_rows.append(row[FEATURE_COLS].tolist())
            y_rows.append(label)

    if not X_rows:
        return pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype=int)

    X = pd.DataFrame(X_rows, columns=FEATURE_COLS)
    y = pd.Series(y_rows, name="y").astype(int)
    # Final cleanup
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y
