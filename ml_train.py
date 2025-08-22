# ml_train.py
#
# Purpose:
# - Train an XGBoost classifier to predict: "Will the target be hit before the stop within H hours?"
# - Uses live-fetched Kraken HOURLY OHLC (last 720 hours ≈ 30 days). No DB backfill required.
#
# How to run (examples):
#   python ml_train.py --horizon 168 --save models/xgb_target_first.json
#   python ml_train.py --symbols BTC,ETH,SOL --horizon 168 --save models/xgb_target_first.json
#
# Notes:
# - Keeps features very close to what the runtime scorer expects (hourly trend/breakout/ATR-like).
# - Prints a recommended SCORE_THRESHOLD based on validation precision/recall curve.

import argparse
import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import httpx
import numpy as np
import pandas as pd
from sqlmodel import SQLModel, Session, select, create_engine
from xgboost import XGBClassifier

# --- app imports for defaults/symbols ---
from settings import (
    UNIVERSE, TARGET_PCT, STOP_PCT
)
from universe import UniversePair

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"

def kraken_pair_for_symbol(symbol: str) -> str:
    """
    Map a bare symbol like 'BTC' to a likely Kraken pair, prefer USD then USDT.
    We first try DB UniversePair mappings (if present).
    Otherwise assume 'BTCUSD' and 'BTCUSDT'.
    """
    # Try DB mapping first
    try:
        engine = create_engine("sqlite:///picklr.db", echo=False, connect_args={"check_same_thread": False})
        with Session(engine) as s:
            row = s.exec(select(UniversePair).where(UniversePair.symbol == symbol)).first()
            if row and row.pair:
                return row.pair
    except Exception:
        pass

    # Fallback guesses
    candidates = [f"{symbol}USD", f"{symbol}USDT", f"X{symbol}ZUSD", f"X{symbol}ZUSDT"]
    return candidates[0]

def fetch_hourly_ohlc(pair: str, max_retries: int = 3, pause_sec: float = 1.0) -> pd.DataFrame:
    """
    Fetch last ~720 hourly candles from Kraken REST OHLC (interval=60).
    Returns DataFrame with columns: ts, open, high, low, close, volume
    """
    params = {"pair": pair, "interval": 60}
    for attempt in range(1, max_retries + 1):
        try:
            r = httpx.get(KRAKEN_OHLC_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if data.get("error"):
                raise RuntimeError(f"Kraken error: {data['error']}")

            # The result dict's key is the standardized pair name; fetch the first one.
            result = data.get("result", {})
            keys = [k for k in result.keys() if k != "last"]
            if not keys:
                raise RuntimeError("No OHLC data returned")
            k = keys[0]
            rows = result[k]

            # Each row format per Kraken docs: [time, open, high, low, close, vwap, volume, count]
            # We keep time, open, high, low, close, volume
            df = pd.DataFrame(rows, columns=[
                "t", "open", "high", "low", "close", "vwap", "volume", "count"
            ])
            df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
            df = df.rename(columns={"t": "ts"})
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[["ts", "open", "high", "low", "close", "volume"]].dropna()
            return df
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(pause_sec)
    raise RuntimeError("Unreachable")

def build_features_labels(df: pd.DataFrame, horizon_hours: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Input: hourly OHLC df with columns ts, open, high, low, close, volume
    Output:
      X: features at each time t
      y: 1 if target is hit before stop in next H hours, else 0
    """
    df = df.copy().sort_values("ts").reset_index(drop=True)

    # Basic returns
    df["ret_1h"] = df["close"].pct_change(1)
    df["ret_4h"] = df["close"].pct_change(4)
    df["ret_24h"] = df["close"].pct_change(24)

    # EMAs and spreads (hourly)
    def ema(arr: pd.Series, span: int) -> pd.Series:
        k = 2.0 / (span + 1)
        s = None
        out = []
        for x in arr:
            s = x if s is None else (x - s) * k + s
            out.append(s)
        return pd.Series(out, index=arr.index)

    df["ema12"] = ema(df["close"], 12)
    df["ema26"] = ema(df["close"], 26)
    df["ema_spread"] = (df["ema12"] - df["ema26"]) / df["close"].clip(lower=1e-12)
    df["ema26_slope"] = df["ema26"] - df["ema26"].shift(6)  # slope over 6 hours

    # Breakout vs 20-bar high
    lookback = 20
    df["hh"] = df["close"].rolling(lookback).max()
    df["breakout_pct"] = (df["close"] - df["hh"]) / df["hh"].replace(0, np.nan)

    # ATR-like (true range on hourly)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = df["tr"].rolling(14).mean()
    df["atr_pct"] = df["atr14"] / df["close"].replace(0, np.nan)

    # Label: target-first within H hours (conservative: require target touched and stop NOT touched)
    entry = df["close"]
    target = entry * (1.0 + float(TARGET_PCT))
    stop   = entry * (1.0 - float(STOP_PCT))

    future_high = df["high"].rolling(window=horizon_hours, min_periods=1).max().shift(-horizon_hours)
    future_low  = df["low"].rolling(window=horizon_hours, min_periods=1).min().shift(-horizon_hours)

    hit_target = (future_high >= target)
    hit_stop   = (future_low <= stop)
    y = (hit_target & (~hit_stop)).astype(int)

    # Feature set
    X = df[[
        "ret_1h","ret_4h","ret_24h",
        "ema_spread","ema26_slope",
        "breakout_pct","atr_pct","volume"
    ]].copy()

    # Drop rows with any NaN in features/labels
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    return X, y

def split_train_val(X: pd.DataFrame, y: pd.Series, val_frac: float = 0.2) -> Tuple[pd.DataFrame, ...]:
    n = len(X)
    n_val = int(n * val_frac)
    if n_val == 0:
        return X, y, X.copy(), y.copy()
    # time-ordered split
    X_train, y_train = X.iloc[:-n_val], y.iloc[:-n_val]
    X_val, y_val = X.iloc[-n_val:], y.iloc[-n_val:]
    return X_train, y_train, X_val, y_val

def eval_threshold(probs: np.ndarray, y_true: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Sweep thresholds to suggest a reasonable SCORE_THRESHOLD.
    Optimizes F1 by default, but prints a small table.
    """
    best_thr, best_f1 = 0.5, -1
    out = {}
    for thr in [i/100 for i in range(10, 90, 5)]:  # 0.10 .. 0.85
        y_hat = (probs >= thr).astype(int)
        tp = int(np.sum((y_hat == 1) & (y_true == 1)))
        fp = int(np.sum((y_hat == 1) & (y_true == 0)))
        fn = int(np.sum((y_hat == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        out[f"{thr:.2f}"] = {"precision": precision, "recall": recall, "f1": f1}
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="", help="Comma list like BTC,ETH,SOL. Empty = use DB universe then settings.UNIVERSE")
    parser.add_argument("--horizon", type=int, default=168, help="Label horizon in hours (168 = 7 days).")
    parser.add_argument("--save", type=str, default="models/xgb_target_first.json", help="Path to save model.")
    args = parser.parse_args()

    # Build symbol list
    symbols: List[str] = []
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        # Try DB universe first
        try:
            engine = create_engine("sqlite:///picklr.db", echo=False, connect_args={"check_same_thread": False})
            with Session(engine) as s:
                rows = s.exec(select(UniversePair).order_by(UniversePair.usd_vol_24h.desc())).all()
                symbols = [r.symbol for r in rows]
        except Exception:
            pass
        if not symbols:
            symbols = UNIVERSE

    print(f"[train] Symbols: {symbols[:15]}{'...' if len(symbols)>15 else ''}")

    # Fetch hourly OHLC and build dataset
    X_all = []
    y_all = []
    kept = 0
    for sym in symbols:
        pair = kraken_pair_for_symbol(sym)
        try:
            df = fetch_hourly_ohlc(pair)
        except Exception as e:
            print(f"[train] Skip {sym} ({pair}): fetch error: {e}")
            continue

        # Need at least horizon + ~40 bars for features/label lag
        if len(df) < max(args.horizon + 40, 120):
            print(f"[train] Skip {sym}: not enough hourly bars ({len(df)})")
            continue

        X, y = build_features_labels(df, args.horizon)
        if len(X) == 0 or len(y) == 0:
            print(f"[train] Skip {sym}: empty features/labels after cleaning")
            continue

        X_all.append(X)
        y_all.append(y)
        kept += 1
        # small pause to be gentle on Kraken
        time.sleep(0.5)

    if kept == 0:
        print("[train] No symbols yielded usable data. Try a shorter horizon (e.g., 96) or specify --symbols.")
        return

    X = pd.concat(X_all, axis=0).reset_index(drop=True)
    y = pd.concat(y_all, axis=0).reset_index(drop=True)
    print(f"[train] Dataset size: X={X.shape}, positives={int(y.sum())}, negatives={int((1-y).sum())}")

    # Train/val split (time-ordered)
    X_train, y_train, X_val, y_val = split_train_val(X, y, val_frac=0.2)

    # Model
    model = XGBClassifier(
        max_depth=4,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=2,
        random_state=42,
        tree_method="hist",
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    # Validation metrics + threshold suggestion
    val_probs = model.predict_proba(X_val)[:, 1]
    best_thr, table = eval_threshold(val_probs, y_val.values.astype(int))
    print("[train] Threshold sweep (precision/recall/f1):")
    for k, v in table.items():
        print(f"  thr={k}: precision={v['precision']:.2f}, recall={v['recall']:.2f}, f1={v['f1']:.2f}")
    print(f"[train] Suggested SCORE_THRESHOLD = {best_thr:.2f}")

    # Save
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    model.save_model(args.save)
    print(f"[train] Saved model to: {args.save}")

if __name__ == "__main__":
    main()
