# routes_stop_uplift.py

from datetime import timedelta
from typing import Optional, List, Tuple, Dict
from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select
import models as M
from settings import TARGET_PCT, MAX_HOLD_MINUTES, FEE_PCT, SLIPPAGE_PCT, ATR_LEN
from db import get_session  # same helper you use elsewhere to get a Session
from statistics import median

router = APIRouter(tags=["diagnostics"])

# ---- shared candle loader ----
def _load_candles(session: Session, symbol: str, start, end):
    q = (
        select(M.Candle)
        .where(M.Candle.symbol == symbol, M.Candle.ts >= start, M.Candle.ts <= end)
        .order_by(M.Candle.ts.asc())
    )
    return session.exec(q).all()

# ---- shared counterfactual checker (fixed stop percent) ----
def _would_be_win_for_stop(session: Session, t: M.Trade, stop_pct: float,
                           tp_pct: float, use_time_exit: bool = True):
    """
    Counterfactual with a *fixed* stop as a fraction (e.g., 0.02 = 2%).
    'Win' means the target (tp_pct) is hit before the stop,
    or (if neither hit) the trade ends above breakeven after costs.
    Uses minute candles.
    """
    entry = float(t.entry_px or 0.0)
    if entry <= 0:
        return False, "bad_entry", None

    # include costs so 'profitable' means net positive after fees + slippage
    be_edge = (FEE_PCT * 2.0) + (SLIPPAGE_PCT * 2.0)
    be_level = entry * (1.0 + be_edge)
    tp_level = entry * (1.0 + tp_pct + be_edge)
    stop_level = entry * (1.0 - stop_pct)

    horizon = t.exit_ts or (t.entry_ts + timedelta(minutes=MAX_HOLD_MINUTES))
    if use_time_exit and MAX_HOLD_MINUTES:
        horizon = min(horizon, t.entry_ts + timedelta(minutes=MAX_HOLD_MINUTES))

    candles = _load_candles(session, t.symbol, t.entry_ts, horizon)
    if not candles:
        return False, "no_candles", None

    for c in candles:
        px_low = getattr(c, "low", None) or c.close
        px_high = getattr(c, "high", None) or c.close

        if px_low <= stop_level:
            return False, "hit_stop", c.ts
        if px_high >= tp_level:
            return True, "hit_target", c.ts

    # neither hit — count as win if last close clears breakeven after costs
    if candles[-1].close >= be_level:
        return True, "ended_profitable", candles[-1].ts
    return False, "stopped_or_never_profitable", candles[-1].ts

# ---- helper: ATR at entry ----
def _atr_at_entry(session: Session, symbol: str, entry_ts, atr_len: int) -> Optional[float]:
    """
    Compute simple ATR (SMA of True Range) using the *atr_len* bars that end at or before entry_ts.
    If not enough history, return None.
    """
    # fetch atr_len+1 bars up to entry (need prev close for the first TR)
    rows_desc = session.exec(
        select(M.Candle)
        .where(M.Candle.symbol == symbol, M.Candle.ts <= entry_ts)
        .order_by(M.Candle.ts.desc())
        .limit(atr_len + 1)
    ).all()
    if len(rows_desc) < atr_len + 1:
        return None

    rows = list(reversed(rows_desc))  # chronological
    trs = []
    for i in range(1, len(rows)):
        h = float(rows[i].high if rows[i].high is not None else rows[i].close)
        l = float(rows[i].low  if rows[i].low  is not None else rows[i].close)
        pc = float(rows[i-1].close)
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)

    if len(trs) < atr_len:
        return None
    return sum(trs[-atr_len:]) / atr_len  # simple average TR

# ---------------- existing simple study (kept) ----------------
@router.get("/diagnostics/stop_uplift")
def stop_uplift(
    session: Session = Depends(get_session),
    min_stop: float = Query(0.005, description="Min stop pct as a fraction (0.005 = 0.5%)"),
    max_stop: float = Query(0.050, description="Max stop pct as a fraction (0.050 = 5.0%)"),
    step: float     = Query(0.0025, description="Grid step (0.0025 = 0.25%)"),
    tp_pct: float   = Query(0.037, description="Target used to define 'win' (0.037 = +3.7% TP1)"),
    use_time_exit: bool = Query(True, description="Honor MAX_HOLD_MINUTES horizon"),
    include_examples: bool = Query(True, description="Attach up to 10 examples per stop")
):
    losers = session.exec(
        select(M.Trade).where(M.Trade.result == "LOSS").order_by(M.Trade.entry_ts.asc())
    ).all()

    stops = []
    s = float(min_stop)
    while s <= float(max_stop) + 1e-12:
        stops.append(round(s, 6))
        s += float(step)

    out = {
        "losers_analyzed": len(losers),
        "grid": {"min_stop": min_stop, "max_stop": max_stop, "step": step},
        "target_for_win": tp_pct,
        "notes": [
            "Counterfactual based on minute bars; true tick path may differ.",
            "Costs accounted via fees+slippage buffer when checking target/breakeven."
        ],
        "stops": {}
    }

    for sp in stops:
        turned = 0
        examples = []
        for t in losers:
            ok, mode, when = _would_be_win_for_stop(session, t, sp, tp_pct, use_time_exit)
            if ok:
                turned += 1
                if include_examples and len(examples) < 10:
                    examples.append({
                        "symbol": t.symbol,
                        "entry_ts": t.entry_ts,
                        "exit_ts": t.exit_ts,
                        "entry_px": float(t.entry_px),
                        "exit_px": float(t.exit_px),
                        "mode": mode,
                        "when": when
                    })
        rate = (turned / len(losers)) if losers else 0.0
        out["stops"][f"{sp:.4f}"] = {
            "turned_winners": turned,
            "turn_rate": round(rate, 4),
            "examples": examples
        }

    return out

# ---------------- NEW: ATR × MIN_STOP grid ----------------
@router.get("/diagnostics/atr_grid")
def atr_grid(
    session: Session = Depends(get_session),
    # MIN_STOP_PCT axis
    min_floor_from: float = Query(0.005, description="Start MIN_STOP_PCT (0.005 = 0.5%)"),
    min_floor_to:   float = Query(0.030, description="End MIN_STOP_PCT (0.030 = 3.0%)"),
    min_floor_step: float = Query(0.0025, description="Step for MIN_STOP_PCT axis (0.25%)"),
    # ATR_MULT axis
    atr_mult_from: float  = Query(1.0, description="Start ATR_STOP_MULT"),
    atr_mult_to:   float  = Query(3.0, description="End ATR_STOP_MULT"),
    atr_mult_step: float  = Query(0.25, description="Step for ATR_STOP_MULT axis"),
    # other knobs
    tp_pct: float         = Query(0.037, description="Win defined as reaching TP1 (e.g. 0.037 = 3.7%)"),
    atr_len: int          = Query(None, description="Override ATR_LEN (defaults to settings.ATR_LEN)"),
    use_time_exit: bool   = Query(True, description="Honor MAX_HOLD_MINUTES horizon"),
    include_examples: bool = Query(True, description="Attach up to 5 examples per cell"),
):
    """
    For each losing trade, compute ATR at entry, convert ATR to a stop percent,
    then apply effective_stop = max(MIN_STOP_PCT, ATR_MULT * ATR / entry).
    Check if that would have flipped the loser into a winner (TP1 or profitable end).
    """
    losers = session.exec(
        select(M.Trade).where(M.Trade.result == "LOSS").order_by(M.Trade.entry_ts.asc())
    ).all()

    if not losers:
        return {"ok": True, "losers_analyzed": 0, "note": "No losing trades found."}

    floors, mults = [], []
    f = float(min_floor_from)
    while f <= float(min_floor_to) + 1e-12:
        floors.append(round(f, 6))
        f += float(min_floor_step)

    m = float(atr_mult_from)
    while m <= float(atr_mult_to) + 1e-12:
        mults.append(round(m, 6))
        m += float(atr_mult_step)

    use_atr_len = int(atr_len or ATR_LEN)

    # pre-compute ATR at entry for each trade
    atr_cache: Dict[int, Optional[float]] = {}
    for t in losers:
        atr_cache[t.id] = _atr_at_entry(session, t.symbol, t.entry_ts, use_atr_len)

    results = {
        "losers_analyzed": len(losers),
        "axes": {
            "min_stop_pct": {"from": min_floor_from, "to": min_floor_to, "step": min_floor_step, "values": floors},
            "atr_stop_mult": {"from": atr_mult_from, "to": atr_mult_to, "step": atr_mult_step, "values": mults},
            "atr_len": use_atr_len,
            "tp_pct": tp_pct,
        },
        "cells": {},  # "floor|mult" -> { turned, turn_rate, eligible, avg_effective_stop_pct, med_effective_stop_pct, examples }
        "best": []    # top 10 by turn_rate then turned
    }

    for floor in floors:
        for mult in mults:
            turned = 0
            eligible = 0
            eff_stops: List[float] = []
            examples = []
            for t in losers:
                entry = float(t.entry_px or 0.0)
                atr = atr_cache.get(t.id)
                if entry <= 0 or not atr:
                    continue  # not eligible (no ATR history)
                eligible += 1

                atr_stop_pct = (mult * atr) / entry
                effective_stop_pct = max(floor, atr_stop_pct)
                eff_stops.append(effective_stop_pct)

                ok, mode, when = _would_be_win_for_stop(session, t, effective_stop_pct, tp_pct, use_time_exit)
                if ok:
                    turned += 1
                    if include_examples and len(examples) < 5:
                        examples.append({
                            "symbol": t.symbol,
                            "entry_ts": t.entry_ts,
                            "exit_ts": t.exit_ts,
                            "entry_px": float(t.entry_px),
                            "effective_stop_pct": round(effective_stop_pct, 6),
                            "mode": mode,
                            "when": when
                        })

            rate = (turned / eligible) if eligible else 0.0
            key = f"{floor:.4f}|{mult:.2f}"
            results["cells"][key] = {
                "turned_winners": turned,
                "turn_rate": round(rate, 4),
                "eligible": eligible,
                "avg_effective_stop_pct": (round(sum(eff_stops)/len(eff_stops), 6) if eff_stops else None),
                "med_effective_stop_pct": (round(median(eff_stops), 6) if eff_stops else None),
                "examples": examples,
            }

    # pick top 10 cells by turn_rate, then by turned_winners
    ranked = sorted(
        results["cells"].items(),
        key=lambda kv: (kv[1]["turn_rate"], kv[1]["turned_winners"]),
        reverse=True
    )[:10]
    results["best"] = [
        {"cell": k, **v} for k, v in ranked
    ]

    return results
