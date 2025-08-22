# risk.py
# Purpose: position sizing with an optional boost when the model score is strong.

from settings import RISK_PCT_PER_TRADE, USE_MODEL, SCORE_THRESHOLD, RISK_MAX_MULTIPLIER

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def score_to_risk_multiplier(score: float) -> float:
    """
    Turns a model score into a safe size multiplier.
    Example:
      - If SCORE_THRESHOLD = 0.22 and score = 0.22 => 1.0 (base risk)
      - If score = 0.44 => about 2.0, but capped by RISK_MAX_MULTIPLIER
    If the model is off, we return 1.0.
    """
    if not USE_MODEL:
        return 1.0
    try:
        if SCORE_THRESHOLD <= 0:
            return 1.0
        raw = score / SCORE_THRESHOLD
        return _clamp(raw, 1.0, RISK_MAX_MULTIPLIER)
    except Exception:
        return 1.0

def size_position(cash_or_equity_usd: float, entry: float, stop: float, score: float = None) -> float:
    """
    Risk a fixed percent of equity per trade, scaled up gently if the score is strong.
    """
    if entry <= 0 or stop <= 0 or entry <= stop:
        return 0.0

    risk_pct = RISK_PCT_PER_TRADE
    if score is not None:
        risk_pct *= score_to_risk_multiplier(float(score))

    risk_usd = cash_or_equity_usd * risk_pct
    per_unit = entry - stop
    if per_unit <= 0:
        return 0.0

    qty = risk_usd / per_unit
    return max(0.0, float(qty))
