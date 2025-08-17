# sim.py

from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List
from sqlmodel import Session, select
from models import Wallet, Order, Position, Trade, Candle
from settings import (
    FEE_PCT, SLIPPAGE_PCT,
    MAX_OPEN_POSITIONS, MAX_TRADE_COST_PCT, MAX_GROSS_EXPOSURE_PCT,
    MIN_TRADE_NOTIONAL, MIN_TRADE_NOTIONAL_PCT, MIN_BREAKEVEN_EDGE_PCT,
    MAX_HOLD_MINUTES,
    # strategy
    TARGET_PCT, STOP_PCT,
    # exits
    PTP_LEVELS, PTP_SIZES,
    BE_TRIGGER_PCT, BE_AFTER_FIRST_TP,
    TSL_ACTIVATE_AFTER_TP, TSL_ACTIVATE_PCT, TSL_PCT
)

EPSILON = 1e-9  # tiny float dust

# ---------- time ----------
def _utcnow():
    return datetime.now(timezone.utc).replace(microsecond=0).replace(tzinfo=None)

# ---------- lookups ----------
def get_last_price(session: Session, symbol: str) -> Optional[float]:
    c = session.exec(
        select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.desc())
    ).first()
    return c.close if c else None

def _portfolio_market_value(session: Session) -> float:
    total = 0.0
    opens: List[Position] = session.exec(select(Position).where(Position.status == "OPEN")).all()
    for p in opens:
        px = get_last_price(session, p.symbol)
        if px is not None:
            total += px * p.qty
    return total

# ---------- wallet helpers ----------
def ensure_wallet(session: Session, start_usd: float = 1000.0) -> Wallet:
    w = session.get(Wallet, 1)
    if not w:
        w = Wallet(id=1, balance_usd=start_usd, equity_usd=start_usd, updated_at=_utcnow())
        session.add(w); session.commit()
    return w

def _sanitize_wallet(w: Wallet) -> None:
    if abs(w.balance_usd) < EPSILON: w.balance_usd = 0.0
    if abs(w.equity_usd)  < EPSILON: w.equity_usd  = 0.0
    w.balance_usd = float(round(w.balance_usd, 8))
    w.equity_usd  = float(round(w.equity_usd, 8))

def _recalc_equity(session: Session, w: Wallet) -> None:
    mv = _portfolio_market_value(session)
    w.equity_usd = w.balance_usd + mv
    w.updated_at = _utcnow()
    _sanitize_wallet(w)

# ---------- policy helpers ----------
def can_open_new_position(session: Session) -> bool:
    open_count = session.exec(select(Position).where(Position.status == "OPEN")).all()
    return len(open_count) < MAX_OPEN_POSITIONS

def size_position(cash_balance: float, entry: float, stop: float, risk_pct: float = 0.02) -> float:
    if entry <= 0 or stop <= 0 or entry <= stop:
        return 0.0
    risk_usd = cash_balance * risk_pct
    per_unit = entry - stop
    qty = risk_usd / per_unit
    return max(0.0, qty)

# ---------- records ----------
def _book_order(session: Session, symbol: str, side: str, qty: float,
                price_req: float, price_fill: float, status: str, note: str):
    session.add(Order(
        ts=_utcnow(), symbol=symbol, side=side, qty=qty,
        price_req=price_req, price_fill=price_fill,
        status=status, reason=note
    ))

def _add_trade(session: Session, symbol: str, entry_ts: datetime, entry_px: float,
               exit_px: float, qty: float):
    pnl = (exit_px - entry_px) * qty
    result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
    session.add(Trade(
        symbol=symbol, entry_ts=entry_ts, exit_ts=_utcnow(),
        entry_px=entry_px, exit_px=exit_px, qty=qty,
        pnl_usd=pnl, result=result
    ))

def _partial_close(session: Session, p: Position, qty: float, exit_px: float, reason: str):
    """Sell part of a position; realize PnL on that slice; update wallet and position."""
    if qty <= 0 or qty > p.qty:
        return
    w = ensure_wallet(session)

    fill_px = exit_px * (1 - SLIPPAGE_PCT)
    fee = fill_px * qty * FEE_PCT
    proceeds = fill_px * qty - fee

    # Credit cash
    w.balance_usd += proceeds

    # Trade record for the sold slice
    _add_trade(session, p.symbol, p.opened_ts, p.avg_price, fill_px, qty)

    # Order record
    _book_order(session, p.symbol, "SELL", qty, p.avg_price, fill_px, "FILLED", reason)

    # Reduce position size; avg_price unchanged for remaining
    p.qty -= qty
    if p.qty <= EPSILON:
        p.qty = 0.0
        p.status = "CLOSED"

    _recalc_equity(session, w)
    session.commit()

def _close_position(session: Session, p: Position, exit_px: float, reason: str):
    """Close entire remaining position."""
    if p.qty <= 0:
        return
    _partial_close(session, p, p.qty, exit_px, reason)

def _activate_tsl_if_needed(p: Position, entry_px: float, last_px: float):
    """Start trailing if rule says 'after TP1' and TP1 is done, or after % gain."""
    if p.tsl_active:
        return
    trigger_by_pct = (last_px >= entry_px * (1 + TSL_ACTIVATE_PCT))
    trigger_by_tp  = p.tp1_done if TSL_ACTIVATE_AFTER_TP else False
    if trigger_by_tp or trigger_by_pct:
        p.tsl_active = True
        p.tsl_high = last_px

# ---------- trading ops ----------
def place_buy(session: Session, symbol: str, qty: float, entry: float, reason: str,
              stop: Optional[float] = None, target: Optional[float] = None):
    """
    Enforces:
      - Breakeven-aware per-unit edge (must beat fees+slippage by MIN_BREAKEVEN_EDGE_PCT*entry)
      - Cash cap, Per-trade cap, Gross exposure cap
      - Notional floor
    Sets up position with TP/BE/TSL state for later management.
    """
    if qty <= 0:
        return

    w = ensure_wallet(session)
    _recalc_equity(session, w)

    last_px = get_last_price(session, symbol) or entry
    fill_px = last_px * (1 + SLIPPAGE_PCT)

    # Use provided stop/target if available; otherwise defaults
    tgt = target or (entry * (1 + TARGET_PCT))
    stp = stop   or (entry * (1 - STOP_PCT))

    # ---- Breakeven-aware per-unit edge ----
    buy_cost = fill_px * (1 + FEE_PCT)
    sell_net = tgt * (1 - SLIPPAGE_PCT) * (1 - FEE_PCT)
    per_unit_edge = sell_net - buy_cost
    min_edge = entry * MIN_BREAKEVEN_EDGE_PCT
    if per_unit_edge < min_edge:
        _book_order(session, symbol, "BUY", 0.0, entry, fill_px, "REJECTED",
                    f"Breakeven check failed: edge={per_unit_edge:.6f} < min={min_edge:.6f}")
        session.commit()
        return

    # ---- Caps ----
    unit_cost = buy_cost
    if unit_cost <= 0:
        return
    cash_cap_qty = w.balance_usd / unit_cost
    max_trade_cost = w.equity_usd * MAX_TRADE_COST_PCT
    trade_cost_cap_qty = max_trade_cost / unit_cost if unit_cost > 0 else 0.0
    current_mv = _portfolio_market_value(session)
    max_gross_exposure = w.equity_usd * MAX_GROSS_EXPOSURE_PCT
    remaining_exposure_dollars = max(0.0, max_gross_exposure - current_mv)
    exposure_cap_qty = remaining_exposure_dollars / fill_px if fill_px > 0 else 0.0

    buy_qty = min(qty, cash_cap_qty, trade_cost_cap_qty, exposure_cap_qty)

    # ---- Notional floor ----
    notional = buy_qty * fill_px
    min_notional = max(MIN_TRADE_NOTIONAL, w.equity_usd * MIN_TRADE_NOTIONAL_PCT)
    if buy_qty < 1e-6 or notional < min_notional:
        _book_order(session, symbol, "BUY", 0.0, entry, fill_px, "REJECTED",
                    f"Notional too small: {notional:.2f} < min {min_notional:.2f}")
        session.commit()
        return

    # ---- Execute buy ----
    fee = fill_px * buy_qty * FEE_PCT
    cost = fill_px * buy_qty + fee
    w.balance_usd -= cost

    _book_order(session, symbol, "BUY", buy_qty, entry, fill_px, "FILLED",
                f"{reason} | cost={cost:.6f}")

    # Open or add position
    p = session.exec(select(Position).where(Position.symbol == symbol, Position.status == "OPEN")).first()
    if p:
        # average in
        total_qty = p.qty + buy_qty
        if total_qty > 0:
            p.avg_price = (p.avg_price * p.qty + fill_px * buy_qty) / total_qty
            p.qty = total_qty
            # keep existing stop/target/flags
    else:
        p = Position(
            symbol=symbol,
            qty=buy_qty,
            avg_price=fill_px,
            opened_ts=_utcnow(),
            stop=stp,
            target=tgt,
            status="OPEN",
            tp1_done=False,
            tp2_done=False,
            be_moved=False,
            tsl_active=False,
            tsl_high=None
        )
        session.add(p)

    _recalc_equity(session, w)
    session.commit()

def mark_to_market_and_manage(session: Session):
    """Every cycle: manage TP1/TP2, Break-even, Trailing, Stop/Target, and Time exit."""
    w = ensure_wallet(session)

    opens: List[Position] = session.exec(select(Position).where(Position.status == "OPEN")).all()
    for p in opens:
        last_px = get_last_price(session, p.symbol)
        if last_px is None:
            continue

        entry_px = p.avg_price  # we treat avg fill as effective entry

        # ---- Partial Take-Profits (TP1, TP2) ----
        # Build absolute price levels from entry for clarity
        tp_levels = [entry_px * (1 + g) for g in PTP_LEVELS]
        tp_sizes  = PTP_SIZES[:] + [0.0] * max(0, len(PTP_LEVELS) - len(PTP_SIZES))  # safe align

        # TP1
        if not p.tp1_done and len(tp_levels) >= 1 and last_px >= tp_levels[0]:
            sell_qty = p.qty * min(1.0, tp_sizes[0])
            if sell_qty > 0:
                _partial_close(session, p, sell_qty, tp_levels[0], "TP1")
                p.tp1_done = True

        # TP2
        if not p.tp2_done and len(tp_levels) >= 2 and last_px >= tp_levels[1] and p.status == "OPEN":
            sell_qty = p.qty * min(1.0, tp_sizes[1])
            if sell_qty > 0:
                _partial_close(session, p, sell_qty, tp_levels[1], "TP2")
                p.tp2_done = True

        if p.status != "OPEN":
            continue  # closed by TP partials if qty went to zero

        # ---- Break-Even stop move ----
        if not p.be_moved:
            gain_ok = last_px >= entry_px * (1 + BE_TRIGGER_PCT)
            tp1_ok  = p.tp1_done if BE_AFTER_FIRST_TP else False
            if gain_ok or tp1_ok:
                # move stop to entry or higher (never lower)
                p.stop = max(p.stop, entry_px)
                p.be_moved = True

        # ---- Trailing stop activation + update ----
        _activate_tsl_if_needed(p, entry_px, last_px)
        if p.tsl_active:
            if p.tsl_high is None or last_px > p.tsl_high:
                p.tsl_high = last_px
            dyn_trail_stop = p.tsl_high * (1 - TSL_PCT)
            # Stop can only go UP (never loosen)
            p.stop = max(p.stop, dyn_trail_stop)

        # ---- Hard Stop / Original Target (for any remaining qty) ----
        # We still honor target for any leftover if not fully scaled out
        if p.qty > 0:
            # stop hit?
            if last_px <= p.stop:
                _close_position(session, p, p.stop, "STOP/TSL")
                continue
            # target hit (remaining)
            if last_px >= p.target:
                _close_position(session, p, p.target, "TARGET")
                continue

        # ---- Time-based exit ----
        if p.status == "OPEN" and p.opened_ts is not None:
            age = (_utcnow() - p.opened_ts)
            if age >= timedelta(minutes=MAX_HOLD_MINUTES):
                _close_position(session, p, last_px, "TIME")
                continue

    # finally, refresh account equity
    _recalc_equity(session, w)
    session.commit()
