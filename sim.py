# sim.py

from datetime import timedelta
from typing import Optional, List
from sqlmodel import Session, select
import models as M
from settings import (
    FEE_PCT, SLIPPAGE_PCT,
    MAX_OPEN_POSITIONS, MAX_TRADE_COST_PCT, MAX_GROSS_EXPOSURE_PCT,
    MIN_TRADE_NOTIONAL, MIN_TRADE_NOTIONAL_PCT, MIN_BREAKEVEN_EDGE_PCT,
    MAX_HOLD_MINUTES,
    TARGET_PCT, STOP_PCT,
    PTP_LEVELS, PTP_SIZES,
    BE_TRIGGER_PCT, BE_AFTER_FIRST_TP,
    TSL_ACTIVATE_AFTER_TP, TSL_ACTIVATE_PCT, TSL_PCT
)

EPSILON = 1e-9

from datetime import datetime, timezone

def _utcnow():
    """Naive UTC now (no tzinfo) for consistent DB storage."""
    return datetime.now(timezone.utc).replace(microsecond=0).replace(tzinfo=None)



def get_last_price(session: Session, symbol: str) -> Optional[float]:
    c = session.exec(
        select(M.Candle).where(M.Candle.symbol == symbol).order_by(M.Candle.ts.desc())
    ).first()
    return c.close if c else None

def _portfolio_market_value(session: Session) -> float:
    total = 0.0
    opens: List[M.Position] = session.exec(
        select(M.Position).where(M.Position.status == "OPEN")
    ).all()
    for p in opens:
        px = get_last_price(session, p.symbol)
        if px is not None:
            total += px * p.qty
    return total

def ensure_wallet(session: Session, start_usd: float = 1000.0) -> M.Wallet:
    w = session.get(M.Wallet, 1)
    if not w:
        w = M.Wallet(id=1, balance_usd=start_usd, equity_usd=start_usd, updated_at=_utcnow())
        session.add(w); session.commit()
    return w

def _sanitize_wallet(w: M.Wallet) -> None:
    if abs(w.balance_usd) < EPSILON: w.balance_usd = 0.0
    if abs(w.equity_usd)  < EPSILON: w.equity_usd  = 0.0
    w.balance_usd = float(round(w.balance_usd, 8))
    w.equity_usd  = float(round(w.equity_usd, 8))

def _recalc_equity(session: Session, w: M.Wallet) -> None:
    mv = _portfolio_market_value(session)
    w.equity_usd = w.balance_usd + mv
    w.updated_at = _utcnow()
    _sanitize_wallet(w)

def can_open_new_position(session: Session) -> bool:
    open_count = session.exec(select(M.Position).where(M.Position.status == "OPEN")).all()
    return len(open_count) < MAX_OPEN_POSITIONS

def size_position(cash_balance: float, entry: float, stop: float, risk_pct: float = 0.02) -> float:
    if entry <= 0 or stop <= 0 or entry <= stop:
        return 0.0
    risk_usd = cash_balance * risk_pct
    per_unit = entry - stop
    qty = risk_usd / per_unit
    return max(0.0, qty)

def _book_order(session: Session, symbol: str, side: str, qty: float,
                price_req: float, price_fill: float, status: str, note: str):
    session.add(M.Order(
        ts=_utcnow(), symbol=symbol, side=side, qty=qty,
        price_req=price_req, price_fill=price_fill,
        status=status, reason=note
    ))

def _add_trade(session: Session, symbol: str, entry_ts: datetime, entry_px: float,
               exit_px: float, qty: float):
    pnl = (exit_px - entry_px) * qty
    result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
    session.add(M.Trade(
        symbol=symbol, entry_ts=entry_ts, exit_ts=_utcnow(),
        entry_px=entry_px, exit_px=exit_px, qty=qty,
        pnl_usd=pnl, result=result
    ))

def _partial_close(session: Session, p: M.Position, qty: float, exit_px: float, reason: str):
    if qty <= 0 or qty > p.qty:
        return
    w = ensure_wallet(session)

    fill_px = exit_px * (1 - SLIPPAGE_PCT)
    fee = fill_px * qty * FEE_PCT
    proceeds = fill_px * qty - fee

    w.balance_usd += proceeds
    _add_trade(session, p.symbol, p.opened_ts, p.avg_price, fill_px, qty)
    _book_order(session, p.symbol, "SELL", qty, p.avg_price, fill_px, "FILLED", reason)

    p.qty -= qty
    if p.qty <= EPSILON:
        p.qty = 0.0
        p.status = "CLOSED"

    _recalc_equity(session, w)
    session.commit()

def _close_position(session: Session, p: M.Position, exit_px: float, reason: str):
    if p.qty <= 0:
        return
    _partial_close(session, p, p.qty, exit_px, reason)

def _activate_tsl_if_needed(p: M.Position, entry_px: float, last_px: float):
    if p.tsl_active:
        return
    trigger_by_pct = (last_px >= entry_px * (1 + TSL_ACTIVATE_PCT))
    trigger_by_tp  = p.tp1_done if TSL_ACTIVATE_AFTER_TP else False
    if trigger_by_tp or trigger_by_pct:
        p.tsl_active = True
        p.tsl_high = last_px

# sim.py (replace the whole place_buy function)

def place_buy(session: Session, symbol: str, qty: float, entry: float, reason: str,
  stop: Optional[float] = None, target: Optional[float] = None, score: Optional[float] = None):

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
    p = session.exec(select(M.Position).where(M.Position.symbol == symbol, M.Position.status == "OPEN")).first()
    if p:
        # average in
        total_qty = p.qty + buy_qty
        if total_qty > 0:
            p.avg_price = (p.avg_price * p.qty + fill_px * buy_qty) / total_qty
            p.qty = total_qty
        # keep existing stop/target/flags
    else:
        p = M.Position(
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
            tsl_high=None,
        )
        # TP1 absolute level (for dashboard)
        if PTP_LEVELS and len(PTP_LEVELS) >= 1:
            p.tp1_price = fill_px * (1 + PTP_LEVELS[0])
        # if the caller passed score explicitly, store as confidence
        if isinstance(score, (int, float)):
            p.score = float(score)
        session.add(p)

    # keep stats fresh
    refresh_position_stats(session, p)



    _recalc_equity(session, w)
    session.commit()


def refresh_position_stats(session: Session, p: M.Position):
    last_px = get_last_price(session, p.symbol)
    if last_px is None:
        return

    p.current_px = last_px
    try:
        p.pl_usd = (last_px - p.avg_price) * p.qty
        p.pl_pct = ((last_px / p.avg_price) - 1.0) * 100.0
    except Exception:
        p.pl_usd = 0.0
        p.pl_pct = 0.0

    # tp1 level (store once)
    if getattr(p, "tp1_price", None) is None and PTP_LEVELS:
        p.tp1_price = p.avg_price * (1 + PTP_LEVELS[0])

    # break-even price (only after BE is moved)
    p.be_price = p.avg_price if p.be_moved else None

    # trailing stop “visible” price if active (tsl mapped to current stop)
    p.tsl_price = p.stop if p.tsl_active else None

    # time in trade (minutes)
    if p.opened_ts:
        delta = _utcnow() - p.opened_ts
        p.time_in_trade_min = int(delta.total_seconds() // 60)


def mark_to_market_and_manage(session: Session):
    """Every cycle: manage TP1/TP2, Break-even, Trailing, Stop/Target, and Time exit."""
    w = ensure_wallet(session)

    opens: List[M.Position] = session.exec(select(M.Position).where(M.Position.status == "OPEN")).all()
    for p in opens:
        last_px = get_last_price(session, p.symbol)
        if last_px is None:
            continue

        # update live fields for the dashboard
        p.current_px = last_px
        p.pl_usd = (last_px - p.avg_price) * p.qty
        p.pl_pct = ((last_px / p.avg_price) - 1.0) * 100.0


        entry_px = p.avg_price  # effective entry for calculations
        p.current_px = last_px
        # live P/L
        p.pl_usd = (last_px - entry_px) * p.qty
        p.pl_pct = ((last_px / entry_px) - 1.0) * 100.0 if entry_px > 0 else None

        # ---- Partial Take-Profits ----
        tp_levels = [entry_px * (1 + g) for g in PTP_LEVELS]
        # (Your existing TP1/TP2 logic here; unchanged)
        if not p.tp1_done and len(tp_levels) >= 1 and last_px >= tp_levels[0]:
            sell_qty = p.qty * min(1.0, PTP_SIZES[0] if PTP_SIZES else 0.0)
            if sell_qty > 0:
                _partial_close(session, p, sell_qty, tp_levels[0], "TP1")
                p.tp1_done = True

        if p.status == "OPEN" and not p.tp2_done and len(tp_levels) >= 2 and last_px >= tp_levels[1]:
            sell_qty = p.qty * min(1.0, PTP_SIZES[1] if len(PTP_SIZES) >= 2 else 0.0)
            if sell_qty > 0:
                _partial_close(session, p, sell_qty, tp_levels[1], "TP2")
                p.tp2_done = True

        if p.status != "OPEN":
            continue

        # ---- Break-Even stop move ----
        if not p.be_moved:
            gain_ok = last_px >= entry_px * (1 + BE_TRIGGER_PCT)
            tp1_ok  = p.tp1_done if BE_AFTER_FIRST_TP else False
            if gain_ok or tp1_ok:
                p.stop = max(p.stop, entry_px)
                p.be_moved = True

        # Surface BE as a price for the dashboard when moved
        p.be_price = entry_px if p.be_moved else None

        # ---- Trailing stop activation + update ----
        _activate_tsl_if_needed(p, entry_px, last_px)
        if p.tsl_active:
            if p.tsl_high is None or last_px > p.tsl_high:
                p.tsl_high = last_px
            dyn_trail_stop = p.tsl_high * (1 - TSL_PCT)
            p.stop = max(p.stop, dyn_trail_stop)
            p.tsl_price = p.stop
            p.tsl_price = p.stop
        else:
            p.tsl_price = None

        # ---- Hard Stop / Target (for any remaining qty) ----
        if p.qty > 0:
            if last_px <= p.stop:
                _close_position(session, p, p.stop, "STOP/TSL")
                continue
            if last_px >= p.target:
                _close_position(session, p, p.target, "TARGET")
                continue

        # ---- Time-based exit ----
        if p.status == "OPEN" and p.opened_ts is not None:
            age = (_utcnow() - p.opened_ts)
            if age >= timedelta(minutes=MAX_HOLD_MINUTES):
                _close_position(session, p, last_px, "TIME")
                continue

    # refresh stats for all remaining open positions
    opens2: List[M.Position] = session.exec(select(M.Position).where(M.Position.status == "OPEN")).all()
    for p2 in opens2:
        refresh_position_stats(session, p2)


    # refresh account equity (wallet: cash + MV of positions)
    _recalc_equity(session, w)
    session.commit()

