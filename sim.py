# sim.py

from datetime import datetime, timezone
from typing import Optional, Tuple, List
from sqlmodel import Session, select
from models import Wallet, Order, Position, Trade, Candle
from settings import FEE_PCT, SLIPPAGE_PCT, MAX_OPEN_POSITIONS

def _utcnow():
    return datetime.now(timezone.utc).replace(microsecond=0).replace(tzinfo=None)

def get_last_price(session: Session, symbol: str) -> Optional[float]:
    c = session.exec(
        select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.desc())
    ).first()
    return c.close if c else None

def ensure_wallet(session: Session, start_usd: float = 1000.0) -> Wallet:
    w = session.get(Wallet, 1)
    if not w:
        w = Wallet(id=1, balance_usd=start_usd, equity_usd=start_usd, updated_at=_utcnow())
        session.add(w)
        session.commit()
    return w

def can_open_new_position(session: Session) -> bool:
    open_count = session.exec(
        select(Position).where(Position.status == "OPEN")
    ).all()
    return len(open_count) < MAX_OPEN_POSITIONS

def size_position(cash_balance: float, entry: float, stop: float, risk_pct: float = 0.02) -> float:
    # simple risk-based sizing: $risk / (entry - stop)
    if entry <= 0 or stop <= 0 or entry <= stop:
        return 0.0
    risk_usd = cash_balance * risk_pct
    per_unit = entry - stop
    qty = risk_usd / per_unit
    return max(0.0, qty)

def _book_order(session: Session, symbol: str, side: str, qty: float, price_req: float, price_fill: float, status: str, reason: str):
    o = Order(
        ts=_utcnow(), symbol=symbol, side=side, qty=qty,
        price_req=price_req, price_fill=price_fill,
        status=status, reason=reason
    )
    session.add(o)

def _open_or_add_position(session: Session, symbol: str, qty: float, fill_px: float, stop: float, target: float):
    p = session.exec(
        select(Position).where(Position.symbol == symbol, Position.status == "OPEN")
    ).first()
    if p:
        total_qty = p.qty + qty
        if total_qty > 0:
            p.avg_price = (p.avg_price * p.qty + fill_px * qty) / total_qty
            p.qty = total_qty
    else:
        p = Position(
            symbol=symbol,
            qty=qty,
            avg_price=fill_px,
            opened_ts=_utcnow(),
            stop=stop,
            target=target,
            status="OPEN"
        )
        session.add(p)

def place_buy(session: Session, symbol: str, qty: float, entry: float, reason: str, stop: Optional[float] = None, target: Optional[float] = None):
    """Simulate a market buy: cap to affordable qty, deduct cash (incl. fees/slippage), create/merge position, log order."""
    if qty <= 0:
        return

    last_px = get_last_price(session, symbol) or entry
    fill_px = last_px * (1 + SLIPPAGE_PCT)

    w = ensure_wallet(session)

    # Cap to affordable size (including fees)
    unit_cost = fill_px * (1 + FEE_PCT)  # cost per unit (price + fee)
    if unit_cost <= 0:
        return
    max_affordable_qty = w.balance_usd / unit_cost
    buy_qty = min(qty, max_affordable_qty)

    MIN_QTY = 1e-6
    if buy_qty < MIN_QTY:
        return

    fee = fill_px * buy_qty * FEE_PCT
    cost = fill_px * buy_qty + fee

    # Deduct cash
    w.balance_usd -= cost
    w.updated_at = _utcnow()

    # Order + position
    _book_order(session, symbol, "BUY", buy_qty, entry, fill_px, "FILLED", reason)
    _open_or_add_position(session, symbol, buy_qty, fill_px, stop or (entry * 0.98), target or (entry * 1.05))

    session.commit()

def _close_position(session: Session, p: Position, exit_px: float, reason: str):
    """Close position, book trade, credit wallet, and mark result."""
    w = ensure_wallet(session)

    fill_px = exit_px * (1 - SLIPPAGE_PCT)
    fee = fill_px * p.qty * FEE_PCT
    proceeds = fill_px * p.qty - fee

    # Credit cash
    w.balance_usd += proceeds
    w.updated_at = _utcnow()

    # PnL (fees already handled via wallet debits/credits)
    pnl = (fill_px - p.avg_price) * p.qty
    result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
    t = Trade(
        symbol=p.symbol,
        entry_ts=p.opened_ts,
        exit_ts=_utcnow(),
        entry_px=p.avg_price,
        exit_px=fill_px,
        qty=p.qty,
        pnl_usd=pnl,
        result=result
    )
    session.add(t)

    # Position closed
    p.status = "CLOSED"

    # Exit order record
    _book_order(session, p.symbol, "SELL", p.qty, p.avg_price, fill_px, "FILLED", reason)

    session.commit()

def mark_to_market_and_manage(session: Session):
    """
    - Updates wallet equity = cash + sum(market_value of all open positions)
    - Closes positions that hit stop or target
    """
    w = ensure_wallet(session)

    opens: List[Position] = session.exec(
        select(Position).where(Position.status == "OPEN")
    ).all()

    # Check exits and compute market value
    to_close: List[Tuple[Position, float, str]] = []
    market_value = 0.0

    for p in opens:
        last_px = get_last_price(session, p.symbol)
        if last_px is None:
            continue

        # Exit rules
        if p.stop and last_px <= p.stop:
            to_close.append((p, p.stop, "STOP"))
            continue
        if p.target and last_px >= p.target:
            to_close.append((p, p.target, "TARGET"))
            continue

        market_value += last_px * p.qty

    # Correct equity calculation
    w.equity_usd = w.balance_usd + market_value
    w.updated_at = _utcnow()
    session.commit()

    # Apply exits
    for p, exit_px, reason in to_close:
        _close_position(session, p, exit_px, reason)
