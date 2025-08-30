# brokers/sim_broker.py
from __future__ import annotations
from typing import Optional, List
from datetime import datetime, timezone
from sqlmodel import Session, select

from .base import Broker, Balance, OrderInfo, MarketMeta
import models as M
from sim import (
    get_last_price,
    place_buy,
    _utcnow,
    get_or_create_wallet,   # existing helper in your sim.py
)
from settings import FEE_PCT, SLIPPAGE_PCT

def _now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)

class SimBroker(Broker):
    name = "sim"

    def __init__(self, engine):
        self.engine = engine

    def get_balance(self) -> Balance:
        with Session(self.engine) as s:
            w = get_or_create_wallet(s)
            return Balance(cash_usd=float(w.balance_usd or 0.0),
                           equity_usd=float(w.equity_usd or 0.0))

    def get_market_meta(self, symbol: str) -> Optional[MarketMeta]:
        # Sim doesn’t enforce precision; provide neutral values.
        return MarketMeta(symbol=symbol, exch_symbol=symbol,
                          price_tick=1e-8, qty_step=1e-8, min_notional=0.0)

    def get_price(self, symbol: str) -> Optional[float]:
        with Session(self.engine) as s:
            return get_last_price(s, symbol)

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        price: Optional[float] = None,
        reason: Optional[str] = None,
        score: Optional[float] = None,
        session: Optional[Session] = None,
    ) -> OrderInfo:
        """
        BUY path: calls your existing place_buy() exactly as today.
        SELL path: if you ever need it later, you can wire to a sim close helper.
        Returns an OrderInfo by reading the last Order row.
        """
        if side.upper() != "BUY":
            # In your current flow you only buy here. Extend as needed.
            return OrderInfo(
                id=None, symbol=symbol, side=side, qty=0.0,
                price=price or 0.0, status="REJECTED", ts=_now_naive(),
                reason="SimBroker currently supports BUY only"
            )

        # Use the provided session when we’re already inside a with Session(engine) block.
        created_session = False
        s = session
        if s is None:
            s = Session(self.engine)
            created_session = True

        try:
            # Use your existing function (no modifications to sim.py)
            # place_buy books the Order + Position and commits.
            entry = price or (self.get_price(symbol) or 0.0)
            place_buy(s, symbol, qty, entry, reason or "", score=score)

            # read back the most recent BUY order for this symbol
            o = s.exec(
                select(M.Order)
                .where(M.Order.symbol == symbol, M.Order.side == "BUY")
                .order_by(M.Order.id.desc())
            ).first()

            if not o:
                return OrderInfo(
                    id=None, symbol=symbol, side="BUY", qty=qty,
                    price=entry, status="REJECTED", ts=_now_naive(),
                    reason="place_buy did not create an order row"
                )

            return OrderInfo(
                id=str(o.id),
                symbol=o.symbol,
                side=o.side,
                qty=float(o.qty),
                price=float(o.price_fill or o.price_req or 0.0),
                status=o.status,
                ts=o.ts,
                reason=o.reason,
                extra={"fee_pct": FEE_PCT, "slip_pct": SLIPPAGE_PCT, "score": (score or 0.0)},
            )
        finally:
            if created_session:
                s.close()

    # no-ops / simple stubs (expand later if useful)
    def cancel_order(self, order_id: str) -> bool:
        return False

    def fetch_open_orders(self) -> List[OrderInfo]:
        return []
