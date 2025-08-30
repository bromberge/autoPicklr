# brokers/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, List, Dict
from datetime import datetime

@dataclass
class Balance:
    cash_usd: float
    equity_usd: float

@dataclass
class OrderInfo:
    id: Optional[str]          # exchange id or local id
    symbol: str
    side: str                  # "BUY" or "SELL"
    qty: float
    price: float               # fill price if FILLED, else requested
    status: str                # "NEW" | "FILLED" | "REJECTED" | "CANCELED" | "PARTIAL"
    ts: datetime
    reason: Optional[str] = None
    extra: Dict[str, float] = None  # e.g., fees, slippage, score, etc.

@dataclass
class MarketMeta:
    symbol: str                # app symbol, e.g., "BTC"
    exch_symbol: str           # venue symbol, e.g., "BTC/USDT" or "XBT/USD"
    price_tick: float
    qty_step: float
    min_notional: float

class Broker(Protocol):
    name: str

    # --- account / market meta ---
    def get_balance(self) -> Balance: ...
    def get_market_meta(self, symbol: str) -> Optional[MarketMeta]: ...

    # --- price snapshot (for sizing / UI) ---
    def get_price(self, symbol: str) -> Optional[float]: ...

    # --- order placement ---
    def place_order(
        self,
        symbol: str,
        side: str,                 # "BUY" | "SELL"
        qty: float,
        order_type: str = "market",
        price: Optional[float] = None,
        reason: Optional[str] = None,
        score: Optional[float] = None,
        session=None,             # pass your SQLModel Session so we can reuse DB when sim
    ) -> OrderInfo: ...

    # Optional helpers (can be no-ops on sim)
    def cancel_order(self, order_id: str) -> bool: ...
    def fetch_open_orders(self) -> List[OrderInfo]: ...
