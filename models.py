#models.py

from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field

class Wallet(SQLModel, table=True):
    id: int = Field(default=1, primary_key=True)
    balance_usd: float
    equity_usd: float
    updated_at: datetime

class Candle(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str
    ts: datetime  # bar close time (minute)
    open: float
    high: float
    low: float
    close: float
    volume: float

class Signal(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str
    ts: datetime
    score: float  # 0..1
    entry: float
    stop: float
    target: float
    reason: str

class Order(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime
    symbol: str
    side: str  # BUY/SELL
    qty: float
    price_req: float
    price_fill: Optional[float] = None
    status: str  # NEW/FILLED/CANCELLED/REJECTED
    sim: bool = True
    reason: Optional[str] = None

class Position(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str
    qty: float
    avg_price: float
    opened_ts: datetime
    stop: float
    target: float
    status: str  # OPEN/CLOSED

class Trade(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str
    entry_ts: datetime
    exit_ts: Optional[datetime] = None
    entry_px: float
    exit_px: Optional[float] = None
    qty: float
    pnl_usd: Optional[float] = None
    result: Optional[str] = None  # WIN/LOSS/TIMEOUT
