# models.py

from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime

class Wallet(SQLModel, table=True):
    id: int = Field(default=1, primary_key=True)
    balance_usd: float
    equity_usd: float
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,                         # set on INSERT
        sa_column=Column(DateTime, nullable=False, onupdate=datetime.utcnow)  # bump on UPDATE
    )

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

    # --- live stats for dashboard ---
    current_px: Optional[float] = None
    pl_usd: Optional[float] = None
    pl_pct: Optional[float] = None
    score: Optional[float] = None            # “confidence”
    be_price: Optional[float] = None         # break-even price if moved
    tp1_price: Optional[float] = None        # absolute price for TP1 (if configured)
    tsl_price: Optional[float] = None        # current trailing stop price (if active)
    tp1_done: bool = False
    tp2_done: bool = False
    be_moved: bool = False
    tsl_active: bool = False
    tsl_high: Optional[float] = None
    time_in_trade_min: Optional[float] = None
    custom_tsl_pct: Optional[float] = None  # per-position trailing % (e.g., 0.04 = 4%)


class Trade(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str
    entry_ts: datetime
    exit_ts: Optional[datetime] = None
    entry_px: float
    exit_px: Optional[float] = None
    qty: float
    pnl_usd: Optional[float] = None
    result: Optional[str] = None  # WIN/LOSS/FLAT/TIME
    duration_min: Optional[float] = None

# --- Simple daily metrics rollup ---
from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class MetricsDaily(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    date_utc: datetime               # midnight UTC for that day

    # account-level snapshot
    equity_usd: float
    cash_usd: float
    exposure_pct: float              # 0..100

    # trades exited on that day
    trades: int
    wins: int
    win_rate: float                  # 0..1
    avg_win_pct: float               # e.g., 0.054 = +5.4%
    avg_loss_pct: float              # e.g., -0.021 = -2.1%
    payoff: float                    # |avg_win| / |avg_loss|, 0 if no losses
    expectancy: float                # win_rate*avg_win + (1-win_rate)*avg_loss

    # optional cost buckets (populate later if you track exact fees/slippage)
    fees_usd: float = 0.0
    slippage_usd: float = 0.0
