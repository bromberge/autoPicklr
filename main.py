# main.py

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import SQLModel, create_engine, Session, select

# --- app imports ---
from models import Wallet, Position, Trade, Order
from settings import POLL_SECONDS, UNIVERSE  # UNIVERSE is only a fallback if cache is empty
from signal_engine import compute_signals

# trading ops + wallet/equity mgmt
from sim import (
    ensure_wallet,
    can_open_new_position,
    size_position,
    place_buy,
    mark_to_market_and_manage,
)

# dynamic universe + candles
from universe import refresh_universe, get_active_universe, universe_stale
from data import update_candles_for

# ---- engine/app ----
engine = create_engine(
    "sqlite:///picklr.db",
    echo=False,
    connect_args={"check_same_thread": False},
)

# You can flip this to pause/resume trades without restarting the app
RUN_ENABLED = True

# -------- lifespan (startup/shutdown) --------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting autoPicklr Trading Simulator...")
    # Create tables + wallet
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        print("[startup] Ensuring wallet exists...")
        ensure_wallet(s)
        print("[startup] Refreshing universe once...")
        # Warm the universe cache so the first loop has data
        await refresh_universe(s)
    print("[startup] Starting trading loop...")
    loop_task = asyncio.create_task(trading_loop())
    print("[startup] Startup complete!")
    try:
        yield
    finally:
        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task


app = FastAPI(title="autoPicklr Trading Simulator", lifespan=lifespan)

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.get("/api/sim")
def sim_status():
    with Session(engine) as s:
        w = s.get(Wallet, 1)
        open_pos = s.exec(select(Position).where(Position.status == "OPEN")).all()
        last_trades = s.exec(select(Trade).order_by(Trade.id.desc())).all()[:10]

        all_trades = s.exec(select(Trade)).all()
        total_pnl = sum(t.pnl_usd or 0 for t in all_trades)
        wins = len([t for t in all_trades if t.result == "WIN"])
        losses = len([t for t in all_trades if t.result == "LOSS"])
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

        return {
            "wallet_equity": w.equity_usd if w else 1000,
            "wallet_balance": w.balance_usd if w else 1000,
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(win_rate, 1),
            "total_trades": wins + losses,
            "open_positions_count": len(open_pos),
            "open_positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "entry": p.avg_price,
                    "stop": p.stop,
                    "target": p.target
                } for p in open_pos
            ],
            "recent_trades": [
                {
                    "symbol": t.symbol,
                    "pnl": round(t.pnl_usd or 0, 2),
                    "result": t.result,
                    "entry_ts": t.entry_ts.isoformat() if t.entry_ts else None
                } for t in last_trades
            ]
        }

@app.get("/api/orders")
def orders():
    with Session(engine) as s:
        xs = s.exec(select(Order).order_by(Order.id.desc())).all()[:100]
        return [
            {
                "id": x.id,
                "ts": x.ts.isoformat(),
                "symbol": x.symbol,
                "side": x.side,
                "qty": x.qty,
                "price_req": x.price_req,
                "price_fill": x.price_fill,
                "status": x.status,
                "reason": x.reason
            } for x in xs
        ]

@app.get("/api/positions")
def positions():
    with Session(engine) as s:
        xs = s.exec(select(Position).where(Position.status == "OPEN")).all()
        return [
            {
                "id": x.id,
                "symbol": x.symbol,
                "qty": x.qty,
                "avg_price": x.avg_price,
                "opened_ts": x.opened_ts.isoformat(),
                "stop": x.stop,
                "target": x.target,
                "status": x.status
            } for x in xs
        ]

@app.get("/api/trades")
def trades():
    with Session(engine) as s:
        xs = s.exec(select(Trade).order_by(Trade.id.desc())).all()[:200]
        return [
            {
                "id": x.id,
                "symbol": x.symbol,
                "entry_ts": x.entry_ts.isoformat() if x.entry_ts else None,
                "exit_ts": x.exit_ts.isoformat() if x.exit_ts else None,
                "entry_px": x.entry_px,
                "exit_px": x.exit_px,
                "qty": x.qty,
                "pnl_usd": x.pnl_usd,
                "result": x.result
            } for x in xs
        ]

@app.get("/api/wallet")
def wallet():
    with Session(engine) as s:
        w = s.get(Wallet, 1)
        return {
            "balance_usd": w.balance_usd,
            "equity_usd": w.equity_usd,
            "updated_at": w.updated_at.isoformat()
        } if w else {}

@app.get("/api/performance")
def performance():
    with Session(engine) as s:
        trades = s.exec(select(Trade).order_by(Trade.entry_ts.asc())).all()
        cumulative_pnl = []
        running_total = 1000  # starting balance
        for trade in trades:
            if trade.pnl_usd is not None:
                running_total += trade.pnl_usd
                cumulative_pnl.append({
                    "date": (trade.exit_ts or trade.entry_ts).isoformat(),
                    "equity": round(running_total, 2),
                    "pnl": round(trade.pnl_usd, 2),
                })
        return {"equity_curve": cumulative_pnl, "current_equity": round(running_total, 2)}


# -------- trading loop --------
async def trading_loop():
    print("[loop] Starting trading loop...")
    await asyncio.sleep(2)
    print("[loop] Initial delay complete, entering main loop")
    while True:
        try:
            if RUN_ENABLED:
                print("[loop] Processing trading cycle...")
                with Session(engine) as s:
                    # 0) Refresh universe if stale (age > UNIVERSE_CACHE_MINUTES or empty)
                    if universe_stale(s):
                        print("[loop] Universe is stale — refreshing…")
                        await refresh_universe(s)

                    # 1) Get active symbols from cache; fallback to static if needed
                    active_syms = get_active_universe(s)
                    if not active_syms:
                        print("[loop] Universe cache empty — refreshing…")
                        await refresh_universe(s)
                        active_syms = get_active_universe(s)
                    if not active_syms:
                        active_syms = UNIVERSE  # last-resort fallback
                    print(f"[loop] Active symbols: {active_syms}")

                    # 2) Update candles for the active symbols
                    print("[loop] Updating candles...")
                    await update_candles_for(s, active_syms)

                    # 3) Manage open positions (TPs, BE, TSL, stops, targets, timeouts)
                    print("[loop] Managing positions...")
                    mark_to_market_and_manage(s)

                    # 4) Look for new entries (at most one per cycle)
                    print("[loop] Checking for new positions...")
                    if can_open_new_position(s):
                        for sym in active_syms:
                            sigs = compute_signals(s, sym)
                            for sig in sigs:
                                w = s.get(Wallet, 1)
                                cash = w.balance_usd if w else 0.0
                                qty = size_position(cash, sig.entry, sig.stop)
                                if qty > 0:
                                    print(f"[loop] Placing buy order for {sym}")
                                    place_buy(s, sym, qty, sig.entry, sig.reason, sig.stop, sig.target)
                                    break  # one new entry per cycle
                    print(f"[loop] Cycle complete, sleeping for {POLL_SECONDS}s")
            else:
                print("[loop] paused")
        except Exception as e:
            print(f"[loop] error: {e}")
            import traceback
            traceback.print_exc()
        await asyncio.sleep(POLL_SECONDS)
