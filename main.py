# main.py

import asyncio
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import SQLModel, create_engine, Session, select
from sqlalchemy import text
from sqlmodel import desc, asc
from contextlib import asynccontextmanager

from models import *
from data import update_candles
from signal_engine import compute_signals
from risk import ensure_wallet, can_open_new_position, size_position
from sim import place_buy, mark_to_market_and_manage
from settings import UNIVERSE, POLL_SECONDS

# NOTE: allow SQLite across threads (background task + server)
engine = create_engine("sqlite:///picklr.db",
                       echo=False,
                       connect_args={"check_same_thread": False})

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[startup] Creating database tables...")
    SQLModel.metadata.create_all(engine)
    print("[startup] Ensuring wallet exists...")
    with Session(engine) as s:
        ensure_wallet(s)
    print("[startup] Starting trading loop...")
    # Start trading loop
    asyncio.create_task(trading_loop())
    print("[startup] Startup complete!")
    yield
    # Shutdown (nothing needed for now)
    print("[shutdown] Server shutting down...")

app = FastAPI(title="autoPicklr Trading Simulator", lifespan=lifespan)

# ---- Admin controls ----
RUN_ENABLED = True  # simple on/off switch for the background loop


@app.get("/admin/status")
def admin_status():
    return {
        "run_enabled": RUN_ENABLED,
        "poll_seconds": POLL_SECONDS,
        "universe": UNIVERSE
    }


@app.post("/admin/pause")
def admin_pause():
    global RUN_ENABLED
    RUN_ENABLED = False
    return {"ok": True, "run_enabled": RUN_ENABLED}


@app.post("/admin/resume")
def admin_resume():
    global RUN_ENABLED
    RUN_ENABLED = True
    return {"ok": True, "run_enabled": RUN_ENABLED}


@app.post("/admin/tick")
async def admin_tick():
    """Run ONE full trading cycle on demand (useful for testing)."""
    try:
        with Session(engine) as s:
            await update_candles(s)
            mark_to_market_and_manage(s)
            if can_open_new_position(s):
                for sym in UNIVERSE:
                    sigs = compute_signals(s, sym)
                    for sig in sigs:
                        w = s.get(Wallet, 1)
                        if w is not None:
                            qty = size_position(w.balance_usd, sig.entry, sig.stop)
                        else:
                            qty = 0
                        if qty > 0:
                            place_buy(s, sym, qty, sig.entry, sig.reason, sig.stop, sig.target)
                            break  # one new entry per cycle
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/admin/diag")
def admin_diag():
    with Session(engine) as s:
        w = s.get(Wallet, 1)
        cash = w.balance_usd if w else None
        # peek at last order for context
        last_order_query = select(Order).order_by(desc(Order.id))
        last_order = s.exec(last_order_query).first()
        return {
            "cash": cash,
            "last_order": {
                "symbol": last_order.symbol,
                "side": last_order.side,
                "qty": last_order.qty,
                "price_fill": last_order.price_fill,
                "note": last_order.reason
            } if last_order else None
        }


# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/ping")
def ping():
    return {"message": "pong"}


# ---- Startup tasks ----
# This is now handled in lifespan function


@app.post("/admin/reset")
def admin_reset():
    """Hard reset: close positions, clear orders/trades, reset wallet to $1000. Keeps candles."""
    with Session(engine) as s:
        # Close all open positions
        opens = s.exec(select(Position).where(Position.status == "OPEN")).all()
        for p in opens:
            p.status = "CLOSED"

        # Clear orders & trades (use raw SQL because 'order' is a reserved word)
        s.execute(text('DELETE FROM "order"'))
        s.execute(text("DELETE FROM trade"))
        s.commit()

        # Reset wallet
        w = s.get(Wallet, 1)
        if w:
            w.balance_usd = 1000.0
            w.equity_usd = 1000.0
            w.updated_at = datetime.utcnow()
        else:
            s.add(Wallet(id=1, balance_usd=1000.0, equity_usd=1000.0, updated_at=datetime.utcnow()))
        s.commit()

    return {"ok": True}



async def trading_loop():
    print("[loop] Starting trading loop...")
    await asyncio.sleep(2)  # small delay so app has booted
    print("[loop] Initial delay complete, entering main loop")
    while True:
        try:
            if RUN_ENABLED:
                print("[loop] Processing trading cycle...")
                with Session(engine) as s:
                    print("[loop] Updating candles...")
                    await update_candles(s)
                    print("[loop] Managing positions...")
                    mark_to_market_and_manage(s)
                    print("[loop] Checking for new positions...")
                    if can_open_new_position(s):
                        for sym in UNIVERSE:
                            sigs = compute_signals(s, sym)
                            for sig in sigs:
                                w = s.get(Wallet, 1)
                                if w is not None:
                                    qty = size_position(w.balance_usd, sig.entry, sig.stop)
                                else:
                                    qty = 0
                                if qty > 0:
                                    print(f"[loop] Placing buy order for {sym}")
                                    place_buy(s, sym, qty, sig.entry,
                                              sig.reason)
                                    break  # one new entry per cycle
                    print(f"[loop] Cycle complete, sleeping for {POLL_SECONDS}s")
            else:
                print("[loop] paused")
        except Exception as e:
            print(f"[loop] error: {e}")
            import traceback
            traceback.print_exc()
        await asyncio.sleep(POLL_SECONDS)


# This is now handled in lifespan function


# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/sim")
def sim_status():
    with Session(engine) as s:
        w = s.get(Wallet, 1)
        open_pos = s.exec(
            select(Position).where(Position.status == "OPEN")).all()
        last_trades_query = select(Trade).order_by(desc(Trade.id))
        last_trades = s.exec(last_trades_query).all()[:10]

        all_trades = s.exec(select(Trade)).all()
        total_pnl = sum(t.pnl_usd or 0 for t in all_trades)
        wins = len([t for t in all_trades if t.result == "WIN"])
        losses = len([t for t in all_trades if t.result == "LOSS"])
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

        return {
            "wallet_equity":
            w.equity_usd if w else 1000,
            "wallet_balance":
            w.balance_usd if w else 1000,
            "total_pnl":
            round(total_pnl, 2),
            "win_rate":
            round(win_rate, 1),
            "total_trades":
            wins + losses,
            "open_positions_count":
            len(open_pos),
            "open_positions": [{
                "symbol": p.symbol,
                "qty": p.qty,
                "entry": p.avg_price,
                "stop": p.stop,
                "target": p.target
            } for p in open_pos],
            "recent_trades": [{
                "symbol":
                t.symbol,
                "pnl":
                round(t.pnl_usd or 0, 2),
                "result":
                t.result,
                "entry_ts":
                t.entry_ts.isoformat() if t.entry_ts else None
            } for t in last_trades]
        }


@app.get("/api/orders")
def orders():
    with Session(engine) as s:
        orders_query = select(Order).order_by(desc(Order.id))
        xs = s.exec(orders_query).all()[:100]
        return [{
            "id": x.id,
            "ts": x.ts.isoformat(),
            "symbol": x.symbol,
            "side": x.side,
            "qty": x.qty,
            "price_req": x.price_req,
            "price_fill": x.price_fill,
            "status": x.status,
            "reason": x.reason
        } for x in xs]


@app.get("/api/positions")
def positions():
    with Session(engine) as s:
        xs = s.exec(select(Position).where(Position.status == "OPEN")).all()
        return [{
            "id": x.id,
            "symbol": x.symbol,
            "qty": x.qty,
            "avg_price": x.avg_price,
            "opened_ts": x.opened_ts.isoformat(),
            "stop": x.stop,
            "target": x.target,
            "status": x.status
        } for x in xs]


@app.get("/api/trades")
def trades():
    with Session(engine) as s:
        trades_query = select(Trade).order_by(desc(Trade.id))
        xs = s.exec(trades_query).all()[:200]
        return [{
            "id": x.id,
            "symbol": x.symbol,
            "entry_ts": x.entry_ts.isoformat() if x.entry_ts else None,
            "exit_ts": x.exit_ts.isoformat() if x.exit_ts else None,
            "entry_px": x.entry_px,
            "exit_px": x.exit_px,
            "qty": x.qty,
            "pnl_usd": x.pnl_usd,
            "result": x.result
        } for x in xs]


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
        performance_query = select(Trade).order_by(asc(Trade.entry_ts))
        trades = s.exec(performance_query).all()

        cumulative_pnl = []
        running_total = 1000  # starting balance

        for trade in trades:
            if trade.pnl_usd is not None:
                running_total += trade.pnl_usd
                cumulative_pnl.append({
                    "date": (trade.exit_ts or trade.entry_ts).isoformat(),
                    "equity":
                    round(running_total, 2),
                    "pnl":
                    round(trade.pnl_usd, 2),
                })

        return {
            "equity_curve": cumulative_pnl,
            "current_equity": round(running_total, 2)
        }

from signal_engine import compute_signals
from settings import ENABLE_DEBUG_SIGNALS, DET_EMA_SHORT, DET_EMA_LONG, MIN_BREAKOUT_PCT, MIN_VOLUME_USD, CHOOSER_THRESHOLD, BREAKOUT_LOOKBACK, STOP_PCT, TARGET_PCT

@app.get("/admin/next_signals")
def admin_next_signals():
    """Preview the next signals per symbol using the CURRENT strategy settings."""
    out = []
    with Session(engine) as s:
        for sym in UNIVERSE:
            sigs = compute_signals(s, sym)
            # return top signal (or empty)
            if sigs:
                sig = sigs[0]
                out.append({
                    "symbol": sym,
                    "score": sig.score,
                    "entry": round(sig.entry, 6),
                    "stop": round(sig.stop, 6),
                    "target": round(sig.target, 6),
                    "reason": sig.reason
                })
            else:
                out.append({"symbol": sym, "signal": None})
    return {
        "debug_mode": ENABLE_DEBUG_SIGNALS,
        "rules": {
            "ema_short": DET_EMA_SHORT,
            "ema_long": DET_EMA_LONG,
            "breakout_lookback": BREAKOUT_LOOKBACK,
            "min_breakout_pct": MIN_BREAKOUT_PCT,
            "min_volume_usd": MIN_VOLUME_USD,
            "chooser_threshold": CHOOSER_THRESHOLD,
            "stop_pct": STOP_PCT,
            "target_pct": TARGET_PCT,
        },
        "signals": out
    }

