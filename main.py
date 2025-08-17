# main.py

import asyncio
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import SQLModel, create_engine, Session, select

from models import *
from data import update_candles
from signal_engine import compute_signals
from risk import ensure_wallet, can_open_new_position, size_position
from sim import place_buy, mark_to_market_and_manage
from settings import UNIVERSE, POLL_SECONDS

# NOTE: allow SQLite across threads (background task + server)
engine = create_engine(
    "sqlite:///picklr.db",
    echo=False,
    connect_args={"check_same_thread": False}
)

app = FastAPI(title="autoPicklr Trading Simulator")

# ---- Admin controls ----
RUN_ENABLED = True  # simple on/off switch for the background loop

@app.get("/admin/status")
def admin_status():
    return {"run_enabled": RUN_ENABLED, "poll_seconds": POLL_SECONDS, "universe": UNIVERSE}

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
                        qty = size_position(w.balance_usd, sig.entry, sig.stop)
                        if qty > 0:
                            place_buy(s, sym, qty, sig.entry, sig.reason)
                            break  # one new entry per cycle
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/ping")
def ping():
    return {"message": "pong"}


# ---- Startup tasks ----
@app.on_event("startup")
def create_db_and_wallet():
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        ensure_wallet(s)

@app.post("/admin/reset")
def admin_reset():
    """Close all positions, clear orders/trades, reset wallet to $1000. Keeps candles."""
    with Session(engine) as s:
        # Close positions
        opens = s.exec(select(Position).where(Position.status == "OPEN")).all()
        for p in opens:
            p.status = "CLOSED"
        # Clear orders & trades
        s.exec(text("DELETE FROM 'order'"))  # table name is order; quoted because it's a keyword
        s.exec(text("DELETE FROM trade"))
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
    await asyncio.sleep(2)  # small delay so app has booted
    while True:
        try:
            if RUN_ENABLED:
                with Session(engine) as s:
                    await update_candles(s)
                    mark_to_market_and_manage(s)
                    if can_open_new_position(s):
                        for sym in UNIVERSE:
                            sigs = compute_signals(s, sym)
                            for sig in sigs:
                                w = s.get(Wallet, 1)
                                qty = size_position(w.balance_usd, sig.entry, sig.stop)
                                if qty > 0:
                                    place_buy(s, sym, qty, sig.entry, sig.reason)
                                    break  # one new entry per cycle
            else:
                print("[loop] paused")
        except Exception as e:
            print(f"[loop] error: {e}")
        await asyncio.sleep(POLL_SECONDS)



@app.on_event("startup")
async def start_trading_loop():
    asyncio.create_task(trading_loop())


# ---- Routes ----
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


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

        return {
            "equity_curve": cumulative_pnl,
            "current_equity": round(running_total, 2)
        }


# No second __main__ block — run with uvicorn/gunicorn command
# My name is Jeff