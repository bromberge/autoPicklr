#main.py

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
import uvicorn

engine = create_engine("sqlite:///picklr.db", echo=False)

app = FastAPI(title="autoPicklr Trading Simulator")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/ping")
def ping():
    return {"message": "pong"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
    
@app.on_event("startup")
def startup():
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        ensure_wallet(s)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/sim")
def sim_status():
    with Session(engine) as s:
        w = s.get(Wallet, 1)
        open_pos = s.exec(select(Position).where(Position.status=="OPEN")).all()
        last_trades = s.exec(select(Trade).order_by(Trade.id.desc())).all()[:10]
        
        # Calculate total P&L
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
            "open_positions": [{"symbol": p.symbol, "qty": p.qty, "entry": p.avg_price, "stop": p.stop, "target": p.target} for p in open_pos],
            "recent_trades": [{"symbol": t.symbol, "pnl": round(t.pnl_usd or 0, 2), "result": t.result, "entry_ts": t.entry_ts.isoformat() if t.entry_ts else None} for t in last_trades]
        }

@app.get("/api/orders")
def orders():
    with Session(engine) as s:
        xs = s.exec(select(Order).order_by(Order.id.desc())).all()[:100]
        return [{"id": x.id, "ts": x.ts.isoformat(), "symbol": x.symbol, "side": x.side, "qty": x.qty, "price_req": x.price_req, "price_fill": x.price_fill, "status": x.status, "reason": x.reason} for x in xs]

@app.get("/api/positions")
def positions():
    with Session(engine) as s:
        xs = s.exec(select(Position).where(Position.status=="OPEN")).all()
        return [{"id": x.id, "symbol": x.symbol, "qty": x.qty, "avg_price": x.avg_price, "opened_ts": x.opened_ts.isoformat(), "stop": x.stop, "target": x.target, "status": x.status} for x in xs]

@app.get("/api/trades")
def trades():
    with Session(engine) as s:
        xs = s.exec(select(Trade).order_by(Trade.id.desc())).all()[:200]
        return [{"id": x.id, "symbol": x.symbol, "entry_ts": x.entry_ts.isoformat() if x.entry_ts else None, "exit_ts": x.exit_ts.isoformat() if x.exit_ts else None, "entry_px": x.entry_px, "exit_px": x.exit_px, "qty": x.qty, "pnl_usd": x.pnl_usd, "result": x.result} for x in xs]

@app.get("/api/wallet")
def wallet():
    with Session(engine) as s:
        w = s.get(Wallet, 1)
        return {"balance_usd": w.balance_usd, "equity_usd": w.equity_usd, "updated_at": w.updated_at.isoformat()} if w else {}

@app.get("/api/performance")
def performance():
    with Session(engine) as s:
        trades = s.exec(select(Trade).order_by(Trade.entry_ts.asc())).all()
        
        # Calculate cumulative P&L over time
        cumulative_pnl = []
        running_total = 1000  # Starting balance
        
        for trade in trades:
            if trade.pnl_usd is not None:
                running_total += trade.pnl_usd
                cumulative_pnl.append({
                    "date": trade.exit_ts.isoformat() if trade.exit_ts else trade.entry_ts.isoformat(),
                    "equity": round(running_total, 2),
                    "pnl": round(trade.pnl_usd, 2)
                })
        
        return {
            "equity_curve": cumulative_pnl,
            "current_equity": round(running_total, 2)
        }

async def trading_loop():
    await asyncio.sleep(2)
    while True:
        try:
            with Session(engine) as s:
                # 1) fetch minute bars
                await update_candles(s)

                # 2) exit checks / mark-to-market
                mark_to_market_and_manage(s)

                # 3) new entries if room
                if can_open_new_position(s):
                    for sym in UNIVERSE:
                        sigs = compute_signals(s, sym)
                        for sig in sigs:
                            # size & place buy
                            w = s.get(Wallet, 1)
                            qty = size_position(w.balance_usd, sig.entry, sig.stop)
                            if qty <= 0: 
                                continue
                            place_buy(s, sym, qty, sig.entry, sig.reason)
                            break  # one new entry per cycle
        except Exception as e:
            print(f"Trading loop error: {e}")
        await asyncio.sleep(POLL_SECONDS)

@app.on_event("startup")
async def start_trading_loop():
    asyncio.create_task(trading_loop())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
