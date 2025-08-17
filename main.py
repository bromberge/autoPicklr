# main.py

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta  # if not already present
from settings import UNIVERSE_REFRESH_MINUTES

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

from settings import (
    MAX_OPEN_POSITIONS,
    MAX_NEW_POSITIONS_PER_CYCLE,
    SIGNAL_MIN_NOTIONAL_USD,
    FEE_PCT,
    SLIPPAGE_PCT,
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
from fastapi import APIRouter
admin = APIRouter()

@admin.get("/admin/next_signals")
def admin_next_signals():
    with Session(engine) as s:
        out = []
        w = s.get(Wallet, 1)
        if not w:
            return {"debug_mode": False, "candidates": [], "note": "no wallet"}
        for sym in UNIVERSE:
            sigs = compute_signals(s, sym)
            if not sigs:
                out.append({"symbol": sym, "signal": None})
                continue
            sig = max(sigs, key=lambda x: getattr(x, "score", 0.0))
            qty_est = size_position(w.balance_usd, sig.entry, sig.stop)
            est_fill = sig.entry * (1 + SLIPPAGE_PCT)
            unit_cost = est_fill * (1 + FEE_PCT)
            notional_est = qty_est * unit_cost if qty_est > 0 else 0.0
            out.append({
                "symbol": sym,
                "signal": {
                    "score": getattr(sig, "score", 0.0),
                    "entry": sig.entry,
                    "stop": sig.stop,
                    "target": sig.target,
                    "reason": sig.reason,
                    "qty_est": qty_est,
                    "notional_est": round(notional_est, 2),
                    "passes_min_notional": notional_est >= SIGNAL_MIN_NOTIONAL_USD
                }
            })
        # sort like the loop
        out_sorted = sorted(
            [o for o in out if o["signal"]],
            key=lambda o: o["signal"]["score"],
            reverse=True
        )
        return {"candidates_sorted": out_sorted[:10]}  # show top 10 for readability

app.include_router(admin)

# --- Admin: view current universe (symbols in DB) ---
@app.get("/admin/universe")
def admin_universe():
    from universe import UniversePair
    with Session(engine) as s:
        rows = s.exec(select(UniversePair).order_by(UniversePair.usd_vol_24h.desc())).all()
        return {
            "count": len(rows),
            "symbols": [r.symbol for r in rows],
        }

# --- Admin: force refresh from Kraken (manual) ---
@app.post("/admin/universe_refresh")
async def admin_universe_refresh():
    from universe import refresh_universe
    with Session(engine) as s:
        rows = await refresh_universe(s)
        return {"ok": True, "count": len(rows)}


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
        out = []
        for x in xs:
            out.append({
                "id": x.id,
                "symbol": x.symbol,
                "qty": x.qty,
                "avg_price": x.avg_price,
                "opened_ts": x.opened_ts.isoformat(),
                "stop": x.stop,
                "target": x.target,
                "status": x.status,
                # profit-protection flags
                "tp1_done": getattr(x, "tp1_done", False),
                "tp2_done": getattr(x, "tp2_done", False),
                "be_moved": getattr(x, "be_moved", False),
                "tsl_active": getattr(x, "tsl_active", False),
                "tsl_high": getattr(x, "tsl_high", None),
            })
        return out

@app.get("/api/pp_summary")
def profit_protection_summary():
    with Session(engine) as s:
        xs = s.exec(select(Position).where(Position.status == "OPEN")).all()
        return {
            "open_positions": len(xs),
            "tp1_done": sum(1 for p in xs if getattr(p, "tp1_done", False)),
            "tp2_done": sum(1 for p in xs if getattr(p, "tp2_done", False)),
            "be_moved": sum(1 for p in xs if getattr(p, "be_moved", False)),
            "tsl_active": sum(1 for p in xs if getattr(p, "tsl_active", False)),
        }


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
LAST_UNIVERSE_REFRESH = None

async def trading_loop():
    global LAST_UNIVERSE_REFRESH
    print("[loop] Starting trading loop...")
    await asyncio.sleep(2)
    print("[loop] Initial delay complete, entering main loop")
    while True:
        try:
            if RUN_ENABLED:
                print("[loop] Processing trading cycle...")
                with Session(engine) as s:

                    # --- scheduled universe refresh (only one strategy) ---
                    now = datetime.utcnow()
                    if (
                        LAST_UNIVERSE_REFRESH is None
                        or (now - LAST_UNIVERSE_REFRESH) >= timedelta(minutes=UNIVERSE_REFRESH_MINUTES)
                    ):
                        print("[loop] Universe cache empty/stale — refreshing…")
                        rows = await refresh_universe(s)  # reuse current session
                        print(f"[universe] Refreshed {len(rows)} USD/USDT pairs from Kraken.")
                        LAST_UNIVERSE_REFRESH = now
                    # --- end universe refresh ---

                    # 1) Get active symbols from cache; fallback to static if needed
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

                    # 4) Look for new entries (ranked; place at most N per cycle)
                    print("[loop] Checking for new positions...")

                    from settings import (
                        MAX_NEW_POSITIONS_PER_CYCLE,
                        SIGNAL_MIN_NOTIONAL_USD,
                        FEE_PCT,
                        SLIPPAGE_PCT,
                        MAX_OPEN_POSITIONS
                    )
                    from risk import size_position  # make sure this is also a top-level import

                    candidates = []
                    w = s.get(Wallet, 1)
                    if w is not None:
                        open_count = len(s.exec(select(Position).where(Position.status == "OPEN")).all())
                        slots_left = max(0, MAX_OPEN_POSITIONS - open_count)

                        if slots_left > 0:
                            # IMPORTANT: iterate over dynamic active_syms, not UNIVERSE
                            for sym in active_syms:
                                sigs = compute_signals(s, sym)
                                if not sigs:
                                    continue

                                # take best signal (by score)
                                sig = max(sigs, key=lambda x: getattr(x, "score", 0.0))

                                # estimate qty/notional before any order
                                qty_est = size_position(w.balance_usd, sig.entry, sig.stop)
                                if qty_est <= 0:
                                    continue

                                est_fill = sig.entry * (1 + SLIPPAGE_PCT)
                                unit_cost = est_fill * (1 + FEE_PCT)
                                notional_est = qty_est * unit_cost

                                if notional_est < SIGNAL_MIN_NOTIONAL_USD:
                                    continue

                                candidates.append((sym, sig, qty_est, notional_est))

                    # sort by score desc and place up to N
                    if candidates and slots_left > 0:
                        candidates.sort(key=lambda t: getattr(t[1], "score", 0.0), reverse=True)

                        placed = 0
                        for sym, sig, qty_est, _ in candidates:
                            if placed >= min(MAX_NEW_POSITIONS_PER_CYCLE, slots_left):
                                break
                            # re-check wallet just in case
                            w = s.get(Wallet, 1)
                            if w is None:
                                break

                            qty = size_position(w.balance_usd, sig.entry, sig.stop)
                            if qty <= 0:
                                continue

                            print(f"[loop] Placing buy order for {sym} (score={getattr(sig,'score',0.0):.2f})")
                            place_buy(s, sym, qty, sig.entry, sig.reason)
                            placed += 1

                    print(f"[loop] Cycle complete, sleeping for {POLL_SECONDS}s")
            else:
                print("[loop] paused")
        except Exception as e:
            print(f"[loop] error: {e}")
            import traceback
            traceback.print_exc()
        await asyncio.sleep(POLL_SECONDS)

