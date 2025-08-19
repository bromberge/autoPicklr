# main.py

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta 
from settings import UNIVERSE_REFRESH_MINUTES
from sim import get_last_price
from models import Wallet, Order, Position, Trade
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import SQLModel, create_engine, Session, select
from fastapi import Query
from typing import Any, Dict
from sqlmodel import select
from settings import (
    MIN_BREAKOUT_PCT, SIGNAL_MIN_NOTIONAL_USD, FEE_PCT, SLIPPAGE_PCT,
    REQUIRE_BREAKOUT, COOLDOWN_MINUTES
)
from risk import size_position

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

from sqlalchemy import text

def _column_missing(conn, table: str, col: str) -> bool:
    info = conn.execute(text(f"PRAGMA table_info('{table}')")).fetchall()
    names = {row[1] for row in info}  # row[1] is column name
    return col not in names

def _eval_candidate(session, w, sym, sig) -> Dict[str, Any]:
    """
    Run the exact same gates the trading_loop uses, but instead of placing orders,
    return reasons/metrics so we can see why a symbol was skipped or kept.
    """
    reasons = []
    ok = True

    # pull numbers with safe defaults
    entry = float(getattr(sig, "entry", 0.0) or 0.0)
    stop  = float(getattr(sig, "stop",  0.0) or 0.0)
    target= float(getattr(sig, "target",0.0) or 0.0)
    score = float(getattr(sig, "score", 0.0) or 0.0)
    brk   = float(getattr(sig, "breakout_pct", 0.0) or 0.0)
    rel   = float(getattr(sig, "ema_spread", 0.0) or 0.0)

    if REQUIRE_BREAKOUT and brk < MIN_BREAKOUT_PCT:
        ok = False; reasons.append(f"breakout<{MIN_BREAKOUT_PCT:.4f}")

    if not cool_down_ok(session, sym, COOLDOWN_MINUTES):
        ok = False; reasons.append(f"cooldown<{COOLDOWN_MINUTES}m")

    # position sizing & notional checks
    qty_est = size_position(w.balance_usd, entry, stop)
    if qty_est <= 0:
        ok = False; reasons.append("qty_est<=0")

    est_fill = entry * (1 + SLIPPAGE_PCT)
    unit_cost = est_fill * (1 + FEE_PCT)
    notional_est = qty_est * unit_cost
    if notional_est < SIGNAL_MIN_NOTIONAL_USD:
        ok = False; reasons.append(f"notional<{SIGNAL_MIN_NOTIONAL_USD}")

    # breakeven/edge sanity (mirrors place_buy test roughly)
    buy_cost = est_fill * (1 + FEE_PCT)
    sell_net = target * (1 - SLIPPAGE_PCT) * (1 - FEE_PCT)
    if sell_net - buy_cost <= 0:
        ok = False; reasons.append("no_edge")

    return {
        "symbol": sym,
        "ok": ok,
        "reasons": reasons,
        "score": score,
        "breakout_pct": brk,
        "ema_spread": rel,
        "entry": entry,
        "stop": stop,
        "target": target,
        "qty_est": qty_est if qty_est > 0 else 0.0,
        "notional_est": notional_est if notional_est > 0 else 0.0,
    }

def migrate_db(engine):
    with engine.connect() as conn:
        # Position columns
        pos_cols = [
            ("current_px",     "REAL"),
            ("pl_usd",         "REAL"),
            ("pl_pct",         "REAL"),
            ("score",          "REAL"),
            ("be_price",       "REAL"),
            ("tp1_price",      "REAL"),
            ("tsl_price",      "REAL"),
            ("tp1_done",       "INTEGER DEFAULT 0"),
            ("tp2_done",       "INTEGER DEFAULT 0"),
            ("be_moved",       "INTEGER DEFAULT 0"),
            ("tsl_active",     "INTEGER DEFAULT 0"),
            ("tsl_high",       "REAL"),
            ("time_in_trade_min","REAL")
        ]
        for col, typ in pos_cols:
            if _column_missing(conn, "position", col):
                conn.execute(text(f"ALTER TABLE position ADD COLUMN {col} {typ}"))
        # Trade columns
        if _column_missing(conn, "trade", "duration_min"):
            conn.execute(text("ALTER TABLE trade ADD COLUMN duration_min REAL"))
        conn.commit()


# --- chooser helpers ---
def last_buy_time(session, symbol: str):
    from models import Order
    # last *filled* buy for this symbol
    q = session.exec(
        select(Order)
        .where(Order.symbol == symbol, Order.side == "BUY", Order.status == "FILLED")
        .order_by(Order.ts.desc())
    ).first()
    return q.ts if q else None

def cool_down_ok(session, symbol: str, minutes: int) -> bool:
    ts = last_buy_time(session, symbol)
    if not ts:
        return True
    return (datetime.utcnow() - ts) >= timedelta(minutes=minutes)


# You can flip this to pause/resume trades without restarting the app
RUN_ENABLED = True

# -------- lifespan (startup/shutdown) --------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting autoPicklr Trading Simulator...")
    # Create tables + wallet
    SQLModel.metadata.create_all(engine)
    print("[startup] Ensuring wallet exists...")
    with Session(engine) as s:
        ensure_wallet(s)

    print("[startup] Running simple migrations (add missing columns)...")
    migrate_db(engine)

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

@app.get("/admin/next_signals")
def admin_next_signals(limit: int = Query(15, ge=1, le=200)):
    """
    Dry-run chooser: shows which symbols *would* be tradable now, and why others are skipped.
    """
    from models import Wallet, Position
    out = {"debug_mode": False, "rules": {
        "require_breakout": REQUIRE_BREAKOUT,
        "min_breakout_pct": MIN_BREAKOUT_PCT,
        "cooldown_minutes": COOLDOWN_MINUTES,
        "signal_min_notional_usd": SIGNAL_MIN_NOTIONAL_USD,
    }, "signals": []}

    with Session(engine) as s:
        w = s.get(Wallet, 1)
        if not w:
            out["signals"] = []
            return out

        active_syms = get_active_universe(s)
        if not active_syms:
            active_syms = UNIVERSE

        rows = []
        for sym in active_syms:
            sigs = compute_signals(s, sym)
            if not sigs:
                rows.append({"symbol": sym, "ok": False, "reasons": ["no_signal"]})
                continue
            # pick best
            sig = max(sigs, key=lambda x: float(getattr(x, "score", 0.0) or 0.0))
            rows.append(_eval_candidate(s, w, sym, sig))

        # Top OK rows first by (score, breakout, ema_spread)
        ok_rows = [r for r in rows if r.get("ok")]
        ok_rows.sort(key=lambda r: (r.get("score", 0), r.get("breakout_pct", 0), r.get("ema_spread", 0)), reverse=True)

        # non-OK (skipped) for visibility too
        bad_rows = [r for r in rows if not r.get("ok")]
        # keep a few skipped so you can see reasons
        out["signals"] = ok_rows[:limit] + bad_rows[:max(5, limit//2)]
        return out


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
        wallet_equity = float(w.equity_usd) if w else 0.0
        wallet_balance = float(w.balance_usd) if w else 0.0

        # Open positions with live stats + confidence + TP/BE/TSL
        pos = s.exec(select(Position).where(Position.status == "OPEN")).all()
        open_positions = []
        for p in pos:
            last_px = get_last_price(s, p.symbol)
            cur = float(last_px) if last_px is not None else float(p.avg_price)
            pl_usd = (cur - float(p.avg_price)) * float(p.qty)
            pl_pct = (cur / float(p.avg_price) - 1.0) * 100.0 if p.avg_price else 0.0
            age_min = None
            if p.opened_ts:
                age_min = (datetime.utcnow() - p.opened_ts).total_seconds() / 60.0

            open_positions.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg": float(p.avg_price),
                "price": cur,
                "pl_usd": pl_usd,
                "pl_pct": pl_pct,
                "confidence": (None if p.score is None else float(p.score)),
                "tp1": (None if p.tp1_price is None else float(p.tp1_price)),
                "be": (float(p.be_price) if p.be_price is not None else (float(p.avg_price) if p.be_moved else None)),
                "tsl": (None if p.tsl_price is None else float(p.tsl_price)),
                "stop": float(p.stop),
                "target": float(p.target),
                "age_min": age_min,
            })

        # Closed trades
        closed = s.exec(select(Trade).where(Trade.exit_ts.is_not(None))).all()
        total_pnl = float(sum((t.pnl_usd or 0.0) for t in closed))
        wins = sum(1 for t in closed if (t.pnl_usd or 0.0) > 0.0)
        win_rate = (wins / len(closed) * 100.0) if closed else 0.0

        # Very recent trades list (last 10)
        recent_trades = s.exec(select(Trade).where(Trade.exit_ts.is_not(None))
                               .order_by(Trade.exit_ts.desc()).limit(10)).all()
        recent_payload = [{
            "symbol": t.symbol,
            "entry": float(t.entry_px),
            "exit": (None if t.exit_px is None else float(t.exit_px)),
            "qty": float(t.qty),
            "pnl": (None if t.pnl_usd is None else float(t.pnl_usd)),
            "result": t.result,
            "duration_min": t.duration_min
        } for t in recent_trades]

        return {
            "wallet_equity": wallet_equity,
            "wallet_balance": wallet_balance,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "total_trades": len(closed),
            "open_positions_count": len(open_positions),
            "open_positions": open_positions,
            "recent_trades": recent_payload
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
                "opened_ts": x.opened_ts.isoformat() if x.opened_ts else None,
                "stop": x.stop,
                "target": x.target,
                "status": x.status,
                # new fields used by the dashboard JS
                "current_price": getattr(x, "current_px", None),
                "pl_usd": getattr(x, "pl_usd", None),
                "pl_pct": getattr(x, "pl_pct", None),
                "confidence": getattr(x, "score", None),
                "tp1_price": getattr(x, "tp1_price", None),
                "be_price": getattr(x, "be_price", None),
                "tsl_price": getattr(x, "tsl_price", None),
                "time_in_trade_min": getattr(x, "time_in_trade_min", None),
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
    """
    Build a simple equity curve:
    - Start from WALLET_START_USD
    - Add each closed trade's PnL at its exit time
    - Append one 'now' point that includes open PnL
    """
    start = float(os.environ.get("WALLET_START_USD", "1000") or 1000.0)

    with Session(engine) as s:
        # Closed trades in chronological order
        trades = s.exec(select(Trade).where(Trade.exit_ts.is_not(None))
                        .order_by(Trade.exit_ts.asc())).all()

        curve = []
        equity = start
        # initial anchor (if you prefer, you can omit this)
        if trades:
            t0 = trades[0].exit_ts
            curve.append({"t": (t0 or datetime.utcnow()).isoformat(), "equity": equity})
        else:
            curve.append({"t": datetime.utcnow().isoformat(), "equity": equity})

        for t in trades:
            equity += float(t.pnl_usd or 0.0)
            curve.append({"t": (t.exit_ts or datetime.utcnow()).isoformat(), "equity": equity})

        # Open PnL snapshot for "now"
        pos = s.exec(select(Position).where(Position.status == "OPEN")).all()
        open_pnl = 0.0
        for p in pos:
            last_px = get_last_price(s, p.symbol)
            cur = float(last_px) if last_px is not None else float(p.avg_price)
            open_pnl += (cur - float(p.avg_price)) * float(p.qty)

        now_equity = equity + open_pnl
        curve.append({"t": datetime.utcnow().isoformat(), "equity": now_equity})

        return {
            "equity_curve": curve,
            "current_equity": now_equity
        }


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
                        MAX_NEW_POSITIONS_PER_CYCLE, SIGNAL_MIN_NOTIONAL_USD, FEE_PCT, SLIPPAGE_PCT,
                        MIN_BREAKOUT_PCT, REQUIRE_BREAKOUT, COOLDOWN_MINUTES
                    )
                    from risk import size_position
                    import random

                    candidates = []
                    w = s.get(Wallet, 1)
                    slots_left = 0
                    if w is not None and can_open_new_position(s):
                        open_ct = len(s.exec(select(Position).where(Position.status == "OPEN")).all())
                        slots_left = max(0, MAX_OPEN_POSITIONS - open_ct)

                    if w is not None and slots_left > 0:
                        for sym in active_syms:  # use the active dynamic universe, not hard-coded UNIVERSE
                            sigs = compute_signals(s, sym)
                            if not sigs:
                                continue

                            # best signal per symbol
                            sig = max(sigs, key=lambda x: getattr(x, "score", 0.0))

                            # --- hard gate: breakout must meet threshold (optional) ---
                            brk = float(getattr(sig, "breakout_pct", 0.0) or 0.0)
                            if REQUIRE_BREAKOUT and brk < MIN_BREAKOUT_PCT:
                                continue

                            # --- cooldown: don’t rebuy same coin too soon ---
                            if not cool_down_ok(s, sym, COOLDOWN_MINUTES):
                                continue

                            # size & notional pre-check
                            qty_est = size_position(w.balance_usd, sig.entry, sig.stop)
                            if qty_est <= 0:
                                continue

                            est_fill = sig.entry * (1 + SLIPPAGE_PCT)
                            unit_cost = est_fill * (1 + FEE_PCT)
                            notional_est = qty_est * unit_cost
                            if notional_est < SIGNAL_MIN_NOTIONAL_USD:
                                continue

                            # tiny jitter so ties don’t always pick same symbol
                            jitter = random.uniform(-0.005, 0.005)

                            # relative strength proxy: distance of price above long EMA (if provided on sig)
                            rel = float(getattr(sig, "ema_spread", 0.0) or 0.0)

                            # --- 1) Max extension vs long EMA ---
                            ema_long = float(getattr(sig, "ema_long", 0.0) or 0.0)
                            if ema_long > 0:
                                extension = (sig.entry / ema_long) - 1.0
                                if extension > MAX_EXTENSION_PCT:
                                    # too extended; skip
                                    continue

                            # --- 2) Minimum Reward/Risk ---
                            rr_num = sig.target - sig.entry
                            rr_den = sig.entry - sig.stop
                            if rr_den <= 0 or (rr_num / rr_den) < MIN_RR:
                                continue

                            # --- 3) EMA slope up (momentum quality) ---
                            ema_long_ago = float(getattr(sig, "ema_long_ago", 0.0) or 0.0)  # you can populate this in compute_signals
                            if ema_long and ema_long_ago:
                                if (ema_long - ema_long_ago) <= 0:
                                    continue


                            # keep everything we need to log and sort
                            candidates.append({
                                "sym": sym,
                                "sig": sig,
                                "qty_est": qty_est,
                                "notional_est": notional_est,
                                "score": float(getattr(sig, "score", 0.0) or 0.0),
                                "brk": brk,
                                "rel": rel,
                                "jitter": jitter
                            })

                    # sort by: score, breakout strength, relative strength, jitter
                    if candidates:
                        candidates.sort(key=lambda c: (c["score"], c["brk"], c["rel"], c["jitter"]), reverse=True)

                        placed = 0
                        for c in candidates:
                            if placed >= min(MAX_NEW_POSITIONS_PER_CYCLE, slots_left):
                                break

                            sym, sig, qty_est = c["sym"], c["sig"], c["qty_est"]
                            # re-size with current cash
                            w = s.get(Wallet, 1)
                            if not w:
                                break
                            qty = size_position(w.balance_usd, sig.entry, sig.stop)
                            if qty <= 0:
                                continue

                            print(f"[loop] Placing buy {sym} score={c['score']:.2f} brk={c['brk']:.4f} rel={c['rel']:.4f}")
                            place_buy(s, sym, qty, sig.entry, sig.reason, score=c["score"])
                            placed += 1


                    print(f"[loop] Cycle complete, sleeping for {POLL_SECONDS}s")
            else:
                print("[loop] paused")
        except Exception as e:
            print(f"[loop] error: {e}")
            import traceback
            traceback.print_exc()
        await asyncio.sleep(POLL_SECONDS)

