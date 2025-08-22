# main.py

import asyncio
import contextlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict
from universe import universe_stale, universe_debug_snapshot
from universe import refresh_universe, get_active_universe, ensure_pairs_for, UniversePair

import os
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template  # optional; safe to keep
from sqlmodel import SQLModel, create_engine, Session, select
from sqlalchemy import text

# --- settings & models ---
from settings import (
    MAX_NEW_POSITIONS_PER_CYCLE, SIGNAL_MIN_NOTIONAL_USD, FEE_PCT, SLIPPAGE_PCT,
    MIN_BREAKOUT_PCT, REQUIRE_BREAKOUT, COOLDOWN_MINUTES,
    DET_EMA_SHORT, DET_EMA_LONG, BREAKOUT_LOOKBACK, EMA_SLOPE_LOOKBACK,
)
from settings import (
    # chooser / gates
    EMA_SLOPE_MIN, MIN_EMA_SPREAD, MAX_EXTENSION_PCT, MIN_RR,
    REQUIRE_BREAKOUT, MIN_BREAKOUT_PCT, ALLOW_TREND_ENTRY, COOLDOWN_MINUTES,
    # model gates + debug
    USE_MODEL, SCORE_THRESHOLD, ENABLE_DEBUG_SIGNALS,
)
from settings import POLL_SECONDS, UNIVERSE_REFRESH_MINUTES, UNIVERSE


from models import Wallet, Order, Position, Trade, Candle

# signals
from signal_engine import compute_signals

# trading ops + wallet/equity mgmt
from sim import (
    ensure_wallet,
    can_open_new_position,
    place_buy,
    mark_to_market_and_manage,
    get_last_price,
)

# unify sizing (avoid ambiguity with sim.size_position)
from risk import size_position as risk_size_position

# dynamic universe + candles
from universe import refresh_universe, get_active_universe
from data import update_candles_for

# ---- engine/app ----
engine = create_engine(
    "sqlite:///picklr.db",
    echo=False,
    connect_args={"check_same_thread": False},
)

def _column_missing(conn, table: str, col: str) -> bool:
    info = conn.execute(text(f"PRAGMA table_info('{table}')")).fetchall()
    names = {row[1] for row in info}  # row[1] is column name
    return col not in names


def _eval_candidate(session, w, sym, sig) -> dict:
    """
    Dry-run the chooser gates and *explain* why a symbol is ok or skipped.
    Adds three pre-filters:
      1) Extension vs EMA(long) <= MAX_EXTENSION_PCT
      2) Reward/Risk >= MIN_RR
      3) EMA(long) slope > 0 over EMA_SLOPE_LOOKBACK bars
    Also computes breakout_pct and ema_spread from Candle history so we don't
    rely on extra fields on models.Signal.
    """
    reasons = []
    ok = True

    # --- Inputs from signal (with safe defaults) ---
    entry  = float(getattr(sig, "entry", 0.0) or 0.0)
    stop   = float(getattr(sig, "stop",  0.0) or 0.0)
    target = float(getattr(sig, "target",0.0) or 0.0)
    score  = float(getattr(sig, "score", 0.0) or 0.0)

    # --- Pull candles for EMA/Breakout metrics ---
    candles = session.exec(
        select(Candle).where(Candle.symbol == sym).order_by(Candle.ts.asc())
    ).all()
    closes = [c.close for c in candles] if candles else []
    min_needed = max(DET_EMA_LONG + EMA_SLOPE_LOOKBACK + 1, BREAKOUT_LOOKBACK + 2, 40)
    if len(closes) < min_needed:
        return {
            "symbol": sym, "ok": False, "reasons": [f"insufficient_history<{min_needed}"],
            "score": score, "breakout_pct": 0.0, "ema_spread": 0.0,
            "entry": entry, "stop": stop, "target": target,
            "qty_est": 0.0, "notional_est": 0.0,
            "ema_long": None, "ema_long_slope": None,
            "extension_pct": None, "rr": None,
        }

    # --- EMA helpers ---
    def _ema(arr: list[float], span: int) -> list[float]:
        k = 2 / (span + 1)
        out, s = [], None
        for x in arr:
            s = x if s is None else (x - s) * k + s
            out.append(s)
        return out

    ema_s = _ema(closes, DET_EMA_SHORT)
    ema_l = _ema(closes, DET_EMA_LONG)
    price = entry if entry > 0 else float(closes[-1])

    # breakout vs previous BREAKOUT_LOOKBACK bars (exclude current bar)
    prior_high = max(closes[-(BREAKOUT_LOOKBACK + 1):-1])
    breakout_pct = (price - prior_high) / prior_high if prior_high > 0 else 0.0

    # ema spread as % of price
    ema_spread = (ema_s[-1] - ema_l[-1]) / price if price else 0.0

    # long EMA slope
    back = EMA_SLOPE_LOOKBACK
    ema_long = float(ema_l[-1])
    ema_then = float(ema_l[-(back + 1)])
    ema_slope = ema_long - ema_then

    # Trend fallback: treat as OK if slope >= EMA_SLOPE_MIN and |ema_spread| >= MIN_EMA_SPREAD
    trend_ok = (ema_slope >= EMA_SLOPE_MIN) and (abs(ema_spread) >= MIN_EMA_SPREAD)

    # Breakout gate (OR trend fallback if enabled)
    if REQUIRE_BREAKOUT and breakout_pct < MIN_BREAKOUT_PCT and not (ALLOW_TREND_ENTRY and trend_ok):
        ok = False; reasons.append(f"breakout<{MIN_BREAKOUT_PCT:.4f}")

    # Slope gate (use the configurable minimum)
    if ema_slope < EMA_SLOPE_MIN:
        ok = False; reasons.append(f"ema_long_slope<{EMA_SLOPE_MIN}({back})")

    # --- Pre-filter: Max extension vs EMA(long) ---
    extension_pct = (price - ema_long) / ema_long if ema_long > 0 else 0.0
    if extension_pct > MAX_EXTENSION_PCT:
        ok = False; reasons.append(f"extension>{MAX_EXTENSION_PCT:.4f}")

    # --- Pre-filter: Reward/Risk ---
    rr = None
    if price > 0 and stop > 0 and target > price and price > stop:
        rr = (target - price) / (price - stop)
        if rr < MIN_RR:
            ok = False; reasons.append(f"rr<{MIN_RR:.2f}")
    else:
        ok = False; reasons.append("invalid_entry_stop_target")

    # --- Sizing & notional checks (use equity) ---
    qty_est = risk_size_position(w.equity_usd, price, stop)
    if qty_est <= 0:
        ok = False; reasons.append("qty_est<=0")

    est_fill = price * (1 + SLIPPAGE_PCT)
    unit_cost = est_fill * (1 + FEE_PCT)
    notional_est = qty_est * unit_cost
    if notional_est < SIGNAL_MIN_NOTIONAL_USD:
        ok = False; reasons.append(f"notional<{SIGNAL_MIN_NOTIONAL_USD}")

    # --- Breakeven/edge sanity ---
    buy_cost = est_fill * (1 + FEE_PCT)
    sell_net = target * (1 - SLIPPAGE_PCT) * (1 - FEE_PCT)
    if sell_net - buy_cost <= 0:
        ok = False; reasons.append("no_edge")

    return {
        "symbol": sym,
        "ok": ok,
        "reasons": reasons,
        "score": score,
        "breakout_pct": breakout_pct,
        "ema_spread": ema_spread,
        "entry": price,
        "stop": stop,
        "target": target,
        "qty_est": max(0.0, float(qty_est)),
        "notional_est": max(0.0, float(notional_est)),
        "ema_long": ema_long,
        "ema_long_slope": ema_slope,
        "extension_pct": extension_pct,
        "rr": rr,
    }



def migrate_db(engine):
    with engine.connect() as conn:
        # Position columns
        pos_cols = [
            ("current_px",        "REAL"),
            ("pl_usd",            "REAL"),
            ("pl_pct",            "REAL"),
            ("score",             "REAL"),
            ("be_price",          "REAL"),
            ("tp1_price",         "REAL"),
            ("tsl_price",         "REAL"),
            ("tp1_done",          "INTEGER DEFAULT 0"),
            ("tp2_done",          "INTEGER DEFAULT 0"),
            ("be_moved",          "INTEGER DEFAULT 0"),
            ("tsl_active",        "INTEGER DEFAULT 0"),
            ("tsl_high",          "REAL"),
            ("time_in_trade_min", "REAL"),
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
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")


# ---- Routes ----
from fastapi import APIRouter
admin = APIRouter()

@app.get("/admin/next_signals")
def admin_next_signals(limit: int = Query(15, ge=1, le=200)):
    """
    Dry-run chooser: shows which symbols *would* be tradable now, and why others are skipped.
    """
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

def _open_position_symbols(session):
    rows = session.exec(select(Position).where(Position.status == "OPEN")).all()
    return sorted({p.symbol for p in rows if getattr(p, "symbol", None)})


# -------- trading loop --------
LAST_UNIVERSE_REFRESH = None
# new: throttle knobs for candle updates
OPEN_POS_UPDATE_SECONDS = 60       # update candles for *open positions* every 60s
FULL_CANDLES_UPDATE_SECONDS = 300  # update candles for *entire active universe* every 5 minutes

LAST_OPEN_POS_UPDATE = None
LAST_FULL_CANDLES_UPDATE = None

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
                        or universe_stale(s)  # <-- refresh immediately if the cache has gone stale
                    ):
                        print("[loop] Universe cache empty/stale — refreshing…")
                        rows = await refresh_universe(s)
                        print(f"[universe] Refreshed {len(rows)} USD/USDT pairs from Kraken.")
                        LAST_UNIVERSE_REFRESH = now

                    # --- end universe refresh ---
                    
                    # --- always keep candles fresh for *open positions* (every 60s) ---
                    global LAST_OPEN_POS_UPDATE
                    if (LAST_OPEN_POS_UPDATE is None) or ((now - LAST_OPEN_POS_UPDATE).total_seconds() >= OPEN_POS_UPDATE_SECONDS):
                        op_syms = _open_position_symbols(s)
                        if op_syms:
                            # Make sure UniversePair rows exist so we know the Kraken pair names
                            rows = s.exec(select(UniversePair).where(UniversePair.symbol.in_(op_syms))).all()
                            present = {r.symbol for r in rows}
                            missing = [sym for sym in op_syms if sym not in present]
                            if missing:
                                try:
                                    ensured = await ensure_pairs_for(s, missing)
                                    if ensured:
                                        print(f"[loop] Ensured Kraken mapping for open positions: {ensured}")
                                except Exception as e:
                                    print(f"[loop] ensure_pairs_for failed: {e}")

                            print(f"[loop] Updating candles for open positions: {op_syms}")
                            await update_candles_for(s, op_syms)
                        LAST_OPEN_POS_UPDATE = now


                    # 1) Get active symbols from cache; fallback to static if needed
                    active_syms = get_active_universe(s)

                    # If the list is unexpectedly tiny, print a quick snapshot and try once more.
                    # (This uses the relaxed selector internally, then falls back to your static UNIVERSE.)
                    from universe import universe_debug_snapshot
                    if len(active_syms) <= 3:
                        snap = universe_debug_snapshot(s)
                        print(f"[universe][guard] active={len(active_syms)} snapshot={snap}")
                        if snap["rows_total"] > 0 and snap["vol_ok"] == 0:
                            print("[universe][guard] soft recovery: re-reading with relaxed filters")
                            active_syms = get_active_universe(s)
                    if not active_syms:
                        active_syms = UNIVERSE  # absolute last resort

                    print(f"[loop] Active symbols: {active_syms}")


                    # 2) Update candles for the active symbols
                    # --- throttle full active-universe candle updates (every 5 minutes) ---
                    global LAST_FULL_CANDLES_UPDATE
                    if (LAST_FULL_CANDLES_UPDATE is None) or ((now - LAST_FULL_CANDLES_UPDATE).total_seconds() >= FULL_CANDLES_UPDATE_SECONDS):
                        print("[loop] Updating candles (full universe)...")
                        await update_candles_for(s, active_syms)
                        LAST_FULL_CANDLES_UPDATE = now
                    else:
                        secs_left = FULL_CANDLES_UPDATE_SECONDS - int((now - LAST_FULL_CANDLES_UPDATE).total_seconds())
                        if secs_left < 0:
                            secs_left = 0
                        print(f"[loop] Skipping full-universe candles (next in ~{secs_left}s); open positions are updated every {OPEN_POS_UPDATE_SECONDS}s")


                    # 3) Manage open positions (TPs, BE, TSL, stops, targets, timeouts)
                    print("[loop] Managing positions...")
                    mark_to_market_and_manage(s)

                    # 4) Look for new entries (ranked; place at most N per cycle)
                    print("[loop] Checking for new positions...")
            
                    from settings import (
                        MAX_OPEN_POSITIONS,                 # <-- add this
                        MAX_NEW_POSITIONS_PER_CYCLE, SIGNAL_MIN_NOTIONAL_USD, FEE_PCT, SLIPPAGE_PCT,
                        MIN_BREAKOUT_PCT, REQUIRE_BREAKOUT, COOLDOWN_MINUTES,
                        DET_EMA_SHORT, DET_EMA_LONG, BREAKOUT_LOOKBACK, EMA_SLOPE_LOOKBACK,
                        # trend fallback + quality gates + model gates + debug
                        ALLOW_TREND_ENTRY, EMA_SLOPE_MIN, MIN_EMA_SPREAD,
                        MAX_EXTENSION_PCT, MIN_RR,
                        USE_MODEL, SCORE_THRESHOLD,
                        ENABLE_DEBUG_SIGNALS,
                    )

                    import random
            
                    # small EMA helper for this scope
                    def _ema(arr: list[float], span: int) -> list[float]:
                        k = 2 / (span + 1)
                        out, s = [], None
                        for x in arr:
                            s = x if s is None else (x - s) * k + s
                            out.append(s)
                        return out
            
                    candidates = []
                    w = s.get(Wallet, 1)
                    slots_left = 0
                    if w is not None and can_open_new_position(s):
                        open_ct = len(s.exec(select(Position).where(Position.status == "OPEN")).all())
                        slots_left = max(0, MAX_OPEN_POSITIONS - open_ct)
            
                    if w is not None and slots_left > 0:
                        for sym in active_syms:
                            sigs = compute_signals(s, sym)
                            if not sigs:
                                continue
            
                            # best signal per symbol
                            sig = max(sigs, key=lambda x: float(getattr(x, "score", 0.0) or 0.0))
            
                            # --- Fresh metrics from Candle history (don’t rely on Signal extras) ---
                            candles = s.exec(
                                select(Candle).where(Candle.symbol == sym).order_by(Candle.ts.asc())
                            ).all()
                            closes = [c.close for c in candles] if candles else []
                            need = max(DET_EMA_LONG + EMA_SLOPE_LOOKBACK + 1, BREAKOUT_LOOKBACK + 2, 40)
                            if len(closes) < need:
                                if ENABLE_DEBUG_SIGNALS:
                                    print(f"[gate:{sym}] skip: not enough candles (need {need}, have {len(closes)})")
                                continue

                            price = float(getattr(sig, "entry", 0.0) or closes[-1])

                            # small EMA helper for this scope (keep your version if you prefer)
                            def _ema(arr: list[float], span: int) -> list[float]:
                                k = 2 / (span + 1)
                                out, s = [], None
                                for x in arr:
                                    s = x if s is None else (x - s) * k + s
                                    out.append(s)
                                return out

                            e1 = _ema(closes, DET_EMA_SHORT)
                            e2 = _ema(closes, DET_EMA_LONG)

                            # breakout vs previous BREAKOUT_LOOKBACK bars (exclude current bar)
                            prior_high = max(closes[-(BREAKOUT_LOOKBACK + 1):-1])
                            brk = (price - prior_high) / prior_high if prior_high > 0 else 0.0

                            # ema spread (% of price)
                            rel = (e1[-1] - e2[-1]) / price if price else 0.0

                            ema_long = float(e2[-1])
                            back = EMA_SLOPE_LOOKBACK
                            ema_long_ago = float(e2[-(back + 1)])
                            # slope per bar
                            ema_slope = (ema_long - ema_long_ago) / max(1, EMA_SLOPE_LOOKBACK)

                            reasons = []

                            # --- Model score gate (only if model is on) ---
                            if USE_MODEL:
                                model_score = float(getattr(sig, "score", 0.0) or 0.0)
                                if model_score < SCORE_THRESHOLD:
                                    if ENABLE_DEBUG_SIGNALS:
                                        print(f"[gate:{sym}] score<{SCORE_THRESHOLD:.2f} (got {model_score:.4f})")
                                    continue

                            # --- Trend fallback: OK if slope ≥ EMA_SLOPE_MIN and short>long by at least MIN_EMA_SPREAD ---
                            trend_ok = (ema_slope >= EMA_SLOPE_MIN) and (rel >= MIN_EMA_SPREAD)

                            # --- hard gate: breakout OR strong trend (if allowed) ---
                            if REQUIRE_BREAKOUT and brk < MIN_BREAKOUT_PCT and not (ALLOW_TREND_ENTRY and trend_ok):
                                if ENABLE_DEBUG_SIGNALS:
                                    why = f"breakout<{MIN_BREAKOUT_PCT:.4f}"
                                    if ALLOW_TREND_ENTRY:
                                        why += f" and not trend_ok (slope={ema_slope:.6f}, rel={rel:.6f})"
                                    print(f"[gate:{sym}] {why}")
                                continue

                            # --- cooldown: don’t rebuy same coin too soon ---
                            if not cool_down_ok(s, sym, COOLDOWN_MINUTES):
                                if ENABLE_DEBUG_SIGNALS:
                                    print(f"[gate:{sym}] cooldown")
                                continue

                            # --- Sizing & notional pre-check (use equity) ---
                            qty_est = risk_size_position(w.equity_usd, price, sig.stop)
                            if qty_est <= 0:
                                if ENABLE_DEBUG_SIGNALS:
                                    print(f"[gate:{sym}] size=0 (risk cap or invalid stop)")
                                continue

                            est_fill = price * (1 + SLIPPAGE_PCT)
                            unit_cost = est_fill * (1 + FEE_PCT)
                            notional_est = qty_est * unit_cost
                            if notional_est < SIGNAL_MIN_NOTIONAL_USD:
                                if ENABLE_DEBUG_SIGNALS:
                                    print(f"[gate:{sym}] notional<min ({notional_est:.2f} < {SIGNAL_MIN_NOTIONAL_USD:.2f})")
                                continue

                            # --- 1) Max extension vs long EMA (avoid buying too far above the average) ---
                            if ema_long > 0:
                                extension = (price / ema_long) - 1.0
                                if extension > MAX_EXTENSION_PCT:
                                    if ENABLE_DEBUG_SIGNALS:
                                        print(f"[gate:{sym}] extension>{MAX_EXTENSION_PCT:.4f} (got {extension:.4f})")
                                    continue

                            # --- 2) Minimum Reward/Risk ---
                            rr_num = sig.target - price
                            rr_den = price - sig.stop
                            rr = rr_num / rr_den if rr_den > 0 else -1
                            if rr_den <= 0 or rr < MIN_RR:
                                if ENABLE_DEBUG_SIGNALS:
                                    print(f"[gate:{sym}] rr<{MIN_RR:.2f} (got {rr:.2f})")
                                continue

                            # --- 3) Long-EMA slope must be at least EMA_SLOPE_MIN ---
                            if ema_slope < EMA_SLOPE_MIN:
                                if ENABLE_DEBUG_SIGNALS:
                                    print(f"[gate:{sym}] ema_slope<{EMA_SLOPE_MIN:.6f} (got {ema_slope:.6f})")
                                continue

                            # tiny jitter so ties don’t always pick same symbol
                            jitter = random.uniform(-0.005, 0.005)

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
                            # re-size with current equity (risk-based)
                            w = s.get(Wallet, 1)
                            if not w:
                                break
                            qty = risk_size_position(w.equity_usd, sig.entry, sig.stop)
                            if qty <= 0:
                                continue
            
                            print(f"[loop] Placing buy {sym} score={c['score']:.2f} brk={c['brk']:.4f} rel={c['rel']:.4f}")
                            place_buy(s, sym, qty, sig.entry, sig.reason, score=c["score"])
                            placed += 1
            
            else:
                print("[loop] paused")
        except Exception as e:
            print(f"[loop] error: {e}")
            import traceback
            traceback.print_exc()

        # <-- this runs every loop, success or error
        wake = datetime.utcnow() + timedelta(seconds=POLL_SECONDS)
        print(f"[loop] Sleeping {POLL_SECONDS}s — next cycle at {wake:%Y-%m-%d %H:%M:%S}Z")
        await asyncio.sleep(POLL_SECONDS)
