# sim.py

from datetime import timedelta, datetime, timezone
from typing import Optional, List
import os

from sqlmodel import Session, select

import models as M
from models import Wallet
from settings import (
    # Costs & limits
    FEE_PCT, SLIPPAGE_PCT,
    MAX_OPEN_POSITIONS, MAX_TRADE_COST_PCT, MAX_GROSS_EXPOSURE_PCT,
    MIN_TRADE_NOTIONAL, MIN_TRADE_NOTIONAL_PCT, MIN_BREAKEVEN_EDGE_PCT,

    # Holding window
    MAX_HOLD_MINUTES,

    # Default % stop/target used when ATR versions aren’t available
    TARGET_PCT, STOP_PCT,

    # Partial TPs & sizes
    PTP_LEVELS, PTP_SIZES,

    # Break-even + trailing
    BE_TRIGGER_PCT, BE_AFTER_FIRST_TP,
    TSL_ACTIVATE_AFTER_TP, TSL_ACTIVATE_PCT, TSL_PCT,

    # Wallet boot balance
    WALLET_START_USD,

    # Risk sizing
    RISK_PCT_PER_TRADE,

    # ATR trailing
    TSL_USE_ATR, TSL_ATR_MULT, TSL_TIGHTEN_MULT, ATR_LEN,

    # NEW: live min order notional for partial TPs
    LIVE_MIN_ORDER_USD,
)


from ml_runtime import hourly_atr_from_db

EPSILON = 1e-9

def _env_true(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    if val is None:
        return False
    val = str(val).strip().lower()
    return val in ("1", "true", "yes", "on")

VERBOSE_TP_LOGS = _env_true("VERBOSE_TP_LOGS", "0")
# Default broker handle (optional). If set to a live broker (paper=False),
# place_buy() will route actual BUY submissions through it.
DEFAULT_BROKER = None

def set_default_broker(broker_obj):
    """
    Register a default broker. If this broker is live (paper=False),
    sim.place_buy() will submit real BUYs via broker.place_order(...)
    instead of executing a paper fill.
    """
    global DEFAULT_BROKER
    DEFAULT_BROKER = broker_obj


def _vlog(msg: str) -> None:
    if VERBOSE_TP_LOGS:
        try:
            print(msg)
        except Exception:
            pass
            
def _gain(entry_px: float, last_px: float) -> float:
    if entry_px <= 0 or last_px <= 0:
        return 0.0
    return (last_px / entry_px) - 1.0

def _set_stop_never_loosen(p, new_stop: float):
    p.stop = max(float(p.stop or 0.0), float(new_stop or 0.0))

# --------------------------
# Time / misc helpers
# --------------------------
def _utcnow() -> datetime:
    # store naive-UTC in DB (consistent with your existing code)
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _is_live_broker(broker) -> bool:
    """
    We consider 'live' if broker.paper == False.
    Your SimBroker uses paper=True; KrakenLiveBroker uses paper=False.
    """
    return bool(broker) and (not getattr(broker, "paper", True))


# --------------------------
# Wallet helpers
# --------------------------
def ensure_wallet(session: Session) -> Wallet:
    """
    Get the first wallet row; create it if missing.
    Always returns a Wallet instance.
    """
    w = session.exec(select(Wallet).order_by(Wallet.id.asc())).first()
    if not w:
        w = Wallet(
            balance_usd=WALLET_START_USD,
            equity_usd=WALLET_START_USD,
            updated_at=_utcnow(),
        )
        session.add(w)
        session.commit()
        session.refresh(w)
        print(f"[startup] Created wallet with ${WALLET_START_USD:,.2f}")
    return w


# Backward-compatible alias used elsewhere in this file / app
def get_or_create_wallet(session: Session) -> Wallet:
    return ensure_wallet(session)


def _sanitize_wallet(w: M.Wallet) -> None:
    if abs(w.balance_usd) < EPSILON:
        w.balance_usd = 0.0
    if abs(w.equity_usd) < EPSILON:
        w.equity_usd = 0.0
    w.balance_usd = float(round(w.balance_usd, 8))
    w.equity_usd = float(round(w.equity_usd, 8))


def _recalc_equity(session: Session, w: Optional[M.Wallet]) -> None:
    # Be defensive—if caller passed None, make sure we have a wallet row
    if w is None:
        w = ensure_wallet(session)

    mv = _portfolio_market_value(session)
    # defensive in case balance_usd was None at some point
    w.equity_usd = (w.balance_usd or 0.0) + mv
    w.updated_at = _utcnow()
    _sanitize_wallet(w)


# --------------------------
# Market value / price helpers
# --------------------------
def get_last_price(session: Session, symbol: str) -> Optional[float]:
    c = session.exec(
        select(M.Candle)
        .where(M.Candle.symbol == symbol)
        .order_by(M.Candle.ts.desc())
    ).first()
    return float(c.close) if c else None


def _prefer_kraken_mid(session: Session, symbol: str, broker) -> tuple[Optional[float], str]:
    """
    Returns (price, source_tag). If a live kraken broker is present, use mid=(bid+ask)/2.
    Fallback to latest candle close.
    """
    # Try live kraken ticker first
    try:
        if broker and getattr(broker, "live", False) and hasattr(broker, "exch"):
            # Best-effort to reuse broker's symbol mapping:
            ccxt_sym = None
            _map = getattr(broker, "_ccxt_symbol_for", None)
            if callable(_map):
                ccxt_sym = _map(session, symbol)
            else:
                # fallback guess
                ccxt_sym = f"{symbol}/USD"

            t = broker.exch.fetch_ticker(ccxt_sym)
            bid = float(t.get("bid") or 0) or None
            ask = float(t.get("ask") or 0) or None
            last = float(t.get("last") or 0) or None
            if bid and ask:
                return ((bid + ask) / 2.0, "kraken_mid")
            if last:
                return (last, "kraken_last")
    except Exception:
        pass

    # Fallback: candle close
    return (get_last_price(session, symbol), "candle")


def _portfolio_market_value(session: Session) -> float:
    total = 0.0
    opens: List[M.Position] = session.exec(
        select(M.Position).where(M.Position.status == "OPEN")
    ).all()
    for p in opens:
        px = get_last_price(session, p.symbol)
        if px is not None:
            total += px * p.qty
    return total

def _dust_threshold_usd() -> float:
    """
    Read POSITION_DUST_USD from env (default $5.00).
    Centralized so all call sites use the same threshold.
    """
    try:
        return float(os.environ.get("POSITION_DUST_USD", "5.00") or 5.00)
    except Exception:
        return 5.00


def _usd_notional(session: Session, p: M.Position, broker=None) -> float:
    """
    USD value of a position using *only* observable market data:
    - prefer Kraken mid (if live broker),
    - else latest candle close,
    - else treat as unpriceable (0.0).
    NOTE: We intentionally do NOT fall back to avg_price to avoid
    counting stale/notional value as current market value.
    """
    px, _src = _prefer_kraken_mid(session, p.symbol, broker)
    if px is None or px <= 0:
        px = get_last_price(session, p.symbol) or 0.0
    return float(px) * float(p.qty or 0.0)



def open_positions_above_usd(session: Session, broker=None) -> List[M.Position]:
    """
    Return OPEN positions whose USD notional >= POSITION_DUST_USD.
    This keeps 'dust' out of downstream management entirely.
    """
    threshold = _dust_threshold_usd()
    rows: List[M.Position] = session.exec(
        select(M.Position).where(M.Position.status == "OPEN")
    ).all()
    out: List[M.Position] = []
    for p in rows:
        try:
            if _usd_notional(session, p, broker) + 1e-12 >= threshold:
                out.append(p)
        except Exception:
            # Be permissive if we cannot price it
            out.append(p)
    return out

# --------------------------
# ATR-based trailing
# --------------------------
def _atr_trailing_stop(session: Session, symbol: str, highest_px: float) -> Optional[float]:
    """
    Compute an ATR-based trailing stop:
    stop = highest_since_activation - (TSL_ATR_MULT * ATR)
    """
    # === START PATCH: sim.atr_trail_clamp ===
    if not TSL_USE_ATR:
        return None
    atr_val = hourly_atr_from_db(session, symbol, n=ATR_LEN)
    if atr_val is None or atr_val <= 0:
        return None

    trail_px = max(0.0, highest_px - (TSL_ATR_MULT * atr_val))

    # Clamp trail so it never sits at/above current price or inside a small buffer.
    last_px = get_last_price(session, symbol)
    if last_px is None or last_px <= 0:
        return trail_px

    fee_buf = FEE_PCT * 2.0
    slip_buf = SLIPPAGE_PCT * 2.0
    MIN_TRAIL_GAP = max(0.0005, 3.0 * (fee_buf + slip_buf))  # ≥5 bps and ≥ frictions

    max_allowed_trail = last_px * (1.0 - MIN_TRAIL_GAP)
    if trail_px > max_allowed_trail:
        trail_px = max_allowed_trail

    return trail_px



# --------------------------
# Position sizing (risk-based)
# --------------------------
def risk_size_position(equity_usd: float, entry: float, stop: float,
                       risk_pct: float = RISK_PCT_PER_TRADE) -> float:
    """
    Risk a fixed % of equity per trade. Position size = risk_usd / per-unit-loss.
    """
    if entry <= 0 or stop <= 0 or entry <= stop:
        return 0.0
    risk_usd = max(0.0, equity_usd) * max(0.0, risk_pct)
    per_unit_loss = entry - stop
    if per_unit_loss <= 0:
        return 0.0
    qty = risk_usd / per_unit_loss
    return max(0.0, qty)


# (Kept for compatibility with older references; identical to risk_size_position with explicit risk_pct)
def size_position(cash_balance: float, entry: float, stop: float, risk_pct: float = 0.02) -> float:
    if entry <= 0 or stop <= 0 or entry <= stop:
        return 0.0
    risk_usd = max(0.0, cash_balance) * max(0.0, risk_pct)
    per_unit = entry - stop
    if per_unit <= 0:
        return 0.0
    qty = risk_usd / per_unit
    return max(0.0, qty)


# --------------------------
# Order / Trade bookkeeping
# --------------------------
def _book_order(session: Session, symbol: str, side: str, qty: float,
                price_req: float, price_fill: float, status: str, note: str):
    session.add(M.Order(
        ts=_utcnow(), symbol=symbol, side=side, qty=qty,
        price_req=price_req, price_fill=price_fill,
        status=status, reason=note
    ))


def _add_trade(session: Session, symbol: str, entry_ts: datetime, entry_px: float,
               exit_px: float, qty: float):
    pnl = (exit_px - entry_px) * qty
    result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
    session.add(M.Trade(
        symbol=symbol, entry_ts=entry_ts, exit_ts=_utcnow(),
        entry_px=entry_px, exit_px=exit_px, qty=qty,
        pnl_usd=float(pnl), result=result
    ))


# --------------------------
# Close helpers
# --------------------------
def _partial_close(session: Session, p: M.Position, qty: float, exit_px: float, reason: str):
    if qty <= 0 or qty > p.qty:
        return

    w = ensure_wallet(session)

    fill_px = exit_px * (1 - SLIPPAGE_PCT)
    fee = fill_px * qty * FEE_PCT
    proceeds = fill_px * qty - fee

    w.balance_usd += proceeds
    _add_trade(session, p.symbol, p.opened_ts, p.avg_price, fill_px, qty)
    _book_order(session, p.symbol, "SELL", qty, p.avg_price, fill_px, "FILLED", reason)

    p.qty -= qty
    if p.qty <= EPSILON:
        p.qty = 0.0
        p.status = "CLOSED"

    _recalc_equity(session, w)
    session.commit()


def _close_position(session: Session, p: M.Position, exit_px: float, reason: str):
    if p.qty <= 0:
        return
    _partial_close(session, p, p.qty, exit_px, reason)


def _activate_tsl_if_needed(p: M.Position, entry_px: float, last_px: float):
    if p.tsl_active:
        return
    trigger_by_pct = (last_px >= entry_px * (1 + TSL_ACTIVATE_PCT))
    trigger_by_tp = p.tp1_done if TSL_ACTIVATE_AFTER_TP else False
    if trigger_by_tp or trigger_by_pct:
        p.tsl_active = True
        p.tsl_high = last_px


# --------------------------
# Entry
# --------------------------
def place_buy(
    session: Session,
    symbol: str,
    qty: float,
    entry: float,
    reason: str,
    stop: Optional[float] = None,
    target: Optional[float] = None,
    score: Optional[float] = None,
):
    """
    Enforces:
      - Breakeven-aware per-unit edge (must beat fees+slippage by MIN_BREAKEVEN_EDGE_PCT*entry)
      - Cash cap, Per-trade cap, Gross exposure cap
      - Notional floor
    Sets up position with TP/BE/TSL state for later management.
    """
    if qty <= 0:
        return

    w = ensure_wallet(session)
    _recalc_equity(session, w)

    last_px = get_last_price(session, symbol) or entry
    fill_px = last_px * (1 + SLIPPAGE_PCT)

    # Use provided stop/target if available; otherwise defaults
    tgt = target or (entry * (1 + TARGET_PCT))
    stp = stop or (entry * (1 - STOP_PCT))

    fee_buf = FEE_PCT * 2.0      # buy + sell
    slip_buf = SLIPPAGE_PCT * 2.0
    # Minimum stop gap (in pct) to avoid micro-stops from tiny ATR / math
    MIN_STOP_GAP_PCT = max(0.006, 3.0 * (fee_buf + slip_buf))  # ≥0.60% and ≥ 3× frictions

    # Express current stop gap vs *entry* (not fill) to avoid drift
    cur_gap = max(0.0, 1.0 - (stp / entry))
    if cur_gap < MIN_STOP_GAP_PCT:
        stp = entry * (1.0 - MIN_STOP_GAP_PCT)


    # ---- Breakeven-aware per-unit edge ----
    buy_cost = fill_px * (1 + FEE_PCT)
    sell_net = (tgt * (1 - SLIPPAGE_PCT)) * (1 - FEE_PCT)
    per_unit_edge = sell_net - buy_cost
    min_edge = entry * MIN_BREAKEVEN_EDGE_PCT
    if per_unit_edge < min_edge:
        _book_order(session, symbol, "BUY", 0.0, entry, fill_px, "REJECTED",
                    f"Breakeven check failed: edge={per_unit_edge:.6f} < min={min_edge:.6f}")
        session.commit()
        return

    # ---- Caps ----
    unit_cost = buy_cost
    if unit_cost <= 0:
        return

    # A) Cash cap
    cash_cap_qty = w.balance_usd / unit_cost

    # B) Per-trade cap
    max_trade_cost = w.equity_usd * MAX_TRADE_COST_PCT
    trade_cost_cap_qty = max_trade_cost / unit_cost if unit_cost > 0 else 0.0

    # C) Gross exposure cap
    current_mv = _portfolio_market_value(session)
    max_gross_exposure = w.equity_usd * MAX_GROSS_EXPOSURE_PCT
    remaining_exposure_dollars = max(0.0, max_gross_exposure - current_mv)
    exposure_cap_qty = remaining_exposure_dollars / fill_px if fill_px > 0 else 0.0

    buy_qty = min(qty, cash_cap_qty, trade_cost_cap_qty, exposure_cap_qty)

    # ---- Notional floor ----
    notional = buy_qty * fill_px
    min_notional = max(MIN_TRADE_NOTIONAL, w.equity_usd * MIN_TRADE_NOTIONAL_PCT)
    if buy_qty < 1e-6 or notional < min_notional:
        _book_order(session, symbol, "BUY", 0.0, entry, fill_px, "REJECTED",
                    f"Notional too small: {notional:.2f} < min {min_notional:.2f}")
        session.commit()
        return

    # ---- Live execution path (if a default live broker is configured) ----
    broker_obj = globals().get("DEFAULT_BROKER", None)
    if _is_live_broker(broker_obj):
        try:
            res = broker_obj.place_order(
                symbol=symbol,
                side="BUY",
                qty=buy_qty,
                order_type="market",
                price=fill_px,        # reference price
                reason=reason,
                session=session,      # broker should add the Order row
            )
            if not res:
                _book_order(session, symbol, "BUY", 0.0, entry, fill_px, "REJECTED",
                            "broker.place_order returned None")
            session.commit()
        except Exception as ee:
            _book_order(session, symbol, "BUY", 0.0, entry, fill_px, "REJECTED",
                        f"broker error: {ee}")
            session.commit()
        return


    # ---- Execute buy ----
    fee = fill_px * buy_qty * FEE_PCT
    cost = fill_px * buy_qty + fee
    w.balance_usd -= cost

    _book_order(session, symbol, "BUY", buy_qty, entry, fill_px, "FILLED",
                f"{reason} | cost={cost:.6f}")

    # Open or add position
    p = session.exec(
        select(M.Position).where(M.Position.symbol == symbol, M.Position.status == "OPEN")
    ).first()

    if p:
        # average in
        total_qty = p.qty + buy_qty
        if total_qty > 0:
            p.avg_price = (p.avg_price * p.qty + fill_px * buy_qty) / total_qty
            p.qty = total_qty
        # keep existing stop/target/flags
    else:
        p = M.Position(
            symbol=symbol,
            qty=buy_qty,
            avg_price=fill_px,
            opened_ts=_utcnow(),
            stop=stp,
            target=tgt,
            status="OPEN",
            tp1_done=False,
            tp2_done=False,
            be_moved=False,
            tsl_active=False,
            tsl_high=None,
        )
        # TP1 absolute level (for dashboard)
        if PTP_LEVELS and len(PTP_LEVELS) >= 1:
            p.tp1_price = fill_px * (1 + PTP_LEVELS[0])

        # Store model score as confidence if that column exists; fall back to .score
        if isinstance(score, (int, float)):
            if hasattr(p, "confidence"):
                p.confidence = float(score)
            else:
                try:
                    p.score = float(score)
                except Exception:
                    pass

        session.add(p)

    # keep stats fresh
    refresh_position_stats(session, p)

    _recalc_equity(session, w)
    session.commit()


# --------------------------
# Live stats / UI helpers
# --------------------------
def refresh_position_stats(session: Session, p: M.Position):
    last_px = get_last_price(session, p.symbol)
    if last_px is None:
        return

    p.current_px = last_px
    try:
        p.pl_usd = (last_px - p.avg_price) * p.qty
        p.pl_pct = ((last_px / p.avg_price) - 1.0) * 100.0
    except Exception:
        p.pl_usd = 0.0
        p.pl_pct = 0.0

    # tp1 level (store once)
    if getattr(p, "tp1_price", None) is None and PTP_LEVELS:
        p.tp1_price = p.avg_price * (1 + PTP_LEVELS[0])

    # break-even price (only after BE is moved)
    p.be_price = p.avg_price if p.be_moved else None

    # trailing stop “visible” price if active (tsl mapped to current stop)
    p.tsl_price = p.stop if p.tsl_active else None

    # time in trade (minutes)
    if p.opened_ts:
        delta = _utcnow() - p.opened_ts
        p.time_in_trade_min = int(delta.total_seconds() // 60)


# --------------------------
# MTM + full position management
# --------------------------
def mark_to_market_and_manage(session: Session, broker=None):
    """
    Every cycle: manage TP1/TP2, Break-even, Trailing, Stop/Target, and Time exit.
    """
    w = ensure_wallet(session)

    opens: List[M.Position] = open_positions_above_usd(session, broker)
    for p in opens:
        # Prefer Kraken mid if available for TP decisions; fallback to last candle
        last_px, _px_src = _prefer_kraken_mid(session, p.symbol, broker)
        if last_px is None:
            last_px = get_last_price(session, p.symbol)
            _px_src = _px_src or "candle"

        if last_px is None:
            if last_px is None:
                try:
                    # No price this cycle—still emit a compact summary
                    stage = "Trailing" if p.tsl_active else ("TP2" if p.tp2_done else ("TP1" if p.tp1_done else "Open"))
                    print(f"{p.symbol}: SKIP, n/a, {stage}")
                except Exception:
                    pass
                did_print_summary = True
                continue


        # Emit a compact SKIP line upfront; if a SELL happens later, we’ll emit the SELL line instead.
        preprinted = False
        try:
            pl_pct0 = ((last_px / (p.avg_price or last_px)) - 1.0) * 100.0 if p.avg_price else 0.0
            stage0 = "Trailing" if p.tsl_active else ("TP2" if p.tp2_done else ("TP1" if p.tp1_done else "Open"))
            print(f"{p.symbol}: SKIP, {pl_pct0:.2f}%, {stage0}")
            preprinted = True
        except Exception:
            preprinted = False

                

        # Update live fields for dashboard
        p.current_px = last_px
        p.pl_usd = (last_px - p.avg_price) * p.qty
        p.pl_pct = ((last_px / p.avg_price) - 1.0) * 100.0 if p.avg_price > 0 else None

        entry_px = p.avg_price
        BARS_DELAY_MIN = 2  # minutes

        age_min = getattr(p, "time_in_trade_min", None)

        def _be_tsl_delay_ok() -> bool:
            try:
                return (age_min is not None) and (age_min >= BARS_DELAY_MIN)
            except Exception:
                return False

        # ---- Partial Take-Profits ----
        # Compact console summary defaults
        action = "SKIP"          # or "SELL"
        action_meta = None       # dict with keys: price, qty, reason, ts (UTC ISO)
        did_print_summary = False
        # Precompute gross move (for logging)
        gross = 0.0
        try:
            if entry_px and last_px:
                gross = (last_px / entry_px) - 1.0
        except Exception:
            pass

        # Helpful "DONE" echo so we know which ones already fired
        if p.tp1_done:
            try:
                if VERBOSE_TP_LOGS:
                    _vlog(f"[tp1] DONE {p.symbol}: gross {gross*100:.2f}% (src={_px_src})")
            except Exception:
                pass

        # TP levels
        tp_levels = [entry_px * (1 + g) for g in PTP_LEVELS]
        tp_price1 = tp_levels[0] if len(tp_levels) >= 1 else None
        tp_price2 = tp_levels[1] if len(tp_levels) >= 2 else None

        # === TP1 (+3.7%): sell 30%, then move stop to breakeven (entry) ===
        if (not p.tp1_done) and (tp_price1 is not None) and (last_px >= tp_price1):
            sell_qty = p.qty * (PTP_SIZES[0] if PTP_SIZES and len(PTP_SIZES) >= 1 else 0.0)

            # --- Live min-notional handling (Kraken per-pair minimum) ---
            if _is_live_broker(broker) and sell_qty > 0.0:
                try:
                    min_usd = float(broker.pair_min_notional_usd(p.symbol))
                    if VERBOSE_TP_LOGS:
                        _vlog(f"[tp1] {p.symbol}: Kraken min notional ≈ ${min_usd:.2f}")

                except Exception:
                    from settings import LIVE_MIN_ORDER_USD
                    min_usd = float(LIVE_MIN_ORDER_USD)
                    if VERBOSE_TP_LOGS:
                        _vlog(f"[tp1] {p.symbol}: fallback min notional ≈ ${min_usd:.2f}")


                total_value = float(last_px) * float(p.qty)
                if total_value + 1e-12 < min_usd:
                    # Whole position is below exchange minimum — nothing we can do.
                    if VERBOSE_TP_LOGS:
                        _vlog(f"[tp1] SKIP {p.symbol}: position ${total_value:.2f} < min ${min_usd:.2f} — cannot sell any size.")

                    sell_qty = 0.0  # do NOT mark tp1_done
                else:
                    # If 30% chunk is too small, upsize (but never above full position)
                    est_value = float(last_px) * float(sell_qty)
                    if est_value + 1e-12 < min_usd:
                        old_qty = float(sell_qty)
                        need_qty = min_usd / max(1e-12, float(last_px))
                        sell_qty = min(float(p.qty), float(need_qty))
                        if VERBOSE_TP_LOGS:
                            _vlog(f"[tp1] UPSIZE {p.symbol}: 30% chunk ${est_value:.2f} < min ${min_usd:.2f} — qty {old_qty:.8f} -> {sell_qty:.8f}")



            # Execute TP1
            if sell_qty > 0:
                if _is_live_broker(broker):
                    res = broker.place_order(
                        symbol=p.symbol, side="SELL", qty=sell_qty,
                        order_type="market", price=tp_price1,
                        reason="TP1", session=session,
                    )
                    if res:
                        p.tp1_done = True
                        try:
                            if VERBOSE_TP_LOGS:
                                _vlog(
                                    f"[tp1] LIVE DONE {p.symbol}: sold {sell_qty:.6f} @≈{tp_price1:.8f} "
                                    f"(~${sell_qty*tp_price1:.2f}) src={_px_src} gross={(last_px/entry_px-1)*100:.2f}%"
                                )
                        except Exception:
                            pass

                        # Make sure the compact SELL line is set even if logging above is skipped.
                        action = "SELL"
                        action_meta = {
                            "price": float(tp_price1),
                            "qty": float(sell_qty),
                            "reason": "TP1",
                            "ts": _utcnow().isoformat(timespec="seconds") + "Z"
                        }



                

                else:
                    _partial_close(session, p, sell_qty, tp_price1, "TP1")
                    p.tp1_done = True
                    _vlog(f"[tp1] DONE {p.symbol}: tp1_done=True (paper)")
                    action = "SELL"
                    action_meta = {
                        "price": float(tp_price1),
                        "qty": float(sell_qty),
                        "reason": "TP1",
                        "ts": _utcnow().isoformat(timespec="seconds") + "Z"
                    }


            # Move stop to breakeven ONLY if TP1 actually executed
            if p.tp1_done:
                prev_stop = float(p.stop or entry_px)
                _set_stop_never_loosen(p, entry_px)
                p.be_moved = True
                try:
                    if VERBOSE_TP_LOGS:
                        _vlog(f"[tp1] BE {p.symbol}: stop {prev_stop:.8g} -> {entry_px:.8g}")
                except Exception:
                    pass

        # Independent SKIP log (don’t attach with elif; print each cycle while below TP1)
        if (not p.tp1_done) and (tp_price1 is not None) and (last_px is not None) and (last_px < tp_price1):
            try:
                if VERBOSE_TP_LOGS:
                    _vlog(f"[tp1] SKIP {p.symbol}: gross {gross*100:.2f}% < {PTP_LEVELS[0]*100:.2f}% (src={_px_src})")
            except Exception:
                pass


        # === TP2 (+8%): sell 30%, then enable a 4% trailing stop ===
        if p.status == "OPEN" and (not p.tp2_done) and (tp_price2 is not None) and (last_px >= tp_price2):
            sell_qty = p.qty * (PTP_SIZES[1] if PTP_SIZES and len(PTP_SIZES) >= 2 else 0.0)

            # --- Live min-notional handling (Kraken per-pair minimum) ---
            if _is_live_broker(broker) and sell_qty > 0.0:
                try:
                    min_usd = float(broker.pair_min_notional_usd(p.symbol))
                    if VERBOSE_TP_LOGS:
                        _vlog(f"[tp2] {p.symbol}: Kraken min notional ≈ ${min_usd:.2f}")
                except Exception:
                    from settings import LIVE_MIN_ORDER_USD
                    min_usd = float(LIVE_MIN_ORDER_USD)
                    if VERBOSE_TP_LOGS:
                        _vlog(f"[tp2] {p.symbol}: fallback min notional ≈ ${min_usd:.2f}")


                total_value = float(last_px) * float(p.qty)
                if total_value + 1e-12 < min_usd:
                    if VERBOSE_TP_LOGS:
                        _vlog(f"[tp2] SKIP {p.symbol}: position ${total_value:.2f} < min ${min_usd:.2f} — cannot sell any size.")
                    sell_qty = 0.0  # do NOT mark tp2_done
                else:
                    est_value = float(last_px) * float(sell_qty)
                    if est_value + 1e-12 < min_usd:
                        old_qty = float(sell_qty)
                        need_qty = min_usd / max(1e-12, float(last_px))
                        sell_qty = min(float(p.qty), float(need_qty))
                        if VERBOSE_TP_LOGS:
                            _vlog(f"[tp2] UPSIZE {p.symbol}: 30% chunk ${est_value:.2f} < min ${min_usd:.2f} — qty {old_qty:.8f} -> {sell_qty:.8f}")

            if sell_qty > 0:
                if _is_live_broker(broker):
                    res = broker.place_order(
                        symbol=p.symbol,
                        side="SELL",
                        qty=sell_qty,
                        order_type="market",
                        price=tp_price2,
                        reason="TP2",
                        session=session,
                    )
                    if res:
                        p.tp2_done = True
                        action = "SELL"
                        action_meta = {
                            "price": float(tp_price2),
                            "qty": float(sell_qty),
                            "reason": "TP2",
                            "ts": _utcnow().isoformat(timespec="seconds") + "Z"
                        }
                else:
                    _partial_close(session, p, sell_qty, tp_price2, "TP2")
                    p.tp2_done = True
                    if VERBOSE_TP_LOGS:
                        _vlog(f"[tp2] DONE {p.symbol}: tp2_done=True (paper)")
                    action = "SELL"
                    action_meta = {
                        "price": float(tp_price2),
                        "qty": float(sell_qty),
                        "reason": "TP2",
                        "ts": _utcnow().isoformat(timespec="seconds") + "Z"
                    }

            # Turn on trailing stop now with a fixed 4% trail
            if not p.tsl_active:
                if VERBOSE_TP_LOGS:
                    _vlog(f"[tp2] TSL {p.symbol}: activating trailing stop (custom 4%)")
            p.tsl_active = True
            p.tsl_high = last_px
            try:
                p.custom_tsl_pct = 0.04   # 4% trail
            except Exception:
                pass

        if p.status != "OPEN":
            continue

        # --- Force-activate a 4% trailing profit at +10% gain ---
        if entry_px and entry_px > 0:
            gain_frac = (last_px / entry_px) - 1.0
            if (not p.tsl_active) and (gain_frac >= 0.10):
                p.tsl_active = True
                p.tsl_high = last_px
                try:
                    p.custom_tsl_pct = 0.04
                except Exception:
                    pass

        # ---- Break-Even stop move ----
        if _be_tsl_delay_ok():
            gain_ok = last_px >= entry_px * (1 + BE_TRIGGER_PCT)
            tp1_ok = p.tp1_done if BE_AFTER_FIRST_TP else False
            if not p.be_moved and (gain_ok or tp1_ok):
                p.stop = max(p.stop, entry_px)
                p.be_moved = True
        # else: delay not met; skip BE move for now

        # Surface BE as a price for the dashboard when moved
        p.be_price = entry_px if p.be_moved else None

        # ---- Trailing stop activation + update ----
        if _be_tsl_delay_ok():
            _activate_tsl_if_needed(p, entry_px, last_px)
        else:
            p.tsl_price = None  # explicit: not active yet

        if p.tsl_active and _be_tsl_delay_ok():
            # Track highest price since trailing started
            if p.tsl_high is None or last_px > p.tsl_high:
                p.tsl_high = last_px

            # Prefer ATR trailing; else fallback to % trail
            dyn_trail_stop = None
            custom_pct = getattr(p, "custom_tsl_pct", None)
            if isinstance(custom_pct, (int, float)) and custom_pct > 0:
                dyn_trail_stop = p.tsl_high * (1 - float(custom_pct))
            else:
                atr_stop = _atr_trailing_stop(session, p.symbol, p.tsl_high)
                if atr_stop is not None:
                    dyn_trail_stop = atr_stop
                else:
                    dyn_trail_stop = p.tsl_high * (1 - TSL_PCT)

            # Safety clamp: stop must sit below current price by a small buffer
            fee_buf = FEE_PCT * 2.0
            slip_buf = SLIPPAGE_PCT * 2.0
            MIN_TRAIL_GAP = max(0.0005, 3.0 * (fee_buf + slip_buf))
            max_allowed_trail = last_px * (1.0 - MIN_TRAIL_GAP)
            if dyn_trail_stop > max_allowed_trail:
                dyn_trail_stop = max_allowed_trail

            # Never reduce the stop
            p.stop = max(p.stop, dyn_trail_stop)
            p.tsl_price = p.stop
        else:
            p.tsl_price = None

        # ---- Stop and “Target-as-ceiling” rules ----
        if p.qty > 0:
            # A) Hard stop / trailing stop hit
            if last_px <= p.stop:
                if _is_live_broker(broker):
                    broker.place_order(
                        symbol=p.symbol, side="SELL", qty=p.qty,
                        order_type="market", price=p.stop,
                        reason="STOP/TSL", session=session,
                    )
                else:
                    _close_position(session, p, p.stop, "STOP/TSL")
                    action = "SELL"
                    action_meta = {
                        "price": float(p.stop),
                        "qty": float(p.qty),
                        "reason": "STOP/TSL",
                        "ts": _utcnow().isoformat(timespec="seconds") + "Z"
                    }
                    # Print compact line immediately since we continue next
                    try:
                        pl_pct = ((last_px / entry_px) - 1.0) * 100.0 if entry_px else 0.0
                        stage = "Trailing" if p.tsl_active else ("TP2" if p.tp2_done else ("TP1" if p.tp1_done else "Open"))
                        print(f"{p.symbol}: SELL, {pl_pct:.2f}%, {stage} | price {action_meta['price']:.8f}, qty {action_meta['qty']:.6f}, reason {action_meta['reason']}, at {action_meta['ts']}")
                    except Exception:
                        pass
                did_print_summary = True
                continue

            # B) Hitting the old target doesn't close; tighten trail instead
            if last_px >= p.target:
                if not p.tsl_active:
                    p.tsl_active = True
                    p.tsl_high = last_px

                # Tighten the trail
                new_tsl = None
                atr_val = hourly_atr_from_db(session, p.symbol, n=ATR_LEN)
                if atr_val is not None and atr_val > 0:
                    tighten_len = max(1e-9, TSL_ATR_MULT * TSL_TIGHTEN_MULT)
                    new_tsl = max(0.0, p.tsl_high - (tighten_len * atr_val))
                else:
                    new_tsl = p.tsl_high * (1 - (TSL_PCT * TSL_TIGHTEN_MULT))

                # Clamp to stay below current price by a buffer
                fee_buf = FEE_PCT * 2.0
                slip_buf = SLIPPAGE_PCT * 2.0
                MIN_TRAIL_GAP = max(0.0005, 3.0 * (fee_buf + slip_buf))
                max_allowed_trail = last_px * (1.0 - MIN_TRAIL_GAP)
                if new_tsl > max_allowed_trail:
                    new_tsl = max_allowed_trail

                # Never reduce the stop
                p.stop = max(p.stop, new_tsl)
                p.tsl_price = p.stop

            if not did_print_summary:
              try:
                  pl_pct = ((last_px / entry_px) - 1.0) * 100.0 if entry_px else 0.0
                  stage = "Trailing" if p.tsl_active else ("TP2" if p.tp2_done else ("TP1" if p.tp1_done else "Open"))
                  if action == "SELL" and action_meta:
                      print(f"{p.symbol}: SELL, {pl_pct:.2f}%, {stage} | price {action_meta['price']:.8f}, qty {action_meta['qty']:.6f}, reason {action_meta['reason']}, at {action_meta['ts']}")
                  else:
                      if not preprinted:
                          print(f"{p.symbol}: SKIP, {pl_pct:.2f}%, {stage}")
              except Exception:
                  pass


        # ---- Time-based exit ----
        if p.status == "OPEN" and p.opened_ts is not None:
            age = (_utcnow() - p.opened_ts)
            if age >= timedelta(minutes=MAX_HOLD_MINUTES):
                if _is_live_broker(broker):
                    broker.place_order(
                        symbol=p.symbol, side="SELL", qty=p.qty,
                        order_type="market", price=last_px,
                        reason="TIME", session=session,
                    )
                else:
                    _close_position(session, p, last_px, "TIME")
                did_print_summary = True
                continue

    # Refresh stats for all remaining open positions
    opens2: List[M.Position] = session.exec(
        select(M.Position).where(M.Position.status == "OPEN")
    ).all()
    for p2 in opens2:
        refresh_position_stats(session, p2)

    # Refresh account equity (wallet: cash + MV of positions)
    _recalc_equity(session, w)
    session.commit()



# --------------------------
# Account guard used by chooser / loop
# --------------------------
def can_open_new_position(session: Session) -> bool:
    from settings import MAX_OPEN_POSITIONS
    import os
    dust_usd = float(os.environ.get("POSITION_DUST_USD", "0.01") or 0.01)

    opens: List[M.Position] = session.exec(
        select(M.Position).where(M.Position.status == "OPEN")
    ).all()

    count_effective = 0
    for p in opens:
        last = get_last_price(session, p.symbol)
        if last is None or last <= 0:
            continue  # cannot price -> treat as dust
        if (last * float(p.qty or 0.0)) >= dust_usd:
            count_effective += 1

    return count_effective < MAX_OPEN_POSITIONS
