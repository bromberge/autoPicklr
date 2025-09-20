#!/usr/bin/env python3
import os, csv, argparse
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import ccxt
from sqlmodel import create_engine, Session, select

# ---- Project imports ----
import settings as S
from models import Order, Position, Trade, Candle

ENGINE = create_engine(
    "sqlite:///picklr.db",
    echo=False,
    connect_args={"check_same_thread": False},
)

# ---------- Utilities ----------
def parse_iso(dt: str) -> datetime:
    return datetime.fromisoformat(dt.strip().replace("Z",""))

def to_iso(dt: Optional[datetime]) -> str:
    return dt.isoformat(sep=" ") if dt else ""

def fnum(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def hours_between(a: Optional[datetime], b: Optional[datetime]) -> Optional[float]:
    if not a or not b: return None
    return round((b - a).total_seconds() / 3600.0, 6)

def rr(entry: float, stop: float, tgt: float) -> Optional[float]:
    if entry and stop and tgt and entry > stop:
        return round((tgt - entry) / (entry - stop), 6)
    return None

def sym_usd(sym: str) -> str:
    s = (sym or "").upper()
    return s if s.endswith("/USD") else f"{s.split('/')[0]}/USD"

def load_kraken():
    key = os.environ.get("KRAKEN_API_KEY","")
    sec = os.environ.get("KRAKEN_API_SECRET","")
    ex = ccxt.kraken({"apiKey": key, "secret": sec, "enableRateLimit": True})
    ex.load_markets()
    return ex

def fetch_trades_kraken_for_symbols(ex, symbols: List[str], since: datetime, until: datetime) -> List[dict]:
    out = []
    since_ms = int((since - timedelta(minutes=30)).timestamp() * 1000)
    for sym in symbols:
        try:
            tr = ex.fetch_my_trades(sym, since_ms, limit=1000) or []
            for t in tr:
                t["symbol"] = sym
                # ensure timestamp ms
                if "timestamp" not in t:
                    t["timestamp"] = int(ccxt.Exchange.parse8601(t.get("datetime","") or "1970-01-01T00:00:00Z") or 0)
            out.extend(tr)
        except Exception as e:
            print(f"[warn] fetch_my_trades failed for {sym}: {e}")
    lo = since - timedelta(minutes=60)
    hi = until + timedelta(minutes=60)
    filt = []
    for t in out:
        ts = datetime.utcfromtimestamp((t.get("timestamp") or 0)/1000.0)
        if lo <= ts <= hi:
            filt.append(t)
    return filt

def match_closest_trade(trades: List[dict], side: str, sym: str, ts_hint: datetime) -> Optional[dict]:
    best = None; best_dt = None
    for t in trades:
        if t.get("symbol") != sym: continue
        if t.get("side") != side: continue
        tdt = datetime.utcfromtimestamp((t.get("timestamp") or 0)/1000.0)
        if best is None or abs((tdt - ts_hint).total_seconds()) < abs((best_dt - ts_hint).total_seconds()):
            best = t; best_dt = tdt
    return best

def kraken_sells_between(trades: List[dict], sym: str, t0: datetime, t1: datetime) -> List[dict]:
    out = []
    for t in trades:
        if t.get("symbol") != sym: continue
        if t.get("side") != "sell": continue
        ts = datetime.utcfromtimestamp((t.get("timestamp") or 0)/1000.0)
        if t0 <= ts <= t1:
            out.append(t)
    return sorted(out, key=lambda x: x.get("timestamp", 0))

def vwap_and_totals(trs: List[dict]) -> Tuple[Optional[float], float, float]:
    tot_qty = 0.0; tot_notional = 0.0; tot_fee = 0.0
    for t in trs:
        q = fnum(t.get("amount")) or 0.0
        p = fnum(t.get("price")) or 0.0
        fee = fnum(t.get("fee")) or 0.0
        tot_qty += q
        tot_notional += q * p
        tot_fee += fee
    vwap = (tot_notional / tot_qty) if tot_qty > 0 else None
    return vwap, tot_qty, tot_fee

def extremes(session: Session, symbol: str, start: datetime, end: Optional[datetime]) -> Tuple[Optional[float], Optional[float]]:
    if not start: return None, None
    e = end or datetime.utcnow()
    cs = session.exec(
        select(Candle)
        .where(Candle.symbol == symbol, Candle.ts >= start, Candle.ts <= e)
        .order_by(Candle.ts.asc())
    ).all()
    if not cs: return None, None
    highs = [float(c.high if c.high is not None else c.close) for c in cs]
    lows  = [float(c.low  if c.low  is not None else c.close) for c in cs]
    return (max(highs) if highs else None, min(lows) if lows else None)

# ---------- CSV headers (one row per position) ----------
HEADERS = [
    "Status","Symbol",
    "BUY SCORE",
    "BUY TIME PLACED (AUTOPICKLR)","BUY TIME EXECUTED (KRAKEN)",
    "BUY PRICE PLACED (AUTOPICKLR)","BUY PRICE EXECUTED (KRAKEN)",
    "BUY QUANTITY (KRAKEN)","BUY COST (KRAKEN)","BUY FEE (KRAKEN)",
    "ATR STOP","EMA SLOPE","BREAKOUT PERCENTAGE","RR","EMA SPREAD",
    "SELL TIME PLACED (AUTOPICKLR)","SELL TIME EXECUTED (KRAKEN)",
    "SELL PRICE PLACED (AUTOPICKLR)","SELL PRICE EXECUTED (KRAKEN)",
    "SELL QUANTITY (% OF POSITION)","SELL QUANTITY","SELL COST","SELL FEE",
    "REASON FOR SELL",
    "P/L (%)","P/L ($)",
    "TIME POSITION WAS OPEN (HOURS)",
    "POSITION HIGHEST P/L %","POSITION LOWEST P/L %",
]

# ---------- Core builder ----------
def build_rows(dt_from: datetime, dt_to: datetime, fetch_kraken: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with Session(ENGINE) as s:
        # Pull everything we may need once
        pos_all = s.exec(select(Position).order_by(Position.opened_ts.asc())).all()
        orders = s.exec(
            select(Order)
            .where(Order.ts >= dt_from - timedelta(days=2), Order.ts <= dt_to + timedelta(days=2))
            .order_by(Order.ts.asc())
        ).all()
        trades = s.exec(
            select(Trade)
            .where(Trade.entry_ts >= dt_from - timedelta(days=2), Trade.exit_ts <= dt_to + timedelta(days=2))
            .order_by(Trade.entry_ts.asc())
        ).all()

        by_sym_orders: Dict[str, List[Order]] = {}
        for o in orders:
            by_sym_orders.setdefault(sym_usd(o.symbol), []).append(o)

        by_sym_trades: Dict[str, List[Trade]] = {}
        for t in trades:
            by_sym_trades.setdefault(sym_usd(t.symbol), []).append(t)

        # Kraken enrichment
        ex = None
        k_trades: List[dict] = []
        if fetch_kraken:
            symbols = sorted({sym_usd(p.symbol) for p in pos_all})
            ex = load_kraken()
            k_trades = fetch_trades_kraken_for_symbols(ex, symbols, dt_from, dt_to)

        # Iterate positions and build a derived closed_ts
        for p in pos_all:
            sym = sym_usd(p.symbol)
            opened_ts = p.opened_ts

            # derive closed_ts (we do NOT assume Position.closed_ts exists)
            derived_closed: Optional[datetime] = None
            # prefer a Trade that matches this position (entry_ts ~ opened_ts)
            match_trade = None
            cand_trs = by_sym_trades.get(sym, [])
            if opened_ts and cand_trs:
                match_trade = min(
                    cand_trs,
                    key=lambda t: abs((t.entry_ts - opened_ts).total_seconds()) if t.entry_ts and opened_ts else 10**12
                )
                # sanity: must be within 24h of entry, else ignore
                if match_trade and match_trade.entry_ts and opened_ts:
                    if abs((match_trade.entry_ts - opened_ts).total_seconds()) <= 24*3600:
                        derived_closed = match_trade.exit_ts
                    else:
                        match_trade = None

            # fallback: last filled SELL order after open
            if derived_closed is None:
                sell_os = [o for o in by_sym_orders.get(sym, []) if o.side == "SELL" and o.status == "FILLED" and o.ts and opened_ts and o.ts >= opened_ts]
                if sell_os:
                    derived_closed = max(sell_os, key=lambda o: o.ts).ts

            # keep only positions overlapping the requested window
            if not opened_ts:
                continue
            left  = opened_ts
            right = derived_closed or dt_to  # treat open positions as running through dt_to
            if not (left <= dt_to and right >= dt_from):
                continue

            status = (p.status or "").upper() or ("CLOSED" if derived_closed else "OPEN")

            # -------- BUY side (Autopicklr placed + Kraken executed) --------
            buy_orders = [o for o in by_sym_orders.get(sym, []) if o.side == "BUY" and o.status == "FILLED"]
            buy_order = None
            if buy_orders:
                # pick the buy closest to opened_ts
                buy_order = min(buy_orders, key=lambda o: abs((o.ts - opened_ts).total_seconds()) if opened_ts and o.ts else 10**12)

            buy_time_placed = buy_order.ts if buy_order else opened_ts
            buy_price_placed = fnum(getattr(buy_order, "price_req", None)) or fnum(getattr(p, "avg_price", None))

            buy_exec_ts = None; buy_exec_px = None; buy_qty = None; buy_fee = None; buy_cost = None
            if fetch_kraken and k_trades and buy_time_placed:
                # find the Kraken buy near our placed time
                t = match_closest_trade(k_trades, "buy", sym, buy_time_placed)
                if t:
                    buy_exec_ts = datetime.utcfromtimestamp((t.get("timestamp") or 0)/1000.0)
                    buy_exec_px = fnum(t.get("price"))
                    buy_qty     = fnum(t.get("amount"))
                    buy_fee     = fnum(t.get("fee"))
                    if buy_exec_px is not None and buy_qty is not None:
                        buy_cost = buy_exec_px * buy_qty

            # DB fallbacks
            if buy_exec_ts is None: buy_exec_ts = buy_time_placed
            if buy_exec_px is None: buy_exec_px = fnum(getattr(buy_order, "price_fill", None)) or fnum(getattr(p, "avg_price", None))
            if buy_qty is None:     buy_qty     = fnum(getattr(p, "qty", None))
            if buy_cost is None and (buy_exec_px is not None and buy_qty is not None):
                buy_cost = buy_exec_px * buy_qty

            # -------- SELL side (aggregate partials) --------
            sell_orders = [o for o in by_sym_orders.get(sym, []) if o.side == "SELL" and o.status == "FILLED" and o.ts and o.ts >= opened_ts]
            last_sell_order = max(sell_orders, key=lambda o: o.ts) if sell_orders else None
            sell_time_placed = last_sell_order.ts if last_sell_order else (derived_closed or None)
            sell_price_placed = fnum(getattr(last_sell_order, "price_req", None)) or fnum(getattr(p, "target", None))

            sell_exec_ts = None; sell_exec_px = None; sell_qty = 0.0; sell_fee = 0.0; sell_cost = 0.0
            if fetch_kraken and k_trades and buy_exec_ts:
                # take all Kraken sells after our buy (until derived_closed or dt_to)
                end_cap = derived_closed or dt_to
                tr = kraken_sells_between(k_trades, sym, buy_exec_ts, end_cap)
                if tr:
                    vwap, qty, fee = vwap_and_totals(tr)
                    sell_exec_px = vwap
                    sell_qty = qty
                    sell_fee = fee
                    sell_cost = (vwap or 0.0) * qty
                    sell_exec_ts = datetime.utcfromtimestamp((tr[-1].get("timestamp") or 0)/1000.0)

            # DB fallbacks if Kraken missing
            if sell_exec_ts is None:
                sell_exec_ts = last_sell_order.ts if last_sell_order else derived_closed
            if sell_exec_px is None:
                # VWAP from DB sells if multiple
                if sell_orders:
                    tot_q = sum(fnum(o.qty) or 0.0 for o in sell_orders)
                    tot_n = sum((fnum(o.price_fill) or 0.0) * (fnum(o.qty) or 0.0) for o in sell_orders)
                    sell_exec_px = (tot_n / tot_q) if tot_q > 0 else fnum(getattr(last_sell_order, "price_fill", None)) or fnum(getattr(p, "target", None))
                    sell_qty = sell_qty or tot_q
                    sell_cost = sell_cost or (sell_exec_px or 0.0) * (tot_q or 0.0)
                else:
                    sell_exec_px = fnum(getattr(last_sell_order, "price_fill", None)) or fnum(getattr(p, "target", None))
            # SELL % of position
            sell_pct_of_pos = None
            if buy_qty and buy_qty > 0 and sell_qty is not None:
                sell_pct_of_pos = round(100.0 * (sell_qty / buy_qty), 6)

            # -------- Analytics fields --------
            score = fnum(getattr(p, "score", None))
            entry = fnum(getattr(p, "avg_price", None))
            stop  = fnum(getattr(p, "stop", None))
            target = fnum(getattr(p, "target", None))
            atr_stop = stop
            risk_reward = rr(entry, stop, target)

            ema_slope = None; ema_spread = None; breakout_pct = None
            try:
                cs = s.exec(select(Candle).where(Candle.symbol == sym).order_by(Candle.ts.asc())).all()
                closes = [float(c.close) for c in cs]
                if closes and entry:
                    def _ema(arr, span):
                        k = 2/(span+1); s_val=None; out=[]
                        for x in arr:
                            s_val = x if s_val is None else (x - s_val)*k + s_val
                            out.append(s_val)
                        return out
                    need = S.DET_EMA_LONG + S.EMA_SLOPE_LOOKBACK + 1
                    if len(closes) >= need:
                        e1 = _ema(closes, S.DET_EMA_SHORT)
                        e2 = _ema(closes, S.DET_EMA_LONG)
                        ema_slope  = round(e2[-1] - e2[-(S.EMA_SLOPE_LOOKBACK+1)], 8)
                        ema_spread = round((e1[-1] - e2[-1]) / entry, 8)
                        if S.BREAKOUT_LOOKBACK + 1 <= len(closes):
                            prior_high = max(closes[-(S.BREAKOUT_LOOKBACK+1):-1])
                            if prior_high > 0:
                                breakout_pct = round((entry / prior_high - 1.0) * 100.0, 6)
            except Exception:
                pass

            # P/L and timing
            exit_ts = derived_closed or sell_exec_ts
            open_hours = hours_between(opened_ts, exit_ts or datetime.utcnow())
            pl_usd = None; pl_pct = None
            if status == "CLOSED":
                # prefer trade pnl if we matched one
                if match_trade and match_trade.pnl_usd is not None:
                    pl_usd = round(float(match_trade.pnl_usd), 8)
                    if match_trade.entry_px:
                        pl_pct = round(((float(match_trade.exit_px or 0) / float(match_trade.entry_px)) - 1.0) * 100.0, 6)
                elif entry and sell_exec_px and sell_qty:
                    # scale by amount sold if partial
                    pl_usd = round((sell_exec_px - entry) * sell_qty, 8)
                    pl_pct = round((sell_exec_px / entry - 1.0) * 100.0, 6)
            else:
                # mark-to-market rough P/L
                last_close = s.exec(select(Candle).where(Candle.symbol == sym).order_by(Candle.ts.desc())).first()
                px_now = float(last_close.close) if last_close else entry
                if entry and px_now and buy_qty:
                    pl_usd = round((px_now - entry) * buy_qty, 8)
                    pl_pct = round((px_now / entry - 1.0) * 100.0, 6)

            hi_px, lo_px = extremes(s, sym, opened_ts, exit_ts)
            hi_pl_pct = round(((hi_px / entry) - 1.0) * 100.0, 6) if (entry and hi_px) else None
            lo_pl_pct = round(((lo_px / entry) - 1.0) * 100.0, 6) if (entry and lo_px) else None

            reason_sell = ""
            if last_sell_order and getattr(last_sell_order, "reason", None):
                reason_sell = last_sell_order.reason
            elif status == "CLOSED" and not reason_sell:
                reason_sell = "CLOSED"

            rows.append({
                "Status": status,
                "Symbol": sym,
                "BUY SCORE": score,
                "BUY TIME PLACED (AUTOPICKLR)": to_iso(buy_time_placed),
                "BUY TIME EXECUTED (KRAKEN)": to_iso(buy_exec_ts),
                "BUY PRICE PLACED (AUTOPICKLR)": buy_price_placed,
                "BUY PRICE EXECUTED (KRAKEN)": buy_exec_px,
                "BUY QUANTITY (KRAKEN)": buy_qty,
                "BUY COST (KRAKEN)": buy_cost,
                "BUY FEE (KRAKEN)": buy_fee,
                "ATR STOP": atr_stop,
                "EMA SLOPE": ema_slope,
                "BREAKOUT PERCENTAGE": breakout_pct,
                "RR": risk_reward,
                "EMA SPREAD": ema_spread,
                "SELL TIME PLACED (AUTOPICKLR)": to_iso(sell_time_placed),
                "SELL TIME EXECUTED (KRAKEN)": to_iso(sell_exec_ts),
                "SELL PRICE PLACED (AUTOPICKLR)": sell_price_placed,
                "SELL PRICE EXECUTED (KRAKEN)": fnum(sell_exec_px),
                "SELL QUANTITY (% OF POSITION)": sell_pct_of_pos,
                "SELL QUANTITY": sell_qty,
                "SELL COST": sell_cost,
                "SELL FEE": sell_fee,
                "REASON FOR SELL": reason_sell,
                "P/L (%)": pl_pct,
                "P/L ($)": pl_usd,
                "TIME POSITION WAS OPEN (HOURS)": open_hours,
                "POSITION HIGHEST P/L %": hi_pl_pct,
                "POSITION LOWEST P/L %": lo_pl_pct,
            })

    return rows

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Export positions (one row per position) to CSV.")
    ap.add_argument("--from", dest="from_dt", required=True, help="Start ISO, e.g. 2025-08-23T00:00:00")
    ap.add_argument("--to", dest="to_dt", required=True, help="End ISO, e.g. 2025-09-07T23:59:59")
    ap.add_argument("--out", dest="outfile", default="positions_export.csv")
    ap.add_argument("--fetch-kraken", action="store_true", help="Enrich executed times/fees via Kraken")
    args = ap.parse_args()

    dt_from = parse_iso(args.from_dt)
    dt_to   = parse_iso(args.to_dt)

    rows = build_rows(dt_from, dt_to, fetch_kraken=args.fetch_kraken)

    with open(args.outfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in HEADERS})

    print(f"Wrote {len(rows)} rows to {args.outfile}")

if __name__ == "__main__":
    main()
