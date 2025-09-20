# brokers.py
from typing import Optional
from sqlmodel import Session
from models import Wallet
from sim import place_buy, get_last_price

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import os
import ccxt
from sqlmodel import Session, select


from models import Wallet, Order, Position, Trade, Candle
from universe import UniversePair
from settings import (
    BROKER,
    KRAKEN_API_KEY, KRAKEN_API_SECRET,
    LIVE_MAX_ORDER_USD, LIVE_MIN_ORDER_USD,
    FEE_PCT,
    SLIPPAGE_PCT,  # needed in BUY stop-gap calc
)


@dataclass
class Balance:
    cash_usd: float
    equity_usd: float

# -------- Sim broker you already had (keep if present) --------
class SimBroker:
    name = "sim"
    paper = True
    live = False
    def __init__(self, engine):
        self.engine = engine
    def get_balance(self) -> Balance:
        try:
            b = self.exch.fetch_balance()
            usd  = float(b.get("free", {}).get("USD", 0.0))
            usdt = float(b.get("free", {}).get("USDT", 0.0))
            cash = usd + usdt
        except Exception:
            cash = 0.0

        # For now, treat equity as cash (until we compute live MV of positions)
        return Balance(cash_usd=cash, equity_usd=cash)
    # In sim you probably used sim.place_buy; we won't reimplement it here since your loop now calls place_order on the selected broker.

# -------- Live Kraken broker (uses CCXT) --------
class KrakenLiveBroker:
    name = "kraken-live"
    paper = False
    live = True

    def _split_app_symbol(self, symbol: str) -> tuple[str, str | None]:
        """
        Returns (base_app, quote_app_or_None) from whatever the app passed in.
        Examples:
          "XDG" -> ("XDG", None)
          "DOGE/USD" -> ("DOGE", "USD")
          "sol/usdt" -> ("SOL", "USDT")
        """
        s = (symbol or "").strip().upper()
        if "/" in s:
            b, q = s.split("/", 1)
            return b.strip(), q.strip()
        return s, None

    def _to_exchange_base(self, base_app: str) -> str:
        """
        Map app base -> Kraken/CCXT base where symbols differ.
        XDG -> DOGE, XBT -> BTC, pass-through otherwise.
        """
        m = {"XDG": "DOGE", "XBT": "BTC"}
        return m.get((base_app or "").upper(), (base_app or "").upper())

    def _prefer_quotes_by_wallet(self) -> list[str]:
        """
        Prefer the quote you actually have more free balance in (USD vs USDT).
        Tie-breaker = USD. Fallback = ["USD", "USDT"].
        """
        try:
            bal = self.exch.fetch_balance() or {}
            free = bal.get("free", {}) or {}
            usd  = float((free.get("USD")  or free.get("ZUSD")  or 0) or 0)
            usdt = float((free.get("USDT") or free.get("ZUSDT") or 0) or 0)
            return ["USD", "USDT"] if usd >= usdt else ["USDT", "USD"]
        except Exception:
            return ["USD", "USDT"]


    def _pair_min_notional_usd(self, ccxt_symbol: str) -> float:
        """
        Return the exchange's min notional (cost) for a given *Kraken* market symbol,
        e.g. 'AVAX/USD' or 'DOGE/USDT'. Falls back to env if unavailable.
        """
        try:
            markets = getattr(self.exch, "markets", None) or self.exch.load_markets()
            m = markets.get(ccxt_symbol) or {}
            lim = (m.get("limits") or {}).get("cost") or {}
            mn = lim.get("min")
            if mn is not None:
                return float(mn)
        except Exception:
            pass

        # robust fallback
        env = (
            os.environ.get("MIN_TRADE_NOTIONAL_USD")
            or os.environ.get("MIN_SELL_NOTIONAL_USD")
            or getattr(self, "min_notional_usd", None)
        )
        return float(env or LIVE_MIN_ORDER_USD)



    def pair_min_notional_usd(self, symbol: str) -> float:
        """
        Backwards-compatible shim: accept base or pair and resolve to a real Kraken pair first.
        """
        # Resolve to a ccxt symbol using the same logic as orders (no DB session here),
        # fall back to the wallet-preferred quote.
        base_app, quote_app = self._split_app_symbol(symbol)
        base_ex = self._to_exchange_base(base_app)
        mkts = getattr(self.exch, "markets", {}) or {}
        if quote_app and f"{base_ex}/{quote_app}" in mkts:
            ksym = f"{base_ex}/{quote_app}"
        else:
            for q in self._prefer_quotes_by_wallet():
                if f"{base_ex}/{q}" in mkts:
                    ksym = f"{base_ex}/{q}"
                    break
            else:
                ksym = f"{base_ex}/USDT"
        return self._pair_min_notional_usd(ksym)



    def __init__(self, engine):
        self.engine = engine
        self.exch = ccxt.kraken({
            "apiKey": KRAKEN_API_KEY,
            "secret": KRAKEN_API_SECRET,
            "enableRateLimit": True,
            # "options": {"adjustForTimeDifference": True},
        })
        try:
            self.exch.load_markets()
        except Exception as e:
            # Keep going even if load_markets warns (network hiccups, etc.)
            print(f"[broker] load_markets warning: {e}")


    def get_balance(self) -> Balance:
        """
        Report live free USD+USDT as cash, and approx equity as cash + MV of DB OPENs
        (the equity calc stays DB-based for now to avoid heavy per-call valuation).
        """
        cash = 0.0
        try:
            b = self.exch.fetch_balance() or {}
            free = b.get("free", {}) or {}
            cash = float((free.get("USD") or free.get("ZUSD") or 0.0)) + float((free.get("USDT") or free.get("ZUSDT") or 0.0))
        except Exception:
            pass

        mv = 0.0
        with Session(self.engine) as s:
            opens = s.exec(select(Position).where(Position.status == "OPEN")).all()
            for p in opens:
                # Use your DB/app symbol; candles align with app symbols
                c = s.exec(select(Candle).where(Candle.symbol == p.symbol).order_by(Candle.ts.desc())).first()
                last = float(c.close) if c else float(p.avg_price or 0.0)
                mv += last * float(p.qty or 0.0)

        return Balance(cash_usd=cash, equity_usd=cash + mv)



    def fetch_open_orders(self):
        try:
            ods = self.exch.fetch_open_orders()
            out = []
            for o in ods:
                ts = o.get("timestamp")
                iso = datetime.utcfromtimestamp(ts/1000).isoformat() if ts else None
                out.append({
                    "time": iso,
                    "symbol": o.get("symbol"),
                    "side": (o.get("side") or "").upper(),
                    "price": float(o.get("price") or 0.0),
                    "status": (o.get("status") or "open").upper(),
                })
            return out
        except Exception as e:
            print(f"[broker] fetch_open_orders failed: {e}")
            return []


    def _ccxt_symbol_for(self, session: Session, base: str) -> str:
        base2 = "BTC" if base.upper() in ("XBT", "BTC") else base.upper()

        # which quote do we actually have cash in?
        try:
            bal = self.exch.fetch_balance() or {}
            free = bal.get("free", {}) or {}
            usd_free  = float((free.get("USD")  or free.get("ZUSD")  or 0) or 0)
            usdt_free = float((free.get("USDT") or free.get("ZUSDT") or 0) or 0)
            pref_quotes = ["USD", "USDT"] if usd_free >= usdt_free else ["USDT", "USD"]
        except Exception:
            pref_quotes = ["USD", "USDT"]

        markets = getattr(self.exch, "markets", {}) or {}
        def _exists(q: str) -> bool:
            return f"{base2}/{q}" in markets

        for q in pref_quotes:
            if _exists(q):
                return f"{base2}/{q}"

        try:
            row = session.exec(select(UniversePair).where(UniversePair.symbol == base2)).first()
            if row and row.quote and _exists(row.quote):
                return f"{base2}/{row.quote}"
        except Exception:
            pass

        for q in ("USD", "USDT"):
            if _exists(q):
                return f"{base2}/{q}"

        return f"{base2}/USDT"



    def place_order(
        self,
        *,
        symbol: str,             # may be "AVAX" or "AVAX/USD"
        side: str,               # "BUY" / "SELL"
        qty: float,
        order_type: str,         # "market" (only for now)
        price: float,            # signal/ref price (for logging)
        reason: str,
        session: Session,        # reuse main loop session
        score: Optional[float] = None,
    ):
        # ---- Resolve to (db_symbol, ccxt_symbol) ----
        markets = getattr(self.exch, "markets", None) or self.exch.load_markets()

        def _alias_base(b: str) -> str:
            b = (b or "").strip().upper()
            return {"XBT": "BTC", "XDG": "DOGE"}.get(b, b)

        def _alias_quote(q: str) -> str:
            q = (q or "").strip().upper()
            return {"ZUSD": "USD", "ZUSDT": "USDT", "USD": "USD", "USDT": "USDT"}.get(q, q)

        sym_raw = (symbol or "").strip().upper()
        if "/" in sym_raw:
            base_raw, quote_raw = sym_raw.split("/", 1)
            base  = _alias_base(base_raw)
            quote = _alias_quote(quote_raw)
            ccxt_symbol = f"{base}/{quote}"
            if ccxt_symbol not in markets:
                # Fallback: let broker choose best USD/USDT for this base
                ccxt_symbol = self._ccxt_symbol_for(session, base)
            db_symbol = base                         # <-- persist base-only in DB
        else:
            base = _alias_base(sym_raw)
            ccxt_symbol = self._ccxt_symbol_for(session, base)
            db_symbol = base                         # <-- persist base-only in DB

        # ---- Resolve last price for guards ----
        last = float(price or 0.0)
        if last <= 0.0:
            # First try our candle cache by DB symbol (base-only)
            c = session.exec(
                select(Candle).where(Candle.symbol == db_symbol).order_by(Candle.ts.desc())
            ).first()
            if c:
                last = float(c.close or 0.0)
        if last <= 0.0:
            # Fallback to exchange ticker for the actual market we’ll trade
            try:
                t = self.exch.fetch_ticker(ccxt_symbol) or {}
                last = float(t.get("last") or t.get("close") or 0.0)
            except Exception:
                last = 0.0

        if last <= 0.0:
            session.add(Order(
                ts=datetime.utcnow(), symbol=db_symbol, side=side, qty=0.0,
                price_req=float(price or 0.0), price_fill=0.0,
                status="REJECTED", reason="LIVE: no price"
            ))
            session.commit()
            return None

        # ---- Min notional / amount floors (use actual Kraken market) ----
        notional_est = qty * last
        try:
            pair_min = float(self._pair_min_notional_usd(ccxt_symbol))
        except Exception:
            pair_min = float(LIVE_MIN_ORDER_USD)

        if notional_est + 1e-12 < pair_min:
            if side.upper() == "SELL":
                pos = session.exec(select(Position).where(
                    Position.symbol == db_symbol, Position.status == "OPEN"
                )).first()
                max_qty = float((pos.qty if pos else 0.0) or 0.0)
                need_qty = pair_min / max(1e-12, last)
                new_qty  = min(max_qty, need_qty)
                if new_qty * last + 1e-12 < pair_min:
                    session.add(Order(
                        ts=datetime.utcnow(), symbol=db_symbol, side=side, qty=0.0,
                        price_req=last, price_fill=0.0,
                        status="REJECTED",
                        reason=f"LIVE: position ${max_qty*last:.2f} < pair min ${pair_min:.2f}"
                    ))
                    session.commit()
                    return None
                qty = float(self.exch.amount_to_precision(ccxt_symbol, new_qty))
            else:
                need_qty = pair_min / max(1e-12, last)
                qty = float(self.exch.amount_to_precision(ccxt_symbol, max(qty, need_qty)))

        # Kraken amount minimum (per-pair)
        mkt = (getattr(self.exch, "markets", {}) or {}).get(ccxt_symbol, {})
        try:
            amt_min = float((((mkt.get("limits") or {}).get("amount") or {}).get("min")) or 0.0)
        except Exception:
            amt_min = 0.0

        if amt_min and qty < amt_min:
            if side.upper() == "SELL":
                pos = session.exec(select(Position).where(
                    Position.symbol == db_symbol, Position.status == "OPEN"
                )).first()
                max_qty = float((pos.qty if pos else 0.0) or 0.0)
                qty = min(max_qty, amt_min)
                if qty <= 0.0:
                    session.add(Order(
                        ts=datetime.utcnow(), symbol=db_symbol, side=side, qty=0.0,
                        price_req=last, price_fill=0.0,
                        status="REJECTED",
                        reason=f"LIVE: volume minimum {amt_min} not met"
                    ))
                    session.commit()
                    return None
            else:
                qty = amt_min

        # Recompute and clamp to LIVE_MAX_ORDER_USD
        notional_est = qty * last
        if notional_est > LIVE_MAX_ORDER_USD:
            qty = LIVE_MAX_ORDER_USD / last

        qty2 = float(self.exch.amount_to_precision(ccxt_symbol, qty))

        # ---- Place the market order ----
        try:
            print(f"[broker] LIVE {side} {ccxt_symbol} qty={qty2} (notional~${qty2*last:.2f})")
            if order_type.lower() != "market":
                raise ValueError("Only market orders supported in this MVP live path.")
            od = (
                self.exch.create_market_buy_order(ccxt_symbol, qty2)
                if side.upper() == "BUY"
                else self.exch.create_market_sell_order(ccxt_symbol, qty2)
            )
        except Exception as e:
            print(f"[broker] create_order failed: {e}")
            session.add(Order(
                ts=datetime.utcnow(), symbol=db_symbol, side=side, qty=0.0,
                price_req=last, price_fill=0.0,
                status="REJECTED", reason=f"LIVE: {e}"
            ))
            session.commit()
            return None

        # ---- Persist Order (always DB base symbol) ----
        status  = (od.get("status") or "").lower() or "closed"
        filled  = float(od.get("filled") or qty2)
        average = float(od.get("average") or od.get("price") or last)

        session.add(Order(
            ts=datetime.utcnow(), symbol=db_symbol, side=side, qty=filled,
            price_req=last, price_fill=average,
            status=("FILLED" if status == "closed" else "PENDING"),
            reason=f"LIVE kraken {reason}"
        ))

        # ---- Mirror into DB (Positions/Trades/Wallet) with base-only symbol ----
        if side.upper() == "BUY":
            p = session.exec(select(Position).where(Position.symbol == db_symbol, Position.status == "OPEN")).first()
            if p:
                total = p.qty + filled
                p.avg_price = (p.avg_price * p.qty + average * filled) / total
                p.qty = total
            else:
                raw_stop   = (getattr(self, "_last_sig_stop", None)   or average * 0.98)
                raw_target = (getattr(self, "_last_sig_target", None) or average * 1.05)

                fee_buf  = FEE_PCT * 2.0
                slip_buf = SLIPPAGE_PCT * 2.0
                MIN_STOP_GAP_PCT = max(0.006, 3.0 * (fee_buf + slip_buf))
                min_gap_stop = average * (1.0 - MIN_STOP_GAP_PCT)
                safe_stop = min(raw_stop, min_gap_stop)

                p = Position(
                    symbol=db_symbol,       # base-only
                    qty=filled,
                    avg_price=average,
                    opened_ts=datetime.utcnow(),
                    stop=safe_stop,
                    target=raw_target,
                    status="OPEN",
                    score=(float(score) if score is not None else None),
                )
                session.add(p)
        else:
            p = session.exec(select(Position).where(Position.symbol == db_symbol, Position.status == "OPEN")).first()
            if p and filled > 0:
                close_qty = min(p.qty, filled)
                pnl = (average - float(p.avg_price)) * close_qty
                result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
                session.add(Trade(
                    symbol=db_symbol,
                    entry_ts=p.opened_ts,
                    exit_ts=datetime.utcnow(),
                    entry_px=float(p.avg_price),
                    exit_px=average,
                    qty=close_qty,
                    pnl_usd=float(pnl),
                    result=result,
                ))
                p.qty = max(0.0, p.qty - close_qty)
                if p.qty <= 1e-12:
                    p.qty = 0.0
                    p.status = "CLOSED"

                w = session.get(Wallet, 1)
                if w:
                    proceeds = average * close_qty * (1 - FEE_PCT)
                    w.balance_usd = float((w.balance_usd or 0.0) + proceeds)

        session.commit()
        return od




# -------- factory --------
def make_broker(engine):
    b = (BROKER or "sim").lower()
    if b in ("kraken-live", "kraken"):
        return KrakenLiveBroker(engine)
    elif b in ("sim",):
        return SimBroker(engine)
    else:
        # default to sim if unknown
        return SimBroker(engine)


