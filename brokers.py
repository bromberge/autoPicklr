# brokers.py
from typing import Optional
from sqlmodel import Session
from models import Wallet
from sim import place_buy, get_last_price

# brokers.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import ccxt
from sqlmodel import Session, select

from models import Wallet, Order, Position, Trade, Candle
from universe import UniversePair
from settings import (
    BROKER,
    KRAKEN_API_KEY, KRAKEN_API_SECRET,
    LIVE_MAX_ORDER_USD, LIVE_MIN_ORDER_USD,
    FEE_PCT,   # <— add this
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

    def __init__(self, engine):
         self.engine = engine
         self.exch = ccxt.kraken({
             "apiKey": KRAKEN_API_KEY,
             "secret": KRAKEN_API_SECRET,
             "enableRateLimit": True,
             # You can add "options": {"adjustForTimeDifference": True}
         })

    def get_balance(self) -> Balance:
        def _grab(bal, code):
            return float(
                (bal.get("free", {}).get(code) or
                 bal.get("total", {}).get(code) or
                 bal.get(code, 0.0)) or 0.0
            )

        cash = 0.0
        try:
            b = self.exch.fetch_balance()
            # cover normalized and legacy kraken codes
            usd  = _grab(b, "USD")  + _grab(b, "ZUSD")
            usdt = _grab(b, "USDT") + _grab(b, "ZUSDT")
            cash = usd + usdt
        except Exception as e:
            print(f"[broker] fetch_balance failed: {e}")

        mv = 0.0
        with Session(self.engine) as s:
            opens = s.exec(select(Position).where(Position.status == "OPEN")).all()
            for p in opens:
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
        """
        Prefer the cached universe quote (USD/USDT).
        CCXT will map BTC/XBT automatically.
        """
        row = session.exec(select(UniversePair).where(UniversePair.symbol == base)).first()
        quote = (row.quote if row and row.quote in ("USDT", "USD") else "USDT")
        base2 = "BTC" if base.upper() in ("XBT", "BTC") else base.upper()
        return f"{base2}/{quote}"

    def place_order(
        self,
        *,
        symbol: str,             # base symbol, e.g. "BTC"
        side: str,               # "BUY" / "SELL"
        qty: float,
        order_type: str,         # "market" (only for now)
        price: float,            # your signal entry (for logging / sanity)
        reason: str,
        score: Optional[float] = None,
        session: Session,        # IMPORTANT: reuse main loop session
    ):
        ccxt_symbol = self._ccxt_symbol_for(session, symbol)

        # Notional sanity checks (quick guardrails)
        last = price
        if not last or last <= 0:
            c = session.exec(select(Candle).where(Candle.symbol == symbol).order_by(Candle.ts.desc())).first()
            last = float(c.close) if c else None
        if not last or last <= 0:
            session.add(Order(
                ts=datetime.utcnow(), symbol=symbol, side=side, qty=0.0,
                price_req=price or 0.0, price_fill=0.0,
                status="REJECTED", reason="LIVE: no price"
            ))
            # Mirror SELL fills into DB position/trade
            if side.upper() == "SELL":
                p = session.exec(select(Position).where(Position.symbol == symbol, Position.status == "OPEN")).first()
                if p:
                    filled_qty = min(float(filled), float(p.qty or 0.0))
                    exit_px = float(average or last)
                    # book trade
                    pnl = (exit_px - float(p.avg_price or 0.0)) * filled_qty
                    session.add(Trade(
                        symbol=symbol,
                        entry_ts=p.opened_ts,
                        exit_ts=datetime.utcnow(),
                        entry_px=float(p.avg_price or 0.0),
                        exit_px=exit_px,
                        qty=filled_qty,
                        pnl_usd=float(pnl),
                        result=("WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"),
                    ))
                    # reduce/close position
                    p.qty = max(0.0, float(p.qty or 0.0) - filled_qty)
                    if p.qty == 0.0:
                        p.status = "CLOSED"

            session.commit()
            return None

        notional_est = qty * last
        if notional_est < LIVE_MIN_ORDER_USD:
            session.add(Order(
                ts=datetime.utcnow(), symbol=symbol, side=side, qty=0.0,
                price_req=price, price_fill=0.0,
                status="REJECTED", reason=f"LIVE: below min notional ${LIVE_MIN_ORDER_USD:.2f}"
            ))
            session.commit()
            return None
        if notional_est > LIVE_MAX_ORDER_USD:
            # Hard cap to keep first live runs tiny
            qty = LIVE_MAX_ORDER_USD / last

        # Exchange precision
        qty2 = float(self.exch.amount_to_precision(ccxt_symbol, qty))

        try:
            print(f"[broker] LIVE {side} {ccxt_symbol} qty={qty2} (notional~${qty2*last:.2f})")
            if order_type.lower() != "market":
                raise ValueError("Only market orders supported in this MVP live path.")
            if side.upper() == "BUY":
                od = self.exch.create_market_buy_order(ccxt_symbol, qty2)
            else:
                od = self.exch.create_market_sell_order(ccxt_symbol, qty2)
        except Exception as e:
            print(f"[broker] create_order failed: {e}")
            session.add(Order(
                ts=datetime.utcnow(), symbol=symbol, side=side, qty=0.0,
                price_req=price or last, price_fill=0.0,
                status="REJECTED", reason=f"LIVE: {e}"
            ))
            session.commit()
            return None

        # Best-effort fill info (Kraken usually fills markets immediately)
        status  = (od.get("status") or "").lower() or "closed"
        filled  = float(od.get("filled") or qty2)
        average = float(od.get("average") or od.get("price") or last)

        session.add(Order(
            ts=datetime.utcnow(), symbol=symbol, side=side, qty=filled,
            price_req=price or last, price_fill=average,
            status=("FILLED" if status == "closed" else "PENDING"),
            reason=f"LIVE kraken {reason}"
        ))

        # --- Mirror into our DB (Position/Trade/Wallet) ---
        if side.upper() == "BUY":
            # Add/average into an open position
            p = session.exec(select(Position).where(Position.symbol == symbol, Position.status == "OPEN")).first()
            if p:
                total = p.qty + filled
                p.avg_price = (p.avg_price * p.qty + average * filled) / total
                p.qty = total
            else:
                p = Position(
                    symbol=symbol,
                    qty=filled,
                    avg_price=average,
                    opened_ts=datetime.utcnow(),
                    stop=average * 0.98,     # placeholder; your mgmt will manage these
                    target=average * 1.05,
                    status="OPEN",
                    score=(float(score) if score is not None else None),
                )
                session.add(p)

        else:
            # SELL: close/partial-close an existing position and book a Trade
            p = session.exec(select(Position).where(Position.symbol == symbol, Position.status == "OPEN")).first()
            if p and filled > 0:
                close_qty = min(p.qty, filled)

                # Book the trade
                pnl = (average - float(p.avg_price)) * close_qty
                result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
                session.add(Trade(
                    symbol=symbol,
                    entry_ts=p.opened_ts,
                    exit_ts=datetime.utcnow(),
                    entry_px=float(p.avg_price),
                    exit_px=average,
                    qty=close_qty,
                    pnl_usd=float(pnl),
                    result=result,
                ))

                # Reduce position size / close
                p.qty = max(0.0, p.qty - close_qty)
                if p.qty <= 1e-12:
                    p.qty = 0.0
                    p.status = "CLOSED"

                # Credit wallet cash with net proceeds (approx taker fee via FEE_PCT)
                w = session.get(Wallet, 1)
                if w:
                    proceeds = average * close_qty * (1 - FEE_PCT)
                    w.balance_usd = float((w.balance_usd or 0.0) + proceeds)
                    # equity_usd will be refreshed by your periodic balance sync;
                    # we leave it as-is for now.

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

class KrakenPaperBroker:
    """
    Paper broker that uses your existing sim.* functions under the hood.
    It matches the call made from trading_loop: .place_order(...).
    """
    name = "kraken-paper"
    paper = False
    live = True

    def __init__(self, engine):
        self.engine = engine

    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        price: Optional[float] = None,
        reason: str = "",
        score: Optional[float] = None,
        session: Optional[Session] = None,
    ):
        if session is None:
            # Safety: trading_loop already passes a Session, but keep this fallback.
            from sqlmodel import Session as _S
            with _S(self.engine) as s:
                return self._place_order_impl(s, symbol, side, qty, order_type, price, reason, score)
        else:
            return self._place_order_impl(session, symbol, side, qty, order_type, price, reason, score)

    def _place_order_impl(
        self,
        s: Session,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        price: Optional[float],
        reason: str,
        score: Optional[float],
    ):
        # For paper mode, BUY just delegates to the simulator's place_buy (which books orders/positions/wallet)
        if side.upper() == "BUY":
            px = price or (get_last_price(s, symbol) or 0.0)
            return place_buy(s, symbol, qty, px, reason, score=score)

        # For now we don't route SELLs here; sim.mark_to_market_and_manage()
        # handles TPs/BE/TSL/Stops internally in paper mode.
        # You can expand this later for live trading exits.
        raise NotImplementedError("SELL routing via broker is not used in paper mode yet.")



