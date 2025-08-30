# brokers/__init__.py
from .base import Broker
from .sim_broker import SimBroker
from settings import BROKER
from typing import Optional

def make_broker(engine) -> Broker:
    b = (BROKER or "sim").lower()
    if b == "sim":
        return SimBroker(engine)
    # later: if b == "binance_testnet": return BinancePaperBroker(...)
    # later: if b == "alpaca_paper": return AlpacaPaperBroker(...)
    return SimBroker(engine)
