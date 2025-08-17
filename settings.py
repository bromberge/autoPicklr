# settings.py
from dotenv import load_dotenv
load_dotenv(override=True)  # read .env into environment

import os

def env(name, cast=str, default=None):
    v = os.getenv(name, default)
    return cast(v) if v is not None else v

# --- Account / runtime ---
WALLET_START_USD   = float(env("WALLET_START_USD", float, 1000))
UNIVERSE           = [s.strip().upper() for s in env("UNIVERSE", str, "BTC,ETH,SOL").split(",")]
BASE_CCY           = env("BASE_CCY", str, "USDT")
POLL_SECONDS       = int(env("POLL_SECONDS", int, 60))
HISTORY_MINUTES    = int(env("HISTORY_MINUTES", int, 600))
USE_COINGECKO      = env("USE_COINGECKO", str, "true").lower() == "true"

# --- Risk + costs ---
RISK_PCT_PER_TRADE = float(env("RISK_PCT_PER_TRADE", float, 0.02))
FEE_PCT            = float(env("FEE_PCT", float, 0.0006))      # 0.06% each side
SLIPPAGE_PCT       = float(env("SLIPPAGE_PCT", float, 0.0008)) # 0.08% each side
MAX_OPEN_POSITIONS = int(env("MAX_OPEN_POSITIONS", int, 3))

# Caps to keep orders realistic
MAX_TRADE_COST_PCT     = float(env("MAX_TRADE_COST_PCT", float, 0.50))  # ≤50% equity per trade
MAX_GROSS_EXPOSURE_PCT = float(env("MAX_GROSS_EXPOSURE_PCT", float, 1.00))  # ≤100% equity total
MIN_TRADE_NOTIONAL     = float(env("MIN_TRADE_NOTIONAL", float, 10.0))   # e.g., $10 min order
MIN_TRADE_NOTIONAL_PCT = float(env("MIN_TRADE_NOTIONAL_PCT", float, 0.002))  # e.g., 0.2% of equity
MIN_BREAKEVEN_EDGE_PCT = float(env("MIN_BREAKEVEN_EDGE_PCT", float, 0.0000)) # extra edge beyond pure breakeven

# --- Strategy (PicklrMVP defaults) ---
DET_EMA_SHORT      = int(env("DET_EMA_SHORT", int, 12))
DET_EMA_LONG       = int(env("DET_EMA_LONG", int, 26))

# PicklrMVP: “holding_days”
HOLDING_DAYS       = int(env("HOLDING_DAYS", int, 4))
MAX_HOLD_MINUTES   = HOLDING_DAYS * 1440  # used by time-based exit

# PicklrMVP: “min_volume_usd”
MIN_VOLUME_USD     = float(env("MIN_VOLUME_USD", float, 20_000_000))

# PicklrMVP: “target_pct” and “stop_pct”
TARGET_PCT         = float(env("TARGET_PCT", float, 0.095))  # 9.5%
STOP_PCT           = float(env("STOP_PCT",   float, 0.02))   # 2.0%

# PicklrMVP: “min_breakout_pct”
MIN_BREAKOUT_PCT   = float(env("MIN_BREAKOUT_PCT", float, 0.0035))  # 0.35%

# PicklrMVP: “min_ema_spread” (EMA separation as a % of price)
MIN_EMA_SPREAD     = float(env("MIN_EMA_SPREAD", float, 0.005))     # 0.5%

# Other chooser knobs
BREAKOUT_LOOKBACK  = int(env("BREAKOUT_LOOKBACK", int, 20))
CHOOSER_THRESHOLD  = float(env("CHOOSER_THRESHOLD", float, 0.60))  # score cutoff

# Debug toggle (OFF by default)
ENABLE_DEBUG_SIGNALS = env("ENABLE_DEBUG_SIGNALS", str, "false").lower() == "true"

# Optional integrations
TELEGRAM_BOT_TOKEN = env("TELEGRAM_BOT_TOKEN", str, "")
TELEGRAM_CHAT_ID   = env("TELEGRAM_CHAT_ID", str, "")

# --- NEW: Exit Enhancements ---
# Partial take-profits: sell portions at these gains from entry (fractions)
# Example: 0.05 = +5%, 0.10 = +10%
PTP_LEVELS = [float(x) for x in env("PTP_LEVELS", str, "0.05,0.10").split(",")]  # % gains
PTP_SIZES  = [float(x) for x in env("PTP_SIZES",  str, "0.30,0.30").split(",")]  # fractions of original qty

# Break-even move: when to move stop to entry (either absolute trigger or after TP1)
BE_TRIGGER_PCT     = float(env("BE_TRIGGER_PCT", float, 0.03))  # e.g., +3% open profit
BE_AFTER_FIRST_TP   = env("BE_AFTER_FIRST_TP", str, "true").lower() == "true"  # also move at TP1

# Trailing stop: start trailing after first TP; trail by this percent from highest price since activation
TSL_ACTIVATE_AFTER_TP = env("TSL_ACTIVATE_AFTER_TP", str, "true").lower() == "true"
TSL_ACTIVATE_PCT  = float(env("TSL_ACTIVATE_PCT", float, 0.05))  # if not after TP, activate after +5%
TSL_PCT           = float(env("TSL_PCT", float, 0.03))           # 3% trail from high

# --- Live data controls ---
COINGECKO_API_KEY = env("COINGECKO_API_KEY", str, "")  # add your key in Replit Secrets
DATA_FETCH_INTERVAL_SECONDS = int(env("DATA_FETCH_INTERVAL_SECONDS", int, 60))  # 1 fetch/min/symbol

# Dynamic universe knobs
UNIVERSE_TOP_N = int(env("UNIVERSE_TOP_N", int, 150))          # raise cap
UNIVERSE_CACHE_MINUTES = int(env("UNIVERSE_CACHE_MINUTES", int, 60))
UNIVERSE_EXCLUDE = [s.strip().upper() for s in env("UNIVERSE_EXCLUDE", str, "").split(",") if s.strip()]
UNIVERSE_INCLUDE = [s.strip().upper() for s in env("UNIVERSE_INCLUDE", str, "").split(",") if s.strip()]

# Allow quotes: include USDT to dramatically expand candidates on Kraken
ALLOWED_QUOTES = [s.strip().upper() for s in env("ALLOWED_QUOTES", str, "USD,USDT").split(",")]

# Lower the 24h USD-volume threshold (Kraken’s USD markets are thinner than Binance’s)
MIN_VOLUME_USD = float(env("MIN_VOLUME_USD", float, 1_000_000))


