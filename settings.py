# settings.py
from dotenv import load_dotenv
load_dotenv(override=True)  # read .env into environment

import os

def env(name, cast=str, default=None):
    v = os.getenv(name, default)
    return cast(v) if v is not None else v

# --- Account / runtime ---
WALLET_START_USD   = float(env("WALLET_START_USD", float, 10000))
UNIVERSE           = [s.strip().upper() for s in env("UNIVERSE", str, "BTC,ETH,SOL").split(",")]
BASE_CCY           = env("BASE_CCY", str, "USDT")
POLL_SECONDS       = int(env("POLL_SECONDS", int, 30))
OPEN_POS_UPDATE_SECONDS = int(os.getenv("OPEN_POS_UPDATE_SECONDS", "30"))
HISTORY_MINUTES    = int(env("HISTORY_MINUTES", int, 600))
USE_COINGECKO      = env("USE_COINGECKO", str, "true").lower() == "true"

# --- Risk + costs ---
RISK_PCT_PER_TRADE = float(env("RISK_PCT_PER_TRADE", float, 0.02))
FEE_PCT            = float(env("FEE_PCT", float, 0.0006))      # 0.06% each side
SLIPPAGE_PCT       = float(env("SLIPPAGE_PCT", float, 0.0008)) # 0.08% each side
MAX_OPEN_POSITIONS = int(env("MAX_OPEN_POSITIONS", int, 3))
# Dust threshold: positions with USD value below this are ignored as "open"
POSITION_DUST_USD = float(env("POSITION_DUST_USD", float, 5.0))

MAX_TRADE_COST_PCT     = float(env("MAX_TRADE_COST_PCT", float, 0.50))  # ≤50% equity per trade
MAX_GROSS_EXPOSURE_PCT = float(env("MAX_GROSS_EXPOSURE_PCT", float, 1.00))  # ≤100% equity total
MIN_TRADE_NOTIONAL     = float(env("MIN_TRADE_NOTIONAL", float, 10.0))   # e.g., $10 min order
MIN_TRADE_NOTIONAL_PCT = float(env("MIN_TRADE_NOTIONAL_PCT", float, 0.002))  # e.g., 0.2% of equity
MIN_BREAKEVEN_EDGE_PCT = float(env("MIN_BREAKEVEN_EDGE_PCT", float, 0.01000)) # extra edge beyond pure breakeven

# --- Strategy (PicklrMVP defaults) ---
DET_EMA_SHORT      = int(env("DET_EMA_SHORT", int, 12))
DET_EMA_LONG       = int(env("DET_EMA_LONG", int, 26))

# PicklrMVP: “holding_days”
HOLDING_DAYS       = int(env("HOLDING_DAYS", int, 4))
MAX_HOLD_MINUTES   = HOLDING_DAYS * 1440  # used by time-based exit

# PicklrMVP: “min_volume_usd”
MIN_VOLUME_USD     = float(env("MIN_VOLUME_USD", float, 1_000_000))

# PicklrMVP: “target_pct” and “stop_pct”
TARGET_PCT         = float(env("TARGET_PCT", float, 0.095))  # 9.5%
STOP_PCT           = float(env("STOP_PCT",   float, 0.02))   # 2.0%

MIN_BREAKOUT_PCT   = float(env("MIN_BREAKOUT_PCT", float, 0.0035))  # 0.35%, not 3.5%
MIN_EMA_SPREAD     = float(env("MIN_EMA_SPREAD", float, 0.005))     # 0.5% default

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
TSL_PCT           = float(env("TSL_PCT", float, 0.06))           # 3% trail from high

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

# Trade selection knobs
MAX_NEW_POSITIONS_PER_CYCLE = int(env("MAX_NEW_POSITIONS_PER_CYCLE", int, 2))
SIGNAL_MIN_NOTIONAL_USD = float(env("SIGNAL_MIN_NOTIONAL_USD", float, 10.0))

# How often to rebuild the symbol list from Kraken
UNIVERSE_REFRESH_MINUTES = int(env("UNIVERSE_REFRESH_MINUTES", int, 15))

# Signal gates / chooser (env-backed)
REQUIRE_BREAKOUT = env("REQUIRE_BREAKOUT", str, "true").lower() == "true"
COOLDOWN_MINUTES = int(env("COOLDOWN_MINUTES", int, 60))
MAX_NEW_POSITIONS_PER_CYCLE = int(env("MAX_NEW_POSITIONS_PER_CYCLE", int, 2))
SIGNAL_MIN_NOTIONAL_USD = float(env("SIGNAL_MIN_NOTIONAL_USD", float, 10.0))

# Quality filters (env-backed)
MAX_EXTENSION_PCT   = float(env("MAX_EXTENSION_PCT", float, 0.08))
MIN_RR              = float(env("MIN_RR", float, 1.5))
EMA_SLOPE_LOOKBACK  = int(env("EMA_SLOPE_LOOKBACK", int, 5))

# Debug
ENABLE_DEBUG_SIGNALS = env("ENABLE_DEBUG_SIGNALS", str, "false").lower() == "true"

# Trend fallback (allows entries on strong uptrends without a strict breakout)
ALLOW_TREND_ENTRY = env("ALLOW_TREND_ENTRY", str, "true").lower() == "true"
EMA_SLOPE_MIN = float(env("EMA_SLOPE_MIN", float, 0.0))  # require slope >= 0 by default

# ====== MODEL / FEATURES / ATR SETTINGS (add at end) ======

# Model toggle and path
USE_MODEL        = env("USE_MODEL", str, "true").lower() == "true"
MODEL_PATH       = env("MODEL_PATH", str, "models/xgb_target_first.json")
SCORE_THRESHOLD  = float(env("SCORE_THRESHOLD", float, 0.10))

# How far in the future we label wins (hours), and how many past days to build rows
HORIZON_HOURS    = int(env("HORIZON_HOURS", int, 168))  # up to 7 days
FEATURE_DAYS     = int(env("FEATURE_DAYS", int, 90))

# ATR length for volatility-based logic
ATR_LEN          = int(env("ATR_LEN", int, 14))

# ATR-based stops/targets (Option D)
USE_ATR_STOPS    = env("USE_ATR_STOPS", str, "true").lower() == "true"
ATR_STOP_MULT    = float(env("ATR_STOP_MULT", float, 1.8))
ATR_TARGET_MULT  = float(env("ATR_TARGET_MULT", float, 4.5))

# Trailing with ATR (Option A). We keep your existing % trail too.
TSL_USE_ATR      = env("TSL_USE_ATR", str, "true").lower() == "true"
TSL_ATR_MULT     = float(env("TSL_ATR_MULT", float, 3.0))
# When price reaches the old target, tighten the trail (e.g., 70% of original trail)
TSL_TIGHTEN_MULT = float(env("TSL_TIGHTEN_MULT", float, 0.7))

# Risk scaling by model score (Option B). 1.0 = no scale. Cap keeps it safe.
RISK_MAX_MULTIPLIER = float(env("RISK_MAX_MULTIPLIER", float, 1.8))

BROKER = env("BROKER", str, "sim")

KRAKEN_API_KEY    = env("KRAKEN_API_KEY", str, "")
KRAKEN_API_SECRET = env("KRAKEN_API_SECRET", str, "")

LIVE_MAX_ORDER_USD = float(env("LIVE_MAX_ORDER_USD", float, 50.0))
LIVE_MIN_ORDER_USD = float(env("LIVE_MIN_ORDER_USD", float, 5.0))

# ===== Backfill controls =====
BACKFILL_DAYS_DEFAULT   = int(env("BACKFILL_DAYS_DEFAULT", int, 365))  # default history: how far back in time in days to fetch data
BACKFILL_CONCURRENCY    = int(env("BACKFILL_CONCURRENCY", int, 6))     # parallel symbols: how many symbols to fetch data for simultaneously, higher is faster but may stress Kraken rate limits (5-6)
BACKFILL_PAUSE_MS       = int(env("BACKFILL_PAUSE_MS", int, 600))      # sleep between page fetches: A pause between pulling pages to prevent hitting Kraken rate limits (600-700)
BACKFILL_MAX_PAIRS      = int(env("BACKFILL_MAX_PAIRS", int, 0))       # 0 = no cap; else cap symbol count 

# --- TP1 controls ---
TP1_PCT = float(os.getenv("TP1_PCT", "0.037"))      # 3.7% default
TP1_SELL_FRAC = float(os.getenv("TP1_SELL_FRAC", "0.30"))  # sell 30% by default
MOVE_BE_ON_TP1 = bool(int(os.getenv("MOVE_BE_ON_TP1", "1")))  # move BE after TP1