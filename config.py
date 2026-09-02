"""
Config - Sab settings yahan se control hoti hain
"""

# ============================================================
# EXCHANGE SETTINGS
# ============================================================
EXCHANGE_ID = "kucoin"            # ccxt exchange name (Binance blocks cloud server IPs jaise Streamlit Cloud, is liye KuCoin istemal kar rahe hain)
MARKET_TYPE = "spot"             # spot only (buy-side trading)

# ============================================================
# COIN LIST
# ============================================================
# Option A: manual list daalain
# Option B: "AUTO" rakhain -> top N USDT spot pairs by volume khud fetch honge
COIN_LIST_MODE = "AUTO"            # "AUTO" ya "MANUAL" -> ab 400 coins par CE(12) vs CE(16) confirm
MANUAL_COIN_LIST = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "TRX/USDT", "MATIC/USDT", "LTC/USDT", "SHIB/USDT", "UNI/USDT",
    "ATOM/USDT", "ETC/USDT", "XLM/USDT", "NEAR/USDT", "APT/USDT",
    "FIL/USDT", "ARB/USDT", "OP/USDT", "SUI/USDT", "INJ/USDT",
]
AUTO_TOP_N_COINS = 400            # AUTO mode mein kitne coins lene hain

# ============================================================
# EXCLUDED COINS — Meme coins aur Binance leveraged/binary tokens
# ============================================================
# Leveraged tokens (Binance ke "3L/3S", "UP/DOWN", "BULL/BEAR" suffix wale)
# ye normal spot coins nahi, price 2x/3x leverage se track karte hain
EXCLUDE_LEVERAGED_SUFFIXES = ["UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S"]

# Jaani-mani meme coins (naam se pehchan kar exclude karte hain)
EXCLUDE_MEME_COINS = [
    "DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "MEME", "ELON",
    "BABYDOGE", "SAFEMOON", "AKITA", "KISHU", "HOGE", "DOGELON",
    "MOG", "BRETT", "POPCAT", "TURBO", "MYRO", "BOME", "SLERF",
    "WOJAK", "PEOPLE", "LADYS", "AIDOGE", "DOGGY", "CATE", "CAT",
    "PIT", "NEIRO", "GOAT", "PNUT", "ACT", "CHILLGUY", "FWOG",
]

def is_excluded_coin(symbol: str) -> bool:
    """
    symbol jaisa 'BTC/USDT' -> base = 'BTC'.
    True return karta hai agar ye meme coin ya leveraged/binary token hai.
    """
    base = symbol.split("/")[0].upper()

    for suffix in EXCLUDE_LEVERAGED_SUFFIXES:
        if base.endswith(suffix) and len(base) > len(suffix):
            return True

    if base in EXCLUDE_MEME_COINS:
        return True

    return False
QUOTE_CURRENCY = "USDT"

# ============================================================
# TIMEFRAMES
# ============================================================
TIMEFRAMES = ["15m", "1h", "4h"]

# Har timeframe ke liye kitni candles fetch/backtest karni hain
CANDLE_LIMITS = {
    "15m": 2000,   # ~20 din
    "1h": 2000,    # ~83 din
    "4h": 1500,    # ~250 din
}

# ============================================================
# STRATEGY PARAMETERS
# ============================================================
STRATEGY_PARAMS = {
    "ema_crossover": {
        "fast": 9,
        "slow": 21,
        "trend_filter": 200,       # is EMA se upar ho tabhi valid buy
        "golden_cross_filter": 50,  # NEW: EMA50 > EMA200 bhi zaroori (extra tight)
        "volume_mult": 3.0,         # TIGHT (pehle 2.0): avg volume se 3x zyada chahiye
        "volume_avg_period": 20,
    },
    "breakout": {
        "lookback": 30,             # TIGHT (pehle 20 tha): N candle ka high
        "volume_mult": 3.0,         # TIGHT (pehle 2.0)
        "volume_avg_period": 20,
        "trend_filter": 200,
        "golden_cross_filter": 50,  # NEW
    },
    "momentum_volume": {
        "roc_period": 10,           # rate of change period
        "roc_threshold": 6.0,       # TIGHT (pehle 4.0): % price change threshold
        "volume_mult": 3.0,         # TIGHT (pehle 2.0)
        "volume_avg_period": 20,
        "trend_filter": 200,
        "golden_cross_filter": 50,  # NEW
    },
    "ichimoku": {
        "tenkan": 9,
        "kijun": 26,
        "senkou_b": 52,
        "displacement": 26,
        "trend_filter": 200,
        "golden_cross_filter": 50,  # NEW
        "volume_mult": 2.0,         # TIGHT (pehle 1.3)
        "volume_avg_period": 20,
    },
    "market_structure": {
        "pivot_lookback": 5,        # left/right bars pivot detect karne ke liye
        "min_swing_pct": 1.5,       # TIGHT (pehle 1.0): chhote noise swings ignore karne ke liye
        "trend_filter": 200,        # NEW
        "golden_cross_filter": 50,  # NEW
    },
    "fvg": {
        "trend_filter": 200,        # sirf uptrend mein FVG zones par buy karein
        "golden_cross_filter": 50,
        "max_zone_age_bars": 100,   # itne bars ke baad purani unfilled FVG zone bhool jao
        "min_gap_pct": 0.6,         # TIGHT (pehle 0.1): sirf bara/asli gap maanein
        "min_candle_body_pct": 0.8, # NEW: beech wali (impulse) candle ka body kam az kam itna bara ho (%)
        "volume_mult": 1.8,         # NEW: bounce candle par volume confirmation
        "volume_avg_period": 20,
    },
    "volume_profile": {
        "lookback_period": 100,     # kitni candles ka profile banaye
        "num_bins": 24,             # price ko kitne bins mein baante
        "value_area_pct": 0.70,     # 70% value area (standard)
        "bounce_tolerance_pct": 0.5,# VAL/POC ke kitna qareeb ho to "touch" maana jaye
        "trend_filter": 200,
        "golden_cross_filter": 50,  # NEW
        "volume_mult": 2.5,         # TIGHT (pehle 1.5)
        "volume_avg_period": 20,
    },
}

# Signal Cooldown: ek signal ke baad, usi coin/strategy par kitne bars tak
# agla signal repeat NAHI hoga (spam/clutter rokne ke liye)
SIGNAL_COOLDOWN_BARS = 15

# Strategies ON/OFF (backtest ke liye)
# NOTE: Sirf wahi strategies rakhi hain jo TOP-3 confluence pairs banati hain.
# Momentum+Volume aur Volume Profile confluence test mein kamzor nikli (PF < 1
# har combo mein), is liye nikal di gayi hain.
ACTIVE_STRATEGIES = [
    "ema_crossover",
    "breakout",
    "ichimoku",
    "market_structure",
    "fvg",   # NEW: Fair Value Gap (retracement entry)
]

# TOP-3 confluence pairs (25-coin test se pata chale):
#   1. ichimoku + market_structure   -> PF 2.51 (best)
#   2. ema_crossover + ichimoku      -> PF 2.13
#   3. ema_crossover + breakout      -> PF 1.75
# FVG (tight version) ko har mukhtalif strategy ke sath alag test karne ke liye:
CONFLUENCE_PAIRS = [
    ("ichimoku", "market_structure"),
    ("ema_crossover", "ichimoku"),
    ("ema_crossover", "breakout"),
    ("fvg", "market_structure"),
    ("fvg", "ema_crossover"),
    ("fvg", "breakout"),
]

# ============================================================
# BACKTEST / TRADE SIMULATION SETTINGS
# ============================================================
# Buy-only (spot) simulation rules
BACKTEST_PARAMS = {
    "atr_period": 14,
    "stop_loss_atr_mult": 1.5,      # SL = entry - 1.5*ATR (sirf tab istemal hoga jab exit_mode="fixed_atr")
    "take_profit_atr_mult": 3.0,    # TP = entry + 3*ATR  (1:2 risk-reward)
    "max_hold_bars": 50,            # is se zyada bars tak position open na rahe (time-based exit)
    "fee_pct": 0.1,                 # per side trading fee %
    "slippage_pct": 0.05,           # entry/exit slippage %

    # Exit mode: "fixed_atr" (purana tareeqa - fixed SL/TP) ya
    # "chandelier" (Chandelier Exit trailing stop, jaisa maanga gaya)
    "exit_mode": "chandelier",
}

# Chandelier Exit settings (trailing stop loss)
# Long position ke liye: stop = Highest High (period) - multiplier * ATR(period)
# Jaise jaise price upar jata hai, stop bhi upar trail karta hai (kabhi neeche nahi aata)
CHANDELIER_EXIT_PARAMS = {
    "period": 12,       # jitna maanga gaya (default TradingView mein 22 hota hai)
    "multiplier": 3.0,
}

# ============================================================
# OUTPUT
# ============================================================
RESULTS_DIR = "results"