"""
BOT CORE - dono live bots (Ichimoku 4H + Donchian Daily) ke mushtarka helpers
==========================================================================
Pehle ye cheezein bhari test files (strategy_lab / ichimoku4h_validation / backtest_engine)
se aati thin. Ab sirf yehi chhoti file chahiye - hisaab bilkul wahi hai.
"""
import copy
import time

import pandas as pd

import config
from strategies import STRATEGY_FUNCTIONS, apply_cooldown

FEE = config.BACKTEST_PARAMS["fee_pct"] / 100
SLIP = config.BACKTEST_PARAMS["slippage_pct"] / 100
STOP_SLIP = 0.0025          # stop-exit par extra slippage (tez girawat mein stop neeche fill hota hai)

STABLES = {"USDC", "USDT", "DAI", "TUSD", "FDUSD", "USDD", "USDP", "PYUSD", "BUSD", "USDE", "EURC",
           "EUR", "UST", "USTC", "USD1", "RLUSD", "USDS", "PAXG", "XAUT"}

# 4H Ichimoku + Market Structure - tasdeeq-shuda production settings
ICHI_BASE = {"tenkan": 9, "kijun": 26, "senkou_b": 52, "vol_mult": 2.0, "pivot": 5, "swing": 1.5,
             "cool": config.SIGNAL_COOLDOWN_BARS}

TF_MS = {"1d": 86_400_000, "4h": 14_400_000, "1h": 3_600_000, "15m": 900_000}


def norm(df):
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"]).astype("datetime64[ns]")
    return d.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def compute_atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def chandelier(df, period, mult):
    return (df["high"].rolling(period).max() - mult * compute_atr(df, period)).values


def fetch_full(exchange, symbol, timeframe, limit):
    """Aage ki taraf page kar ke poori history (beech mein khala nahi) - sirf BAND candles."""
    step = TF_MS[timeframe]
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    since = now_ms - limit * step
    rows, empty_jumps = [], 0
    while since < now_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        time.sleep(exchange.rateLimit / 1000)
        batch = [b for b in batch if b[0] >= since] if batch else []
        if not batch:
            since += 1000 * step          # coin shayad baad mein list hua - aage jump
            empty_jumps += 1
            if empty_jumps > 20:
                break
            continue
        rows.extend(batch)
        since = batch[-1][0] + step
        if len(batch) < 5 and since >= now_ms - 2 * step:
            break
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"] + step <= now_ms]            # sirf BAND candles
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.reset_index(drop=True)


def ichi_signal(df, p=ICHI_BASE):
    """Ichimoku AUR Market Structure dono ek hi candle par (cooldown ke sath)."""
    ip = copy.deepcopy(config.STRATEGY_PARAMS["ichimoku"])
    ip.update({"tenkan": p["tenkan"], "kijun": p["kijun"], "senkou_b": p["senkou_b"], "volume_mult": p["vol_mult"]})
    mp = copy.deepcopy(config.STRATEGY_PARAMS["market_structure"])
    mp.update({"pivot_lookback": p["pivot"], "min_swing_pct": p["swing"]})
    a = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, ip), p["cool"])
    b = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, mp), p["cool"])
    return (a & b).values
