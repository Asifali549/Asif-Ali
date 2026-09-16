"""
BTC Regime Filter - Core Logic
================================
Signal ko sirf tab valid maanta hai jab, signal ke waqt, BTC ka daily
close apne 200-day EMA se upar ho (yani "bull/recovery" regime).

Isay backtest_engine.py ke signals par EXTRA gate ke taur par lagaya
jata hai - koi strategy tabdeel nahi hoti, sirf trade liya ya nahi
liya jata hai.
"""

import pandas as pd


def compute_btc_regime_series(btc_daily_df, ema_period=200):
    """
    btc_daily_df: BTC/USDT daily OHLCV dataframe (columns: timestamp, close, ...)
    Returns a pandas Series (indexed by timestamp) of booleans:
        True  = us din BTC close > 200 EMA (bullish regime)
        False = neeche (bearish/uncertain regime)
    """
    ema = btc_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = btc_daily_df["close"] > ema
    result = pd.Series(is_bullish.values, index=btc_daily_df["timestamp"].values)
    return result


def is_bullish_regime(regime_series, signal_timestamp):
    """
    Kisi bhi (intraday) signal timestamp ke liye, us din (ya us se pehle
    wale aakhri available daily bar) ka regime dekhta hai.

    regime_series: compute_btc_regime_series() ka result (daily, DatetimeIndex)
    signal_timestamp: pandas Timestamp (signal ka waqt, kisi bhi TF ka)

    Returns True/False. Agar data na mile (bohot purana signal), True
    return karta hai (conservative: filter na lagaye, taake data-missing
    signals ko galat tarah se reject na kare).
    """
    valid = regime_series[regime_series.index <= signal_timestamp]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])