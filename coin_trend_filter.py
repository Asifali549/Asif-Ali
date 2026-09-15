"""
Per-Coin 4h Trend Alignment Filter - Core Logic
====================================================
BTC ke macro trend se alag: yahan HAR coin apna 4h trend khud check
karta hai. Signal (1h par) sirf tab valid hai jab us coin ka 4h
close apni 4h EMA(50) se upar ho - yani us coin ka apna bara trend
bhi bullish ho, sirf 1h ka chota signal na ho.
"""

import pandas as pd


def compute_ema_trend_series(df_4h, ema_period=50):
    """
    df_4h: coin ka 4h OHLCV dataframe (columns: timestamp, close, ...)
    Returns pandas Series (DatetimeIndex) of booleans:
        True  = us 4h candle par close > EMA(period)
        False = neeche
    """
    ema = df_4h["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = df_4h["close"] > ema
    result = pd.Series(is_bullish.values, index=pd.to_datetime(df_4h["timestamp"]).values)
    return result


def is_trend_aligned(trend_series, signal_timestamp):
    """
    Signal ke waqt (1h, kisi bhi waqt) ke liye, us se pehle ya usi waqt
    tak ka sab se aakhri 4h candle ka trend dekhta hai.

    Agar data na mile (bohot purana signal ya trend_series khali), True
    return karta hai (conservative - filter na lagaye).
    """
    if trend_series is None or len(trend_series) == 0:
        return True
    valid = trend_series[trend_series.index <= signal_timestamp]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])