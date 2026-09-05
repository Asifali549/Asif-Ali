"""
Advanced Anti-Fakeout Confluence Screener - Core Engine

10-Point Confluence Scoring:
  - Market Structure (BOS/CHoCH)      = 2 points
  - RVOL Volume Spike (>1.8x)          = 2 points
  - BTC Bullish (Daily > EMA50)        = 2 points  (USDT.D weakness bonus - live only, backtest mein shamil nahi)
  - Price near Demand Zone / EMA50     = 2 points
  - ADX > 20 AND RSI (40-65)           = 2 points

BUY signal jab score >= threshold (default 6/10) AND fresh BOS/CHoCH ho.

IMPORTANT (transparency):
  - Candle CLOSE par hi calculate hota hai (no repainting) - hum sirf
    band ho chuki candles ka data istemal karte hain.
  - USDT.Dominance sirf LIVE scan mein (CoinGecko se) - backtest mein
    is component ko shamil NAHI kiya gaya kyunke historical USDT.D
    data mufat mein available nahi (Pine Script mein CRYPTOCAP:USDT.D
    se ye theek se milta hai, is liye Pine version mein poora hai).
"""

import numpy as np
import pandas as pd


# ============================================================
# INDICATORS
# ============================================================
def compute_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_adx(df, period=14):
    """Standard Wilder's ADX."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ============================================================
# MARKET STRUCTURE (BOS + CHoCH)
# ============================================================
def detect_market_structure(df, pivot_lookback=5, min_swing_pct=1.0):
    """
    BOS (Break of Structure)  = pehle se bullish trend mein continuation break
    CHoCH (Change of Character) = bearish se bullish switch ka pehla break

    Return: bos_signal, choch_signal, swing_low_series (SL ke liye)
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    length = len(df)

    is_pivot_high = np.zeros(length, dtype=bool)
    is_pivot_low = np.zeros(length, dtype=bool)
    for i in range(pivot_lookback, length - pivot_lookback):
        window_h = high[i - pivot_lookback:i + pivot_lookback + 1]
        window_l = low[i - pivot_lookback:i + pivot_lookback + 1]
        if high[i] == window_h.max():
            is_pivot_high[i] = True
        if low[i] == window_l.min():
            is_pivot_low[i] = True

    bos_signal = np.zeros(length, dtype=bool)
    choch_signal = np.zeros(length, dtype=bool)
    swing_low_series = np.full(length, np.nan)

    last_pivot_high_val = None
    prev_pivot_high_val = None
    last_pivot_low_val = None
    was_bearish_structure = False   # pichli dafa structure bearish tha?
    structure_bullish = False

    for i in range(length):
        if is_pivot_low[i]:
            last_pivot_low_val = low[i]

        if is_pivot_high[i]:
            val = high[i]
            if prev_pivot_high_val is not None:
                swing_pct = abs(val - prev_pivot_high_val) / prev_pivot_high_val * 100
                if swing_pct >= min_swing_pct:
                    was_bearish_structure = not structure_bullish
                    structure_bullish = val > prev_pivot_high_val
            prev_pivot_high_val = last_pivot_high_val
            last_pivot_high_val = val

        swing_low_series[i] = last_pivot_low_val if last_pivot_low_val is not None else np.nan

        if last_pivot_high_val is not None and structure_bullish:
            fresh_break = close[i] > last_pivot_high_val and (i == 0 or close[i - 1] <= last_pivot_high_val)
            if fresh_break:
                if was_bearish_structure:
                    choch_signal[i] = True   # pehli dafa reversal - CHoCH
                    was_bearish_structure = False
                else:
                    bos_signal[i] = True     # continuation - BOS

    return (
        pd.Series(bos_signal, index=df.index),
        pd.Series(choch_signal, index=df.index),
        pd.Series(swing_low_series, index=df.index),
    )


# ============================================================
# CONFLUENCE SCORE ENGINE
# ============================================================
def compute_confluence(df, btc_daily_df, params, usdt_d_weak=None):
    """
    df: traded coin ka OHLCV (candle-close data)
    btc_daily_df: BTC/USDT ka DAILY OHLCV (BTC trend + 1D RSI overbought check ke liye)
    usdt_d_weak: None (backtest mein) ya bool (live scan mein, CoinGecko se) -
                 agar None ho to is component ka 2 points automatically SKIP
                 ho jate hain (backtest ki fairness ke liye)

    Return: DataFrame with columns: score (0-10), bos, choch, buy_signal, sl, tp1, tp2
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ---- 1. Market Structure (BOS/CHoCH) = 2 points ----
    bos, choch, swing_low = detect_market_structure(
        df, params["pivot_lookback"], params["min_swing_pct"]
    )
    structure_signal = bos | choch
    structure_points = structure_signal.astype(int) * 2

    # ---- 2. RVOL Volume Spike = 2 points ----
    vol_ma = volume.rolling(params["vol_avg_period"]).mean()
    rvol = volume / vol_ma.replace(0, np.nan)
    bullish_candle = close > df["open"]
    rvol_ok = (rvol > params["rvol_threshold"]) & bullish_candle
    rvol_points = rvol_ok.fillna(False).astype(int) * 2

    # ---- 3. BTC Bullish (Daily > EMA50) [+ USDT.D weak agar di gayi ho] = 2 points ----
    btc_ema50 = btc_daily_df["close"].ewm(span=50, adjust=False).mean()
    btc_bullish_daily = (btc_daily_df["close"] > btc_ema50).rename("btc_bullish")
    btc_rsi_daily = compute_rsi(btc_daily_df, 14)
    btc_not_overbought = (btc_rsi_daily <= 70).rename("btc_not_ob")

    btc_merge = pd.merge_asof(
        df[["timestamp"]].sort_values("timestamp"),
        pd.DataFrame({
            "timestamp": btc_daily_df["timestamp"],
            "btc_bullish": btc_bullish_daily,
            "btc_not_ob": btc_not_overbought,
        }).sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    btc_bullish = btc_merge["btc_bullish"].fillna(False).values
    btc_not_ob = btc_merge["btc_not_ob"].fillna(True).values  # default True agar data na mile

    if usdt_d_weak is not None:
        btc_macro_ok = btc_bullish & btc_not_ob & usdt_d_weak
    else:
        btc_macro_ok = btc_bullish & btc_not_ob
    btc_points = pd.Series(btc_macro_ok, index=df.index).astype(int) * 2

    # ---- 4. Price near Demand Zone / EMA50 (aur MAJOR RESISTANCE ke paas NA ho) = 2 points ----
    ema50 = close.ewm(span=50, adjust=False).mean()
    near_ema = (close - ema50).abs() / close <= (params["demand_zone_tolerance_pct"] / 100)
    near_swing_low = (close - swing_low).abs() / close <= (params["demand_zone_tolerance_pct"] / 100)
    near_demand = (near_ema | near_swing_low).fillna(False)

    # Simplified resistance check: recent lookback ki highest high se bohot qareeb na ho
    recent_high = high.rolling(params["resistance_lookback"]).max()
    not_near_resistance = (recent_high - close) / close > (params["resistance_buffer_pct"] / 100)

    demand_zone_ok = near_demand & not_near_resistance.fillna(True)
    demand_points = demand_zone_ok.astype(int) * 2

    # ---- 5. ADX > 20 AND RSI (40-65) = 2 points ----
    adx = compute_adx(df, params["adx_period"])
    rsi = compute_rsi(df, params["rsi_period"])
    trend_momentum_ok = (adx > params["adx_threshold"]) & (rsi >= params["rsi_zone_low"]) & (rsi <= params["rsi_zone_high"])
    momentum_points = trend_momentum_ok.fillna(False).astype(int) * 2

    # ---- TOTAL SCORE ----
    score = structure_points + rvol_points + btc_points + demand_points + momentum_points

    # ---- BUY SIGNAL: fresh structure break + score >= threshold ----
    buy_signal = structure_signal & (score >= params["score_threshold"])

    # ---- SL / TP (1.5x ATR below swing low, min 1:2 RR) ----
    atr = compute_atr(df, params["atr_period"])
    sl = swing_low - params["sl_atr_mult"] * atr
    risk = close - sl
    tp1 = close + risk * params["tp1_rr"]
    tp2 = close + risk * params["tp2_rr"]

    result = pd.DataFrame({
        "score": score,
        "bos": bos,
        "choch": choch,
        "buy_signal": buy_signal,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "adx": adx,
        "rsi": rsi,
        "rvol": rvol,
    }, index=df.index)

    return result


DEFAULT_PARAMS = {
    "pivot_lookback": 5,
    "min_swing_pct": 1.0,
    "vol_avg_period": 20,
    "rvol_threshold": 1.8,
    "demand_zone_tolerance_pct": 2.0,
    "resistance_lookback": 50,
    "resistance_buffer_pct": 3.0,
    "adx_period": 14,
    "adx_threshold": 20,
    "rsi_period": 14,
    "rsi_zone_low": 40,
    "rsi_zone_high": 65,
    "score_threshold": 6,
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "tp1_rr": 2.0,
    "tp2_rr": 3.0,
}