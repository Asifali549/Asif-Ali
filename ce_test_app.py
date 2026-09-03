"""
CE Buy-Only Test App - Alag, chhoti test app (asal screener_app.py ko
CHHUA nahi gaya, koi risk nahi). Sirf CE Buy-Only strategy ko live
test karne ke liye.

⚠️ Ye strategy BTC par backtest mein PF 0.67 (nuksan) de chuki hai -
sirf tajurbe ke liye hai, trading decisions is par na karein.
"""

import numpy as np
import pandas as pd
import streamlit as st

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv

st.set_page_config(page_title="CE Buy-Only Test", layout="wide")
st.title("🕯️ CE Buy-Only Test App (Alag/Experimental)")
st.warning(
    "⚠️ Ye strategy BTC par backtest mein PF 0.67 (nuksan) de chuki hai — "
    "asal Confluence Screener se ALAG hai, isay chhua nahi gaya. Sirf "
    "tajurbe ke liye, trading decisions is par na karein."
)

CE_PARAMS = {
    "atr_period": 12,
    "atr_mult": 3.0,
    "vol_ma_len": 20,
    "vol_multiplier": 1.0,
    "buy_pressure_ratio": 0.6,
    "btc_ema_len": 50,
}


def ce_buy_only(df, btc_df, params):
    atr_period = params["atr_period"]
    atr_mult = params["atr_mult"]

    close = df["close"].values
    length = len(df)

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean().values

    highest_high = df["close"].rolling(atr_period).max().values
    lowest_low = df["close"].rolling(atr_period).min().values

    long_stop = highest_high - atr * atr_mult
    short_stop = lowest_low + atr * atr_mult

    long_stop_prev = np.full(length, np.nan)
    short_stop_prev = np.full(length, np.nan)
    dir_arr = np.ones(length, dtype=int)

    for i in range(length):
        ls_prev = long_stop_prev[i - 1] if i > 0 and not np.isnan(long_stop_prev[i - 1]) else long_stop[i]
        ss_prev = short_stop_prev[i - 1] if i > 0 and not np.isnan(short_stop_prev[i - 1]) else short_stop[i]

        ls = max(long_stop[i], ls_prev) if i > 0 and close[i - 1] > ls_prev else long_stop[i]
        ss = min(short_stop[i], ss_prev) if i > 0 and close[i - 1] < ss_prev else short_stop[i]

        prev_dir = dir_arr[i - 1] if i > 0 else 1
        if close[i] > ss_prev:
            dir_arr[i] = 1
        elif close[i] < ls_prev:
            dir_arr[i] = -1
        else:
            dir_arr[i] = prev_dir

        long_stop_prev[i] = ls
        short_stop_prev[i] = ss

    buy_signal_raw = np.zeros(length, dtype=bool)
    for i in range(1, length):
        if dir_arr[i] == 1 and dir_arr[i - 1] == -1:
            buy_signal_raw[i] = True

    vol_ma = df["volume"].rolling(params["vol_ma_len"]).mean()
    vol_ok = df["volume"] > vol_ma * params["vol_multiplier"]
    close_position = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    upside_vol_ok = (close_position > params["buy_pressure_ratio"]) & vol_ok

    btc_ema = btc_df["close"].ewm(span=params["btc_ema_len"], adjust=False).mean()
    btc_ok_series = (btc_df["close"] > btc_ema).rename("btc_ok")
    merged = pd.merge_asof(
        df[["timestamp"]].sort_values("timestamp"),
        pd.DataFrame({"timestamp": btc_df["timestamp"], "btc_ok": btc_ok_series}).sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    btc_ok = merged["btc_ok"].fillna(False).values

    confirmed_buy = buy_signal_raw & upside_vol_ok.fillna(False).values & btc_ok
    return pd.Series(confirmed_buy, index=df.index)


col1, col2 = st.columns(2)
timeframe = col1.selectbox("Timeframe", ["15m", "1h", "4h"], index=1)
n_coins = col2.slider("Kitne coins scan karein", 20, 200, 100)

if st.button("🔍 Scan Chalayen", type="primary"):
    exchange = get_exchange()
    try:
        with st.spinner("BTC ka trend data le rahe hain..."):
            btc_df = fetch_ohlcv(exchange, "BTC/USDT", timeframe, limit=100)
        with st.spinner("Coin list le rahe hain..."):
            coins = get_coin_list(exchange)[:n_coins]
    except Exception as e:
        st.error(f"Exchange se connect nahi ho paya: {e}")
        st.stop()

    results = []
    progress = st.progress(0.0)
    for i, symbol in enumerate(coins):
        try:
            df = fetch_ohlcv(exchange, symbol, timeframe, limit=100)
        except Exception:
            df = None
        if df is not None and len(df) >= 60:
            try:
                sig = ce_buy_only(df, btc_df, CE_PARAMS)
                if sig.tail(3).any():
                    signal_idx = sig.tail(3)[sig.tail(3)].index[-1]
                    results.append({
                        "Coin": symbol,
                        "Bars Ago": len(df) - 1 - signal_idx,
                        "Signal Price": round(df.loc[signal_idx, "close"], 6),
                        "Current Price": round(df["close"].iloc[-1], 6),
                    })
            except Exception:
                pass
        progress.progress((i + 1) / len(coins))
    progress.empty()

    if results:
        df_r = pd.DataFrame(results).sort_values("Bars Ago")
        st.success(f"✅ {len(df_r)} coins mein signal mila")
        st.dataframe(df_r, use_container_width=True, hide_index=True)
    else:
        st.info("Is waqt koi fresh signal nahi mila.")