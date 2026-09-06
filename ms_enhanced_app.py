"""
Market Structure ENHANCED - Live Test App (standalone, purani files
ko chhua nahi gaya, koi risk nahi)

⚠️ Ye ek NAYA, sirf BTC par test (12 trades, PF 3.53) hua idea hai -
400-coin confirm abhi baaki hai. Trading decisions is par na karein
jab tak bara sample confirm na ho.

Chalayen: streamlit run ms_enhanced_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import compute_atr

st.set_page_config(page_title="MS Enhanced Screener", layout="wide")
st.title("🎯 Market Structure Enhanced (Resistance + BTC Macro)")
st.warning(
    "⚠️ Ye NAYA, sirf BTC par test (12 trades, PF 3.53) hua idea hai — "
    "chhota sample. 400-coin confirm hone tak trading decisions is par "
    "na karein, sirf tajurbe ke liye istemal karein."
)

ENHANCED_PARAMS = {
    "swing_lookback": 20,
    "resistance_lookback": 300,
    "resistance_exclude_recent": 20,
    "atr_period": 14,
    "min_rr_to_resistance": 1.5,
}
CE_PERIOD = 16
CE_MULT = 3.0
RR_MULTIPLE = 2.0


def market_structure_enhanced(df, btc_daily, ms_params, enh_params):
    base_signal = STRATEGY_FUNCTIONS["market_structure"](df, ms_params)

    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(enh_params["atr_period"]).mean()

    swing_low = low.rolling(enh_params["swing_lookback"]).min()
    swing_high = high.shift(enh_params["resistance_exclude_recent"]).rolling(enh_params["resistance_lookback"]).max()

    risk = close - (swing_low - 0.5 * atr)
    upside_to_resistance = swing_high - close
    resistance_ok = upside_to_resistance >= enh_params["min_rr_to_resistance"] * risk

    btc_ema50 = btc_daily["close"].ewm(span=50, adjust=False).mean()
    btc_bullish = (btc_daily["close"] > btc_ema50).rename("btc_bullish")
    merged = pd.merge_asof(
        df[["timestamp"]].sort_values("timestamp"),
        pd.DataFrame({"timestamp": btc_daily["timestamp"], "btc_bullish": btc_bullish}).sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    btc_ok = merged["btc_bullish"].fillna(False).values

    return base_signal & resistance_ok.fillna(False) & btc_ok


col1, col2 = st.columns(2)
timeframe = col1.selectbox("Timeframe", ["15m", "1h", "4h"], index=1)
n_coins = col2.slider("Kitne coins scan karein", 20, 200, 100)

if st.button("🔍 Scan Chalayen", type="primary"):
    exchange = get_exchange()
    try:
        with st.spinner("BTC daily data le rahe hain..."):
            btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=400)
        with st.spinner("Coin list le rahe hain..."):
            coins = get_coin_list(exchange)[:n_coins]
    except Exception as e:
        st.error(f"Exchange se connect nahi ho paya: {e}")
        st.stop()

    ms_params = config.STRATEGY_PARAMS["market_structure"]

    results = []
    progress = st.progress(0.0)
    for i, symbol in enumerate(coins):
        try:
            df = fetch_ohlcv(exchange, symbol, timeframe, limit=max(config.CANDLE_LIMITS.get(timeframe, 500), 350))
        except Exception:
            df = None
        if df is not None and len(df) >= 320:
            try:
                sig = market_structure_enhanced(df, btc_daily, ms_params, ENHANCED_PARAMS)
                sig = apply_cooldown(sig, config.SIGNAL_COOLDOWN_BARS)
                if sig.tail(3).any():
                    idx = sig.tail(3)[sig.tail(3)].index[-1]

                    atr = compute_atr(df, CE_PERIOD)
                    highest_high = df["high"].rolling(CE_PERIOD).max()
                    chandelier = (highest_high - CE_MULT * atr).loc[idx]
                    entry_price = df.loc[idx, "close"]
                    current_price = df["close"].iloc[-1]
                    risk = entry_price - chandelier
                    tp_price = entry_price + risk * RR_MULTIPLE

                    results.append({
                        "Coin": symbol,
                        "Bars Ago": len(df) - 1 - idx,
                        "Entry": round(entry_price, 6),
                        "Current": round(current_price, 6),
                        "Trail Stop": round(chandelier, 6),
                        "Take Profit": round(tp_price, 6),
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