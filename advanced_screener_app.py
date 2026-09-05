"""
Advanced Confluence Screener - Live Test App (standalone, koi risk nahi
purani files ko)

Chalayen: streamlit run advanced_screener_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import requests

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from confluence_engine import compute_confluence, DEFAULT_PARAMS

st.set_page_config(page_title="Advanced Confluence Screener", layout="wide")
st.title("🎯 Advanced Anti-Fakeout Confluence Screener")
st.caption(
    "10-Point Confluence Scoring: Market Structure(2) + RVOL(2) + BTC/USDT.D Macro(2) "
    "+ Demand Zone(2) + ADX/RSI(2). Real BTC par backtest: PF 1.97, Win% 57.5 "
    "(chhota 40-trade sample - 400-coin confirm abhi baaki)."
)


def get_usdt_dominance_trend():
    """CoinGecko se LIVE USDT dominance - sirf snapshot, trend nahi (backtest mein shamil nahi)."""
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        data = resp.json()
        usdt_pct = data["data"]["market_cap_percentage"].get("usdt", None)
        return usdt_pct
    except Exception:
        return None


col1, col2, col3 = st.columns(3)
timeframe = col1.selectbox("Timeframe", ["1h", "4h", "1d"], index=0)
n_coins = col2.slider("Kitne coins scan karein", 20, 200, 100)
score_threshold = col3.slider("Minimum Score", 2, 10, 6, step=2)

usdt_dominance = get_usdt_dominance_trend()
if usdt_dominance is not None:
    st.info(f"📊 USDT Dominance (abhi): {usdt_dominance:.2f}% (sirf snapshot hai, historical trend Python mein nahi mil sakta)")

if st.button("🔍 Scan Chalayen", type="primary"):
    exchange = get_exchange()
    try:
        with st.spinner("BTC daily data le rahe hain..."):
            btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=200)
        with st.spinner("Coin list le rahe hain..."):
            coins = get_coin_list(exchange)[:n_coins]
    except Exception as e:
        st.error(f"Exchange se connect nahi ho paya: {e}")
        st.stop()

    params = dict(DEFAULT_PARAMS)
    params["score_threshold"] = score_threshold

    results = []
    progress = st.progress(0.0)
    for i, symbol in enumerate(coins):
        try:
            df = fetch_ohlcv(exchange, symbol, timeframe, limit=250)
        except Exception:
            df = None
        if df is not None and len(df) >= 220:
            try:
                res = compute_confluence(df, btc_daily, params, usdt_d_weak=None)
                if res["buy_signal"].iloc[-3:].any():
                    idx = res.iloc[-3:][res.iloc[-3:]["buy_signal"]].index[-1]
                    structure_type = "CHoCH" if res.loc[idx, "choch"] else "BOS"
                    results.append({
                        "Coin": symbol,
                        "Timeframe": timeframe,
                        "Score": f"{int(res.loc[idx,'score'])}/10",
                        "Structure": structure_type,
                        "Entry": round(df.loc[idx, "close"], 6),
                        "SL": round(res.loc[idx, "sl"], 6),
                        "TP1": round(res.loc[idx, "tp1"], 6),
                        "TP2": round(res.loc[idx, "tp2"], 6),
                        "Bars Ago": len(df) - 1 - idx,
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
        st.info("Is waqt koi fresh signal nahi mila. Score threshold kam kar ke dekhein.")