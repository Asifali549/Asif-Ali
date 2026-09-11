"""
Live + Manual Colorful Dashboard - LIVE section khud-b-khud (GitHub
Actions se) update hoti hai, MANUAL section button se on-demand
chalti hai. Dono mein Overall Score, Rank (best=1), aur Compact
View available hai.

Chalayen: streamlit run live_colorful_dashboard.py
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import compute_atr
from dashboard_helpers import (
    get_fear_greed, get_futures_exchange, get_funding_and_oi, get_long_short_ratio,
    get_orderbook_info, get_tf_volume_change, get_24h_range_distance,
    get_historical_performance, get_btc_correlation, get_whale_activity,
    color_value, compute_overall_score, ALL_TIMEFRAMES,
)

st.set_page_config(page_title="Live Colorful Dashboard", layout="wide")
st.title("🎨 Live + Manual Colorful Dashboard")
st.warning(
    "⚠️ Sirf MALOOMAT — asal Union AB signal logic bilkul nahi badla "
    "gaya. 🟢 Green = signal ke HAQ mein, 🔴 Red = KHILAF."
)

CE_A = {"period": 16, "multiplier": 3.0}
CE_B = {"period": 12, "multiplier": 3.0}
RR_MULTIPLE = 2.0

COLORABLE_COLUMNS = (
    ["OrderBook Bid/Ask", "Liquidity Compare", "Coin's Own PF", "Funding Rate", "Dist from 24h High"]
    + [f"Vol {tf}" for tf in ALL_TIMEFRAMES]
    + [f"Chg {tf}" for tf in ALL_TIMEFRAMES]
)

COMPACT_COLUMNS = ["Rank", "Coin", "Combo", "Overall Score %", "Verdict", "Entry", "Trail Stop", "Take Profit"]


def style_and_show(df, compact):
    if compact:
        show_cols = [c for c in COMPACT_COLUMNS if c in df.columns]
        df = df[show_cols]

    def apply_row_colors(row):
        return [color_value(col, row[col]) if col in df.columns else "" for col in df.columns]

    st.dataframe(df.style.apply(apply_row_colors, axis=1), use_container_width=True, hide_index=True)


# ============================================================
# SECTION 1: LIVE (khud-b-khud, GitHub Actions se)
# ============================================================
st.header("🔴 LIVE — Auto-Updated (har 1 ghanta)")
compact_live = st.checkbox("Compact View (sirf zaroori columns)", value=True, key="compact_live")

if os.path.exists("dashboard_signals.json"):
    with open("dashboard_signals.json") as f:
        live_data = json.load(f)

    last_updated = datetime.fromisoformat(live_data["last_updated_utc"])
    age_minutes = (datetime.now(timezone.utc) - last_updated).total_seconds() / 60

    col1, col2, col3 = st.columns(3)
    col1.metric("Last Updated", f"{age_minutes:.0f} min pehle")
    col2.metric("Coins Scanned", live_data["coins_scanned"])
    col3.metric("Signals", len(live_data["signals"]))

    if live_data.get("fear_greed_value") is not None:
        st.markdown(f"### 📊 BTC Fear & Greed: **{live_data['fear_greed_value']}/100** ({live_data['fear_greed_label']})")

    if age_minutes > 90:
        st.warning("⚠️ Ye data 90 minute se purana hai — background scan delay ho sakta hai.")

    if live_data["signals"]:
        df_live = pd.DataFrame(live_data["signals"])
        style_and_show(df_live, compact_live)
    else:
        st.info("Is waqt koi fresh signal nahi (last scan mein).")
else:
    st.info(
        "Live scan abhi setup nahi hua ya pehli baar chalne ka wait ho raha hai. "
        "GitHub repo mein '.github/workflows/scan_dashboard.yml' hona chahiye — "
        "1 ghante mein pehla result aa jayega."
    )

st.markdown("---")


# ============================================================
# SECTION 2: MANUAL (on-demand, apni marzi ke toggles ke sath)
# ============================================================
st.header("🔍 Manual Scan (apni marzi ke toggles)")

st.sidebar.header("Manual Scan Settings")
show_funding = st.sidebar.checkbox("Funding Rate", value=True)
show_oi = st.sidebar.checkbox("Open Interest", value=True)
show_liquidity = st.sidebar.checkbox("Liquidity Up/Down", value=True)
show_orderbook_ratio = st.sidebar.checkbox("Order Book Ratio", value=True)
show_24h_range = st.sidebar.checkbox("24h Range", value=True)
show_history = st.sidebar.checkbox("Coin's Own Performance", value=True)
show_btc_corr = st.sidebar.checkbox("BTC Correlation", value=False)
show_long_short = st.sidebar.checkbox("Long/Short Ratio", value=False)
show_whale = st.sidebar.checkbox("Whale Transfers", value=False)
etherscan_key = st.sidebar.text_input("Etherscan API Key", type="password") if show_whale else ""

st.sidebar.markdown("---")
st.sidebar.subheader("Volume - Har Timeframe")
vol_tf_toggles = {tf: st.sidebar.checkbox(f"Vol {tf}", value=True, key=f"v{tf}") for tf in ALL_TIMEFRAMES}
st.sidebar.subheader("Price Change - Har Timeframe")
chg_tf_toggles = {tf: st.sidebar.checkbox(f"Chg {tf}", value=True, key=f"c{tf}") for tf in ALL_TIMEFRAMES}

st.sidebar.markdown("---")
coin_source = st.sidebar.radio("Coin Kaise Chunein", ["Auto Scan", "Manual Paste"])
if coin_source.startswith("Manual"):
    manual_coins_text = st.sidebar.text_area("Coins (comma-separated)", height=80)
else:
    n_coins = st.sidebar.slider("Kitne coins", 10, 400, 20)
signal_timeframe = st.sidebar.selectbox("Signal Timeframe", ["15m", "1h", "4h"], index=1)

compact_manual = st.checkbox("Compact View (sirf zaroori columns)", value=True, key="compact_manual")

if st.button("🎨 Manual Scan Chalayen", type="primary"):
    exchange = get_exchange()
    futures_exchange = get_futures_exchange() if (show_funding or show_oi or show_long_short) else None

    if coin_source.startswith("Manual"):
        coins = [c.strip() for c in manual_coins_text.split(",") if c.strip()]
        if not coins:
            st.error("Koi coin nahi likha gaya.")
            st.stop()
    else: 