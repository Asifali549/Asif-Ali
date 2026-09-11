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