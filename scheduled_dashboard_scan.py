"""
Scheduled Dashboard Scan - GitHub Actions ke zariye har 1 ghanta khud
chalti hai. Union AB signals dhoond kar, HAR signal ke liye MUKAMMAL
context (funding, OI, liquidity, multi-TF, 24h range, history, BTC
correlation) nikalti hai, Overall Score/Rank laga kar
'dashboard_signals.json' mein save karti hai.

Live section (Streamlit app mein) isi file ko seedha padh leta hai -
koi manual scan chalane ki zaroorat nahi.
"""

import json
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import compute_atr
from dashboard_helpers import (
    get_fear_greed, get_futures_exchange, get_funding_and_oi, get_orderbook_info,
    get_tf_volume_change, get_24h_range_distance, get_historical_performance,
    get_btc_correlation, ALL_TIMEFRAMES, compute_overall_score,
)

NTFY_TOPIC_STRONG = "asifali549-strong-signals-9k3m7x"


def send_strong_notification(title, message):
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC_STRONG}",