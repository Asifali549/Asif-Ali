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
from datetime import datetime, timezone

import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import compute_atr
from dashboard_helpers import (
    get_fear_greed, get_futures_exchange, get_funding_and_oi, get_orderbook_info,
    get_tf_volume_change, get_24h_range_distance, get_historical_performance,
    get_btc_correlation, ALL_TIMEFRAMES, compute_overall_score,
)

TOP_N_COINS = 400
SIGNAL_TIMEFRAME = "1h"
CE_A = {"period": 16, "multiplier": 3.0}
CE_B = {"period": 12, "multiplier": 3.0}
RR_MULTIPLE = 2.0

COLORABLE_COLUMNS = (
    ["OrderBook Bid/Ask", "Liquidity Compare", "Coin's Own PF", "Funding Rate", "Dist from 24h High"]
    + [f"Vol {tf}" for tf in ALL_TIMEFRAMES]