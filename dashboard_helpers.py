"""
Dashboard Helpers - Live (scheduled) aur Manual (interactive) dono
dashboards yehi functions istemal karte hain, taake dono mein
HAMESHA same logic rahe (koi mismatch na ho).
"""

import requests
import pandas as pd

import config
from data_fetcher import fetch_ohlcv
from backtest_engine import simulate_trades

ALL_TIMEFRAMES = ["15m", "1h", "4h", "1d"]

WHALE_TRACKED_TOKENS = {
    "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec",
    "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "LINK": "0x514910771af9ca656af840dff83e8264ecf986ca",
    "UNI": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
}


def get_fear_greed():
    try:
        resp = requests.get("https://api.alternative.me/fng/", timeout=10)
        item = resp.json()["data"][0]
        return int(item["value"]), item["value_classification"]
    except Exception:
        return None, None


def get_futures_exchange():
    import ccxt
    return ccxt.kucoinfutures({"enableRateLimit": True})

