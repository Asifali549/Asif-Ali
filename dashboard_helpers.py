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


def get_funding_and_oi(futures_exchange, symbol):
    base = symbol.split("/")[0]
    futures_symbol = f"{base}/USDT:USDT"
    funding, oi = None, None
    try:
        funding = futures_exchange.fetch_funding_rate(futures_symbol).get("fundingRate")
    except Exception:
        pass
    try:
        oi_info = futures_exchange.fetch_open_interest(futures_symbol)
        oi = oi_info.get("openInterestValue") or oi_info.get("openInterestAmount")
    except Exception:
        pass
    return funding, oi


def get_long_short_ratio(futures_exchange, symbol):
    try:
        base = symbol.split("/")[0]
        futures_symbol = f"{base}/USDT:USDT"
        if hasattr(futures_exchange, "fetch_long_short_ratio"):
            return futures_exchange.fetch_long_short_ratio(futures_symbol).get("longShortRatio")
        return None
    except Exception:
        return None


def get_orderbook_info(exchange, symbol, depth=20):
    try:
        ob = exchange.fetch_order_book(symbol, limit=depth)
        bids = ob.get("bids", [])[:depth]
        asks = ob.get("asks", [])[:depth]
        if not bids or not asks:
            return None, None, None
        bid_vol = sum(p * q for p, q in bids)
        ask_vol = sum(p * q for p, q in asks)
        ratio = bid_vol / ask_vol if ask_vol > 0 else None
        return bid_vol, ask_vol, ratio
    except Exception:
        return None, None, None


def get_tf_volume_change(exchange, symbol, tf):
    try:
        df_tf = fetch_ohlcv(exchange, symbol, tf, limit=25)
        if df_tf is None or len(df_tf) < 21:
            return None, None
        vol_ma = df_tf["volume"].iloc[:-1].tail(20).mean()
        vol_ratio = df_tf["volume"].iloc[-1] / vol_ma if vol_ma > 0 else None
        o, c = df_tf["open"].iloc[-1], df_tf["close"].iloc[-1]
        price_change_pct = (c - o) / o * 100 if o > 0 else None
        return vol_ratio, price_change_pct
    except Exception:
        return None, None


def get_24h_range_distance(exchange, symbol):
    try:
        df_1d = fetch_ohlcv(exchange, symbol, "1d", limit=2)
        if df_1d is None or len(df_1d) < 1:
            return None, None
        last = df_1d.iloc[-1]
        current = last["close"]
        return round((last["high"] - current) / current * 100, 2), round((current - last["low"]) / current * 100, 2)
    except Exception:
        return None, None


def get_historical_performance(df, combo_sig, ce):
    try:
        trades = simulate_trades(df, combo_sig, config.BACKTEST_PARAMS, ce)
        if not trades:
            return None, None
        trades_df = pd.DataFrame(trades)
        wins = trades_df[trades_df["return_pct"] > 0]
        losses = trades_df[trades_df["return_pct"] <= 0]
        win_rate = len(wins) / len(trades_df) * 100
        pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
        return round(win_rate, 1), round(pf, 2) if pf else None
    except Exception:
        return None, None


def get_btc_correlation(exchange, symbol, timeframe):
    try:
        df_coin = fetch_ohlcv(exchange, symbol, timeframe, limit=100)
        df_btc = fetch_ohlcv(exchange, "BTC/USDT", timeframe, limit=100)
        if df_coin is None or df_btc is None:
            return None
        merged = pd.merge(df_coin[["timestamp", "close"]], df_btc[["timestamp", "close"]], on="timestamp", suffixes=("_c", "_b"))
        if len(merged) < 20:
            return None
        corr = merged["close_c"].pct_change().dropna().corr(merged["close_b"].pct_change().dropna())
        return round(corr, 2) if corr is not None else None
    except Exception:
        return None


def get_whale_activity(symbol, api_key):
    base = symbol.split("/")[0]
    if base not in WHALE_TRACKED_TOKENS or not api_key:
        return "N/A"
    try:
        resp = requests.get("https://api.etherscan.io/api", params={
            "module": "account", "action": "tokentx", "contractaddress": WHALE_TRACKED_TOKENS[base],
            "page": 1, "offset": 5, "sort": "desc", "apikey": api_key,
        }, timeout=10)
        data = resp.json()
        if data.get("status") == "1" and data.get("result"):
            return f"{len(data['result'])} transfers"
        return "Koi data nahi"
    except Exception:
        return "N/A"


# ============================================================
# COLOR + SCORE LOGIC (shared - LIVE aur MANUAL dono yehi istemal karein)
# ============================================================
def classify_color(col_name, val):
    """Return 'green', 'red', ya 'neutral'."""
    if val in (None, "N/A") or (isinstance(val, str) and val.startswith("N/A")):
        return "neutral"
    try:
        if "OrderBook Bid/Ask" in col_name:
            num = float(str(val).replace("x", ""))
            return "green" if num > 1.0 else "red"
        if col_name == "Liquidity Compare":
            return "green" if val == "Support > Resistance" else "red"
        if col_name.startswith("Chg "):
            num = float(str(val).replace("%", "").replace("+", ""))
            return "green" if num > 0 else "red" if num < 0 else "neutral"
        if col_name.startswith("Vol "):
            num = float(str(val).replace("x", ""))
            return "green" if num >= 1.2 else "red" if num < 0.8 else "neutral"
        if col_name == "Coin's Own PF":
            num = float(val)
            return "green" if num > 1 else "red"
        if col_name == "Funding Rate":
            num = float(str(val).replace("%", ""))
            return "red" if num > 0.05 else "green" if num <= 0.01 else "neutral"
        if col_name == "Dist from 24h High":
            num = float(str(val).replace("%", ""))
            return "red" if num < 1.0 else "green" if num > 3.0 else "neutral"
    except (ValueError, TypeError):
        return "neutral"
    return "neutral"


def color_value(col_name, val):
    c = classify_color(col_name, val)
    if c == "green":
        return "background-color:#1a4d2e;color:white"
    if c == "red":
        return "background-color:#4d1a1a;color:white"
    return ""


def compute_overall_score(final_row, colorable_columns):
    """Row ke sab colorable columns dekh kar green/red gin kar ek
    overall score (0-100%) aur verdict nikalta hai."""
    green, red = 0, 0
    for col in colorable_columns:
        if col not in final_row:
            continue
        c = classify_color(col, final_row[col])
        if c == "green":
            green += 1
        elif c == "red":
            red += 1
    total = green + red
    score_pct = round(green / total * 100, 1) if total > 0 else 50.0
    if score_pct >= 70:
        verdict = "🟢🟢🟢 Strong"
    elif score_pct >= 50:
        verdict = "🟢 Good"
    elif score_pct >= 30:
        verdict = "🟡 Mixed"
    else:
        verdict = "🔴 Weak"
    return score_pct, verdict