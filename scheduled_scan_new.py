"""
Scheduled Scan - NEW ADVANCED CONFLUENCE (Alag/Independent)

Ye purane system (scheduled_scan.py) se BILKUL ALAG hai - usay chhua
nahi gaya, wo apni jagah waisa hi chalta rahega. Ye sirf NEW
AdvancedConfluence_v1 (Score>=6, BOS+CHoCH, Chandelier Exit) ke liye
hai, apni alag notification ke sath (alag ntfy topic taake purane
signals se mix na ho).

GitHub Actions ke zariye har 1 ghanta khud chalti hai.

Result 'latest_signals_new.json' mein save hota hai (ALAG file naam,
purani 'latest_signals.json' se mix nahi hoga).
"""

import json
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown
from backtest_engine import compute_atr
from confluence_engine import compute_confluence, DEFAULT_PARAMS

TOP_N_COINS = 400
TIMEFRAME = "1h"
LOOKBACK_BARS = 3
RR_MULTIPLE = 2.0
CE_MULT = 3.0
CE_PERIOD = 16

# ALAG ntfy topic - purane system ke notifications se mix nahi honge
# ⚠️ Apna khud ka unique naam banayen (ntfy app mein subscribe karein)
NTFY_TOPIC_NEW = "asifali549-advanced-confluence-7k2m9x"


def send_notification(title, message):
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC_NEW}",
            data=message.encode("utf-8"),
            headers={
                "Title": title.encode("utf-8"),
                "Priority": "high",
                "Tags": "chart_with_upwards_trend",
            },
            timeout=10,
        )
    except Exception as e:
        print(f"  [NOTIFY FAILED] {e}")


def to_pkt_str(ts):
    ts_utc = pd.Timestamp(ts)
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    return ts_utc.tz_convert("Asia/Karachi").strftime("%Y-%m-%d %I:%M %p PKT")


def load_notified_keys():
    if not os.path.exists("notified_signals_new.json"):
        return {}
    with open("notified_signals_new.json") as f:
        return json.load(f)


def save_notified_keys(keys_dict):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    pruned = {k: v for k, v in keys_dict.items() if datetime.fromisoformat(v) > cutoff}
    with open("notified_signals_new.json", "w") as f:
        json.dump(pruned, f, indent=2)


def scan_symbol(exchange, symbol, btc_daily):
    df = fetch_ohlcv(exchange, symbol, TIMEFRAME, limit=max(config.CANDLE_LIMITS.get(TIMEFRAME, 500), 300))
    if df is None or len(df) < 220:
        return []

    try:
        result = compute_confluence(df, btc_daily, DEFAULT_PARAMS, usdt_d_weak=None)
    except Exception as e:
        print(f"  [ERROR] {symbol}: {e}")
        return []

    structure_signal = result["bos"] | result["choch"]
    sig = structure_signal & (result["score"] >= 6)
    sig = apply_cooldown(sig, config.SIGNAL_COOLDOWN_BARS)

    recent = sig.tail(LOOKBACK_BARS)
    if not recent.any():
        return []

    signal_idx = recent[recent].index[-1]
    signal_bar = df.loc[signal_idx]
    bars_ago = len(df) - 1 - signal_idx
    entry_price = signal_bar["close"]
    current_price = df["close"].iloc[-1]

    atr = compute_atr(df, CE_PERIOD)
    highest_high = df["high"].rolling(CE_PERIOD).max()
    chandelier = highest_high - CE_MULT * atr
    initial_stop = chandelier.loc[signal_idx]
    if pd.isna(initial_stop):
        return []

    risk = entry_price - initial_stop
    tp_price = entry_price + risk * RR_MULTIPLE
    trail_stop = chandelier.loc[signal_idx:].max()
    pnl_pct = (current_price - entry_price) / entry_price * 100
    structure_type = "CHoCH" if result.loc[signal_idx, "choch"] else "BOS"

    return [{
        "Coin": symbol,
        "Timeframe": TIMEFRAME,
        "Combo": "NEW_AdvancedConfluence_v1",
        "Structure": structure_type,
        "Score": f"{int(result.loc[signal_idx, 'score'])}/10",
        "Signal Bar": to_pkt_str(signal_bar["timestamp"]),
        "Signal Bar UTC": str(signal_bar["timestamp"]),
        "Bars Ago": int(bars_ago),
        "Entry": round(float(entry_price), 6),
        "Current": round(float(current_price), 6),
        "P/L %": round(float(pnl_pct), 2),
        "Trail Stop": round(float(trail_stop), 6),
        "Take Profit": round(float(tp_price), 6),
        "Status": "OPEN",
    }]


def main():
    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"[NEW SYSTEM] Scanning {len(coins)} coins on {TIMEFRAME}...")

    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=800)

    all_results = []
    for symbol in coins:
        try:
            all_results.extend(scan_symbol(exchange, symbol, btc_daily))
        except Exception as e:
            print(f"  [SKIP] {symbol}: {e}")

    output = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "coins_scanned": len(coins),
        "timeframe": TIMEFRAME,
        "signals": all_results,
    }

    with open("latest_signals_new.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[NEW SYSTEM] Done. {len(all_results)} fresh signals saved to latest_signals_new.json")

    notified = load_notified_keys()
    new_count = 0
    for sig in all_results:
        key = f"{sig['Coin']}|{sig['Combo']}|{sig['Signal Bar UTC']}"
        if key in notified:
            continue
        title = f"🎯 [NEW] {sig['Coin']} - Score {sig['Score']}"
        message = (
            f"Structure: {sig['Structure']}\n"
            f"Signal Candle: {sig['Signal Bar']}\n"
            f"Entry: {sig['Entry']}\n"
            f"Take Profit: {sig['Take Profit']}\n"
            f"Trail Stop: {sig['Trail Stop']}"
        )
        send_notification(title, message)
        notified[key] = datetime.now(timezone.utc).isoformat()
        new_count += 1

    save_notified_keys(notified)
    print(f"[NEW SYSTEM] Notifications bheji: {new_count} naye signals")


if __name__ == "__main__":
    main()