"""
Scheduled Scan - Ye script GitHub Actions ke zariye background mein
har 15 minute baad khud chalti hai (PC/mobile band ho tab bhi).
Sirf top 100 coins, sirf 1h timeframe (backtest mein sab se behtareen
sabit hua) - taake 15 minute ke andar poora ho jaye.

Result 'latest_signals.json' mein save hota hai, jise screener_app.py
seedha padh kar FORAN dikha deta hai - koi wait nahi karna parta.

NAYA signal milte hi ntfy.sh ke zariye mobile par push notification
bhi jati hai (sirf NAYE signals par - purane repeat nahi hote,
'notified_signals.json' mein track hota hai kya already bhej chuke hain).
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

TOP_N_COINS = 400
TIMEFRAME = "1h"   # backtest mein sab se behtareen (Ichimoku+MS PF 4.46, EMA+Breakout PF 2.41)
LOOKBACK_BARS = 3
RR_MULTIPLE = 2.0
CE_MULT = 3.0

# ntfy.sh topic - isay apna unique naam dein (jo aap ne app mein subscribe kiya)
NTFY_TOPIC = "asifali549-crypto-alerts-8x2m9k"

COMBOS = [
    ("ichimoku", "market_structure", "Ichimoku+MarketStructure", 16),
    ("ema_crossover", "breakout", "EMA+Breakout", 12),
]


def send_notification(title, message):
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
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
    """
    Exchange ka timestamp UTC hota hai. Isay Pakistan Time (UTC+5) mein
    convert kar ke insani-parhne-laiq banata hai.
    """
    ts_utc = pd.Timestamp(ts)
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    ts_pkt = ts_utc.tz_convert("Asia/Karachi")
    return ts_pkt.strftime("%Y-%m-%d %I:%M %p PKT")


def load_notified_keys():
    if not os.path.exists("notified_signals.json"):
        return {}
    with open("notified_signals.json") as f:
        return json.load(f)


def save_notified_keys(keys_dict):
    # 24 ghante se purane entries hata dein (file chhoti rahe)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    pruned = {
        k: v for k, v in keys_dict.items()
        if datetime.fromisoformat(v) > cutoff
    }
    with open("notified_signals.json", "w") as f:
        json.dump(pruned, f, indent=2)


def scan_symbol(exchange, symbol):
    df = fetch_ohlcv(exchange, symbol, TIMEFRAME, limit=max(config.CANDLE_LIMITS.get(TIMEFRAME, 500), 300))
    if df is None or len(df) < 220:
        return []

    found = []
    for strat_a, strat_b, combo_name, ce_period in COMBOS:
        atr = compute_atr(df, ce_period)
        highest_high = df["high"].rolling(ce_period).max()
        chandelier = highest_high - CE_MULT * atr

        sig_a = STRATEGY_FUNCTIONS[strat_a](df, config.STRATEGY_PARAMS[strat_a])
        sig_b = STRATEGY_FUNCTIONS[strat_b](df, config.STRATEGY_PARAMS[strat_b])
        sig_a = apply_cooldown(sig_a, config.SIGNAL_COOLDOWN_BARS)
        sig_b = apply_cooldown(sig_b, config.SIGNAL_COOLDOWN_BARS)

        combined = sig_a & sig_b
        recent = combined.tail(LOOKBACK_BARS)
        if not recent.any():
            continue

        signal_idx = recent[recent].index[-1]
        signal_bar = df.loc[signal_idx]
        bars_ago = len(df) - 1 - signal_idx

        entry_price = signal_bar["close"]
        current_price = df["close"].iloc[-1]

        initial_stop = chandelier.loc[signal_idx]
        if pd.isna(initial_stop):
            continue

        risk = entry_price - initial_stop
        tp_price = entry_price + risk * RR_MULTIPLE
        trail_stop = chandelier.loc[signal_idx:].max()
        pnl_pct = (current_price - entry_price) / entry_price * 100

        status = "OPEN"
        if df["low"].iloc[-1] <= trail_stop:
            status = "STOPPED (trail)"
        elif df["high"].iloc[-1] >= tp_price:
            status = "TARGET HIT"

        found.append({
            "Coin": symbol,
            "Timeframe": TIMEFRAME,
            "Combo": combo_name,
            "Signal Bar": to_pkt_str(signal_bar["timestamp"]),
            "Signal Bar UTC": str(signal_bar["timestamp"]),  # unique-key ke liye (notification dedup)
            "Bars Ago": int(bars_ago),
            "Entry": round(float(entry_price), 6),
            "Current": round(float(current_price), 6),
            "P/L %": round(float(pnl_pct), 2),
            "Trail Stop": round(float(trail_stop), 6),
            "Take Profit": round(float(tp_price), 6),
            "Status": status,
        })

    return found


def main():
    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"Scanning {len(coins)} coins on {TIMEFRAME}...")

    all_results = []
    for symbol in coins:
        try:
            r = scan_symbol(exchange, symbol)
            all_results.extend(r)
        except Exception as e:
            print(f"  [SKIP] {symbol}: {e}")

    output = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "coins_scanned": len(coins),
        "timeframe": TIMEFRAME,
        "signals": all_results,
    }

    with open("latest_signals.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. {len(all_results)} fresh signals saved to latest_signals.json")

    # ---- NAYE signals par notification bhejein (purane repeat na hon) ----
    notified = load_notified_keys()
    new_count = 0

    for sig in all_results:
        key = f"{sig['Coin']}|{sig['Combo']}|{sig['Signal Bar UTC']}"
        if key in notified:
            continue  # ye pehle hi bhej chuke hain

        title = f"🚀 {sig['Coin']} - {sig['Combo']}"
        message = (
            f"Signal Candle: {sig['Signal Bar']}\n"
            f"Timeframe: {sig['Timeframe']}\n"
            f"Entry: {sig['Entry']}\n"
            f"Take Profit: {sig['Take Profit']}\n"
            f"Trail Stop: {sig['Trail Stop']}"
        )
        send_notification(title, message)
        notified[key] = datetime.now(timezone.utc).isoformat()
        new_count += 1

    save_notified_keys(notified)
    print(f"Notifications bheji: {new_count} naye signals")


if __name__ == "__main__":
    main()