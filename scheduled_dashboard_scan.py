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
from confluence_engine import compute_confluence, DEFAULT_PARAMS as CONF_PARAMS
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
            data=message.encode("utf-8"),
            headers={"Title": title.encode("utf-8"), "Priority": "high", "Tags": "star"},
            timeout=10,
        )
    except Exception as e:
        print(f"  [NOTIFY FAILED] {e}")


def load_notified_keys():
    if not os.path.exists("notified_strong_signals.json"):
        return {}
    with open("notified_strong_signals.json") as f:
        return json.load(f)


def save_notified_keys(keys_dict):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    pruned = {k: v for k, v in keys_dict.items() if datetime.fromisoformat(v) > cutoff}
    with open("notified_strong_signals.json", "w") as f:
        json.dump(pruned, f, indent=2)


TOP_N_COINS = 400
SIGNAL_TIMEFRAME = "1h"
CE_A = {"period": 16, "multiplier": 3.0}
CE_B = {"period": 12, "multiplier": 3.0}
CE_D = {"period": 16, "multiplier": 3.0}
RR_MULTIPLE = 2.0

COLORABLE_COLUMNS = (
    ["OrderBook Bid/Ask", "Liquidity Compare", "Coin's Own PF", "Funding Rate", "Dist from 24h High"]
    + [f"Vol {tf}" for tf in ALL_TIMEFRAMES]
    + [f"Chg {tf}" for tf in ALL_TIMEFRAMES]
)


def main():
    exchange = get_exchange()
    futures_exchange = get_futures_exchange()

    fg_value, fg_label = get_fear_greed()

    print("BTC daily benchmark data fetch kar rahe hain (NEW system ke liye)...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=800)

    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"Scanning {len(coins)} coins on {SIGNAL_TIMEFRAME}...")

    signal_coins = []
    for symbol in coins:
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=max(config.CANDLE_LIMITS.get(SIGNAL_TIMEFRAME, 500), 300))
        except Exception:
            df = None
        if df is None or len(df) < 220:
            continue
        try:
            ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
            ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
            ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
            breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)
            combo_a = ichi_sig & ms_sig
            combo_b = ema_sig & breakout_sig

            for combo_sig, combo_name, ce in [(combo_a, "Ichimoku+MS", CE_A), (combo_b, "EMA+Breakout", CE_B)]:
                if combo_sig.tail(3).any():
                    idx = combo_sig.tail(3)[combo_sig.tail(3)].index[-1]
                    atr = compute_atr(df, ce["period"])
                    highest_high = df["high"].rolling(ce["period"]).max()
                    chandelier = (highest_high - ce["multiplier"] * atr).loc[idx]
                    entry_price = df.loc[idx, "close"]
                    current_price = df["close"].iloc[-1]
                    risk = entry_price - chandelier
                    tp_price = entry_price + risk * RR_MULTIPLE
                    signal_coins.append({
                        "Coin": symbol, "Combo": combo_name, "Bars Ago": int(len(df) - 1 - idx),
                        "Entry": round(float(entry_price), 6), "Current": round(float(current_price), 6),
                        "Trail Stop": round(float(chandelier), 6), "Take Profit": round(float(tp_price), 6),
                        "_df": df, "_sig": combo_sig, "_ce": ce,
                    })
        except Exception as e:
            print(f"  [SKIP-OLD] {symbol}: {e}")

        # ---- NEW system: AdvancedConfluence_v1 ----
        try:
            result_new = compute_confluence(df, btc_daily, CONF_PARAMS, usdt_d_weak=None)
            structure_signal = result_new["bos"] | result_new["choch"]
            new_sig = apply_cooldown(structure_signal & (result_new["score"] >= 6), config.SIGNAL_COOLDOWN_BARS)

            if new_sig.tail(3).any():
                idx = new_sig.tail(3)[new_sig.tail(3)].index[-1]
                atr = compute_atr(df, CE_D["period"])
                highest_high = df["high"].rolling(CE_D["period"]).max()
                chandelier = (highest_high - CE_D["multiplier"] * atr).loc[idx]
                entry_price = df.loc[idx, "close"]
                current_price = df["close"].iloc[-1]
                risk = entry_price - chandelier
                tp_price = entry_price + risk * RR_MULTIPLE
                signal_coins.append({
                    "Coin": symbol, "Combo": "NEW_AdvancedConfluence", "Bars Ago": int(len(df) - 1 - idx),
                    "Entry": round(float(entry_price), 6), "Current": round(float(current_price), 6),
                    "Trail Stop": round(float(chandelier), 6), "Take Profit": round(float(tp_price), 6),
                    "_df": df, "_sig": new_sig, "_ce": CE_D,
                })
        except Exception as e:
            print(f"  [SKIP-NEW] {symbol}: {e}")

    print(f"Found {len(signal_coins)} signals. Computing full context...")

    final_rows = []
    for row in signal_coins:
        symbol = row["Coin"]
        final_row = {k: v for k, v in row.items() if not k.startswith("_")}

        funding, oi = get_funding_and_oi(futures_exchange, symbol)
        final_row["Funding Rate"] = f"{funding*100:.3f}%" if funding is not None else "N/A"
        final_row["Open Interest"] = f"${oi:,.0f}" if oi is not None else "N/A"

        bid_liq, ask_liq, ob_ratio = get_orderbook_info(exchange, symbol)
        final_row["Liquidity Up ($)"] = f"${ask_liq:,.0f}" if ask_liq is not None else "N/A"
        final_row["Liquidity Down ($)"] = f"${bid_liq:,.0f}" if bid_liq is not None else "N/A"
        final_row["Liquidity Compare"] = ("Support > Resistance" if bid_liq > ask_liq else "Resistance > Support") if (bid_liq is not None and ask_liq is not None) else "N/A"
        final_row["OrderBook Bid/Ask"] = f"{ob_ratio:.2f}x" if ob_ratio is not None else "N/A"

        for tf in ALL_TIMEFRAMES:
            vol_ratio, chg = get_tf_volume_change(exchange, symbol, tf)
            final_row[f"Vol {tf}"] = f"{vol_ratio:.2f}x" if vol_ratio is not None else "N/A"
            final_row[f"Chg {tf}"] = f"{chg:+.2f}%" if chg is not None else "N/A"

        dist_high, dist_low = get_24h_range_distance(exchange, symbol)
        final_row["Dist from 24h High"] = f"{dist_high}%" if dist_high is not None else "N/A"
        final_row["Dist from 24h Low"] = f"{dist_low}%" if dist_low is not None else "N/A"

        win_rate, pf = get_historical_performance(row["_df"], row["_sig"], row["_ce"])
        final_row["Coin's Own Win%"] = f"{win_rate}%" if win_rate is not None else "N/A"
        final_row["Coin's Own PF"] = f"{pf}" if pf is not None else "N/A"

        corr = get_btc_correlation(exchange, symbol, SIGNAL_TIMEFRAME)
        final_row["BTC Correlation"] = f"{corr}" if corr is not None else "N/A"

        score, verdict = compute_overall_score(final_row, COLORABLE_COLUMNS)
        final_row["Overall Score %"] = score
        final_row["Verdict"] = verdict

        final_rows.append(final_row)

    # Best-to-worst rank lagayein
    final_rows.sort(key=lambda r: r["Overall Score %"], reverse=True)
    for i, r in enumerate(final_rows):
        r["Rank"] = i + 1

    output = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "coins_scanned": len(coins),
        "fear_greed_value": fg_value,
        "fear_greed_label": fg_label,
        "signals": final_rows,
    }

    with open("dashboard_signals.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. {len(final_rows)} signals with full context saved to dashboard_signals.json")

    # Sirf "Strong" verdict wale signals ki notification bhejte hain
    notified = load_notified_keys()
    new_count = 0
    for row in final_rows:
        if "Strong" not in row.get("Verdict", ""):
            continue
        key = f"{row['Coin']}|{row['Combo']}|{row.get('Entry')}"
        if key in notified:
            continue
        title = f"⭐ STRONG: {row['Coin']} - {row['Combo']}"
        message = (
            f"Overall Score: {row['Overall Score %']}% ({row['Verdict']})\n"
            f"Entry: {row['Entry']}\n"
            f"Take Profit: {row['Take Profit']}\n"
            f"Trail Stop: {row['Trail Stop']}"
        )
        send_strong_notification(title, message)
        notified[key] = datetime.now(timezone.utc).isoformat()
        new_count += 1

    save_notified_keys(notified)
    print(f"Strong signal notifications bheji: {new_count}")


if __name__ == "__main__":
    main()