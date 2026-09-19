"""
Test: SHADOW47 AI Structure Engine PRO v2.1 (user ne diya hua Pine Script)
- iska APNA poora signal (weighted 0-100 score, MTF+Structure+Momentum+
Volume+Liquidity+Correlation+Premium-Discount) aur APNA SL/TP plan (Hybrid
stop mode, TP2 = 2R) hamare historical data par backtest kar rahe hain.

Ye Union AB ya NEW_AdvancedConfluence se ALAG, mukammal apna ek system hai
- iska maqsad dekhna hai ke SHADOW47 hamare liye AS-IS kaisa kaam karta hai.

MTF ke liye 4 timeframes fetch ki jati hain (Pine script defaults ke
mutabiq): 15m, 1h, 4h, 1D. Is wajah se ye test pichle tests se BHARI hai
(har coin ke 5 fetches - 1h/15m/4h/1D + BTC alag se ek dafa) - is liye
coins ki tadaad thodi kam (default 60) rakhi gayi hai taake GitHub Actions
par waqt mein mukammal ho sake.

Chalayen: python test_shadow47.py
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from shadow47_engine import compute_shadow47, DEFAULT_PARAMS

TOP_N_COINS = 60
BASE_TIMEFRAME = "1h"
BASE_LIMIT = 3000
TF1_LIMIT = 2000   # 15m
TF3_LIMIT = 1000   # 4h
TF4_LIMIT = 500    # 1D
BTC_LIMIT = 3000

FEE_SLIP_PARAMS = {
    "fee_pct": config.BACKTEST_PARAMS.get("fee_pct", 0.1),
    "slippage_pct": config.BACKTEST_PARAMS.get("slippage_pct", 0.05),
    "max_hold_bars": config.BACKTEST_PARAMS.get("max_hold_bars", 200),
}


def simulate_shadow47_trades(df, result, bt_params):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    n = len(df)
    buy = result["buySignal"].values
    sl_arr = result["longSL"].values
    tp_arr = result["longTP2"].values
    fee = bt_params["fee_pct"] / 100
    slip = bt_params["slippage_pct"] / 100
    max_hold = bt_params["max_hold_bars"]

    trades = []
    idxs = np.where(buy)[0]
    for i in idxs:
        if i + 1 >= n:
            continue
        sl_price = sl_arr[i]
        tp_price = tp_arr[i]
        if np.isnan(sl_price) or np.isnan(tp_price):
            continue

        entry_bar = i + 1
        entry_price = open_[entry_bar] * (1 + slip)

        exit_price = None
        exit_bar = None
        exit_reason = None
        for j in range(entry_bar, min(entry_bar + max_hold, n)):
            if low[j] <= sl_price:
                exit_price = sl_price
                exit_bar = j
                exit_reason = "SL"
                break
            if high[j] >= tp_price:
                exit_price = tp_price
                exit_bar = j
                exit_reason = "TP2"
                break

        if exit_price is None:
            last_bar = min(entry_bar + max_hold - 1, n - 1)
            exit_price = close[last_bar]
            exit_bar = last_bar
            exit_reason = "TIME"

        exit_price = exit_price * (1 - slip)
        gross = (exit_price - entry_price) / entry_price
        net = gross - (2 * fee)

        trades.append({
            "entry_time": df["timestamp"].iloc[entry_bar],
            "exit_time": df["timestamp"].iloc[exit_bar],
            "return_pct": net * 100,
            "exit_reason": exit_reason,
        })

    return trades


def main():
    exchange = get_exchange()

    print("BTC (1h) reference data fetch kar rahe hain...")
    btc_df = fetch_ohlcv(exchange, "BTC/USDT", BASE_TIMEFRAME, limit=BTC_LIMIT)

    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"Scanning {len(coins)} coins - har coin ke 4 timeframes (15m/1h/4h/1D)...\n")

    all_trades = []
    done = 0

    for symbol in coins:
        done += 1
        try:
            df_1h = fetch_ohlcv(exchange, symbol, BASE_TIMEFRAME, limit=BASE_LIMIT)
            df_15m = fetch_ohlcv(exchange, symbol, "15m", limit=TF1_LIMIT)
            df_4h = fetch_ohlcv(exchange, symbol, "4h", limit=TF3_LIMIT)
            df_1d = fetch_ohlcv(exchange, symbol, "1d", limit=TF4_LIMIT)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fetch fail ({e})")
            continue

        if df_1h is None or len(df_1h) < 300:
            continue

        htf_dfs = {"tf1": df_15m, "tf2": df_1h, "tf3": df_4h, "tf4": df_1d}

        try:
            result = compute_shadow47(df_1h, htf_dfs, btc_df, DEFAULT_PARAMS)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: SHADOW47 calc fail ({e}), skip")
            continue

        trades = simulate_shadow47_trades(df_1h, result, FEE_SLIP_PARAMS)
        all_trades.extend(trades)

        if done % 5 == 0 or done == len(coins):
            print(f"[{done}/{len(coins)}] ... total_trades={len(all_trades)}")

    def summarize(trades, label):
        print(f"\n=== {label} ===")
        if not trades:
            print("Koi trades nahi mile.")
            return
        df_t = pd.DataFrame(trades)
        wins = df_t[df_t["return_pct"] > 0]
        losses = df_t[df_t["return_pct"] <= 0]
        win_rate = len(wins) / len(df_t) * 100
        pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
        print(f"Trades: {len(df_t)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Profit Factor: {pf:.3f}" if pf else "Profit Factor: N/A")
        print(f"Avg Return/Trade: {df_t['return_pct'].mean():.3f}%")
        print(f"Total Return (sum): {df_t['return_pct'].sum():.2f}%")
        print("Exit reasons:", df_t["exit_reason"].value_counts().to_dict())

    output_lines = []

    def emit(line=""):
        print(line)
        output_lines.append(line)

    emit("\n\n########## SHADOW47 RESULTS (apna Signal + apna SL/TP2 plan) ##########")
    summarize(all_trades, "SHADOW47 AI PRO v2.1 - Poora Sample")

    result_file = "test_shadow47_RESULTS.txt"
    lines = []
    if all_trades:
        df_t = pd.DataFrame(all_trades)
        wins = df_t[df_t["return_pct"] > 0]
        losses = df_t[df_t["return_pct"] <= 0]
        win_rate = len(wins) / len(df_t) * 100
        pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
        lines.append("SHADOW47 AI PRO v2.1 - RESULTS")
        lines.append(f"Trades: {len(df_t)}")
        lines.append(f"Win Rate: {win_rate:.2f}%")
        lines.append(f"Profit Factor: {pf:.3f}" if pf else "Profit Factor: N/A")
        lines.append(f"Avg Return/Trade: {df_t['return_pct'].mean():.3f}%")
        lines.append(f"Total Return (sum): {df_t['return_pct'].sum():.2f}%")
        lines.append(f"Exit reasons: {df_t['exit_reason'].value_counts().to_dict()}")
    else:
        lines.append("SHADOW47 AI PRO v2.1 - Koi trades nahi mile.")

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
