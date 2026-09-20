"""
Test: SHADOW47 AI Structure Engine PRO v2.1 ko ALAG (standalone) screener
banane ke liye - dekhna hai kaunsi behtari kaam karti hai.

Pehle test (test_shadow47.py) mein poora SHADOW47 (apna signal + apna
SL/TP2 plan) PF 1.073 par aaya (Top 60 coins, 2445 trades) - kamzor.
Exit reasons se pata chala: 59.8% dafa SL lagta hai - matlab SHADOW47 ka
apna SL (1.3xATR/Structure Hybrid) shayad bohot tang hai.

4 buckets (sab mein SHADOW47 ka apna SIGNAL/score hi hai - sirf exit ya
filter badla ja raha hai):
  A) SHADOW47 Signal (threshold=72) + APNA SL/TP2 Plan       - Baseline (Top-N zyada coins)
  B) SHADOW47 Signal (threshold=72) + Hamara Chandelier Exit  - (period=16, mult=4.5)
  C) SHADOW47 Signal (threshold=72) + APNA SL/TP2 + 52-Week High<=15% Filter
  D) SHADOW47 Signal (threshold=80, zyada sakht) + APNA SL/TP2 Plan

ETH Regime aur RS filters is test mein SHAMIL NAHI kiye gaye - kyunke
SHADOW47 mein pehle se BTC Correlation/RS aur 4x MTF Trend components
maujood hain (redundant hone ka andesha), is liye pehle sirf orthogonal
(alag zaviye ki) tabdeeliyan test kar rahe hain.

Har coin ke 52-week high nikalne ke liye extended 1h data chahiye
(365*24 + buffer candles), is liye coins ki tadaad kam (default 70)
rakhi gayi hai taake GitHub Actions par waqt mein mukammal ho sake.

Chalayen: python test_shadow47_v2.py
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from backtest_engine import simulate_trades
from shadow47_engine import compute_shadow47, build_signal_with_threshold, DEFAULT_PARAMS

TOP_N_COINS = 70
BASE_TIMEFRAME = "1h"

HIGH_52W_LOOKBACK_BARS = 365 * 24
HIGH_52W_CUTOFF_PCT = 15.0
BASE_LIMIT = HIGH_52W_LOOKBACK_BARS + 50   # 52-week high ke liye poora saal chahiye

TF1_LIMIT = 2000   # 15m
TF3_LIMIT = 1000   # 4h
TF4_LIMIT = 500    # 1D
BTC_LIMIT = BASE_LIMIT

CE_CHANDELIER = {"period": 16, "multiplier": 4.5}   # Union AB ka tasdeeq-shuda exit

FEE_SLIP_PARAMS = {
    "fee_pct": config.BACKTEST_PARAMS.get("fee_pct", 0.1),
    "slippage_pct": config.BACKTEST_PARAMS.get("slippage_pct", 0.05),
    "max_hold_bars": config.BACKTEST_PARAMS.get("max_hold_bars", 200),
}


def simulate_shadow47_own_exit(df, buy_signal, sl_series, tp_series, bt_params):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    n = len(df)
    buy = buy_signal.values
    sl_arr = sl_series.values
    tp_arr = tp_series.values
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
        for j in range(entry_bar, min(entry_bar + max_hold, n)):
            if low[j] <= sl_price:
                exit_price = sl_price
                exit_bar = j
                break
            if high[j] >= tp_price:
                exit_price = tp_price
                exit_bar = j
                break

        if exit_price is None:
            last_bar = min(entry_bar + max_hold - 1, n - 1)
            exit_price = close[last_bar]
            exit_bar = last_bar

        exit_price = exit_price * (1 - slip)
        gross = (exit_price - entry_price) / entry_price
        net = gross - (2 * fee)

        trades.append({
            "entry_time": df["timestamp"].iloc[entry_bar],
            "exit_time": df["timestamp"].iloc[exit_bar],
            "return_pct": net * 100,
        })

    return trades


def compute_dist_from_52w_high(high_arr, close_arr, idx, lookback):
    window = high_arr[max(0, idx - lookback + 1):idx + 1]
    if len(window) == 0:
        return None
    hi = window.max()
    if hi <= 0:
        return None
    return (hi - close_arr[idx]) / hi * 100


def apply_52w_filter(sig_series, high_arr, close_arr):
    filtered = sig_series.copy()
    idxs = np.where(sig_series.values)[0]
    for i in idxs:
        dist = compute_dist_from_52w_high(high_arr, close_arr, i, HIGH_52W_LOOKBACK_BARS)
        if dist is None or dist > HIGH_52W_CUTOFF_PCT:
            filtered.iloc[i] = False
    return filtered


def main():
    exchange = get_exchange()

    print("BTC (1h, extended) reference data fetch kar rahe hain...")
    btc_df = fetch_ohlcv(exchange, "BTC/USDT", BASE_TIMEFRAME, limit=BTC_LIMIT)

    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"Scanning {len(coins)} coins - har coin ke 4 timeframes (15m/1h-extended/4h/1D)...\n")

    buckets = {k: [] for k in ["A", "B", "C", "D"]}
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

        if df_1h is None or len(df_1h) < 400:
            continue

        htf_dfs = {"tf1": df_15m, "tf2": df_1h, "tf3": df_4h, "tf4": df_1d}

        try:
            result = compute_shadow47(df_1h, htf_dfs, btc_df, DEFAULT_PARAMS)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: SHADOW47 calc fail ({e}), skip")
            continue

        sig_baseline = result["buySignal"]

        # A) Baseline - apna SL/TP2
        buckets["A"].extend(simulate_shadow47_own_exit(df_1h, sig_baseline, result["longSL"], result["longTP2"], FEE_SLIP_PARAMS))

        # B) Hamara Chandelier Exit
        buckets["B"].extend(simulate_trades(df_1h, sig_baseline, config.BACKTEST_PARAMS, CE_CHANDELIER))

        # C) + 52-Week High<=15% filter (apna SL/TP2 ke sath)
        high_arr = df_1h["high"].values
        close_arr = df_1h["close"].values
        sig_52w = apply_52w_filter(sig_baseline, high_arr, close_arr)
        buckets["C"].extend(simulate_shadow47_own_exit(df_1h, sig_52w, result["longSL"], result["longTP2"], FEE_SLIP_PARAMS))

        # D) Zyada sakht threshold (80) - apna SL/TP2
        sig_strict = build_signal_with_threshold(
            result["longScoreFinal"], result["shortScoreFinal"], result["volRatio"],
            threshold=80.0, min_score_edge=DEFAULT_PARAMS["min_score_edge"],
            min_vol_ratio=DEFAULT_PARAMS["min_vol_ratio"],
        )
        buckets["D"].extend(simulate_shadow47_own_exit(df_1h, sig_strict, result["longSL"], result["longTP2"], FEE_SLIP_PARAMS))

        if done % 5 == 0 or done == len(coins):
            counts = {k: len(v) for k, v in buckets.items()}
            print(f"[{done}/{len(coins)}] ... {counts}")

    output_lines = []

    def emit(line=""):
        print(line)
        output_lines.append(line)

    def summarize(trades, label):
        emit(f"\n=== {label} ===")
        if not trades:
            emit("Koi trades nahi mile.")
            return
        df_t = pd.DataFrame(trades)
        wins = df_t[df_t["return_pct"] > 0]
        losses = df_t[df_t["return_pct"] <= 0]
        win_rate = len(wins) / len(df_t) * 100
        pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
        emit(f"Trades: {len(df_t)}")
        emit(f"Win Rate: {win_rate:.2f}%")
        emit(f"Profit Factor: {pf:.3f}" if pf else "Profit Factor: N/A")
        emit(f"Avg Return/Trade: {df_t['return_pct'].mean():.3f}%")
        emit(f"Total Return (sum): {df_t['return_pct'].sum():.2f}%")

    emit("\n\n########## SHADOW47 STANDALONE SCREENER - RESULTS ##########")
    summarize(buckets["A"], "A) SHADOW47 Signal + Apna SL/TP2 Plan - Baseline")
    summarize(buckets["B"], "B) SHADOW47 Signal + Hamara Chandelier Exit (16, 4.5)")
    summarize(buckets["C"], "C) SHADOW47 Signal + Apna SL/TP2 + 52-Week High<=15% Filter")
    summarize(buckets["D"], "D) SHADOW47 Signal (Threshold=80, sakht) + Apna SL/TP2 Plan")

    result_file = "test_shadow47_v2_RESULTS.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
