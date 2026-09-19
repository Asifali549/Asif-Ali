"""
Test: FVG (Fair Value Gap) Strategy - pehle se config.py/strategies.py mein
maujood hai lekin kabhi mukammal test nahi hui.

8 buckets (4 versions x baseline/+filters):
  A) FVG akela                    - Baseline
  B) FVG akela                    - + Naye Filters (ETH+RS+RS%95+52W)
  C) FVG + Market Structure       - Baseline
  D) FVG + Market Structure       - + Naye Filters
  E) FVG + EMA Crossover          - Baseline
  F) FVG + EMA Crossover          - + Naye Filters
  G) FVG + Breakout               - Baseline
  H) FVG + Breakout               - + Naye Filters

Top 150 coins, 1h timeframe. Exit: Chandelier (period=16, multiplier=4.5),
jaisa Union AB mein hai (FVG abhi khud calibrate nahi hua, is liye Union AB
ka defaults istemal kar rahe hain as starting point).

Chalayen: python test_fvg_strategy.py
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import compute_atr, simulate_trades

TOP_N_COINS = 150
SIGNAL_TIMEFRAME = "1h"
CE_FVG = {"period": 16, "multiplier": 4.5}   # Union AB ke defaults se shuru

ETH_EMA_PERIOD = 200
RS_EMA_PERIOD = 50
RS_PERCENTILE_LOOKBACK_DAYS = 180
RS_PERCENTILE_CUTOFF = 95.0
HIGH_52W_LOOKBACK_BARS = 365 * 24
HIGH_52W_CUTOFF_PCT = 15.0
EXTENDED_1H_LIMIT = HIGH_52W_LOOKBACK_BARS + 50


# ---------------- Helper functions ----------------
def compute_eth_regime(eth_daily_df, ema_period=200):
    ema = eth_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = eth_daily_df["close"] > ema
    return pd.Series(is_bullish.values, index=pd.to_datetime(eth_daily_df["timestamp"]).values)


def is_bullish_at(regime_series, ts):
    valid = regime_series[regime_series.index <= ts]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])


def compute_rs_trend_and_ratio(coin_daily_df, btc_daily_df, ema_period=50):
    coin_d = coin_daily_df[["timestamp", "close"]].rename(columns={"close": "coin_close"})
    btc_d = btc_daily_df[["timestamp", "close"]].rename(columns={"close": "btc_close"})
    merged = pd.merge_asof(
        coin_d.sort_values("timestamp"), btc_d.sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    merged["rs_ratio"] = merged["coin_close"] / merged["btc_close"].replace(0, np.nan)
    merged["rs_ema"] = merged["rs_ratio"].ewm(span=ema_period, adjust=False).mean()
    is_bull = merged["rs_ratio"] > merged["rs_ema"]
    bullish_series = pd.Series(is_bull.values, index=pd.to_datetime(merged["timestamp"]).values)
    ratio_series = pd.Series(merged["rs_ratio"].values, index=pd.to_datetime(merged["timestamp"]).values)
    return bullish_series, ratio_series


def percentile_rank_of_last(values):
    if len(values) < 2:
        return 50.0
    last = values[-1]
    return float((values <= last).sum()) / len(values) * 100


def compute_dist_from_52w_high(high_arr, close_arr, idx, lookback):
    window = high_arr[max(0, idx - lookback + 1):idx + 1]
    if len(window) == 0:
        return None
    hi = window.max()
    if hi <= 0:
        return None
    return (hi - close_arr[idx]) / hi * 100


def passes_new_filters(sig_ts, idx, eth_regime, rs_bullish_series, rs_ratio_series, high_arr, close_arr):
    if not is_bullish_at(eth_regime, sig_ts):
        return False
    if not is_bullish_at(rs_bullish_series, sig_ts):
        return False
    rs_valid = rs_ratio_series[rs_ratio_series.index <= sig_ts]
    rs_recent = rs_valid.iloc[-RS_PERCENTILE_LOOKBACK_DAYS:] if len(rs_valid) > 0 else pd.Series([], dtype=float)
    if len(rs_recent) < 2:
        return False
    if percentile_rank_of_last(rs_recent.values) < RS_PERCENTILE_CUTOFF:
        return False
    dist_52w = compute_dist_from_52w_high(high_arr, close_arr, idx, HIGH_52W_LOOKBACK_BARS)
    if dist_52w is None or dist_52w > HIGH_52W_CUTOFF_PCT:
        return False
    return True


def build_filtered(sig_series, df, eth_regime, rs_bullish_series, rs_ratio_series, high_arr, close_arr):
    filtered = sig_series.copy()
    idxs = np.where(sig_series.values)[0]
    for i in idxs:
        sig_ts = pd.Timestamp(df["timestamp"].iloc[i])
        if not passes_new_filters(sig_ts, i, eth_regime, rs_bullish_series, rs_ratio_series, high_arr, close_arr):
            filtered.iloc[i] = False
    return filtered


# ---------------- Main ----------------
def main():
    exchange = get_exchange()

    print("BTC daily benchmark data fetch kar rahe hain...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=RS_PERCENTILE_LOOKBACK_DAYS + 420)

    print("ETH daily data + 200 EMA regime nikal rahe hain...")
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=800)
    eth_regime = compute_eth_regime(eth_daily, ema_period=ETH_EMA_PERIOD)

    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"Scanning {len(coins)} coins on {SIGNAL_TIMEFRAME} (extended {EXTENDED_1H_LIMIT} candles)...\n")

    buckets = {k: [] for k in ["A", "B", "C", "D", "E", "F", "G", "H"]}
    done = 0

    for symbol in coins:
        done += 1
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=EXTENDED_1H_LIMIT)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: 1h fetch fail ({e})")
            continue
        if df is None or len(df) < 300:
            continue

        try:
            coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=RS_PERCENTILE_LOOKBACK_DAYS + 60)
            rs_bullish_series, rs_ratio_series = compute_rs_trend_and_ratio(coin_daily, btc_daily, ema_period=RS_EMA_PERIOD)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: daily/RS fetch fail ({e}), skip")
            continue

        try:
            fvg_sig = apply_cooldown(STRATEGY_FUNCTIONS["fvg"](df, config.STRATEGY_PARAMS["fvg"]), config.SIGNAL_COOLDOWN_BARS)
            ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
            ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
            breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: strategy calc fail ({e}), skip")
            continue

        high_arr = df["high"].values
        close_arr = df["close"].values

        sig_map = {
            "A": fvg_sig,
            "C": fvg_sig & ms_sig,
            "E": fvg_sig & ema_sig,
            "G": fvg_sig & breakout_sig,
        }

        for base_key, filt_key in [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]:
            base_sig = sig_map[base_key]
            filt_sig = build_filtered(base_sig, df, eth_regime, rs_bullish_series, rs_ratio_series, high_arr, close_arr)
            buckets[base_key].extend(simulate_trades(df, base_sig, config.BACKTEST_PARAMS, CE_FVG))
            buckets[filt_key].extend(simulate_trades(df, filt_sig, config.BACKTEST_PARAMS, CE_FVG))

        if done % 10 == 0 or done == len(coins):
            counts = {k: len(v) for k, v in buckets.items()}
            print(f"[{done}/{len(coins)}] ... {counts}")

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

    print("\n\n########## FINAL RESULTS ##########")
    summarize(buckets["A"], "A) FVG Akela - Baseline")
    summarize(buckets["B"], "B) FVG Akela - + Naye Filters (ETH+RS+RS%95+52W)")
    summarize(buckets["C"], "C) FVG + Market Structure - Baseline")
    summarize(buckets["D"], "D) FVG + Market Structure - + Naye Filters")
    summarize(buckets["E"], "E) FVG + EMA Crossover - Baseline")
    summarize(buckets["F"], "F) FVG + EMA Crossover - + Naye Filters")
    summarize(buckets["G"], "G) FVG + Breakout - Baseline")
    summarize(buckets["H"], "H) FVG + Breakout - + Naye Filters")


if __name__ == "__main__":
    main()
