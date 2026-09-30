"""
Support/Resistance Bounce - pichli test mein maloom hua ke BAGHAIR filter
ye strategy bade sample (4315 trades) par GHATA deti hai (PF=0.876, WR=27.72%,
Total Return -581%) - koi fluke nahi, mustaqil nuqsaan. 4 naye filters ek
sath lagane se munafa nazar aaya (PF=2.831) magar sample sirf 29 trades
reh gaya - itne kam trades par PF par bharosa nahi kiya ja sakta.

Ye script har filter ko ALAG ALAG (ek waqt mein sirf 1) test karta hai -
taake dekh sakein koi akela filter bina sample ko bilkul khatam kiye
koi asli faida deta hai ya nahi:

  1) Baseline (koi filter nahi) - jaisa pehle test mein tha
  2) + ETH Regime akela (ETH daily close > EMA200)
  3) + RS Trend akela (coin/BTC ratio > apni EMA50)
  4) + RS Percentile>=95 akela (180-din lookback mein top 5%)
  5) + 52-Week-High Distance akela (<=15% door)

Signal: S/R Bounce, tolerance=1.0% (default, bada sample), lookback=50,
uptrend filter (close>EMA200) pehle se signal mein shamil hai. Exit:
Chandelier (period=16, multiplier=4.5) - Union AB defaults, jaisa pehli
test mein tha.

Chalayen: python test_sr_bounce_filters_individual.py
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown
from backtest_engine import compute_atr, simulate_trades

TOP_N_COINS = 150
SIGNAL_TIMEFRAME = "1h"
CE_SR = {"period": 16, "multiplier": 4.5}

SUPPORT_LOOKBACK = 50
TOLERANCE_PCT = 1.0    # pehli test ka "A" bucket (bada sample)

ETH_EMA_PERIOD = 200
RS_EMA_PERIOD = 50
RS_PERCENTILE_LOOKBACK_DAYS = 180
RS_PERCENTILE_CUTOFF = 95.0
HIGH_52W_LOOKBACK_BARS = 365 * 24
HIGH_52W_CUTOFF_PCT = 15.0
EXTENDED_1H_LIMIT = HIGH_52W_LOOKBACK_BARS + 50


# ---------------- Strategy: Support/Resistance Bounce (pehli test jaisa hi) ----------------
def support_resistance_bounce(df, lookback, tolerance_pct):
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    close = df["close"]

    support = low.rolling(lookback).min().shift(1)
    ema200 = close.ewm(span=200, adjust=False).mean()

    touched_support = low <= support * (1 + tolerance_pct / 100)
    bullish_candle = close > open_
    held_above = close > support
    uptrend = close > ema200

    raw = touched_support & bullish_candle & held_above & uptrend & support.notna()
    fresh = raw.fillna(False) & (~raw.shift(1).fillna(False))
    return fresh


# ---------------- Helper functions (pehli test se, alag alag filter ke liye) ----------------
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


def rs_percentile_ok(sig_ts, rs_ratio_series):
    rs_valid = rs_ratio_series[rs_ratio_series.index <= sig_ts]
    rs_recent = rs_valid.iloc[-RS_PERCENTILE_LOOKBACK_DAYS:] if len(rs_valid) > 0 else pd.Series([], dtype=float)
    if len(rs_recent) < 2:
        return False
    return percentile_rank_of_last(rs_recent.values) >= RS_PERCENTILE_CUTOFF


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

    variant_names = [
        "1) Baseline (koi filter nahi)",
        "2) + ETH Regime akela",
        "3) + RS Trend akela (vs BTC)",
        "4) + RS Percentile>=95 akela",
        "5) + 52-Week-High Distance akela (<=15%)",
    ]
    all_trades = {name: [] for name in variant_names}

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

        high_arr = df["high"].values
        close_arr = df["close"].values

        try:
            sig = apply_cooldown(support_resistance_bounce(df, SUPPORT_LOOKBACK, TOLERANCE_PCT), config.SIGNAL_COOLDOWN_BARS)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: S/R Bounce calc fail ({e}), skip")
            continue

        # backtest_engine.simulate_trades trade dicts mein "signal_idx" nahi deta
        # (sirf entry/exit time/price), is liye trades ko baad mein filter karne
        # ke bajaye - jaisa original test_support_resistance_bounce.py mein kiya
        # gaya tha - har filter ke liye SIGNAL SERIES ko khud filter karte hain,
        # phir har variant ke liye simulate_trades alag se chalate hain.
        idxs = np.where(sig.values)[0]

        eth_sig = sig.copy()
        rs_trend_sig = sig.copy()
        rs_pct_sig = sig.copy()
        high52w_sig = sig.copy()

        for i in idxs:
            sig_ts = pd.Timestamp(df["timestamp"].iloc[i])

            if not is_bullish_at(eth_regime, sig_ts):
                eth_sig.iloc[i] = False
            if not is_bullish_at(rs_bullish_series, sig_ts):
                rs_trend_sig.iloc[i] = False
            if not rs_percentile_ok(sig_ts, rs_ratio_series):
                rs_pct_sig.iloc[i] = False
            dist_52w = compute_dist_from_52w_high(high_arr, close_arr, i, HIGH_52W_LOOKBACK_BARS)
            if not (dist_52w is not None and dist_52w <= HIGH_52W_CUTOFF_PCT):
                high52w_sig.iloc[i] = False

        all_trades["1) Baseline (koi filter nahi)"].extend(simulate_trades(df, sig, config.BACKTEST_PARAMS, CE_SR))
        all_trades["2) + ETH Regime akela"].extend(simulate_trades(df, eth_sig, config.BACKTEST_PARAMS, CE_SR))
        all_trades["3) + RS Trend akela (vs BTC)"].extend(simulate_trades(df, rs_trend_sig, config.BACKTEST_PARAMS, CE_SR))
        all_trades["4) + RS Percentile>=95 akela"].extend(simulate_trades(df, rs_pct_sig, config.BACKTEST_PARAMS, CE_SR))
        all_trades["5) + 52-Week-High Distance akela (<=15%)"].extend(simulate_trades(df, high52w_sig, config.BACKTEST_PARAMS, CE_SR))

        if done % 10 == 0 or done == len(coins):
            counts = {k: len(v) for k, v in all_trades.items()}
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

    emit("\n\n########## S/R BOUNCE - HAR FILTER ALAG ALAG - RESULTS ##########")
    for name in variant_names:
        summarize(all_trades[name], name)

    result_file = "test_sr_bounce_filters_individual_RESULTS.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
