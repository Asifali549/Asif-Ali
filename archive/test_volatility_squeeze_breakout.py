"""
Test: Volatility Squeeze Breakout - NAYI strategy (pehli baar test ho rahi
hai). Idea #1 jo pehle recommend ki gayi thi.

Idea: Jab price ek tang range mein consolidate karta hai (ATR% apni hi
recent history ke muqable bohat neecha ho - "squeeze"), phir jab price us
tang range se UPAR nikal kar breakout karta hai, sath mein volume bhi
normal se zyada ho (confirmation) - to ye aksar ek naye move ki shuruaat
hoti hai. Ye classic "volatility contraction -> expansion" idea hai
(Bollinger Squeeze / TTM Squeeze jaisa), lekin yahan ATR% ke rolling
percentile se squeeze measure kiya gaya hai (Bollinger Band width ki
bajaye, taake calculation seedha aur robust rahe).

Ye breakout/trend-continuation family ka idea hai (S/R Bounce jaisa
mean-reversion nahi) - is session ke established sabaq ke mutabiq.

4 buckets:
  A) Squeeze Breakout (range=20) - Baseline (koi naya filter nahi)
  B) Squeeze Breakout (range=20) - + Naye Filters (ETH Regime + RS Trend + RS%95)
  C) Squeeze Breakout (range=10, tez/chhota range) - Baseline
  D) Squeeze Breakout (range=10) - + Naye Filters

NOTE: 52-Week-High Distance filter jaan-boojh kar shamil NAHI - pichli
tests (CE Buy-Only aur S/R Bounce) dono mein ye filter nuqsaan-dah saabit
hua (chhoti-muddat/quick-reentry trades ko harm karta hai).

Top 150 coins, 1h. Exit: Chandelier (period=16, multiplier=4.5).

Chalayen: python test_volatility_squeeze_breakout.py
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown
from backtest_engine import compute_atr, simulate_trades

TOP_N_COINS = 150
SIGNAL_TIMEFRAME = "1h"
CE_PB = {"period": 16, "multiplier": 4.5}

ATR_PERIOD = 14
QUANTILE_WINDOW = 100      # ATR% ka apna rolling history window (squeeze ka "normal" nikalne ke liye)
SQUEEZE_QUANTILE = 0.25    # ATR% is quantile se neeche ho to "squeeze" mana jayega
RANGE_LOOKBACK_FAST = 20   # breakout range: pichli N candles ki high
RANGE_LOOKBACK_SLOW = 10
VOLUME_WINDOW = 20
VOLUME_MULT = 1.5          # breakout candle ka volume, average se kam az kam itna guna ho

ETH_EMA_PERIOD = 200
RS_EMA_PERIOD = 50
RS_PERCENTILE_LOOKBACK_DAYS = 180
RS_PERCENTILE_CUTOFF = 95.0
DAILY_FETCH_LIMIT = RS_PERCENTILE_LOOKBACK_DAYS + 60
EXTENDED_1H_LIMIT = 4380   # ~6 maheene (1h)


# ---------------- Strategy: Volatility Squeeze Breakout ----------------
def volatility_squeeze_breakout(df, range_lookback, atr_period=ATR_PERIOD,
                                 quantile_window=QUANTILE_WINDOW,
                                 squeeze_quantile=SQUEEZE_QUANTILE,
                                 volume_window=VOLUME_WINDOW,
                                 volume_mult=VOLUME_MULT):
    close = df["close"]
    high = df["high"]
    volume = df["volume"]

    atr = compute_atr(df, atr_period)
    atr_pct = atr / close * 100

    # "normal" ATR% ka threshold - apni hi rolling history ka nichla quantile
    atr_pct_threshold = atr_pct.rolling(quantile_window).quantile(squeeze_quantile)
    # squeeze PICHLI bar par tha (breakout wali bar khud volatile hogi, isliye shift(1))
    in_squeeze = atr_pct.shift(1) <= atr_pct_threshold.shift(1)

    # consolidation range ki high - pichli N candles (current candle shamil nahi)
    range_high = high.rolling(range_lookback).max().shift(1)
    breakout = close > range_high

    vol_avg = volume.rolling(volume_window).mean().shift(1)
    vol_confirm = volume > (vol_avg * volume_mult)

    raw = (
        in_squeeze & breakout & vol_confirm
        & atr_pct_threshold.notna() & range_high.notna() & vol_avg.notna()
    )
    fresh = raw.fillna(False) & (~raw.shift(1).fillna(False))
    return fresh


# ---------------- Helper functions (established pattern) ----------------
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


def passes_new_filters(sig_ts, eth_regime, rs_bullish_series, rs_ratio_series):
    """ETH Regime + RS Trend + RS Percentile>=95 (52W-High jaan-boojh kar shamil nahi)."""
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
    return True


def build_filtered(sig_series, df, eth_regime, rs_bullish_series, rs_ratio_series):
    filtered = sig_series.copy()
    idxs = np.where(sig_series.values)[0]
    for i in idxs:
        sig_ts = pd.Timestamp(df["timestamp"].iloc[i])
        if not passes_new_filters(sig_ts, eth_regime, rs_bullish_series, rs_ratio_series):
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

    buckets = {k: [] for k in ["A", "B", "C", "D"]}
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
            coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=DAILY_FETCH_LIMIT)
            rs_bullish_series, rs_ratio_series = compute_rs_trend_and_ratio(coin_daily, btc_daily, ema_period=RS_EMA_PERIOD)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: daily/RS fetch fail ({e}), skip")
            continue

        try:
            sig_fast = apply_cooldown(volatility_squeeze_breakout(df, RANGE_LOOKBACK_FAST), config.SIGNAL_COOLDOWN_BARS)
            sig_slow = apply_cooldown(volatility_squeeze_breakout(df, RANGE_LOOKBACK_SLOW), config.SIGNAL_COOLDOWN_BARS)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: Squeeze calc fail ({e}), skip")
            continue

        sig_fast_filtered = build_filtered(sig_fast, df, eth_regime, rs_bullish_series, rs_ratio_series)
        sig_slow_filtered = build_filtered(sig_slow, df, eth_regime, rs_bullish_series, rs_ratio_series)

        buckets["A"].extend(simulate_trades(df, sig_fast, config.BACKTEST_PARAMS, CE_PB))
        buckets["B"].extend(simulate_trades(df, sig_fast_filtered, config.BACKTEST_PARAMS, CE_PB))
        buckets["C"].extend(simulate_trades(df, sig_slow, config.BACKTEST_PARAMS, CE_PB))
        buckets["D"].extend(simulate_trades(df, sig_slow_filtered, config.BACKTEST_PARAMS, CE_PB))

        if done % 10 == 0 or done == len(coins):
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

    emit("\n\n########## VOLATILITY SQUEEZE BREAKOUT - RESULTS ##########")
    summarize(buckets["A"], f"A) Squeeze Breakout (range={RANGE_LOOKBACK_FAST}) - Baseline")
    summarize(buckets["B"], f"B) Squeeze Breakout (range={RANGE_LOOKBACK_FAST}) - + Naye Filters (ETH+RS+RS%95)")
    summarize(buckets["C"], f"C) Squeeze Breakout (range={RANGE_LOOKBACK_SLOW}) - Baseline")
    summarize(buckets["D"], f"D) Squeeze Breakout (range={RANGE_LOOKBACK_SLOW}) - + Naye Filters")

    result_file = "test_volatility_squeeze_breakout_RESULTS.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
