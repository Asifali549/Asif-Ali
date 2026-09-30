"""
Test: Do pending/purane-proposed ideas ko FRESH tareeqe se verify kar rahe
hain, taake production mein shamil karne se pehle tasdeeq ho jaye
(purane test scripts is session mein maujood nahi the, is liye dobara
se, saaf tareeqe se banaye gaye hain):

  1) CE Buy-Only + Exit Multiplier=6.0
     Signal: price Chandelier Exit trailing-stop line (period=16,
     multiplier=4.5) ke UPAR cross kare (trend-reclaim entry) - koi
     Ichimoku/MarketStructure/EMA/Breakout combo nahi, sirf Chandelier
     akela.
     Exit: alag Chandelier trail, multiplier=6.0 (zyada wide - trade ko
     saans lene ki jagah) vs multiplier=3.0 (original, tight) compare.

  2) Union AB "Backup Tier" (ETH Filter ke bagair)
     Signal: wahi Union AB combos (Ichimoku+MS ya EMA+Breakout) jo
     RS Filter + RS Percentile>=95 pass karte hain - LEKIN ETH Regime
     Filter check NAHI hota (hamesha active rehta hai, chahe ETH
     bearish ho). Isay "Baseline" (jisme ETH bhi shamil hai) se compare
     kar rahe hain.

6 buckets:
  A) CE Buy-Only - Exit Mult=3.0 (original, jaisa pehle test hua tha)
  B) CE Buy-Only - Exit Mult=6.0 (naya proposal)
  C) Union AB Baseline (ETH+RS+RS%95) - reference (jaisa abhi live hai)
  D) Union AB Backup Tier (sirf RS+RS%95, ETH filter NAHI)

Top 150 coins, 1h.

Chalayen: python test_pending_tiers.py
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import compute_atr, compute_chandelier_long_stop, simulate_trades

TOP_N_COINS = 150
SIGNAL_TIMEFRAME = "1h"

CE_SIGNAL = {"period": 16, "multiplier": 4.5}     # CE Buy-Only ke liye SIGNAL wali chandelier line
CE_EXIT_ORIGINAL = {"period": 16, "multiplier": 3.0}
CE_EXIT_WIDE = {"period": 16, "multiplier": 6.0}

CE_A = {"period": 16, "multiplier": 4.5}   # Union AB: Ichimoku+MS
CE_B = {"period": 12, "multiplier": 4.5}   # Union AB: EMA+Breakout

ETH_EMA_PERIOD = 200
RS_EMA_PERIOD = 50
RS_PERCENTILE_LOOKBACK_DAYS = 180
RS_PERCENTILE_CUTOFF = 95.0
CANDLE_LIMIT = 3000


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


def passes_rs_filters(sig_ts, rs_bullish_series, rs_ratio_series):
    if not is_bullish_at(rs_bullish_series, sig_ts):
        return False
    rs_valid = rs_ratio_series[rs_ratio_series.index <= sig_ts]
    rs_recent = rs_valid.iloc[-RS_PERCENTILE_LOOKBACK_DAYS:] if len(rs_valid) > 0 else pd.Series([], dtype=float)
    if len(rs_recent) < 2:
        return False
    return percentile_rank_of_last(rs_recent.values) >= RS_PERCENTILE_CUTOFF


def build_backup_tier(sig_series, df, rs_bullish_series, rs_ratio_series):
    """RS+RS%95 pass, ETH check NAHI (hamesha active)."""
    filtered = sig_series.copy()
    idxs = np.where(sig_series.values)[0]
    for i in idxs:
        sig_ts = pd.Timestamp(df["timestamp"].iloc[i])
        if not passes_rs_filters(sig_ts, rs_bullish_series, rs_ratio_series):
            filtered.iloc[i] = False
    return filtered


def build_baseline_tier(sig_series, df, eth_regime, rs_bullish_series, rs_ratio_series):
    """ETH + RS + RS%95 - jaisa abhi live production mein hai."""
    filtered = sig_series.copy()
    idxs = np.where(sig_series.values)[0]
    for i in idxs:
        sig_ts = pd.Timestamp(df["timestamp"].iloc[i])
        if not is_bullish_at(eth_regime, sig_ts):
            filtered.iloc[i] = False
            continue
        if not passes_rs_filters(sig_ts, rs_bullish_series, rs_ratio_series):
            filtered.iloc[i] = False
    return filtered


def ce_buy_only_signal(df, ce_params):
    """Price Chandelier trailing-stop line ke upar cross kare (trend-reclaim)."""
    stop = compute_chandelier_long_stop(df, ce_params["period"], ce_params["multiplier"])
    close = df["close"]
    cross_above = (close > stop) & (close.shift(1) <= stop.shift(1))
    return cross_above.fillna(False)


# ---------------- Main ----------------
def main():
    exchange = get_exchange()

    print("BTC daily benchmark data fetch kar rahe hain...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=RS_PERCENTILE_LOOKBACK_DAYS + 420)

    print("ETH daily data + 200 EMA regime nikal rahe hain...")
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=800)
    eth_regime = compute_eth_regime(eth_daily, ema_period=ETH_EMA_PERIOD)

    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"Scanning {len(coins)} coins on {SIGNAL_TIMEFRAME}...\n")

    buckets = {k: [] for k in ["A", "B", "C", "D"]}
    done = 0

    for symbol in coins:
        done += 1
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=CANDLE_LIMIT)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fetch fail ({e})")
            continue
        if df is None or len(df) < 300:
            continue

        try:
            coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=RS_PERCENTILE_LOOKBACK_DAYS + 60)
            rs_bullish_series, rs_ratio_series = compute_rs_trend_and_ratio(coin_daily, btc_daily, ema_period=RS_EMA_PERIOD)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: daily/RS fetch fail ({e}), skip")
            continue

        # ---- A/B: CE Buy-Only ----
        try:
            ce_sig = apply_cooldown(ce_buy_only_signal(df, CE_SIGNAL), config.SIGNAL_COOLDOWN_BARS)
            buckets["A"].extend(simulate_trades(df, ce_sig, config.BACKTEST_PARAMS, CE_EXIT_ORIGINAL))
            buckets["B"].extend(simulate_trades(df, ce_sig, config.BACKTEST_PARAMS, CE_EXIT_WIDE))
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: CE Buy-Only calc fail ({e})")

        # ---- C/D: Union AB Baseline vs Backup Tier ----
        try:
            ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
            ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
            ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
            breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)
            combo_a = ichi_sig & ms_sig
            combo_b = ema_sig & breakout_sig

            for combo_sig, ce in [(combo_a, CE_A), (combo_b, CE_B)]:
                baseline_sig = build_baseline_tier(combo_sig, df, eth_regime, rs_bullish_series, rs_ratio_series)
                backup_sig = build_backup_tier(combo_sig, df, rs_bullish_series, rs_ratio_series)
                buckets["C"].extend(simulate_trades(df, baseline_sig, config.BACKTEST_PARAMS, ce))
                buckets["D"].extend(simulate_trades(df, backup_sig, config.BACKTEST_PARAMS, ce))
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: Union AB calc fail ({e})")

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

    emit("\n\n########## PENDING TIERS - RESULTS ##########")
    summarize(buckets["A"], "A) CE Buy-Only - Exit Mult=3.0 (original)")
    summarize(buckets["B"], "B) CE Buy-Only - Exit Mult=6.0 (naya proposal)")
    summarize(buckets["C"], "C) Union AB Baseline (ETH+RS+RS%95) - reference")
    summarize(buckets["D"], "D) Union AB Backup Tier (RS+RS%95, ETH NAHI)")

    result_file = "test_pending_tiers_RESULTS.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
