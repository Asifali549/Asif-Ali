"""
Confirmation Test: Pending Tiers (CE Buy-Only Mult=3.0 + Union AB Backup
Tier) ko PERIOD-SPLIT ke zariye tasdeeq kar rahe hain, taake production
mein shamil karne se pehle pata chale ke ye results poori sample mein
consistent hain ya sirf kisi khaas (ho sakta hai strong-bull) period ka
fluke hain.

Bucket B (CE Buy-Only Mult=6.0) is test mein SHAMIL NAHI - wo pehle hi
test_pending_tiers.py mein reject ho chuka hai (PF 0.898, loss).

Tareeqa: har coin ke 1h data ko do barabar hisso mein split karte hain
(mid = len(df)//2). Phir:
  - Period 1: sirf mid se PEHLE wale candles ka entry-signal chalta hai
    (baad wale zero kar diye jate hain)
  - Period 2: sirf mid ke BAAD wale candles ka entry-signal chalta hai
  - Full: poori sample (reference, jaisa test_pending_tiers.py mein tha)

Agar Full-sample ka result Period1 aur Period2 dono mein alag-alag
consistent (dono profitable, PF ek dusre ke qareeb) nazar aaye, to hi
filter/signal ko "confirmed" mana jayega. Agar sirf ek period mein acha
aur dusre mein bura/negative ho, to ye reject hoga.

6 buckets:
  A1) CE Buy-Only Mult=3.0 - Full sample (reference)
  A2) CE Buy-Only Mult=3.0 - Period 1
  A3) CE Buy-Only Mult=3.0 - Period 2
  D1) Union AB Backup Tier - Full sample (reference)
  D2) Union AB Backup Tier - Period 1
  D3) Union AB Backup Tier - Period 2

Top 150 coins, 1h.

Chalayen: python confirm_pending_tiers.py
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

CE_A = {"period": 16, "multiplier": 4.5}   # Union AB: Ichimoku+MS
CE_B = {"period": 12, "multiplier": 4.5}   # Union AB: EMA+Breakout

ETH_EMA_PERIOD = 200
RS_EMA_PERIOD = 50
RS_PERCENTILE_LOOKBACK_DAYS = 180
RS_PERCENTILE_CUTOFF = 95.0
CANDLE_LIMIT = 3000


# ---------------- Helper functions (test_pending_tiers.py se wahi) ----------------
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


def ce_buy_only_signal(df, ce_params):
    """Price Chandelier trailing-stop line ke upar cross kare (trend-reclaim)."""
    stop = compute_chandelier_long_stop(df, ce_params["period"], ce_params["multiplier"])
    close = df["close"]
    cross_above = (close > stop) & (close.shift(1) <= stop.shift(1))
    return cross_above.fillna(False)


def split_periods(sig_series):
    """Signal ko do hisso mein split karta hai - period1-only aur period2-only versions."""
    n = len(sig_series)
    mid = n // 2
    p1 = sig_series.copy()
    p1.iloc[mid:] = False
    p2 = sig_series.copy()
    p2.iloc[:mid] = False
    return p1, p2


# ---------------- Main ----------------
def main():
    exchange = get_exchange()

    print("BTC daily benchmark data fetch kar rahe hain...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=RS_PERCENTILE_LOOKBACK_DAYS + 420)

    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"Scanning {len(coins)} coins on {SIGNAL_TIMEFRAME} (period-split confirmation)...\n")

    buckets = {k: [] for k in ["A1", "A2", "A3", "D1", "D2", "D3"]}
    done = 0

    for symbol in coins:
        done += 1
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=CANDLE_LIMIT)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fetch fail ({e})")
            continue
        if df is None or len(df) < 400:
            continue

        try:
            coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=RS_PERCENTILE_LOOKBACK_DAYS + 60)
            rs_bullish_series, rs_ratio_series = compute_rs_trend_and_ratio(coin_daily, btc_daily, ema_period=RS_EMA_PERIOD)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: daily/RS fetch fail ({e}), skip")
            continue

        # ---- A: CE Buy-Only, Exit Mult=3.0 ----
        try:
            ce_sig = apply_cooldown(ce_buy_only_signal(df, CE_SIGNAL), config.SIGNAL_COOLDOWN_BARS)
            ce_p1, ce_p2 = split_periods(ce_sig)
            buckets["A1"].extend(simulate_trades(df, ce_sig, config.BACKTEST_PARAMS, CE_EXIT_ORIGINAL))
            buckets["A2"].extend(simulate_trades(df, ce_p1, config.BACKTEST_PARAMS, CE_EXIT_ORIGINAL))
            buckets["A3"].extend(simulate_trades(df, ce_p2, config.BACKTEST_PARAMS, CE_EXIT_ORIGINAL))
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: CE Buy-Only calc fail ({e})")

        # ---- D: Union AB Backup Tier ----
        try:
            ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
            ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
            ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
            breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)
            combo_a = ichi_sig & ms_sig
            combo_b = ema_sig & breakout_sig

            for combo_sig, ce in [(combo_a, CE_A), (combo_b, CE_B)]:
                backup_sig = build_backup_tier(combo_sig, df, rs_bullish_series, rs_ratio_series)
                backup_p1, backup_p2 = split_periods(backup_sig)
                buckets["D1"].extend(simulate_trades(df, backup_sig, config.BACKTEST_PARAMS, ce))
                buckets["D2"].extend(simulate_trades(df, backup_p1, config.BACKTEST_PARAMS, ce))
                buckets["D3"].extend(simulate_trades(df, backup_p2, config.BACKTEST_PARAMS, ce))
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

    emit("\n\n########## CONFIRM PENDING TIERS - RESULTS ##########")
    emit("\n----- CE Buy-Only (Exit Mult=3.0) -----")
    summarize(buckets["A1"], "A1) Full Sample (reference)")
    summarize(buckets["A2"], "A2) Period 1 (purani half)")
    summarize(buckets["A3"], "A3) Period 2 (nayi half)")
    emit("\n----- Union AB Backup Tier (RS+RS%95, ETH NAHI) -----")
    summarize(buckets["D1"], "D1) Full Sample (reference)")
    summarize(buckets["D2"], "D2) Period 1 (purani half)")
    summarize(buckets["D3"], "D3) Period 2 (nayi half)")

    result_file = "confirm_pending_tiers_RESULTS.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
