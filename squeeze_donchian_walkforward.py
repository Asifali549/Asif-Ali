"""
Volatility Squeeze Breakout + Donchian Channel Breakout - EK SATH
WALK-FORWARD tasdeeq (poore 1 saal, 4 folds).

Wajah: single 6-maheene ke test mein dono strategies ka nateeja mila:
  - Volatility Squeeze: Baseline dono variants (range=20/10) HAAR rahe the
    (PF < 1), sirf +Filters wale buckets acha lag rahe the (PF 2.5+,
    101-125 trades).
  - Donchian: Baseline KHUD bhi bade sample (9847/5254 trades) par
    mثbat tha (PF 1.05-1.10), aur +Filters se aur behtar ho gaya
    (PF 2.0-2.4, 285-422 trades) - is poore session ka sab se mazboot
    single-period nateeja.

Magar is session ka established sabaq (Pullback-in-Uptrend) yahi hai ke
EK 6-maheene ke dour ka mثbat baseline bhi sirf us dour ke market-trend
ki den ho sakta hai. Isi liye ab dono strategies ko EK SATH, poore SAAL
(8760 candles) par, 4 SEQUENTIAL folds mein test kiya ja raha hai -
taake dekh sakein kaun sa variant har dour mein consistent rehta hai
aur kaun sirf "ek acha dour" ki den tha.

8 variants (test ho rahe hain):
  1) Squeeze range=20 - Baseline
  2) Squeeze range=20 - + Filters (ETH+RS+RS%95)
  3) Squeeze range=10 - Baseline
  4) Squeeze range=10 - + Filters
  5) Donchian 20-period - Baseline
  6) Donchian 20-period - + Filters
  7) Donchian 55-period - Baseline
  8) Donchian 55-period - + Filters

NOTE: 52-Week-High Distance filter jaan-boojh kar shamil NAHI (established
sabaq: chhoti-muddat/quick-reentry trades ko harm karta hai).

Top 150 coins, 1h. Exit: Chandelier (period=16, multiplier=4.5).

Result 'squeeze_donchian_walkforward_RESULTS.txt' mein save hota hai.
Chalayen: python squeeze_donchian_walkforward.py
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown

TOP_N_COINS = 150
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = 8760              # ~1 saal (1h)
N_FOLDS = 4                      # har fold ~3 maheene
CE_PB = {"period": 16, "multiplier": 4.5}

# Squeeze params
ATR_PERIOD = 14
QUANTILE_WINDOW = 100
SQUEEZE_QUANTILE = 0.25
RANGE_LOOKBACK_FAST = 20
RANGE_LOOKBACK_SLOW = 10
VOLUME_WINDOW = 20
VOLUME_MULT = 1.5

# Donchian params
DONCHIAN_PERIOD_FAST = 20
DONCHIAN_PERIOD_SLOW = 55

ETH_EMA_PERIOD = 200
RS_EMA_PERIOD = 50
RS_PERCENTILE_LOOKBACK_DAYS = 180
RS_PERCENTILE_CUTOFF = 95.0
DAILY_FETCH_LIMIT = RS_PERCENTILE_LOOKBACK_DAYS + 420   # 1-saal test window + 180-din lookback warm-up


# ---------------- Strategies (jaisa single-period tests mein the) ----------------
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

    atr_pct_threshold = atr_pct.rolling(quantile_window).quantile(squeeze_quantile)
    in_squeeze = atr_pct.shift(1) <= atr_pct_threshold.shift(1)

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


def donchian_channel_breakout(df, channel_period):
    high = df["high"]
    close = df["close"]
    donchian_high = high.rolling(channel_period).max().shift(1)
    raw = (close > donchian_high) & donchian_high.notna()
    fresh = raw.fillna(False) & (~raw.shift(1).fillna(False))
    return fresh


# ---------------- Chandelier exit (backtest_engine.simulate_trades ke hoobahoo
# barabar, magar signal_idx bhi return karta hai - taake fold assign kar sakein) ----------------
def compute_atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_chandelier_long_stop(df, period, multiplier):
    atr = compute_atr(df, period)
    highest_high = df["high"].rolling(period).max()
    return highest_high - (multiplier * atr)


def simulate_trades_with_idx(df, signal, bt_params, ce_params):
    """backtest_engine.simulate_trades (exit_mode='chandelier') ke hoobahoo
    barabar - sirf signal_idx bhi trade dict mein shamil karta hai."""
    fee = bt_params["fee_pct"] / 100
    slip = bt_params["slippage_pct"] / 100
    max_hold = bt_params["max_hold_bars"]

    ce_stop = compute_chandelier_long_stop(df, ce_params["period"], ce_params["multiplier"]).values
    close = df["close"].values
    low = df["low"].values
    n = len(df)

    trades = []
    signal_idx = np.where(signal.values)[0]

    for i in signal_idx:
        if i + 1 >= n or np.isnan(ce_stop[i]):
            continue
        entry_bar = i + 1
        entry_price = df["open"].values[entry_bar] * (1 + slip)

        trail_stop = ce_stop[i]
        exit_price = None
        exit_bar = None
        for j in range(entry_bar, min(entry_bar + max_hold, n)):
            if not np.isnan(ce_stop[j]):
                trail_stop = max(trail_stop, ce_stop[j])
            if low[j] <= trail_stop:
                exit_price = trail_stop
                exit_bar = j
                break

        if exit_price is None:
            last_bar = min(entry_bar + max_hold - 1, n - 1)
            exit_price = close[last_bar]
            exit_bar = last_bar

        exit_price = exit_price * (1 - slip)
        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - (2 * fee)
        trades.append({
            "signal_idx": i,
            "return_pct": net_return * 100,
            "bars_held": exit_bar - entry_bar,
        })

    return trades


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


def fold_of(signal_idx, n, n_folds):
    bounds = [int(n * i / n_folds) for i in range(n_folds + 1)]
    for f in range(n_folds):
        if bounds[f] <= signal_idx < bounds[f + 1]:
            return f
    return n_folds - 1


def pf_of(trades):
    if not trades:
        return None, 0, None
    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["return_pct"] > 0]
    losses = df_t[df_t["return_pct"] <= 0]
    win_rate = len(wins) / len(df_t) * 100
    pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
    return pf, len(df_t), win_rate


def main():
    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"Top {len(coins)} coins, {SIGNAL_TIMEFRAME}, {CANDLE_LIMIT} candles (~1 saal), {N_FOLDS} folds...\n")

    print("BTC daily benchmark data fetch kar rahe hain...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DAILY_FETCH_LIMIT)

    print("ETH daily data + 200 EMA regime nikal rahe hain...")
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=DAILY_FETCH_LIMIT)
    eth_regime = compute_eth_regime(eth_daily, ema_period=ETH_EMA_PERIOD)

    variant_names = [
        "1) Squeeze range=20 - Baseline",
        "2) Squeeze range=20 - + Filters (ETH+RS+RS%95)",
        "3) Squeeze range=10 - Baseline",
        "4) Squeeze range=10 - + Filters",
        "5) Donchian 20-period - Baseline",
        "6) Donchian 20-period - + Filters",
        "7) Donchian 55-period - Baseline",
        "8) Donchian 55-period - + Filters",
    ]
    results = {v: {f: [] for f in range(N_FOLDS)} for v in variant_names}

    done = 0
    for symbol in coins:
        done += 1
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=CANDLE_LIMIT)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: 1h fetch fail ({e})")
            continue
        if df is None or len(df) < 500:
            continue

        try:
            coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=DAILY_FETCH_LIMIT)
            rs_bullish_series, rs_ratio_series = compute_rs_trend_and_ratio(coin_daily, btc_daily, ema_period=RS_EMA_PERIOD)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: daily/RS fetch fail ({e}), skip")
            continue

        n = len(df)

        try:
            sq_fast_sig = apply_cooldown(volatility_squeeze_breakout(df, RANGE_LOOKBACK_FAST), config.SIGNAL_COOLDOWN_BARS)
            sq_slow_sig = apply_cooldown(volatility_squeeze_breakout(df, RANGE_LOOKBACK_SLOW), config.SIGNAL_COOLDOWN_BARS)
            dc_fast_sig = apply_cooldown(donchian_channel_breakout(df, DONCHIAN_PERIOD_FAST), config.SIGNAL_COOLDOWN_BARS)
            dc_slow_sig = apply_cooldown(donchian_channel_breakout(df, DONCHIAN_PERIOD_SLOW), config.SIGNAL_COOLDOWN_BARS)

            sq_fast_trades = simulate_trades_with_idx(df, sq_fast_sig, config.BACKTEST_PARAMS, CE_PB)
            sq_slow_trades = simulate_trades_with_idx(df, sq_slow_sig, config.BACKTEST_PARAMS, CE_PB)
            dc_fast_trades = simulate_trades_with_idx(df, dc_fast_sig, config.BACKTEST_PARAMS, CE_PB)
            dc_slow_trades = simulate_trades_with_idx(df, dc_slow_sig, config.BACKTEST_PARAMS, CE_PB)

            for t in sq_fast_trades:
                i = t["signal_idx"]
                sig_ts = pd.Timestamp(df["timestamp"].iloc[i])
                f = fold_of(i, n, N_FOLDS)
                results["1) Squeeze range=20 - Baseline"][f].append(t)
                if passes_new_filters(sig_ts, eth_regime, rs_bullish_series, rs_ratio_series):
                    results["2) Squeeze range=20 - + Filters (ETH+RS+RS%95)"][f].append(t)

            for t in sq_slow_trades:
                i = t["signal_idx"]
                sig_ts = pd.Timestamp(df["timestamp"].iloc[i])
                f = fold_of(i, n, N_FOLDS)
                results["3) Squeeze range=10 - Baseline"][f].append(t)
                if passes_new_filters(sig_ts, eth_regime, rs_bullish_series, rs_ratio_series):
                    results["4) Squeeze range=10 - + Filters"][f].append(t)

            for t in dc_fast_trades:
                i = t["signal_idx"]
                sig_ts = pd.Timestamp(df["timestamp"].iloc[i])
                f = fold_of(i, n, N_FOLDS)
                results["5) Donchian 20-period - Baseline"][f].append(t)
                if passes_new_filters(sig_ts, eth_regime, rs_bullish_series, rs_ratio_series):
                    results["6) Donchian 20-period - + Filters"][f].append(t)

            for t in dc_slow_trades:
                i = t["signal_idx"]
                sig_ts = pd.Timestamp(df["timestamp"].iloc[i])
                f = fold_of(i, n, N_FOLDS)
                results["7) Donchian 55-period - Baseline"][f].append(t)
                if passes_new_filters(sig_ts, eth_regime, rs_bullish_series, rs_ratio_series):
                    results["8) Donchian 55-period - + Filters"][f].append(t)

        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fail ({e})")

        if done % 10 == 0 or done == len(coins):
            print(f"[{done}/{len(coins)}] ... done")

    output_lines = []

    def emit(line=""):
        print(line)
        output_lines.append(line)

    emit("\n\n########## SQUEEZE + DONCHIAN - WALK-FORWARD RESULTS ##########")
    emit(f"(Top {TOP_N_COINS} coins, {SIGNAL_TIMEFRAME}, {CANDLE_LIMIT} candles, {N_FOLDS} sequential folds)\n")

    header = "Variant | " + " | ".join([f"Fold{i+1} PF (trades, win%)" for i in range(N_FOLDS)])
    emit(header)
    emit("-" * len(header))

    for v in variant_names:
        cells = []
        for f in range(N_FOLDS):
            pf, n, wr = pf_of(results[v][f])
            if pf is None:
                cells.append("N/A" if n == 0 else f"N/A ({n}, {wr:.0f}%)")
            else:
                cells.append(f"{pf:.2f} ({n}, {wr:.0f}%)")
        emit(f"{v}\n    " + " | ".join(cells))

    emit("\n----- Har variant ka poore saal (4 folds combined) ka overall result -----")
    for v in variant_names:
        all_v_trades = []
        for f in range(N_FOLDS):
            all_v_trades.extend(results[v][f])
        pf, n, wr = pf_of(all_v_trades)
        if pf is None:
            emit(f"{v}: Trades={n}, koi PF nahi")
        else:
            emit(f"{v}: Trades={n}, Win Rate={wr:.2f}%, PF={pf:.3f}")

    result_file = "squeeze_donchian_walkforward_RESULTS.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
