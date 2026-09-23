"""
Pullback-in-Uptrend (EMA20/EMA50 + ETH/RS/RS%95 filters) ka WALK-FORWARD
tasdeeq: single 6-maheene ke test mein bohot acha nateeja mila tha
(EMA20+Filters: 607 trades/WR42.5%/PF1.947; EMA50+Filters: 413 trades/
WR42.6%/PF2.458) - magar ye sirf EK dour tha, aur EMA20 Baseline khud
bhi mثbat tha (PF>1) jo is baat ka ishara ho sakta hai ke us dour mein
overall market hi mضbooti se upar chal raha tha.

Ye script poore 1 SAAL (8760 candles, ~1h) ko 4 SEQUENTIAL folds
(~3 maheene har ek) mein taqseem kar ke, har fold mein ALAG SE dekhta
hai ke faida consistent hai ya sirf kisi ek khaas (bullish) daur ki
den tha.

4 variants (sab dekhte hain):
  1) EMA20 Baseline
  2) EMA20 + Filters (ETH+RS+RS%95)
  3) EMA50 Baseline
  4) EMA50 + Filters

Result 'pullback_uptrend_walkforward_RESULTS.txt' mein save hota hai.
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

TREND_EMA_PERIOD = 200
PULLBACK_EMA_FAST = 20
PULLBACK_EMA_SLOW = 50
TOLERANCE_PCT = 1.0
EMA_RISING_LOOKBACK = 5

ETH_EMA_PERIOD = 200
RS_EMA_PERIOD = 50
RS_PERCENTILE_LOOKBACK_DAYS = 180
RS_PERCENTILE_CUTOFF = 95.0
DAILY_FETCH_LIMIT = RS_PERCENTILE_LOOKBACK_DAYS + 420   # 1-saal test window + 180-din lookback warm-up


# ---------------- Strategy: Pullback-in-Uptrend (jaisa single-period test mein tha) ----------------
def pullback_uptrend_entry(df, pullback_ema_period, tolerance_pct,
                            trend_ema_period=TREND_EMA_PERIOD, rising_lookback=EMA_RISING_LOOKBACK):
    close = df["close"]
    open_ = df["open"]
    low = df["low"]

    pullback_ema = close.ewm(span=pullback_ema_period, adjust=False).mean()
    trend_ema = close.ewm(span=trend_ema_period, adjust=False).mean()

    ema_rising = pullback_ema > pullback_ema.shift(rising_lookback)
    uptrend = close > trend_ema

    touched_ema = low <= pullback_ema * (1 + tolerance_pct / 100)
    bullish_candle = close > open_
    held_above = close > pullback_ema

    raw = (
        touched_ema & bullish_candle & held_above & uptrend & ema_rising
        & pullback_ema.notna() & trend_ema.notna()
    )
    fresh = raw.fillna(False) & (~raw.shift(1).fillna(False))
    return fresh


# ---------------- Chandelier exit (backtest_engine.simulate_trades ke barabar,
# magar signal_idx bhi return karta hai - taake fold assign kar sakein) ----------------
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
        "1) EMA20 Baseline",
        "2) EMA20 + Filters (ETH+RS+RS%95)",
        "3) EMA50 Baseline",
        "4) EMA50 + Filters",
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

        try:
            sig_fast = apply_cooldown(pullback_uptrend_entry(df, PULLBACK_EMA_FAST, TOLERANCE_PCT), config.SIGNAL_COOLDOWN_BARS)
            sig_slow = apply_cooldown(pullback_uptrend_entry(df, PULLBACK_EMA_SLOW, TOLERANCE_PCT), config.SIGNAL_COOLDOWN_BARS)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: Pullback calc fail ({e}), skip")
            continue

        n = len(df)

        try:
            fast_trades = simulate_trades_with_idx(df, sig_fast, config.BACKTEST_PARAMS, CE_PB)
            slow_trades = simulate_trades_with_idx(df, sig_slow, config.BACKTEST_PARAMS, CE_PB)

            for t in fast_trades:
                i = t["signal_idx"]
                sig_ts = pd.Timestamp(df["timestamp"].iloc[i])
                f = fold_of(i, n, N_FOLDS)
                results["1) EMA20 Baseline"][f].append(t)
                if passes_new_filters(sig_ts, eth_regime, rs_bullish_series, rs_ratio_series):
                    results["2) EMA20 + Filters (ETH+RS+RS%95)"][f].append(t)

            for t in slow_trades:
                i = t["signal_idx"]
                sig_ts = pd.Timestamp(df["timestamp"].iloc[i])
                f = fold_of(i, n, N_FOLDS)
                results["3) EMA50 Baseline"][f].append(t)
                if passes_new_filters(sig_ts, eth_regime, rs_bullish_series, rs_ratio_series):
                    results["4) EMA50 + Filters"][f].append(t)

        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fail ({e})")

        if done % 10 == 0 or done == len(coins):
            print(f"[{done}/{len(coins)}] ... done")

    output_lines = []

    def emit(line=""):
        print(line)
        output_lines.append(line)

    emit("\n\n########## PULLBACK-IN-UPTREND - WALK-FORWARD RESULTS ##########")
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

    result_file = "pullback_uptrend_walkforward_RESULTS.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
