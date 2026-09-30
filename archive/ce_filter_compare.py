"""
CE Buy-Only ko MAZBOOT banane ke liye do sab se zyada ummeed-afza filters
test karte hain - ETH Regime Filter aur RS (vs BTC) Filter - alag alag
AUR combined, taake pata chale kaun sa filter (agar koi) asal mein
faida deta hai, aur kaun sa sirf trades ghata deta hai bina PF/win-rate
behtar kiye.

4 variants (sab EK HI entry/exit settings - 11/4.5 entry, 16/3.0 exit,
High-based extremum - jo pehle hi behtar sabit ho chuka hai):
  1) Baseline (koi filter nahi - maujooda live system)
  2) + ETH Regime Filter (ETH daily close > EMA200)
  3) + RS Filter (coin/BTC ratio > apni 50-EMA, AUR RS percentile >= 95)
  4) + Dono (ETH Regime + RS) combined

Filter logic Union AB ke ALREADY-VALIDATED functions se hu-ba-hu li gayi
hai (compute_eth_regime, compute_rs_trend_and_ratio, passes_rs_filters)
taake koi nayi/alag definition na bane.

Result 'ce_filter_compare_RESULTS.txt' mein save hota hai.
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown

TOP_N_COINS = 150
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = 4380              # ~6 maheene (1h) - pehle wale extremum-test se match

RS_EMA_PERIOD = 50
RS_PERCENTILE_LOOKBACK_DAYS = 180
RS_PERCENTILE_CUTOFF = 95.0
ETH_EMA_PERIOD = 200
DAILY_FETCH_LIMIT = 425          # RS_PERCENTILE_LOOKBACK_DAYS(180) max HIGH_52W(365) + 60, production jaisa

ENTRY_PARAMS = {"period": 11, "multiplier": 4.5}
EXIT_PARAMS = {"period": 16, "multiplier": 3.0}


# ---------- Chandelier core (High-based, jo pehle test mein behtar sabit hua) ----------

def compute_atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def chandelier_stop(df, period, multiplier):
    atr = compute_atr(df, period)
    highest = df["high"].rolling(period).max()
    return highest - (multiplier * atr)


def ce_buy_only_signal(df, period, multiplier):
    stop = chandelier_stop(df, period, multiplier)
    close = df["close"]
    cross_above = (close > stop) & (close.shift(1) <= stop.shift(1))
    return cross_above.fillna(False)


def simulate_chandelier_trades(df, signal, exit_stop_series, bt_params):
    fee = bt_params["fee_pct"] / 100
    slip = bt_params["slippage_pct"] / 100
    max_hold = bt_params["max_hold_bars"]

    ce_stop = exit_stop_series.values
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


# ---------- Union AB ke ALREADY-VALIDATED filter functions (hu-ba-hu) ----------

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
    is_bullish = merged["rs_ratio"] > merged["rs_ema"]
    bullish_series = pd.Series(is_bullish.values, index=pd.to_datetime(merged["timestamp"]).values)
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
    rs_pct = percentile_rank_of_last(rs_recent.values)
    return rs_pct >= RS_PERCENTILE_CUTOFF


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
    print(f"Top {len(coins)} coins, {SIGNAL_TIMEFRAME}, {CANDLE_LIMIT} candles (~6 maheene)...\n")

    print("ETH aur BTC daily data fetch kar rahe hain (regime + RS ke liye)...")
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=DAILY_FETCH_LIMIT)
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DAILY_FETCH_LIMIT)
    eth_regime = compute_eth_regime(eth_daily, ema_period=ETH_EMA_PERIOD)

    variant_names = ["1) Baseline (koi filter nahi)", "2) + ETH Regime Filter", "3) + RS Filter", "4) + ETH Regime + RS (dono)"]
    all_trades = {name: [] for name in variant_names}

    done = 0
    for symbol in coins:
        done += 1
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=CANDLE_LIMIT)
            coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=DAILY_FETCH_LIMIT)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fetch fail ({e})")
            continue
        if df is None or len(df) < 300 or coin_daily is None or len(coin_daily) < 60:
            continue

        try:
            sig = apply_cooldown(
                ce_buy_only_signal(df, ENTRY_PARAMS["period"], ENTRY_PARAMS["multiplier"]),
                config.SIGNAL_COOLDOWN_BARS,
            )
            exit_stop = chandelier_stop(df, EXIT_PARAMS["period"], EXIT_PARAMS["multiplier"])
            all_signal_trades = simulate_chandelier_trades(df, sig, exit_stop, config.BACKTEST_PARAMS)

            rs_bullish, rs_ratio = compute_rs_trend_and_ratio(coin_daily, btc_daily, ema_period=RS_EMA_PERIOD)

            for t in all_signal_trades:
                sig_ts = pd.Timestamp(df["timestamp"].iloc[t["signal_idx"]])
                eth_ok = is_bullish_at(eth_regime, sig_ts)
                rs_ok = passes_rs_filters(sig_ts, rs_bullish, rs_ratio)

                all_trades["1) Baseline (koi filter nahi)"].append(t)
                if eth_ok:
                    all_trades["2) + ETH Regime Filter"].append(t)
                if rs_ok:
                    all_trades["3) + RS Filter"].append(t)
                if eth_ok and rs_ok:
                    all_trades["4) + ETH Regime + RS (dono)"].append(t)

        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fail ({e})")

        if done % 10 == 0 or done == len(coins):
            print(f"[{done}/{len(coins)}] ... done")

    output_lines = []

    def emit(line=""):
        print(line)
        output_lines.append(line)

    emit("\n\n########## CE BUY-ONLY - FILTER COMPARISON - RESULTS ##########")
    emit(f"(Entry {ENTRY_PARAMS['period']}/{ENTRY_PARAMS['multiplier']}, Exit {EXIT_PARAMS['period']}/{EXIT_PARAMS['multiplier']}, "
         f"High-based, top {TOP_N_COINS} coins, {SIGNAL_TIMEFRAME}, {CANDLE_LIMIT} candles)\n")

    for name in variant_names:
        pf, n, wr = pf_of(all_trades[name])
        if pf is None:
            emit(f"{name}: koi trades nahi mile")
        else:
            avg_bars = np.mean([t["bars_held"] for t in all_trades[name]])
            emit(f"{name}:")
            emit(f"   Trades = {n}")
            emit(f"   Win Rate = {wr:.2f}%")
            emit(f"   Profit Factor = {pf:.3f}")
            emit(f"   Avg Bars Held = {avg_bars:.1f}")
            emit("")

    result_file = "ce_filter_compare_RESULTS.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
