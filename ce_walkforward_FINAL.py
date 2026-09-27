"""
CE Buy-Only - FINAL walk-forward (order-fix + liquidity variants +
outlier-sensitivity).

Ye teeno cheezein ek sath karta hai:
  1) Simulation order THEEK (v2 wali fix - pehle check, phir update)
  2) Liquidity variants (Top-60/100/150, +ETH Regime) - taake dekh
     sakein ke jab illiquid/naye-listed coins hata dein to numbers
     kitne "normal" ho jate hain
  3) Har variant ke liye "outlier-sensitivity": poora saal ka PF, aur
     phir sab se zyada munafa wali TOP 10 trades nikal kar dobara PF -
     taake pata chale poora result mutthi bhar "lucky" trades par to
     nahi tika hua

Result 'ce_walkforward_FINAL_RESULTS.txt' mein save hota hai.
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown

TOP_N_COINS = 150
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = 8760
N_FOLDS = 4
ETH_DAILY_FETCH_LIMIT = 650
ETH_EMA_PERIOD = 200

LIQUIDITY_TOP_60 = 60
LIQUIDITY_TOP_100 = 100
OUTLIER_TRIM_N = 10   # sab se zyada munafa wali itni trades nikal kar dubara PF

ENTRY_PARAMS = {"period": 11, "multiplier": 4.5}
EXIT_PARAMS = {"period": 16, "multiplier": 3.0}


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


def simulate_fixed_order(df, signal, entry_stop_series, exit_stop_series, bt_params):
    """v2 wali FIX: pehle check (pichli maloom stop se), phir update (current bar se)."""
    fee = bt_params["fee_pct"] / 100
    slip = bt_params["slippage_pct"] / 100
    max_hold = bt_params["max_hold_bars"]

    entry_stop_vals = entry_stop_series.values
    exit_stop_vals = exit_stop_series.values
    close = df["close"].values
    low = df["low"].values
    n = len(df)

    trades = []
    signal_idx = np.where(signal.values)[0]

    for i in signal_idx:
        if i + 1 >= n or np.isnan(entry_stop_vals[i]):
            continue
        entry_bar = i + 1
        entry_price = df["open"].values[entry_bar] * (1 + slip)

        initial_stop = entry_stop_vals[i]
        if np.isnan(initial_stop) or initial_stop >= entry_price:
            continue

        trail_stop = float(initial_stop)
        exit_price = None
        exit_bar = None
        for j in range(entry_bar, min(entry_bar + max_hold, n)):
            if low[j] <= trail_stop:
                exit_price = trail_stop
                exit_bar = j
                break
            if not np.isnan(exit_stop_vals[j]):
                trail_stop = max(trail_stop, exit_stop_vals[j])

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


def compute_eth_regime(eth_daily_df, ema_period=200):
    ema = eth_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = eth_daily_df["close"] > ema
    return pd.Series(is_bullish.values, index=pd.to_datetime(eth_daily_df["timestamp"]).values)


def is_bullish_at(regime_series, ts):
    valid = regime_series[regime_series.index <= ts]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])


def pf_of(trades):
    if not trades:
        return None, 0, None
    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["return_pct"] > 0]
    losses = df_t[df_t["return_pct"] <= 0]
    win_rate = len(wins) / len(df_t) * 100
    pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
    return pf, len(df_t), win_rate


def pf_excluding_top_n(trades, n):
    """Sab se zyada munafa wali N trades nikal kar PF dobara - outlier-sensitivity."""
    if len(trades) <= n:
        return None, 0, None
    sorted_trades = sorted(trades, key=lambda t: t["return_pct"], reverse=True)
    trimmed = sorted_trades[n:]
    return pf_of(trimmed)


def main():
    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"Top {len(coins)} coins (liquidity order), {SIGNAL_TIMEFRAME}, {CANDLE_LIMIT} candles (~1 saal)...\n")

    print("ETH daily data fetch kar rahe hain...")
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=ETH_DAILY_FETCH_LIMIT)
    eth_regime = compute_eth_regime(eth_daily, ema_period=ETH_EMA_PERIOD)

    variant_names = [
        "1) Baseline (Top-150, koi filter nahi)",
        "2) + ETH Regime Filter (Top-150)",
        "3) Top-60 sirf (liquidity)",
        "4) Top-100 sirf (liquidity)",
        "5) Top-60 + ETH Regime",
        "6) Top-100 + ETH Regime",
    ]
    results = {v: [] for v in variant_names}

    done = 0
    for rank, symbol in enumerate(coins):
        done += 1
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=CANDLE_LIMIT)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fetch fail ({e})")
            continue
        if df is None or len(df) < 500:
            continue

        try:
            sig = apply_cooldown(
                ce_buy_only_signal(df, ENTRY_PARAMS["period"], ENTRY_PARAMS["multiplier"]),
                config.SIGNAL_COOLDOWN_BARS,
            )
            entry_stop = chandelier_stop(df, ENTRY_PARAMS["period"], ENTRY_PARAMS["multiplier"])
            exit_stop = chandelier_stop(df, EXIT_PARAMS["period"], EXIT_PARAMS["multiplier"])
            all_signal_trades = simulate_fixed_order(df, sig, entry_stop, exit_stop, config.BACKTEST_PARAMS)

            top60_ok = rank < LIQUIDITY_TOP_60
            top100_ok = rank < LIQUIDITY_TOP_100

            for t in all_signal_trades:
                i = t["signal_idx"]
                sig_ts = pd.Timestamp(df["timestamp"].iloc[i])
                eth_ok = is_bullish_at(eth_regime, sig_ts)

                results["1) Baseline (Top-150, koi filter nahi)"].append(t)
                if eth_ok:
                    results["2) + ETH Regime Filter (Top-150)"].append(t)
                if top60_ok:
                    results["3) Top-60 sirf (liquidity)"].append(t)
                if top100_ok:
                    results["4) Top-100 sirf (liquidity)"].append(t)
                if top60_ok and eth_ok:
                    results["5) Top-60 + ETH Regime"].append(t)
                if top100_ok and eth_ok:
                    results["6) Top-100 + ETH Regime"].append(t)

        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fail ({e})")

        if done % 20 == 0 or done == len(coins):
            print(f"[{done}/{len(coins)}] ... done")

    output_lines = []

    def emit(line=""):
        print(line)
        output_lines.append(line)

    emit("\n\n########## CE BUY-ONLY - FINAL WALK-FORWARD (order-fixed + outlier-sensitivity) ##########")
    emit(f"(Entry {ENTRY_PARAMS['period']}/{ENTRY_PARAMS['multiplier']}, Exit {EXIT_PARAMS['period']}/{EXIT_PARAMS['multiplier']}, "
         f"top {TOP_N_COINS} coins pool, {SIGNAL_TIMEFRAME}, {CANDLE_LIMIT} candles)\n")

    for v in variant_names:
        trades = results[v]
        pf, n, wr = pf_of(trades)
        emit(f"\n{v}")
        if pf is None:
            emit(f"  Trades={n}: PF/Win% calculate nahi ho saka (kaafi data nahi)")
            continue
        emit(f"  POORA: Trades={n}, Win Rate={wr:.2f}%, PF={pf:.3f}")

        pf_trim, n_trim, wr_trim = pf_excluding_top_n(trades, OUTLIER_TRIM_N)
        if pf_trim is not None:
            emit(f"  Top-{OUTLIER_TRIM_N} outlier trades NIKAAL KAR: Trades={n_trim}, Win Rate={wr_trim:.2f}%, PF={pf_trim:.3f}")
            emit(f"  -> In {OUTLIER_TRIM_N} trades ka PF mein hissa: {pf:.2f} -> {pf_trim:.2f}")

    result_file = "ce_walkforward_FINAL_RESULTS.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
