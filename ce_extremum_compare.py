"""
CE Buy-Only: "High-based extremum" (hamara maujooda tareeqa) vs
"Close-based extremum" (TradingView indicator ka "Use Close Price for
Extremums" option) - dono ko EK HI settings (entry 11/4.5, exit 16/3.0)
par, 150 coins par, bara sample (1h, ~6 maheene) par, side-by-side
test karta hai.

Kisi live file ko chuta nahi - poori tarah standalone, sirf report
deta hai. Result 'ce_extremum_compare_RESULTS.txt' mein save hota hai.
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown

TOP_N_COINS = 150
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = 4380          # ~6 maheene (1h candles) - bara, realistic sample

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


def chandelier_stop(df, period, multiplier, use_close):
    """use_close=False -> hamara maujooda tareeqa (Highest High).
    use_close=True -> indicator ka 'Use Close Price for Extremums' tareeqa."""
    atr = compute_atr(df, period)
    if use_close:
        highest = df["close"].rolling(period).max()
    else:
        highest = df["high"].rolling(period).max()
    return highest - (multiplier * atr)


def ce_buy_only_signal(df, period, multiplier, use_close):
    stop = chandelier_stop(df, period, multiplier, use_close)
    close = df["close"]
    cross_above = (close > stop) & (close.shift(1) <= stop.shift(1))
    return cross_above.fillna(False)


def simulate_chandelier_trades(df, signal, exit_stop_series, bt_params):
    """backtest_engine.simulate_trades ke chandelier-mode ki hu-ba-hu copy,
    bas exit_stop_series bahar se di jati hai (taake use_close variant
    ke sath bhi kaam kare)."""
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

        trades.append({"return_pct": net_return * 100, "bars_held": exit_bar - entry_bar})

    return trades


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

    variants = {
        "HIGH (hamara maujooda tareeqa)": False,
        "CLOSE (indicator ka 'Use Close Price for Extremums')": True,
    }
    all_trades = {name: [] for name in variants}

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

        for name, use_close in variants.items():
            try:
                sig = apply_cooldown(
                    ce_buy_only_signal(df, ENTRY_PARAMS["period"], ENTRY_PARAMS["multiplier"], use_close),
                    config.SIGNAL_COOLDOWN_BARS,
                )
                exit_stop = chandelier_stop(df, EXIT_PARAMS["period"], EXIT_PARAMS["multiplier"], use_close)
                trades = simulate_chandelier_trades(df, sig, exit_stop, config.BACKTEST_PARAMS)
                all_trades[name].extend(trades)
            except Exception as e:
                print(f"[{done}/{len(coins)}] {symbol}: {name} fail ({e})")

        if done % 10 == 0 or done == len(coins):
            print(f"[{done}/{len(coins)}] ... done")

    output_lines = []

    def emit(line=""):
        print(line)
        output_lines.append(line)

    emit("\n\n########## CE BUY-ONLY - HIGH vs CLOSE Extremum - RESULTS ##########")
    emit(f"(Entry {ENTRY_PARAMS['period']}/{ENTRY_PARAMS['multiplier']}, Exit {EXIT_PARAMS['period']}/{EXIT_PARAMS['multiplier']}, "
         f"top {TOP_N_COINS} coins, {SIGNAL_TIMEFRAME}, {CANDLE_LIMIT} candles)\n")

    for name in variants:
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

    result_file = "ce_extremum_compare_RESULTS.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
