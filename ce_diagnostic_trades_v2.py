"""
CE Buy-Only - DIAGNOSTIC v2 (asal bug theek kiya gaya).

v1 mein pata chala: 88% trades sirf 0.3 bars mein hi band ho rahi thin
(taqreeban FORAN), kuch to 525% tak "munafa" dikha rahi thin. Wajah:
har candle par PEHLE usi candle ki apni high se stop upar uthaya ja raha
tha, phir USI candle ke low ko us (abhi upar hue) stop se check kiya ja
raha tha - candle ki apni high, usi candle ke low ke khilaf istemal ho
rahi thi (future leakage, kyunke ek hi candle ke andar high pehle bani
ya low, hum nahi jaante).

FIX: ab tarteeb ulti hai -
    1) PEHLE current bar ka low, PICHLI bar tak maloom stop se check hota hai
    2) US ke BAAD current bar ke data se agli bar ke liye stop update hota hai

Result 'ce_diagnostic_v2_trades.csv' aur 'ce_diagnostic_v2_summary.txt'
mein save hota hai.
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown

TOP_N_COINS = 150
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = 8760

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


def simulate_with_detail_v2(symbol, df, signal, entry_stop_series, exit_stop_series, bt_params):
    """
    FIX: check-before-update tarteeb (dekho file ka docstring).
    """
    fee = bt_params["fee_pct"] / 100
    slip = bt_params["slippage_pct"] / 100
    max_hold = bt_params["max_hold_bars"]

    entry_stop_vals = entry_stop_series.values
    exit_stop_vals = exit_stop_series.values
    close = df["close"].values
    low = df["low"].values
    n = len(df)

    rows = []
    signal_idx = np.where(signal.values)[0]

    for i in signal_idx:
        if i + 1 >= n or np.isnan(entry_stop_vals[i]):
            continue
        entry_bar = i + 1
        entry_price = df["open"].values[entry_bar] * (1 + slip)

        initial_stop = entry_stop_vals[i]
        if np.isnan(initial_stop) or initial_stop >= entry_price:
            continue

        trail_stop = float(initial_stop)   # sirf signal-bar (i) tak maloom stop
        exit_price = None
        exit_bar = None
        exit_reason = None
        for j in range(entry_bar, min(entry_bar + max_hold, n)):
            # FIX 1) PEHLE: current bar (j) ka low, PICHLI maloom stop se check
            if low[j] <= trail_stop:
                exit_price = trail_stop
                exit_bar = j
                exit_reason = "STOPPED"
                break
            # FIX 2) US ke BAAD: current bar (j) ke data se agli bar ke liye update
            if not np.isnan(exit_stop_vals[j]):
                trail_stop = max(trail_stop, exit_stop_vals[j])

        if exit_price is None:
            last_bar = min(entry_bar + max_hold - 1, n - 1)
            exit_price = close[last_bar]
            exit_bar = last_bar
            exit_reason = "TIME"

        exit_price_net = exit_price * (1 - slip)
        gross_return = (exit_price_net - entry_price) / entry_price
        net_return = gross_return - (2 * fee)

        rows.append({
            "symbol": symbol,
            "signal_time": df["timestamp"].iloc[i],
            "entry_time": df["timestamp"].iloc[entry_bar],
            "entry_price": round(entry_price, 6),
            "initial_stop": round(float(initial_stop), 6),
            "exit_time": df["timestamp"].iloc[exit_bar],
            "exit_price": round(exit_price_net, 6),
            "exit_reason": exit_reason,
            "bars_held": exit_bar - entry_bar,
            "return_pct": round(net_return * 100, 3),
        })

    return rows


def main():
    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"Top {len(coins)} coins, {SIGNAL_TIMEFRAME}, {CANDLE_LIMIT} candles - FIXED v2 (check-before-update)...\n")

    all_rows = []
    done = 0
    for symbol in coins:
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
            rows = simulate_with_detail_v2(symbol, df, sig, entry_stop, exit_stop, config.BACKTEST_PARAMS)
            all_rows.extend(rows)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fail ({e})")

        if done % 20 == 0 or done == len(coins):
            print(f"[{done}/{len(coins)}] ... done ({len(all_rows)} trades so far)")

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv("ce_diagnostic_v2_trades.csv", index=False)
    print(f"\n[SAVE] {len(df_all)} trades -> ce_diagnostic_v2_trades.csv")

    lines = []

    def emit(line=""):
        print(line)
        lines.append(line)

    emit("\n########## CE BUY-ONLY - DIAGNOSTIC v2 (FIXED order) SUMMARY ##########")
    emit(f"Total trades: {len(df_all)}")

    for reason in ["STOPPED", "TIME"]:
        sub = df_all[df_all["exit_reason"] == reason]
        if len(sub) == 0:
            emit(f"\n{reason}: 0 trades")
            continue
        wins = sub[sub["return_pct"] > 0]
        emit(f"\n{reason}: {len(sub)} trades ({len(sub)/len(df_all)*100:.1f}% of all)")
        emit(f"  Win Rate: {len(wins)/len(sub)*100:.1f}%")
        emit(f"  Avg return: {sub['return_pct'].mean():.2f}%")
        emit(f"  Avg bars held: {sub['bars_held'].mean():.1f}")
        emit(f"  Max single-trade return: {sub['return_pct'].max():.2f}%")
        emit(f"  Min single-trade return: {sub['return_pct'].min():.2f}%")

    if len(df_all) > 0:
        wins_all = df_all[df_all["return_pct"] > 0]
        losses_all = df_all[df_all["return_pct"] <= 0]
        win_rate = len(wins_all) / len(df_all) * 100
        pf = wins_all["return_pct"].sum() / abs(losses_all["return_pct"].sum()) if len(losses_all) and losses_all["return_pct"].sum() != 0 else None
        emit(f"\n----- OVERALL (poora saal, sab coins) -----")
        emit(f"Trades={len(df_all)}, Win Rate={win_rate:.2f}%, PF={pf:.3f}" if pf else f"Trades={len(df_all)}, Win Rate={win_rate:.2f}%, PF=N/A")
        emit(f"Avg bars held (sab trades): {df_all['bars_held'].mean():.1f}")

    with open("ce_diagnostic_v2_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[SAVE] ce_diagnostic_v2_summary.txt bhi ban gayi.")


if __name__ == "__main__":
    main()
