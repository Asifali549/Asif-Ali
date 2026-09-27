"""
CE Buy-Only walk-forward - THEEK KIYA GAYA (FIXED) version.

Asal wajah maloom hui: purani `ce_walkforward_eth_liquidity.py` mein
trailing stop seedha EXIT indicator (period=16, multiplier=3.0 - tight)
se shuru hota tha. Chhota multiplier hamesha qeemat ke QAREEB (zyada)
stop deta hai - is wajah se trailing stop aksar ENTRY PRICE se UPAR hi
nikal aata tha, aur trade turant, bila wajah, apni hi upar wali line se
"jeet" kar band ho jati thi. Ye hi Win%93/PF45 jaisi ghair-mumkin numbers
ki asal wajah thi.

Production (scheduled_dashboard_scan.py ki compute_trade_progress) mein
ye sahi tareeqe se hota hai:
    - initial_stop = ENTRY indicator (11, 4.5 - chaura) ka stop
    - agar initial_stop >= entry_price -> invalid setup, trade SKIP
    - trailing phir EXIT indicator (16, 3.0) se hoti hai, initial_stop se
      shuru ho kar

Isi FIXED logic ko yahan lagaya gaya hai - baqi sab (variants, folds,
liquidity/ETH filters) purani file jaisa hi rakha gaya hai, taake seedha
comparison ho sake ke asal (theek) numbers kya nikalte hain.

Result 'ce_walkforward_FIXED_RESULTS.txt' mein save hota hai.
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
ETH_DAILY_FETCH_LIMIT = 650      # ~1 saal test window + ~250 din EMA200 warm-up ke liye
ETH_EMA_PERIOD = 200

LIQUIDITY_TOP_60 = 60
LIQUIDITY_TOP_100 = 100

ENTRY_PARAMS = {"period": 11, "multiplier": 4.5}
EXIT_PARAMS = {"period": 16, "multiplier": 3.0}


# ---------- Chandelier core (High-based) ----------

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


def simulate_chandelier_trades_fixed(df, signal, entry_stop_series, exit_stop_series, bt_params):
    """
    THEEK KIYA GAYA version - dekho file ka docstring.

    entry_stop_series: ENTRY indicator (11, 4.5) ka stop - sirf initial
        validity check aur trailing ke SHURUATI point ke liye istemal
        hota hai (production jaisa).
    exit_stop_series: EXIT indicator (16, 3.0) ka stop - trailing isi se
        aage barhti hai (purani file jaisa hi, ye hissa theek tha).
    """
    fee = bt_params["fee_pct"] / 100
    slip = bt_params["slippage_pct"] / 100
    max_hold = bt_params["max_hold_bars"]

    entry_stop_vals = entry_stop_series.values
    exit_stop_vals = exit_stop_series.values
    close = df["close"].values
    low = df["low"].values
    n = len(df)

    trades = []
    skipped_invalid = 0
    signal_idx = np.where(signal.values)[0]

    for i in signal_idx:
        if i + 1 >= n or np.isnan(entry_stop_vals[i]):
            continue
        entry_bar = i + 1
        entry_price = df["open"].values[entry_bar] * (1 + slip)

        initial_stop = entry_stop_vals[i]
        if np.isnan(initial_stop) or initial_stop >= entry_price:
            skipped_invalid += 1
            continue   # FIX: production jaisa - invalid setup bilkul skip

        trail_stop = float(initial_stop)   # FIX: chaure ENTRY indicator se shuru
        exit_price = None
        exit_bar = None
        for j in range(entry_bar, min(entry_bar + max_hold, n)):
            if not np.isnan(exit_stop_vals[j]):
                trail_stop = max(trail_stop, exit_stop_vals[j])
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

    return trades, skipped_invalid


def compute_eth_regime(eth_daily_df, ema_period=200):
    ema = eth_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = eth_daily_df["close"] > ema
    return pd.Series(is_bullish.values, index=pd.to_datetime(eth_daily_df["timestamp"]).values)


def is_bullish_at(regime_series, ts):
    valid = regime_series[regime_series.index <= ts]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])


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
    coins = get_coin_list(exchange)[:TOP_N_COINS]   # liquidity order (24h volume, descending)
    print(f"Top {len(coins)} coins (liquidity order), {SIGNAL_TIMEFRAME}, {CANDLE_LIMIT} candles (~1 saal), {N_FOLDS} folds...\n")

    print("ETH daily data fetch kar rahe hain (regime filter ke liye)...")
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
    results = {v: {f: [] for f in range(N_FOLDS)} for v in variant_names}
    total_skipped_invalid = 0

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
            all_signal_trades, skipped = simulate_chandelier_trades_fixed(
                df, sig, entry_stop, exit_stop, config.BACKTEST_PARAMS
            )
            total_skipped_invalid += skipped
            n = len(df)

            top60_ok = rank < LIQUIDITY_TOP_60
            top100_ok = rank < LIQUIDITY_TOP_100

            for t in all_signal_trades:
                i = t["signal_idx"]
                sig_ts = pd.Timestamp(df["timestamp"].iloc[i])
                eth_ok = is_bullish_at(eth_regime, sig_ts)
                f = fold_of(i, n, N_FOLDS)

                results["1) Baseline (Top-150, koi filter nahi)"][f].append(t)
                if eth_ok:
                    results["2) + ETH Regime Filter (Top-150)"][f].append(t)
                if top60_ok:
                    results["3) Top-60 sirf (liquidity)"][f].append(t)
                if top100_ok:
                    results["4) Top-100 sirf (liquidity)"][f].append(t)
                if top60_ok and eth_ok:
                    results["5) Top-60 + ETH Regime"][f].append(t)
                if top100_ok and eth_ok:
                    results["6) Top-100 + ETH Regime"][f].append(t)

        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fail ({e})")

        if done % 10 == 0 or done == len(coins):
            print(f"[{done}/{len(coins)}] ... done")

    output_lines = []

    def emit(line=""):
        print(line)
        output_lines.append(line)

    emit("\n\n########## CE BUY-ONLY - WALK-FORWARD RESULTS (FIXED - entry-indicator initial stop) ##########")
    emit(f"(Entry {ENTRY_PARAMS['period']}/{ENTRY_PARAMS['multiplier']}, Exit {EXIT_PARAMS['period']}/{EXIT_PARAMS['multiplier']}, "
         f"High-based, top {TOP_N_COINS} coins pool, {SIGNAL_TIMEFRAME}, {CANDLE_LIMIT} candles, {N_FOLDS} sequential folds)")
    emit(f"Total signals jo INVALID setup ki wajah se skip hue (initial_stop >= entry_price): {total_skipped_invalid}\n")

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
            emit(f"{v}: Trades={n}, koi PF nahi (losses hi nahi hain ya trades nahi)")
        else:
            emit(f"{v}: Trades={n}, Win Rate={wr:.2f}%, PF={pf:.3f}")

    result_file = "ce_walkforward_FIXED_RESULTS.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
