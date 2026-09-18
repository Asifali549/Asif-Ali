"""
CE Buy-Only — Exit Chandelier Multiplier Tuning
================================================================================
CE Buy-Only strategy Top 60 Liquid + ETH Filter (behtareen tier) par
fix rakh kar, sirf EXIT ke chandelier multiplier ko badal kar test
karta hai (3.0, 4.0, 4.5, 5.0, 6.0) - taake pata chale ke kya tang
exit stop hi is strategy ki asal kamzori hai, jaisa Union AB mein
mult 3.0->4.5 se faida hua tha.

Entry signal ki apni logic (ATR mult=3.0 reversal detection) NAHI
badli - sirf trade ke bane baad, exit trailing stop kitna door rakhna
hai, wahi test ho raha hai.

CHALANE SE PEHLE:
    pip install ccxt pandas numpy
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from backtest_engine import simulate_trades

DURATION_DAYS = 270
N_TOP_LIQUID = 60
N_FETCH_COINS = 300
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = DURATION_DAYS * 24 + 300
MIN_BARS_REQUIRED = (DURATION_DAYS - 5) * 24

CE_PARAMS = {
    "atr_period": 12,
    "atr_mult": 3.0,
    "vol_ma_len": 20,
    "vol_multiplier": 1.0,
    "buy_pressure_ratio": 0.6,
    "btc_ema_len": 50,
}
EXIT_MULTIPLIERS = [3.0, 4.0, 4.5, 5.0, 6.0]
EXIT_PERIOD = 12

ETH_EMA_PERIOD = 200

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "ce_buyonly_exit_mult_result.txt"


def ce_buy_only(df, btc_df, params):
    atr_period = params["atr_period"]
    atr_mult = params["atr_mult"]

    close = df["close"].values
    length = len(df)

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean().values

    highest_high = df["close"].rolling(atr_period).max().values
    lowest_low = df["close"].rolling(atr_period).min().values

    long_stop = highest_high - atr * atr_mult
    short_stop = lowest_low + atr * atr_mult

    long_stop_prev = np.full(length, np.nan)
    short_stop_prev = np.full(length, np.nan)
    dir_arr = np.ones(length, dtype=int)

    for i in range(length):
        ls_prev = long_stop_prev[i - 1] if i > 0 and not np.isnan(long_stop_prev[i - 1]) else long_stop[i]
        ss_prev = short_stop_prev[i - 1] if i > 0 and not np.isnan(short_stop_prev[i - 1]) else short_stop[i]

        ls = max(long_stop[i], ls_prev) if i > 0 and close[i - 1] > ls_prev else long_stop[i]
        ss = min(short_stop[i], ss_prev) if i > 0 and close[i - 1] < ss_prev else short_stop[i]

        prev_dir = dir_arr[i - 1] if i > 0 else 1
        if close[i] > ss_prev:
            dir_arr[i] = 1
        elif close[i] < ls_prev:
            dir_arr[i] = -1
        else:
            dir_arr[i] = prev_dir

        long_stop_prev[i] = ls
        short_stop_prev[i] = ss

    buy_signal_raw = np.zeros(length, dtype=bool)
    for i in range(1, length):
        if dir_arr[i] == 1 and dir_arr[i - 1] == -1:
            buy_signal_raw[i] = True

    vol_ma = df["volume"].rolling(params["vol_ma_len"]).mean()
    vol_ok = df["volume"] > vol_ma * params["vol_multiplier"]
    close_position = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    upside_vol_ok = (close_position > params["buy_pressure_ratio"]) & vol_ok

    btc_ema = btc_df["close"].ewm(span=params["btc_ema_len"], adjust=False).mean()
    btc_ok_series = (btc_df["close"] > btc_ema).rename("btc_ok")
    merged = pd.merge_asof(
        df[["timestamp"]].sort_values("timestamp"),
        pd.DataFrame({"timestamp": btc_df["timestamp"], "btc_ok": btc_ok_series}).sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    btc_ok = merged["btc_ok"].fillna(False).values

    confirmed_buy = buy_signal_raw & upside_vol_ok.fillna(False).values & btc_ok
    return pd.Series(confirmed_buy, index=df.index)


def compute_eth_regime(eth_daily_df, ema_period=200):
    ema = eth_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = eth_daily_df["close"] > ema
    return pd.Series(is_bullish.values, index=pd.to_datetime(eth_daily_df["timestamp"]).values)


def is_eth_bullish_at(eth_regime, ts):
    valid = eth_regime[eth_regime.index <= ts]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])


def apply_eth_filter(df, signal, eth_regime):
    filtered = signal.copy()
    for idx in signal[signal].index:
        ts = pd.Timestamp(df["timestamp"].iloc[idx])
        if not is_eth_bullish_at(eth_regime, ts):
            filtered.iloc[idx] = False
    return filtered


def aggregate_trades(all_trades):
    if not all_trades:
        return {"total_trades": 0, "win_rate_pct": None, "profit_factor": None, "total_return_pct": None}
    returns = pd.Series([t["return_pct"] for t in all_trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = len(wins) / len(returns) * 100
    gross_profit = wins.sum() if len(wins) else 0
    gross_loss = abs(losses.sum()) if len(losses) else 0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("nan")
    return {
        "total_trades": len(returns),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 3) if pf == pf else None,
        "total_return_pct": round(returns.sum(), 2),
    }


def main():
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(str(msg))

    exchange = get_exchange()

    log("ETH daily data + 200 EMA regime nikal rahe hain...")
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=DURATION_DAYS + 250)
    eth_regime = compute_eth_regime(eth_daily, ema_period=ETH_EMA_PERIOD)
    log(f"  ETH bullish regime: {eth_regime.sum()}/{len(eth_regime)} din\n")

    log("BTC daily data nikal rahe hain (strategy ke apne BTC filter ke liye)...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DURATION_DAYS + 250)

    coins = get_coin_list(exchange)[:N_FETCH_COINS]
    log(f"Scanning coins - pehle {N_TOP_LIQUID} full-history coins tak (Top {N_TOP_LIQUID} Liquid + ETH Filter)...\n")

    trades_by_mult = {m: [] for m in EXIT_MULTIPLIERS}
    rank_included = 0

    for n, symbol in enumerate(coins):
        if rank_included >= N_TOP_LIQUID:
            break
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=CANDLE_LIMIT)
        except Exception:
            df = None
        if df is None or len(df) < MIN_BARS_REQUIRED:
            continue
        rank_included += 1

        try:
            signal = ce_buy_only(df, btc_daily, CE_PARAMS)
            eth_filtered_signal = apply_eth_filter(df, signal, eth_regime)

            for mult in EXIT_MULTIPLIERS:
                ce_exit = {"period": EXIT_PERIOD, "multiplier": mult}
                trades = simulate_trades(df, eth_filtered_signal, BT_PARAMS, ce_exit)
                trades_by_mult[mult].extend(trades)
        except Exception as e:
            log(f"  [SKIP] {symbol}: {e}")

        if rank_included % 10 == 0:
            log(f"  {rank_included}/{N_TOP_LIQUID} coins mile (scan kiye: {n+1})")

    log(f"\n[INFO] Total {rank_included} full-history coins mile")

    log("\n" + "=" * 65)
    log("NATIJA: CE Buy-Only - Exit Chandelier Multiplier Tuning")
    log("=" * 65)

    best_pf = -999
    best_mult = None
    for mult in EXIT_MULTIPLIERS:
        stats = aggregate_trades(trades_by_mult[mult])
        log(f"\nExit Multiplier={mult}: Trades={stats['total_trades']}  Win%={stats['win_rate_pct']}  "
            f"PF={stats['profit_factor']}  Total Return={stats['total_return_pct']}%")
        if stats['profit_factor'] and stats['profit_factor'] > best_pf:
            best_pf = stats['profit_factor']
            best_mult = mult

    log(f"\n>>> Sab se behtar Exit Multiplier: {best_mult} (PF={best_pf}) <<<")
    log("\nNOTE: Agar kisi bhi multiplier par PF 1.5+ nahi aata, to sirf exit")
    log("stop door karna is strategy ko nahi bacha sakta - masla entry")
    log("signal ki apni quality mein hai, exit mechanism mein nahi.")

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        print(f"\n[SAVED] Result '{OUTPUT_FILE}' file mein save ho gaya hai.")
    except Exception as e:
        print(f"\n[ERROR] Result file save nahi ho saki: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        error_text = traceback.format_exc()
        print("\n" + "=" * 60)
        print("SCRIPT MEIN ERROR AAYA:")
        print("=" * 60)
        print(error_text)
        try:
            with open("ce_buyonly_exit_mult_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'ce_buyonly_exit_mult_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)
