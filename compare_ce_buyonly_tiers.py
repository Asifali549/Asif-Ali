"""
CE Buy-Only — Liquidity Tiers + ETH Filter Test
================================================================================
"CE Buy-Only" strategy (pehle chhote/mukhtasar test mein PF 0.91 -
kamzor/rejected) par, hamara tasdeeq-shuda "recipe" (Top-N Liquid
Coins + ETH Regime Filter) laga kar dekhte hain - kya ye pehle se
rejected strategy ko bacha sakta hai, jaisa Union AB ke sath hua tha.

Strategy khud apna BTC trend filter bhi rakhti hai (EMA50) - wo bhi
saath rehta hai. Hum sirf UPAR se Top-N Liquid + ETH filter laga
rahe hain, jaisa Union AB par kiya tha.

Tiers: TOP 60, TOP 100, TOP 150 (ETH filter ke sath, aur bina ETH
filter ke, taake dono compare ho sakein).

CHALANE SE PEHLE:
    pip install ccxt pandas numpy
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from backtest_engine import simulate_trades

DURATION_DAYS = 270
TIERS = [60, 100, 150]
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
CE_EXIT = {"period": 12, "multiplier": 3.0}  # backtest_engine ke exit ke liye, strategy ke apne ATR settings se milta hua

ETH_EMA_PERIOD = 200

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "ce_buyonly_tiers_result.txt"


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
    max_tier = max(TIERS)
    log(f"Scanning coins - pehle {max_tier} full-history coins tak...\n")

    trades_by_tier_noeth = {tier: [] for tier in TIERS}
    trades_by_tier_witheth = {tier: [] for tier in TIERS}
    rank_included = 0

    for n, symbol in enumerate(coins):
        if rank_included >= max_tier:
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
            trades_noeth = simulate_trades(df, signal, BT_PARAMS, CE_EXIT)

            eth_filtered_signal = apply_eth_filter(df, signal, eth_regime)
            trades_witheth = simulate_trades(df, eth_filtered_signal, BT_PARAMS, CE_EXIT)
        except Exception as e:
            log(f"  [SKIP] {symbol}: {e}")
            trades_noeth, trades_witheth = [], []

        for tier in TIERS:
            if rank_included <= tier:
                trades_by_tier_noeth[tier].extend(trades_noeth)
                trades_by_tier_witheth[tier].extend(trades_witheth)

        if rank_included % 20 == 0:
            log(f"  rank {rank_included}/{max_tier} tak coins mil chuke (scan kiye: {n+1})")

    log(f"\n[INFO] Total {rank_included} full-history coins mile")

    log("\n" + "=" * 65)
    log("NATIJA: CE Buy-Only - Liquidity Tiers + ETH Filter")
    log("=" * 65)

    for tier in TIERS:
        log(f"\n--- TOP {tier} LIQUID ---")
        s_noeth = aggregate_trades(trades_by_tier_noeth[tier])
        s_witheth = aggregate_trades(trades_by_tier_witheth[tier])
        log(f"  Bina ETH filter : Trades={s_noeth['total_trades']}  Win%={s_noeth['win_rate_pct']}  "
            f"PF={s_noeth['profit_factor']}  Total Return={s_noeth['total_return_pct']}%")
        log(f"  ETH filter sath : Trades={s_witheth['total_trades']}  Win%={s_witheth['win_rate_pct']}  "
            f"PF={s_witheth['profit_factor']}  Total Return={s_witheth['total_return_pct']}%")

    log("\nNOTE: Pehle chhote/mukhtasar test mein is strategy ka PF 0.91 tha")
    log("(poore coin universe par, bina liquidity restriction ke). Agar")
    log("yahan kisi tier ka PF 1.5+ aata hai, to liquidity+ETH recipe ne")
    log("isay waqai bacha liya - warna ye strategy khud hi kamzor hai.")

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
            with open("ce_buyonly_tiers_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'ce_buyonly_tiers_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)
