"""
ETH Regime Filter on NEW System (AdvancedConfluence) - Comparison Backtest
================================================================================
NEW_AdvancedConfluence_v1 (BOS/CHoCH + Score>=6, 1h) par test karta hai:
    A) BASELINE - jaisa abhi hai (koi ETH filter nahi, sirf apna internal
       BTC-bullish component jo confluence score ka hissa hai)
    B) +ETH_REGIME - signal sirf tab valid jab ETH ka daily close apni
       daily EMA(200) se upar ho (Union AB mein jo already live hai,
       wahi cheez yahan NEW system par test kar rahe hain)

CHALANE SE PEHLE:
    pip install ccxt pandas numpy

NOTE: Isay repo ke usi folder mein rakhein jahan confluence_engine.py,
backtest_engine.py, config.py, data_fetcher.py, strategies.py maujood hain.
Ye file (btc_regime_filter.py) bhi chahiye hogi - wahi generic EMA regime
logic hai jo pehle BTC ke liye test ki thi, ab ETH ke data ke sath use
ho rahi hai.
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown
from backtest_engine import simulate_trades
from confluence_engine import compute_confluence, DEFAULT_PARAMS as CONF_PARAMS
from btc_regime_filter import compute_btc_regime_series, is_bullish_regime

TEST_N_COINS = 200
SIGNAL_TIMEFRAME = "1h"
CE_D = {"period": 16, "multiplier": 3.0}

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "eth_regime_new_system_result.txt"


def apply_regime_filter(df, signal, regime_series):
    filtered = signal.copy()
    for idx in signal[signal].index:
        ts = pd.Timestamp(df["timestamp"].iloc[idx])
        if not is_bullish_regime(regime_series, ts):
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
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=800)
    eth_regime = compute_btc_regime_series(eth_daily, ema_period=200)
    bullish_days = eth_regime.sum()
    total_days = len(eth_regime)
    log(f"  ETH bullish regime: {bullish_days}/{total_days} din ({bullish_days/total_days*100:.1f}%)\n")

    log("BTC daily data nikal rahe hain (NEW system ke apne internal score ke liye)...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=800)

    coins = get_coin_list(exchange)[:TEST_N_COINS]
    log(f"Scanning {len(coins)} coins on {SIGNAL_TIMEFRAME}...\n")

    trades_baseline = []
    trades_filtered = []

    for n, symbol in enumerate(coins):
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME,
                              limit=max(config.CANDLE_LIMITS.get(SIGNAL_TIMEFRAME, 500), 300))
        except Exception:
            df = None
        if df is None or len(df) < 220:
            continue

        try:
            result_new = compute_confluence(df, btc_daily, CONF_PARAMS, usdt_d_weak=None)
            structure_signal = result_new["bos"] | result_new["choch"]
            new_sig = apply_cooldown(structure_signal & (result_new["score"] >= 6), config.SIGNAL_COOLDOWN_BARS)

            trades_a = simulate_trades(df, new_sig, BT_PARAMS, CE_D)
            trades_baseline.extend(trades_a)

            filtered_sig = apply_regime_filter(df, new_sig, eth_regime)
            trades_b = simulate_trades(df, filtered_sig, BT_PARAMS, CE_D)
            trades_filtered.extend(trades_b)
        except Exception as e:
            log(f"  [SKIP] {symbol}: {e}")

        if (n + 1) % 20 == 0:
            log(f"  processed {n + 1}/{len(coins)} coins...")

    log("\n" + "=" * 65)
    log("NATIJA: ETH Regime Filter on NEW System (AdvancedConfluence)")
    log("=" * 65)

    stats_a = aggregate_trades(trades_baseline)
    stats_b = aggregate_trades(trades_filtered)
    log(f"\nBASELINE     : Trades={stats_a['total_trades']:>4}  "
        f"Win%={stats_a['win_rate_pct']}  PF={stats_a['profit_factor']}  "
        f"Total Return={stats_a['total_return_pct']}%")
    log(f"+ETH_REGIME  : Trades={stats_b['total_trades']:>4}  "
        f"Win%={stats_b['win_rate_pct']}  PF={stats_b['profit_factor']}  "
        f"Total Return={stats_b['total_return_pct']}%")

    log("\nNOTE: Agar '+ETH_REGIME' wala PF aur Total Return dono BASELINE se")
    log("behtar hain, to filter is NEW system ke liye bhi faida mand hai -")
    log("bilkul jaisa Union AB par pehle se live hai. Agar kam ho jayein, to")
    log("sirf Union AB tak mehdood rakhna behtar hoga.")

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
            with open("eth_regime_new_system_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'eth_regime_new_system_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)