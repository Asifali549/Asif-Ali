"""
Per-Coin 4h Trend Alignment Filter - Comparison Backtest
=============================================================
Union AB (1h) par test karta hai:
    A) BASELINE - koi extra trend filter nahi
    B) +COIN_4H_TREND - signal sirf tab valid jab USI COIN ka 4h
       close apni 4h EMA(50) se upar ho (BTC nahi, coin khud)

CHALANE SE PEHLE:
    pip install ccxt pandas numpy

NOTE: Isay repo ke usi folder mein rakhein jahan strategies.py,
backtest_engine.py, config.py, data_fetcher.py maujood hain.
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import simulate_trades
from coin_trend_filter import compute_ema_trend_series, is_trend_aligned

TEST_N_COINS = 150
SIGNAL_TIMEFRAME = "1h"
EMA_PERIOD_4H = 50

CE_A = {"period": 16, "multiplier": 3.0}
CE_B = {"period": 12, "multiplier": 3.0}

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "coin_trend_filter_result.txt"


def get_combo_signal(df, combo_name):
    ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
    ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
    ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
    breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)

    if combo_name == "Ichimoku+MS":
        return ichi_sig & ms_sig, CE_A
    else:
        return ema_sig & breakout_sig, CE_B


def apply_trend_filter(df, signal, trend_series):
    filtered = signal.copy()
    for idx in signal[signal].index:
        ts = pd.Timestamp(df["timestamp"].iloc[idx])
        if not is_trend_aligned(trend_series, ts):
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
    coins = get_coin_list(exchange)[:TEST_N_COINS]
    log(f"Scanning {len(coins)} coins on {SIGNAL_TIMEFRAME} (+ har coin ka apna 4h trend)...\n")

    trades_baseline = {"Ichimoku+MS": [], "EMA+Breakout": []}
    trades_filtered = {"Ichimoku+MS": [], "EMA+Breakout": []}

    for n, symbol in enumerate(coins):
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME,
                              limit=max(config.CANDLE_LIMITS.get(SIGNAL_TIMEFRAME, 500), 300))
        except Exception:
            df = None
        if df is None or len(df) < 220:
            continue

        try:
            df_4h = fetch_ohlcv(exchange, symbol, "4h", limit=300)
            trend_series = compute_ema_trend_series(df_4h, ema_period=EMA_PERIOD_4H) if df_4h is not None and len(df_4h) > EMA_PERIOD_4H else None
        except Exception:
            trend_series = None

        for combo_name in ["Ichimoku+MS", "EMA+Breakout"]:
            try:
                signal, ce_params = get_combo_signal(df, combo_name)

                trades_a = simulate_trades(df, signal, BT_PARAMS, ce_params)
                trades_baseline[combo_name].extend(trades_a)

                filtered_signal = apply_trend_filter(df, signal, trend_series) if trend_series is not None else signal
                trades_b = simulate_trades(df, filtered_signal, BT_PARAMS, ce_params)
                trades_filtered[combo_name].extend(trades_b)
            except Exception as e:
                log(f"  [SKIP] {symbol} {combo_name}: {e}")

        if (n + 1) % 20 == 0:
            log(f"  processed {n + 1}/{len(coins)} coins...")

    log("\n" + "=" * 65)
    log("NATIJA: Per-Coin 4h Trend Alignment Filter Comparison")
    log("=" * 65)

    for combo_name in ["Ichimoku+MS", "EMA+Breakout"]:
        log(f"\n--- {combo_name} ---")
        stats_a = aggregate_trades(trades_baseline[combo_name])
        stats_b = aggregate_trades(trades_filtered[combo_name])
        log(f"  BASELINE          : Trades={stats_a['total_trades']:>4}  "
            f"Win%={stats_a['win_rate_pct']}  PF={stats_a['profit_factor']}")
        log(f"  +COIN_4H_TREND    : Trades={stats_b['total_trades']:>4}  "
            f"Win%={stats_b['win_rate_pct']}  PF={stats_b['profit_factor']}")

    all_baseline = trades_baseline["Ichimoku+MS"] + trades_baseline["EMA+Breakout"]
    all_filtered = trades_filtered["Ichimoku+MS"] + trades_filtered["EMA+Breakout"]

    log(f"\n--- UNION AB (dono combos milakar) ---")
    stats_a = aggregate_trades(all_baseline)
    stats_b = aggregate_trades(all_filtered)
    log(f"  BASELINE          : Trades={stats_a['total_trades']:>4}  "
        f"Win%={stats_a['win_rate_pct']}  PF={stats_a['profit_factor']}  "
        f"Total Return={stats_a['total_return_pct']}%")
    log(f"  +COIN_4H_TREND    : Trades={stats_b['total_trades']:>4}  "
        f"Win%={stats_b['win_rate_pct']}  PF={stats_b['profit_factor']}  "
        f"Total Return={stats_b['total_return_pct']}%")

    log("\nNOTE: Agar 'COIN_4H_TREND' wala PF zyada hai, to filter madad kar")
    log("raha hai. Trades ki tadaad kam hone ko akele nuksan na samjhein -")
    log("PF aur Total Return dono dekh kar faisla karein.")

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
            with open("coin_trend_filter_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'coin_trend_filter_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1) 