"""
Confirmation Candle Filter - Comparison Backtest
====================================================
Union AB (1h) par test karta hai:
    A) BASELINE - signal ke foran baad wale bar ke open par entry
       (jaisa abhi live system karta hai)
    B) +CONFIRMATION - signal ke baad wali candle (confirmation candle)
       agar bullish band ho, to us se agli candle ke open par entry.
       Agar bearish band ho, to signal reject (trade na lo).

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
from backtest_engine import compute_atr, compute_chandelier_long_stop, simulate_trades
from confirmation_candle_filter import passes_confirmation, get_confirmed_entry_bar

TEST_N_COINS = 200
SIGNAL_TIMEFRAME = "1h"

CE_A = {"period": 16, "multiplier": 3.0}
CE_B = {"period": 12, "multiplier": 3.0}

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "confirmation_candle_result.txt"


def get_combo_signal(df, combo_name):
    ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
    ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
    ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
    breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)

    if combo_name == "Ichimoku+MS":
        return ichi_sig & ms_sig, CE_A
    else:
        return ema_sig & breakout_sig, CE_B


def simulate_with_confirmation(df, signal, ce_params):
    atr = compute_atr(df, BT_PARAMS["atr_period"])
    ce_stop = compute_chandelier_long_stop(df, ce_params["period"], ce_params["multiplier"]).values
    fee = BT_PARAMS["fee_pct"] / 100
    slip = BT_PARAMS["slippage_pct"] / 100
    max_hold = BT_PARAMS["max_hold_bars"]

    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    atr_vals = atr.values
    n = len(df)

    signal_idx_arr = np.where(signal.values)[0]
    trades = []

    for i in signal_idx_arr:
        conf_idx, entry_bar = get_confirmed_entry_bar(i, n)
        if conf_idx is None or np.isnan(atr_vals[i]) or np.isnan(ce_stop[i]):
            continue

        if not passes_confirmation(o[conf_idx], h[conf_idx], l[conf_idx], c[conf_idx]):
            continue

        entry_price = o[entry_bar] * (1 + slip)

        trail_stop = ce_stop[i]
        for k in range(i + 1, entry_bar + 1):
            if not np.isnan(ce_stop[k]):
                trail_stop = max(trail_stop, ce_stop[k])

        exit_price, exit_bar, exit_reason = None, None, None
        for j in range(entry_bar, min(entry_bar + max_hold, n)):
            if not np.isnan(ce_stop[j]):
                trail_stop = max(trail_stop, ce_stop[j])
            if l[j] <= trail_stop:
                exit_price, exit_bar, exit_reason = trail_stop, j, "CE_STOP"
                break

        if exit_price is None:
            last_bar = min(entry_bar + max_hold - 1, n - 1)
            exit_price, exit_bar, exit_reason = c[last_bar], last_bar, "TIME"

        exit_price = exit_price * (1 - slip)
        gross_return = (exit_price - entry_price) / entry_price
        net_return = (gross_return - 2 * fee) * 100

        trades.append({
            "return_pct": net_return,
            "exit_reason": exit_reason,
            "bars_held": exit_bar - entry_bar,
        })

    return trades


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
    log(f"Scanning {len(coins)} coins on {SIGNAL_TIMEFRAME}...\n")

    trades_baseline = {"Ichimoku+MS": [], "EMA+Breakout": []}
    trades_confirmed = {"Ichimoku+MS": [], "EMA+Breakout": []}

    for n, symbol in enumerate(coins):
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME,
                              limit=max(config.CANDLE_LIMITS.get(SIGNAL_TIMEFRAME, 500), 300))
        except Exception:
            df = None
        if df is None or len(df) < 220:
            continue

        for combo_name in ["Ichimoku+MS", "EMA+Breakout"]:
            try:
                signal, ce_params = get_combo_signal(df, combo_name)

                trades_a = simulate_trades(df, signal, BT_PARAMS, ce_params)
                trades_baseline[combo_name].extend(trades_a)

                trades_b = simulate_with_confirmation(df, signal, ce_params)
                trades_confirmed[combo_name].extend(trades_b)
            except Exception as e:
                log(f"  [SKIP] {symbol} {combo_name}: {e}")

        if (n + 1) % 20 == 0:
            log(f"  processed {n + 1}/{len(coins)} coins...")

    log("\n" + "=" * 65)
    log("NATIJA: Confirmation Candle Filter Comparison")
    log("=" * 65)

    for combo_name in ["Ichimoku+MS", "EMA+Breakout"]:
        log(f"\n--- {combo_name} ---")
        stats_a = aggregate_trades(trades_baseline[combo_name])
        stats_b = aggregate_trades(trades_confirmed[combo_name])
        log(f"  BASELINE      : Trades={stats_a['total_trades']:>4}  "
            f"Win%={stats_a['win_rate_pct']}  PF={stats_a['profit_factor']}")
        log(f"  +CONFIRMATION : Trades={stats_b['total_trades']:>4}  "
            f"Win%={stats_b['win_rate_pct']}  PF={stats_b['profit_factor']}")

    all_baseline = trades_baseline["Ichimoku+MS"] + trades_baseline["EMA+Breakout"]
    all_confirmed = trades_confirmed["Ichimoku+MS"] + trades_confirmed["EMA+Breakout"]

    log(f"\n--- UNION AB (dono combos milakar) ---")
    stats_a = aggregate_trades(all_baseline)
    stats_b = aggregate_trades(all_confirmed)
    log(f"  BASELINE      : Trades={stats_a['total_trades']:>4}  "
        f"Win%={stats_a['win_rate_pct']}  PF={stats_a['profit_factor']}  "
        f"Total Return={stats_a['total_return_pct']}%")
    log(f"  +CONFIRMATION : Trades={stats_b['total_trades']:>4}  "
        f"Win%={stats_b['win_rate_pct']}  PF={stats_b['profit_factor']}  "
        f"Total Return={stats_b['total_return_pct']}%")

    log("\nNOTE: '+CONFIRMATION' mein Trades kam honge (kuch signals reject")
    log("hote hain, aur entry bhi 1 bar der se hoti hai). Faisla PF aur")
    log("Total Return dono dekh kar karein, sirf Trades ki tadaad kam hone")
    log("ko nuksan na samjhein.")

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
            with open("confirmation_candle_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'confirmation_candle_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)