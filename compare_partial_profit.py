"""
Partial Profit Filter - Comparison Backtest
================================================
Union AB (1h) par test karta hai:
    A) BASELINE - poori position chandelier trailing stop tak (jaisa
       abhi live system karta hai)
    B) +PARTIAL_PROFIT - jab price +1.5% tak pahunche, aadhi position
       wahan band, baqi aadhi ka stop breakeven par, phir trailing

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
from partial_profit_filter import simulate_partial_profit_trade

TEST_N_COINS = 200
SIGNAL_TIMEFRAME = "1h"
PARTIAL_TARGET_PCT = 1.5

CE_A = {"period": 16, "multiplier": 3.0}
CE_B = {"period": 12, "multiplier": 3.0}

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "partial_profit_result.txt"


def get_combo_signal(df, combo_name):
    ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
    ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
    ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
    breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)

    if combo_name == "Ichimoku+MS":
        return ichi_sig & ms_sig, CE_A
    else:
        return ema_sig & breakout_sig, CE_B


def simulate_partial_for_signals(df, signal, ce_params):
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
        entry_bar = i + 1
        if entry_bar >= n or np.isnan(atr_vals[i]) or np.isnan(ce_stop[i]):
            continue

        entry_price = o[entry_bar] * (1 + slip)
        initial_stop = ce_stop[i]

        result = simulate_partial_profit_trade(
            o, h, l, c, entry_bar=entry_bar, entry_price=entry_price,
            initial_stop=initial_stop, ce_stops=ce_stop, max_hold=max_hold,
            partial_target_pct=PARTIAL_TARGET_PCT, fee=fee, slip=slip,
        )
        trades.append(result)

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
    log(f"Scanning {len(coins)} coins on {SIGNAL_TIMEFRAME}... (partial target = {PARTIAL_TARGET_PCT}%)\n")

    trades_baseline = {"Ichimoku+MS": [], "EMA+Breakout": []}
    trades_partial = {"Ichimoku+MS": [], "EMA+Breakout": []}

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

                trades_b = simulate_partial_for_signals(df, signal, ce_params)
                trades_partial[combo_name].extend(trades_b)
            except Exception as e:
                log(f"  [SKIP] {symbol} {combo_name}: {e}")

        if (n + 1) % 20 == 0:
            log(f"  processed {n + 1}/{len(coins)} coins...")

    log("\n" + "=" * 65)
    log("NATIJA: Partial Profit Filter Comparison")
    log("=" * 65)

    for combo_name in ["Ichimoku+MS", "EMA+Breakout"]:
        log(f"\n--- {combo_name} ---")
        stats_a = aggregate_trades(trades_baseline[combo_name])
        stats_b = aggregate_trades(trades_partial[combo_name])
        log(f"  BASELINE       : Trades={stats_a['total_trades']:>4}  "
            f"Win%={stats_a['win_rate_pct']}  PF={stats_a['profit_factor']}")
        log(f"  +PARTIAL_PROFIT: Trades={stats_b['total_trades']:>4}  "
            f"Win%={stats_b['win_rate_pct']}  PF={stats_b['profit_factor']}")

    all_baseline = trades_baseline["Ichimoku+MS"] + trades_baseline["EMA+Breakout"]
    all_partial = trades_partial["Ichimoku+MS"] + trades_partial["EMA+Breakout"]

    log(f"\n--- UNION AB (dono combos milakar) ---")
    stats_a = aggregate_trades(all_baseline)
    stats_b = aggregate_trades(all_partial)
    log(f"  BASELINE       : Trades={stats_a['total_trades']:>4}  "
        f"Win%={stats_a['win_rate_pct']}  PF={stats_a['profit_factor']}  "
        f"Total Return={stats_a['total_return_pct']}%")
    log(f"  +PARTIAL_PROFIT: Trades={stats_b['total_trades']:>4}  "
        f"Win%={stats_b['win_rate_pct']}  PF={stats_b['profit_factor']}  "
        f"Total Return={stats_b['total_return_pct']}%")

    partial_taken_count = sum(1 for t in all_partial if t.get("partial_taken"))
    log(f"\nPartial target hit hua: {partial_taken_count}/{len(all_partial)} trades mein "
        f"({partial_taken_count/len(all_partial)*100:.1f}%)")

    log("\nNOTE: Trades ki tadaad dono taraf barabar honi chahiye (yahan signal")
    log("reject nahi hota, sirf exit tareeqa badalta hai). Faisla Win% aur PF")
    log("dono dekh kar karein.")

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
            with open("partial_profit_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'partial_profit_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)