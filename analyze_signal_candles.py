"""
Signal Candle Pattern Analysis
==================================
Union AB (1h) ke tamam historical signals par:
  1. Har signal ki candle features nikalta hai (body%, wick%, volume)
  2. Entry ke foran baad wali candle(s) ka reaction record karta hai
  3. Signals ko 2 groups mein baanta hai:
       GOOD   = jo trade +2% ya zyada return de gaya (chahe kitne bhi
                bars mein)
       QUICK_LOSS = jo entry ke 1-2 candles ke andar hi CE_STOP (SL)
                    laga aur bara loss hua
  4. Dono groups ki average features compare karta hai - taake pata
     chale konsi cheez GOOD signals mein alag thi

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
from backtest_engine import compute_atr, compute_chandelier_long_stop
from signal_candle_features import extract_signal_features

TEST_N_COINS = 200
SIGNAL_TIMEFRAME = "1h"

CE_A = {"period": 16, "multiplier": 3.0}
CE_B = {"period": 12, "multiplier": 3.0}

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

GOOD_RETURN_THRESHOLD = 2.0
QUICK_LOSS_MAX_BARS = 2

OUTPUT_FILE = "signal_candle_analysis_result.txt"


def get_combo_signal(df, combo_name):
    ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
    ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
    ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
    breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)

    if combo_name == "Ichimoku+MS":
        return ichi_sig & ms_sig, CE_A
    else:
        return ema_sig & breakout_sig, CE_B


def simulate_with_features(df, signal, ce_params):
    atr = compute_atr(df, BT_PARAMS["atr_period"])
    ce_stop = compute_chandelier_long_stop(df, ce_params["period"], ce_params["multiplier"]).values
    fee = BT_PARAMS["fee_pct"] / 100
    slip = BT_PARAMS["slippage_pct"] / 100
    max_hold = BT_PARAMS["max_hold_bars"]

    close = df["close"].values
    low = df["low"].values
    atr_vals = atr.values
    n = len(df)

    df_values = {
        "open": df["open"].values, "high": df["high"].values,
        "low": df["low"].values, "close": df["close"].values,
        "volume": df["volume"].values,
    }

    signal_idx_arr = np.where(signal.values)[0]
    results = []

    for i in signal_idx_arr:
        if i + 1 >= n or np.isnan(atr_vals[i]) or np.isnan(ce_stop[i]):
            continue

        entry_bar = i + 1
        entry_price = df["open"].values[entry_bar] * (1 + slip)

        trail_stop = ce_stop[i]
        exit_price, exit_bar, exit_reason = None, None, None

        for j in range(entry_bar, min(entry_bar + max_hold, n)):
            if not np.isnan(ce_stop[j]):
                trail_stop = max(trail_stop, ce_stop[j])
            if low[j] <= trail_stop:
                exit_price, exit_bar, exit_reason = trail_stop, j, "CE_STOP"
                break

        if exit_price is None:
            last_bar = min(entry_bar + max_hold - 1, n - 1)
            exit_price, exit_bar, exit_reason = close[last_bar], last_bar, "TIME"

        exit_price = exit_price * (1 - slip)
        gross_return = (exit_price - entry_price) / entry_price
        net_return = (gross_return - 2 * fee) * 100
        bars_held = exit_bar - entry_bar

        features = extract_signal_features(df_values, signal_idx=i, entry_idx=entry_bar,
                                            atr_at_signal=atr_vals[i], lookahead=2)

        results.append({
            "return_pct": net_return,
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            **features,
        })

    return results


def classify(trade):
    if trade["return_pct"] >= GOOD_RETURN_THRESHOLD:
        return "GOOD"
    if trade["exit_reason"] == "CE_STOP" and trade["bars_held"] <= QUICK_LOSS_MAX_BARS:
        return "QUICK_LOSS"
    return "OTHER"


NUMERIC_FEATURES = [
    "signal_body_pct", "signal_upper_wick_pct", "signal_volume_ratio",
    "next_candle_move_pct", "next_candle_move_atr", "worst_move_pct_in_lookahead",
]


def summarize_group(trades, label, log):
    if not trades:
        log(f"{label}: koi trade nahi mila")
        return
    log(f"\n{label} (n={len(trades)}):")
    for feat in NUMERIC_FEATURES:
        vals = [t[feat] for t in trades if t.get(feat) is not None]
        if not vals:
            continue
        log(f"  {feat:<28}: avg={np.mean(vals):.3f}  median={np.median(vals):.3f}")
    bearish_pct = sum(1 for t in trades if t.get("next_candle_bearish")) / len(trades) * 100
    log(f"  {'next_candle_bearish_pct':<28}: {bearish_pct:.1f}%")


def main():
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(str(msg))

    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TEST_N_COINS]
    log(f"Scanning {len(coins)} coins on {SIGNAL_TIMEFRAME}...\n")

    all_trades = []

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
                trades = simulate_with_features(df, signal, ce_params)
                all_trades.extend(trades)
            except Exception as e:
                log(f"  [SKIP] {symbol} {combo_name}: {e}")

        if (n + 1) % 20 == 0:
            log(f"  processed {n + 1}/{len(coins)} coins...")

    log("\n" + "=" * 65)
    log("NATIJA: Signal Candle Pattern Analysis")
    log("=" * 65)
    log(f"Total signals processed: {len(all_trades)}")

    good = [t for t in all_trades if classify(t) == "GOOD"]
    quick_loss = [t for t in all_trades if classify(t) == "QUICK_LOSS"]
    other = [t for t in all_trades if classify(t) == "OTHER"]

    log(f"GOOD (return >= {GOOD_RETURN_THRESHOLD}%): {len(good)}")
    log(f"QUICK_LOSS (SL within {QUICK_LOSS_MAX_BARS} bars): {len(quick_loss)}")
    log(f"OTHER (baqi sab): {len(other)}")

    summarize_group(good, "GOOD SIGNALS", log)
    summarize_group(quick_loss, "QUICK_LOSS SIGNALS", log)

    log("\n" + "-" * 65)
    log("TAJZIYA (Interpretation):")
    log("-" * 65)
    if good and quick_loss:
        good_bearish_pct = sum(1 for t in good if t.get("next_candle_bearish")) / len(good) * 100
        ql_bearish_pct = sum(1 for t in quick_loss if t.get("next_candle_bearish")) / len(quick_loss) * 100
        log(f"GOOD signals mein entry ke baad wali candle bearish thi: {good_bearish_pct:.1f}% dafa")
        log(f"QUICK_LOSS signals mein entry ke baad wali candle bearish thi: {ql_bearish_pct:.1f}% dafa")

        good_vol = np.mean([t["signal_volume_ratio"] for t in good])
        ql_vol = np.mean([t["signal_volume_ratio"] for t in quick_loss])
        log(f"\nGOOD signals ka average signal-candle volume ratio: {good_vol:.2f}x")
        log(f"QUICK_LOSS signals ka average signal-candle volume ratio: {ql_vol:.2f}x")

        good_body = np.mean([t["signal_body_pct"] for t in good])
        ql_body = np.mean([t["signal_body_pct"] for t in quick_loss])
        log(f"\nGOOD signals ki signal-candle body%: {good_body:.1f}%")
        log(f"QUICK_LOSS signals ki signal-candle body%: {ql_body:.1f}%")

    log("\nNOTE: Ye sirf correlation dikhata hai, guarantee nahi. Agar koi")
    log("wazeh farq (jaise 15%+ ka gap) kisi feature mein nazar aaye, to")
    log("wahi ek naya filter banane ka acha candidate hai - phir usay")
    log("alag se test karna hoga (jaise humne BTC regime filter kiya tha).")

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
            with open("signal_candle_analysis_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'signal_candle_analysis_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass