"""
Live System — Long-Term Walk-Forward Test
==============================================
Aapke ABHI KE live-configured Union AB system (Chandelier Multiplier
4.5, ETH Daily-EMA200 Regime Filter) ko lambe dorania (270 din, ~9
mahine) aur zyada coins (150) par test karta hai, phir poore dorania
ko 3 barabar hisson mein baant kar har hisse ka alag natija dikhata
hai — taake pata chale ke system har dor mein kaisa raha, sirf ek
"lucky" hafte ka asar to nahi.

⚠️ Ye bhaari test hai (270 din ka 1h data, 150 coins) - shayad
1-2 ghante lag sakte hain GitHub Actions par.

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
from walk_forward_splitter import make_period_boundaries, assign_trades_to_periods

DURATION_DAYS = 270
N_PERIODS = 3
TEST_N_COINS = 150
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = DURATION_DAYS * 24 + 300

CE_A = {"period": 16, "multiplier": 4.5}
CE_B = {"period": 12, "multiplier": 4.5}
ETH_EMA_PERIOD = 200

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "long_term_walkforward_result.txt"


def compute_eth_regime(eth_daily_df, ema_period=200):
    ema = eth_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = eth_daily_df["close"] > ema
    return pd.Series(is_bullish.values, index=pd.to_datetime(eth_daily_df["timestamp"]).values)


def is_eth_bullish_at(eth_regime, ts):
    valid = eth_regime[eth_regime.index <= ts]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])


def get_combo_signal(df, combo_name):
    ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
    ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
    ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
    breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)

    if combo_name == "Ichimoku+MS":
        return ichi_sig & ms_sig, CE_A
    else:
        return ema_sig & breakout_sig, CE_B


def simulate_live_system_trades(df, signal, ce_params, eth_regime):
    atr = compute_atr(df, BT_PARAMS["atr_period"])
    ce_stop = compute_chandelier_long_stop(df, ce_params["period"], ce_params["multiplier"]).values
    fee = BT_PARAMS["fee_pct"] / 100
    slip = BT_PARAMS["slippage_pct"] / 100
    max_hold = BT_PARAMS["max_hold_bars"]

    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    atr_vals = atr.values
    n = len(df)
    timestamps = df["timestamp"].values

    signal_idx_arr = np.where(signal.values)[0]
    trades = []

    for i in signal_idx_arr:
        if i + 1 >= n or np.isnan(atr_vals[i]) or np.isnan(ce_stop[i]):
            continue

        sig_ts = pd.Timestamp(timestamps[i])
        if not is_eth_bullish_at(eth_regime, sig_ts):
            continue

        entry_bar = i + 1
        entry_price = o[entry_bar] * (1 + slip)
        initial_stop = ce_stop[i]
        initial_risk_pct = (entry_price - initial_stop) / entry_price * 100

        trail_stop = initial_stop
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
        net_return_pct = (gross_return - 2 * fee) * 100
        r_multiple = net_return_pct / initial_risk_pct if initial_risk_pct > 0 else None

        trades.append({
            "entry_time": timestamps[entry_bar],
            "return_pct": net_return_pct,
            "exit_reason": exit_reason,
            "bars_held": exit_bar - entry_bar,
            "initial_risk_pct": initial_risk_pct,
            "r_multiple": r_multiple,
        })

    return trades


def aggregate_trades(all_trades):
    if not all_trades:
        return {"total_trades": 0, "win_rate_pct": None, "profit_factor": None,
                "total_return_pct": None, "avg_r_multiple": None}
    returns = pd.Series([t["return_pct"] for t in all_trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = len(wins) / len(returns) * 100
    gross_profit = wins.sum() if len(wins) else 0
    gross_loss = abs(losses.sum()) if len(losses) else 0
    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("nan")
    r_multiples = [t["r_multiple"] for t in all_trades if t.get("r_multiple") is not None]
    avg_r = np.mean(r_multiples) if r_multiples else None
    return {
        "total_trades": len(returns),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 3) if pf == pf else None,
        "total_return_pct": round(returns.sum(), 2),
        "avg_r_multiple": round(avg_r, 3) if avg_r is not None else None,
    }


def main():
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(str(msg))

    exchange = get_exchange()

    log(f"Live system settings: Chandelier Multiplier=4.5, ETH Regime Filter=ON")
    log(f"Duration: {DURATION_DAYS} din, {N_PERIODS} periods mein baant kar, {TEST_N_COINS} coins\n")

    log("ETH daily data + 200 EMA regime nikal rahe hain...")
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=DURATION_DAYS + 250)
    eth_regime = compute_eth_regime(eth_daily, ema_period=ETH_EMA_PERIOD)
    log(f"  ETH bullish regime: {eth_regime.sum()}/{len(eth_regime)} din\n")

    coins = get_coin_list(exchange)[:TEST_N_COINS]
    log(f"Scanning {len(coins)} coins on {SIGNAL_TIMEFRAME} ({DURATION_DAYS} din ka data)...\n")

    all_trades = []
    overall_start_ts = None
    overall_end_ts = None

    for n, symbol in enumerate(coins):
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=CANDLE_LIMIT)
        except Exception:
            df = None
        if df is None or len(df) < 250:
            continue

        ts_min = pd.Timestamp(df["timestamp"].iloc[0])
        ts_max = pd.Timestamp(df["timestamp"].iloc[-1])
        if overall_start_ts is None or ts_min < overall_start_ts:
            overall_start_ts = ts_min
        if overall_end_ts is None or ts_max > overall_end_ts:
            overall_end_ts = ts_max

        for combo_name in ["Ichimoku+MS", "EMA+Breakout"]:
            try:
                signal, ce_params = get_combo_signal(df, combo_name)
                trades = simulate_live_system_trades(df, signal, ce_params, eth_regime)
                all_trades.extend(trades)
            except Exception as e:
                log(f"  [SKIP] {symbol} {combo_name}: {e}")

        if (n + 1) % 20 == 0:
            log(f"  processed {n + 1}/{len(coins)} coins... (trades so far: {len(all_trades)})")

    log("\n" + "=" * 65)
    log("NATIJA: Live System Long-Term Walk-Forward Test")
    log("=" * 65)

    overall_stats = aggregate_trades(all_trades)
    log(f"\n--- OVERALL ({DURATION_DAYS} din, sab coins milakar) ---")
    log(f"  Trades={overall_stats['total_trades']}  Win%={overall_stats['win_rate_pct']}  "
        f"PF={overall_stats['profit_factor']}  Total Return={overall_stats['total_return_pct']}%  "
        f"Avg R-multiple={overall_stats['avg_r_multiple']}")

    if overall_start_ts is not None and overall_end_ts is not None and all_trades:
        boundaries = make_period_boundaries(overall_start_ts, overall_end_ts, N_PERIODS)
        periods = assign_trades_to_periods(all_trades, boundaries)

        log(f"\n--- PER-PERIOD BREAKDOWN ({N_PERIODS} barabar hisse) ---")
        for idx, (p_start, p_end) in enumerate(boundaries):
            p_stats = aggregate_trades(periods[idx])
            log(f"\nPeriod {idx+1}: {p_start.date()} se {p_end.date()}")
            log(f"  Trades={p_stats['total_trades']}  Win%={p_stats['win_rate_pct']}  "
                f"PF={p_stats['profit_factor']}  Total Return={p_stats['total_return_pct']}%  "
                f"Avg R-multiple={p_stats['avg_r_multiple']}")

    log("\n" + "-" * 65)
    log("KAISE SAMJHEIN:")
    log("-" * 65)
    log("- Agar teeno periods ka PF mustaqil taur par >1 hai, to system")
    log("  consistent hai (sirf ek acche dor ka asar nahi).")
    log("- Agar kisi period mein PF bohot kam/manfi ho, wo dikhata hai ke")
    log("  system us waqt ki market condition (jaise bear ya sideways)")
    log("  mein kamzor hai.")
    log("- Avg R-multiple: agar ye 0 se upar hai, matlab average trade")
    log("  apne risk se zyada wapas de raha hai (mustaqil taur par).")
    log("- Win% is system ki nature ke mutabiq 30-40% ke qareeb rehna")
    log("  normal hai (trend-following, chhote losses + kabhi kabhar")
    log("  bara winner) - is number ko akele 'bura' na samjhein, PF aur")
    log("  R-multiple ke sath milakar dekhein.")

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
            with open("long_term_walkforward_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'long_term_walkforward_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)