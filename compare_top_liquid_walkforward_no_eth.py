"""
Top Liquid Coins — Long-Term Walk-Forward Test, BINA ETH FILTER
======================================================================
Wohi test jaisa "compare_top_liquid_walkforward.py" mein tha, lekin
ETH Regime Filter HATA diya gaya hai - taake pata chale ke sirf
"top liquid coins" wali behtari (bina ETH filter ke) kaisa natija
deti hai, aur kya Period 2 wali khamoshi (ETH filter ki wajah se)
khatam ho jati hai.

Baqi sab kuch same hai: Chandelier Multiplier 4.5, top 60 liquid
coins, 270 din, 3 periods.

⚠️ Bhaari test - 1-2 ghante lag sakte hain.

CHALANE SE PEHLE:
    pip install ccxt pandas numpy
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import compute_atr, compute_chandelier_long_stop

DURATION_DAYS = 270
N_PERIODS = 3
N_FETCH_COINS = 300       # kitne coins tak scan (volume order mein) taake top-liquid poori tadaad mil sake
N_TOP_LIQUID = 60         # sirf inhi ko final result mein shamil karenge
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = DURATION_DAYS * 24 + 300
MIN_BARS_REQUIRED = (DURATION_DAYS - 5) * 24

CE_A = {"period": 16, "multiplier": 4.5}
CE_B = {"period": 12, "multiplier": 4.5}
ETH_EMA_PERIOD = 200

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "top_liquid_walkforward_no_eth_result.txt"


def compute_eth_regime(eth_daily_df, ema_period=200):
    ema = eth_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = eth_daily_df["close"] > ema
    return pd.Series(is_bullish.values, index=pd.to_datetime(eth_daily_df["timestamp"]).values)


def is_eth_bullish_at(eth_regime, ts):
    # ETH FILTER HATA DIYA GAYA - hamesha True (koi filter nahi)
    return True


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
        trail_stop = ce_stop[i]
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

        trades.append({"entry_time": timestamps[entry_bar], "return_pct": net_return_pct, "exit_reason": exit_reason})

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


def make_period_boundaries(start_ts, end_ts, n_periods):
    total_span = end_ts - start_ts
    step = total_span / n_periods
    boundaries = []
    for i in range(n_periods):
        p_start = start_ts + step * i
        p_end = start_ts + step * (i + 1)
        boundaries.append((p_start, p_end))
    return boundaries


def assign_trades_to_periods(trades, boundaries):
    periods = [[] for _ in boundaries]
    for t in trades:
        entry_time = pd.Timestamp(t["entry_time"])
        for idx, (p_start, p_end) in enumerate(boundaries):
            if p_start <= entry_time < p_end:
                periods[idx].append(t)
                break
            elif idx == len(boundaries) - 1 and entry_time >= p_end:
                periods[idx].append(t)
    return periods


def main():
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(str(msg))

    exchange = get_exchange()

    log(f"Live settings: Chandelier Multiplier=4.5, ETH Regime Filter=OFF (hataya gaya)")
    log(f"Sirf TOP {N_TOP_LIQUID} liquid coins, {DURATION_DAYS} din, {N_PERIODS} periods\n")

    eth_regime = None  # istemal nahi hoga (filter off hai)

    coins = get_coin_list(exchange)[:N_FETCH_COINS]
    log(f"Scanning {len(coins)} coins (volume order) - sirf pehle {N_TOP_LIQUID} 'TOP_LIQUID' shamil honge...\n")

    all_trades = []
    overall_start_ts = None
    overall_end_ts = None
    top_liquid_included = 0

    for n, symbol in enumerate(coins):
        if top_liquid_included >= N_TOP_LIQUID:
            break  # itni coins mil chuki jitni chahiye thi, aage scan ki zaroorat nahi

        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=CANDLE_LIMIT)
        except Exception:
            df = None
        if df is None or len(df) < MIN_BARS_REQUIRED:
            continue  # short history - top-liquid list mein nahi ginte, agla try karo

        top_liquid_included += 1

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

        if top_liquid_included % 10 == 0:
            log(f"  top-liquid coins mile: {top_liquid_included}/{N_TOP_LIQUID} "
                f"(scan kiye: {n+1}, trades so far: {len(all_trades)})")

    log(f"\n[INFO] Total {top_liquid_included} top-liquid coins shamil hue "
        f"(poora {DURATION_DAYS}-din history ke sath)")

    log("\n" + "=" * 65)
    log("NATIJA: Top Liquid Coins - Long-Term Walk-Forward Test")
    log("=" * 65)

    overall_stats = aggregate_trades(all_trades)
    log(f"\n--- OVERALL ({DURATION_DAYS} din, top {N_TOP_LIQUID} liquid coins) ---")
    log(f"  Trades={overall_stats['total_trades']}  Win%={overall_stats['win_rate_pct']}  "
        f"PF={overall_stats['profit_factor']}  Total Return={overall_stats['total_return_pct']}%")

    if overall_start_ts is not None and overall_end_ts is not None and all_trades:
        boundaries = make_period_boundaries(overall_start_ts, overall_end_ts, N_PERIODS)
        periods = assign_trades_to_periods(all_trades, boundaries)

        log(f"\n--- PER-PERIOD BREAKDOWN ({N_PERIODS} barabar hisse) ---")
        for idx, (p_start, p_end) in enumerate(boundaries):
            p_stats = aggregate_trades(periods[idx])
            log(f"\nPeriod {idx+1}: {p_start.date()} se {p_end.date()}")
            log(f"  Trades={p_stats['total_trades']}  Win%={p_stats['win_rate_pct']}  "
                f"PF={p_stats['profit_factor']}  Total Return={p_stats['total_return_pct']}%")

    log("\n" + "-" * 65)
    log("KAISE SAMJHEIN:")
    log("-" * 65)
    log("- Agar teeno periods ka Win% consistently 45%+ aur PF >1.5 hai,")
    log("  to 'top liquid coins' wali behtari mustaqil hai - sirf ek")
    log("  chhote acche dor ka asar nahi.")
    log("- Agar kisi period mein Trades bohot kam (jaise <15) hon, us")
    log("  period ke PF/Win% par zyada bharosa na karein - chota sample.")

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
            with open("top_liquid_walkforward_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'top_liquid_walkforward_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)
