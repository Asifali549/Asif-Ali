"""
Liquidity Tiers Comparison — NEW System (AdvancedConfluence), ETH Filter ON
================================================================================
Yehi liquidity-tiers test (TOP 60/100/150/200), lekin Union AB ki
jagah NEW_AdvancedConfluence_v1 system (BOS/CHoCH + Score>=6) par,
ETH Regime Filter ke sath. Taake pata chale ke liquidity-restriction
wali behtari sirf Union AB tak mehdood hai, ya NEW system par bhi
kaam karti hai.

⚠️ Bhaari test - 2-3 ghante lag sakte hain.

CHALANE SE PEHLE:
    pip install ccxt pandas numpy
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown
from backtest_engine import compute_atr, compute_chandelier_long_stop
from confluence_engine import compute_confluence, DEFAULT_PARAMS as CONF_PARAMS

DURATION_DAYS = 270
N_PERIODS = 3
TIERS = [60, 100, 150, 200]
N_FETCH_COINS = 350
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = DURATION_DAYS * 24 + 300
MIN_BARS_REQUIRED = (DURATION_DAYS - 5) * 24

CE_D = {"period": 16, "multiplier": 3.0}  # NEW system ka apna chandelier (Union AB se alag)
ETH_EMA_PERIOD = 200

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "liquidity_tiers_new_system_result.txt"


def compute_eth_regime(eth_daily_df, ema_period=200):
    ema = eth_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = eth_daily_df["close"] > ema
    return pd.Series(is_bullish.values, index=pd.to_datetime(eth_daily_df["timestamp"]).values)


def is_eth_bullish_at(eth_regime, ts):
    valid = eth_regime[eth_regime.index <= ts]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])


def get_new_system_signal(df, btc_daily):
    result_new = compute_confluence(df, btc_daily, CONF_PARAMS, usdt_d_weak=None)
    structure_signal = result_new["bos"] | result_new["choch"]
    new_sig = apply_cooldown(structure_signal & (result_new["score"] >= 6), config.SIGNAL_COOLDOWN_BARS)
    return new_sig


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

        trades.append({"entry_time": timestamps[entry_bar], "return_pct": net_return_pct})

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

    log(f"Live settings: Chandelier Multiplier=4.5, ETH Regime Filter=ON")
    log(f"Tiers: {TIERS}, {DURATION_DAYS} din, {N_PERIODS} periods\n")

    log("ETH daily data + 200 EMA regime nikal rahe hain...")
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=DURATION_DAYS + 250)
    eth_regime = compute_eth_regime(eth_daily, ema_period=ETH_EMA_PERIOD)
    log(f"  ETH bullish regime: {eth_regime.sum()}/{len(eth_regime)} din\n")

    log("BTC daily data nikal rahe hain (NEW system ke apne internal score ke liye)...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DURATION_DAYS + 250)

    coins = get_coin_list(exchange)[:N_FETCH_COINS]
    max_tier = max(TIERS)
    log(f"Scanning {len(coins)} coins (volume order) - pehle {max_tier} full-history "
        f"coins tak scan honge...\n")

    # har tier ke liye alag trades list
    trades_by_tier = {tier: [] for tier in TIERS}
    overall_start_ts = None
    overall_end_ts = None
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

        rank_included += 1  # is coin ki "liquidity rank" (1-based)

        ts_min = pd.Timestamp(df["timestamp"].iloc[0])
        ts_max = pd.Timestamp(df["timestamp"].iloc[-1])
        if overall_start_ts is None or ts_min < overall_start_ts:
            overall_start_ts = ts_min
        if overall_end_ts is None or ts_max > overall_end_ts:
            overall_end_ts = ts_max

        coin_trades = []
        try:
            signal = get_new_system_signal(df, btc_daily)
            coin_trades = simulate_live_system_trades(df, signal, CE_D, eth_regime)
        except Exception as e:
            log(f"  [SKIP] {symbol}: {e}")

        # is coin ke trades un sab tiers mein daalo jinme ye rank fit baithti hai
        for tier in TIERS:
            if rank_included <= tier:
                trades_by_tier[tier].extend(coin_trades)

        if rank_included % 20 == 0:
            log(f"  rank {rank_included}/{max_tier} tak coins mil chuke (scan kiye: {n+1})")

    log(f"\n[INFO] Total {rank_included} full-history coins mile aur rank ke hisab se tiers mein baante gaye")

    log("\n" + "=" * 65)
    log("NATIJA: Liquidity Tiers Comparison (ETH Filter ON)")
    log("=" * 65)

    boundaries = None
    if overall_start_ts is not None and overall_end_ts is not None:
        boundaries = make_period_boundaries(overall_start_ts, overall_end_ts, N_PERIODS)

    for tier in TIERS:
        tier_trades = trades_by_tier[tier]
        overall_stats = aggregate_trades(tier_trades)
        log(f"\n--- TOP {tier} LIQUID COINS ---")
        log(f"  OVERALL: Trades={overall_stats['total_trades']}  Win%={overall_stats['win_rate_pct']}  "
            f"PF={overall_stats['profit_factor']}  Total Return={overall_stats['total_return_pct']}%")

        if boundaries and tier_trades:
            periods = assign_trades_to_periods(tier_trades, boundaries)
            for idx, (p_start, p_end) in enumerate(boundaries):
                p_stats = aggregate_trades(periods[idx])
                log(f"  Period {idx+1} ({p_start.date()} se {p_end.date()}): "
                    f"Trades={p_stats['total_trades']}  Win%={p_stats['win_rate_pct']}  "
                    f"PF={p_stats['profit_factor']}")

    log("\n" + "-" * 65)
    log("KAISE SAMJHEIN:")
    log("-" * 65)
    log("- Jis tier ka Win% aur PF sab se zyada consistent (teeno periods")
    log("  mein achha) ho, wahi liquidity had behtareen hai.")
    log("- Zyada coins (jaise 200) zyada trades denge (roz-marra trading")
    log("  ke liye behtar), lekin shayad Win%/PF thoda kam ho (kam liquid")
    log("  coins shamil hone ki wajah se) - ye trade-off dekhna hai.")
    log("- Agar kisi tier ke kisi period mein Trades bohot kam (<15) hon,")
    log("  us number par zyada bharosa na karein.")

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
            with open("liquidity_tiers_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'liquidity_tiers_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)
