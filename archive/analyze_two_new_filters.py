"""
Do Naye Filters — 52-Week High Momentum aur VWAP Deviation
================================================================================
Union AB (Top 100 Liquid + ETH Filter + RS Filter + RS Percentile>=95,
Multiplier 4.5 - hamara ab tak ka tasdeeq-shuda behtareen combination)
ke signals par, do naye, ma'roof (established) features record karte
hain:

1. 52-WEEK HIGH MOMENTUM: Coin ki qeemat apni pichli 365-din ki
   sab se buland satah (365-day high) se kitne % door hai. Academic
   tehqeeq (George & Hwang) ke mutabiq, jo stocks apni 52-week high
   ke qareeb hon, wo aksar behtar performance dete hain.

2. VWAP DEVIATION: Coin ki qeemat, pichle 7 din ke Volume-Weighted
   Average Price (VWAP) se kitne % upar/neeche hai. Institutional
   trading mein ye "fair value" ka nishan mana jata hai.

CHALANE SE PEHLE:
    pip install ccxt pandas numpy
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import compute_atr, compute_chandelier_long_stop

DURATION_DAYS = 365
N_TOP_LIQUID = 100
N_FETCH_COINS = 350
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = DURATION_DAYS * 24 + 300
MIN_BARS_REQUIRED = (DURATION_DAYS - 5) * 24

CE_A = {"period": 16, "multiplier": 4.5}
CE_B = {"period": 12, "multiplier": 4.5}
ETH_EMA_PERIOD = 200
RS_EMA_PERIOD = 50
RS_PERCENTILE_LOOKBACK_DAYS = 180
RS_PERCENTILE_CUTOFF = 95.0

HIGH_52W_LOOKBACK_BARS = 365 * 24  # 365 din, 1h candles mein
VWAP_LOOKBACK_BARS = 7 * 24        # 7 din, 1h candles mein

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "two_new_filters_result.txt"


def compute_eth_regime(eth_daily_df, ema_period=200):
    ema = eth_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = eth_daily_df["close"] > ema
    return pd.Series(is_bullish.values, index=pd.to_datetime(eth_daily_df["timestamp"]).values)


def is_bullish_at(regime_series, ts):
    valid = regime_series[regime_series.index <= ts]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])


def compute_rs_trend_and_ratio(coin_daily_df, btc_daily_df, ema_period=50):
    coin_d = coin_daily_df[["timestamp", "close"]].rename(columns={"close": "coin_close"})
    btc_d = btc_daily_df[["timestamp", "close"]].rename(columns={"close": "btc_close"})
    merged = pd.merge_asof(
        coin_d.sort_values("timestamp"), btc_d.sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    merged["rs_ratio"] = merged["coin_close"] / merged["btc_close"].replace(0, np.nan)
    merged["rs_ema"] = merged["rs_ratio"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = merged["rs_ratio"] > merged["rs_ema"]
    bullish_series = pd.Series(is_bullish.values, index=pd.to_datetime(merged["timestamp"]).values)
    ratio_series = pd.Series(merged["rs_ratio"].values, index=pd.to_datetime(merged["timestamp"]).values)
    return bullish_series, ratio_series


def get_combo_signal(df, combo_name):
    ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
    ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
    ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
    breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)

    if combo_name == "Ichimoku+MS":
        return ichi_sig & ms_sig, CE_A
    else:
        return ema_sig & breakout_sig, CE_B


def percentile_rank_of_last(values):
    if len(values) < 2:
        return 50.0
    last = values[-1]
    return float((values <= last).sum()) / len(values) * 100


def compute_dist_from_52w_high(high_arr, close_arr, idx, lookback):
    window = high_arr[max(0, idx - lookback + 1):idx + 1]
    if len(window) == 0:
        return None
    hi = window.max()
    if hi <= 0:
        return None
    return (hi - close_arr[idx]) / hi * 100


def compute_vwap_deviation(high_arr, low_arr, close_arr, volume_arr, idx, lookback):
    start = max(0, idx - lookback + 1)
    h = high_arr[start:idx + 1]
    l = low_arr[start:idx + 1]
    c = close_arr[start:idx + 1]
    v = volume_arr[start:idx + 1]
    if len(v) == 0 or v.sum() == 0:
        return None
    typical_price = (h + l + c) / 3
    vwap = (typical_price * v).sum() / v.sum()
    if vwap <= 0:
        return None
    return (close_arr[idx] - vwap) / vwap * 100


def simulate_with_two_features(df, signal, ce_params, eth_regime, rs_bullish, rs_ratio_series):
    atr = compute_atr(df, BT_PARAMS["atr_period"])
    ce_stop = compute_chandelier_long_stop(df, ce_params["period"], ce_params["multiplier"]).values
    fee = BT_PARAMS["fee_pct"] / 100
    slip = BT_PARAMS["slippage_pct"] / 100
    max_hold = BT_PARAMS["max_hold_bars"]

    o, h, l, c, v = (df["open"].values, df["high"].values, df["low"].values,
                       df["close"].values, df["volume"].values)
    atr_vals = atr.values
    n = len(df)
    timestamps = df["timestamp"].values

    signal_idx_arr = np.where(signal.values)[0]
    trades = []

    for i in signal_idx_arr:
        if i + 1 >= n or np.isnan(atr_vals[i]) or np.isnan(ce_stop[i]):
            continue
        sig_ts = pd.Timestamp(timestamps[i])
        if not is_bullish_at(eth_regime, sig_ts):
            continue
        if not is_bullish_at(rs_bullish, sig_ts):
            continue

        rs_valid = rs_ratio_series[rs_ratio_series.index <= sig_ts]
        rs_recent = rs_valid.iloc[-RS_PERCENTILE_LOOKBACK_DAYS:] if len(rs_valid) > 0 else pd.Series([], dtype=float)
        if len(rs_recent) < 2:
            continue
        rs_pct = percentile_rank_of_last(rs_recent.values)
        if rs_pct < RS_PERCENTILE_CUTOFF:
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

        dist_52w = compute_dist_from_52w_high(h, c, i, HIGH_52W_LOOKBACK_BARS)
        vwap_dev = compute_vwap_deviation(h, l, c, v, i, VWAP_LOOKBACK_BARS)

        trades.append({
            "return_pct": net_return_pct,
            "dist_52w_high": dist_52w,
            "vwap_deviation": vwap_dev,
        })

    return trades


def aggregate_trades(trades):
    if not trades:
        return {"total_trades": 0, "win_rate_pct": None, "profit_factor": None}
    returns = pd.Series([t["return_pct"] for t in trades])
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
    }


def bucket_report(trades, feature_key, bins, log):
    for lo, hi, label in bins:
        group = [t for t in trades if t[feature_key] is not None and lo <= t[feature_key] < hi]
        stats = aggregate_trades(group)
        log(f"  {label}: Trades={stats['total_trades']}  Win%={stats['win_rate_pct']}  PF={stats['profit_factor']}")


def main():
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(str(msg))

    exchange = get_exchange()

    log("ETH daily data + 200 EMA regime nikal rahe hain...")
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=DURATION_DAYS + 250)
    eth_regime = compute_eth_regime(eth_daily, ema_period=ETH_EMA_PERIOD)

    log("BTC daily data nikal rahe hain...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DURATION_DAYS + 250)

    coins = get_coin_list(exchange)[:N_FETCH_COINS]
    log(f"Scanning - sirf TOP {N_TOP_LIQUID} liquid, full-history wale ({DURATION_DAYS} din)...\n")

    all_trades = []
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
            coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=DURATION_DAYS + 250)
            rs_bullish, rs_ratio_series = compute_rs_trend_and_ratio(coin_daily, btc_daily, ema_period=RS_EMA_PERIOD)
        except Exception as e:
            log(f"  [RS-SKIP] {symbol}: {e}")
            continue

        for combo_name in ["Ichimoku+MS", "EMA+Breakout"]:
            try:
                signal, ce_params = get_combo_signal(df, combo_name)
                trades = simulate_with_two_features(df, signal, ce_params, eth_regime, rs_bullish, rs_ratio_series)
                all_trades.extend(trades)
            except Exception as e:
                log(f"  [SKIP] {symbol} {combo_name}: {e}")

        if rank_included % 10 == 0:
            log(f"  {rank_included}/{N_TOP_LIQUID} coins mile (trades so far: {len(all_trades)})")

    log(f"\n[INFO] Total {rank_included} coins, {len(all_trades)} trades")

    log("\n" + "=" * 65)
    log("NATIJA: 52-Week High Momentum aur VWAP Deviation")
    log("=" * 65)

    overall = aggregate_trades(all_trades)
    log(f"\nOVERALL (baseline population): Trades={overall['total_trades']}  "
        f"Win%={overall['win_rate_pct']}  PF={overall['profit_factor']}")

    log("\n\n### 1. DIST FROM 365-DAY (52-WEEK) HIGH ###")
    high_bins = [
        (0, 5, "0% - 5% (52w high ke bilkul qareeb)"),
        (5, 15, "5% - 15%"),
        (15, 30, "15% - 30%"),
        (30, 50, "30% - 50%"),
        (50, 1000, "50%+ (52w high se bohot door)"),
    ]
    bucket_report(all_trades, "dist_52w_high", high_bins, log)

    log("\n\n### 2. VWAP DEVIATION (7-din weekly VWAP se) ###")
    vwap_bins = [
        (-1000, -5, "-5% se kam (VWAP se bohot neeche)"),
        (-5, -1, "-5% se -1%"),
        (-1, 1, "-1% se +1% (VWAP ke qareeb)"),
        (1, 5, "+1% se +5%"),
        (5, 1000, "+5%+ (VWAP se bohot upar)"),
    ]
    bucket_report(all_trades, "vwap_deviation", vwap_bins, log)

    log("\n" + "-" * 65)
    log("KAISE SAMJHEIN:")
    log("-" * 65)
    log("- 52-Week High: agar '0-5%' bucket sab se behtar PF de, to")
    log("  academic momentum-effect ki tasdeeq hui.")
    log("- VWAP: dekhein kaunsi range (upar/neeche/qareeb) behtar hai.")
    log("- Chhote sample (<15 trades) wale buckets par kam bharosa karein.")

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
            with open("two_new_filters_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'two_new_filters_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)
