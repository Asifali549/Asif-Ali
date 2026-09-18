"""
Teen Naye Filters — Ek Hi Test Mein (Trend Quality, RS Percentile, Volatility Percentile)
================================================================================
Union AB (Top 60 Liquid + ETH Filter + RS Filter, Multiplier 4.5 -
hamara ab tak ka behtareen combination) ke signals par, teen naye
features record karte hain aur buckets mein baant kar dikhate hain:

1. TREND QUALITY (R²): Pichle 20 candles ki closing price ek seedhi
   lakeer (linear trend) ke kitni qareeb rahi (0=bilkul bikhri hui,
   1=bilkul seedhi lakeer). Idea: saaf trend = behtar signal.

2. RS PERCENTILE (apni history ke andar): Coin ka RS Ratio (khud/BTC)
   iski apni pichli 180 din ki RS history ke muqable kitne percentile
   par hai. 90+ matlab coin abhi apni sab se zyada "outperforming"
   halat ke qareeb hai.

3. VOLATILITY PERCENTILE: Coin ka abhi ka ATR, iski apni pichli 90 din
   ki ATR history ke muqable kitne percentile par hai. Bohot zyada
   (90+) matlab ghair-mamool tor par utaar-chadhaav wala waqt.

Sab TOP 60 Liquid + ETH Filter + RS Filter (existing best) ki
population par - taake sirf naye dimensions ka asar dekha ja sake.

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
N_TOP_LIQUID = 60
N_FETCH_COINS = 300
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = DURATION_DAYS * 24 + 300
MIN_BARS_REQUIRED = (DURATION_DAYS - 5) * 24

CE_A = {"period": 16, "multiplier": 4.5}
CE_B = {"period": 12, "multiplier": 4.5}
ETH_EMA_PERIOD = 200
RS_EMA_PERIOD = 50

TREND_QUALITY_LOOKBACK = 20   # kitne candles par R² nikalna hai
RS_PERCENTILE_LOOKBACK_DAYS = 180
VOL_PERCENTILE_LOOKBACK_BARS = 90 * 24  # 90 din, 1h candles mein

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "three_filters_result.txt"


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


def apply_regime_filter(df, signal, regime_series):
    filtered = signal.copy()
    for idx in signal[signal].index:
        ts = pd.Timestamp(df["timestamp"].iloc[idx])
        if not is_bullish_at(regime_series, ts):
            filtered.iloc[idx] = False
    return filtered


def compute_r_squared(prices):
    """Linear regression R² - prices ek numpy array (chronological)."""
    n = len(prices)
    if n < 3:
        return np.nan
    x = np.arange(n)
    if np.std(prices) == 0:
        return 0.0
    corr = np.corrcoef(x, prices)[0, 1]
    if np.isnan(corr):
        return 0.0
    return corr ** 2


def percentile_rank_of_last(values):
    """Array ke aakhri value ka percentile rank (0-100) poore array ke andar."""
    if len(values) < 2:
        return 50.0
    last = values[-1]
    return float((values <= last).sum()) / len(values) * 100


def simulate_with_three_features(df, signal, ce_params, eth_regime, rs_bullish, rs_ratio_series):
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
        if not is_bullish_at(eth_regime, sig_ts):
            continue
        if not is_bullish_at(rs_bullish, sig_ts):
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

        # --- Feature 1: Trend Quality (R²) ---
        tq_window = c[max(0, i - TREND_QUALITY_LOOKBACK + 1):i + 1]
        r_squared = compute_r_squared(tq_window)

        # --- Feature 2: RS Percentile (apni history mein) ---
        rs_valid = rs_ratio_series[rs_ratio_series.index <= sig_ts]
        rs_window_days = RS_PERCENTILE_LOOKBACK_DAYS
        rs_recent = rs_valid.iloc[-rs_window_days:] if len(rs_valid) > 0 else pd.Series([], dtype=float)
        rs_percentile = percentile_rank_of_last(rs_recent.values) if len(rs_recent) >= 2 else None

        # --- Feature 3: Volatility Percentile (ATR ki apni history mein) ---
        atr_window = atr_vals[max(0, i - VOL_PERCENTILE_LOOKBACK_BARS + 1):i + 1]
        atr_window_valid = atr_window[~np.isnan(atr_window)]
        vol_percentile = percentile_rank_of_last(atr_window_valid) if len(atr_window_valid) >= 2 else None

        trades.append({
            "return_pct": net_return_pct,
            "r_squared": r_squared,
            "rs_percentile": rs_percentile,
            "vol_percentile": vol_percentile,
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

    log("BTC daily data nikal rahe hain (RS ke liye)...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DURATION_DAYS + 250)

    coins = get_coin_list(exchange)[:N_FETCH_COINS]
    log(f"Scanning - sirf TOP {N_TOP_LIQUID} liquid, full-history wale...\n")

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
                trades = simulate_with_three_features(df, signal, ce_params, eth_regime, rs_bullish, rs_ratio_series)
                all_trades.extend(trades)
            except Exception as e:
                log(f"  [SKIP] {symbol} {combo_name}: {e}")

        if rank_included % 10 == 0:
            log(f"  {rank_included}/{N_TOP_LIQUID} coins mile (trades so far: {len(all_trades)})")

    log(f"\n[INFO] Total {rank_included} coins, {len(all_trades)} trades")

    log("\n" + "=" * 65)
    log("NATIJA: Teen Naye Filters (ek hi test mein, alag alag)")
    log("=" * 65)

    overall = aggregate_trades(all_trades)
    log(f"\nOVERALL (baseline population): Trades={overall['total_trades']}  "
        f"Win%={overall['win_rate_pct']}  PF={overall['profit_factor']}")

    log("\n\n### 1. TREND QUALITY (R²) ###")
    r2_bins = [
        (0, 0.3, "0.0 - 0.3 (bikhra hua)"),
        (0.3, 0.5, "0.3 - 0.5"),
        (0.5, 0.7, "0.5 - 0.7"),
        (0.7, 0.85, "0.7 - 0.85"),
        (0.85, 1.01, "0.85 - 1.0 (bilkul saaf trend)"),
    ]
    bucket_report(all_trades, "r_squared", r2_bins, log)

    log("\n\n### 2. RS PERCENTILE (apni 180-din history mein) ###")
    rs_bins = [
        (0, 50, "0 - 50 (kamzor RS)"),
        (50, 70, "50 - 70"),
        (70, 85, "70 - 85"),
        (85, 95, "85 - 95"),
        (95, 101, "95 - 100 (sab se mazboot RS)"),
    ]
    bucket_report(all_trades, "rs_percentile", rs_bins, log)

    log("\n\n### 3. VOLATILITY PERCENTILE (ATR ki apni 90-din history mein) ###")
    vol_bins = [
        (0, 25, "0 - 25 (kam volatility)"),
        (25, 50, "25 - 50"),
        (50, 75, "50 - 75"),
        (75, 90, "75 - 90"),
        (90, 101, "90 - 100 (ghair-mamool volatility)"),
    ]
    bucket_report(all_trades, "vol_percentile", vol_bins, log)

    log("\n" + "-" * 65)
    log("KAISE SAMJHEIN:")
    log("-" * 65)
    log("- Har feature ke sab se behtar bucket ka PF dekhein - agar")
    log("  wazeh farq hai, to wo filter kaam ka hai.")
    log("- Chhote sample (<15 trades) wale buckets par kam bharosa karein.")
    log("- Teeno alag alag test hue hain (ek doosre se mila kar nahi) -")
    log("  agar koi acha nikle, use alag se filter ke tor par phir test")
    log("  karna hoga.")

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
            with open("three_filters_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'three_filters_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)
