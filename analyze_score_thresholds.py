"""
Score Thresholds — Data-Driven Analysis (Vol, Chg, Dist-from-24h-High)
================================================================================
Union AB ke historical signals (Top 60 Liquid, ETH Filter, live
settings) par, signal ke waqt ye 3 features record karta hai:
    - Vol 1h (us waqt ka 1h volume, pichle 20 candles ki average se
      kitna zyada/kam)
    - Chg 1h (signal candle ka % change)
    - Dist from 24h High (signal ke waqt price 24h high se kitna door)

Phir har feature ko buckets mein baant kar dekhta hai ke asal
(GOOD vs BAD outcome) trades kis range mein zyada aate hain - taake
maloom ho ke abhi ke thresholds (Vol>=1.2x, Dist>3%) sahi jagah par
hain ya inhein badalna chahiye.

⚠️ OrderBook Bid/Ask aur Funding Rate is tarah test NAHI ho sakte -
inki tareekhi (historical) value available nahi hoti hamare data
source se.

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
GOOD_RETURN_THRESHOLD = 2.0

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "score_thresholds_result.txt"


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


def compute_features_at_signal(df_values, signal_idx):
    """Vol ratio, Chg%, aur Dist-from-24h-high - signal candle par."""
    o, h, l, c, v = (df_values["open"], df_values["high"], df_values["low"],
                      df_values["close"], df_values["volume"])

    # Vol ratio: signal candle ka volume vs pichle 20 candles ki average
    vol_window = v[max(0, signal_idx - 20):signal_idx]
    avg_vol = vol_window.mean() if len(vol_window) > 0 else v[signal_idx]
    vol_ratio = v[signal_idx] / avg_vol if avg_vol > 0 else 1.0

    # Chg%: signal candle ka apna % move
    chg_pct = (c[signal_idx] - o[signal_idx]) / o[signal_idx] * 100 if o[signal_idx] > 0 else 0.0

    # Dist from 24h high: pichle 24 candles (1h TF par 24h) ka high, us se kitna neeche hain
    window_24h = h[max(0, signal_idx - 24):signal_idx + 1]
    high_24h = window_24h.max() if len(window_24h) > 0 else h[signal_idx]
    dist_from_high_pct = (high_24h - c[signal_idx]) / high_24h * 100 if high_24h > 0 else 0.0

    return {"vol_ratio": vol_ratio, "chg_pct": chg_pct, "dist_from_high_pct": dist_from_high_pct}


def simulate_with_score_features(df, signal, ce_params, eth_regime):
    atr = compute_atr(df, BT_PARAMS["atr_period"])
    ce_stop = compute_chandelier_long_stop(df, ce_params["period"], ce_params["multiplier"]).values
    fee = BT_PARAMS["fee_pct"] / 100
    slip = BT_PARAMS["slippage_pct"] / 100
    max_hold = BT_PARAMS["max_hold_bars"]

    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    atr_vals = atr.values
    n = len(df)
    timestamps = df["timestamp"].values

    df_values = {"open": o, "high": h, "low": l, "close": c, "volume": df["volume"].values}

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

        features = compute_features_at_signal(df_values, i)
        trades.append({"return_pct": net_return_pct, **features})

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
        group = [t for t in trades if lo <= t[feature_key] < hi]
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

        for combo_name in ["Ichimoku+MS", "EMA+Breakout"]:
            try:
                signal, ce_params = get_combo_signal(df, combo_name)
                trades = simulate_with_score_features(df, signal, ce_params, eth_regime)
                all_trades.extend(trades)
            except Exception as e:
                log(f"  [SKIP] {symbol} {combo_name}: {e}")

        if rank_included % 10 == 0:
            log(f"  {rank_included}/{N_TOP_LIQUID} coins mile (trades so far: {len(all_trades)})")

    log(f"\n[INFO] Total {rank_included} coins, {len(all_trades)} trades")

    log("\n" + "=" * 65)
    log("NATIJA: Score Thresholds Data-Driven Analysis")
    log("=" * 65)

    overall = aggregate_trades(all_trades)
    log(f"\nOVERALL: Trades={overall['total_trades']}  Win%={overall['win_rate_pct']}  PF={overall['profit_factor']}")

    log("\n--- VOL RATIO ke buckets (abhi ka threshold: >=1.2 green, <0.8 red) ---")
    vol_bins = [
        (0, 0.8, "0.0x - 0.8x (abhi RED)"),
        (0.8, 1.2, "0.8x - 1.2x (abhi NEUTRAL)"),
        (1.2, 2.0, "1.2x - 2.0x (abhi GREEN)"),
        (2.0, 5.0, "2.0x - 5.0x (abhi GREEN)"),
        (5.0, 1000, "5.0x+ (abhi GREEN)"),
    ]
    bucket_report(all_trades, "vol_ratio", vol_bins, log)

    log("\n--- CHG% ke buckets (abhi ka threshold: >0 green, <0 red) ---")
    chg_bins = [
        (-1000, -2, "-2% se kam (bara manfi)"),
        (-2, 0, "-2% se 0% (chhota manfi, abhi RED)"),
        (0, 2, "0% se 2% (chhota musbat, abhi GREEN)"),
        (2, 5, "2% se 5%"),
        (5, 1000, "5%+ (bara musbat)"),
    ]
    bucket_report(all_trades, "chg_pct", chg_bins, log)

    log("\n--- DIST FROM 24H HIGH ke buckets (abhi ka threshold: <1% red, >3% green) ---")
    dist_bins = [
        (0, 1.0, "0% - 1% (abhi RED, high ke bilkul qareeb)"),
        (1.0, 3.0, "1% - 3% (abhi NEUTRAL)"),
        (3.0, 6.0, "3% - 6% (abhi GREEN)"),
        (6.0, 15.0, "6% - 15%"),
        (15.0, 1000, "15%+ (high se bohot door)"),
    ]
    bucket_report(all_trades, "dist_from_high_pct", dist_bins, log)

    log("\n" + "-" * 65)
    log("KAISE SAMJHEIN:")
    log("-" * 65)
    log("- Har feature ke buckets ka PF dekhein. Agar 'abhi GREEN' wale")
    log("  bucket ka PF waqai baqi buckets se zyada hai, to threshold")
    log("  sahi jagah par hai.")
    log("- Agar koi 'abhi RED' bucket ka PF acha nikle, ya 'abhi GREEN'")
    log("  ka PF kamzor nikle, to us threshold ko badalna faida mand")
    log("  ho sakta hai.")
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
            with open("score_thresholds_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'score_thresholds_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)
