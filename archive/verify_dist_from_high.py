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

DURATION_DAYS = 365
N_TOP_LIQUID = 100
N_FETCH_COINS = 350
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = DURATION_DAYS * 24 + 300
MIN_BARS_REQUIRED = (DURATION_DAYS - 5) * 24

CE_A = {"period": 16, "multiplier": 4.5}
CE_B = {"period": 12, "multiplier": 4.5}
ETH_EMA_PERIOD = 200
GOOD_RETURN_THRESHOLD = 2.0

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "verify_dist_from_high_result.txt"


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

    log("\n--- DIST FROM 24H HIGH: FINER BUCKETS (tasdeeq ke liye, bara sample) ---")
    dist_bins = [
        (0, 0.5, "0% - 0.5% (high ke bilkul qareeb)"),
        (0.5, 1.0, "0.5% - 1%"),
        (1.0, 2.0, "1% - 2%"),
        (2.0, 3.0, "2% - 3%"),
        (3.0, 4.5, "3% - 4.5%"),
        (4.5, 6.0, "4.5% - 6%"),
        (6.0, 10.0, "6% - 10%"),
        (10.0, 1000, "10%+ (high se bohot door)"),
    ]
    bucket_report(all_trades, "dist_from_high_pct", dist_bins, log)

    log("\n--- COMPARISON: Moujoda direction vs Ulti (reversed) direction ---")
    # Moujoda: <1% red, 1-3% neutral, >3% green
    current_red = [t for t in all_trades if t["dist_from_high_pct"] < 1.0]
    current_green = [t for t in all_trades if t["dist_from_high_pct"] > 3.0]
    log("\nMOUJODA LOGIC (High ke qareeb = RED, door = GREEN):")
    s = aggregate_trades(current_red)
    log(f"  'RED' zone (<1%)  : Trades={s['total_trades']}  Win%={s['win_rate_pct']}  PF={s['profit_factor']}")
    s = aggregate_trades(current_green)
    log(f"  'GREEN' zone (>3%): Trades={s['total_trades']}  Win%={s['win_rate_pct']}  PF={s['profit_factor']}")

    # Reversed: <1% green (close to high = momentum), >3% red (far = weak)
    reversed_green = current_red
    reversed_red = current_green
    log("\nULTI (REVERSED) LOGIC (High ke qareeb = GREEN, door = RED):")
    s = aggregate_trades(reversed_green)
    log(f"  'GREEN' zone (<1%): Trades={s['total_trades']}  Win%={s['win_rate_pct']}  PF={s['profit_factor']}")
    s = aggregate_trades(reversed_red)
    log(f"  'RED' zone (>3%)  : Trades={s['total_trades']}  Win%={s['win_rate_pct']}  PF={s['profit_factor']}")

    log("\n" + "-" * 65)
    log("KAISE SAMJHEIN:")
    log("-" * 65)
    log("- Agar 'ULTI LOGIC' mein GREEN zone ka PF, RED zone se saaf")
    log("  zyada hai (aur ye pehle chhote test se bhi match karta hai),")
    log("  to direction ulti karna sahi hai - is dorania mein bhi")
    log("  tasdeeq ho gayi.")
    log("- Bara sample (Top 100, 365 din) chhote sample (Top 60, 270")
    log("  din) se zyada qabil-e-bharosa hai.")

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
            with open("verify_dist_from_high_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'verify_dist_from_high_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)
