"""
Quality Score — Applied to Union AB (Top 60 Liquid + ETH Filter, LIVE settings)
================================================================================
Quality Score khud koi signal nahi banata (jaisa user ne bataya - ye
sirf RANKING ke liye hai). Is liye hum isay Union AB ke asal, live
tasdeeq-shuda signals (Top 60 Liquid + ETH Filter + Multiplier 4.5)
par LAGA kar dekhte hain: kya zyada Quality Score wale signals waqai
behtar Win%/PF dete hain (jaisa dawa hai - kamzor magar musbat asar,
PF 1.15->1.24)?

Score formula bilkul wohi hai jo diya gaya (percentile rank based,
har coin ke apne history ke andar relative).

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

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "quality_score_result.txt"

# ---- Quality Score params (jaisa diya gaya) ----
QS_PARAMS = {
    "vol_avg_period": 20,
    "atr_period": 14,
    "resistance_lookback": 300,
    "resistance_exclude_recent": 5,
    "adx_period": 14,
}


def compute_adx(df, period=14):
    """Standard Wilder's ADX (confluence_engine.py se, self-contained)."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr_e = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_e
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_e

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_val


def compute_quality_score_series(df):
    """Poore df ke liye quality score series (0-100) nikalta hai (percentile-based)."""
    close, high, low, volume, open_ = df["close"], df["high"], df["low"], df["volume"], df["open"]

    vol_ma = volume.rolling(QS_PARAMS["vol_avg_period"]).mean()
    rvol = volume / vol_ma.replace(0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(QS_PARAMS["atr_period"]).mean()
    candle_strength = (close - open_).abs() / atr_val.replace(0, np.nan)

    recent_high = high.shift(QS_PARAMS["resistance_exclude_recent"]).rolling(QS_PARAMS["resistance_lookback"]).max()
    headroom_pct = (recent_high - close) / close * 100

    adx_val = compute_adx(df, QS_PARAMS["adx_period"])

    metrics = pd.DataFrame({"rvol": rvol, "candle_strength": candle_strength, "headroom_pct": headroom_pct, "adx": adx_val})
    for col in ["rvol", "candle_strength", "headroom_pct", "adx"]:
        metrics[col + "_pct"] = metrics[col].rank(pct=True) * 100

    score = metrics[["rvol_pct", "candle_strength_pct", "headroom_pct_pct", "adx_pct"]].mean(axis=1)
    return score


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


def simulate_with_quality_score(df, signal, ce_params, eth_regime, quality_scores):
    atr = compute_atr(df, BT_PARAMS["atr_period"])
    ce_stop = compute_chandelier_long_stop(df, ce_params["period"], ce_params["multiplier"]).values
    fee = BT_PARAMS["fee_pct"] / 100
    slip = BT_PARAMS["slippage_pct"] / 100
    max_hold = BT_PARAMS["max_hold_bars"]

    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    atr_vals = atr.values
    n = len(df)
    timestamps = df["timestamp"].values
    qs_vals = quality_scores.values

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

        qscore = qs_vals[i]
        trades.append({"return_pct": net_return_pct, "quality_score": qscore if not np.isnan(qscore) else None})

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
    log(f"Scanning - sirf TOP {N_TOP_LIQUID} liquid, full-history wale (Union AB, live settings)...\n")

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
            quality_scores = compute_quality_score_series(df)
            for combo_name in ["Ichimoku+MS", "EMA+Breakout"]:
                signal, ce_params = get_combo_signal(df, combo_name)
                trades = simulate_with_quality_score(df, signal, ce_params, eth_regime, quality_scores)
                all_trades.extend(trades)
        except Exception as e:
            log(f"  [SKIP] {symbol}: {e}")

        if rank_included % 10 == 0:
            log(f"  {rank_included}/{N_TOP_LIQUID} coins mile (trades so far: {len(all_trades)})")

    log(f"\n[INFO] Total {rank_included} coins, {len(all_trades)} trades")

    log("\n" + "=" * 65)
    log("NATIJA: Quality Score Applied to Union AB (Top 60 + ETH)")
    log("=" * 65)

    overall = aggregate_trades(all_trades)
    log(f"\nOVERALL: Trades={overall['total_trades']}  Win%={overall['win_rate_pct']}  PF={overall['profit_factor']}")

    log("\n--- QUALITY SCORE ke buckets ---")
    bins = [
        (0, 30, "0-30 (kam quality)"),
        (30, 50, "30-50"),
        (50, 70, "50-70"),
        (70, 85, "70-85"),
        (85, 101, "85-100 (zyada quality)"),
    ]
    for lo, hi, label in bins:
        group = [t for t in all_trades if t["quality_score"] is not None and lo <= t["quality_score"] < hi]
        stats = aggregate_trades(group)
        log(f"  {label}: Trades={stats['total_trades']}  Win%={stats['win_rate_pct']}  PF={stats['profit_factor']}")

    no_score = [t for t in all_trades if t["quality_score"] is None]
    if no_score:
        log(f"\n  (Score na milne wale trades: {len(no_score)} - shuru ke bars jahan resistance_lookback poora nahi tha)")

    log("\n" + "-" * 65)
    log("KAISE SAMJHEIN:")
    log("-" * 65)
    log("- Agar upar wale buckets (85-100) ka PF neeche wale (0-30) se")
    log("  saaf zyada hai, to Quality Score waqai kaam ki cheez hai -")
    log("  ranking ke liye (ya halke filter ke liye) istemal ho sakta hai.")
    log("- Chhote sample wale buckets par kam bharosa karein.")

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
            with open("quality_score_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'quality_score_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)
