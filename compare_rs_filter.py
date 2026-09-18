"""
Relative Strength (vs BTC) Filter — Union AB (Top 60 Liquid + ETH Filter)
================================================================================
Idea: Sirf un coins ke signals lo jo apne BTC ke muqable "outperform"
kar rahe hon (Relative Strength trend upward). RS Ratio = coin daily
close / BTC daily close. Agar ye ratio apni EMA(50) se upar ho, to
coin BTC se behtar chal raha hai (RS bullish).

Union AB (live settings: Chandelier Multiplier 4.5, Top 60 Liquid,
ETH Regime Filter) par ye NAYA filter UPAR se laga kar test karte
hain:
    A) BASELINE - jaisa abhi hai (RS filter nahi)
    B) +RS_FILTER - sirf tab signal lo jab coin ka RS trend (vs BTC)
       bhi bullish ho

KuCoin data, Top 60 Liquid coins, 270 din - hamare established
tareeqe ke mutabiq.

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
RS_EMA_PERIOD = 50  # RS ratio ki apni trend-EMA (daily)

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "rs_filter_result.txt"


def compute_eth_regime(eth_daily_df, ema_period=200):
    ema = eth_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = eth_daily_df["close"] > ema
    return pd.Series(is_bullish.values, index=pd.to_datetime(eth_daily_df["timestamp"]).values)


def is_bullish_at(regime_series, ts):
    valid = regime_series[regime_series.index <= ts]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])


def compute_rs_trend(coin_daily_df, btc_daily_df, ema_period=50):
    """
    RS Ratio = coin close / BTC close (date ke hisab se align kiya
    gaya), phir uski apni EMA se moqabla - taake maloom ho ratio khud
    upar ja raha hai (outperformance barh rahi hai) ya neeche.
    Returns pandas Series (DatetimeIndex) of booleans.
    """
    coin_d = coin_daily_df[["timestamp", "close"]].rename(columns={"close": "coin_close"})
    btc_d = btc_daily_df[["timestamp", "close"]].rename(columns={"close": "btc_close"})
    merged = pd.merge_asof(
        coin_d.sort_values("timestamp"), btc_d.sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    merged["rs_ratio"] = merged["coin_close"] / merged["btc_close"].replace(0, np.nan)
    merged["rs_ema"] = merged["rs_ratio"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = merged["rs_ratio"] > merged["rs_ema"]
    return pd.Series(is_bullish.values, index=pd.to_datetime(merged["timestamp"]).values)


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


def simulate_trades_simple(df, signal, ce_params):
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
        if i + 1 >= n or np.isnan(atr_vals[i]) or np.isnan(ce_stop[i]):
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
        trades.append({"return_pct": net_return_pct})

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

    log("ETH daily data + 200 EMA regime nikal rahe hain...")
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=DURATION_DAYS + 250)
    eth_regime = compute_eth_regime(eth_daily, ema_period=ETH_EMA_PERIOD)

    log("BTC daily data nikal rahe hain (RS ratio ke liye)...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DURATION_DAYS + 250)

    coins = get_coin_list(exchange)[:N_FETCH_COINS]
    log(f"Scanning - sirf TOP {N_TOP_LIQUID} liquid, full-history wale...\n")

    trades_baseline = []
    trades_rs_filtered = []
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
            rs_trend = compute_rs_trend(coin_daily, btc_daily, ema_period=RS_EMA_PERIOD)
        except Exception as e:
            log(f"  [RS-SKIP] {symbol}: {e}")
            rs_trend = None

        for combo_name in ["Ichimoku+MS", "EMA+Breakout"]:
            try:
                signal, ce_params = get_combo_signal(df, combo_name)
                eth_filtered = apply_regime_filter(df, signal, eth_regime)

                trades_a = simulate_trades_simple(df, eth_filtered, ce_params)
                trades_baseline.extend(trades_a)

                if rs_trend is not None:
                    rs_filtered = apply_regime_filter(df, eth_filtered, rs_trend)
                    trades_b = simulate_trades_simple(df, rs_filtered, ce_params)
                    trades_rs_filtered.extend(trades_b)
                else:
                    trades_rs_filtered.extend(trades_a)  # RS data na mile to baseline jaisa hi rakho
            except Exception as e:
                log(f"  [SKIP] {symbol} {combo_name}: {e}")

        if rank_included % 10 == 0:
            log(f"  {rank_included}/{N_TOP_LIQUID} coins mile "
                f"(baseline: {len(trades_baseline)}, +RS: {len(trades_rs_filtered)})")

    log(f"\n[INFO] Total {rank_included} coins shamil hue")

    log("\n" + "=" * 65)
    log("NATIJA: Relative Strength (vs BTC) Filter on Union AB")
    log("=" * 65)

    stats_a = aggregate_trades(trades_baseline)
    stats_b = aggregate_trades(trades_rs_filtered)
    log(f"\nBASELINE (Top 60 + ETH filter) : Trades={stats_a['total_trades']}  "
        f"Win%={stats_a['win_rate_pct']}  PF={stats_a['profit_factor']}  "
        f"Total Return={stats_a['total_return_pct']}%")
    log(f"+RS_FILTER (vs BTC)            : Trades={stats_b['total_trades']}  "
        f"Win%={stats_b['win_rate_pct']}  PF={stats_b['profit_factor']}  "
        f"Total Return={stats_b['total_return_pct']}%")

    log("\nNOTE: Agar '+RS_FILTER' ka PF BASELINE se behtar hai, to Relative")
    log("Strength filter waqai faida mand hai - hamare live system mein")
    log("shamil karne ke qabil. Trades kam hone ko akele nuksan na samjhein.")

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
            with open("rs_filter_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'rs_filter_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)
