"""
CHoCH-Only Test — BOS Hata Kar (Comparison)
================================================================================
Pichle diagnosis mein pata chala ke BOS-triggered signals kamzor
(PF 0.728) aur CHoCH-triggered signals mazboot (PF 2.147) the. Ye
script dono ko compare karta hai:
    A) BASELINE - jaisa abhi hai (BOS ya CHoCH, dono shamil)
    B) CHOCH_ONLY - sirf CHoCH signals (BOS bilkul hata diya)

Score breakdown (6,7,8,9,10) bhi dikhata hai, khali (0 trades) wale
levels bhi saaf tor par dikhaye jate hain.

TOP 60 liquid coins, ETH filter ke sath, 270 din.

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
N_TOP_LIQUID = 60
N_FETCH_COINS = 300
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = DURATION_DAYS * 24 + 300
MIN_BARS_REQUIRED = (DURATION_DAYS - 5) * 24

CE_D = {"period": 16, "multiplier": 3.0}
ETH_EMA_PERIOD = 200

BT_PARAMS = dict(config.BACKTEST_PARAMS)
BT_PARAMS["exit_mode"] = "chandelier"

OUTPUT_FILE = "choch_only_result.txt"


def compute_eth_regime(eth_daily_df, ema_period=200):
    ema = eth_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = eth_daily_df["close"] > ema
    return pd.Series(is_bullish.values, index=pd.to_datetime(eth_daily_df["timestamp"]).values)


def is_eth_bullish_at(eth_regime, ts):
    valid = eth_regime[eth_regime.index <= ts]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])


def simulate_with_diagnostics(df, result_new, eth_regime, choch_only=False):
    """
    NEW system ke signals simulate karta hai.
    choch_only=False -> BASELINE (bos | choch)
    choch_only=True  -> sirf CHoCH signals (BOS bilkul hata diya)
    """
    if choch_only:
        structure_signal = result_new["choch"] & (~result_new["bos"])
    else:
        structure_signal = result_new["bos"] | result_new["choch"]
    signal = apply_cooldown(structure_signal & (result_new["score"] >= 6), config.SIGNAL_COOLDOWN_BARS)

    atr = compute_atr(df, BT_PARAMS["atr_period"])
    ce_stop = compute_chandelier_long_stop(df, CE_D["period"], CE_D["multiplier"]).values
    fee = BT_PARAMS["fee_pct"] / 100
    slip = BT_PARAMS["slippage_pct"] / 100
    max_hold = BT_PARAMS["max_hold_bars"]

    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    atr_vals = atr.values
    n = len(df)
    timestamps = df["timestamp"].values

    bos_vals = result_new["bos"].values
    choch_vals = result_new["choch"].values
    score_vals = result_new["score"].values

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

        sig_type = "BOTH" if (bos_vals[i] and choch_vals[i]) else ("BOS" if bos_vals[i] else "CHOCH")

        trades.append({
            "return_pct": net_return_pct,
            "exit_reason": exit_reason,
            "bars_held": exit_bar - entry_bar,
            "score": int(score_vals[i]),
            "signal_type": sig_type,
        })

    return trades


def aggregate_trades(all_trades):
    if not all_trades:
        return {"total_trades": 0, "win_rate_pct": None, "profit_factor": None}
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

    log("BTC daily data nikal rahe hain...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DURATION_DAYS + 250)

    coins = get_coin_list(exchange)[:N_FETCH_COINS]
    log(f"Scanning coins - sirf TOP {N_TOP_LIQUID} liquid, full-history wale shamil honge...\n")

    all_trades_baseline = []
    all_trades_choch_only = []
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
            result_new = compute_confluence(df, btc_daily, CONF_PARAMS, usdt_d_weak=None)
            trades_a = simulate_with_diagnostics(df, result_new, eth_regime, choch_only=False)
            all_trades_baseline.extend(trades_a)
            trades_b = simulate_with_diagnostics(df, result_new, eth_regime, choch_only=True)
            all_trades_choch_only.extend(trades_b)
        except Exception as e:
            log(f"  [SKIP] {symbol}: {e}")

        if rank_included % 10 == 0:
            log(f"  {rank_included}/{N_TOP_LIQUID} top-liquid coins mile "
                f"(baseline trades: {len(all_trades_baseline)}, choch-only: {len(all_trades_choch_only)})")

    log(f"\n[INFO] Total {rank_included} coins shamil hue")

    log("\n" + "=" * 65)
    log("NATIJA: CHoCH-Only Test (BOS hataya gaya)")
    log("=" * 65)

    stats_a = aggregate_trades(all_trades_baseline)
    stats_b = aggregate_trades(all_trades_choch_only)
    log(f"\nBASELINE (BOS+CHoCH)  : Trades={stats_a['total_trades']}  Win%={stats_a['win_rate_pct']}  PF={stats_a['profit_factor']}")
    log(f"CHOCH_ONLY (bina BOS) : Trades={stats_b['total_trades']}  Win%={stats_b['win_rate_pct']}  PF={stats_b['profit_factor']}")

    log("\n--- CHOCH_ONLY ka SCORE ke hisab se breakdown ---")
    for score_val in [6, 7, 8, 9, 10]:
        group = [t for t in all_trades_choch_only if t["score"] == score_val]
        stats = aggregate_trades(group)
        log(f"  Score={score_val}: Trades={stats['total_trades']}  Win%={stats['win_rate_pct']}  PF={stats['profit_factor']}")

    log("\n--- BASELINE ka SCORE ke hisab se breakdown (moqabla ke liye) ---")
    for score_val in [6, 7, 8, 9, 10]:
        group = [t for t in all_trades_baseline if t["score"] == score_val]
        stats = aggregate_trades(group)
        log(f"  Score={score_val}: Trades={stats['total_trades']}  Win%={stats['win_rate_pct']}  PF={stats['profit_factor']}")

    log("\n" + "-" * 65)
    log("KAISE SAMJHEIN:")
    log("-" * 65)
    log("- Agar CHOCH_ONLY ka PF BASELINE se saaf behtar hai, to BOS ko")
    log("  hamesha ke liye hata dena chahiye.")
    log("- Score breakdown mein 0 Trades wale levels dikhate hain ke wo")
    log("  score is dorania mein kabhi bana hi nahi.")

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
            with open("choch_only_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'choch_only_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)
