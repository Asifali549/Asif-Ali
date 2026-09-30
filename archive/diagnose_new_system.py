"""
NEW System Weakness Diagnosis — Score aur Signal-Type ke hisab se breakdown
================================================================================
NEW_AdvancedConfluence_v1 ke har signal ko uske Score (6-10) aur
signal-type (BOS ya CHoCH ya dono) ke hisab se group karta hai, phir
har group ka Win%/PF dikhata hai - taake pata chale ke:
  1. Kya zyada Score wale signals (jaise 8-10) behtar hain kam Score
     (jaise 6) se?
  2. Kya BOS-triggered signals CHoCH-triggered se behtar/kamzor hain?
  3. Zyada tar losses kitni jaldi (kitne bars mein) hoti hain?

Sirf TOP 60 liquid coins par, ETH filter ke sath (jo pehle test mein
best-case tha) - taake sabse acchi condition mein bhi dekhein ke
weakness kahan hai.

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

OUTPUT_FILE = "new_system_diagnosis_result.txt"


def compute_eth_regime(eth_daily_df, ema_period=200):
    ema = eth_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = eth_daily_df["close"] > ema
    return pd.Series(is_bullish.values, index=pd.to_datetime(eth_daily_df["timestamp"]).values)


def is_eth_bullish_at(eth_regime, ts):
    valid = eth_regime[eth_regime.index <= ts]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])


def simulate_with_diagnostics(df, result_new, eth_regime):
    """
    NEW system ke signals simulate karta hai, aur har trade ke sath
    score, bos/choch type, aur outcome details record karta hai.
    """
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
            result_new = compute_confluence(df, btc_daily, CONF_PARAMS, usdt_d_weak=None)
            trades = simulate_with_diagnostics(df, result_new, eth_regime)
            all_trades.extend(trades)
        except Exception as e:
            log(f"  [SKIP] {symbol}: {e}")

        if rank_included % 10 == 0:
            log(f"  {rank_included}/{N_TOP_LIQUID} top-liquid coins mile (trades so far: {len(all_trades)})")

    log(f"\n[INFO] Total {rank_included} coins shamil, {len(all_trades)} trades mile")

    log("\n" + "=" * 65)
    log("NATIJA: NEW System Weakness Diagnosis")
    log("=" * 65)

    overall = aggregate_trades(all_trades)
    log(f"\nOVERALL: Trades={overall['total_trades']}  Win%={overall['win_rate_pct']}  PF={overall['profit_factor']}")

    log("\n--- SCORE ke hisab se breakdown ---")
    for score_val in sorted(set(t["score"] for t in all_trades)):
        group = [t for t in all_trades if t["score"] == score_val]
        stats = aggregate_trades(group)
        log(f"  Score={score_val}: Trades={stats['total_trades']}  Win%={stats['win_rate_pct']}  PF={stats['profit_factor']}")

    log("\n--- Signal Type (BOS vs CHoCH vs BOTH) ke hisab se ---")
    for sig_type in ["BOS", "CHOCH", "BOTH"]:
        group = [t for t in all_trades if t["signal_type"] == sig_type]
        stats = aggregate_trades(group)
        log(f"  {sig_type}: Trades={stats['total_trades']}  Win%={stats['win_rate_pct']}  PF={stats['profit_factor']}")

    log("\n--- Exit Reason ke hisab se ---")
    for reason in ["CE_STOP", "TIME"]:
        group = [t for t in all_trades if t["exit_reason"] == reason]
        stats = aggregate_trades(group)
        log(f"  {reason}: Trades={stats['total_trades']}  Win%={stats['win_rate_pct']}  PF={stats['profit_factor']}")

    quick_losses = [t for t in all_trades if t["exit_reason"] == "CE_STOP" and t["bars_held"] <= 3 and t["return_pct"] < 0]
    log(f"\n--- Fauri (3 bars ke andar) SL trades ---")
    if all_trades:
        log(f"  Count: {len(quick_losses)} / {len(all_trades)} ({len(quick_losses)/len(all_trades)*100:.1f}%)")
    else:
        log("  Koi trades nahi mile is dorania mein.")

    log("\n" + "-" * 65)
    log("KAISE SAMJHEIN:")
    log("-" * 65)
    log("- Agar zyada Score (8-10) wale trades ka Win%/PF kam Score (6)")
    log("  se behtar nahi hai, to Score system khud kaam nahi kar raha.")
    log("- Agar BOS aur CHoCH mein bara farq hai, to ek type ko hata kar")
    log("  sirf behtar wala rakhna faida mand ho sakta hai.")
    log("- Agar zyada tar trades CE_STOP se aur jaldi (3 bars ke andar)")
    log("  nuksan mein jaate hain, to entry timing ka masla hai.")

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
            with open("new_system_diagnosis_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'new_system_diagnosis_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)
