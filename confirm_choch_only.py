"""
Confirmation Test: CHoCH-Only (NEW_AdvancedConfluence se BOS nikaal kar
sirf CHoCH rakhna) - BARA sample (Top 300 coins) par, aur time ko do
hisson mein taqseem kar ke (Period 1 = purana half, Period 2 = naya
half) consistency check karta hai - taake pehle chhote test (150
coins, PF 1.255) ka result ek dafa ka accident na ho.

4 comparisons:
  A) BOS+CHoCH Baseline (jaisa abhi live hai) - poora sample
  B) CHoCH-Only - poora sample
  C) CHoCH-Only - Period 1 (purana half)
  D) CHoCH-Only - Period 2 (naya half)

Agar B behtar hai A se, AUR C aur D dono bhi ek dusre ke qareeb/dono
achhe hain (sirf ek half mein achha na ho), tabhi CHoCH-Only ko
consistent maana ja sakta hai.

Chalayen: python confirm_choch_only.py
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown
from backtest_engine import compute_atr, simulate_trades
from confluence_engine import compute_confluence, DEFAULT_PARAMS as CONF_PARAMS

TOP_N_COINS = 300
SIGNAL_TIMEFRAME = "1h"
CE_D = {"period": 16, "multiplier": 3.0}
SCORE_THRESHOLD = 6

CANDLE_LIMIT = max(config.CANDLE_LIMITS.get(SIGNAL_TIMEFRAME, 2000), 2000)


def main():
    exchange = get_exchange()

    print("BTC daily benchmark data fetch kar rahe hain...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=800)

    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"Scanning {len(coins)} coins on {SIGNAL_TIMEFRAME} ({CANDLE_LIMIT} candles har coin)...\n")

    trades_baseline_all = []
    trades_choch_all = []
    trades_choch_p1 = []
    trades_choch_p2 = []
    done = 0

    for symbol in coins:
        done += 1
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=CANDLE_LIMIT)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fetch fail ({e})")
            continue
        if df is None or len(df) < 300:
            continue

        try:
            result_new = compute_confluence(df, btc_daily, CONF_PARAMS, usdt_d_weak=None)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: confluence calc fail ({e})")
            continue

        structure_signal = result_new["bos"] | result_new["choch"]
        score_ok = result_new["score"] >= SCORE_THRESHOLD

        sig_baseline = apply_cooldown(structure_signal & score_ok, config.SIGNAL_COOLDOWN_BARS)
        sig_choch = apply_cooldown(result_new["choch"] & score_ok, config.SIGNAL_COOLDOWN_BARS)

        trades_baseline_all.extend(simulate_trades(df, sig_baseline, config.BACKTEST_PARAMS, CE_D))
        trades_choch_all.extend(simulate_trades(df, sig_choch, config.BACKTEST_PARAMS, CE_D))

        # Period split: bar-index ke hisab se pehla half vs doosra half
        mid = len(df) // 2
        sig_choch_p1 = sig_choch.copy()
        sig_choch_p1.iloc[mid:] = False
        sig_choch_p2 = sig_choch.copy()
        sig_choch_p2.iloc[:mid] = False

        trades_choch_p1.extend(simulate_trades(df, sig_choch_p1, config.BACKTEST_PARAMS, CE_D))
        trades_choch_p2.extend(simulate_trades(df, sig_choch_p2, config.BACKTEST_PARAMS, CE_D))

        if done % 10 == 0 or done == len(coins):
            print(f"[{done}/{len(coins)}] ... baseline={len(trades_baseline_all)} "
                  f"choch_all={len(trades_choch_all)} p1={len(trades_choch_p1)} p2={len(trades_choch_p2)}")

    def summarize(trades, label):
        print(f"\n=== {label} ===")
        if not trades:
            print("Koi trades nahi mile.")
            return
        df_t = pd.DataFrame(trades)
        wins = df_t[df_t["return_pct"] > 0]
        losses = df_t[df_t["return_pct"] <= 0]
        win_rate = len(wins) / len(df_t) * 100
        pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
        print(f"Trades: {len(df_t)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Profit Factor: {pf:.3f}" if pf else "Profit Factor: N/A")
        print(f"Avg Return/Trade: {df_t['return_pct'].mean():.3f}%")
        print(f"Total Return (sum): {df_t['return_pct'].sum():.2f}%")

    print("\n\n########## CONFIRMATION RESULTS ##########")
    summarize(trades_baseline_all, "A) BOS+CHoCH Baseline (jaisa abhi live hai) - Poora Sample")
    summarize(trades_choch_all, "B) CHoCH-Only - Poora Sample")
    summarize(trades_choch_p1, "C) CHoCH-Only - Period 1 (purana half)")
    summarize(trades_choch_p2, "D) CHoCH-Only - Period 2 (naya half)")


if __name__ == "__main__":
    main()
