"""
Do Filters Test - 200 coins, 1h (live system jaisa):
  1. ETH Regime Filter - ETH ki apni daily EMA(200) se upar ho tabhi
     signal lein (har coin ke liye, jaisa BTC Regime tha mgr ETH se)
  2. Isi Coin ki 4h EMA(200) Filter - jis coin par 1h signal hai,
     usi coin ki apni 4h timeframe ki EMA(200) se upar ho tabhi lein

Dono BASELINE (bina filter) ke sath compare hote hain.
BILKUL ASAL, GHAIR-TABDEEL simulate_trades function istemal hoti hai.

Result 'result.txt' mein save hota hai.
"""

import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_all_data, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import simulate_trades

N_COINS = 200
SIGNAL_TF = "1h"
CANDLE_LIMIT_1H = 2000
CANDLE_LIMIT_4H = 1500

CE_A = {"period": 16, "multiplier": 4.5}
CE_B = {"period": 12, "multiplier": 4.5}


def summarize(trades, label, lines):
    if not trades:
        lines.append(f"\n{label}: koi trades nahi")
        return
    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["return_pct"] > 0]
    losses = df_t[df_t["return_pct"] <= 0]
    win_rate = len(wins) / len(df_t) * 100
    pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
    lines.append(f"\n{label}:")
    lines.append(f"  Trades: {len(df_t)}")
    lines.append(f"  Win Rate: {win_rate:.1f}%")
    lines.append(f"  Profit Factor: {pf:.2f}" if pf else "  Profit Factor: N/A")
    lines.append(f"  Avg return/trade: {df_t['return_pct'].mean():.3f}%")


def main():
    lines = []
    lines.append("=== ETH Regime + Same-Coin 4h EMA Filter Test - 200 coins ===\n")

    exchange = get_exchange()
    coin_list = get_coin_list(exchange)[:N_COINS]
    print(f"Coins: {len(coin_list)}")

    print("ETH daily benchmark data fetch kar rahe hain...")
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=800)
    eth_ema200 = eth_daily["close"].ewm(span=200, adjust=False).mean()
    eth_regime = pd.DataFrame({
        "timestamp": eth_daily["timestamp"],
        "regime_ok": eth_daily["close"] > eth_ema200,
    })

    print("1h data fetch kar rahe hain...")
    data_1h = fetch_all_data(coin_list, [SIGNAL_TF], {SIGNAL_TF: CANDLE_LIMIT_1H})

    baseline_trades = []
    eth_filter_trades = []
    coin_4h_ema_trades = []

    for symbol, tf_data in data_1h.items():
        df = tf_data.get(SIGNAL_TF)
        if df is None or len(df) < 250:
            continue
        try:
            ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
            ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
            ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
            breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)

            combo_a = ichi_sig & ms_sig
            combo_b = ema_sig & breakout_sig

            # ---- BASELINE ----
            if combo_a.sum() > 0:
                baseline_trades.extend(simulate_trades(df, combo_a, config.BACKTEST_PARAMS, CE_A))
            if combo_b.sum() > 0:
                baseline_trades.extend(simulate_trades(df, combo_b, config.BACKTEST_PARAMS, CE_B))

            # ---- TEST 1: ETH Regime Filter ----
            merged_eth = pd.merge_asof(
                df[["timestamp"]].sort_values("timestamp"),
                eth_regime.sort_values("timestamp"),
                on="timestamp", direction="backward",
            )
            eth_ok = merged_eth["regime_ok"].fillna(False).values
            combo_a_eth = combo_a & eth_ok
            combo_b_eth = combo_b & eth_ok
            if combo_a_eth.sum() > 0:
                eth_filter_trades.extend(simulate_trades(df, combo_a_eth, config.BACKTEST_PARAMS, CE_A))
            if combo_b_eth.sum() > 0:
                eth_filter_trades.extend(simulate_trades(df, combo_b_eth, config.BACKTEST_PARAMS, CE_B))

            # ---- TEST 2: Isi coin ki apni 4h EMA(200) ----
            try:
                df_4h = fetch_ohlcv(exchange, symbol, "4h", limit=CANDLE_LIMIT_4H)
            except Exception:
                df_4h = None
            if df_4h is not None and len(df_4h) >= 210:
                ema200_4h = df_4h["close"].ewm(span=200, adjust=False).mean()
                coin_regime_4h = pd.DataFrame({
                    "timestamp": df_4h["timestamp"],
                    "regime_ok": df_4h["close"] > ema200_4h,
                })
                merged_coin = pd.merge_asof(
                    df[["timestamp"]].sort_values("timestamp"),
                    coin_regime_4h.sort_values("timestamp"),
                    on="timestamp", direction="backward",
                )
                coin_ok = merged_coin["regime_ok"].fillna(False).values
                combo_a_coin = combo_a & coin_ok
                combo_b_coin = combo_b & coin_ok
                if combo_a_coin.sum() > 0:
                    coin_4h_ema_trades.extend(simulate_trades(df, combo_a_coin, config.BACKTEST_PARAMS, CE_A))
                if combo_b_coin.sum() > 0:
                    coin_4h_ema_trades.extend(simulate_trades(df, combo_b_coin, config.BACKTEST_PARAMS, CE_B))
        except Exception as e:
            print(f"  [SKIP] {symbol}: {e}")

    summarize(baseline_trades, "BASELINE (bina filter, Union AB 1h)", lines)
    summarize(eth_filter_trades, "+ ETH Regime Filter (ETH apni daily EMA200 se upar)", lines)
    summarize(coin_4h_ema_trades, "+ Isi Coin ki 4h EMA(200) Filter", lines)

    result_text = "\n".join(lines)
    print(result_text)
    with open("result.txt", "w") as f:
        f.write(result_text)


if __name__ == "__main__":
    main()