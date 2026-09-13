"""
Chandelier Multiplier Comparison - 400 coins, 1h aur 4h.

Mukhtalif ATR multiplier ki qadrein (3.0 - abhi wala, 3.5, 4.0, 4.5)
test karte hain - taake dekhein ke "kholi" Chandelier (zyada room
dena) waqai behtar hai ya nahi (bare, asli pumps ko poora pakadna).

Har multiplier ke liye BILKUL ASAL, GHAIR-TABDEEL simulate_trades
function istemal hoti hai (koi extra shart nahi).

Result 'result.txt' mein save hota hai.
"""

import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_all_data
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import simulate_trades

TIMEFRAMES = ["1h", "4h"]
CANDLE_LIMITS = {"1h": 2000, "4h": 1500}
N_COINS = 400

CE_PERIOD_A = 16
CE_PERIOD_B = 12
MULTIPLIERS_TO_TEST = [3.0, 3.5, 4.0, 4.5]


def main():
    lines = []
    lines.append("=== Chandelier Multiplier Comparison - 400 coins ===")
    lines.append(f"Timeframes: {TIMEFRAMES}, Multipliers: {MULTIPLIERS_TO_TEST}\n")

    exchange = get_exchange()
    coin_list = get_coin_list(exchange)[:N_COINS]
    print(f"Coins: {len(coin_list)}")

    data = fetch_all_data(coin_list, TIMEFRAMES, CANDLE_LIMITS)
    print("Data fetch ho gaya, ab har multiplier test kar rahe hain...")

    results = {}

    for symbol, tf_data in data.items():
        for tf, df in tf_data.items():
            if df is None or len(df) < 250:
                continue
            try:
                ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
                ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
                ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
                breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)

                combo_a = ichi_sig & ms_sig
                combo_b = ema_sig & breakout_sig

                for mult in MULTIPLIERS_TO_TEST:
                    ce_a = {"period": CE_PERIOD_A, "multiplier": mult}
                    ce_b = {"period": CE_PERIOD_B, "multiplier": mult}

                    key = (tf, mult)
                    if key not in results:
                        results[key] = []

                    if combo_a.sum() > 0:
                        results[key].extend(simulate_trades(df, combo_a, config.BACKTEST_PARAMS, ce_a))
                    if combo_b.sum() > 0:
                        results[key].extend(simulate_trades(df, combo_b, config.BACKTEST_PARAMS, ce_b))
            except Exception as e:
                print(f"  [SKIP] {symbol} {tf}: {e}")

    for tf in TIMEFRAMES:
        lines.append(f"\n########## TIMEFRAME: {tf} ##########")
        for mult in MULTIPLIERS_TO_TEST:
            trades = results.get((tf, mult), [])
            if not trades:
                lines.append(f"\nMultiplier {mult}: koi trades nahi")
                continue
            df_t = pd.DataFrame(trades)
            wins = df_t[df_t["return_pct"] > 0]
            losses = df_t[df_t["return_pct"] <= 0]
            win_rate = len(wins) / len(df_t) * 100
            pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None

            lines.append(f"\nMultiplier {mult} (abhi live system = 3.0):")
            lines.append(f"  Trades: {len(df_t)}")
            lines.append(f"  Win Rate: {win_rate:.1f}%")
            lines.append(f"  Profit Factor: {pf:.2f}" if pf else "  Profit Factor: N/A")
            lines.append(f"  Avg return/trade: {df_t['return_pct'].mean():.3f}%")

    lines.append("\n\n=== Kaise Parhein (How to Read) ===")
    lines.append("Agar 3.5/4.0/4.5 ka Profit Factor 3.0 se ZYADA hai, to 'khula' Chandelier")
    lines.append("waqai behtar hai (bare moves poore pakadta hai, chhoti pullbacks bardasht")
    lines.append("karta hai). Agar 3.0 hi sab se behtar raha, to abhi wala setting sahi hai -")
    lines.append("badalne ki zaroorat nahi.")

    result_text = "\n".join(lines)
    print(result_text)
    with open("result.txt", "w") as f:
        f.write(result_text)


if __name__ == "__main__":
    main()