"""
Market Breadth Filter Test - 400 coins, 1h (live system jaisa).

"Market Breadth" = har din, kitne % coins apni DAILY EMA(200) se
UPAR hain - ye ek khud-apna-banaya "poori market ki sehat" ka
paimana hai (TOTAL market cap ki tarah, lekin hamare apne data se
banaya gaya - kisi bahar ki API ki zaroorat nahi).

Filter: sirf tab signal lein jab Breadth >= 50% (yani market mein
"zyada coins upar hain, neeche nahi" - wasee tor par sehatmand halat).

BILKUL ASAL, GHAIR-TABDEEL simulate_trades function istemal hoti hai.

Result 'result_breadth.txt' mein save hota hai.
"""

import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_all_data
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import simulate_trades

N_COINS = 400
TIMEFRAMES = ["1h", "1d"]
CANDLE_LIMITS = {"1h": 2000, "1d": 800}
BREADTH_THRESHOLD = 50.0  # % coins upar hone chahiye

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
    lines.append("=== Market Breadth Filter Test - 400 coins ===\n")

    exchange = get_exchange()
    coin_list = get_coin_list(exchange)[:N_COINS]
    print(f"Coins: {len(coin_list)}")

    print("1h aur daily data fetch kar rahe hain (breadth calculate karne ke liye)...")
    data = fetch_all_data(coin_list, TIMEFRAMES, CANDLE_LIMITS)

    # ---- STEP 1: Har coin ki daily "above EMA200" series banayein ----
    print("Har coin ki daily EMA(200) status nikal rahe hain...")
    above_series_list = []
    for symbol, tf_data in data.items():
        df_daily = tf_data.get("1d")
        if df_daily is None or len(df_daily) < 210:
            continue
        ema200 = df_daily["close"].ewm(span=200, adjust=False).mean()
        above = (df_daily["close"] > ema200).astype(int)
        s = pd.Series(above.values, index=df_daily["timestamp"].values, name=symbol)
        above_series_list.append(s)

    if not above_series_list:
        lines.append("Breadth calculate nahi ho saka - daily data kaafi nahi.")
        with open("result_breadth.txt", "w") as f:
            f.write("\n".join(lines))
        return

    # ---- STEP 2: Sab coins ko ek common daily index par jama karein, breadth % nikalein ----
    above_df = pd.concat(above_series_list, axis=1)
    breadth_pct = above_df.mean(axis=1, skipna=True) * 100
    breadth_df = pd.DataFrame({"timestamp": breadth_pct.index, "breadth_pct": breadth_pct.values}).sort_values("timestamp")
    print(f"Breadth series ready: {len(breadth_df)} din, range {breadth_df['breadth_pct'].min():.1f}% - {breadth_df['breadth_pct'].max():.1f}%")

    # ---- STEP 3: Har coin ke 1h signals par Breadth filter lagayein ----
    baseline_trades = []
    breadth_filter_trades = []

    for symbol, tf_data in data.items():
        df = tf_data.get("1h")
        if df is None or len(df) < 250:
            continue
        try:
            ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
            ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
            ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
            breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)

            combo_a = ichi_sig & ms_sig
            combo_b = ema_sig & breakout_sig

            if combo_a.sum() > 0:
                baseline_trades.extend(simulate_trades(df, combo_a, config.BACKTEST_PARAMS, CE_A))
            if combo_b.sum() > 0:
                baseline_trades.extend(simulate_trades(df, combo_b, config.BACKTEST_PARAMS, CE_B))

            merged = pd.merge_asof(
                df[["timestamp"]].sort_values("timestamp"),
                breadth_df,
                on="timestamp", direction="backward",
            )
            breadth_ok = (merged["breadth_pct"] >= BREADTH_THRESHOLD).fillna(False).values

            combo_a_b = combo_a & breadth_ok
            combo_b_b = combo_b & breadth_ok
            if combo_a_b.sum() > 0:
                breadth_filter_trades.extend(simulate_trades(df, combo_a_b, config.BACKTEST_PARAMS, CE_A))
            if combo_b_b.sum() > 0:
                breadth_filter_trades.extend(simulate_trades(df, combo_b_b, config.BACKTEST_PARAMS, CE_B))
        except Exception as e:
            print(f"  [SKIP] {symbol}: {e}")

    summarize(baseline_trades, "BASELINE (bina filter, Union AB 1h)", lines)
    summarize(breadth_filter_trades, f"+ Market Breadth Filter (>= {BREADTH_THRESHOLD}% coins apni EMA200 se upar)", lines)

    result_text = "\n".join(lines)
    print(result_text)
    with open("result_breadth.txt", "w") as f:
        f.write(result_text)


if __name__ == "__main__":
    main()