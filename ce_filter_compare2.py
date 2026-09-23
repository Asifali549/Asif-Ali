"""
CE Buy-Only ke 4 aur filters ko ALAG ALAG (baseline se) test karte hain,
taake dekh sakein in mein se koi genuinely faida deta hai ya nahi -
kisi ko bhi combine NAHI kiya (jaisa pehle ETH+RS test mein seekha ke
combine karne se sample bohot chhota ho kar numbers ghair-bharosemand
ho jate hain):

  1) Baseline (koi filter nahi)
  2) + Higher-Timeframe Alignment (4h EMA50 se upar ho)
  3) + Volume Confirmation (breakout candle ka volume apni 20-bar
     average se zyada ho)
  4) + 52-Week-High Distance Filter (<=15% door ho apni saal ki
     bulandi se) - Union AB wala hi validated tareeqa
  5) + Liquidity Restriction: sirf Top-60 (sab se zyada liquid) coins
  6) + Liquidity Restriction: sirf Top-100 coins

Entry/exit hamesha 11/4.5 aur 16/3.0 (High-based), jaisa pehle confirm
ho chuka. Result 'ce_filter_compare2_RESULTS.txt' mein save hota hai.
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown

TOP_N_COINS = 150
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = 4380              # ~6 maheene (1h)
HTF_TIMEFRAME = "4h"
HTF_CANDLE_LIMIT = 1200          # ~6+ maheene (4h)
HTF_EMA_PERIOD = 50
VOLUME_SMA_PERIOD = 20
DAILY_FETCH_LIMIT = 425
HIGH_52W_LOOKBACK_DAYS = 365
HIGH_52W_CUTOFF_PCT = 15.0
LIQUIDITY_TOP_60 = 60
LIQUIDITY_TOP_100 = 100

ENTRY_PARAMS = {"period": 11, "multiplier": 4.5}
EXIT_PARAMS = {"period": 16, "multiplier": 3.0}


# ---------- Chandelier core (High-based) ----------

def compute_atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def chandelier_stop(df, period, multiplier):
    atr = compute_atr(df, period)
    highest = df["high"].rolling(period).max()
    return highest - (multiplier * atr)


def ce_buy_only_signal(df, period, multiplier):
    stop = chandelier_stop(df, period, multiplier)
    close = df["close"]
    cross_above = (close > stop) & (close.shift(1) <= stop.shift(1))
    return cross_above.fillna(False)


def simulate_chandelier_trades(df, signal, exit_stop_series, bt_params):
    fee = bt_params["fee_pct"] / 100
    slip = bt_params["slippage_pct"] / 100
    max_hold = bt_params["max_hold_bars"]

    ce_stop = exit_stop_series.values
    close = df["close"].values
    low = df["low"].values
    n = len(df)

    trades = []
    signal_idx = np.where(signal.values)[0]

    for i in signal_idx:
        if i + 1 >= n or np.isnan(ce_stop[i]):
            continue
        entry_bar = i + 1
        entry_price = df["open"].values[entry_bar] * (1 + slip)

        trail_stop = ce_stop[i]
        exit_price = None
        exit_bar = None
        for j in range(entry_bar, min(entry_bar + max_hold, n)):
            if not np.isnan(ce_stop[j]):
                trail_stop = max(trail_stop, ce_stop[j])
            if low[j] <= trail_stop:
                exit_price = trail_stop
                exit_bar = j
                break

        if exit_price is None:
            last_bar = min(entry_bar + max_hold - 1, n - 1)
            exit_price = close[last_bar]
            exit_bar = last_bar

        exit_price = exit_price * (1 - slip)
        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - (2 * fee)
        trades.append({
            "signal_idx": i,
            "return_pct": net_return * 100,
            "bars_held": exit_bar - entry_bar,
        })

    return trades


# ---------- 4 naye filters ----------

def compute_htf_bullish(htf_df, ema_period=HTF_EMA_PERIOD):
    """4h close > 4h EMA50 - bara trend upar ki taraf hai ya nahi."""
    ema = htf_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = htf_df["close"] > ema
    return pd.Series(is_bullish.values, index=pd.to_datetime(htf_df["timestamp"]).values)


def is_bullish_at(regime_series, ts):
    valid = regime_series[regime_series.index <= ts]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])


def compute_volume_ok_series(df, sma_period=VOLUME_SMA_PERIOD):
    """Breakout candle ka volume apni N-bar average se zyada ho."""
    vol_sma = df["volume"].rolling(sma_period).mean()
    return df["volume"] > vol_sma


def compute_dist_from_52w_high_daily(coin_daily_df, sig_ts, current_close, lookback_days=HIGH_52W_LOOKBACK_DAYS):
    ts = pd.to_datetime(coin_daily_df["timestamp"])
    valid = coin_daily_df.loc[ts <= sig_ts]
    if len(valid) == 0:
        return None
    window = valid.iloc[-lookback_days:]
    hi = window["high"].max()
    if pd.isna(hi) or hi <= 0:
        return None
    return (hi - current_close) / hi * 100


def pf_of(trades):
    if not trades:
        return None, 0, None
    df_t = pd.DataFrame(trades)
    wins = df_t[df_t["return_pct"] > 0]
    losses = df_t[df_t["return_pct"] <= 0]
    win_rate = len(wins) / len(df_t) * 100
    pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
    return pf, len(df_t), win_rate


def main():
    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TOP_N_COINS]   # pehle se hi liquidity (24h volume) ke hisab se descending sorted
    print(f"Top {len(coins)} coins (liquidity order), {SIGNAL_TIMEFRAME}, {CANDLE_LIMIT} candles (~6 maheene)...\n")

    variant_names = [
        "1) Baseline (koi filter nahi)",
        "2) + Higher-Timeframe Alignment (4h EMA50)",
        "3) + Volume Confirmation",
        "4) + 52-Week-High Filter (<=15%)",
        "5) + Liquidity: Top-60 sirf",
        "6) + Liquidity: Top-100 sirf",
    ]
    all_trades = {name: [] for name in variant_names}

    done = 0
    for rank, symbol in enumerate(coins):
        done += 1
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=CANDLE_LIMIT)
            htf_df = fetch_ohlcv(exchange, symbol, HTF_TIMEFRAME, limit=HTF_CANDLE_LIMIT)
            coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=DAILY_FETCH_LIMIT)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fetch fail ({e})")
            continue
        if df is None or len(df) < 300 or htf_df is None or len(htf_df) < 60 or coin_daily is None or len(coin_daily) < 60:
            continue

        try:
            sig = apply_cooldown(
                ce_buy_only_signal(df, ENTRY_PARAMS["period"], ENTRY_PARAMS["multiplier"]),
                config.SIGNAL_COOLDOWN_BARS,
            )
            exit_stop = chandelier_stop(df, EXIT_PARAMS["period"], EXIT_PARAMS["multiplier"])
            all_signal_trades = simulate_chandelier_trades(df, sig, exit_stop, config.BACKTEST_PARAMS)

            htf_bullish = compute_htf_bullish(htf_df)
            vol_ok_series = compute_volume_ok_series(df)

            for t in all_signal_trades:
                i = t["signal_idx"]
                sig_ts = pd.Timestamp(df["timestamp"].iloc[i])
                current_close = float(df["close"].iloc[i])

                htf_ok = is_bullish_at(htf_bullish, sig_ts)
                vol_ok = bool(vol_ok_series.iloc[i]) if not pd.isna(vol_ok_series.iloc[i]) else False
                dist_52w = compute_dist_from_52w_high_daily(coin_daily, sig_ts, current_close)
                high52w_ok = (dist_52w is not None) and (dist_52w <= HIGH_52W_CUTOFF_PCT)
                liquidity_top60_ok = rank < LIQUIDITY_TOP_60
                liquidity_top100_ok = rank < LIQUIDITY_TOP_100

                all_trades["1) Baseline (koi filter nahi)"].append(t)
                if htf_ok:
                    all_trades["2) + Higher-Timeframe Alignment (4h EMA50)"].append(t)
                if vol_ok:
                    all_trades["3) + Volume Confirmation"].append(t)
                if high52w_ok:
                    all_trades["4) + 52-Week-High Filter (<=15%)"].append(t)
                if liquidity_top60_ok:
                    all_trades["5) + Liquidity: Top-60 sirf"].append(t)
                if liquidity_top100_ok:
                    all_trades["6) + Liquidity: Top-100 sirf"].append(t)

        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fail ({e})")

        if done % 10 == 0 or done == len(coins):
            print(f"[{done}/{len(coins)}] ... done")

    output_lines = []

    def emit(line=""):
        print(line)
        output_lines.append(line)

    emit("\n\n########## CE BUY-ONLY - 4 NAYE FILTERS (ALAG ALAG) - RESULTS ##########")
    emit(f"(Entry {ENTRY_PARAMS['period']}/{ENTRY_PARAMS['multiplier']}, Exit {EXIT_PARAMS['period']}/{EXIT_PARAMS['multiplier']}, "
         f"High-based, top {TOP_N_COINS} coins (liquidity order), {SIGNAL_TIMEFRAME}, {CANDLE_LIMIT} candles)\n")

    for name in variant_names:
        pf, n, wr = pf_of(all_trades[name])
        if pf is None:
            emit(f"{name}: koi trades nahi mile")
        else:
            avg_bars = np.mean([t["bars_held"] for t in all_trades[name]])
            emit(f"{name}:")
            emit(f"   Trades = {n}")
            emit(f"   Win Rate = {wr:.2f}%")
            emit(f"   Profit Factor = {pf:.3f}")
            emit(f"   Avg Bars Held = {avg_bars:.1f}")
            emit("")

    result_file = "ce_filter_compare2_RESULTS.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
