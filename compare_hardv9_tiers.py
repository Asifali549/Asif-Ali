"""
HARD V9 — Liquidity Tiers + ETH Filter Test
================================================================================
HARD V9 strategy (pehle chhote test mein avg PF 1.83 lekin 73% coins
nuksan mein - kamzor/mixed) par, hamara tasdeeq-shuda "recipe" (Top-N
Liquid Coins + ETH Regime Filter) laga kar dekhte hain.

Ye strategy apna fixed SL/TP (ATR*1.2 / risk*2.2) istemal karti hai,
Chandelier trailing nahi - is liye simulate_trades (jo chandelier ke
liye bana hai) use nahi ho sakta, khud ka simulation function chahiye
(jo user ne diya tha, wohi istemal ho raha hai).

4h aur Daily trend series bhi har coin ke apne data se nikalte hain
(BTC ka nahi - HARD V9 apne coin ka trend dekhta hai, jaisa asal code
mein tha).

CHALANE SE PEHLE:
    pip install ccxt pandas numpy
"""

import numpy as np
import pandas as pd

from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv

DURATION_DAYS = 270
TIERS = [60, 100, 150]
N_FETCH_COINS = 300
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = DURATION_DAYS * 24 + 300
MIN_BARS_REQUIRED = (DURATION_DAYS - 5) * 24

ETH_EMA_PERIOD = 200

OUTPUT_FILE = "hardv9_tiers_result.txt"

# ---- HARD V9 core (jaisa diya gaya, bina tabdeeli) ----
ADX_LEN = 14
ADX_MIN = 23.0
EMA_FAST_LEN = 20
EMA_SLOW_LEN = 50
MFI_LEN = 14
VOL_LEN = 20
VOL_DELTA_MIN = 0.15
SWING_LEN = 10
ATR_LEN = 14
ATR_EXT_MULT = 1.5
MIN_BARS_BETWEEN_SIGNALS = 14
MAX_EMA_DISTANCE_PCT = 3.8
MIN_DISTANCE_FROM_RECENT_HIGH_PCT = 2.5
MIN_SCORE = 5
RR_TARGET = 2.2
ATR_SL_MULT = 1.2


def rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def mfi(df, n):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    money = tp * df["volume"]
    change = tp.diff()
    pos = money.where(change > 0, 0.0).rolling(n).sum()
    neg = money.where(change < 0, 0.0).rolling(n).sum()
    ratio = pos / neg.replace(0, np.nan)
    out = 100 - 100 / (1 + ratio)
    return out.where(~neg.eq(0), 100.0)


def atr(df, n):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def adx(df, n):
    h, l, c = df["high"], df["low"], df["close"]
    ph, pl, pc = h.shift(1), l.shift(1), c.shift(1)
    up, down = h - ph, pl - l
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    av = rma(tr, n)
    pdi = 100 * rma(plus, n) / av.replace(0, np.nan)
    mdi = 100 * rma(minus, n) / av.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx, n)


def add_indicators(df):
    x = df.copy()
    x["ema_fast"] = ema(x["close"], EMA_FAST_LEN)
    x["ema_slow"] = ema(x["close"], EMA_SLOW_LEN)
    x["mfi"] = mfi(x, MFI_LEN)
    x["atr"] = atr(x, ATR_LEN)
    x["adx"] = adx(x, ADX_LEN)

    bull_vol = np.where(x["close"] > x["open"], x["volume"], 0.0)
    bear_vol = np.where(x["close"] <= x["open"], x["volume"], 0.0)
    sb = pd.Series(bull_vol, index=x.index).rolling(VOL_LEN).sum()
    ss = pd.Series(bear_vol, index=x.index).rolling(VOL_LEN).sum()
    total = sb + ss
    x["net_buy_ratio"] = (sb - ss) / total.replace(0, np.nan)

    x["over_extended"] = x["close"] > x["ema_fast"] + x["atr"] * ATR_EXT_MULT
    x["far_from_ema"] = (x["close"] - x["ema_slow"]).abs() / x["ema_slow"] * 100 > MAX_EMA_DISTANCE_PCT
    x["recent_high"] = x["high"].rolling(50).max()
    x["near_recent_high"] = (x["recent_high"] - x["close"]) / x["recent_high"] * 100 < MIN_DISTANCE_FROM_RECENT_HIGH_PCT
    x["bullish_close"] = x["close"] > x["open"]
    return x


def add_bos(df):
    x = df.copy()
    n = len(x)
    p = SWING_LEN
    highs = x["high"].to_numpy()
    closes = x["close"].to_numpy()

    last_hi = np.nan
    bos = []
    for i in range(n):
        pi = i - p
        if pi >= p:
            hw = highs[i - 2 * p:i + 1]
            if len(hw) == 2 * p + 1 and highs[pi] == np.max(hw):
                last_hi = highs[pi]
        bos.append(False if np.isnan(last_hi) else closes[i] > last_hi)

    x["bos_up"] = bos
    return x


def compute_full_signal_series(df, trend4h_series, daily_bias_series):
    x = add_bos(add_indicators(df))

    waiting = False
    signal_high = np.nan
    signal_low = np.nan
    last_signal_bar = None

    made_buy_arr = np.zeros(len(x), dtype=bool)
    sl_arr = np.full(len(x), np.nan)
    tp_arr = np.full(len(x), np.nan)

    for i in range(len(x)):
        row = x.iloc[i]
        needed = [row["ema_fast"], row["ema_slow"], row["mfi"], row["atr"], row["adx"], row["net_buy_ratio"]]
        if any(pd.isna(v) for v in needed):
            continue

        trend4h = trend4h_series.iloc[i]
        daily_bias = daily_bias_series.iloc[i]

        score = (
            int(trend4h) + int(daily_bias) + int(row["adx"] >= ADX_MIN)
            + int(not row["over_extended"]) + int(not row["far_from_ema"]) + int(not row["near_recent_high"])
        )
        trend_gate = trend4h and daily_bias
        spacing = 999999 if last_signal_bar is None else i - last_signal_bar

        raw = (
            trend_gate and row["adx"] >= ADX_MIN and bool(row["bos_up"])
            and row["net_buy_ratio"] >= VOL_DELTA_MIN and row["mfi"] > 50
            and bool(row["bullish_close"]) and score >= MIN_SCORE
            and spacing >= MIN_BARS_BETWEEN_SIGNALS
        )

        made_buy = False
        if raw and not waiting:
            waiting = True
            signal_high = float(row["high"])
            signal_low = float(row["low"])

        if waiting and not raw:
            if float(row["close"]) > signal_high:
                made_buy = True
                last_signal_bar = i
                waiting = False
            elif float(row["close"]) < signal_low:
                waiting = False

        if made_buy:
            price = float(row["close"])
            sl = price - float(row["atr"]) * ATR_SL_MULT
            tp = price + (price - sl) * RR_TARGET
            made_buy_arr[i] = True
            sl_arr[i] = sl
            tp_arr[i] = tp

    return pd.DataFrame({"buy_signal": made_buy_arr, "sl": sl_arr, "tp": tp_arr}, index=df.index)


def simulate_hard_v9_trades(df, result, eth_regime, max_hold_bars=100, fee_pct=0.1, slippage_pct=0.05):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    timestamps = df["timestamp"].values
    length = len(df)

    signal_idx = result.index[result["buy_signal"]].tolist()
    trades = []

    for i in signal_idx:
        if i + 1 >= length:
            continue

        sig_ts = pd.Timestamp(timestamps[i])
        if not is_eth_bullish_at(eth_regime, sig_ts):
            continue

        entry_bar = i + 1
        entry_price = open_[entry_bar] * (1 + slippage_pct / 100)
        sl_price = result["sl"].iloc[i]
        tp_price = result["tp"].iloc[i]

        if pd.isna(sl_price) or sl_price >= entry_price:
            continue

        exit_price, exit_reason = None, None
        for j in range(entry_bar, min(entry_bar + max_hold_bars, length)):
            if low[j] <= sl_price:
                exit_price, exit_reason = sl_price, "SL"
                break
            if high[j] >= tp_price:
                exit_price, exit_reason = tp_price, "TP"
                break

        if exit_price is None:
            last_bar = min(entry_bar + max_hold_bars - 1, length - 1)
            exit_price, exit_reason = close[last_bar], "TIME"

        exit_price *= (1 - slippage_pct / 100)
        net_return = (exit_price - entry_price) / entry_price * 100 - 2 * fee_pct
        trades.append({"return_pct": net_return, "exit_reason": exit_reason})

    return trades


def compute_eth_regime(eth_daily_df, ema_period=200):
    ema_series = eth_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = eth_daily_df["close"] > ema_series
    return pd.Series(is_bullish.values, index=pd.to_datetime(eth_daily_df["timestamp"]).values)


def is_eth_bullish_at(eth_regime, ts):
    valid = eth_regime[eth_regime.index <= ts]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])


def build_trend_series(df_1h):
    """Coin ke apne 4h/daily resampled data se trend series banata hai."""
    df_4h = df_1h.set_index("timestamp").resample("4h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()
    df_1d = df_1h.set_index("timestamp").resample("1D").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna().reset_index()

    ema50_4h = ema(df_4h["close"], 50)
    trend4h_4h_res = (df_4h["close"] > ema50_4h)
    ema20_1d = ema(df_1d["close"], 20)
    daily_bias_1d_res = (df_1d["close"] > ema20_1d)

    merged_4h = pd.merge_asof(
        df_1h[["timestamp"]].sort_values("timestamp"),
        pd.DataFrame({"timestamp": df_4h["timestamp"], "trend4h": trend4h_4h_res}).sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    merged_1d = pd.merge_asof(
        df_1h[["timestamp"]].sort_values("timestamp"),
        pd.DataFrame({"timestamp": df_1d["timestamp"], "daily_bias": daily_bias_1d_res}).sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    trend4h_series = merged_4h["trend4h"].fillna(False)
    daily_bias_series = merged_1d["daily_bias"].fillna(False)
    return trend4h_series, daily_bias_series


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
    log(f"  ETH bullish regime: {eth_regime.sum()}/{len(eth_regime)} din\n")

    coins = get_coin_list(exchange)[:N_FETCH_COINS]
    max_tier = max(TIERS)
    log(f"Scanning coins - pehle {max_tier} full-history coins tak...\n")

    trades_by_tier = {tier: [] for tier in TIERS}
    rank_included = 0

    for n, symbol in enumerate(coins):
        if rank_included >= max_tier:
            break
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=CANDLE_LIMIT)
        except Exception:
            df = None
        if df is None or len(df) < MIN_BARS_REQUIRED:
            continue
        rank_included += 1

        try:
            trend4h_series, daily_bias_series = build_trend_series(df)
            result = compute_full_signal_series(df, trend4h_series, daily_bias_series)
            trades = simulate_hard_v9_trades(df, result, eth_regime)
        except Exception as e:
            log(f"  [SKIP] {symbol}: {e}")
            trades = []

        for tier in TIERS:
            if rank_included <= tier:
                trades_by_tier[tier].extend(trades)

        if rank_included % 20 == 0:
            log(f"  rank {rank_included}/{max_tier} tak coins mil chuke (scan kiye: {n+1})")

    log(f"\n[INFO] Total {rank_included} full-history coins mile")

    log("\n" + "=" * 65)
    log("NATIJA: HARD V9 - Liquidity Tiers + ETH Filter")
    log("=" * 65)

    for tier in TIERS:
        stats = aggregate_trades(trades_by_tier[tier])
        log(f"\nTOP {tier}: Trades={stats['total_trades']}  Win%={stats['win_rate_pct']}  "
            f"PF={stats['profit_factor']}  Total Return={stats['total_return_pct']}%")

    log("\nNOTE: Pehle chhote test mein is strategy ka avg PF 1.83 tha magar")
    log("73% coins nuksan mein the (kuch outliers ne average khींच liya tha).")
    log("Agar yahan kisi tier ka PF consistently 1.5+ aata hai, to")
    log("liquidity+ETH recipe ne behtari di hai.")

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
            with open("hardv9_tiers_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(error_text)
            print("\n[SAVED] Error tafseel 'hardv9_tiers_ERROR.txt' mein save ho gayi.")
        except Exception:
            pass
        raise SystemExit(1)
