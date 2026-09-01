"""
Backtest Engine - Signals ko actual trades mein simulate karta hai
(Spot / Buy-only: har trade ek LONG entry hai, SHORT nahi)

Exit rules:
  - Stop Loss  = entry - (ATR * stop_loss_atr_mult)
  - Take Profit= entry + (ATR * take_profit_atr_mult)
  - Time exit  = max_hold_bars ke baad, jo bhi pehle aaye
Fees + slippage dono taraf (entry aur exit) par lagte hain.
"""

import numpy as np
import pandas as pd


def compute_atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_chandelier_long_stop(df, period, multiplier):
    """
    Chandelier Exit (long side) — trailing stop:
        stop = Highest High(period) - multiplier * ATR(period)
    Ye har bar par calculate hota hai; trade ke andar hum isay
    "sirf upar ja sakta hai, neeche nahi" wale trailing rule ke sath istemal karte hain.
    """
    atr = compute_atr(df, period)
    highest_high = df["high"].rolling(period).max()
    return highest_high - (multiplier * atr)


def simulate_trades(df, signal, bt_params, ce_params=None):
    """
    df: OHLCV dataframe
    signal: boolean Series, True = fresh buy signal on that bar
    bt_params: config.BACKTEST_PARAMS
    ce_params: config.CHANDELIER_EXIT_PARAMS (sirf exit_mode="chandelier" ke liye)

    Return: list of trade dicts
    """
    exit_mode = bt_params.get("exit_mode", "fixed_atr")
    atr = compute_atr(df, bt_params["atr_period"])
    fee = bt_params["fee_pct"] / 100
    slip = bt_params["slippage_pct"] / 100
    max_hold = bt_params["max_hold_bars"]

    if exit_mode == "chandelier":
        ce_stop = compute_chandelier_long_stop(df, ce_params["period"], ce_params["multiplier"]).values

    trades = []
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr_vals = atr.values
    n = len(df)

    signal_idx = np.where(signal.values)[0]

    for i in signal_idx:
        if i + 1 >= n or np.isnan(atr_vals[i]):
            continue

        entry_bar = i + 1  # signal candle ke close par decide, agle bar ke open par entry (zyada realistic)
        entry_price = df["open"].values[entry_bar] * (1 + slip)

        exit_price = None
        exit_bar = None
        exit_reason = None

        if exit_mode == "chandelier":
            # Trailing stop: shuru mein entry bar ka CE stop, phir sirf upar trail hoga
            if np.isnan(ce_stop[i]):
                continue
            trail_stop = ce_stop[i]

            for j in range(entry_bar, min(entry_bar + max_hold, n)):
                if not np.isnan(ce_stop[j]):
                    trail_stop = max(trail_stop, ce_stop[j])  # sirf upar trail ho, neeche nahi

                if low[j] <= trail_stop:
                    exit_price = trail_stop
                    exit_bar = j
                    exit_reason = "CE_STOP"
                    break

            if exit_price is None:
                last_bar = min(entry_bar + max_hold - 1, n - 1)
                exit_price = close[last_bar]
                exit_bar = last_bar
                exit_reason = "TIME"

        else:
            risk = atr_vals[i] * bt_params["stop_loss_atr_mult"]
            reward = atr_vals[i] * bt_params["take_profit_atr_mult"]
            sl_price = entry_price - risk
            tp_price = entry_price + reward

            for j in range(entry_bar, min(entry_bar + max_hold, n)):
                if low[j] <= sl_price:
                    exit_price = sl_price
                    exit_bar = j
                    exit_reason = "SL"
                    break
                if high[j] >= tp_price:
                    exit_price = tp_price
                    exit_bar = j
                    exit_reason = "TP"
                    break

            if exit_price is None:
                last_bar = min(entry_bar + max_hold - 1, n - 1)
                exit_price = close[last_bar]
                exit_bar = last_bar
                exit_reason = "TIME"

        exit_price = exit_price * (1 - slip)

        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - (2 * fee)  # entry + exit fee

        trades.append({
            "entry_time": df["timestamp"].iloc[entry_bar],
            "exit_time": df["timestamp"].iloc[exit_bar],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "bars_held": exit_bar - entry_bar,
            "return_pct": net_return * 100,
            "exit_reason": exit_reason,
        })

    return trades


def summarize_trades(trades, symbol, timeframe, strategy_name):
    if len(trades) == 0:
        return {
            "symbol": symbol, "timeframe": timeframe, "strategy": strategy_name,
            "total_trades": 0, "win_rate_pct": None, "avg_return_pct": None,
            "total_return_pct": None, "profit_factor": None,
            "avg_bars_held": None, "expectancy_pct": None,
        }

    returns = np.array([t["return_pct"] for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    win_rate = len(wins) / len(returns) * 100
    gross_profit = wins.sum() if len(wins) else 0
    gross_loss = abs(losses.sum()) if len(losses) else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy_name,
        "total_trades": len(trades),
        "win_rate_pct": round(win_rate, 2),
        "avg_return_pct": round(returns.mean(), 3),
        "total_return_pct": round(returns.sum(), 2),
        "profit_factor": round(profit_factor, 2) if not np.isnan(profit_factor) else None,
        "avg_bars_held": round(np.mean([t["bars_held"] for t in trades]), 1),
        "expectancy_pct": round(returns.mean(), 3),
    }