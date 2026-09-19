"""
Scheduled Dashboard Scan - GitHub Actions ke zariye har 1 ghanta khud
chalti hai.

Union AB signals ab DO TIERS mein baante jate hain (tasdeeq-shuda
backtest ke mutabiq):
    - BASELINE: ETH Regime Filter + RS (vs BTC) trend filter +
      RS Percentile>=95 pass karne wale signals (Win% ~48%, PF ~4)
    - BASELINE+52W: Baseline ke sath-sath qeemat apni 365-din
      (52-week) high ke 15% ke andar bhi ho (Win% ~57%, PF ~4.75)

Dono tiers alag alag track/notify hote hain - taake maloom rahe
konsi tier ka signal hai.

NEW_AdvancedConfluence_v1 system par ye naye filters LAGU NAHI hote -
wo pehle jaisa hi, bina kisi extra filter ke chalta hai.

HAR signal ke liye MUKAMMAL context (funding, OI, liquidity, multi-TF,
24h range, history, BTC correlation) nikalti hai, Overall Score/Rank
laga kar 'dashboard_signals.json' mein save karti hai.

Live section (Streamlit app mein) isi file ko seedha padh leta hai -
koi manual scan chalane ki zaroorat nahi.
"""

import json
import os
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import compute_atr
from confluence_engine import compute_confluence, DEFAULT_PARAMS as CONF_PARAMS
from dashboard_helpers import (
    get_fear_greed, get_futures_exchange, get_funding_and_oi, get_orderbook_info,
    get_tf_volume_change, get_24h_range_distance, get_historical_performance,
    get_btc_correlation, ALL_TIMEFRAMES, compute_overall_score,
)
from telegram_alert import send_telegram_alert

NTFY_TOPIC_STRONG = "asifali549-strong-signals-9k3m7x"


def send_strong_notification(title, message):
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC_STRONG}",
            data=message.encode("utf-8"),
            headers={"Title": title.encode("utf-8"), "Priority": "high", "Tags": "star"},
            timeout=10,
        )
    except Exception as e:
        print(f"  [NOTIFY FAILED] {e}")


def load_notified_keys():
    if not os.path.exists("notified_strong_signals.json"):
        return {}
    with open("notified_strong_signals.json") as f:
        return json.load(f)


def save_notified_keys(keys_dict):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    pruned = {k: v for k, v in keys_dict.items() if datetime.fromisoformat(v) > cutoff}
    with open("notified_strong_signals.json", "w") as f:
        json.dump(pruned, f, indent=2)


TOP_N_COINS = 150          # Tasdeeq-shuda: 400 se ghata kar 150 (zyada coins quality kam karte hain)
SIGNAL_TIMEFRAME = "1h"
CE_A = {"period": 16, "multiplier": 4.5}   # Tasdeeq-shuda: 3.0 se 4.5
CE_B = {"period": 12, "multiplier": 4.5}   # Tasdeeq-shuda: 3.0 se 4.5
CE_D = {"period": 16, "multiplier": 3.0}   # NEW system - jaisa pehle tha, tabdeel nahi
RR_MULTIPLE = 2.0

ETH_EMA_PERIOD = 200
RS_EMA_PERIOD = 50
RS_PERCENTILE_LOOKBACK_DAYS = 180
RS_PERCENTILE_CUTOFF = 95.0
HIGH_52W_LOOKBACK_BARS = 365 * 24
HIGH_52W_CUTOFF_PCT = 15.0
EXTENDED_1H_LIMIT = HIGH_52W_LOOKBACK_BARS + 50  # 52-week high nikalne ke liye poora saal ka data

COLORABLE_COLUMNS = (
    ["OrderBook Bid/Ask", "Liquidity Compare", "Coin's Own PF", "Funding Rate", "Dist from 24h High"]
    + [f"Vol {tf}" for tf in ALL_TIMEFRAMES]
    + [f"Chg {tf}" for tf in ALL_TIMEFRAMES]
)


def compute_eth_regime(eth_daily_df, ema_period=200):
    ema = eth_daily_df["close"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = eth_daily_df["close"] > ema
    return pd.Series(is_bullish.values, index=pd.to_datetime(eth_daily_df["timestamp"]).values)


def is_bullish_at(regime_series, ts):
    valid = regime_series[regime_series.index <= ts]
    if len(valid) == 0:
        return True
    return bool(valid.iloc[-1])


def compute_rs_trend_and_ratio(coin_daily_df, btc_daily_df, ema_period=50):
    coin_d = coin_daily_df[["timestamp", "close"]].rename(columns={"close": "coin_close"})
    btc_d = btc_daily_df[["timestamp", "close"]].rename(columns={"close": "btc_close"})
    merged = pd.merge_asof(
        coin_d.sort_values("timestamp"), btc_d.sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    merged["rs_ratio"] = merged["coin_close"] / merged["btc_close"].replace(0, np.nan)
    merged["rs_ema"] = merged["rs_ratio"].ewm(span=ema_period, adjust=False).mean()
    is_bullish = merged["rs_ratio"] > merged["rs_ema"]
    bullish_series = pd.Series(is_bullish.values, index=pd.to_datetime(merged["timestamp"]).values)
    ratio_series = pd.Series(merged["rs_ratio"].values, index=pd.to_datetime(merged["timestamp"]).values)
    return bullish_series, ratio_series


def percentile_rank_of_last(values):
    if len(values) < 2:
        return 50.0
    last = values[-1]
    return float((values <= last).sum()) / len(values) * 100


def compute_dist_from_52w_high(high_arr, close_arr, idx, lookback):
    window = high_arr[max(0, idx - lookback + 1):idx + 1]
    if len(window) == 0:
        return None
    hi = window.max()
    if hi <= 0:
        return None
    return (hi - close_arr[idx]) / hi * 100


def evaluate_union_ab_tier(exchange, symbol, df, idx, eth_regime, btc_daily):
    """
    Union AB ke ek signal (idx) ke liye tier decide karta hai:
    Returns (tier_or_None, extra_info_dict)
        tier_or_None: "Baseline", "Baseline+52W", ya None (reject)
    Sirf jitna zaroori ho utna hi fetch karta hai (jaldi khatam hone
    wale checks pehle - ETH check jo already fetched hai, phir RS
    (coin daily fetch), phir 52W (extended 1h fetch) - sirf agar
    pichle sab pass ho chuke hon.
    """
    sig_ts = pd.Timestamp(df["timestamp"].iloc[idx])

    if not is_bullish_at(eth_regime, sig_ts):
        return None, {}

    try:
        coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=RS_PERCENTILE_LOOKBACK_DAYS + 60)
        rs_bullish, rs_ratio_series = compute_rs_trend_and_ratio(coin_daily, btc_daily, ema_period=RS_EMA_PERIOD)
    except Exception:
        return None, {}

    if not is_bullish_at(rs_bullish, sig_ts):
        return None, {}

    rs_valid = rs_ratio_series[rs_ratio_series.index <= sig_ts]
    rs_recent = rs_valid.iloc[-RS_PERCENTILE_LOOKBACK_DAYS:] if len(rs_valid) > 0 else pd.Series([], dtype=float)
    if len(rs_recent) < 2:
        return None, {}
    rs_pct = percentile_rank_of_last(rs_recent.values)
    if rs_pct < RS_PERCENTILE_CUTOFF:
        return None, {}

    # Yahan tak pahunche matlab kam az kam BASELINE tier pass ho chuki hai
    extra_info = {"RS Percentile": round(rs_pct, 1)}

    try:
        df_extended = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=EXTENDED_1H_LIMIT)
        if df_extended is not None and len(df_extended) >= 30:
            # Extended data ke andar wahi signal timestamp dhoondo
            ext_ts = pd.to_datetime(df_extended["timestamp"])
            matches = df_extended.index[ext_ts == sig_ts]
            if len(matches) > 0:
                ext_idx = matches[0]
                dist_52w = compute_dist_from_52w_high(
                    df_extended["high"].values, df_extended["close"].values,
                    ext_idx, HIGH_52W_LOOKBACK_BARS,
                )
                if dist_52w is not None:
                    extra_info["Dist from 52W High"] = f"{round(dist_52w, 2)}%"
                    if dist_52w <= HIGH_52W_CUTOFF_PCT:
                        return "Baseline+52W", extra_info
    except Exception:
        pass

    return "Baseline", extra_info


def main():
    exchange = get_exchange()
    futures_exchange = get_futures_exchange()

    fg_value, fg_label = get_fear_greed()

    print("BTC daily benchmark data fetch kar rahe hain...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=800)

    print("ETH daily data + 200 EMA regime nikal rahe hain...")
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=800)
    eth_regime = compute_eth_regime(eth_daily, ema_period=ETH_EMA_PERIOD)

    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"Scanning {len(coins)} coins on {SIGNAL_TIMEFRAME}...")

    signal_coins = []
    for symbol in coins:
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=max(config.CANDLE_LIMITS.get(SIGNAL_TIMEFRAME, 500), 300))
        except Exception:
            df = None
        if df is None or len(df) < 220:
            continue
        try:
            ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
            ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
            ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
            breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)
            combo_a = ichi_sig & ms_sig
            combo_b = ema_sig & breakout_sig

            for combo_sig, combo_name, ce in [(combo_a, "Ichimoku+MS", CE_A), (combo_b, "EMA+Breakout", CE_B)]:
                if combo_sig.tail(3).any():
                    idx = combo_sig.tail(3)[combo_sig.tail(3)].index[-1]

                    tier, extra_info = evaluate_union_ab_tier(exchange, symbol, df, idx, eth_regime, btc_daily)
                    if tier is None:
                        continue  # ETH/RS/RS-Percentile filter fail - signal reject

                    atr = compute_atr(df, ce["period"])
                    highest_high = df["high"].rolling(ce["period"]).max()
                    chandelier = (highest_high - ce["multiplier"] * atr).loc[idx]
                    entry_price = df.loc[idx, "close"]
                    current_price = df["close"].iloc[-1]
                    risk = entry_price - chandelier
                    tp_price = entry_price + risk * RR_MULTIPLE
                    row = {
                        "Coin": symbol, "Combo": combo_name, "Tier": tier, "Bars Ago": int(len(df) - 1 - idx),
                        "Entry": round(float(entry_price), 6), "Current": round(float(current_price), 6),
                        "Trail Stop": round(float(chandelier), 6), "Take Profit": round(float(tp_price), 6),
                        "_df": df, "_sig": combo_sig, "_ce": ce,
                    }
                    row.update(extra_info)
                    signal_coins.append(row)
        except Exception as e:
            print(f"  [SKIP-OLD] {symbol}: {e}")

        # ---- NEW system: AdvancedConfluence_v1 (bina kisi naye filter ke, jaisa pehle tha) ----
        try:
            result_new = compute_confluence(df, btc_daily, CONF_PARAMS, usdt_d_weak=None)
            # CHoCH-Only: bara-sample confirmation test se tasdeeq shuda
            # (Top 300 coins, period-split consistent) - BOS wale signals
            # hata diye gaye, kyunke BOS wale weak the (PF 0.728) aur CHoCH
            # akela mazboot tha (poora sample PF 1.215->1.708, Period 1 PF
            # 1.763, Period 2 PF 1.687 - dono consistent).
            choch_signal = result_new["choch"]
            new_sig = apply_cooldown(choch_signal & (result_new["score"] >= 6), config.SIGNAL_COOLDOWN_BARS)

            if new_sig.tail(3).any():
                idx = new_sig.tail(3)[new_sig.tail(3)].index[-1]
                atr = compute_atr(df, CE_D["period"])
                highest_high = df["high"].rolling(CE_D["period"]).max()
                chandelier = (highest_high - CE_D["multiplier"] * atr).loc[idx]
                entry_price = df.loc[idx, "close"]
                current_price = df["close"].iloc[-1]
                risk = entry_price - chandelier
                tp_price = entry_price + risk * RR_MULTIPLE
                signal_coins.append({
                    "Coin": symbol, "Combo": "NEW_AdvancedConfluence (CHoCH)", "Tier": "N/A", "Bars Ago": int(len(df) - 1 - idx),
                    "Entry": round(float(entry_price), 6), "Current": round(float(current_price), 6),
                    "Trail Stop": round(float(chandelier), 6), "Take Profit": round(float(tp_price), 6),
                    "_df": df, "_sig": new_sig, "_ce": CE_D,
                })
        except Exception as e:
            print(f"  [SKIP-NEW] {symbol}: {e}")

    print(f"Found {len(signal_coins)} signals (filters ke baad). Computing full context...")

    final_rows = []
    for row in signal_coins:
        symbol = row["Coin"]
        final_row = {k: v for k, v in row.items() if not k.startswith("_")}

        funding, oi = get_funding_and_oi(futures_exchange, symbol)
        final_row["Funding Rate"] = f"{funding*100:.3f}%" if funding is not None else "N/A"
        final_row["Open Interest"] = f"${oi:,.0f}" if oi is not None else "N/A"

        bid_liq, ask_liq, ob_ratio = get_orderbook_info(exchange, symbol)
        final_row["Liquidity Up ($)"] = f"${ask_liq:,.0f}" if ask_liq is not None else "N/A"
        final_row["Liquidity Down ($)"] = f"${bid_liq:,.0f}" if bid_liq is not None else "N/A"
        final_row["Liquidity Compare"] = ("Support > Resistance" if bid_liq > ask_liq else "Resistance > Support") if (bid_liq is not None and ask_liq is not None) else "N/A"
        final_row["OrderBook Bid/Ask"] = f"{ob_ratio:.2f}x" if ob_ratio is not None else "N/A"

        for tf in ALL_TIMEFRAMES:
            vol_ratio, chg = get_tf_volume_change(exchange, symbol, tf)
            final_row[f"Vol {tf}"] = f"{vol_ratio:.2f}x" if vol_ratio is not None else "N/A"
            final_row[f"Chg {tf}"] = f"{chg:+.2f}%" if chg is not None else "N/A"

        dist_high, dist_low = get_24h_range_distance(exchange, symbol)
        final_row["Dist from 24h High"] = f"{dist_high}%" if dist_high is not None else "N/A"
        final_row["Dist from 24h Low"] = f"{dist_low}%" if dist_low is not None else "N/A"

        win_rate, pf = get_historical_performance(row["_df"], row["_sig"], row["_ce"])
        final_row["Coin's Own Win%"] = f"{win_rate}%" if win_rate is not None else "N/A"
        final_row["Coin's Own PF"] = f"{pf}" if pf is not None else "N/A"

        corr = get_btc_correlation(exchange, symbol, SIGNAL_TIMEFRAME)
        final_row["BTC Correlation"] = f"{corr}" if corr is not None else "N/A"

        score, verdict = compute_overall_score(final_row, COLORABLE_COLUMNS)
        final_row["Overall Score %"] = score
        final_row["Verdict"] = verdict

        final_rows.append(final_row)

    # Best-to-worst rank lagayein
    final_rows.sort(key=lambda r: r["Overall Score %"], reverse=True)
    for i, r in enumerate(final_rows):
        r["Rank"] = i + 1

    output = {
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
        "coins_scanned": len(coins),
        "fear_greed_value": fg_value,
        "fear_greed_label": fg_label,
        "signals": final_rows,
    }

    with open("dashboard_signals.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. {len(final_rows)} signals with full context saved to dashboard_signals.json")

    # Notifications: "Strong" verdict (jaisa pehle) + har Union AB Tier ka apna alag paigham
    notified = load_notified_keys()
    new_count = 0
    for row in final_rows:
        tier = row.get("Tier", "N/A")
        is_new_strong = "Strong" in row.get("Verdict", "")
        is_union_ab_tier = tier in ("Baseline", "Baseline+52W")

        if not (is_new_strong or is_union_ab_tier):
            continue

        key = f"{row['Coin']}|{row['Combo']}|{row.get('Entry')}|{tier}"
        if key in notified:
            continue

        if tier == "Baseline+52W":
            emoji = "🏆"
        elif tier == "Baseline":
            emoji = "📊"
        else:
            emoji = "⭐"

        title = f"{emoji} {tier if is_union_ab_tier else 'STRONG'}: {row['Coin']} - {row['Combo']}"
        message = (
            f"Overall Score: {row['Overall Score %']}% ({row['Verdict']})\n"
            f"Entry: {row['Entry']}\n"
            f"Take Profit: {row['Take Profit']}\n"
            f"Trail Stop: {row['Trail Stop']}"
        )
        if "RS Percentile" in row:
            message += f"\nRS Percentile: {row['RS Percentile']}"
        if "Dist from 52W High" in row:
            message += f"\nDist from 52W High: {row['Dist from 52W High']}"

        send_strong_notification(title, message)
        send_telegram_alert(f"<b>{title}</b>\n{message}")
        notified[key] = datetime.now(timezone.utc).isoformat()
        new_count += 1

    save_notified_keys(notified)
    print(f"Notifications bheji: {new_count}")


if __name__ == "__main__":
    main()
