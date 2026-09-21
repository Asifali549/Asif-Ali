"""
Scheduled Dashboard Scan - GitHub Actions ke zariye khud chalti hai
(scan -> commit -> 5 min wait -> agla scan khud shuru, self-loop).

AB CHAAR (4) ALAG ALAG, AZAD (independent) SYSTEMS chalte hain, har ek
apne signals khud detect karta hai aur dashboard par ALAG dikhta hai:

    1) Union AB (Baseline + Baseline+52W)
       Ichimoku+MarketStructure ya EMA+Breakout combo, ETH Regime +
       RS Filter + RS Percentile>=95 (+ optional 52W High<=15%) ke
       sath filtered. (Sab se purana, sab se zyada tasdeeq-shuda.)

    2) NEW AdvancedConfluence (CHoCH-Only)
       Chandelier(16,3.0) CHoCH signal + score>=6. Koi extra filter
       nahi (tasdeeq-shuda: extra filters yahan nuqsan-deh sabit hue).

    3) Union AB Backup Tier (NAYA)
       Wahi Union AB combos, sirf RS Filter + RS Percentile>=95 pass
       karte hain - ETH Regime check NAHI (hamesha active rehta hai,
       chahe ETH bearish ho). Period-split se tasdeeq-shuda (PF 3.04
       poora sample, 1.92/3.87 dono half consistent).

    4) CE Buy-Only (NAYA)
       Sirf Chandelier Exit cross-above (entry: period=11, mult=4.5;
       exit: period=16, mult=3.0) - koi ETH/RS/Volume filter nahi.
       Walk-forward (4 sequential folds, 1 saal) se tasdeeq-shuda,
       Period=11 sab se mustahkam nikla. NOTE: sirf 1 indicator par
       mabni hai, Union AB jitna "mehfooz" nahi - isi liye alag/halka
       system ke tor par rakha gaya hai.

HAR system ke signals mein do "Category" hoti hain:
    - "New Signal"  -> abhi (pichle 3 candles mein) bana
    - "Open Trade"  -> pehle bana tha, abhi tak SL/TP hit nahi hua
      (isliye chalta/dikhta rehta hai)
Jaise hi koi trade SL ya TP hit kar le, wo agli scan se list se apne
aap hat jati hai (status dobara, HAR baar, taaza data se calculate
hota hai - koi alag "state" file rakhne ki zaroorat nahi).

Union AB aur NEW system ke signals ke liye MUKAMMAL context (funding,
OI, liquidity, multi-TF, 24h range, history, BTC correlation, Overall
Score) nikalti hai. Backup Tier aur CE Buy-Only ke liye YE HALKA/LIGHT
rakha gaya hai (sirf zaroori columns) - kyunke CE Buy-Only mein bohot
zyada signals aa sakte hain (backtest mein per-coin kaafi frequent),
poore context ke liye har signal par 8-10 extra API calls lagane se
scan bohot slow ho jata aur rate-limit ka khatra badh jata.

Result 'dashboard_signals.json' mein save hota hai. Live section
(Streamlit app) isi file ko seedha padh leta hai.
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
from backtest_engine import compute_atr, compute_chandelier_long_stop
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


TOP_N_COINS = 150          # 4 systems, sab isi 150-coin list par (jo abhi live hai)
SIGNAL_TIMEFRAME = "1h"
CE_A = {"period": 16, "multiplier": 4.5}   # Union AB: Ichimoku+MS
CE_B = {"period": 12, "multiplier": 4.5}   # Union AB: EMA+Breakout
CE_D = {"period": 16, "multiplier": 3.0}   # NEW system (CHoCH) - jaisa pehle tha

CE_BUYONLY_ENTRY = {"period": 11, "multiplier": 4.5}   # Tasdeeq-shuda: walk-forward, Period=11 sab se mustahkam
CE_BUYONLY_EXIT = {"period": 16, "multiplier": 3.0}

RR_MULTIPLE = 2.0

ETH_EMA_PERIOD = 200
RS_EMA_PERIOD = 50
RS_PERCENTILE_LOOKBACK_DAYS = 180
RS_PERCENTILE_CUTOFF = 95.0
HIGH_52W_LOOKBACK_BARS = 365 * 24
HIGH_52W_CUTOFF_PCT = 15.0
EXTENDED_1H_LIMIT = HIGH_52W_LOOKBACK_BARS + 50  # 52-week high nikalne ke liye poora saal ka data

# "Open trade" dhoondne ke liye kitni purani candles tak peeche dekhein
OPEN_TRADE_LOOKBACK_BARS = 500     # ~20 din (1h par) - is se zyada purani trade ab tak khud-b-khud band ho chuki hogi
NEW_SIGNAL_WINDOW_BARS = 3         # itni "bars ago" tak = "New Signal", is se zyada purani lekin abhi tak OPEN = "Open Trade"

COLORABLE_COLUMNS = (
    ["OrderBook Bid/Ask", "Liquidity Compare", "Coin's Own PF", "Funding Rate", "Dist from 24h High"]
    + [f"Vol {tf}" for tf in ALL_TIMEFRAMES]
    + [f"Chg {tf}" for tf in ALL_TIMEFRAMES]
)


def to_pkt_str(ts):
    """Exchange ka timestamp UTC hota hai - Pakistan Time (UTC+5) mein convert karta hai."""
    ts_utc = pd.Timestamp(ts)
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    ts_pkt = ts_utc.tz_convert("Asia/Karachi")
    return ts_pkt.strftime("%Y-%m-%d %I:%M %p PKT")


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


def passes_rs_filters(sig_ts, rs_bullish_series, rs_ratio_series):
    """RS trend (vs BTC) + RS Percentile>=95 - ETH check shamil NAHI."""
    if not is_bullish_at(rs_bullish_series, sig_ts):
        return False, None
    rs_valid = rs_ratio_series[rs_ratio_series.index <= sig_ts]
    rs_recent = rs_valid.iloc[-RS_PERCENTILE_LOOKBACK_DAYS:] if len(rs_valid) > 0 else pd.Series([], dtype=float)
    if len(rs_recent) < 2:
        return False, None
    rs_pct = percentile_rank_of_last(rs_recent.values)
    if rs_pct < RS_PERCENTILE_CUTOFF:
        return False, None
    return True, round(rs_pct, 1)


def evaluate_union_ab_tier(symbol, df, idx, eth_regime, rs_bullish, rs_ratio_series, exchange):
    """
    Union AB Baseline/Baseline+52W tier decide karta hai. rs_bullish aur
    rs_ratio_series pehle se (ek hi baar, coin-level par) compute ki hui
    di jati hain - taake har combo ke liye dobara fetch na karna pade.
    """
    sig_ts = pd.Timestamp(df["timestamp"].iloc[idx])

    if not is_bullish_at(eth_regime, sig_ts):
        return None, {}

    rs_ok, rs_pct = passes_rs_filters(sig_ts, rs_bullish, rs_ratio_series)
    if not rs_ok:
        return None, {}

    extra_info = {"RS Percentile": rs_pct}

    try:
        df_extended = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=EXTENDED_1H_LIMIT)
        if df_extended is not None and len(df_extended) >= 30:
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


# ---------------- Trade progress / status (SAB systems ke liye saanjha) ----------------
def compute_trade_progress(df, signal_idx, ce_period, ce_multiplier, rr_multiple=RR_MULTIPLE,
                            entry_ce_period=None, entry_ce_multiplier=None):
    """
    Ek signal (signal_idx par bani) ka AAJ TAK ka poora safar dobara,
    taaza data se calculate karta hai - koi state file ki zaroorat nahi.

    Trailing-stop line ko signal_idx se lekar AAKHRI bar tak, bar-bar-bar
    "sirf upar ja sakti hai" wale trail rule ke sath chalate hain. Jaise
    hi kisi bar ka LOW us waqt ke trail-stop ko chhue/neeche jaye -> SL
    hit (CLOSED/STOPPED). Jaise hi kisi bar ka HIGH take-profit ko
    chhue/upar jaye -> TARGET hit (CLOSED/TARGET). Dono ek sath ho jayein
    to conservative tareeqe se SL ko pehle mana jata hai. Agar aakhri bar
    tak kuch hit na ho -> abhi tak OPEN hai.

    entry_ce_period/entry_ce_multiplier: JAB signal khud chandelier-cross
    se bana ho aur uske apne (entry) params trailing/exit params
    (ce_period, ce_multiplier) se ALAG hon (jaise CE Buy-Only: entry
    11/4.5, exit 16/3.0) - to "initial stop < entry price" wali validity
    check ENTRY params se honi chahiye (kyunke wahi cross ko guarantee
    deti hai), warna EXIT params ka zyada tight (chhota multiplier)
    stop is check ko GHALAT taur par fail kar deta hai aur bilkul theek
    signal bhi "invalid" keh kar chupa deta hai. Trailing phir bhi
    exit params (ce_period, ce_multiplier) se hi hoti hai - sirf shuruati
    validity check ka reference alag hai. Agar ye do params nahi diye
    jayen, purana rawaiyya (dono ek hi) barqarar rehta hai.

    Returns None agar setup invalid ho (chandelier NaN ya stop>=entry).
    """
    atr = compute_atr(df, ce_period)
    highest_high = df["high"].rolling(ce_period).max()
    chandelier_series = highest_high - ce_multiplier * atr

    entry_p = entry_ce_period if entry_ce_period is not None else ce_period
    entry_m = entry_ce_multiplier if entry_ce_multiplier is not None else ce_multiplier
    if entry_p == ce_period and entry_m == ce_multiplier:
        entry_stop_series = chandelier_series
    else:
        entry_atr = compute_atr(df, entry_p)
        entry_highest_high = df["high"].rolling(entry_p).max()
        entry_stop_series = entry_highest_high - entry_m * entry_atr

    entry_price = float(df["close"].iloc[signal_idx])
    initial_stop = entry_stop_series.iloc[signal_idx]
    if pd.isna(initial_stop) or initial_stop >= entry_price:
        return None

    risk = entry_price - float(initial_stop)
    tp_price = entry_price + risk * rr_multiple

    running_stop = float(initial_stop)
    status = "OPEN"
    exit_reason = None
    exit_idx = None

    for i in range(signal_idx + 1, len(df)):
        bar_stop = chandelier_series.iloc[i]
        if not pd.isna(bar_stop) and bar_stop > running_stop:
            running_stop = float(bar_stop)

        low_i = df["low"].iloc[i]
        high_i = df["high"].iloc[i]
        stop_hit = low_i <= running_stop
        tp_hit = high_i >= tp_price

        if stop_hit:
            exit_idx, exit_reason, status = i, "STOPPED", "CLOSED"
            break
        elif tp_hit:
            exit_idx, exit_reason, status = i, "TARGET", "CLOSED"
            break

    current_price = float(df["close"].iloc[-1])
    pnl_pct = (current_price - entry_price) / entry_price * 100

    return {
        "status": status,
        "exit_reason": exit_reason,
        "exit_idx": exit_idx,
        "entry_price": round(entry_price, 6),
        "current_price": round(current_price, 6),
        "trail_stop": round(running_stop, 6),
        "tp_price": round(tp_price, 6),
        "pnl_pct": round(pnl_pct, 2),
        "bars_since_entry": int(len(df) - 1 - signal_idx),
    }


def find_latest_open_or_new(df, signal_series, ce_period, ce_multiplier, lookback_bars=OPEN_TRADE_LOOKBACK_BARS,
                             entry_ce_period=None, entry_ce_multiplier=None):
    """
    signal_series mein sab se AAKHRI (most recent) True index dhoondta
    hai (sirf pichli `lookback_bars` candles ke andar), phir uska trade
    progress nikalta hai. Agar wo signal abhi tak OPEN hai (band nahi
    hua), to use return karta hai (category = NEW ya OPEN, bars-ago ke
    hisab se). Agar CLOSED ho chuka hai to None (kuch bhi dikhane ki
    zaroorat nahi - purani trade khatam ho chuki).

    entry_ce_period/entry_ce_multiplier: compute_trade_progress ko
    aage forward - dekho waha ke docstring mein wajah.
    """
    n = len(signal_series)
    start = max(0, n - lookback_bars)
    recent_window = signal_series.iloc[start:]
    true_idxs = recent_window[recent_window].index
    if len(true_idxs) == 0:
        return None

    signal_idx = true_idxs[-1]
    progress = compute_trade_progress(
        df, signal_idx, ce_period, ce_multiplier,
        entry_ce_period=entry_ce_period, entry_ce_multiplier=entry_ce_multiplier,
    )
    if progress is None or progress["status"] != "OPEN":
        return None

    category = "New Signal" if progress["bars_since_entry"] <= NEW_SIGNAL_WINDOW_BARS else "Open Trade"
    return {
        "signal_idx": signal_idx,
        "signal_timestamp": df["timestamp"].iloc[signal_idx],
        "category": category,
        **progress,
    }


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
    print(f"Scanning {len(coins)} coins on {SIGNAL_TIMEFRAME} (4 systems)...")

    heavy_rows = []      # Union AB + NEW system - poora context milega
    light_rows = []      # Union AB Backup Tier + CE Buy-Only - halka rakha gaya

    for symbol in coins:
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=max(config.CANDLE_LIMITS.get(SIGNAL_TIMEFRAME, 500), 300))
        except Exception:
            df = None
        if df is None or len(df) < 220:
            continue

        # ---- Ek hi baar, coin-level par RS data nikal lete hain (Union AB + Backup Tier dono isay istemal karenge) ----
        rs_bullish, rs_ratio_series = None, None
        try:
            coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=RS_PERCENTILE_LOOKBACK_DAYS + 60)
            rs_bullish, rs_ratio_series = compute_rs_trend_and_ratio(coin_daily, btc_daily, ema_period=RS_EMA_PERIOD)
        except Exception as e:
            print(f"  [RS-FAIL] {symbol}: {e}")

        # ================= SYSTEM 1: Union AB (Baseline / Baseline+52W) =================
        # ================= SYSTEM 3: Union AB Backup Tier (RS+RS%95, ETH NAHI) ===========
        try:
            ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
            ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
            ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
            breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)
            combo_a = ichi_sig & ms_sig
            combo_b = ema_sig & breakout_sig

            for combo_sig, combo_name, ce in [(combo_a, "Ichimoku+MS", CE_A), (combo_b, "EMA+Breakout", CE_B)]:
                # --- Union AB Baseline/Baseline+52W: sirf sab se aakhri open/new signal ---
                found = find_latest_open_or_new(df, combo_sig, ce["period"], ce["multiplier"])
                if found is not None:
                    idx = found["signal_idx"]
                    tier, extra_info = (None, {})
                    if rs_bullish is not None:
                        tier, extra_info = evaluate_union_ab_tier(symbol, df, idx, eth_regime, rs_bullish, rs_ratio_series, exchange)
                    if tier is not None:
                        row = {
                            "System": "Union AB", "Coin": symbol, "Combo": combo_name, "Tier": tier,
                            "Category": found["category"],
                            "Signal Time (PKT)": to_pkt_str(found["signal_timestamp"]),
                            "Bars Ago": found["bars_since_entry"],
                            "Entry": found["entry_price"], "Current": found["current_price"],
                            "P/L %": found["pnl_pct"],
                            "Trail Stop": found["trail_stop"], "Take Profit": found["tp_price"],
                            "_df": df, "_sig": combo_sig, "_ce": ce,
                        }
                        row.update(extra_info)
                        heavy_rows.append(row)

                # --- Union AB Backup Tier: RS+RS%95 ke sath, ETH check NAHI ---
                if rs_bullish is not None and found is not None:
                    idx = found["signal_idx"]
                    sig_ts = pd.Timestamp(df["timestamp"].iloc[idx])
                    backup_ok, rs_pct = passes_rs_filters(sig_ts, rs_bullish, rs_ratio_series)
                    if backup_ok:
                        light_rows.append({
                            "System": "Union AB Backup Tier", "Coin": symbol, "Combo": combo_name, "Tier": "N/A",
                            "Category": found["category"],
                            "Signal Time (PKT)": to_pkt_str(found["signal_timestamp"]),
                            "Bars Ago": found["bars_since_entry"],
                            "Entry": found["entry_price"], "Current": found["current_price"],
                            "P/L %": found["pnl_pct"],
                            "Trail Stop": found["trail_stop"], "Take Profit": found["tp_price"],
                            "RS Percentile": rs_pct,
                        })
        except Exception as e:
            print(f"  [SKIP-OLD] {symbol}: {e}")

        # ================= SYSTEM 2: NEW AdvancedConfluence (CHoCH-Only) =================
        try:
            result_new = compute_confluence(df, btc_daily, CONF_PARAMS, usdt_d_weak=None)
            choch_signal = result_new["choch"]
            new_sig = apply_cooldown(choch_signal & (result_new["score"] >= CONF_PARAMS["score_threshold"]), config.SIGNAL_COOLDOWN_BARS)

            found = find_latest_open_or_new(df, new_sig, CE_D["period"], CE_D["multiplier"])
            if found is not None:
                heavy_rows.append({
                    "System": "NEW AdvancedConfluence", "Coin": symbol, "Combo": "CHoCH", "Tier": "N/A",
                    "Category": found["category"],
                    "Signal Time (PKT)": to_pkt_str(found["signal_timestamp"]),
                    "Bars Ago": found["bars_since_entry"],
                    "Entry": found["entry_price"], "Current": found["current_price"],
                    "P/L %": found["pnl_pct"],
                    "Trail Stop": found["trail_stop"], "Take Profit": found["tp_price"],
                    "_df": df, "_sig": new_sig, "_ce": CE_D,
                })
        except Exception as e:
            print(f"  [SKIP-NEW] {symbol}: {e}")

        # ================= SYSTEM 4: CE Buy-Only (halka, koi filter nahi) =================
        try:
            entry_stop = compute_chandelier_long_stop(df, CE_BUYONLY_ENTRY["period"], CE_BUYONLY_ENTRY["multiplier"])
            close = df["close"]
            cross_above = (close > entry_stop) & (close.shift(1) <= entry_stop.shift(1))
            ce_sig = apply_cooldown(cross_above.fillna(False), config.SIGNAL_COOLDOWN_BARS)

            found = find_latest_open_or_new(
                df, ce_sig, CE_BUYONLY_EXIT["period"], CE_BUYONLY_EXIT["multiplier"],
                entry_ce_period=CE_BUYONLY_ENTRY["period"], entry_ce_multiplier=CE_BUYONLY_ENTRY["multiplier"],
            )
            if found is not None:
                light_rows.append({
                    "System": "CE Buy-Only", "Coin": symbol, "Combo": "Chandelier Cross", "Tier": "N/A",
                    "Category": found["category"],
                    "Signal Time (PKT)": to_pkt_str(found["signal_timestamp"]),
                    "Bars Ago": found["bars_since_entry"],
                    "Entry": found["entry_price"], "Current": found["current_price"],
                    "P/L %": found["pnl_pct"],
                    "Trail Stop": found["trail_stop"], "Take Profit": found["tp_price"],
                })
        except Exception as e:
            print(f"  [SKIP-CE] {symbol}: {e}")

    print(f"Found: Union AB/NEW={len(heavy_rows)}, Backup Tier/CE Buy-Only={len(light_rows)}. Computing full context (heavy rows)...")

    final_rows = []
    for row in heavy_rows:
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

    # Light rows ko score/verdict "N/A" ke sath as-is shamil karte hain
    for row in light_rows:
        row["Overall Score %"] = "N/A"
        row["Verdict"] = "N/A"
        final_rows.append(row)

    # Best-to-worst rank (sirf scored/heavy rows ka rank maayne rakhta hai)
    final_rows.sort(key=lambda r: (r["Overall Score %"] if isinstance(r["Overall Score %"], (int, float)) else -1), reverse=True)
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

    print(f"\nDone. {len(final_rows)} signals with context saved to dashboard_signals.json")

    # ---- Notifications: sirf "New Signal" category par (Open Trade dobara notify nahi hoti) ----
    notified = load_notified_keys()
    new_count = 0
    for row in final_rows:
        if row.get("Category") != "New Signal":
            continue

        system = row.get("System", "")
        tier = row.get("Tier", "N/A")
        is_new_strong = "Strong" in str(row.get("Verdict", ""))
        is_union_ab_tier = tier in ("Baseline", "Baseline+52W")
        is_new_system = ("CE Buy-Only" in system) or ("Backup Tier" in system)

        if not (is_new_strong or is_union_ab_tier or is_new_system):
            continue

        key = f"{row['Coin']}|{system}|{row['Combo']}|{row.get('Entry')}|{tier}"
        if key in notified:
            continue

        if tier == "Baseline+52W":
            emoji = "🏆"
        elif tier == "Baseline":
            emoji = "📊"
        elif system == "Union AB Backup Tier":
            emoji = "🛡️"
        elif system == "CE Buy-Only":
            emoji = "⚡"
        else:
            emoji = "⭐"

        title = f"{emoji} {system}: {row['Coin']} - {row['Combo']}"
        message = (
            f"Signal Time: {row.get('Signal Time (PKT)', 'N/A')}\n"
            f"Entry: {row['Entry']}\n"
            f"Take Profit: {row['Take Profit']}\n"
            f"Trail Stop: {row['Trail Stop']}"
        )
        if row.get("Overall Score %") not in (None, "N/A"):
            message = f"Overall Score: {row['Overall Score %']}% ({row['Verdict']})\n" + message
        if "RS Percentile" in row:
            message += f"\nRS Percentile: {row['RS Percentile']}"
        if "Dist from 52W High" in row:
            message += f"\nDist from 52W High: {row['Dist from 52W High']}"
        if system == "CE Buy-Only":
            message += "\n⚠️ Sirf 1 indicator par mabni signal - احتیاط سے capital lagayein."

        send_strong_notification(title, message)
        send_telegram_alert(f"<b>{title}</b>\n{message}")
        notified[key] = datetime.now(timezone.utc).isoformat()
        new_count += 1

    save_notified_keys(notified)
    print(f"Notifications bheji: {new_count}")


if __name__ == "__main__":
    main()
