"""
Streamlit Live Screener - Ichimoku+Market Structure aur EMA+Breakout
confluence signals. Background mein har 15 minute (GitHub Actions se)
top 100 coins x 1h par khud scan hota hai - is page par foran (bina wait)
wahi taaza result dikhta hai. Chahen to neeche manual "Deep Scan" bhi
chala sakte hain (sab coins x sab timeframes, jisme waqt lagta hai).

Chalayen: streamlit run screener_app.py
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import compute_atr


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Crypto Confluence Screener", layout="wide")
st.title("📊 Crypto Confluence Screener")

COMBOS = [
    # (strategy_a, strategy_b, display_name, ce_period)
    # 400-coin backtest se CONFIRM: Ichimoku+MS ke liye CE(16) bohot behtar (PF 1.41->2.94)
    #                                EMA+Breakout ke liye CE(12) hi behtar (CE16 se nuksan)
    #                                MarketStructure+CVD-Proxy: overall PF 5.42! (400 coins par confirm)
    ("ichimoku", "market_structure", "Ichimoku+MarketStructure", 16),
    ("ema_crossover", "breakout", "EMA+Breakout", 12),
    ("market_structure", "cvd_proxy", "MarketStructure+CVD", 16),
]
TIMEFRAMES_TO_SCAN = ["15m", "1h", "4h"]


def to_pkt_str(ts):
    """Exchange ka timestamp UTC hota hai - Pakistan Time (UTC+5) mein dikhate hain."""
    ts_utc = pd.Timestamp(ts)
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    return ts_utc.tz_convert("Asia/Karachi").strftime("%Y-%m-%d %I:%M %p PKT")


def style_results(df_results):
    def color_status(val):
        if val == "OPEN":
            return "background-color: #1a4d2e; color: white"
        elif val == "TARGET HIT":
            return "background-color: #0d4d0d; color: white"
        else:
            return "background-color: #4d1a1a; color: white"

    def color_pnl(val):
        color = "#1a4d2e" if val >= 0 else "#4d1a1a"
        return f"background-color: {color}; color: white"

    return df_results.style.map(color_status, subset=["Status"]).map(color_pnl, subset=["P/L %"])


# ============================================================
# SECTION 1: BACKGROUND AUTO-SCAN RESULT (foran, koi wait nahi)
# ============================================================
st.header("🔴 LIVE — Background Auto-Scan (har 1 ghanta, 400 coins, 1h)")

if os.path.exists("latest_signals.json"):
    with open("latest_signals.json") as f:
        auto_data = json.load(f)

    last_updated = datetime.fromisoformat(auto_data["last_updated_utc"])
    age_minutes = (datetime.now(timezone.utc) - last_updated).total_seconds() / 60

    col1, col2, col3 = st.columns(3)
    col1.metric("Last Updated", f"{age_minutes:.0f} min pehle")
    col2.metric("Coins Scanned", auto_data["coins_scanned"])
    col3.metric("Fresh Signals", len(auto_data["signals"]))

    if age_minutes > 75:
        st.warning("⚠️ Ye data 75 minute se purana hai — GitHub Actions ka background scan shayad delay ho gaya ho, ya abhi setup na hua ho.")

    if auto_data["signals"]:
        df_auto = pd.DataFrame(auto_data["signals"]).sort_values(["Bars Ago", "Coin"])
        st.dataframe(style_results(df_auto), use_container_width=True, hide_index=True)
    else:
        st.info("Is waqt koi fresh signal nahi (last scan mein).")
else:
    st.info(
        "Background auto-scan abhi setup nahi hua ya pehli baar chalne ka wait ho raha hai. "
        "GitHub repo mein '.github/workflows/scan.yml' hona chahiye — 15 minute mein pehla result aa jayega."
    )

st.markdown("---")


# ============================================================
# SECTION 2: MANUAL DEEP SCAN (sab coins x sab timeframes, waqt lagta hai)
# ============================================================
st.header("🔍 Manual Deep Scan (poora control, magar waqt lagta hai)")

st.sidebar.header("Deep Scan Settings")

coin_mode = st.sidebar.radio("Coin List", ["KuCoin ke SAB coins (meme/leveraged exclude)", "MANUAL"])

if coin_mode.startswith("KuCoin"):
    top_n = config.AUTO_TOP_N_COINS
else:
    manual_text = st.sidebar.text_area(
        "Coins (comma-separated, jaise BTC/USDT,ETH/USDT)",
        value=",".join(config.MANUAL_COIN_LIST),
    )

timeframes_selected = st.sidebar.multiselect(
    "Kaunse timeframes scan karein", TIMEFRAMES_TO_SCAN, default=["1h"]
)

lookback_bars = st.sidebar.slider(
    "Signal 'fresh' consider karein agar kitne bars pehle aaya ho", 1, 10, 3
)

rr_multiple = st.sidebar.number_input("Take Profit = Risk x", value=2.0, step=0.5)
ce_mult = st.sidebar.number_input("Chandelier ATR Multiplier", value=3.0, step=0.5)
st.sidebar.caption("Chandelier Period har combo apna alag, tasdeeq-shuda (16 ya 12) khud istemal karta hai.")

st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ Zyada coins/timeframes scan karne mein kaafi waqt lag sakta hai. Sabar karein."
)

run_scan = st.sidebar.button("🔍 Deep Scan Chalayen", type="primary", use_container_width=True)


# ============================================================
# SCAN LOGIC
# ============================================================
def scan_symbol_timeframe(exchange, symbol, timeframe, lookback_bars, ce_mult, rr_multiple):
    """Ek symbol/timeframe ka data ek dafa fetch karta hai, phir dono combos apne apne CE ke sath check karta hai."""
    df = fetch_ohlcv(exchange, symbol, timeframe, limit=max(config.CANDLE_LIMITS.get(timeframe, 500), 300))
    if df is None or len(df) < 220:
        return []

    found = []
    for strat_a, strat_b, combo_name, ce_period in COMBOS:
        atr = compute_atr(df, ce_period)
        highest_high = df["high"].rolling(ce_period).max()
        chandelier = highest_high - ce_mult * atr

        params_a = config.STRATEGY_PARAMS[strat_a]
        params_b = config.STRATEGY_PARAMS[strat_b]

        sig_a = STRATEGY_FUNCTIONS[strat_a](df, params_a)
        sig_b = STRATEGY_FUNCTIONS[strat_b](df, params_b)
        sig_a = apply_cooldown(sig_a, config.SIGNAL_COOLDOWN_BARS)
        sig_b = apply_cooldown(sig_b, config.SIGNAL_COOLDOWN_BARS)

        combined = sig_a & sig_b
        recent = combined.tail(lookback_bars)
        if not recent.any():
            continue

        signal_idx = recent[recent].index[-1]
        signal_bar = df.loc[signal_idx]
        bars_ago = len(df) - 1 - signal_idx

        entry_price = signal_bar["close"]
        current_price = df["close"].iloc[-1]

        initial_stop = chandelier.loc[signal_idx]
        if pd.isna(initial_stop):
            continue

        risk = entry_price - initial_stop
        tp_price = entry_price + risk * rr_multiple
        trail_stop = chandelier.loc[signal_idx:].max()
        pnl_pct = (current_price - entry_price) / entry_price * 100

        status = "OPEN"
        if df["low"].iloc[-1] <= trail_stop:
            status = "STOPPED (trail)"
        elif df["high"].iloc[-1] >= tp_price:
            status = "TARGET HIT"

        found.append({
            "Coin": symbol,
            "Timeframe": timeframe,
            "Combo": combo_name,
            "Signal Bar": to_pkt_str(signal_bar["timestamp"]),
            "Bars Ago": bars_ago,
            "Entry": round(entry_price, 6),
            "Current": round(current_price, 6),
            "P/L %": round(pnl_pct, 2),
            "Trail Stop": round(trail_stop, 6),
            "Take Profit": round(tp_price, 6),
            "Status": status,
        })

    return found


# ============================================================
# MAIN
# ============================================================
if run_scan:
    if not timeframes_selected:
        st.error("Kam az kam ek timeframe select karein.")
        st.stop()

    exchange = get_exchange()

    try:
        if coin_mode.startswith("KuCoin"):
            with st.spinner("Coin list le rahe hain (meme/leveraged coins exclude ho rahe hain)..."):
                coins = get_coin_list(exchange)[:top_n]
        else:
            coins = [c.strip() for c in manual_text.split(",") if c.strip()]
    except Exception as e:
        st.error(
            "❌ Exchange (KuCoin) tak connect nahi ho pa raha is waqt "
            "(shayad network ya exchange ki taraf se koi masla hai — ye code ka "
            "masla nahi).\n\n"
            "**Behtareen tareeqa:** Upar 'LIVE — Background Auto-Scan' section already "
            "kaam kar raha hai (GitHub Actions se) — usi par bharosa karein.\n\n"
            "Agar Deep Scan zaroori hai to isay apne PC par local chalayen bhi kar sakte hain."
        )
        st.stop()

    total_calls = len(coins) * len(timeframes_selected)
    st.info(f"Coins: **{len(coins)}** | Timeframes: **{', '.join(timeframes_selected)}** | "
            f"Kul scans: **{total_calls}** (har coin x har timeframe, dono combos check honge)")

    progress = st.progress(0.0, text="Scanning...")
    results = []
    call_count = 0

    for symbol in coins:
        for tf in timeframes_selected:
            call_count += 1
            try:
                r = scan_symbol_timeframe(exchange, symbol, tf, lookback_bars, ce_mult, rr_multiple)
                results.extend(r)
            except Exception as e:
                st.sidebar.warning(f"{symbol} {tf}: {e}")
            progress.progress(call_count / total_calls, text=f"Scanning {symbol} ({tf}) — {call_count}/{total_calls}")

    progress.empty()

    if results:
        df_results = pd.DataFrame(results).sort_values(["Bars Ago", "Coin"])

        open_trades = df_results[df_results["Status"] == "OPEN"]
        st.success(f"✅ {len(df_results)} fresh signals mile ({len(open_trades)} abhi bhi OPEN hain)")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            filter_tf = st.multiselect("Timeframe filter", sorted(df_results["Timeframe"].unique()),
                                        default=sorted(df_results["Timeframe"].unique()))
        with col2:
            filter_combo = st.multiselect("Combo filter", sorted(df_results["Combo"].unique()),
                                           default=sorted(df_results["Combo"].unique()))

        df_filtered = df_results[df_results["Timeframe"].isin(filter_tf) & df_results["Combo"].isin(filter_combo)]

        st.dataframe(style_results(df_filtered), use_container_width=True, hide_index=True)

        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 CSV Download Karein", csv, "screener_results.csv", "text/csv")
    else:
        st.warning("Koi fresh confluence signal nahi mila is waqt. Coin list ya timeframe change kar ke dekhein.")

else:
    st.info("👈 Sidebar mein settings choose karein aur 'Scan Chalayen' dabayein.")
    st.markdown(
        """
        ### Ye screener kya karta hai
        - Default: **KuCoin ke sab USDT spot coins** (meme coins jaise DOGE/SHIB/PEPE, aur
          leveraged/binary tokens jaise BTCUP/BTCDOWN automatically exclude)
        - Har coin par **har timeframe** (15m, 1h, 4h) check hota hai
        - Har timeframe par **teeno combos** (Ichimoku+MarketStructure, EMA+Breakout, MarketStructure+CVD) test hote hain
          — koi combo kisi khaas timeframe tak mehdood nahi
        - Har fresh signal ke liye Chandelier trailing stop aur Take Profit calculate hota hai
        - Jo trades abhi tak stop/target nahi hue, unhe **OPEN** dikhata hai
        """
    )


# ============================================================
# SECTION 3: VOLATILITY SQUEEZE SCANNER (watch-list tool, trading signal NAHI)
# ============================================================
st.markdown("---")
st.header("🔭 Volatility Squeeze Scanner (Watch-List)")
st.caption(
    "⚠️ Ye trading SIGNAL nahi hai — sirf ye batata hai ke kis coin ki volatility "
    "ghair-mamooli tor par kam ho gayi hai (Bollinger Bands, Keltner Channel ke andar "
    "aa gaye hain). Squeeze = 'koi bari harkat aane wali hai', lekin **direction "
    "(upar ya neeche) pata nahi**. Sirf watch-list ke liye istemal karein, akele "
    "entry ki wajah na banayein."
)

col1, col2, col3 = st.columns(3)
sq_timeframe = col1.selectbox("Timeframe", ["15m", "1h", "4h"], index=1, key="sq_tf")
sq_n_coins = col2.slider("Kitne coins scan karein", 20, 300, 150, key="sq_n")
sq_bb_period = col3.number_input("BB/KC Period", value=20, step=1, key="sq_period")

if st.button("🔍 Squeeze Scan Chalayen", type="primary", key="sq_btn"):
    exchange = get_exchange()
    try:
        with st.spinner("Coin list le rahe hain..."):
            sq_coins = get_coin_list(exchange)[:sq_n_coins]
    except Exception as e:
        st.error(f"Exchange se connect nahi ho paya: {e}")
        st.stop()

    sq_results = []
    progress = st.progress(0.0)
    for i, symbol in enumerate(sq_coins):
        try:
            df = fetch_ohlcv(exchange, symbol, sq_timeframe, limit=sq_bb_period + 30)
        except Exception:
            df = None
        if df is not None and len(df) >= sq_bb_period + 5:
            close, high, low = df["close"], df["high"], df["low"]

            # Bollinger Bands (2 std dev)
            bb_mid = close.rolling(sq_bb_period).mean()
            bb_std = close.rolling(sq_bb_period).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std

            # Keltner Channel (1.5x ATR)
            tr = pd.concat([
                high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(sq_bb_period).mean()
            kc_upper = bb_mid + 1.5 * atr
            kc_lower = bb_mid - 1.5 * atr

            is_squeeze = (bb_upper.iloc[-1] < kc_upper.iloc[-1]) and (bb_lower.iloc[-1] > kc_lower.iloc[-1])

            if is_squeeze:
                # Kitne bars se squeeze mein hai (consecutive count)
                squeeze_series = (bb_upper < kc_upper) & (bb_lower > kc_lower)
                bars_in_squeeze = 0
                for val in squeeze_series.iloc[::-1]:
                    if val:
                        bars_in_squeeze += 1
                    else:
                        break

                sq_results.append({
                    "Coin": symbol,
                    "Bars in Squeeze": bars_in_squeeze,
                    "Current Price": round(close.iloc[-1], 6),
                    "BB Width %": round((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_mid.iloc[-1] * 100, 2),
                })
        progress.progress((i + 1) / len(sq_coins))
    progress.empty()

    if sq_results:
        df_sq = pd.DataFrame(sq_results).sort_values("Bars in Squeeze", ascending=False)
        st.success(f"✅ {len(df_sq)} coins is waqt volatility squeeze mein hain")
        st.dataframe(df_sq, use_container_width=True, hide_index=True)
        st.caption("Jitne zyada 'Bars in Squeeze' honge, utni lambi consolidation - is se signal ki quality zyada nahi hoti, sirf watch list mein zyada priority de sakte hain.")
    else:
        st.info("Is waqt koi coin squeeze mein nahi mila.")


# ============================================================
# SECTION 4: NEW ADVANCED CONFLUENCE SYSTEM (bilkul ALAG, purana chhua nahi)
# ============================================================
st.markdown("---")
st.header("🎯 NEW — Advanced Confluence System (Alag/Independent)")
st.caption(
    "Ye purane 3-combo system se BILKUL ALAG hai, alag background scan "
    "(GitHub Actions), alag notifications. 400-coin comparison test mein "
    "PF 2.13 (Ichimoku+MS ke PF 3.51 se kam, lekin bara sample - 2484 trades)."
)

if os.path.exists("latest_signals_new.json"):
    with open("latest_signals_new.json") as f:
        new_data = json.load(f)

    new_last_updated = datetime.fromisoformat(new_data["last_updated_utc"])
    new_age_minutes = (datetime.now(timezone.utc) - new_last_updated).total_seconds() / 60

    ncol1, ncol2, ncol3 = st.columns(3)
    ncol1.metric("Last Updated", f"{new_age_minutes:.0f} min pehle")
    ncol2.metric("Coins Scanned", new_data["coins_scanned"])
    ncol3.metric("Fresh Signals", len(new_data["signals"]))

    if new_age_minutes > 75:
        st.warning("⚠️ Ye data 75 minute se purana hai — NEW system ka background scan shayad delay ho gaya ho.")

    if new_data["signals"]:
        df_new = pd.DataFrame(new_data["signals"]).sort_values(["Bars Ago", "Coin"])
        st.dataframe(style_results(df_new), use_container_width=True, hide_index=True)
    else:
        st.info("Is waqt koi fresh signal nahi (last scan mein).")
else:
    st.info(
        "NEW system ka background scan abhi setup nahi hua ya pehli baar chalne ka wait ho raha hai. "
        "GitHub repo mein '.github/workflows/scan_new.yml' hona chahiye — 1 ghante mein pehla result aa jayega."
    )