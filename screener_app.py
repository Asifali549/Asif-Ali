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
        - Har timeframe par **dono combos** (Ichimoku+MarketStructure, EMA+Breakout) test hote hain
          — koi combo kisi khaas timeframe tak mehdood nahi
        - Har fresh signal ke liye Chandelier trailing stop aur Take Profit calculate hota hai
        - Jo trades abhi tak stop/target nahi hue, unhe **OPEN** dikhata hai
        """
  