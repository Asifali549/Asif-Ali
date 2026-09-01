"""
Streamlit Live Screener - Ichimoku+Market Structure aur EMA+Breakout
confluence signals, HAR timeframe (15m/1h/4h) par HAR coin ke liye check
hote hain (kisi strategy ko kisi khaas timeframe tak mehdood nahi rakha).
Chandelier trailing stop ke sath.

Default: Binance ke SAB USDT spot coins (meme coins aur leveraged/binary
tokens jaise BTCUP/BTCDOWN automatically exclude hote hain).

Chalayen: streamlit run screener_app.py

NOTE: Isay apne computer par chalayen jahan Binance tak internet access ho
(sandbox mein exchange access nahi hai).
"""

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
st.caption(
    "Har coin, har timeframe (15m/1h/4h) par **Ichimoku+MarketStructure** "
    "aur **EMA+Breakout** dono confluence combos check hote hain — "
    "kisi combo ko kisi khaas timeframe tak mehdood nahi rakha gaya."
)

COMBOS = [
    ("ichimoku", "market_structure", "Ichimoku+MarketStructure"),
    ("ema_crossover", "breakout", "EMA+Breakout"),
]
TIMEFRAMES_TO_SCAN = ["15m", "1h", "4h"]


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Settings")

coin_mode = st.sidebar.radio("Coin List", ["Binance ke SAB coins (meme/leveraged exclude)", "MANUAL"])

if coin_mode.startswith("Binance"):
    top_n = config.AUTO_TOP_N_COINS  # practically sab (meme/leveraged filter data_fetcher mein lagta hai)
else:
    manual_text = st.sidebar.text_area(
        "Coins (comma-separated, jaise BTC/USDT,ETH/USDT)",
        value=",".join(config.MANUAL_COIN_LIST),
    )

timeframes_selected = st.sidebar.multiselect(
    "Kaunse timeframes scan karein", TIMEFRAMES_TO_SCAN, default=TIMEFRAMES_TO_SCAN
)

lookback_bars = st.sidebar.slider(
    "Signal 'fresh' consider karein agar kitne bars pehle aaya ho", 1, 10, 3
)

rr_multiple = st.sidebar.number_input("Take Profit = Risk x", value=2.0, step=0.5)
ce_period = st.sidebar.number_input("Chandelier Period", value=12, step=1)
ce_mult = st.sidebar.number_input("Chandelier ATR Multiplier", value=3.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ Binance ke SAB coins + sab timeframes scan karne mein kaafi waqt "
    "lag sakta hai (100+ coins x 3 timeframes x 2 combos). Sabar karein."
)

run_scan = st.sidebar.button("🔍 Scan Chalayen", type="primary", use_container_width=True)


# ============================================================
# SCAN LOGIC
# ============================================================
def scan_symbol_timeframe(exchange, symbol, timeframe, lookback_bars, ce_period, ce_mult, rr_multiple):
    """Ek symbol/timeframe ka data ek dafa fetch karta hai, phir dono combos check karta hai."""
    df = fetch_ohlcv(exchange, symbol, timeframe, limit=max(config.CANDLE_LIMITS.get(timeframe, 500), 300))
    if df is None or len(df) < 220:
        return []

    atr = compute_atr(df, ce_period)
    highest_high = df["high"].rolling(ce_period).max()
    chandelier = highest_high - ce_mult * atr

    found = []
    for strat_a, strat_b, combo_name in COMBOS:
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
            "Signal Bar": signal_bar["timestamp"],
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

    if coin_mode.startswith("Binance"):
        with st.spinner("Coin list le rahe hain (meme/leveraged coins exclude ho rahe hain)..."):
            coins = get_coin_list(exchange)[:top_n]
    else:
        coins = [c.strip() for c in manual_text.split(",") if c.strip()]

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
                r = scan_symbol_timeframe(exchange, symbol, tf, lookback_bars, ce_period, ce_mult, rr_multiple)
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

        styled = df_filtered.style.map(color_status, subset=["Status"]).map(color_pnl, subset=["P/L %"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 CSV Download Karein", csv, "screener_results.csv", "text/csv")
    else:
        st.warning("Koi fresh confluence signal nahi mila is waqt. Coin list ya timeframe change kar ke dekhein.")

else:
    st.info("👈 Sidebar mein settings choose karein aur 'Scan Chalayen' dabayein.")
    st.markdown(
        """
        ### Ye screener kya karta hai
        - Default: **Binance ke sab USDT spot coins** (meme coins jaise DOGE/SHIB/PEPE, aur
          leveraged/binary tokens jaise BTCUP/BTCDOWN automatically exclude)
        - Har coin par **har timeframe** (15m, 1h, 4h) check hota hai
        - Har timeframe par **dono combos** (Ichimoku+MarketStructure, EMA+Breakout) test hote hain
          — koi combo kisi khaas timeframe tak mehdood nahi
        - Har fresh signal ke liye Chandelier trailing stop aur Take Profit calculate hota hai
        - Jo trades abhi tak stop/target nahi hue, unhe **OPEN** dikhata hai
        """
    )
