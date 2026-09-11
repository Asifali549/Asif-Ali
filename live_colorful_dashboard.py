"""
Live + Manual Colorful Dashboard - LIVE section khud-b-khud (GitHub
Actions se) update hoti hai, MANUAL section button se on-demand
chalti hai. Dono mein Overall Score, Rank (best=1), aur Compact
View available hai.

Chalayen: streamlit run live_colorful_dashboard.py
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
from dashboard_helpers import (
    get_fear_greed, get_futures_exchange, get_funding_and_oi, get_long_short_ratio,
    get_orderbook_info, get_tf_volume_change, get_24h_range_distance,
    get_historical_performance, get_btc_correlation, get_whale_activity,
    color_value, compute_overall_score, ALL_TIMEFRAMES,
)

st.set_page_config(page_title="Live Colorful Dashboard", layout="wide")
st.title("🎨 Live + Manual Colorful Dashboard")
st.warning(
    "⚠️ Sirf MALOOMAT — asal Union AB signal logic bilkul nahi badla "
    "gaya. 🟢 Green = signal ke HAQ mein, 🔴 Red = KHILAF."
)

CE_A = {"period": 16, "multiplier": 3.0}
CE_B = {"period": 12, "multiplier": 3.0}
RR_MULTIPLE = 2.0

COLORABLE_COLUMNS = (
    ["OrderBook Bid/Ask", "Liquidity Compare", "Coin's Own PF", "Funding Rate", "Dist from 24h High"]
    + [f"Vol {tf}" for tf in ALL_TIMEFRAMES]
    + [f"Chg {tf}" for tf in ALL_TIMEFRAMES]
)

COMPACT_COLUMNS = ["Rank", "Coin", "Combo", "Overall Score %", "Verdict", "Entry", "Trail Stop", "Take Profit"]


def style_and_show(df, compact):
    if compact:
        show_cols = [c for c in COMPACT_COLUMNS if c in df.columns]
        df = df[show_cols]

    def apply_row_colors(row):
        return [color_value(col, row[col]) if col in df.columns else "" for col in df.columns]

    st.dataframe(df.style.apply(apply_row_colors, axis=1), use_container_width=True, hide_index=True)


# ============================================================
# SECTION 1: LIVE (khud-b-khud, GitHub Actions se)
# ============================================================
st.header("🔴 LIVE — Auto-Updated (har 1 ghanta)")
compact_live = st.checkbox("Compact View (sirf zaroori columns)", value=True, key="compact_live")

if os.path.exists("dashboard_signals.json"):
    with open("dashboard_signals.json") as f:
        live_data = json.load(f)

    last_updated = datetime.fromisoformat(live_data["last_updated_utc"])
    age_minutes = (datetime.now(timezone.utc) - last_updated).total_seconds() / 60

    col1, col2, col3 = st.columns(3)
    col1.metric("Last Updated", f"{age_minutes:.0f} min pehle")
    col2.metric("Coins Scanned", live_data["coins_scanned"])
    col3.metric("Signals", len(live_data["signals"]))

    if live_data.get("fear_greed_value") is not None:
        st.markdown(f"### 📊 BTC Fear & Greed: **{live_data['fear_greed_value']}/100** ({live_data['fear_greed_label']})")

    if age_minutes > 90:
        st.warning("⚠️ Ye data 90 minute se purana hai — background scan delay ho sakta hai.")

    if live_data["signals"]:
        df_live = pd.DataFrame(live_data["signals"])
        style_and_show(df_live, compact_live)
    else:
        st.info("Is waqt koi fresh signal nahi (last scan mein).")
else:
    st.info(
        "Live scan abhi setup nahi hua ya pehli baar chalne ka wait ho raha hai. "
        "GitHub repo mein '.github/workflows/scan_dashboard.yml' hona chahiye — "
        "1 ghante mein pehla result aa jayega."
    )

st.markdown("---")


# ============================================================
# SECTION 2: MANUAL (on-demand, apni marzi ke toggles ke sath)
# ============================================================
st.header("🔍 Manual Scan (apni marzi ke toggles)")

st.sidebar.header("Manual Scan Settings")
show_funding = st.sidebar.checkbox("Funding Rate", value=True)
show_oi = st.sidebar.checkbox("Open Interest", value=True)
show_liquidity = st.sidebar.checkbox("Liquidity Up/Down", value=True)
show_orderbook_ratio = st.sidebar.checkbox("Order Book Ratio", value=True)
show_24h_range = st.sidebar.checkbox("24h Range", value=True)
show_history = st.sidebar.checkbox("Coin's Own Performance", value=True)
show_btc_corr = st.sidebar.checkbox("BTC Correlation", value=False)
show_long_short = st.sidebar.checkbox("Long/Short Ratio", value=False)
show_whale = st.sidebar.checkbox("Whale Transfers", value=False)
etherscan_key = st.sidebar.text_input("Etherscan API Key", type="password") if show_whale else ""

st.sidebar.markdown("---")
st.sidebar.subheader("Volume - Har Timeframe")
vol_tf_toggles = {tf: st.sidebar.checkbox(f"Vol {tf}", value=True, key=f"v{tf}") for tf in ALL_TIMEFRAMES}
st.sidebar.subheader("Price Change - Har Timeframe")
chg_tf_toggles = {tf: st.sidebar.checkbox(f"Chg {tf}", value=True, key=f"c{tf}") for tf in ALL_TIMEFRAMES}

st.sidebar.markdown("---")
coin_source = st.sidebar.radio("Coin Kaise Chunein", ["Auto Scan", "Manual Paste"])
if coin_source.startswith("Manual"):
    manual_coins_text = st.sidebar.text_area("Coins (comma-separated)", height=80)
else:
    n_coins = st.sidebar.slider("Kitne coins", 10, 400, 20)
signal_timeframe = st.sidebar.selectbox("Signal Timeframe", ["15m", "1h", "4h"], index=1)

compact_manual = st.checkbox("Compact View (sirf zaroori columns)", value=True, key="compact_manual")

if st.button("🎨 Manual Scan Chalayen", type="primary"):
    exchange = get_exchange()
    futures_exchange = get_futures_exchange() if (show_funding or show_oi or show_long_short) else None

    if coin_source.startswith("Manual"):
        coins = [c.strip() for c in manual_coins_text.split(",") if c.strip()]
        if not coins:
            st.error("Koi coin nahi likha gaya.")
            st.stop()
    else:
        try:
            with st.spinner("Coin list le rahe hain..."):
                coins = get_coin_list(exchange)[:n_coins]
        except Exception as e:
            st.error(f"Exchange se connect nahi ho paya: {e}")
            st.stop()

    signal_coins = []
    progress = st.progress(0.0, text="Signals dhoond rahe hain...")
    for i, symbol in enumerate(coins):
        try:
            df = fetch_ohlcv(exchange, symbol, signal_timeframe, limit=max(config.CANDLE_LIMITS.get(signal_timeframe, 500), 300))
        except Exception:
            df = None
        if df is not None and len(df) >= 220:
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
                        atr = compute_atr(df, ce["period"])
                        highest_high = df["high"].rolling(ce["period"]).max()
                        chandelier = (highest_high - ce["multiplier"] * atr).loc[idx]
                        entry_price = df.loc[idx, "close"]
                        current_price = df["close"].iloc[-1]
                        risk = entry_price - chandelier
                        tp_price = entry_price + risk * RR_MULTIPLE
                        signal_coins.append({
                            "Coin": symbol, "Combo": combo_name, "Bars Ago": len(df) - 1 - idx,
                            "Entry": round(entry_price, 6), "Current": round(current_price, 6),
                            "Trail Stop": round(chandelier, 6), "Take Profit": round(tp_price, 6),
                            "_df": df, "_sig": combo_sig, "_ce": ce,
                        })
            except Exception:
                pass
        progress.progress((i + 1) / len(coins), text=f"Signals... {i+1}/{len(coins)}")
    progress.empty()

    if not signal_coins:
        st.info("Is waqt koi fresh signal nahi mila.")
        st.stop()

    st.success(f"✅ {len(signal_coins)} signals mile — context le rahe hain...")

    active_vol_tfs = [tf for tf, on in vol_tf_toggles.items() if on]
    active_chg_tfs = [tf for tf, on in chg_tf_toggles.items() if on]

    final_rows = []
    progress2 = st.progress(0.0, text="Context le rahe hain...")
    for i, row in enumerate(signal_coins):
        symbol = row["Coin"]
        final_row = {k: v for k, v in row.items() if not k.startswith("_")}

        if show_funding or show_oi:
            funding, oi = get_funding_and_oi(futures_exchange, symbol)
            if show_funding:
                final_row["Funding Rate"] = f"{funding*100:.3f}%" if funding is not None else "N/A"
            if show_oi:
                final_row["Open Interest"] = f"${oi:,.0f}" if oi is not None else "N/A"

        if show_long_short:
            ls = get_long_short_ratio(futures_exchange, symbol)
            final_row["Long/Short Ratio"] = f"{ls:.2f}" if ls is not None else "N/A"

        if show_liquidity or show_orderbook_ratio:
            bid_liq, ask_liq, ob_ratio = get_orderbook_info(exchange, symbol)
            if show_liquidity:
                final_row["Liquidity Up ($)"] = f"${ask_liq:,.0f}" if ask_liq is not None else "N/A"
                final_row["Liquidity Down ($)"] = f"${bid_liq:,.0f}" if bid_liq is not None else "N/A"
                final_row["Liquidity Compare"] = ("Support > Resistance" if bid_liq > ask_liq else "Resistance > Support") if (bid_liq is not None and ask_liq is not None) else "N/A"
            if show_orderbook_ratio:
                final_row["OrderBook Bid/Ask"] = f"{ob_ratio:.2f}x" if ob_ratio is not None else "N/A"

        for tf in active_vol_tfs:
            vol_ratio, _ = get_tf_volume_change(exchange, symbol, tf)
            final_row[f"Vol {tf}"] = f"{vol_ratio:.2f}x" if vol_ratio is not None else "N/A"
        for tf in active_chg_tfs:
            _, chg = get_tf_volume_change(exchange, symbol, tf)
            final_row[f"Chg {tf}"] = f"{chg:+.2f}%" if chg is not None else "N/A"

        if show_24h_range:
            dist_high, dist_low = get_24h_range_distance(exchange, symbol)
            final_row["Dist from 24h High"] = f"{dist_high}%" if dist_high is not None else "N/A"
            final_row["Dist from 24h Low"] = f"{dist_low}%" if dist_low is not None else "N/A"

        if show_history:
            win_rate, pf = get_historical_performance(row["_df"], row["_sig"], row["_ce"])
            final_row["Coin's Own Win%"] = f"{win_rate}%" if win_rate is not None else "N/A"
            final_row["Coin's Own PF"] = f"{pf}" if pf is not None else "N/A"

        if show_btc_corr:
            corr = get_btc_correlation(exchange, symbol, signal_timeframe)
            final_row["BTC Correlation"] = f"{corr}" if corr is not None else "N/A"

        if show_whale:
            final_row["Whale Activity"] = get_whale_activity(symbol, etherscan_key)

        score, verdict = compute_overall_score(final_row, COLORABLE_COLUMNS)
        final_row["Overall Score %"] = score
        final_row["Verdict"] = verdict

        final_rows.append(final_row)
        progress2.progress((i + 1) / len(signal_coins), text=f"Context... {i+1}/{len(signal_coins)}")
    progress2.empty()

    final_rows.sort(key=lambda r: r["Overall Score %"], reverse=True)
    for i, r in enumerate(final_rows):
        r["Rank"] = i + 1
    final_rows = [{"Rank": r.pop("Rank"), **r} for r in final_rows]

    df_final = pd.DataFrame(final_rows)
    style_and_show(df_final, compact_manual)

    csv = df_final.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV Download Karein", csv, "manual_dashboard.csv", "text/csv")

st.caption(
    "🟢 Green = signal ke HAQ mein. 🔴 Red = KHILAF. Rank 1 = sab se zyada "
    "'Overall Score' wala (best) trade. Compact View se sirf zaroori columns dikhte hain."
)