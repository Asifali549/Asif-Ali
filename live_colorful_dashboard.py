"""
Live + Manual Colorful Dashboard - LIVE section khud-b-khud (GitHub
Actions se) update hoti hai, MANUAL section button se on-demand
chalti hai. Dono mein Overall Score, Rank (best=1), aur Compact
View available hai.

Chalayen: streamlit run live_colorful_dashboard.py
"""

import base64
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests
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

def _safe_json_load(path):
    """JSON file ko parhta hai; agar file corrupt/invalid JSON ho to poori app crash
    karne ke bajaye None wapas karta hai (taake baaki dashboard sections chalte rahein)."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


st.set_page_config(page_title="Live Colorful Dashboard", layout="wide")
st.title("🎨 Live + Manual Colorful Dashboard")
st.warning(
    "⚠️ Sirf MALOOMAT — asal Union AB signal logic bilkul nahi badla "
    "gaya. 🟢 Green = signal ke HAQ mein, 🔴 Red = KHILAF."
)

st.sidebar.markdown("---")
st.sidebar.header("💰 Position Sizing Calculator")
total_capital = st.sidebar.number_input("Total Capital ($)", min_value=0.0, value=1000.0, step=100.0)
risk_pct_per_trade = st.sidebar.number_input("Risk % per Trade", min_value=0.1, max_value=100.0, value=1.0, step=0.5)
st.sidebar.caption("Har trade mein Entry aur Trail Stop ke farq ke mutabiq, khud-kar position size calculate hoga.")

CE_A = {"period": 16, "multiplier": 4.5}
CE_B = {"period": 12, "multiplier": 4.5}
RR_MULTIPLE = 2.0

COLORABLE_COLUMNS = (
    ["OrderBook Bid/Ask", "Liquidity Compare", "Coin's Own PF", "Funding Rate", "Dist from 24h High"]
    + [f"Vol {tf}" for tf in ALL_TIMEFRAMES]
    + [f"Chg {tf}" for tf in ALL_TIMEFRAMES]
)

COMPACT_COLUMNS = ["Rank", "System", "Category", "Coin", "Signal Time (PKT)", "Combo", "Overall Score %", "Verdict", "Entry", "Current", "P/L %", "Trail Stop", "Take Profit"]

SYSTEM_ORDER = [
    "Union AB", "NEW AdvancedConfluence", "Union AB Backup Tier", "CE Buy-Only",
    "Pullback-in-Uptrend", "Donchian Breakout",
]
SYSTEM_BADGE = {
    "Union AB": "🥇 Union AB",
    "NEW AdvancedConfluence": "🧭 NEW AdvancedConfluence",
    "Union AB Backup Tier": "🛡️ Union AB Backup Tier",
    "CE Buy-Only": "⚡ CE Buy-Only",
    "Pullback-in-Uptrend": "🔁 Pullback-in-Uptrend",
    "Donchian Breakout": "📈 Donchian Breakout",
}
SYSTEM_CAPTION = {
    "Union AB": "Sab se zyada tasdeeq-shuda system (ETH+RS+RS%95, +52W tier).",
    "NEW AdvancedConfluence": "CHoCH-based confluence score, koi extra filter nahi.",
    "Union AB Backup Tier": "Union AB jaisa combo, sirf RS+RS%95 (ETH check NAHI) — hamesha active rehta hai.",
    "CE Buy-Only": "⚠️ Sirf 1 indicator (Chandelier cross) par mabni — koi tasdeeqi filter nahi. Ehtiyaat se capital lagayein.",
    "Pullback-in-Uptrend": "EMA20 pullback (rising EMA + uptrend) + ETH/RS/RS%95 filters. Poore saal ke walk-forward se tasdeeq-shuda (799 trades, PF 1.831, har fold consistent).",
    "Donchian Breakout": "20-period Donchian high breakout + ETH/RS/RS%95 filters. Poore saal ke walk-forward se tasdeeq-shuda (552 trades, PF 2.171, har fold consistent) — is session ka sab se mazboot nateeja.",
}


def style_and_show(df, compact):
    if compact:
        show_cols = [c for c in COMPACT_COLUMNS if c in df.columns]
        df = df[show_cols]

    def apply_row_colors(row):
        return [color_value(col, row[col]) if col in df.columns else "" for col in df.columns]

    st.dataframe(df.style.apply(apply_row_colors, axis=1), use_container_width=True, hide_index=True)


def to_pkt_str(ts):
    ts_utc = pd.Timestamp(ts)
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.tz_localize("UTC")
    ts_pkt = ts_utc.tz_convert("Asia/Karachi")
    return ts_pkt.strftime("%Y-%m-%d %I:%M %p PKT")


def tradingview_url(coin):
    base = coin.split("/")[0]
    return f"https://www.tradingview.com/chart/?symbol=KUCOIN:{base}USDT"


def show_charts_and_copy(df, key_prefix):
    """Har coin ke liye TradingView chart link + poora trade setup copy karne ka option."""
    if df is None or len(df) == 0:
        return
    st.markdown("**📊 Chart dekhein / Trade setup copy karein:**")
    coin_options = [f"{row['Coin']} — {row.get('Combo', '')}" for _, row in df.iterrows()]
    picked = st.selectbox("Coin chunein", coin_options, key=f"{key_prefix}_pick")
    picked_idx = coin_options.index(picked)
    row = df.iloc[picked_idx]

    tv_url = tradingview_url(row["Coin"])
    st.markdown(f"[📈 {row['Coin']} ka TradingView chart kholein]({tv_url})")

    entry_val = row.get("Entry")
    stop_val = row.get("Trail Stop")
    position_line = ""
    if entry_val and stop_val and entry_val > stop_val:
        risk_per_unit = entry_val - stop_val
        risk_dollars = total_capital * (risk_pct_per_trade / 100)
        position_size_units = risk_dollars / risk_per_unit
        position_value = position_size_units * entry_val
        st.info(
            f"💰 **Position Size** (Capital ${total_capital:,.0f}, Risk {risk_pct_per_trade}%): "
            f"**{position_size_units:.4f} {row['Coin'].split('/')[0]}** "
            f"(~${position_value:,.2f}, agar SL laga to nuksan ~${risk_dollars:,.2f})"
        )
        position_line = f"Position Size: {position_size_units:.4f} {row['Coin'].split('/')[0]} (~${position_value:,.2f})\n"

    setup_text = (
        f"Coin: {row['Coin']}\n"
        f"Combo: {row.get('Combo', '')}\n"
        f"Signal Time: {row.get('Signal Time (PKT)', 'N/A')}\n"
        f"Entry: {row.get('Entry', 'N/A')}\n"
        f"Trail Stop: {row.get('Trail Stop', 'N/A')}\n"
        f"Take Profit: {row.get('Take Profit', 'N/A')}\n"
        f"{position_line}"
    )
    st.code(setup_text, language=None)


# ============================================================
# SECTION 1: LIVE (khud-b-khud, GitHub Actions se)
# ============================================================
st.header("🔴 LIVE — Auto-Updated (har scan ke 5 min baad agla)")
col_a, col_b = st.columns(2)
compact_live = col_a.checkbox("Compact View (sirf zaroori columns)", value=True, key="compact_live")
pass_only_live = col_b.checkbox(
    "✅ Sirf Pass (Strong/Good) Dikhayen", value=True, key="pass_only_live",
    help="ON hone par sirf 🟢🟢🟢 Strong aur 🟢 Good verdict wale signals dikhenge. "
         "🟡 Mixed aur 🔴 Weak (failure) signals screen se hat jayenge.",
)

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
        df_all = pd.DataFrame(live_data["signals"])

        present_systems = [s for s in SYSTEM_ORDER if "System" in df_all.columns and s in df_all["System"].unique()]
        if present_systems:
            selected_systems = st.multiselect(
                "🗂️ Systems Dikhayen", present_systems, default=present_systems, key="system_filter_live",
            )
        else:
            selected_systems = []

        st.caption(
            "ℹ️ 'Pass (Strong/Good)' filter sirf un systems par lagu hota hai jinka Overall Score/Verdict "
            "calculate hota hai (Union AB, NEW AdvancedConfluence). Union AB Backup Tier aur CE Buy-Only "
            "hamesha dikhte hain (in par Verdict N/A hai) — ye filter unhein kabhi nahi chupata."
        )

        any_shown = False
        for system_name in SYSTEM_ORDER:
            if "System" not in df_all.columns or system_name not in selected_systems:
                continue
            df_sys = df_all[df_all["System"] == system_name]
            if len(df_sys) == 0:
                continue

            st.markdown(f"#### {SYSTEM_BADGE.get(system_name, system_name)}")
            st.caption(SYSTEM_CAPTION.get(system_name, ""))

            df_sys_show = df_sys
            if pass_only_live and "Verdict" in df_sys_show.columns:
                verdict_str = df_sys_show["Verdict"].astype(str)
                is_na = verdict_str == "N/A"
                is_pass = verdict_str.str.contains("Strong|Good", na=False)
                total_count = len(df_sys_show)
                df_sys_show = df_sys_show[is_na | is_pass]
                hidden_count = total_count - len(df_sys_show)
                if hidden_count > 0:
                    st.caption(f"🔴🟡 {hidden_count} kamzor (Mixed/Weak) signal chupaye gaye.")

            if len(df_sys_show) == 0:
                st.info("Is waqt is system ka koi signal nahi (filter ke baad).")
                st.markdown("---")
                continue

            key_slug = system_name.replace(" ", "_")

            if "Category" in df_sys_show.columns:
                new_df = df_sys_show[df_sys_show["Category"] == "New Signal"]
                open_df = df_sys_show[df_sys_show["Category"] == "Open Trade"]
            else:
                new_df, open_df = df_sys_show, pd.DataFrame()

            if len(new_df) > 0:
                st.markdown(f"**🟢 Naye Signals ({len(new_df)})**")
                style_and_show(new_df, compact_live)
                show_charts_and_copy(new_df, f"live_{key_slug}_new")
                any_shown = True

            if len(open_df) > 0:
                st.markdown(f"**🔵 Chal Rahi Trades — Open ({len(open_df)})**")
                style_and_show(open_df, compact_live)
                show_charts_and_copy(open_df, f"live_{key_slug}_open")
                any_shown = True

            st.markdown("---")

        if not any_shown:
            st.info("Is waqt koi signal nahi (selected systems/filter ke mutabiq).")
    else:
        st.info("Is waqt koi fresh signal nahi (last scan mein).")
else:
    st.info(
        "Live scan abhi setup nahi hua ya pehli baar chalne ka wait ho raha hai. "
        "GitHub repo mein '.github/workflows/scan_dashboard.yml' hona chahiye — "
        "thodi der mein pehla result aa jayega (har scan khatam hone ke 5 minute baad agla shuru hota hai)."
    )

st.markdown("---")
st.header("📊 System Performance — Closed Trades (Har System Alag Alag)")
st.caption(
    "Jab bhi koi signal SL ya trailing-stop/TP par CLOSE hota hai, wo yahan permanently "
    "record ho jata hai (koi live signal is se nahi hatai jaati) — taake waqt ke sath pata "
    "chal sake konsa system asal mein behtar (high Win Rate/PF) hai aur konsa kamzor."
)
if os.path.exists("closed_trades_log.csv"):
    df_closed = pd.read_csv("closed_trades_log.csv")
    if len(df_closed) > 0 and "System" in df_closed.columns:
        for system_name in SYSTEM_ORDER:
            df_sys_closed = df_closed[df_closed["System"] == system_name]
            st.markdown(f"**{SYSTEM_BADGE.get(system_name, system_name)}**")
            if len(df_sys_closed) == 0:
                st.caption("Abhi tak is system ki koi closed trade record nahi hui.")
                continue

            total = len(df_sys_closed)
            wins = df_sys_closed[df_sys_closed["P/L %"] > 0]
            losses = df_sys_closed[df_sys_closed["P/L %"] <= 0]
            win_rate = len(wins) / total * 100
            gross_win = wins["P/L %"].sum()
            gross_loss = abs(losses["P/L %"].sum())
            pf = (gross_win / gross_loss) if gross_loss > 0 else None

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Closed", total)
            c2.metric("Wins / Losses", f"{len(wins)} / {len(losses)}")
            c3.metric("Win Rate", f"{win_rate:.1f}%")
            c4.metric("Profit Factor", f"{pf:.2f}" if pf is not None else "N/A (abhi koi loss nahi)")

        st.markdown("---")
        show_closed = st.checkbox("Poora Closed-Trades Log Dikhayein", value=False, key="show_closed_log")
        if show_closed:
            system_filter_closed = st.multiselect(
                "System(s)", SYSTEM_ORDER, default=SYSTEM_ORDER, key="closed_log_system_filter",
            )
            df_closed_show = df_closed[df_closed["System"].isin(system_filter_closed)]
            st.dataframe(df_closed_show.sort_values("Logged At (UTC)", ascending=False), use_container_width=True, hide_index=True)

        closed_csv = df_closed.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Closed Trades CSV Download Karein", closed_csv, "closed_trades_log.csv", "text/csv", key="dl_closed_trades")
    else:
        st.info("Abhi tak koi closed trade record nahi.")
else:
    st.info(
        "Abhi tak koi closed trade record nahi — pehli baar koi signal SL/trailing-stop par "
        "band hone ke baad (background scan ke agle cycle mein) yahan record banna shuru hoga."
    )

st.markdown("---")
st.header("🕐 Trading Session Analysis (Pakistan Time — PKT)")
st.caption(
    "Har closed trade ka ENTRY waqt dekh kar us waqt kaunsa major market session "
    "'khula' tha — sab kuch **Pakistan Time (PKT)** mein, koi UTC confusion nahi — "
    "taake pata chale kis session mein li gayi trades zyada TP/Win par band hoti hain "
    "aur kis mein zyada SL/Loss par."
)


def classify_session(hour_pkt):
    """Sab boundaries Pakistan Time (PKT) mein - UTC se koi lena dena nahi."""
    if 5 <= hour_pkt < 12:
        return "🌏 Asian (05:00 AM–12:00 PM PKT)"
    elif 12 <= hour_pkt < 17:
        return "🇬🇧 London (12:00 PM–05:00 PM PKT)"
    elif 17 <= hour_pkt < 21:
        return "🇬🇧+🇺🇸 London+NY Overlap (05:00 PM–09:00 PM PKT)"
    elif hour_pkt >= 21 or hour_pkt < 2:
        return "🇺🇸 New York (09:00 PM–02:00 AM PKT)"
    else:
        return "🌙 Late NY / Off-Hours (02:00 AM–05:00 AM PKT)"


def parse_pkt_hour(pkt_str):
    """'2026-09-24 05:00 PM PKT' jaisi string se PKT hour (0-23) nikalta hai - koi conversion nahi."""
    try:
        clean = str(pkt_str).replace(" PKT", "").strip()
        dt = datetime.strptime(clean, "%Y-%m-%d %I:%M %p")
        return dt.hour
    except Exception:
        return None


session_sources = []
if os.path.exists("closed_trades_log.csv"):
    _df = pd.read_csv("closed_trades_log.csv")
    if len(_df) > 0 and "Signal Time (PKT)" in _df.columns:
        _df = _df.rename(columns={"Signal Time (PKT)": "entry_time_pkt"})
        _df["is_win"] = _df["Exit Reason"] == "TARGET"
        session_sources.append(("Live Screener (sab systems)", _df[["entry_time_pkt", "is_win"]]))
if os.path.exists("manual_bot_closed_trades.csv"):
    _df = pd.read_csv("manual_bot_closed_trades.csv")
    if len(_df) > 0:
        _df["is_win"] = _df["result"] == "WIN"
        session_sources.append(("Manual Trade Bot", _df[["entry_time_pkt", "is_win"]]))
if os.path.exists("auto_bot_closed_trades.csv"):
    _df = pd.read_csv("auto_bot_closed_trades.csv")
    if len(_df) > 0:
        _df["is_win"] = _df["result"] == "WIN"
        session_sources.append(("Auto-Scan Bot", _df[["entry_time_pkt", "is_win"]]))

if len(session_sources) == 0:
    st.info("Abhi tak koi closed trade record nahi mila — session analysis ke liye pehle kuch trades band honi chahiye.")
else:
    source_names = [s[0] for s in session_sources]
    picked = st.multiselect("Kaunse record(s) shamil karein", source_names, default=source_names, key="session_source_pick")
    combined = pd.concat([df for name, df in session_sources if name in picked], ignore_index=True) if picked else pd.DataFrame()

    if len(combined) == 0:
        st.caption("Koi record select nahi kiya gaya.")
    else:
        combined["pkt_hour"] = combined["entry_time_pkt"].apply(parse_pkt_hour)
        combined = combined.dropna(subset=["pkt_hour"])
        combined["session"] = combined["pkt_hour"].apply(classify_session)

        session_order = [
            "🌏 Asian (05:00 AM–12:00 PM PKT)", "🇬🇧 London (12:00 PM–05:00 PM PKT)",
            "🇬🇧+🇺🇸 London+NY Overlap (05:00 PM–09:00 PM PKT)", "🇺🇸 New York (09:00 PM–02:00 AM PKT)",
            "🌙 Late NY / Off-Hours (02:00 AM–05:00 AM PKT)",
        ]
        rows = []
        for sess in session_order:
            sub = combined[combined["session"] == sess]
            if len(sub) == 0:
                continue
            wins = int(sub["is_win"].sum())
            total = len(sub)
            rows.append({
                "Session": sess, "Total Trades": total, "TP/Win": wins, "SL/Loss": total - wins,
                "Win Rate %": round(wins / total * 100, 1),
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            best = max(rows, key=lambda r: r["Win Rate %"])
            worst = min(rows, key=lambda r: r["Win Rate %"])
            st.caption(
                f"✅ Sab se behtar: **{best['Session']}** ({best['Win Rate %']}% Win Rate, {best['Total Trades']} trades) — "
                f"⚠️ Sab se kamzor: **{worst['Session']}** ({worst['Win Rate %']}% Win Rate, {worst['Total Trades']} trades). "
                f"Note: chhota sample (~10-15 se kam trades) size wale sessions ka number abhi bharosemand nahi."
            )
        else:
            st.caption("Session classify nahi ho saka.")

st.markdown("---")# SECTION 2: MANUAL (on-demand, apni marzi ke toggles ke sath)
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

    with st.spinner("ETH Regime check kar rahe hain..."):
        try:
            eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=800)
            eth_ema200 = eth_daily["close"].ewm(span=200, adjust=False).mean()
            eth_regime_ok = bool(eth_daily["close"].iloc[-1] > eth_ema200.iloc[-1])
        except Exception as e:
            st.warning(f"⚠️ ETH Regime check nahi ho saka ({e}) — is dafa BINA filter ke scan chalega.")
            eth_regime_ok = True
    if eth_regime_ok:
        st.success("✅ ETH Regime: BULLISH (signals ON)")
    else:
        st.warning("⚠️ ETH Regime: BEARISH — koi naya signal nahi milega (ETH apni EMA200 se neeche hai)")

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
                combo_a = (ichi_sig & ms_sig) & eth_regime_ok
                combo_b = (ema_sig & breakout_sig) & eth_regime_ok

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
                            "Signal Time (PKT)": to_pkt_str(df.loc[idx, "timestamp"]),
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
    show_charts_and_copy(df_final, "manual")

    csv = df_final.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV Download Karein", csv, "manual_dashboard.csv", "text/csv")

st.caption(
    "🟢 Green = signal ke HAQ mein. 🔴 Red = KHILAF. Rank 1 = sab se zyada "
    "'Overall Score' wala (best) trade. Compact View se sirf zaroori columns dikhte hain."
)

st.markdown("---")
st.header("📔 Trade Journal (Khud-kaar Record)")
if os.path.exists("trade_journal.csv"):
    df_journal = pd.read_csv("trade_journal.csv")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Signals Logged", len(df_journal))
    if "Verdict" in df_journal.columns:
        strong_count = df_journal["Verdict"].astype(str).str.contains("Strong", na=False).sum()
        col2.metric("Strong Signals", strong_count)
        col3.metric("Normal Signals", len(df_journal) - strong_count)

    show_journal = st.checkbox("Poora Journal Dikhayein", value=False)
    if show_journal:
        st.dataframe(df_journal.sort_values("Logged At (UTC)", ascending=False), use_container_width=True, hide_index=True)

    journal_csv = df_journal.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Journal CSV Download Karein", journal_csv, "trade_journal.csv", "text/csv")
else:
    st.info("Abhi tak koi journal entry nahi — pehla background scan chalne ke baad yahan record nazar aayega.")

st.markdown("---")
st.header("🤖 Manual Trade Bot (Coin Aap Daalein)")
st.caption(
    "Yeh koi auto-scan nahi karta — SIRF unhi coins par kaam karta hai jo aap khud "
    "'manual_watchlist.json' (GitHub par) mein daalein — chahe wo CE Buy-Only, Union AB, "
    "NEW AdvancedConfluence, Pullback-in-Uptrend ya Donchian Breakout, jis bhi system ka signal ho. "
    "Feed karne ke agle run (max 5 min) mein bot us coin par virtual trade le leta hai, us SYSTEM ke "
    "apne tasdeeq-shuda SL/TP rules ke sath (default $100, watchlist mein amount badal sakte hain). "
    "Asal paisa is mein bilkul risk mein nahi hai (paper/virtual)."
)
GITHUB_REPO = "Asifali549/Asif-Ali"
GITHUB_BRANCH = "main"
GITHUB_WATCHLIST_PATH = "manual_watchlist.json"


def push_watchlist_to_github(merged_list):
    """manual_watchlist.json ko seedha GitHub repo mein commit karta hai
    (GitHub Contents API ke zariye), taake GitHub Actions bot ko turant
    nazar aaye — koi manual GitHub-app editing na karni pare.
    Requires: Streamlit Cloud app Settings -> Secrets mein GITHUB_TOKEN
    (repo-write access wala Personal Access Token) set hona chahiye.
    Returns (success: bool, message: str).
    """
    token = st.secrets.get("GITHUB_TOKEN") if hasattr(st, "secrets") else None
    if not token:
        return False, "NO_TOKEN"

    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_WATCHLIST_PATH}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        get_resp = requests.get(api_url, headers=headers, params={"ref": GITHUB_BRANCH}, timeout=15)
        sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

        new_content = json.dumps(merged_list, indent=2)
        payload = {
            "message": "Update manual watchlist (dashboard se)",
            "content": base64.b64encode(new_content.encode("utf-8")).decode("utf-8"),
            "branch": GITHUB_BRANCH,
        }
        if sha:
            payload["sha"] = sha

        put_resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
        if put_resp.status_code in (200, 201):
            return True, "OK"
        return False, f"GitHub API error {put_resp.status_code}: {put_resp.text[:200]}"
    except Exception as exc:
        return False, f"Error: {exc}"


MANUAL_SYSTEM_BOXES = [
    ("CE Buy-Only", "CE Buy-Only", None),
    ("NEW AdvancedConfluence", "NEW AdvancedConfluence", None),
    ("Union AB — Ichimoku+MS", "Union AB", "Ichimoku+MS"),
    ("Union AB — EMA+Breakout", "Union AB", "EMA+Breakout"),
    ("Union AB Backup Tier — Ichimoku+MS", "Union AB Backup Tier", "Ichimoku+MS"),
    ("Union AB Backup Tier — EMA+Breakout", "Union AB Backup Tier", "EMA+Breakout"),
    ("Pullback-in-Uptrend", "Pullback-in-Uptrend", None),
    ("Donchian Breakout", "Donchian Breakout", None),
]

with st.expander("➕ Coin Yahan Daalein (Har System Ka Alag Khana)", expanded=False):
    st.caption(
        "Jis system ka signal mila hai usi khane mein coin ka naam likhein (ek line mein ek coin, "
        "jaise BTC/USDT). Custom amount dena ho to ':' laga kar likhein — jaise BTC/USDT:200 "
        "(warna default $100 lagega). 'Save Watchlist' dabate hi yeh seedha manual_watchlist.json "
        "mein chali jayengi — koi JSON likhne ki zaroorat nahi."
    )
    with st.form("manual_watchlist_form"):
        box_values = {}
        for label, system_name, combo_name in MANUAL_SYSTEM_BOXES:
            box_values[label] = st.text_area(label, value="", height=70, key=f"wl_box_{label}")
        submitted = st.form_submit_button("💾 Save Watchlist")

    if submitted:
        existing = []
        if os.path.exists("manual_watchlist.json"):
            with open("manual_watchlist.json") as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = []

        new_entries = []
        for label, system_name, combo_name in MANUAL_SYSTEM_BOXES:
            raw_text = box_values[label]
            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                amount = None
                if ":" in line:
                    sym_part, amt_part = line.rsplit(":", 1)
                    sym_part = sym_part.strip()
                    try:
                        amount = float(amt_part.strip())
                    except ValueError:
                        sym_part = line
                        amount = None
                else:
                    sym_part = line
                symbol = sym_part.upper().replace(" ", "")
                if "/" not in symbol:
                    symbol = f"{symbol}/USDT"
                entry = {"symbol": symbol, "amount": amount, "system": system_name, "combo": combo_name}
                new_entries.append(entry)

        def _entry_key(e):
            if isinstance(e, dict):
                return (e.get("symbol"), e.get("system", "CE Buy-Only"), e.get("combo"))
            return (e, "CE Buy-Only", None)

        existing_keys = {_entry_key(e) for e in existing}
        merged = list(existing)
        added_count = 0
        for e in new_entries:
            k = _entry_key(e)
            if k not in existing_keys:
                merged.append(e)
                existing_keys.add(k)
                added_count += 1

        if added_count == 0:
            st.info("Koi nayi entry nahi mili (ya pehle se watchlist mein maujood hai).")
        else:
            pushed, msg = push_watchlist_to_github(merged)
            if pushed:
                # local copy bhi update kar dein taake yahan turant nazar aaye
                with open("manual_watchlist.json", "w") as f:
                    json.dump(merged, f, indent=2)
                st.success(f"✅ {added_count} nayi coin(s) seedha GitHub par save ho gayin. GitHub Actions ke "
                           f"agle run (max 5 min) mein bot inhein process karega.")
                st.rerun()
            elif msg == "NO_TOKEN":
                with open("manual_watchlist.json", "w") as f:
                    json.dump(merged, f, indent=2)
                st.warning(
                    "⚠️ Yeh sirf is app ke local copy mein save hui hai — GitHub par NAHI gayi, isliye bot "
                    "ko nazar nahi aayegi. GitHub se seedha auto-save karne ke liye ek baar "
                    "'GITHUB_TOKEN' Streamlit app Settings → Secrets mein add karwana hoga (mujhe bata dein, "
                    "main step-by-step bata deta hoon). Filhal neeche di gayi JSON copy kar ke khud "
                    "GitHub app mein 'manual_watchlist.json' file mein paste kar dein:"
                )
                st.code(json.dumps(merged, indent=2), language="json")
            else:
                st.error(f"❌ GitHub par save nahi ho saki: {msg}\n\nNeeche di JSON copy kar ke khud GitHub "
                          f"app mein 'manual_watchlist.json' mein paste kar dein:")
                st.code(json.dumps(merged, indent=2), language="json")

if os.path.exists("manual_watchlist.json"):
    try:
        with open("manual_watchlist.json") as f:
            pending_watchlist = json.load(f)
    except Exception:
        pending_watchlist = None
        st.error(
            "⚠️ 'manual_watchlist.json' file mein JSON theek nahi hai (koi extra bracket/comma reh gaya "
            "hoga), isliye is file ko padha nahi ja saka. Upar wale form se 'Save Watchlist' dabayein — "
            "woh khud file ko sahi format mein dobara likh dega. Ya GitHub par file kholkar poori "
            "content mita kar sirf `[]` likh dein aur commit kar dein."
        )
    if pending_watchlist:
        pending_labels = []
        for e in pending_watchlist:
            if isinstance(e, dict):
                pending_labels.append(f"{e.get('symbol')} ({e.get('system', 'CE Buy-Only')})")
            else:
                pending_labels.append(f"{e} (CE Buy-Only)")
        st.caption(f"⏳ Pending (agle run mein process hongi): {', '.join(pending_labels)}")

_mb_state_loaded = _safe_json_load("manual_bot_state.json") if os.path.exists("manual_bot_state.json") else None
if os.path.exists("manual_bot_state.json") and _mb_state_loaded is None:
    st.error("⚠️ 'manual_bot_state.json' file corrupt ho gayi hai — bot ke agle run par yeh khud theek ho jayegi.")
if _mb_state_loaded is not None:
    mb_state = _mb_state_loaded

    cash = mb_state.get("cash", 0)
    open_positions = mb_state.get("positions", {})
    total_equity = mb_state.get("total_equity_usd", cash)
    starting_capital = mb_state.get("starting_capital_usd", 1000.0)
    overall_pnl = total_equity - starting_capital
    overall_pnl_pct = (overall_pnl / starting_capital * 100) if starting_capital else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Equity", f"${total_equity:,.2f}", f"{overall_pnl_pct:+.2f}%")
    c2.metric("Free Cash", f"${cash:,.2f}")
    c3.metric("Open Positions", f"{len(open_positions)} / {mb_state.get('max_concurrent_positions', 8)}")
    c4.metric("Per-Trade Size", f"${mb_state.get('position_size_usd', 100):,.2f}")

    if len(open_positions) > 0:
        st.markdown("**🟢 Abhi Khuli Hui Manual Trades**")
        rows = []
        for symbol, pos in open_positions.items():
            system_label = pos.get("system", "CE Buy-Only")
            if pos.get("combo"):
                system_label += f" ({pos['combo']})"
            rows.append({
                "Coin": symbol,
                "System": system_label,
                "Entry": pos.get("entry_price"),
                "Current": pos.get("current_price"),
                "Trail Stop (SL)": pos.get("trail_stop"),
                "TP": pos.get("tp_price") if pos.get("tp_price") is not None else "N/A (trailing)",
                "Unrealized P/L %": pos.get("unrealized_pnl_pct"),
                "Capital ($)": pos.get("capital_allocated"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Abhi koi manual trade khuli nahi hai — coin 'manual_watchlist.json' mein daal kar feed karein.")

    if os.path.exists("manual_bot_closed_trades.csv"):
        df_mb_closed = pd.read_csv("manual_bot_closed_trades.csv")
        if len(df_mb_closed) > 0:
            wins = df_mb_closed[df_mb_closed["result"] == "WIN"]
            losses = df_mb_closed[df_mb_closed["result"] == "LOSS"]
            win_rate = len(wins) / len(df_mb_closed) * 100
            gross_win = wins["realized_pnl_usd"].sum()
            gross_loss = abs(losses["realized_pnl_usd"].sum())
            pf = (gross_win / gross_loss) if gross_loss > 0 else None

            st.markdown("**📒 Band Ho Chuki Manual Trades**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Closed", len(df_mb_closed))
            c2.metric("Win Rate", f"{win_rate:.1f}%")
            c3.metric("Profit Factor", f"{pf:.2f}" if pf is not None else "N/A")
            c4.metric("Realized P&L", f"${df_mb_closed['realized_pnl_usd'].sum():,.2f}")

            show_mb = st.checkbox("Poori Manual-Trade History Dikhayein", value=False, key="show_manual_bot_log")
            if show_mb:
                st.dataframe(df_mb_closed.sort_values("exit_time_pkt", ascending=False), use_container_width=True, hide_index=True)

            mb_csv = df_mb_closed.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Manual Trades CSV Download Karein", mb_csv, "manual_bot_closed_trades.csv", "text/csv", key="dl_manual_bot")
        else:
            st.caption("Abhi tak koi manual trade band nahi hui.")
else:
    st.info(
        "Manual trade bot abhi tak nahi chala — GitHub repo mein "
        "'.github/workflows/manual_trade_bot.yml' hona chahiye, chalne ke thodi der baad yahan "
        "result nazar aayega."
    )

st.markdown("---")
st.header("🎲 Auto-Scan Trade Bot (Dummy/Paper — Sab 5 Systems Khud Scan Karta Hai)")
st.caption(
    "Yeh manual bot ka 'auto' sāthi hai — khud 150 coins scan karta hai aur jaise hi kisi bhi "
    "system (Union AB, NEW AdvancedConfluence, CE Buy-Only, Pullback-in-Uptrend, Donchian Breakout) "
    "ka fresh signal bane, khud hi wahi (paper/virtual) trade le leta hai — koi manual feed ki zaroorat "
    "nahi. Capital/ledger manual bot se BILKUL ALAG hai. Asal paisa yahan bhi risk mein nahi (paper)."
)
_ab_state_loaded = _safe_json_load("auto_bot_state.json") if os.path.exists("auto_bot_state.json") else None
if os.path.exists("auto_bot_state.json") and _ab_state_loaded is None:
    st.error("⚠️ 'auto_bot_state.json' file corrupt ho gayi hai — bot ke agle run par yeh khud theek ho jayegi.")
if _ab_state_loaded is not None:
    ab_state = _ab_state_loaded

    cash = ab_state.get("cash", 0)
    open_positions = ab_state.get("positions", {})
    total_equity = ab_state.get("total_equity_usd", cash)
    starting_capital = ab_state.get("starting_capital_usd", 2000.0)
    overall_pnl = total_equity - starting_capital
    overall_pnl_pct = (overall_pnl / starting_capital * 100) if starting_capital else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Equity", f"${total_equity:,.2f}", f"{overall_pnl_pct:+.2f}%")
    c2.metric("Free Cash", f"${cash:,.2f}")
    c3.metric("Open Positions", f"{len(open_positions)} / {ab_state.get('max_concurrent_positions', 15)}")
    c4.metric("Per-Trade Size", f"${ab_state.get('position_size_usd', 100):,.2f}")

    if len(open_positions) > 0:
        st.markdown("**🟢 Abhi Khuli Hui Auto Trades**")
        rows = []
        for key, pos in open_positions.items():
            system_label = pos.get("system", "CE Buy-Only")
            if pos.get("combo"):
                system_label += f" ({pos['combo']})"
            rows.append({
                "Coin": pos.get("symbol", key.split("|")[0]),
                "System": system_label,
                "Entry": pos.get("entry_price"),
                "Current": pos.get("current_price"),
                "Trail Stop (SL)": pos.get("trail_stop"),
                "TP": pos.get("tp_price") if pos.get("tp_price") is not None else "N/A (trailing)",
                "Unrealized P/L %": pos.get("unrealized_pnl_pct"),
                "Capital ($)": pos.get("capital_allocated"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Abhi koi auto trade khuli nahi hai.")

    if os.path.exists("auto_bot_closed_trades.csv"):
        df_ab_closed = pd.read_csv("auto_bot_closed_trades.csv")
        if len(df_ab_closed) > 0:
            wins = df_ab_closed[df_ab_closed["result"] == "WIN"]
            losses = df_ab_closed[df_ab_closed["result"] == "LOSS"]
            win_rate = len(wins) / len(df_ab_closed) * 100
            gross_win = wins["realized_pnl_usd"].sum()
            gross_loss = abs(losses["realized_pnl_usd"].sum())
            pf = (gross_win / gross_loss) if gross_loss > 0 else None

            st.markdown("**📒 Band Ho Chuki Auto Trades**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Closed", len(df_ab_closed))
            c2.metric("Win Rate", f"{win_rate:.1f}%")
            c3.metric("Profit Factor", f"{pf:.2f}" if pf is not None else "N/A")
            c4.metric("Realized P&L", f"${df_ab_closed['realized_pnl_usd'].sum():,.2f}")

            if "system" in df_ab_closed.columns:
                st.markdown("**System ke hisaab se breakdown**")
                for sys_name in df_ab_closed["system"].unique():
                    sub = df_ab_closed[df_ab_closed["system"] == sys_name]
                    sub_wins = sub[sub["result"] == "WIN"]
                    sub_wr = len(sub_wins) / len(sub) * 100
                    st.caption(f"**{sys_name}**: {len(sub)} closed, Win Rate {sub_wr:.1f}%, "
                               f"P&L ${sub['realized_pnl_usd'].sum():,.2f}")

            show_ab = st.checkbox("Poori Auto-Trade History Dikhayein", value=False, key="show_auto_bot_log")
            if show_ab:
                st.dataframe(df_ab_closed.sort_values("exit_time_pkt", ascending=False), use_container_width=True, hide_index=True)

            ab_csv = df_ab_closed.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Auto Trades CSV Download Karein", ab_csv, "auto_bot_closed_trades.csv", "text/csv", key="dl_auto_bot")
        else:
            st.caption("Abhi tak koi auto trade band nahi hui.")
else:
    st.info(
        "Auto-scan trade bot abhi tak nahi chala — GitHub repo mein "
        "'.github/workflows/auto_scan_trade_bot.yml' hona chahiye, chalne ke thodi der baad yahan "
        "result nazar aayega."
    )
