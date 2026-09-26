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
import time
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

COMPACT_COLUMNS = ["Rank", "System", "Category", "Coin", "Signal Time (PKT)", "Combo", "Overall Score %", "Verdict", "Entry", "Live Price", "Live P/L %", "Current", "P/L %", "Trail Stop", "Take Profit"]


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_prices(symbols):
    """Exchange (KuCoin) se abhi ki taaza qeemat - ek hi API call, 60 second cache."""
    symbols = list(symbols)
    if not symbols:
        return {}
    try:
        ex = get_exchange()
        tickers = ex.fetch_tickers(symbols)
        return {s: float(t["last"]) for s, t in tickers.items() if t and t.get("last")}
    except Exception:
        return {}


def add_live_price_columns(df):
    """'Live Price' (abhi ki qeemat) aur 'Live P/L %' (Entry ke muqable) columns jorta hai.
    'Current' = scan ke waqt ki qeemat (kuch minute purani ho sakti hai)."""
    if df is None or len(df) == 0 or "Coin" not in df.columns:
        return df, False
    prices = fetch_live_prices(tuple(sorted(df["Coin"].dropna().unique())))
    if not prices:
        return df, False
    df = df.copy()
    df["Live Price"] = df["Coin"].map(prices)
    entry = pd.to_numeric(df.get("Entry"), errors="coerce")
    df["Live P/L %"] = ((df["Live Price"] - entry) / entry * 100).round(2)
    return df, True

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


def style_and_show(df, compact, select_key=None):
    """Table dikhata hai. select_key diya jaye to lines select (tap) ho sakti
    hain - return: select ki gayi lines ki positions (list)."""
    if compact:
        show_cols = [c for c in COMPACT_COLUMNS if c in df.columns]
        df = df[show_cols]

    def apply_row_colors(row):
        return [color_value(col, row[col]) if col in df.columns else "" for col in df.columns]

    styled = df.style.apply(apply_row_colors, axis=1)
    if select_key:
        try:
            event = st.dataframe(
                styled, use_container_width=True, hide_index=True,
                on_select="rerun", selection_mode="multi-row", key=select_key,
            )
            return list(event.selection.rows)
        except TypeError:
            pass   # purana Streamlit version - selection support nahi, sirf table dikhao
    st.dataframe(styled, use_container_width=True, hide_index=True)
    return []


# ============================================================
# MANUAL BOT WATCHLIST - GitHub par seedha save (Live table + form dono isi ko istemal karte hain)
# ============================================================
GITHUB_REPO = "Asifali549/Asif-Ali"
GITHUB_BRANCH = "main"
GITHUB_WATCHLIST_PATH = "manual_watchlist.json"
DEFAULT_MANUAL_SYSTEM = "CE Buy-Only"
UNION_SYSTEMS = ("Union AB", "Union AB Backup Tier")
MANUAL_BOT_SYSTEMS = (
    "CE Buy-Only", "NEW AdvancedConfluence", "Union AB", "Union AB Backup Tier",
    "Pullback-in-Uptrend", "Donchian Breakout",
)
UNION_COMBOS = ("Ichimoku+MS", "EMA+Breakout")


def _norm_symbol(sym):
    s = str(sym or "").upper().replace(" ", "").strip()
    if s and "/" not in s:
        s = f"{s}/USDT"
    return s


def _wl_key(e):
    """Watchlist entry (dict ya plain string) ki pehchan: (symbol, system, combo)."""
    if isinstance(e, dict):
        system = e.get("system") or DEFAULT_MANUAL_SYSTEM
        combo = e.get("combo") if system in UNION_SYSTEMS else None
        return (_norm_symbol(e.get("symbol")), system, combo or None)
    return (_norm_symbol(e), DEFAULT_MANUAL_SYSTEM, None)


def _read_local_watchlist():
    try:
        with open("manual_watchlist.json") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def add_entries_to_github_watchlist(new_entries):
    """
    Nayi entries ko GitHub wali manual_watchlist.json mein jorta hai.
    Hamesha GitHub ki TAAZA file parh kar merge karta hai (app ki purani
    local copy par bharosa nahi) - taake bot ki beech mein ki gayi
    tabdeeli mite nahi. Takraao (409/422) par 3 dafa dobara koshish.
    Returns (status, added_count, message); status = OK / NO_TOKEN / ERROR.
    """
    token = None
    try:
        token = st.secrets.get("GITHUB_TOKEN")
    except Exception:
        token = None
    if not token:
        return "NO_TOKEN", 0, ""

    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_WATCHLIST_PATH}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

    for attempt in range(3):
        try:
            get_resp = requests.get(api_url, headers=headers, params={"ref": GITHUB_BRANCH}, timeout=15)
            if get_resp.status_code == 200:
                data = get_resp.json()
                sha = data.get("sha")
                try:
                    text = base64.b64decode(data.get("content", "")).decode("utf-8").strip()
                    existing = json.loads(text) if text else []
                    if not isinstance(existing, list):
                        existing = []
                except Exception:
                    existing = []   # kharab file - saaf list se shuru
            elif get_resp.status_code == 404:
                existing, sha = [], None
            else:
                return "ERROR", 0, f"GitHub API error {get_resp.status_code}: {get_resp.text[:200]}"

            keys = {_wl_key(e) for e in existing}
            merged = list(existing)
            added = 0
            for e in new_entries:
                k = _wl_key(e)
                if k[0] and k not in keys:
                    merged.append(e)
                    keys.add(k)
                    added += 1

            if added == 0:
                return "OK", 0, "ALREADY"

            payload = {
                "message": "Update manual watchlist (dashboard se)",
                "content": base64.b64encode(json.dumps(merged, indent=2).encode("utf-8")).decode("utf-8"),
                "branch": GITHUB_BRANCH,
            }
            if sha:
                payload["sha"] = sha

            put_resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
            if put_resp.status_code in (200, 201):
                try:
                    with open("manual_watchlist.json", "w") as f:
                        json.dump(merged, f, indent=2)   # local copy, taake "Pending" foran nazar aaye
                except Exception:
                    pass
                return "OK", added, ""
            if put_resp.status_code in (409, 422):
                time.sleep(1.5)   # file beech mein badal gayi (bot ne likha) - dobara taaza parh kar koshish
                continue
            return "ERROR", 0, f"GitHub API error {put_resp.status_code}: {put_resp.text[:200]}"
        except Exception as exc:
            return "ERROR", 0, f"Error: {exc}"

    return "ERROR", 0, "File baar baar badal rahi thi - thori der baad dobara koshish karein"


AUTO_CONTROL_PATH = "auto_bot_control.json"


def read_auto_control():
    """Auto Bot button ki halat (app ki local copy). File na ho = chal raha hai."""
    try:
        with open(AUTO_CONTROL_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"paused": False}
    except Exception:
        return {"paused": False}


def write_auto_control(paused):
    """auto_bot_control.json ko GitHub par likhta hai. Returns (ok, message)."""
    try:
        token = st.secrets.get("GITHUB_TOKEN")
    except Exception:
        token = None
    if not token:
        return False, "GITHUB_TOKEN Streamlit Secrets mein nahi mila"

    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{AUTO_CONTROL_PATH}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    body_obj = {
        "paused": bool(paused),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "by": "dashboard",
    }
    for attempt in range(3):
        try:
            get_resp = requests.get(api_url, headers=headers, params={"ref": GITHUB_BRANCH}, timeout=15)
            if get_resp.status_code == 200:
                sha = get_resp.json().get("sha")
            elif get_resp.status_code == 404:
                sha = None
            else:
                return False, f"GitHub API error {get_resp.status_code}: {get_resp.text[:200]}"
            payload = {
                "message": ("Auto Bot ROKA gaya" if paused else "Auto Bot dobara CHALAYA gaya") + " (dashboard se)",
                "content": base64.b64encode(json.dumps(body_obj, indent=2).encode("utf-8")).decode("utf-8"),
                "branch": GITHUB_BRANCH,
            }
            if sha:
                payload["sha"] = sha
            put_resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
            if put_resp.status_code in (200, 201):
                try:
                    with open(AUTO_CONTROL_PATH, "w") as f:
                        json.dump(body_obj, f, indent=2)
                except Exception:
                    pass
                return True, ""
            if put_resp.status_code in (409, 422):
                time.sleep(1.5)
                continue
            return False, f"GitHub API error {put_resp.status_code}: {put_resp.text[:200]}"
        except Exception as exc:
            return False, f"Error: {exc}"
    return False, "GitHub par file baar baar badal rahi thi - dobara koshish karein"


def live_row_to_watchlist_entry(row, amount=None):
    """Live screener ki line -> manual bot watchlist entry (sahi system/combo ke sath)."""
    system = row.get("System")
    if system not in MANUAL_BOT_SYSTEMS:
        return None
    combo = row.get("Combo") if system in UNION_SYSTEMS else None
    if system in UNION_SYSTEMS and combo not in UNION_COMBOS:
        return None
    symbol = _norm_symbol(row.get("Coin"))
    if not symbol:
        return None
    return {"symbol": symbol, "amount": amount, "system": system, "combo": combo}


def send_selected_to_manual_bot(df_rows, selected_positions, key):
    """Select ki gayi live lines ke neeche 'Manual Bot mein bhejein' button."""
    if not selected_positions:
        st.caption("👆 Kisi coin ki line ke bayen (left) khane par tap karein — "
                   "'🤖 Manual Bot mein bhejein' button aa jayega.")
        return

    picked = df_rows.iloc[[p for p in selected_positions if p < len(df_rows)]]
    entries, labels = [], []
    for _, r in picked.iterrows():
        e = live_row_to_watchlist_entry(r)
        if e is not None:
            entries.append(e)
            labels.append(f"{e['symbol']} ({e['system']}{' — ' + e['combo'] if e['combo'] else ''})")

    if not entries:
        st.warning("Is line ka system Manual Bot mein pehchana nahi gaya.")
        return

    st.markdown("**Chuni gayi:** " + ", ".join(labels))
    c1, c2 = st.columns([1, 2])
    amount = c1.number_input("Amount ($, virtual)", min_value=10.0, value=100.0, step=10.0, key=f"{key}_amt")
    if c2.button(f"🤖 Manual Bot mein bhejein ({len(entries)})", type="primary", key=f"{key}_send"):
        amt = None if amount == 100.0 else float(amount)   # $100 = bot ka default
        for e in entries:
            e["amount"] = amt
        status, added, msg = add_entries_to_github_watchlist(entries)
        if status == "OK" and added > 0:
            st.success(f"✅ {added} coin(s) Manual Bot ki watchlist mein apne apne system ke khane mein chali gayin. "
                       f"Bot ke agle run (max ~5-10 min) mein trade khul jayegi.")
        elif status == "OK":
            st.info("Ye coin(s) pehle se watchlist mein maujood hain — bot agle run mein utha lega.")
        elif status == "NO_TOKEN":
            st.error("⚠️ GITHUB_TOKEN Streamlit Secrets mein nahi mila — coin GitHub par nahi gaya.")
        else:
            st.error(f"❌ GitHub par save nahi ho saka: {msg}")


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

        lp1, lp2 = st.columns([3, 1])
        if lp2.button("🔄 Live Qeemat Taaza", key="refresh_live_prices"):
            fetch_live_prices.clear()
        df_all, live_ok = add_live_price_columns(df_all)
        if live_ok:
            lp1.caption(
                f"💹 **Live Price** = abhi ki qeemat ({pd.Timestamp.now(tz='Asia/Karachi').strftime('%I:%M %p')} PKT tak, "
                "har minute taaza) · **Live P/L %** = Entry se abhi tak ka farq · "
                "**Current** = scan ke waqt ki qeemat (thori purani)."
            )
        else:
            lp1.caption("⚠️ Live qeemat is waqt exchange se nahi mil saki — 'Current' scan ke waqt ki qeemat hai.")

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
                sel_new = style_and_show(new_df, compact_live, select_key=f"sel_{key_slug}_new")
                send_selected_to_manual_bot(new_df, sel_new, f"mb_{key_slug}_new")
                show_charts_and_copy(new_df, f"live_{key_slug}_new")
                any_shown = True

            if len(open_df) > 0:
                st.markdown(f"**🔵 Chal Rahi Trades — Open ({len(open_df)})**")
                sel_open = style_and_show(open_df, compact_live, select_key=f"sel_{key_slug}_open")
                send_selected_to_manual_bot(open_df, sel_open, f"mb_{key_slug}_open")
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
st.info(
    "💡 **Aasan tareeqa:** Upar LIVE table mein kisi coin ki line par tap karein aur "
    "'🤖 Manual Bot mein bhejein' dabayein — coin khud apne system/combo ke khane mein aa jayega."
)

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
                symbol = _norm_symbol(sym_part)
                # Sirf asal coin naam (harf/number) - ghalti se paste hue emoji/nishan rad
                if not symbol or not symbol.replace("/", "").isalnum() or not symbol.isascii():
                    st.warning(f"⚠️ '{line}' coin ka sahi naam nahi lagta — chhor diya gaya.")
                    continue
                entry = {"symbol": symbol, "amount": amount, "system": system_name, "combo": combo_name}
                new_entries.append(entry)

        if not new_entries:
            st.info("Koi nayi entry nahi mili.")
        else:
            status, added_count, msg = add_entries_to_github_watchlist(new_entries)
            if status == "OK" and added_count > 0:
                st.success(f"✅ {added_count} nayi coin(s) seedha GitHub par save ho gayin. GitHub Actions ke "
                           f"agle run (max 5 min) mein bot inhein process karega.")
            elif status == "OK":
                st.info("Ye coin(s) pehle se watchlist mein maujood hain.")
            elif status == "NO_TOKEN":
                merged = _read_local_watchlist()
                keys = {_wl_key(e) for e in merged}
                for e in new_entries:
                    if _wl_key(e) not in keys:
                        merged.append(e)
                        keys.add(_wl_key(e))
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
                st.error(f"❌ GitHub par save nahi ho saki: {msg}\n\nNeeche di JSON (nayi entries) copy kar ke "
                          f"khud GitHub app mein 'manual_watchlist.json' ki list mein shamil kar dein:")
                st.code(json.dumps(new_entries, indent=2), language="json")

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
                "TP": f"{pos.get('tp_price')}" if pos.get("tp_price") is not None else "N/A (trailing)",
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

# ---- ⏸️ / ▶️ Auto Bot ko rokne / chalane ka button ----
_ctrl = read_auto_control()
_is_paused = bool(_ctrl.get("paused"))
_bc1, _bc2 = st.columns([3, 2])
if _is_paused:
    _bc1.error("⏸️ **Auto Bot ROKA HUA hai** — koi nayi trade nahi lega. "
               "Khuli trades ka SL/TP barabar chalta rahega.")
    _btn = _bc2.button("▶️ Bot Dobara Chalayein", type="primary", key="auto_bot_resume", use_container_width=True)
else:
    _bc1.success("▶️ **Auto Bot chal raha hai** — naye signal par trade le sakta hai.")
    _btn = _bc2.button("⏸️ Nayi Entry Rokein", key="auto_bot_pause", use_container_width=True)
if _btn:
    _ok, _msg = write_auto_control(not _is_paused)
    if _ok:
        st.session_state["auto_ctrl_msg"] = (
            "⏸️ Bot roka gaya — agle check par (aam taur par chand minute) nayi trade lena band kar dega."
            if not _is_paused else
            "▶️ Bot dobara chalaya gaya — agle run se naye signal par trade lena shuru karega."
        )
        st.rerun()
    else:
        st.error(f"❌ Button ki halat GitHub par save nahi ho saki: {_msg}")
if st.session_state.get("auto_ctrl_msg"):
    st.info(st.session_state.pop("auto_ctrl_msg"))

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

    def _pkt(iso):
        try:
            return pd.Timestamp(iso).tz_convert("Asia/Karachi").strftime("%d %b %I:%M %p PKT")
        except Exception:
            return str(iso)

    _now = pd.Timestamp.now(tz="UTC")
    _crash = ab_state.get("crash_until")
    _pause = ab_state.get("pause_until")
    _mlevel = ab_state.get("market_level")
    _minfo = ab_state.get("market_info", "")
    if _crash and pd.Timestamp(_crash) > _now:
        st.error(f"🚨 **Market Crash Guard:** BTC/ETH mein bari giravat — nayi entry **{_pkt(_crash)}** tak band (khuli trades apne SL par chal rahi hain). ({_minfo})")
    elif _pause and pd.Timestamp(_pause) > _now:
        st.warning(f"⏸️ **Loss-streak break:** lagatar SL ki wajah se nayi entry **{_pkt(_pause)}** tak band.")
    elif _mlevel == "CAUTION":
        st.warning(f"⚠️ **Ehtiyat:** BTC/ETH tezi se gir rahe hain — nayi entry abhi ruki hui hai. ({_minfo})")
    elif _mlevel == "OK":
        st.caption(f"✅ Market normal — bot nayi entry le sakta hai. ({_minfo})")

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
                "TP": f"{pos.get('tp_price')}" if pos.get("tp_price") is not None else "N/A (trailing)",
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
