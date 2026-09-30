"""
Live Dashboard - sirf wo 2 strategies jo sakht tests mein PASS huin:
  1) 📈 Ichimoku 4H   (Ichimoku + Market Structure, 4H, CE 16/5.5 + TP 3R)
  2) 🐢 Donchian Daily (20-din breakout, BTC>EMA50, CE 22/4x trailing)

Data dono bots ki files se aata hai (GitHub Actions har run ke baad commit karte hain):
  ichimoku4h_paper_state.json / ichimoku4h_paper_trades.csv / ichimoku4h_signals.json
  donchian_paper_state.json   / donchian_paper_trades.csv   / donchian_signals.json

Chalayen: streamlit run live_colorful_dashboard.py
"""

import json
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

from data_fetcher import get_exchange

GITHUB_REPO = "Asifali549/Asif-Ali"
GITHUB_BRANCH = "main"

SYSTEMS = {
    "Ichimoku 4H": {
        "badge": "📈 Ichimoku 4H",
        "state": "ichimoku4h_paper_state.json",
        "trades": "ichimoku4h_paper_trades.csv",
        "signals": "ichimoku4h_signals.json",
        "workflow": "ichimoku4h_bot.yml",
        "stale_min": 330,
        "every": "har 4 ghante (5:10, 9:10, 1:10 ... PKT)",
        "entry_col": "entry_bar", "exit_col": "exit_bar",
        "new_signal_hours": 8,
        "rules": [
            ("Timeframe", "4H candle"),
            ("Entry", "EK HI candle par Ichimoku (Tenkan>Kijun, cloud ke ooper, EMA200 trend, volume 2x) + Market Structure (swing high break)"),
            ("Market filter", "Koi nahi (coin ka apna EMA200/EMA50 trend kaafi hai - test mein filters se faida nahi hua)"),
            ("Stop (SL)", "Chandelier 16 candles, 5.5x ATR - sirf ooper jata hai"),
            ("Take Profit", "3R (Entry se SL ke faasle ka 3 guna)"),
            ("Exchange par", "OCO order (TP ooper, SL neeche); 🔼 aaye to SL ooper karein"),
            ("Size", "Har trade 1% risk, max 10 trades, ek coin max 20%"),
        ],
        "backtest": {"win": 41.4, "pf": 1.90, "cagr": 26.6, "dd": -11.0,
                     "note": "5.3 saal (2021-2026, 2022 crash samet), sakht usool, random-control se behtar"},
    },
    "Donchian Daily": {
        "badge": "🐢 Donchian Daily",
        "state": "donchian_paper_state.json",
        "trades": "donchian_paper_trades.csv",
        "signals": "donchian_signals.json",
        "workflow": "donchian_daily_bot.yml",
        "stale_min": 1560,
        "every": "rozana 5:15 AM PKT (daily candle band hone ke baad)",
        "entry_col": "entry_day", "exit_col": "exit_day",
        "new_signal_hours": 30,
        "rules": [
            ("Timeframe", "Daily candle"),
            ("Entry", "Daily close pichle 20 din ke sab se oonche high se ooper band ho"),
            ("Market filter", "BTC daily close > BTC EMA50 (warna nayi entry nahi)"),
            ("Stop (SL)", "Chandelier 22 din, 4x ATR - sirf ooper jata hai (rozana update karein)"),
            ("Take Profit", "Koi nahi - trend ke sath chalti hai jab tak SL na lage"),
            ("Exchange par", "Stop-loss order; Telegram mein naya SL aaye to update karein"),
            ("Size", "Har trade 1% risk, max 10 trades, ek coin max 20%"),
        ],
        "backtest": {"win": 37.0, "pf": 1.55, "cagr": 26.0, "dd": -31.0,
                     "note": "6.5 saal (2020-2026, 2022 crash samet), 3 alag tests mein PASS"},
    },
}


# ============================================================
# Helpers
# ============================================================
def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def load_csv(path):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_prices(symbols):
    """KuCoin se abhi ki qeemat - ek hi API call, 60 second cache."""
    symbols = list(symbols)
    if not symbols:
        return {}
    try:
        ex = get_exchange()
        tickers = ex.fetch_tickers(symbols)
        return {s: float(t["last"]) for s, t in tickers.items() if t and t.get("last")}
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def get_fear_greed():
    try:
        item = requests.get("https://api.alternative.me/fng/", timeout=10).json()["data"][0]
        return int(item["value"]), item["value_classification"]
    except Exception:
        return None, None


def trigger_github_workflow(workflow_file):
    """GitHub Actions ko workflow_dispatch - dashboard se hi system dobara chalane ke liye."""
    try:
        token = st.secrets.get("GITHUB_TOKEN")
    except Exception:
        token = None
    if not token:
        return False, "GITHUB_TOKEN Streamlit Secrets mein nahi mila."
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{workflow_file}/dispatches"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    try:
        resp = requests.post(url, headers=headers, json={"ref": GITHUB_BRANCH}, timeout=15)
        if resp.status_code in (204, 201):
            return True, ""
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, str(e)


def tradingview_url(coin, interval=None):
    base = str(coin).split("/")[0]
    url = f"https://www.tradingview.com/chart/?symbol=KUCOIN:{base}USDT"
    if interval:
        url += f"&interval={interval}"
    return url


def to_pkt(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("Asia/Karachi")


def pkt_str(ts):
    try:
        return to_pkt(ts).strftime("%Y-%m-%d %I:%M %p PKT")
    except Exception:
        return str(ts)


def fmt_px(x):
    try:
        x = float(x)
    except Exception:
        return "—"
    if x >= 100:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6g}"


def link_column(label="📊 Chart"):
    try:
        return st.column_config.LinkColumn(label, display_text="TradingView ↗")
    except TypeError:
        return st.column_config.LinkColumn(label)


def color_pl(v):
    try:
        v = float(v)
    except Exception:
        return ""
    if v > 0:
        return "background-color: rgba(34,197,94,0.25)"
    if v < 0:
        return "background-color: rgba(239,68,68,0.25)"
    return ""


def stretch_df(data, **kw):
    """Naye Streamlit mein width='stretch', purane mein use_container_width=True."""
    try:
        return st.dataframe(data, width="stretch", **kw)
    except TypeError:
        return st.dataframe(data, use_container_width=True, **kw)


def show_table(df, pl_cols=(), link_cols=("Chart",)):
    cfg = {c: link_column() for c in link_cols if c in df.columns}
    styled = df.style
    for c in pl_cols:
        if c in df.columns:
            styled = styled.map(color_pl, subset=[c]) if hasattr(styled, "map") else styled.applymap(color_pl, subset=[c])
    stretch_df(styled, hide_index=True, column_config=cfg)


def position_size_box(coin, entry, sl, key):
    """Sidebar ke Capital / Risk % ke mutabiq position size + copy karne wala setup."""
    try:
        entry, sl = float(entry), float(sl)
    except Exception:
        return
    if not (entry > sl > 0):
        return
    risk_dollars = total_capital * risk_pct_per_trade / 100
    units = risk_dollars / (entry - sl)
    value = units * entry
    cap_note = ""
    if value > total_capital * 0.20:
        value_capped = total_capital * 0.20
        cap_note = f" ⚠️ 20% had: ${value_capped:,.0f} tak rakhein"
    st.info(f"💰 **Position size** (Capital ${total_capital:,.0f}, Risk {risk_pct_per_trade}%): "
            f"**{units:.6g} {coin.split('/')[0]}** (~${value:,.2f}), SL laga to nuqsan ~${risk_dollars:,.2f}{cap_note}")


# ============================================================
# Page
# ============================================================
st.set_page_config(page_title="Live Dashboard", layout="wide")
st.title("🎯 Live Dashboard — Ichimoku 4H + Donchian Daily")
st.caption("Sirf wo 2 strategies jo sakht tests (lookahead-free, random-control, portfolio, 2022 crash) mein PASS huin. "
           "Signals aur paper trades dono bots khud chalate hain (GitHub Actions).")

st.sidebar.header("💰 Position Sizing Calculator")
total_capital = st.sidebar.number_input("Total Capital ($)", min_value=0.0, value=1000.0, step=100.0)
risk_pct_per_trade = st.sidebar.number_input("Risk % per Trade", min_value=0.1, max_value=100.0, value=1.0, step=0.5)
st.sidebar.caption("Har trade mein Entry aur SL ke farq ke mutabiq position size khud nikalta hai. "
                   "Dono strategies ka test 1% risk par hua hai.")
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Live qeemat taaza karein"):
    fetch_live_prices.clear()

fg_val, fg_label = get_fear_greed()
top1, top2, top3 = st.columns(3)
if fg_val is not None:
    top1.metric("BTC Fear & Greed", f"{fg_val}/100", fg_label)
top2.metric("Waqt (PKT)", pd.Timestamp.now(tz="Asia/Karachi").strftime("%d %b %I:%M %p"))
don_state = load_json(SYSTEMS["Donchian Daily"]["state"], {}) or {}
if "btc_regime_ok" in don_state:
    top3.metric("BTC regime (Donchian)", "🟢 BTC > EMA50" if don_state["btc_regime_ok"] else "🔴 BTC < EMA50")

# ------------------------------------------------------------
# System status + manual restart
# ------------------------------------------------------------
st.header("🩺 Systems ki halat")
status_cols = st.columns(len(SYSTEMS) + 1)
for col, (name, cfg) in zip(status_cols, SYSTEMS.items()):
    state = load_json(cfg["state"], {}) or {}
    last = state.get("last_updated")
    with col:
        st.markdown(f"**{cfg['badge']}**")
        st.caption(f"Chalta hai: {cfg['every']}")
        if last:
            mins = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last)).total_seconds() / 60
            if mins > cfg["stale_min"]:
                st.error(f"⚠️ Aakhri dafa {mins/60:.1f} ghante pehle chala — shayad ruk gaya")
            else:
                st.success(f"✅ Aakhri dafa {mins/60:.1f} ghante pehle chala")
        else:
            st.warning("Abhi tak pehla run record nahi hua")
        if st.button("▶️ Abhi chala do", key=f"run_{cfg['workflow']}"):
            ok, msg = trigger_github_workflow(cfg["workflow"])
            st.success("✅ GitHub ko command bhej di — 2-3 min baad page refresh karein.") if ok else st.error(f"❌ {msg}")
with status_cols[-1]:
    st.markdown("**🐕 Watchdog**")
    st.caption("Har 30 min: koi bot ruk jaye to khud dobara chalata hai + Telegram")
    if st.button("▶️ Abhi chala do", key="run_watchdog"):
        ok, msg = trigger_github_workflow("watchdog.yml")
        st.success("✅ Bhej diya!") if ok else st.error(f"❌ {msg}")

# ------------------------------------------------------------
# Har system alag
# ------------------------------------------------------------
all_closed = []
tabs = st.tabs([cfg["badge"] for cfg in SYSTEMS.values()])
for tab, (name, cfg) in zip(tabs, SYSTEMS.items()):
    with tab:
        state = load_json(cfg["state"], {}) or {}
        signals = load_json(cfg["signals"], []) or []
        trades = load_csv(cfg["trades"])
        interval = "240" if name == "Ichimoku 4H" else "D"

        with st.expander("📋 Strategy ke usool", expanded=False):
            st.table(pd.DataFrame(cfg["rules"], columns=["", "Usool"]))

        # ---------- 1) Naye signals ----------
        st.subheader("🟢 Naye Signals")
        sig_df = pd.DataFrame(signals)
        if len(sig_df):
            sig_df["_t"] = pd.to_datetime(sig_df["signal_time_utc"], errors="coerce", utc=True)
            hours = st.slider("Kitne ghante purane signals dikhayen", 4, 24 * 14, cfg["new_signal_hours"],
                              key=f"hrs_{name}")
            recent = sig_df[sig_df["_t"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)]
            recent = recent.sort_values("_t", ascending=False)
        else:
            recent = pd.DataFrame()
        if len(recent) == 0:
            st.info("Is waqt koi naya signal nahi. (Signals kam aate hain - kai din bhi lag sakte hain.)")
        else:
            prices = fetch_live_prices(tuple(sorted(recent["symbol"].unique())))
            show = pd.DataFrame({
                "Coin": recent["symbol"],
                "Signal Time (PKT)": recent["signal_time_utc"].map(pkt_str),
                "Entry (approx)": recent["entry_est"].map(fmt_px),
                "Live Price": recent["symbol"].map(lambda s: fmt_px(prices.get(s)) if prices.get(s) else "—"),
                "Live P/L %": [round((prices[s] / e - 1) * 100, 2) if prices.get(s) else None
                               for s, e in zip(recent["symbol"], recent["entry_est"])],
                "SL": recent["sl"].map(fmt_px),
                "TP": recent["tp"].map(lambda v: fmt_px(v) if v is not None and v == v else "Nahi (trailing)"),
                "SL tak %": recent["risk_pct"],
                "Size % (equity)": recent["size_pct"],
                "Chart": recent["symbol"].map(lambda s: tradingview_url(s, interval)),
            })
            show_table(show, pl_cols=("Live P/L %",))
            pick = st.selectbox("Coin chunein (chart / position size / setup copy)", list(recent["symbol"]),
                                key=f"pick_sig_{name}")
            row = recent[recent["symbol"] == pick].iloc[0]
            st.link_button(f"📈 {pick} — TradingView par kholein", tradingview_url(pick, interval))
            position_size_box(pick, row["entry_est"], row["sl"], f"ps_{name}")
            tp_txt = fmt_px(row["tp"]) if row.get("tp") is not None and row["tp"] == row["tp"] else "Nahi (trailing stop)"
            st.code(f"System: {name}\nCoin: {pick}\nSignal: {pkt_str(row['signal_time_utc'])}\n"
                    f"Entry (approx): {fmt_px(row['entry_est'])}\nSL: {fmt_px(row['sl'])}\nTP: {tp_txt}", language=None)

        # ---------- 2) Khuli paper trades ----------
        st.subheader("🔵 Chal Rahi Trades (paper)")
        positions = state.get("positions", {})
        cash = float(state.get("cash", 0) or 0)
        if not positions:
            st.info("Abhi koi khuli trade nahi.")
        else:
            prices = fetch_live_prices(tuple(sorted(positions)))
            rows = []
            for sym, p in positions.items():
                live = prices.get(sym)
                ref = live if live else p.get("last_px")
                rows.append({
                    "Coin": sym,
                    "Entry": fmt_px(p["entry"]),
                    "Live Price": fmt_px(live) if live else "—",
                    "Live P/L %": round((ref / p["entry"] - 1) * 100, 2) if ref else None,
                    "Live P/L ($)": round(p["qty"] * ref - p["cost"], 2) if ref else None,
                    "SL (abhi)": fmt_px(p["trail"]),
                    "SL tak %": round((ref - p["trail"]) / ref * 100, 2) if ref else None,
                    "TP": fmt_px(p["tp"]) if p.get("tp") else "Nahi (trailing)",
                    "Entry waqt": pkt_str(p.get("entry_bar") or p.get("entry_day")),
                    "Halat": ("⚠️ SL se neeche — agle run mein band" if ref and ref <= p["trail"]
                              else ("🟢 Nafa" if ref and ref >= p["entry"] else "🔴 Nuqsan")),
                    "Chart": tradingview_url(sym, interval),
                })
            df_open = pd.DataFrame(rows)
            show_table(df_open, pl_cols=("Live P/L %", "Live P/L ($)"))
            open_val = sum(p["qty"] * (prices.get(s) or p.get("last_px", p["entry"])) for s, p in positions.items())
            st.caption(f"💹 Live Equity: **${cash + open_val:,.2f}** · Cash ${cash:,.2f} · "
                       f"{len(positions)}/10 slots bhare hue · fees shamil nahi")

        # ---------- 3) Performance ----------
        st.subheader("📊 Performance — Band Trades")
        hist = pd.DataFrame(state.get("history", []))
        start_eq = 1000.0
        if len(trades) == 0:
            st.info("Abhi tak koi trade band nahi hui - pehli trade band hone par yahan record banna shuru hoga.")
        else:
            r = pd.to_numeric(trades["ret_pct"], errors="coerce").dropna()
            wins, losses = r[r > 0], r[r <= 0]
            pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else None
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Band trades", len(r))
            c2.metric("Jeet / Haar", f"{len(wins)} / {len(losses)}")
            c3.metric("Win Rate", f"{len(wins)/len(r)*100:.1f}%")
            c4.metric("Profit Factor", f"{pf:.2f}" if pf else "N/A")
            c5.metric("Kul nafa ($)", f"{pd.to_numeric(trades['pnl_usd'], errors='coerce').sum():+,.2f}")
            bt = cfg["backtest"]
            st.info(f"**Backtest:** Win {bt['win']}%, PF {bt['pf']}, CAGR +{bt['cagr']}%, MaxDD {bt['dd']}% — {bt['note']}  \n"
                    f"**Live (paper):** Win {len(wins)/len(r)*100:.1f}%, PF {pf:.2f}" if pf else
                    f"**Backtest:** Win {bt['win']}%, PF {bt['pf']} — {bt['note']}")
            tshow = trades.copy()
            tshow["Entry waqt"] = tshow[cfg["entry_col"]].map(pkt_str)
            tshow["Exit waqt"] = tshow[cfg["exit_col"]].map(pkt_str)
            tshow["Chart"] = tshow["symbol"].map(lambda s: tradingview_url(s, interval))
            cols = ["symbol", "Entry waqt", "Exit waqt"] + [c for c in ("exit_reason",) if c in tshow.columns] + \
                   ["entry", "exit", "ret_pct", "pnl_usd", "Chart"]
            with st.expander(f"📜 Saari band trades ({len(tshow)})"):
                show_table(tshow[cols].iloc[::-1], pl_cols=("ret_pct", "pnl_usd"))
                st.download_button("📥 CSV download", trades.to_csv(index=False).encode(), cfg["trades"],
                                   "text/csv", key=f"dl_{name}")
            t2 = trades[[cfg["entry_col"], "ret_pct"]].rename(columns={cfg["entry_col"]: "entry_utc"})
            t2["System"] = name
            all_closed.append(t2)
        if len(hist) > 1:
            eq = hist.set_index(hist.columns[0])["equity"]
            dd = (eq / eq.cummax() - 1).min() * 100
            st.caption(f"📈 Paper equity curve (shuru ${start_eq:,.0f}) — ab ${eq.iloc[-1]:,.2f} "
                       f"({(eq.iloc[-1]/start_eq-1)*100:+.1f}%), Max Drawdown {dd:.1f}%")
            st.line_chart(eq, height=220)

# ------------------------------------------------------------
# Session analysis (PKT)
# ------------------------------------------------------------
st.markdown("---")
st.header("🕐 Trading Session Analysis (Pakistan Time — PKT)")
st.caption("Har band trade ka ENTRY waqt dekh kar us waqt kaunsa market session khula tha — sab kuch PKT mein. "
           "Note: Donchian Daily ki entry hamesha subah ~5 AM PKT (daily candle ke baad) hoti hai.")


def classify_session(hour_pkt):
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


SESSION_ORDER = [
    "🌏 Asian (05:00 AM–12:00 PM PKT)", "🇬🇧 London (12:00 PM–05:00 PM PKT)",
    "🇬🇧+🇺🇸 London+NY Overlap (05:00 PM–09:00 PM PKT)", "🇺🇸 New York (09:00 PM–02:00 AM PKT)",
    "🌙 Late NY / Off-Hours (02:00 AM–05:00 AM PKT)",
]

if not all_closed:
    st.info("Abhi tak koi band trade nahi — session analysis ke liye pehle kuch trades band honi chahiye.")
else:
    comb = pd.concat(all_closed, ignore_index=True)
    picked = st.multiselect("Kaunse system(s) shamil karein", list(SYSTEMS), default=list(SYSTEMS), key="sess_pick")
    comb = comb[comb["System"].isin(picked)]
    if len(comb):
        comb["hour"] = comb["entry_utc"].map(lambda t: to_pkt(t).hour)
        comb["session"] = comb["hour"].map(classify_session)
        comb["win"] = pd.to_numeric(comb["ret_pct"], errors="coerce") > 0
        rows = []
        for sname in SESSION_ORDER:
            sub = comb[comb["session"] == sname]
            if len(sub):
                w = int(sub["win"].sum())
                rows.append({"Session": sname, "Total Trades": len(sub), "TP/Win": w, "SL/Loss": len(sub) - w,
                             "Win Rate %": round(w / len(sub) * 100, 1),
                             "Avg P/L %": round(pd.to_numeric(sub["ret_pct"], errors="coerce").mean(), 2)})
        if rows:
            stretch_df(pd.DataFrame(rows), hide_index=True)
            best = max(rows, key=lambda r: r["Win Rate %"])
            worst = min(rows, key=lambda r: r["Win Rate %"])
            st.caption(f"✅ Sab se behtar: **{best['Session']}** ({best['Win Rate %']}%, {best['Total Trades']} trades) — "
                       f"⚠️ Sab se kamzor: **{worst['Session']}** ({worst['Win Rate %']}%, {worst['Total Trades']} trades). "
                       f"Chhota sample (~15 se kam trades) abhi bharosemand nahi.")

st.markdown("---")
st.caption("⚠️ Ye paper trading hai. Backtest numbers mein survivorship bias hai (aaj ki coin list) — asal natija kuch kam ho sakta hai.")
