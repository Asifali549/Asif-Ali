"""
Live Dashboard - 3 hisse (tabs):
  📊 Aaj ka Scoreboard : aaj ke tamam signals + tamam chal rahi trades - kaun nafa mein, kaun nuqsan mein
  🤖 Auto Trading      : bots (Ichimoku 4H, Dip, Donchian, Capitulation) khud paper trades lete/bechte hain
  ✋ Manual Trading    : aap khud jo signal/coin pasand karein us ki trade yahan darj karein (GitHub mein
                         manual_trades.json mein mehfooz - Streamlit secret GITHUB_TOKEN, Contents: Read and write)
Data bots ki files se aata hai (GitHub Actions har run ke baad commit karte hain).

Chalayen: streamlit run live_colorful_dashboard.py
"""

import base64
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
        "alloc": 0.60, "sizing": ("risk", 0.01, 0.20),
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
            ("Size", "Har trade Ichimoku hisse ka 1% risk, max 10 trades, ek coin max 20%"),
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
        "alloc": 0.0, "sizing": ("risk", 0.01, 0.20),
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
            ("Size", "SIRF PAPER — asli paisa nahi (Ichimoku ke sath girti hai; Portfolio Lab mein hissa 0%)"),
        ],
        "backtest": {"win": 37.0, "pf": 1.55, "cagr": 26.0, "dd": -31.0,
                     "note": "6.5 saal (2020-2026, 2022 crash samet), 3 alag tests mein PASS"},
    },
    "Dip Daily": {
        "badge": "🎯 Dip Daily",
        "state": "dip_paper_state.json",
        "trades": "dip_paper_trades.csv",
        "signals": "dip_signals.json",
        "workflow": "dip_daily_bot.yml",
        "alloc": 0.40, "sizing": ("fixed", 0.20, 0.20),
        "stale_min": 1560,
        "every": "rozana 5:20 AM PKT (daily candle band hone ke baad)",
        "entry_col": "entry_day", "exit_col": "exit_day",
        "new_signal_hours": 30,
        "rules": [
            ("Timeframe", "Daily candle"),
            ("Entry", "Mazboot uptrend (close > EMA200, EMA50 > EMA200) mein RSI(3) < 10 — 2-3 din ki tez girawat"),
            ("Market filter", "BTC daily close > BTC EMA50 (warna nayi entry nahi)"),
            ("Stop (SL)", "Signal ke close se 3x ATR(14) neeche — fixed, hilta nahi"),
            ("Exit", "Jis din close 5-din average (SMA5) se ooper band ho, agle din open par becho; max 10 din"),
            ("Exchange par", "Stop-loss order; Telegram 'AAJ OPEN PAR BECHEIN' kahe to market par bech dein"),
            ("Size", "Har trade Dip hisse ka 20% (kul capital ka 8%), max 10 trades"),
        ],
        "backtest": {"win": 69.3, "pf": 2.07, "cagr": 9.7, "dd": -27.7,
                     "note": "6 saal (2020-2026), win-rate ooncha lekin return kam; breakout bots ka ulta (girawat par khareed)"},
    },
    "Volume Capitulation": {
        "badge": "🌊 Capitulation",
        "state": "capit_paper_state.json",
        "trades": "capit_paper_trades.csv",
        "signals": "capit_signals.json",
        "workflow": "capit_daily_bot.yml",
        "alloc": 0.0, "sizing": ("fixed", 0.20, 0.20),
        "stale_min": 1560,
        "every": "rozana 5:25 AM PKT (daily candle band hone ke baad)",
        "entry_col": "entry_day", "exit_col": "exit_day",
        "new_signal_hours": 30,
        "rules": [
            ("Timeframe", "Daily candle"),
            ("Entry", "Uptrend coin (close > EMA200) ek din mein 8%+ gire AUR us din volume pichle 20 din ke ausat ka 2x+ ho (ghabrahat wali farokht)"),
            ("Market filter", "Koi nahi (BTC filter ke baghair test hua)"),
            ("Stop (SL)", "Signal ke close se 3x ATR(14) neeche — fixed, hilta nahi"),
            ("Exit", "Jis din close 5-din average (SMA5) se ooper band ho, agle din open par becho; max 10 din"),
            ("Exchange par", "Stop-loss order; Telegram 'AAJ OPEN PAR BECHEIN' kahe to market par bech dein"),
            ("Size", "SIRF PAPER — asli paisa nahi (dobara test mein 2025-26 ka nateeja kamzor: OOS PF 0.83)"),
        ],
        "backtest": {"win": 63.5, "pf": 1.47, "cagr": 7.4, "dd": -23.2,
                     "note": "6 saal, ~1 trade/hafta; 2021-24 acha lekin 2025-26 kamzor (OOS PF 0.83) - sirf paper par nazar"},
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


def show_dispatch_error(msg):
    if "403" in msg or "not accessible" in msg:
        st.error("❌ GitHub ne mana kar diya (403): Streamlit Secrets wale GITHUB_TOKEN ko "
                 "'Actions' chalane ki ijazat nahi. GitHub → Settings → Developer settings → "
                 "Personal access tokens → apna token → Repository permissions → **Actions: Read and write** "
                 "kar ke Save karein (ya naya token bana kar Streamlit Secrets mein GITHUB_TOKEN badal dein).")
    else:
        st.error(f"❌ {msg}")


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


def position_size_box(coin, entry, sl, key, cfg):
    """Kul capital -> is system ka hissa -> is trade ka size (Portfolio Lab ki taqseem)."""
    try:
        entry, sl = float(entry), float(sl)
    except Exception:
        return
    if not (entry > sl > 0):
        return
    sleeve = total_capital * cfg["alloc"]
    if sleeve <= 0:
        st.warning("📝 Ye system **sirf paper** par hai — asli paisa nahi lagana (hissa 0%).")
        return
    mode, size, cap = cfg["sizing"]
    if mode == "risk":
        value = min(sleeve * size * entry / (entry - sl), sleeve * cap)
    else:
        value = sleeve * size
    units = value / entry
    loss = units * (entry - sl)
    st.info(f"💰 **Position size** — {cfg['badge']} ka hissa ${sleeve:,.0f} (kul ${total_capital:,.0f} ka "
            f"{cfg['alloc']*100:.0f}%): **{units:.6g} {coin.split('/')[0]}** (~${value:,.2f} = kul ka "
            f"{value / total_capital * 100:.1f}%) | SL laga to nuqsan ~${loss:,.2f}")


# ============================================================
# Manual trades (GitHub mein mehfooz)
# ============================================================
MANUAL_FILE = "manual_trades.json"


def gh_token():
    try:
        return st.secrets.get("GITHUB_TOKEN")
    except Exception:
        return None


def manual_load():
    """GitHub se taaza manual trades (sha ke sath); token na ho to local file."""
    tok = gh_token()
    if tok:
        try:
            r = requests.get(f"https://api.github.com/repos/{GITHUB_REPO}/contents/{MANUAL_FILE}?ref={GITHUB_BRANCH}",
                             headers={"Authorization": f"token {tok}", "Accept": "application/vnd.github.v3+json"},
                             timeout=15)
            if r.status_code == 200:
                j = r.json()
                return json.loads(base64.b64decode(j["content"]).decode() or "[]"), j["sha"]
            if r.status_code == 404:
                return [], None
        except Exception:
            pass
    return load_json(MANUAL_FILE, []) or [], None


def manual_save(trades, sha, msg):
    tok = gh_token()
    if not tok:
        return False, "GITHUB_TOKEN Streamlit Secrets mein nahi mila (Contents: Read and write chahiye)."
    body = {"message": msg, "branch": GITHUB_BRANCH,
            "content": base64.b64encode(json.dumps(trades, indent=2).encode()).decode()}
    if sha:
        body["sha"] = sha
    try:
        r = requests.put(f"https://api.github.com/repos/{GITHUB_REPO}/contents/{MANUAL_FILE}", json=body, timeout=20,
                         headers={"Authorization": f"token {tok}", "Accept": "application/vnd.github.v3+json"})
        if r.status_code in (200, 201):
            return True, ""
        if r.status_code == 403:
            return False, "403: token ko 'Contents: Read and write' ki ijazat chahiye."
        if r.status_code == 409:
            return False, "Kisi aur tabdeeli se takraao (409) - page refresh kar ke dobara koshish karein."
        return False, f"HTTP {r.status_code}: {r.text[:150]}"
    except Exception as e:
        return False, str(e)


def status_label(pl, live=None, sl=None, tp=None):
    if live is not None and sl is not None and sl == sl and live <= sl:
        return "⚠️ SL se neeche"
    if live is not None and tp is not None and tp == tp and tp and live >= tp:
        return "🎯 TP par"
    if pl is None or pl != pl:
        return "—"
    return "🟢 Nafa" if pl > 0 else ("🔴 Nuqsan" if pl < 0 else "⚪ Barabar")


# ============================================================
# Page
# ============================================================
st.set_page_config(page_title="Live Dashboard", layout="wide")
st.title("🎯 Live Dashboard")
st.caption("📊 **Scoreboard** = aaj kya hua (sab systems ek jagah) · 🤖 **Auto Trading** = bots khud paper par "
           "khareedte/bechte hain · ✋ **Manual Trading** = aap apni marzi se jo trade lein, wo yahan darj karein.")

st.sidebar.header("💼 Sarmaye ki taqseem")
total_capital = st.sidebar.number_input("Kul Capital ($)", min_value=0.0, value=1000.0, step=100.0)
for _n, _c in SYSTEMS.items():
    _amt = total_capital * _c["alloc"]
    st.sidebar.markdown(f"**{_c['badge']}** — {_c['alloc']*100:.0f}% = **${_amt:,.0f}**"
                        + ("" if _c["alloc"] else " _(sirf paper)_"))
st.sidebar.caption("Backtest (6 saal): Ichimoku 60% + Dip 40% — CAGR ~+25%, MaxDD ~-11%, Sharpe ~1.73. "
                   "Donchian (Ichimoku ke sath girti hai) aur Capitulation (2025-26 kamzor) sirf paper par. "
                   "Har signal ka size khud nikalta hai.")
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
    top3.metric("BTC regime", "🟢 BTC > EMA50" if don_state["btc_regime_ok"] else "🔴 BTC < EMA50")

# ---------- sab systems ka data ek dafa ----------
DATA = {}
for _n, _c in SYSTEMS.items():
    DATA[_n] = dict(state=load_json(_c["state"], {}) or {}, signals=load_json(_c["signals"], []) or [],
                    trades=load_csv(_c["trades"]))
manual, manual_sha = manual_load()
need = set()
for _n, _d in DATA.items():
    need |= set(_d["state"].get("positions", {}))
    for _sg in _d["signals"]:
        need.add(_sg["symbol"])
need |= {m["symbol"] for m in manual if m.get("status") == "open"}
PRICES = fetch_live_prices(tuple(sorted(need)))

T_SCORE, T_AUTO, T_MANUAL, T_SESS = st.tabs(["📊 Aaj ka Scoreboard", "🤖 Auto Trading (bots)",
                                              "✋ Manual Trading (meri trades)", "🕐 Session Analysis"])

# ============================================================
# 📊 SCOREBOARD
# ============================================================
with T_SCORE:
    st.subheader("📊 Aaj ka Scoreboard")
    hrs = st.radio("Signals ka daur", [24, 48, 168], index=0, horizontal=True,
                   format_func=lambda h: {24: "Aakhri 24 ghante", 48: "48 ghante", 168: "7 din"}[h], key="score_hrs")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hrs)
    sig_rows = []
    for _n, _d in DATA.items():
        for _sg in _d["signals"]:
            t = pd.to_datetime(_sg.get("signal_time_utc"), errors="coerce", utc=True)
            if t is pd.NaT or t != t or t < cutoff:
                continue
            live = PRICES.get(_sg["symbol"])
            e = float(_sg["entry_est"])
            pl = round((live / e - 1) * 100, 2) if live else None
            sig_rows.append({"System": SYSTEMS[_n]["badge"] + ("" if SYSTEMS[_n]["alloc"] else " (paper)"),
                             "Coin": _sg["symbol"], "Signal (PKT)": pkt_str(_sg["signal_time_utc"]),
                             "Entry": fmt_px(e), "Live": fmt_px(live) if live else "—", "P/L %": pl,
                             "Halat": status_label(pl, live, _sg.get("sl"), _sg.get("tp")),
                             "SL": fmt_px(_sg.get("sl")),
                             "Chart": tradingview_url(_sg["symbol"], "240" if _n == "Ichimoku 4H" else "D")})
    open_rows = []
    for _n, _d in DATA.items():
        for sym, p in _d["state"].get("positions", {}).items():
            live = PRICES.get(sym)
            ref = live or p.get("last_px")
            pl = round((ref / p["entry"] - 1) * 100, 2) if ref else None
            open_rows.append({"Kis ki": "🤖 " + SYSTEMS[_n]["badge"], "Coin": sym, "Entry": fmt_px(p["entry"]),
                              "Live": fmt_px(live) if live else "—", "P/L %": pl,
                              "P/L $ (paper)": round(p["qty"] * ref - p["cost"], 2) if ref else None,
                              "Halat": status_label(pl, ref, p.get("trail"), p.get("tp")),
                              "SL": fmt_px(p.get("trail")), "Entry waqt": pkt_str(p.get("entry_bar") or p.get("entry_day")),
                              "Chart": tradingview_url(sym, "240" if _n == "Ichimoku 4H" else "D")})
    for m in manual:
        if m.get("status") != "open":
            continue
        live = PRICES.get(m["symbol"])
        pl = round((live / m["entry"] - 1) * 100, 2) if live else None
        open_rows.append({"Kis ki": "✋ Meri (manual)", "Coin": m["symbol"], "Entry": fmt_px(m["entry"]),
                          "Live": fmt_px(live) if live else "—", "P/L %": pl,
                          "P/L $ (paper)": round(m["qty"] * live - m["amount"], 2) if live else None,
                          "Halat": status_label(pl, live, m.get("sl"), m.get("tp")),
                          "SL": fmt_px(m.get("sl")), "Entry waqt": m.get("entry_time", ""),
                          "Chart": tradingview_url(m["symbol"], "D")})

    sg_win = sum(1 for r in sig_rows if (r["P/L %"] or 0) > 0)
    sg_loss = sum(1 for r in sig_rows if (r["P/L %"] or 0) < 0)
    op_win = sum(1 for r in open_rows if (r["P/L %"] or 0) > 0)
    op_loss = sum(1 for r in open_rows if (r["P/L %"] or 0) < 0)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Naye signals", len(sig_rows))
    m2.metric("Signals: nafa / nuqsan", f"🟢 {sg_win} / 🔴 {sg_loss}")
    m3.metric("Chal rahi trades", len(open_rows))
    m4.metric("Trades: nafa / nuqsan", f"🟢 {op_win} / 🔴 {op_loss}")
    tot_pl = sum(r["P/L $ (paper)"] or 0 for r in open_rows)
    m5.metric("Khuli trades ka kul P/L", f"${tot_pl:+,.2f}")

    st.markdown("#### 🟢 Signals (is daur ke) — abhi kahan hain?")
    if sig_rows:
        show_table(pd.DataFrame(sig_rows).sort_values("P/L %", ascending=False, na_position="last"), pl_cols=("P/L %",))
        st.caption("P/L % = signal ki entry qeemat se abhi ki live qeemat tak (agar aap ne signal par khareeda hota).")
    else:
        st.info("Is daur mein koi signal nahi aaya.")
    st.markdown("#### 🔵 Chal rahi trades (bots + meri manual)")
    if open_rows:
        show_table(pd.DataFrame(open_rows).sort_values("P/L %", ascending=False, na_position="last"),
                   pl_cols=("P/L %", "P/L $ (paper)"))
        st.caption("⚠️ SL se neeche = stop toot chuka — bot agle run mein band karega / manual trade aap khud bech dein.")
    else:
        st.info("Abhi koi trade khuli nahi.")

# ============================================================
# 🤖 AUTO TRADING
# ============================================================
with T_AUTO:
    st.info("🤖 **Auto Trading** — yahan ke bots **khud** signal dhoondte, paper par khareedte aur bechte hain "
            "(asli paisa nahi). Har system ki apni tab neeche hai. Aap ko kuch nahi karna — sirf dekhna hai. "
            "Apni asli trade darj karne ke liye ✋ **Manual Trading** tab kholein.")
    st.subheader("🩺 Systems ki halat")
    status_cols = st.columns(len(SYSTEMS) + 1)
    for col, (name, cfg) in zip(status_cols, SYSTEMS.items()):
        state = DATA[name]["state"]
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
                if ok:
                    st.success("✅ GitHub ko command bhej di — 2-3 min baad page refresh karein.")
                else:
                    show_dispatch_error(msg)
    with status_cols[-1]:
        st.markdown("**🐕 Watchdog**")
        st.caption("Har 30 min: koi bot ruk jaye to khud dobara chalata hai + Telegram")
        if st.button("▶️ Abhi chala do", key="run_watchdog"):
            ok, msg = trigger_github_workflow("watchdog.yml")
            if ok:
                st.success("✅ Bhej diya!")
            else:
                show_dispatch_error(msg)

all_closed = []
with T_AUTO:
    st.markdown('---')
    # ------------------------------------------------------------
    # Har system alag
    # ------------------------------------------------------------
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
                position_size_box(pick, row["entry_est"], row["sl"], f"ps_{name}", cfg)
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

# ============================================================
# ✋ MANUAL TRADING
# ============================================================
with T_MANUAL:
    st.info("✋ **Manual Trading** — yahan aap **apni marzi** se trade darj karte hain: kisi bhi system ka signal "
            "chunein ya koi bhi coin likhein. Asli khareed/farokht aap exchange (KuCoin) par khud karein — yahan sirf "
            "hisaab rakha jata hai (live P/L, SL/TP alert, band trades ka record). Data GitHub mein `manual_trades.json` "
            "mein mehfooz hota hai.")
    if not gh_token():
        st.warning("Streamlit Secrets mein GITHUB_TOKEN nahi mila — trades mehfooz nahi hongi.")

    # ---------- 1) nayi trade ----------
    st.subheader("➕ Nayi trade darj karein")
    recent_sigs = []
    for _n, _d in DATA.items():
        for _sg in _d["signals"]:
            t = pd.to_datetime(_sg.get("signal_time_utc"), errors="coerce", utc=True)
            if t == t and t >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3):
                recent_sigs.append((_n, _sg))
    labels = ["✍️ Apni marzi (koi bhi coin)"] + [
        f"{SYSTEMS[n]['badge']} — {sg['symbol']} ({pkt_str(sg['signal_time_utc'])})" for n, sg in recent_sigs]
    choice = st.selectbox("Signal chunein (aakhri 3 din) ya apni marzi", range(len(labels)),
                          format_func=lambda k: labels[k], key="man_pick")
    if choice == 0:
        src, sym0, sl0, tp0 = "Apni marzi", "", 0.0, 0.0
    else:
        n0, sg0 = recent_sigs[choice - 1]
        src, sym0 = SYSTEMS[n0]["badge"], sg0["symbol"]
        sl0 = float(sg0.get("sl") or 0)
        tp0 = float(sg0.get("tp") or 0) if sg0.get("tp") == sg0.get("tp") and sg0.get("tp") else 0.0
    with st.form("man_new", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        sym = c1.text_input("Coin (jaise SOL/USDT)", value=sym0).strip().upper()
        if sym and "/" not in sym:
            sym = sym + "/USDT"
        live0 = PRICES.get(sym) or (fetch_live_prices((sym,)).get(sym) if sym else None)
        entry = c2.number_input("Entry qeemat", min_value=0.0, value=float(live0 or 0.0), format="%.8f")
        amount = c3.number_input("Kitne $ lagaye", min_value=0.0, value=50.0, step=10.0)
        c4, c5, c6 = st.columns(3)
        sl = c4.number_input("Stop-loss (0 = nahi)", min_value=0.0, value=sl0, format="%.8f")
        tp = c5.number_input("Take-profit (0 = nahi)", min_value=0.0, value=tp0, format="%.8f")
        note = c6.text_input("Note (ikhtiyari)", value="")
        if entry and sl and sl < entry and amount:
            st.caption(f"SL laga to nuqsan ~${amount * (entry - sl) / entry:,.2f} ({(entry - sl) / entry * 100:.1f}%)")
        submitted = st.form_submit_button("✅ Trade darj karein")
    if submitted:
        if not sym or entry <= 0 or amount <= 0:
            st.error("Coin, entry qeemat aur raqam zaroori hain.")
        else:
            rec = {"id": pd.Timestamp.now(tz="UTC").strftime("%Y%m%d%H%M%S"), "source": src, "symbol": sym,
                   "entry": float(entry), "amount": float(amount), "qty": float(amount) / float(entry),
                   "sl": float(sl) or None, "tp": float(tp) or None, "note": note, "status": "open",
                   "entry_time": pd.Timestamp.now(tz="Asia/Karachi").strftime("%Y-%m-%d %I:%M %p PKT")}
            ok, msg = manual_save(manual + [rec], manual_sha, f"Manual trade: {sym}")
            if ok:
                st.success(f"✅ {sym} darj ho gaya. (Page khud taaza ho raha hai)")
                st.rerun()
            else:
                st.error(f"❌ {msg}")

    # ---------- 2) khuli manual trades ----------
    st.subheader("🔵 Meri khuli trades")
    open_m = [m for m in manual if m.get("status") == "open"]
    if not open_m:
        st.info("Abhi koi manual trade khuli nahi.")
    else:
        rows = []
        for m in open_m:
            live = PRICES.get(m["symbol"])
            pl = round((live / m["entry"] - 1) * 100, 2) if live else None
            rows.append({"Coin": m["symbol"], "Source": m.get("source", ""), "Entry": fmt_px(m["entry"]),
                         "Live": fmt_px(live) if live else "—", "P/L %": pl,
                         "P/L $": round(m["qty"] * live - m["amount"], 2) if live else None,
                         "Raqam $": m["amount"], "SL": fmt_px(m.get("sl")), "TP": fmt_px(m.get("tp")),
                         "Halat": status_label(pl, live, m.get("sl"), m.get("tp")),
                         "Entry waqt": m.get("entry_time", ""), "Chart": tradingview_url(m["symbol"], "D")})
        show_table(pd.DataFrame(rows), pl_cols=("P/L %", "P/L $"))
        hit = [r for r in rows if r["Halat"] in ("⚠️ SL se neeche", "🎯 TP par")]
        for r in hit:
            st.warning(f"{r['Halat']}: **{r['Coin']}** — exchange par check kar ke band karein.")
        st.markdown("**Trade band karein**")
        k1, k2, k3 = st.columns([2, 2, 1])
        idx = k1.selectbox("Kaun si", range(len(open_m)), key="man_close_pick",
                           format_func=lambda k: f"{open_m[k]['symbol']} @ {fmt_px(open_m[k]['entry'])} ({open_m[k].get('entry_time','')})")
        live_c = PRICES.get(open_m[idx]["symbol"]) or open_m[idx]["entry"]
        exit_px = k2.number_input("Exit qeemat", min_value=0.0, value=float(live_c), format="%.8f", key="man_exit")
        if k3.button("🔒 Band karein", key="man_close_btn"):
            new = []
            for m in manual:
                if m["id"] == open_m[idx]["id"]:
                    m = dict(m, status="closed", exit=float(exit_px),
                             exit_time=pd.Timestamp.now(tz="Asia/Karachi").strftime("%Y-%m-%d %I:%M %p PKT"),
                             pnl=round(m["qty"] * float(exit_px) - m["amount"], 2),
                             ret_pct=round((float(exit_px) / m["entry"] - 1) * 100, 2))
                new.append(m)
            ok, msg = manual_save(new, manual_sha, f"Manual trade band: {open_m[idx]['symbol']}")
            if ok:
                st.success("✅ Band ho gayi.")
                st.rerun()
            else:
                st.error(f"❌ {msg}")

    # ---------- 3) band manual trades ----------
    st.subheader("📊 Meri band trades — performance")
    closed_m = [m for m in manual if m.get("status") == "closed"]
    if not closed_m:
        st.info("Abhi koi manual trade band nahi hui.")
    else:
        cdf = pd.DataFrame(closed_m)
        r = pd.to_numeric(cdf["ret_pct"], errors="coerce")
        w, l = r[r > 0], r[r <= 0]
        pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else None
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Band trades", len(r))
        a2.metric("Jeet / Haar", f"{len(w)} / {len(l)}")
        a3.metric("Win rate", f"{len(w) / len(r) * 100:.1f}%")
        a4.metric("Kul nafa ($)", f"{pd.to_numeric(cdf['pnl'], errors='coerce').sum():+,.2f}")
        st.caption(f"Profit factor: {pf:.2f}" if pf else "Profit factor: —")
        show_table(cdf[["symbol", "source", "entry_time", "exit_time", "entry", "exit", "ret_pct", "pnl", "note"]]
                   .iloc[::-1].rename(columns={"symbol": "Coin", "source": "Source", "entry_time": "Entry waqt",
                                               "exit_time": "Exit waqt", "entry": "Entry", "exit": "Exit",
                                               "ret_pct": "P/L %", "pnl": "P/L $", "note": "Note"}),
                   pl_cols=("P/L %", "P/L $"), link_cols=())

with T_SESS:
    # ------------------------------------------------------------
    # Session analysis (PKT)
    # ------------------------------------------------------------
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
