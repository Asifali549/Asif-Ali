"""
DONCHIAN DAILY BOT - portfolio test mein PASS hui strategy ke live signals + paper trading
========================================================================================
Screener se BILKUL ALAG. Rozana ek dafa (daily candle band hone ke baad,
~00:15 UTC = 05:15 PKT) chalta hai.

STRATEGY (portfolio_test.py mein tasdeeq-shuda, 2021-2026):
  Entry  : coin ka daily CLOSE pichle 20 din ke sab se oonche HIGH se ooper band ho
           (fresh breakout), AUR BTC close > BTC EMA50, AUR coin top-100 liquid ho
           -> agle din ke OPEN par khareedo
  Stop   : Chandelier trailing stop = 22 din ka highest high - 4 x ATR(22)
           rozana sirf ooper jata hai (neeche kabhi nahi)
  Exit   : jab qeemat stop ko chhoo le
  Size   : har trade par equity ka 1% risk (entry se stop tak), ek coin max 20%,
           ek waqt mein max 10 positions
  Priority: slots se ziada signals hon to pichle 60 din ka sab se mazboot coin pehle

Har run:
  1) paper positions ke stop check (pichle run ke baad ki har daily candle)
  2) pichle run ke pending signals ko is din ke OPEN par paper-khareed
  3) aaj ke band candle par naye signals
  4) Telegram par khulasa: naye BUY, band hui trades, har khuli position ka
     naya stop (exchange par stop-loss khud update karein), paper equity
State: donchian_paper_state.json | Band trades: donchian_paper_trades.csv
"""

import json
import os
import csv

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, ema, chandelier, STABLES, FEE, SLIP, STOP_SLIP

# ---------------- Settings ----------------
DONCHIAN_N = 20
CE_PERIOD, CE_MULT = 22, 4.0
BTC_EMA = 50
MAX_POSITIONS = 10
RISK_PCT = 0.01
MAX_POS_PCT = 0.20
UNIVERSE = 100
TOP_N_COINS = 150
HISTORY_DAYS = 300
START_EQUITY = 1000.0          # paper account (USDT)
STATE_FILE = "donchian_paper_state.json"
TRADES_CSV = "donchian_paper_trades.csv"
SIGNALS_FILE = "donchian_signals.json"      # dashboard ke liye signals ka record


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"cash": START_EQUITY, "positions": {}, "pending": [], "last_day": None, "history": []}


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)


def log_trade(row):
    new = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def fmt_px(x):
    if x >= 100:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:.4f}"
    return f"{x:.6g}"


def append_signals(new_rows, keep=300):
    """Dashboard ke liye har naye signal ka record (aakhri 300)."""
    rows = []
    if os.path.exists(SIGNALS_FILE):
        try:
            with open(SIGNALS_FILE) as f:
                rows = json.load(f)
        except Exception:
            rows = []
    rows = (rows + new_rows)[-keep:]
    with open(SIGNALS_FILE, "w") as f:
        json.dump(rows, f, indent=2)


TELEGRAM_ON = False   # 2026-10-04: user - Telegram par sirf kaam ke 2 systems (Ichimoku TP5 + Dip v2); ye bot khamosh, record chalta rahe


def send(msg):
    if not TELEGRAM_ON:
        print("[Telegram khamosh]\n" + msg)
        return
    try:
        from telegram_alert import send_telegram_alert
        send_telegram_alert(msg)
    except Exception as e:
        print(f"Telegram fail: {e}")
    print(msg)


def main():
    from data_fetcher import get_exchange, get_coin_list
    ex = get_exchange()
    coins = [c for c in get_coin_list(ex) if c.split("/")[0].upper() not in STABLES][:TOP_N_COINS]
    if "BTC/USDT" not in coins:
        coins.insert(0, "BTC/USDT")

    daily = {}
    for sym in coins:
        try:
            d = fetch_full(ex, sym, "1d", HISTORY_DAYS)
            if d is not None and len(d) >= 30:
                daily[sym] = norm(d)
        except Exception as e:
            print(f"{sym}: SKIP ({e})")
    btc = daily["BTC/USDT"]
    D = btc["timestamp"].iloc[-1]                     # aakhri BAND daily candle
    Ds = str(D.date())

    st = load_state()
    if st["last_day"] == Ds:
        print(f"{Ds} pehle hi process ho chuka - sirf 'aakhri run' ka waqt update.")
        st["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
        save_state(st)
        return

    # ---------- per-coin indicators ----------
    ind = {}
    for sym, d in daily.items():
        d = d.copy()
        d["stop"] = chandelier(d, CE_PERIOD, CE_MULT)
        c, h = d["close"], d["high"]
        brk = c > h.shift(1).rolling(DONCHIAN_N).max()
        d["signal"] = brk & ~brk.shift(1, fill_value=False)
        d["mom60"] = c / c.shift(60) - 1
        d["dvol30"] = (c * d["volume"]).rolling(30, min_periods=20).mean()
        ind[sym] = d.set_index("timestamp")

    last_day = pd.Timestamp(st["last_day"]) if st["last_day"] else D - pd.Timedelta(days=1)
    new_days = [t for t in btc["timestamp"] if last_day < t <= D]
    exits_msg, fills_msg = [], []

    # ---------- 1) pending entries: signal-day ke AGLE din ke open par ----------
    still_pending = []
    for p in st["pending"]:
        sym, sig_day = p["symbol"], pd.Timestamp(p["signal_day"])
        di = ind.get(sym)
        nxt = sig_day + pd.Timedelta(days=1)
        if di is None or nxt not in di.index:
            if nxt > D:
                still_pending.append(p)
            continue
        if sym in st["positions"] or len(st["positions"]) >= MAX_POSITIONS:
            continue
        entry = float(di.at[nxt, "open"]) * (1 + SLIP)
        stop = float(p["stop"])
        if stop >= entry:
            continue
        eq_now = st["cash"] + sum(q["qty"] * q["last_px"] for q in st["positions"].values())
        val = min(eq_now * RISK_PCT * entry / (entry - stop), MAX_POS_PCT * eq_now, st["cash"] / (1 + FEE))
        if val < 5:
            continue
        st["cash"] -= val * (1 + FEE)
        st["positions"][sym] = {"qty": val / entry, "entry": entry, "entry_day": str(nxt.date()),
                                "trail": stop, "init_stop": stop, "last_px": entry, "cost": val * (1 + FEE)}
        fills_msg.append(f"📥 {sym} paper-khareeda @ {fmt_px(entry)} (stop {fmt_px(stop)}, ${val:,.0f})")
    st["pending"] = still_pending

    # ---------- 2) khuli positions: har naye din par stop check, phir trail update ----------
    for day in new_days:
        for sym in list(st["positions"]):
            pos = st["positions"][sym]
            di = ind.get(sym)
            if di is None or day not in di.index or pd.Timestamp(pos["entry_day"]) > day:
                continue
            row = di.loc[day]
            if row["low"] <= pos["trail"]:
                px = min(pos["trail"] * (1 - STOP_SLIP), row["open"]) * (1 - SLIP)
                proceeds = pos["qty"] * px * (1 - FEE)
                st["cash"] += proceeds
                pnl = proceeds - pos["cost"]
                ret = pnl / pos["cost"] * 100
                log_trade({"symbol": sym, "entry_day": pos["entry_day"], "exit_day": str(day.date()),
                           "entry": round(pos["entry"], 8), "exit": round(px, 8),
                           "pnl_usd": round(pnl, 2), "ret_pct": round(ret, 2)})
                exits_msg.append(f"{'✅' if pnl > 0 else '❌'} {sym} STOP par band @ {fmt_px(px)} | {ret:+.1f}% (${pnl:+,.2f})")
                del st["positions"][sym]
                continue
            if not np.isnan(row["stop"]) and row["stop"] > pos["trail"]:
                pos["trail"] = float(row["stop"])
            pos["last_px"] = float(row["close"])

    # ---------- 3) aaj (D) ke naye signals ----------
    btc_i = ind["BTC/USDT"]
    btc_ok = bool(btc_i["close"].iloc[-1] > ema(btc_i["close"], BTC_EMA).iloc[-1])
    liq = sorted([(s, di.at[D, "dvol30"]) for s, di in ind.items()
                  if D in di.index and not np.isnan(di.at[D, "dvol30"]) and len(di) >= 60],
                 key=lambda x: -x[1])[:UNIVERSE]
    liquid = {s for s, _ in liq}
    cands = []
    if btc_ok:
        for s in liquid:
            di = ind[s]
            if s in st["positions"] or not bool(di.at[D, "signal"]):
                continue
            stop = di.at[D, "stop"]
            close = di.at[D, "close"]
            if np.isnan(stop) or stop >= close:
                continue
            cands.append((s, float(close), float(stop), float(di.at[D, "mom60"]) if not np.isnan(di.at[D, "mom60"]) else -9))
    cands.sort(key=lambda x: -x[3])
    slots = MAX_POSITIONS - len(st["positions"])
    chosen = cands[:max(slots, 0)]
    st["pending"] = [{"symbol": s, "signal_day": Ds, "stop": stop} for s, _, stop, _ in chosen]

    # ---------- equity ----------
    inv = sum(p["qty"] * p["last_px"] for p in st["positions"].values())
    equity = st["cash"] + inv
    st["history"].append({"day": Ds, "equity": round(equity, 2)})
    st["last_day"] = Ds
    st["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
    st["btc_regime_ok"] = btc_ok
    save_state(st)
    append_signals([{
        "system": "Donchian Daily", "symbol": s, "signal_time_utc": str(D + pd.Timedelta(days=1)),
        "entry_est": round(close, 10), "sl": round(stop, 10), "tp": None,
        "risk_pct": round((close - stop) / close * 100, 2),
        "size_pct": round(min(RISK_PCT / ((close - stop) / close), MAX_POS_PCT) * 100, 2),
    } for s, close, stop, _ in chosen])

    # ---------- Telegram ----------
    L = [f"📊 <b>Donchian Daily Bot</b> — {Ds} (daily candle band)",
         "📝 <b>Sirf PAPER</b> — asli paisa nahi (Portfolio Lab: Ichimoku ke sath girti hai, hissa 0%)",
         f"BTC regime: {'🟢 BTC > EMA50 (nayi entry allowed)' if btc_ok else '🔴 BTC < EMA50 (nayi entry NAHI)'}"]
    if chosen:
        L.append(f"\n🟢 <b>NAYE BUY SIGNALS</b> ({len(chosen)}) — abhi (agle din ke open par) khareedein:")
        for s, close, stop, m in chosen:
            risk_pct = (close - stop) / close
            size_pct = min(RISK_PCT / risk_pct, MAX_POS_PCT) * 100
            L.append(f"• <b>{s}</b> ~{fmt_px(close)} | Stop: {fmt_px(stop)} ({risk_pct*100:.1f}% neeche) | "
                     f"Size: equity ka {size_pct:.1f}%")
    elif btc_ok:
        L.append("\nAaj koi naya signal nahi.")
    if len(cands) > len(chosen):
        L.append(f"({len(cands) - len(chosen)} aur signals the lekin slots bhare hue)")
    if fills_msg:
        L.append("\n" + "\n".join(fills_msg))
    if exits_msg:
        L.append("\n<b>Band hui trades:</b>\n" + "\n".join(exits_msg))
    if st["positions"]:
        L.append(f"\n<b>Khuli positions ({len(st['positions'])}/{MAX_POSITIONS}) — exchange par stop update karein:</b>")
        for s, p in st["positions"].items():
            chg = (p["last_px"] / p["entry"] - 1) * 100
            L.append(f"• {s}: entry {fmt_px(p['entry'])} | abhi {fmt_px(p['last_px'])} ({chg:+.1f}%) | "
                     f"<b>naya stop {fmt_px(p['trail'])}</b>")
    L.append(f"\n💼 Paper equity: ${equity:,.2f} (shuru ${START_EQUITY:,.0f}, {(equity/START_EQUITY-1)*100:+.1f}%)")
    send("\n".join(L))


if __name__ == "__main__":
    main()
