"""
ICHIMOKU 4H BOT - validation mein PASS hui strategy ke live signals + paper trading
==================================================================================
Screener se BILKUL ALAG. Har 4 ghante (4H candle band hone ke 10 min baad) chalta hai.

STRATEGY (ichimoku4h_validation.py + ichimoku4h_exits.py mein tasdeeq-shuda, 2021-2026):
  Entry : EK HI 4H candle par dono signal:
          - Ichimoku: Tenkan(9) > Kijun(26), close cloud ke ooper, close > EMA200,
            EMA50 > EMA200, volume > 2x (20-candle avg) - pehli dafa ye sab sach
          - Market Structure: close pichle swing high (5/5 pivot, swing >= 1.5%) ke
            ooper band, structure bullish (higher highs), EMA200/EMA50 trend
          - Cooldown: ek coin par 15 candles tak naya signal nahi
          - coin top-100 liquid ho (30 din ka dollar volume)
          -> agli 4H candle ke OPEN par khareedo
  Stop  : Chandelier trailing = 16 candles ka highest high - 5.5 x ATR(16), sirf ooper
  TP    : Entry + 3 x (Entry - shuru ka Stop)   [3R]
  Exit  : jo pehle lage - stop ya TP
  Size  : har trade 1% risk, ek coin max 20% equity, max 10 positions
  Priority: slots se ziada signals hon to pichle 60 din ka sab se mazboot coin pehle
Backtest (5.3 saal, sakht usool): PF 1.90, CAGR +26.6%, MaxDD -11%, Sharpe 1.54

State: ichimoku4h_paper_state.json | Band trades: ichimoku4h_paper_trades.csv
"""

import csv
import json
import os

import numpy as np
import pandas as pd

from strategy_lab import fetch_full, norm, chandelier, STABLES, FEE, SLIP, STOP_SLIP
from ichimoku4h_validation import signal as ichi_signal, BASE

CE_P, CE_M, TP_R = 16, 5.5, 3.0
PARAMS = dict(BASE, ce_p=CE_P, ce_m=CE_M, tp=TP_R)
MAX_POSITIONS = 10
RISK_PCT = 0.01
MAX_POS_PCT = 0.20
UNIVERSE = 100
TOP_N_COINS = 150
HISTORY_BARS = 700            # EMA200 + Ichimoku + 60 din momentum ke liye kaafi
BPD = 6
MAX_HOLD = 500
START_EQUITY = 1000.0
STATE_FILE = "ichimoku4h_paper_state.json"
TRADES_CSV = "ichimoku4h_paper_trades.csv"
SIGNALS_FILE = "ichimoku4h_signals.json"    # dashboard ke liye signals ka record


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"cash": START_EQUITY, "positions": {}, "pending": [], "last_bar": None, "history": []}


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


def pkt(ts):
    return (pd.Timestamp(ts) + pd.Timedelta(hours=5)).strftime("%d %b %H:%M PKT")


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


def send(msg):
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

    ind = {}
    for sym in coins:
        try:
            d = fetch_full(ex, sym, "4h", HISTORY_BARS)
            if d is None or len(d) < 260:
                continue
            d = norm(d)
            d["stop"] = chandelier(d, CE_P, CE_M)
            d["signal"] = ichi_signal(d, PARAMS)
            d["mom60"] = d["close"] / d["close"].shift(60 * BPD) - 1
            d["dvol30"] = (d["close"] * d["volume"]).rolling(30 * BPD, min_periods=20 * BPD).mean()
            ind[sym] = d.set_index("timestamp")
        except Exception as e:
            print(f"{sym}: SKIP ({e})")
    if not ind:
        raise SystemExit("Koi data nahi mila")

    D = max(di.index[-1] for di in ind.values())          # aakhri BAND 4H candle
    Ds = str(D)
    st = load_state()
    if st["last_bar"] == Ds:
        print(f"{Ds} pehle hi process ho chuka - sirf 'aakhri run' ka waqt update.")
        st["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
        save_state(st)
        return
    last_bar = pd.Timestamp(st["last_bar"]) if st["last_bar"] else D - pd.Timedelta(hours=4)
    fills_msg, exits_msg = [], []

    # ---------- 1) pending: signal candle ki AGLI candle ke open par paper-khareed ----------
    for p in st["pending"]:
        sym, sig_bar = p["symbol"], pd.Timestamp(p["signal_bar"])
        di = ind.get(sym)
        nxt = sig_bar + pd.Timedelta(hours=4)
        if di is None or nxt not in di.index or sym in st["positions"] or len(st["positions"]) >= MAX_POSITIONS:
            continue
        entry = float(di.at[nxt, "open"]) * (1 + SLIP)
        stop = float(p["stop"])
        if stop >= entry:
            continue
        eq_now = st["cash"] + sum(q["qty"] * q["last_px"] for q in st["positions"].values())
        val = min(eq_now * RISK_PCT * entry / (entry - stop), MAX_POS_PCT * eq_now, st["cash"] / (1 + FEE))
        if val < 5:
            continue
        tp = entry + TP_R * (entry - stop)
        st["cash"] -= val * (1 + FEE)
        st["positions"][sym] = {"qty": val / entry, "entry": entry, "entry_bar": str(nxt), "trail": stop,
                                "init_stop": stop, "tp": tp, "last_px": entry, "cost": val * (1 + FEE)}
        fills_msg.append(f"📥 {sym} paper-khareeda @ {fmt_px(entry)} | SL {fmt_px(stop)} | TP {fmt_px(tp)} (${val:,.0f})")
    st["pending"] = []

    # ---------- 2) khuli positions: har nayi candle par stop, phir TP, phir trail update ----------
    all_bars = sorted({t for di in ind.values() for t in di.index if last_bar < t <= D})
    moved = []
    for bar in all_bars:
        for sym in list(st["positions"]):
            pos = st["positions"][sym]
            di = ind.get(sym)
            if di is None or bar not in di.index or pd.Timestamp(pos["entry_bar"]) > bar:
                continue
            row = di.loc[bar]
            px, why = None, None
            if row["low"] <= pos["trail"]:
                px, why = min(pos["trail"] * (1 - STOP_SLIP), row["open"]), "STOP"
            elif row["high"] >= pos["tp"]:
                px, why = max(pos["tp"], row["open"]), "TP"
            elif (bar - pd.Timestamp(pos["entry_bar"])) / pd.Timedelta(hours=4) >= MAX_HOLD:
                px, why = row["close"], "MAX-HOLD"
            if px is not None:
                px *= (1 - SLIP)
                proceeds = pos["qty"] * px * (1 - FEE)
                st["cash"] += proceeds
                pnl = proceeds - pos["cost"]
                ret = pnl / pos["cost"] * 100
                log_trade({"symbol": sym, "entry_bar": pos["entry_bar"], "exit_bar": str(bar), "exit_reason": why,
                           "entry": round(pos["entry"], 8), "exit": round(px, 8),
                           "pnl_usd": round(pnl, 2), "ret_pct": round(ret, 2)})
                exits_msg.append(f"{'✅' if pnl > 0 else '❌'} {sym} {why} par band @ {fmt_px(px)} | {ret:+.1f}% (${pnl:+,.2f})")
                del st["positions"][sym]
                continue
            s = row["stop"]
            if not np.isnan(s) and s > pos["trail"]:
                pos["trail"] = float(s)
                if sym not in moved:
                    moved.append(sym)
            pos["last_px"] = float(row["close"])

    # ---------- 3) is candle (D) ke naye signals ----------
    liq = sorted([(s, di.at[D, "dvol30"]) for s, di in ind.items()
                  if D in di.index and not np.isnan(di.at[D, "dvol30"])], key=lambda x: -x[1])[:UNIVERSE]
    cands = []
    for s, _ in liq:
        di = ind[s]
        if s in st["positions"] or not bool(di.at[D, "signal"]):
            continue
        stop, close = di.at[D, "stop"], di.at[D, "close"]
        if np.isnan(stop) or stop >= close:
            continue
        m = di.at[D, "mom60"]
        cands.append((s, float(close), float(stop), float(m) if not np.isnan(m) else -9))
    cands.sort(key=lambda x: -x[3])
    chosen = cands[:max(MAX_POSITIONS - len(st["positions"]), 0)]
    st["pending"] = [{"symbol": s, "signal_bar": Ds, "stop": stop} for s, _, stop, _ in chosen]

    inv = sum(p["qty"] * p["last_px"] for p in st["positions"].values())
    equity = st["cash"] + inv
    st["history"].append({"bar": Ds, "equity": round(equity, 2)})
    st["history"] = st["history"][-3000:]
    st["last_bar"] = Ds
    st["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
    save_state(st)
    append_signals([{
        "system": "Ichimoku 4H", "symbol": s, "signal_time_utc": str(D + pd.Timedelta(hours=4)),
        "entry_est": round(close, 10), "sl": round(stop, 10), "tp": round(close + TP_R * (close - stop), 10),
        "risk_pct": round((close - stop) / close * 100, 2),
        "size_pct": round(min(RISK_PCT / ((close - stop) / close), MAX_POS_PCT) * 100, 2),
    } for s, close, stop, _ in chosen])

    # ---------- Telegram: sirf jab kuch hua ho, ya din ki pehli candle (khulasa) ----------
    daily_summary = D.hour == 20        # 20:00 UTC wali candle = din ka aakhri (01:00 PKT ke baad)
    if not (chosen or fills_msg or exits_msg or moved or daily_summary):
        print("Koi naya waqia nahi - Telegram nahi bheja.")
        return
    L = [f"📈 <b>Ichimoku 4H Bot</b> — candle band: {pkt(D + pd.Timedelta(hours=4))}"]
    if chosen:
        L.append(f"\n🟢 <b>NAYE BUY SIGNALS</b> ({len(chosen)}) — abhi khareedein:")
        for s, close, stop, _ in chosen:
            risk = (close - stop) / close
            tp = close + TP_R * (close - stop)
            size = min(RISK_PCT / risk, MAX_POS_PCT) * 100
            L.append(f"• <b>{s}</b> ~{fmt_px(close)}\n   SL: {fmt_px(stop)} ({risk*100:.1f}% neeche) | "
                     f"TP: {fmt_px(tp)} (+{risk*TP_R*100:.1f}%) | Size: equity ka {size:.1f}%")
        L.append("   (OCO order: upar TP, neeche SL)")
    if len(cands) > len(chosen):
        L.append(f"({len(cands) - len(chosen)} aur signals the, slots bhare hue)")
    if fills_msg:
        L.append("\n" + "\n".join(fills_msg))
    if exits_msg:
        L.append("\n<b>Band hui trades:</b>\n" + "\n".join(exits_msg))
    if st["positions"] and (moved or daily_summary or exits_msg or fills_msg):
        L.append(f"\n<b>Khuli positions ({len(st['positions'])}/{MAX_POSITIONS}):</b>")
        for s, p in st["positions"].items():
            chg = (p["last_px"] / p["entry"] - 1) * 100
            flag = " 🔼 <b>SL ooper karein</b>" if s in moved else ""
            L.append(f"• {s}: entry {fmt_px(p['entry'])} | abhi {fmt_px(p['last_px'])} ({chg:+.1f}%) | "
                     f"SL {fmt_px(p['trail'])} | TP {fmt_px(p['tp'])}{flag}")
    L.append(f"\n💼 Paper equity: ${equity:,.2f} ({(equity/START_EQUITY-1)*100:+.1f}%)")
    send("\n".join(L))


if __name__ == "__main__":
    main()
