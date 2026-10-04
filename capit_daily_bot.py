"""
VOLUME CAPITULATION BOT - uptrend coin mein ek din ki ghabrahat wali farokht khareedo (paper trading + Telegram)
==============================================================================================================
Rozana ek dafa (daily candle band hone ke baad, 00:25 UTC = 05:25 PKT) chalta hai.

STRATEGY (new_ideas.py + capit_validate.py mein tasdeeq-shuda, 2020-2026):
  Entry : coin ka daily close > EMA200 (uptrend)
          AUR us din ka return <= -8% (tez girawat)
          AUR us din ka volume >= 2 x pichle 20 din ka ausat volume (ghabrahat wali farokht)
          AUR coin top-100 liquid  -> agle din ke OPEN par khareedo. Koi BTC filter NAHI.
  Exit  : jis din daily CLOSE apni 5-din average (SMA5) se ooper band ho -> AGLE din ke open par becho
  Stop  : signal din ke close se 3 x ATR(14) neeche (fixed)
  Time  : 10 din baad bhi na nikla ho to us din ke close par becho
  Size  : SIRF PAPER (ALLOC 0) - paper mein har trade 20%, max 10 positions
Backtest: 313 trades (~1/hafta), win 65%, PF 1.87, OOS PF 1.59, bootstrap p5 1.40, 4/4 folds musbat,
3x kharche par PF 1.66. ICHI/DIP se correlation ~0. ICHI 60 / DIP 30 / CAPIT 10 -> Sharpe 1.81 (pehle 1.71).

State: capit_paper_state.json | Band trades: capit_paper_trades.csv | Signals: capit_signals.json
"""
import csv
import json
import os

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, ema, STABLES, FEE, SLIP, STOP_SLIP

# ---------------- Settings ----------------
DROP = 0.08                    # din ka return <= -8%
VOL_MULT = 2.0                 # volume >= 2 x pichle 20 din ka ausat
EXIT_SMA = 5
STOP_ATR = 3.0
MAX_HOLD = 10
MAX_POSITIONS = 10
POS_PCT = 0.20                 # ziada se ziada 20% ek trade mein
RISK_PCT = 0.02                # 2026-10-04 stop_fix_lab: size = 2% risk / SL doori (cap 20%) - ek trade max -2% (pehle -12.6%)
ALLOC = 0.0                    # SIRF PAPER (2026-10-02: dobara test mein OOS 2025+ PF 0.83 - asli paisa nahi)
UNIVERSE = 100
TOP_N_COINS = 150
HISTORY_DAYS = 1000              # EMA200 poora pakne ke liye lambi history (backtest jaisa)
START_EQUITY = 1000.0
STATE_FILE = "capit_paper_state.json"
TRADES_CSV = "capit_paper_trades.csv"
SIGNALS_FILE = "capit_signals.json"


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


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


def append_signals(new_rows, keep=300):
    rows = []
    if os.path.exists(SIGNALS_FILE):
        try:
            with open(SIGNALS_FILE) as f:
                rows = json.load(f)
        except Exception:
            rows = []
    with open(SIGNALS_FILE, "w") as f:
        json.dump((rows + new_rows)[-keep:], f, indent=2)


def fmt_px(x):
    return f"{x:,.2f}" if x >= 100 else (f"{x:.4f}" if x >= 1 else f"{x:.6g}")


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


def close_pos(st, sym, px, day, why, msgs):
    pos = st["positions"].pop(sym)
    st.setdefault("closed_on", {})[sym] = str(day.date())      # usi din dobara entry nahi (backtest jaisa)
    proceeds = pos["qty"] * px * (1 - FEE)
    st["cash"] += proceeds
    pnl = proceeds - pos["cost"]
    ret = pnl / pos["cost"] * 100
    log_trade({"symbol": sym, "entry_day": pos["entry_day"], "exit_day": str(day.date()),
               "entry": round(pos["entry"], 8), "exit": round(px, 8),
               "pnl_usd": round(pnl, 2), "ret_pct": round(ret, 2), "reason": why})
    msgs.append(f"{'✅' if pnl > 0 else '❌'} {sym} band ({why}) @ {fmt_px(px)} | {ret:+.1f}% (${pnl:+,.2f})")


def main():
    from data_fetcher import get_exchange, get_coin_list
    ex = get_exchange()
    coins = [c for c in get_coin_list(ex) if c.split("/")[0].upper() not in STABLES][:TOP_N_COINS]
    if "BTC/USDT" not in coins:
        coins.insert(0, "BTC/USDT")

    ind = {}
    for sym in coins:
        try:
            d = fetch_full(ex, sym, "1d", HISTORY_DAYS)
            if d is None or len(d) < 30:
                continue
            d = norm(d)
            c = d["close"]
            d["uptrend"] = (c > ema(c, 200)) & (np.arange(len(d)) >= 200)
            d["ret"] = c.pct_change()
            d["vrat"] = d["volume"] / d["volume"].shift(1).rolling(20).mean()
            d["stop"] = c - STOP_ATR * atr(d)
            d["exit_sig"] = c > c.rolling(EXIT_SMA).mean()
            d["dvol30"] = (c * d["volume"]).rolling(30, min_periods=20).mean()
            ind[sym] = d.set_index("timestamp")
        except Exception as e:
            print(f"{sym}: SKIP ({e})")
    btc = ind["BTC/USDT"]
    D = btc.index[-1]                               # aakhri BAND daily candle
    Ds = str(D.date())

    st = load_state()
    if st["last_day"] == Ds:
        print(f"{Ds} pehle hi process ho chuka - sirf 'aakhri run' ka waqt update.")
        st["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
        save_state(st)
        return

    last_day = pd.Timestamp(st["last_day"]) if st["last_day"] else D - pd.Timedelta(days=1)
    new_days = [t for t in btc.index if last_day < t <= D]
    exits, fills = [], []

    for day in new_days:
        # 1) kal "SMA5 se ooper band" hua tha -> aaj ke OPEN par becho
        for sym in list(st["positions"]):
            pos, di = st["positions"][sym], ind.get(sym)
            if pos.get("exit_next") and di is not None and day in di.index:
                close_pos(st, sym, float(di.at[day, "open"]) * (1 - SLIP), day, "uchhal (SMA5)", exits)
        # 2) kal ke signals -> aaj ke OPEN par khareedo
        keep = []
        for p in st["pending"]:
            sym, nxt = p["symbol"], pd.Timestamp(p["signal_day"]) + pd.Timedelta(days=1)
            if nxt != day:
                if nxt > day:
                    keep.append(p)
                continue
            di = ind.get(sym)
            if di is None or day not in di.index or sym in st["positions"] or len(st["positions"]) >= MAX_POSITIONS:
                continue
            entry = float(di.at[day, "open"]) * (1 + SLIP)
            if p["stop"] >= entry:
                continue
            eq_now = st["cash"] + sum(q["qty"] * q["last_px"] for q in st["positions"].values())
            val = min(eq_now * RISK_PCT * entry / (entry - p["stop"]), POS_PCT * eq_now, st["cash"] / (1 + FEE))
            if val < 5:
                continue
            st["cash"] -= val * (1 + FEE)
            st["positions"][sym] = {"qty": val / entry, "entry": entry, "entry_day": str(day.date()),
                                    "trail": float(p["stop"]), "init_stop": float(p["stop"]), "tp": None,
                                    "last_px": entry, "cost": val * (1 + FEE), "bars": 0, "exit_next": False}
            fills.append(f"📥 {sym} paper-khareeda @ {fmt_px(entry)} (stop {fmt_px(p['stop'])}, ${val:,.0f})")
        st["pending"] = keep
        # 3) har khuli position: stop -> uchhal signal -> time
        for sym in list(st["positions"]):
            pos, di = st["positions"][sym], ind.get(sym)
            if di is None or day not in di.index or pd.Timestamp(pos["entry_day"]) > day or pos.get("exit_next"):
                continue
            row = di.loc[day]
            if row["low"] <= pos["trail"]:
                close_pos(st, sym, min(pos["trail"] * (1 - STOP_SLIP), row["open"]) * (1 - SLIP), day, "STOP", exits)
                continue
            pos["last_px"] = float(row["close"])
            if bool(row["exit_sig"]):
                pos["exit_next"] = True
            elif pos["bars"] >= MAX_HOLD:
                close_pos(st, sym, float(row["close"]) * (1 - SLIP), day, f"{MAX_HOLD} din", exits)
                continue
            pos["bars"] += 1

    # ---------- aaj (D) ke naye signals ----------
    liq = sorted([(s, di.at[D, "dvol30"]) for s, di in ind.items()
                  if D in di.index and not np.isnan(di.at[D, "dvol30"]) and len(di) >= 60],
                 key=lambda x: -x[1])[:UNIVERSE]
    cands = []
    for s, _ in liq:
        if s == "BTC/USDT":
            continue
        r = ind[s].loc[D]
        if s in st["positions"] or st.get("closed_on", {}).get(s) == Ds:
            continue
        if not bool(r["uptrend"]) or not (r["ret"] <= -DROP) or not (r["vrat"] >= VOL_MULT):
            continue
        if np.isnan(r["stop"]) or r["stop"] >= r["close"]:
            continue
        cands.append((s, float(r["close"]), float(r["stop"]), float(r["ret"]), float(r["vrat"])))
    cands.sort(key=lambda x: x[3])                   # sab se gehri girawat pehle
    chosen = cands[:max(MAX_POSITIONS - len(st["positions"]), 0)]
    st["pending"] = [{"symbol": s, "signal_day": Ds, "stop": stop} for s, _, stop, _, _ in chosen]

    equity = st["cash"] + sum(p["qty"] * p["last_px"] for p in st["positions"].values())
    st["history"].append({"day": Ds, "equity": round(equity, 2)})
    st["last_day"] = Ds
    st["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
    save_state(st)
    append_signals([{"system": "Volume Capitulation", "symbol": s, "signal_time_utc": str(D + pd.Timedelta(days=1)),
                     "entry_est": round(c, 10), "sl": round(stop, 10), "tp": None,
                     "risk_pct": round((c - stop) / c * 100, 2), "size_pct": round(min(RISK_PCT / ((c - stop) / c), POS_PCT) * 100, 2)}
                    for s, c, stop, _, _ in chosen])

    # ---------- Telegram ----------
    L = [f"🌊 <b>Volume Capitulation Bot</b> — {Ds} (daily candle band)"]
    sell_now = [s for s, p in st["positions"].items() if p.get("exit_next")]
    if sell_now:
        L.append("\n🔔 <b>AAJ OPEN PAR BECHEIN</b> (close 5-din average se ooper band hua):")
        L += [f"• <b>{s}</b> (entry {fmt_px(st['positions'][s]['entry'])}, abhi {fmt_px(st['positions'][s]['last_px'])})"
              for s in sell_now]
    if chosen:
        L.append(f"\n🟢 <b>NAYE BUY SIGNALS</b> ({len(chosen)}) — aaj open par khareedein:")
        L.append("⚠️ Sirf PAPER — asli paisa nahi (haaliya test mein 2025-26 kamzor)" if ALLOC == 0 else
                 f"Size: {RISK_PCT*100:.0f}% risk (SL doori ke hisab se, max {POS_PCT*100:.0f}%)")
        L += [f"• <b>{s}</b> ~{fmt_px(c)} | Stop: {fmt_px(stop)} ({(c - stop) / c * 100:.1f}% neeche) | "
              f"girawat {r*100:.1f}%, volume {v:.1f}x" for s, c, stop, r, v in chosen]
        L.append("Becho: jis din close 5-din average se ooper band ho, agle din open par (max 10 din).")
    else:
        L.append("\nAaj koi naya signal nahi.")
    if fills:
        L.append("\n" + "\n".join(fills))
    if exits:
        L.append("\n<b>Band hui trades:</b>\n" + "\n".join(exits))
    hold = [s for s in st["positions"] if s not in sell_now]
    if hold:
        L.append(f"\n<b>Khuli positions ({len(st['positions'])}/{MAX_POSITIONS}):</b>")
        for s in hold:
            p = st["positions"][s]
            L.append(f"• {s}: entry {fmt_px(p['entry'])} | abhi {fmt_px(p['last_px'])} "
                     f"({(p['last_px'] / p['entry'] - 1) * 100:+.1f}%) | stop {fmt_px(p['trail'])} | din {p['bars']}/{MAX_HOLD}")
    L.append(f"\n💼 Paper equity: ${equity:,.2f} (shuru ${START_EQUITY:,.0f}, {(equity / START_EQUITY - 1) * 100:+.1f}%)")
    if chosen or exits or fills or sell_now:
        send("\n".join(L))
    else:
        print("\n".join(L))
        print("Koi naya waqia nahi - Telegram nahi bheja.")


if __name__ == "__main__":
    main()
