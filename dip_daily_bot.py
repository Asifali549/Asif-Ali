"""
DIP DAILY BOT - uptrend mein tez girawat khareedo, uchhal par becho (paper trading + Telegram)
=============================================================================================
Rozana ek dafa (daily candle band hone ke baad, 00:20 UTC = 05:20 PKT) chalta hai.

v2 (2026-10-04 se, dip_v2_validate.py mein PASS - win-rate version). v1 (RSI<10, SMA5, TP nahi) ka
record dip_v1_paper_state.json / dip_v1_paper_trades.csv / dip_v1_signals.json mein mehfooz.
  Entry : coin ka daily close > EMA200 AUR EMA50 > EMA200 (mazboot uptrend)
          AUR BTC daily close > BTC EMA50  AUR  RSI(3) < 7 (2-3 din ki gehri girawat)
          AUR coin top-100 liquid  -> agle din ke OPEN par khareedo
  TP    : entry se +5% par foran becho
  Exit  : jis din daily CLOSE apni 3-din average (SMA3) se ooper band ho -> AGLE din ke open par becho
  Stop  : entry signal ke close se 3 x ATR(14) neeche (fixed, hilta nahi)
  Time  : 10 din baad bhi na nikla ho to us din ke close par becho
  Size  : har trade Dip hisse (kul capital ka 40%) ka 20% = kul capital ka 8%, max 10 positions
          (Ichimoku 60% + Dip 40%; Donchian aur Capitulation sirf paper)
Backtest v2 (6 saal): win ~81%, PF ~4.0, CAGR ~11%, MaxDD ~-18%, bura mahina ~-1.4% (v1: 70% / 2.35 / 14% / -27% / -18%). Kam return lekin baqi 2 bots se ulta
(wo breakout par khareedte hain, ye girawat par) - portfolio ko santulan deta hai.

State: dip_paper_state.json | Band trades: dip_paper_trades.csv | Signals: dip_signals.json
"""
import csv
import json
import os

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, ema, STABLES, FEE, SLIP, STOP_SLIP

# ---------------- Settings ----------------
RSI_N, RSI_LO = 3, 7
EXIT_SMA = 3
TP_PCT = 0.05
STOP_ATR = 3.0
MAX_HOLD = 10
BTC_EMA = 50
MAX_POSITIONS = 10
POS_PCT = 0.20                 # Dip hisse ka 20% har trade
ALLOC = 0.40                   # kul capital mein Dip ka hissa (ICHI 60 / DIP 40)
UNIVERSE = 100
TOP_N_COINS = 150
HISTORY_DAYS = 400
START_EQUITY = 1000.0
STATE_FILE = "dip_paper_state.json"
TRADES_CSV = "dip_paper_trades.csv"
SIGNALS_FILE = "dip_signals.json"


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


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


def send(msg):
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
            e50, e200 = ema(c, 50), ema(c, 200)
            d["uptrend"] = (c > e200) & (e50 > e200) & (np.arange(len(d)) >= 200)
            d["rsi"] = rsi(c, RSI_N)
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
                close_pos(st, sym, float(di.at[day, "open"]) * (1 - SLIP), day, f"uchhal (SMA{EXIT_SMA})", exits)
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
            val = min(POS_PCT * eq_now, st["cash"] / (1 + FEE))
            if val < 5:
                continue
            st["cash"] -= val * (1 + FEE)
            st["positions"][sym] = {"qty": val / entry, "entry": entry, "entry_day": str(day.date()),
                                    "trail": float(p["stop"]), "init_stop": float(p["stop"]), "tp": entry * (1 + TP_PCT),
                                    "last_px": entry, "cost": val * (1 + FEE), "bars": 0, "exit_next": False}
            fills.append(f"📥 {sym} paper-khareeda @ {fmt_px(entry)} (stop {fmt_px(p['stop'])}, TP {fmt_px(entry * (1 + TP_PCT))}, ${val:,.0f})")
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
            if pos.get("tp") and row["high"] >= pos["tp"]:
                close_pos(st, sym, max(pos["tp"], row["open"]) * (1 - SLIP), day, f"TP +{TP_PCT*100:.0f}%", exits)
                continue
            pos["last_px"] = float(row["close"])
            if bool(row["exit_sig"]):
                pos["exit_next"] = True
            elif pos["bars"] >= MAX_HOLD:
                close_pos(st, sym, float(row["close"]) * (1 - SLIP), day, f"{MAX_HOLD} din", exits)
                continue
            pos["bars"] += 1

    # ---------- aaj (D) ke naye signals ----------
    btc_ok = bool(btc["close"].iloc[-1] > ema(btc["close"], BTC_EMA).iloc[-1])
    liq = sorted([(s, di.at[D, "dvol30"]) for s, di in ind.items()
                  if D in di.index and not np.isnan(di.at[D, "dvol30"]) and len(di) >= 60],
                 key=lambda x: -x[1])[:UNIVERSE]
    cands = []
    if btc_ok:
        for s, _ in liq:
            r = ind[s].loc[D]
            if s in st["positions"] or st.get("closed_on", {}).get(s) == Ds:
                continue
            if not bool(r["uptrend"]) or not (r["rsi"] < RSI_LO):
                continue
            if np.isnan(r["stop"]) or r["stop"] >= r["close"]:
                continue
            cands.append((s, float(r["close"]), float(r["stop"]), float(r["rsi"])))
    cands.sort(key=lambda x: x[3])                   # sab se gehri girawat pehle
    chosen = cands[:max(MAX_POSITIONS - len(st["positions"]), 0)]
    st["pending"] = [{"symbol": s, "signal_day": Ds, "stop": stop} for s, _, stop, _ in chosen]

    equity = st["cash"] + sum(p["qty"] * p["last_px"] for p in st["positions"].values())
    st["history"].append({"day": Ds, "equity": round(equity, 2)})
    st["last_day"] = Ds
    st["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
    st["btc_regime_ok"] = btc_ok
    save_state(st)
    append_signals([{"system": "Dip Daily", "symbol": s, "signal_time_utc": str(D + pd.Timedelta(days=1)),
                     "entry_est": round(c, 10), "sl": round(stop, 10), "tp": round(c * (1 + TP_PCT), 10),
                     "risk_pct": round((c - stop) / c * 100, 2), "size_pct": POS_PCT * 100}
                    for s, c, stop, _ in chosen])

    # ---------- Telegram ----------
    L = [f"🎯 <b>Dip Daily Bot v2</b> — {Ds} (daily candle band)",
         f"BTC: {'🟢 BTC > EMA50 (nayi entry allowed)' if btc_ok else '🔴 BTC < EMA50 (nayi entry NAHI)'}"]
    sell_now = [s for s, p in st["positions"].items() if p.get("exit_next")]
    if sell_now:
        L.append(f"\n🔔 <b>AAJ OPEN PAR BECHEIN</b> (close {EXIT_SMA}-din average se ooper band hua):")
        L += [f"• <b>{s}</b> (entry {fmt_px(st['positions'][s]['entry'])}, abhi {fmt_px(st['positions'][s]['last_px'])})"
              for s in sell_now]
    if chosen:
        L.append(f"\n🟢 <b>NAYE BUY SIGNALS</b> ({len(chosen)}) — aaj open par khareedein:")
        L.append(f"Size: Dip hisse ka {POS_PCT*100:.0f}% = kul capital ka {POS_PCT*ALLOC*100:.0f}%")
        L += [f"• <b>{s}</b> ~{fmt_px(c)} | Stop: {fmt_px(stop)} ({(c - stop) / c * 100:.1f}% neeche) | "
              f"TP: +{TP_PCT*100:.0f}% (entry se) | RSI3 {r:.1f}" for s, c, stop, r in chosen]
        L.append(f"Becho: +{TP_PCT*100:.0f}% par (limit order), ya jis din close {EXIT_SMA}-din average se ooper band ho "
                 f"to agle din open par (max {MAX_HOLD} din).")
    elif btc_ok:
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
                     f"({(p['last_px'] / p['entry'] - 1) * 100:+.1f}%) | stop {fmt_px(p['trail'])} | TP {fmt_px(p['tp']) if p.get('tp') else '-'} | din {p['bars']}/{MAX_HOLD}")
    L.append(f"\n💼 Paper equity: ${equity:,.2f} (shuru ${START_EQUITY:,.0f}, {(equity / START_EQUITY - 1) * 100:+.1f}%)")
    if chosen or exits or fills or sell_now:
        send("\n".join(L))
    else:
        print("\n".join(L))
        print("Koi naya waqia nahi - Telegram nahi bheja.")


if __name__ == "__main__":
    main()
