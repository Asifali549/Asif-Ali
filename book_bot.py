"""
BOOK BOT - daily "khata" paper bots ka mushtarka engine (2026-10-06)
====================================================================
Do bots isi engine par chalte hain (har ek ki apni chhoti file + workflow + files):
  dipplus_bot.py : DIP+ = Dip v2 (RSI3 < 7) + Residual Dip (BTC ke muqable z < -2) EK khate mein  (Combo Lab 6)
  w52_bot.py     : W52  = sal ki chouti ke 5% andar pehli dafa, 5 din baad becho                (W52 Lab 10 / Combo)
  flush_bot.py   : market safai (top-100 ke 40%+ coins 3 din mein -10%), BTC > EMA200, Dip exit   (Search Lab 13/13b)
  streak_bot.py  : 4 din lagataar neeche close, EMA50 > EMA200, Dip exit                         (Search Lab 12/12b)
  w52trail_bot.py: W52 + chalta SL (nafa 5% par chale, chouti se 2% neeche), 5 din               (W52 Combo Test)
Hisaab bilkul backtest jaisa (strategy_lab5 / search_lab9 / winrate_lab.sim):
  - signal BAND daily candle par, khareed AGLE din ke open par (+ slippage)
  - liquidity: pichle 30 din ka ausat dollar volume (kal tak) -> top-100
  - har din tarteeb: (1) kal ka "SMA se ooper" -> aaj open par becho (2) kal ke signals -> aaj open par khareedo
    (3) har khuli trade: SL -> TP -> exit signal -> din ki ginti (10 din par us din ke close par)
State: <prefix>_paper_state.json | trades: <prefix>_paper_trades.csv | signals: <prefix>_signals.json
"""
import csv
import json
import os

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, ema, STABLES, FEE, SLIP, STOP_SLIP

CFG = {}                 # wrapper file bharti hai
START_EQUITY = 1000.0
TOP_N_COINS = 150
UNIVERSE = 100


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def fmt_px(x):
    return f"{x:,.2f}" if x >= 100 else (f"{x:.4f}" if x >= 1 else f"{x:.6g}")


# ------------------------------------------------------------------ indicators (backtest jaise)
def indicators(d, btc_lr):
    c, h = d["close"], d["high"]
    e50, e200 = ema(c, 50), ema(c, 200)
    n_ok = np.arange(len(d)) >= 200
    d["up"] = (c > e200) & n_ok
    d["golden"] = e50 > e200
    d["rsi3"] = rsi(c, 3)
    d["atr"] = atr(d)
    d["sma3_up"] = c > c.rolling(3).mean()
    d["dvol30"] = (c * d["volume"]).rolling(30, min_periods=20).mean().shift(1)
    # residual z (strategy_lab5.build)
    rc = np.log(c).diff()
    rb = btc_lr.reindex(d["timestamp"]).to_numpy()
    s_rc, s_rb = pd.Series(rc.to_numpy()), pd.Series(rb)
    beta = (s_rc.rolling(60, min_periods=40).cov(s_rb) / s_rb.rolling(60, min_periods=40).var()).clip(-1, 3)
    res = s_rc - beta * s_rb
    d["z"] = (res.rolling(3).sum() / (res.rolling(60, min_periods=40).std() * np.sqrt(3))).to_numpy()
    # 4 din lagataar girawat (search_lab12 P_DOWN_STREAK 4): aaj 4th neeche close (5th par dobara nahi)
    down = (c < c.shift(1)).astype(float)
    st4 = down.rolling(4, min_periods=4).sum() >= 4
    d["streak4"] = st4 & ~st4.shift(1, fill_value=False)
    # W52 (search_lab9.build)
    hi365 = h.rolling(365, min_periods=250).max()
    near = c >= hi365 * (1 - CFG.get("w52_p", 0.05))
    d["w52"] = (near & ~near.shift(1, fill_value=False)).fillna(False)
    return d


def signal_today(r):
    """(kya signal, kis wajah se, tarteeb) - r = coin ki aaj ki row."""
    kind = CFG["kind"]
    if kind == "dipplus":
        tags = []
        if bool(r["up"]) and bool(r["golden"]) and r["rsi3"] < 7:
            tags.append("DIP")
        if bool(r["up"]) and np.isfinite(r["z"]) and r["z"] < -2.0:
            tags.append("RESID")
        prio = r["rsi3"] - 100 if "DIP" in tags else (r["z"] if tags else 0)
        return bool(tags), "+".join(tags), prio
    if kind == "streak":
        return bool(r["up"]) and bool(r["golden"]) and bool(r["streak4"]), "4-DIN", 0.0
    if kind == "flush":
        return bool(r["up"]) and bool(r["golden"]) and bool(r.get("flush", False)), "SAFAI", 0.0
    if kind == "w52":
        return bool(r["up"]) and bool(r["w52"]), "W52", 0.0
    raise ValueError(kind)


# ------------------------------------------------------------------ state
def load_state():
    if os.path.exists(CFG["state"]):
        with open(CFG["state"]) as f:
            return json.load(f)
    return {"cash": START_EQUITY, "positions": {}, "pending": [], "last_day": None, "history": []}


def save_state(st):
    with open(CFG["state"], "w") as f:
        json.dump(st, f, indent=2)


def log_trade(row):
    new = not os.path.exists(CFG["trades"])
    with open(CFG["trades"], "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def append_signals(new_rows, keep=300):
    rows = []
    if os.path.exists(CFG["signals"]):
        try:
            with open(CFG["signals"]) as f:
                rows = json.load(f)
        except Exception:
            rows = []
    with open(CFG["signals"], "w") as f:
        json.dump((rows + new_rows)[-keep:], f, indent=2)


def send(msg):
    try:
        from telegram_alert import send_telegram_alert
        send_telegram_alert(msg)
    except Exception as e:
        print(f"Telegram fail: {e}")
    print(msg)


def close_pos(st, sym, px, day, why, msgs):
    pos = st["positions"].pop(sym)
    st.setdefault("closed_on", {})[sym] = str(day.date())
    proceeds = pos["qty"] * px * (1 - FEE)
    st["cash"] += proceeds
    pnl = proceeds - pos["cost"]
    ret = pnl / pos["cost"] * 100
    log_trade({"symbol": sym, "entry_day": pos["entry_day"], "exit_day": str(day.date()),
               "entry": round(pos["entry"], 8), "exit": round(px, 8),
               "pnl_usd": round(pnl, 2), "ret_pct": round(ret, 2), "reason": why, "tag": pos.get("tag", "")})
    msgs.append(f"{'✅' if pnl > 0 else '❌'} {sym} band ({why}) @ {fmt_px(px)} | {ret:+.1f}% (${pnl:+,.2f})")


# ------------------------------------------------------------------ ek din ka hisaab (replay mein bhi istemal)
def process_day(st, ind, day, exits, fills):
    stop_k, tp, hold, use_sma = CFG["stop_atr"], CFG["tp"], CFG["hold"], CFG["sma_exit"]
    act, gap = CFG.get("trail_act"), CFG.get("trail_gap")
    # 1) kal SMA3 se ooper band -> aaj open par becho
    for sym in list(st["positions"]):
        pos, di = st["positions"][sym], ind.get(sym)
        if pos.get("exit_next") and di is not None and day in di.index:
            close_pos(st, sym, float(di.at[day, "open"]) * (1 - SLIP), day, "uchhal (SMA3)", exits)
    # 2) kal ke signals -> aaj open par khareedo
    keep = []
    for p in st["pending"]:
        sym, nxt = p["symbol"], pd.Timestamp(p["signal_day"]) + pd.Timedelta(days=1)
        if nxt != day:
            if nxt > day:
                keep.append(p)
            continue
        di = ind.get(sym)
        if di is None or day not in di.index or sym in st["positions"] or len(st["positions"]) >= CFG["max_pos"]:
            continue
        entry = float(di.at[day, "open"]) * (1 + SLIP)
        stop = p.get("stop")
        if stop is not None and stop >= entry:
            continue
        eq_now = st["cash"] + sum(q["qty"] * q["last_px"] for q in st["positions"].values())
        val = min(CFG["pos_pct"] * eq_now, st["cash"] / (1 + FEE))
        if val < 5:
            continue
        st["cash"] -= val * (1 + FEE)
        st["positions"][sym] = {"qty": val / entry, "entry": entry, "entry_day": str(day.date()),
                                "trail": float(stop) if stop is not None else None,
                                "init_stop": float(stop) if stop is not None else None,
                                "tp": entry * (1 + tp) if tp else None, "last_px": entry, "cost": val * (1 + FEE),
                                "bars": 0, "exit_next": False, "tag": p.get("tag", "")}
        fills.append(f"📥 {sym} [{p.get('tag', '')}] paper-khareeda @ {fmt_px(entry)}"
                     + (f" (SL {fmt_px(stop)}" if stop is not None else " (SL nahi")
                     + (f", TP {fmt_px(entry * (1 + tp))}" if tp else "") + f", ${val:,.0f})")
    st["pending"] = keep
    # 3) khuli trades: SL -> TP -> SMA3 -> din
    for sym in list(st["positions"]):
        pos, di = st["positions"][sym], ind.get(sym)
        if di is None or day not in di.index or pd.Timestamp(pos["entry_day"]) > day or pos.get("exit_next"):
            continue
        row = di.loc[day]
        if pos.get("trail") is not None and row["low"] <= pos["trail"]:
            close_pos(st, sym, min(pos["trail"] * (1 - STOP_SLIP), row["open"]) * (1 - SLIP), day,
                      "CHALTA SL" if pos["trail"] > pos["entry"] else "STOP", exits)
            continue
        if pos.get("tp") and row["high"] >= pos["tp"]:
            close_pos(st, sym, max(pos["tp"], row["open"]) * (1 - SLIP), day, f"TP +{tp*100:.0f}%", exits)
            continue
        pos["last_px"] = float(row["close"])
        if act is not None:      # chalta SL (Trail Lab 11): chouti A% par pohnche -> SL = entry x (1 + nafa - G), agle din se
            pos["peak"] = max(pos.get("peak", pos["entry"]), float(row["high"]))
            gain = pos["peak"] / pos["entry"] - 1
            if gain >= act:
                new_sl = pos["entry"] * (1 + gain - gap)
                if pos.get("trail") is None or new_sl > pos["trail"]:
                    pos["trail"] = new_sl
        if use_sma and bool(row["sma3_up"]):
            pos["exit_next"] = True
        elif pos["bars"] >= hold:
            close_pos(st, sym, float(row["close"]) * (1 - SLIP), day, f"{hold} din", exits)
            continue
        pos["bars"] += 1


def new_signals(st, ind, D):
    btc = ind["BTC/USDT"]
    btc_ok = bool(btc.at[D, "close"] > ema(btc["close"], CFG.get("btc_ema", 50)).loc[D])
    Ds = str(D.date())
    liq = sorted([(s, di.at[D, "dvol30"]) for s, di in ind.items()
                  if D in di.index and np.isfinite(di.at[D, "dvol30"])], key=lambda x: -x[1])[:UNIVERSE]
    cands = []
    if btc_ok:
        for s, _ in liq:
            r = ind[s].loc[D]
            if s in st["positions"]:
                continue
            ok, tag, prio = signal_today(r)
            if not ok:
                continue
            stop = float(r["close"] - CFG["stop_atr"] * r["atr"]) if CFG["stop_atr"] else None
            if stop is not None and (not np.isfinite(stop) or stop <= 0 or stop >= r["close"]):
                continue
            cands.append((s, float(r["close"]), stop, tag, prio))
    cands.sort(key=lambda x: x[4])
    chosen = cands[:max(CFG["max_pos"] - len(st["positions"]), 0)]
    st["pending"] = [{"symbol": s, "signal_day": Ds, "stop": stop, "tag": tag} for s, _, stop, tag, _ in chosen]
    return chosen, btc_ok


def build_ind(daily):
    btc = norm(daily["BTC/USDT"]).set_index("timestamp")["close"]
    btc_lr = np.log(btc).diff()
    ind = {s: indicators(norm(d).copy(), btc_lr).set_index("timestamp") for s, d in daily.items()}
    # market safai (search_lab13 F_FLUSH): top-100 (dvol30) mein kitne % coins ka 3-din nafa <= -10%
    dv = pd.DataFrame({s: di["dvol30"] for s, di in ind.items()}).sort_index()
    r3 = pd.DataFrame({s: di["close"] / di["close"].shift(3) - 1 for s, di in ind.items()}).sort_index()
    in100 = dv.rank(axis=1, ascending=False) <= UNIVERSE
    breadth = ((r3 <= -0.10) & in100).sum(axis=1) / in100.sum(axis=1).replace(0, np.nan) * 100
    for s, di in ind.items():
        hit = (breadth.reindex(di.index).fillna(0) >= CFG.get("flush_pct", 40))
        di["flush"] = hit & ~hit.shift(1, fill_value=False)
    return ind


def main():
    from data_fetcher import get_exchange, get_coin_list
    ex = get_exchange()
    coins = [c for c in get_coin_list(ex) if c.split("/")[0].upper() not in STABLES][:TOP_N_COINS]
    if "BTC/USDT" not in coins:
        coins.insert(0, "BTC/USDT")
    daily = {}
    for sym in coins:
        try:
            d = fetch_full(ex, sym, "1d", CFG["history_days"])
            if d is not None and len(d) >= 30:
                daily[sym] = d
        except Exception as e:
            print(f"{sym}: SKIP ({e})")
    ind = build_ind(daily)
    btc = ind["BTC/USDT"]
    D = btc.index[-1]
    Ds = str(D.date())

    st = load_state()
    if st["last_day"] == Ds:
        print(f"{Ds} pehle hi process ho chuka.")
        st["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
        save_state(st)
        return
    last_day = pd.Timestamp(st["last_day"]) if st["last_day"] else D - pd.Timedelta(days=1)
    exits, fills = [], []
    for day in [t for t in btc.index if last_day < t <= D]:
        process_day(st, ind, day, exits, fills)
    chosen, btc_ok = new_signals(st, ind, D)

    equity = st["cash"] + sum(p["qty"] * p["last_px"] for p in st["positions"].values())
    st["history"].append({"day": Ds, "equity": round(equity, 2)})
    st["last_day"] = Ds
    st["last_updated"] = pd.Timestamp.now(tz="UTC").isoformat()
    st["btc_regime_ok"] = btc_ok
    save_state(st)
    tp = CFG["tp"]
    append_signals([{"system": CFG["name"], "symbol": s, "signal_time_utc": str(D + pd.Timedelta(days=1)),
                     "entry_est": round(c, 10), "sl": round(stop, 10) if stop is not None else None,
                     "tp": round(c * (1 + tp), 10) if tp else None,
                     "risk_pct": round((c - stop) / c * 100, 2) if stop is not None else None,
                     "size_pct": CFG["pos_pct"] * 100, "tag": tag} for s, c, stop, tag, _ in chosen])

    # ---------- Telegram ----------
    L = [f"{CFG['emoji']} <b>{CFG['name']} (SIRF PAPER)</b> — {Ds} (daily candle band)",
         f"BTC: {'🟢 BTC > EMA50 (nayi entry allowed)' if btc_ok else '🔴 BTC < EMA50 (nayi entry NAHI)'}"]
    sell_now = [s for s, p in st["positions"].items() if p.get("exit_next")]
    if sell_now:
        L.append("\n🔔 <b>AAJ OPEN PAR BECHEIN</b> (close 3-din average se ooper band hua):")
        L += [f"• <b>{s}</b> (entry {fmt_px(st['positions'][s]['entry'])}, abhi {fmt_px(st['positions'][s]['last_px'])})"
              for s in sell_now]
    due = [s for s, p in st["positions"].items() if not p.get("exit_next") and p["bars"] >= CFG["hold"]]
    if due:
        L.append(f"\n⏰ <b>AAJ CLOSE PAR BECHEIN</b> ({CFG['hold']} din poore):")
        L += [f"• <b>{s}</b>" for s in due]
    if chosen:
        L.append(f"\n🟢 <b>NAYE BUY SIGNALS</b> ({len(chosen)}) — aaj open par khareedein (har trade khate ka "
                 f"{CFG['pos_pct']*100:.0f}%):")
        for s, c, stop, tag, _ in chosen:
            parts = [f"• <b>{s}</b> [{tag}] ~{fmt_px(c)}"]
            parts.append(f"SL {fmt_px(stop)} ({(c - stop) / c * 100:.1f}% neeche)" if stop is not None else "SL nahi")
            if tp:
                parts.append(f"TP +{tp*100:.0f}%")
            L.append(" | ".join(parts))
        L.append(CFG["howto"])
    elif btc_ok:
        L.append("\nAaj koi naya signal nahi.")
    if fills:
        L.append("\n" + "\n".join(fills))
    if exits:
        L.append("\n<b>Band hui trades:</b>\n" + "\n".join(exits))
    hold = [s for s in st["positions"] if s not in sell_now]
    if hold:
        L.append(f"\n<b>Khuli positions ({len(st['positions'])}/{CFG['max_pos']}):</b>")
        for s in hold:
            p = st["positions"][s]
            L.append(f"• {s} [{p.get('tag', '')}]: entry {fmt_px(p['entry'])} | abhi {fmt_px(p['last_px'])} "
                     f"({(p['last_px'] / p['entry'] - 1) * 100:+.1f}%) | SL {fmt_px(p['trail']) if p.get('trail') else '-'} | "
                     f"din {p['bars']}/{CFG['hold']}")
    L.append(f"\n💼 Paper equity: ${equity:,.2f} (shuru ${START_EQUITY:,.0f}, {(equity / START_EQUITY - 1) * 100:+.1f}%)")
    if chosen or exits or fills or sell_now or due:
        send("\n".join(L))
    else:
        print("\n".join(L))
        print("Koi naya waqia nahi - Telegram nahi bheja.")
