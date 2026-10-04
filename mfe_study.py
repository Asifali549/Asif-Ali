"""
MFE STUDY - harne wali trades pehle kitna ooper gayin? Kya chhota TP / breakeven madad karta?
=============================================================================================
User ka sawal (2026-10-04): 10 mein se 6 harne wali trades - kya woh entry se hi ulat jati hain, ya pehle
2-4% ooper ja kar wapas SL hit karti hain? Agar ooper ja kar wapas aati hain to kya chhota TP lagana chahiye?

Har system (ICHI, DON, DIP, CAPIT) ki trades bilkul portfolio_lab jaisi (entry agli candle open, fee+slip,
stop slip, gap fill, pehle stop check phir trail update). Har trade ka MFE = entry ke baad (exit candle se pehle
tak) sab se ooncha high / entry - 1.

Hissa 1: harne wali trades ka MFE (kitna ooper gayin) - bins.
Hissa 2: exit variants (BASE ke sath):
   P{X}  : X% par aadhi position bech do (partial TP 50%), baqi BASE jaisi
   BE{X} : X% ooper jane ke baad stop entry (+kharcha) par le aao
   TP{X} : poori position X% par bech do (sirf hawale ke liye)
 Same candle mein stop aur target dono choo jayen to STOP pehle maana (mohtaat).
 Har variant: trades, win%, PF, avg, OOS 2025+ PF, aur portfolio (ICHI 1% risk / DIP,CAPIT 20% size,
 DON 1% risk) ka CAGR / MaxDD / Sharpe.
Natija: mfe_study_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, chandelier as ce_bot, ichi_signal, ICHI_BASE, STABLES, FEE, SLIP, STOP_SLIP
from portfolio_lab import ema, rsi, atr_w, to_daily, portfolio, stats, TOP_N, H4_BARS, UNIVERSE

OUT = "mfe_study_RESULTS.txt"
OOS_START = pd.Timestamp("2025-01-01")
BE_PX = (1 + FEE) * (1 + SLIP) / ((1 - FEE) * (1 - SLIP))   # entry ke barabar (kharcha samet)

VARIANTS = [("BASE", None)]
for x in (0.02, 0.03, 0.04, 0.05, 0.08):
    VARIANTS.append((f"P{int(x*100)}", ("partial", x, 0.5)))
for x in (0.02, 0.03, 0.05, 0.08):
    VARIANTS.append((f"BE{int(x*100)}", ("be", x)))
for x in (0.03, 0.05):
    VARIANTS.append((f"TP{int(x*100)}", ("tp", x)))


def sim_var(o, h, l, c, ts, i, stop, trail, exit_sig, tp_r, max_hold, var):
    n = len(o)
    e = i + 1
    if e >= n:
        return None
    entry = o[e] * (1 + SLIP)
    st = stop
    if not np.isfinite(st) or entry <= st:
        return None
    R = entry - st
    tp = entry + tp_r * R if tp_r else None
    kind = var[0] if var else None
    lvl = entry * (1 + var[1]) if var else None
    if kind == "tp":
        tp = lvl if tp is None else min(tp, lvl)
    part_done, part_ret, frac = False, 0.0, (var[2] if kind == "partial" else 0.0)
    be_on = False
    last = min(e + max_hold, n - 1)
    px, j = None, last
    mfe = 0.0
    for j in range(e, last + 1):
        if l[j] <= st:
            px = min(st * (1 - STOP_SLIP), o[j]) * (1 - SLIP)
            break
        if tp is not None and h[j] >= tp:
            px = max(tp, o[j]) * (1 - SLIP)
            break
        if kind == "partial" and not part_done and h[j] >= lvl:
            part_done = True
            part_ret = max(lvl, o[j]) * (1 - SLIP) * (1 - FEE) / (entry * (1 + FEE)) - 1
        if exit_sig is not None and exit_sig[j] and j + 1 < n:
            mfe = max(mfe, h[j] / entry - 1)
            j += 1
            px = o[j] * (1 - SLIP)
            break
        mfe = max(mfe, h[j] / entry - 1)
        if kind == "be" and not be_on and h[j] >= lvl:
            be_on = True
            st = max(st, entry * BE_PX / (1 - STOP_SLIP))
        if trail is not None and np.isfinite(trail[j]) and trail[j] > st:
            st = trail[j]
    if px is None:
        px = c[last] * (1 - SLIP)
    rest = px * (1 - FEE) / (entry * (1 + FEE)) - 1
    ret = frac * part_ret + (1 - frac) * rest if part_done else rest
    return {"t_in": ts[e], "t_out": ts[j], "entry_px": entry, "risk": R / entry, "ret": ret, "mfe": mfe,
            "bars": j - e}


def gen(system, h4, daily, allowed, btc_ok, var):
    out = []
    if system == "ICHI":
        for sym, d in h4.items():
            if len(d) < 400:
                continue
            sig = ichi_signal(d, dict(ICHI_BASE))
            stop = ce_bot(d, 16, 5.5)
            mom = (d["close"] / d["close"].shift(60 * 6) - 1).to_numpy()
            o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
            ts = d["timestamp"].to_numpy()
            day = d["timestamp"].dt.floor("1D")
            for i in np.where(sig)[0]:
                if sym not in allowed.get(day.iloc[i], ()):
                    continue
                t = sim_var(o, h, l, c, ts, i, stop[i], stop, None, 3.0, 500, var)
                if t:
                    t.update(sym=sym, prio=mom[i] if np.isfinite(mom[i]) else -9)
                    out.append(t)
        return out
    for sym, d in daily.items():
        c = d["close"]
        o, hh, l, cc = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        ts = d["timestamp"].to_numpy()
        bok = btc_ok.reindex(d["timestamp"]).fillna(False).to_numpy(bool)
        if system == "DON":
            brk = (c > d["high"].shift(1).rolling(20).max()).fillna(False)
            sig = (brk & ~brk.shift(1, fill_value=False)).to_numpy() & bok
            stop = ce_bot(d, 22, 4.0)
            mom = (c / c.shift(60) - 1).to_numpy()
            for i in np.where(sig)[0]:
                if sym not in allowed.get(d["timestamp"].iloc[i], ()):
                    continue
                t = sim_var(o, hh, l, cc, ts, i, stop[i], stop, None, None, 365, var)
                if t:
                    t.update(sym=sym, prio=mom[i] if np.isfinite(mom[i]) else -9)
                    out.append(t)
            continue
        e200 = ema(c, 200)
        stop = (c - 3.0 * atr_w(d)).to_numpy()
        ex = (c > c.rolling(5).mean()).to_numpy()
        if system == "DIP":
            e50 = ema(c, 50)
            trend = ((c > e200) & (e50 > e200)).to_numpy() & bok
            trend[:200] = False
            r = rsi(c, 3).to_numpy()
            sig = trend & (r < 10)
            prio = -r
        else:  # CAPIT
            ret1 = (c / c.shift(1) - 1)
            vavg = d["volume"].rolling(20).mean().shift(1)
            up = (c > e200).to_numpy().copy()
            up[:200] = False
            sig = up & (ret1 <= -0.08).to_numpy() & (d["volume"] >= 2.0 * vavg).fillna(False).to_numpy()
            prio = -ret1.to_numpy()
        for i in np.where(sig)[0]:
            if sym not in allowed.get(d["timestamp"].iloc[i], ()):
                continue
            t = sim_var(o, hh, l, cc, ts, i, stop[i], None, ex, None, 10, var)
            if t:
                t.update(sym=sym, prio=prio[i] if np.isfinite(prio[i]) else -9)
                out.append(t)
    return out


def tstats(tr):
    r = np.array([t["ret"] for t in tr])
    if len(r) == 0:
        return dict(n=0, win=0, pf=0, avg=0, oos=0)
    g, lo = r[r > 0].sum(), -r[r <= 0].sum()
    ro = np.array([t["ret"] for t in tr if pd.Timestamp(t["t_in"]) >= OOS_START])
    go, loo = ro[ro > 0].sum(), -ro[ro <= 0].sum()
    return dict(n=len(r), win=(r > 0).mean() * 100, pf=g / lo if lo else np.inf, avg=r.mean() * 100,
                oos=go / loo if loo else np.inf)


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x)
        lines.append(x)

    if h4 is None:
        from data_fetcher import get_exchange, get_coin_list
        ex = get_exchange()
        coins = [s for s in get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:TOP_N]
        for must in ("ETH/USDT", "BTC/USDT"):
            if must not in coins:
                coins.insert(0, must)
        h4 = {}
        for k, sym in enumerate(coins, 1):
            try:
                df = fetch_full(ex, sym, "4h", H4_BARS)
                if df is not None and len(df) >= 6 * 250:
                    h4[sym] = norm(df)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)}")
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})")

    daily = {s: to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc_ok = bd > ema(bd, 50)
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    allowed = {day: set(row.dropna().nlargest(UNIVERSE).index) for day, row in dv.iterrows()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes = closes[closes.index >= start]

    emit("=" * 100)
    emit("MFE STUDY - harne wali trades pehle kitna ooper gayin + chhota TP / breakeven ka asar")
    emit("=" * 100)
    emit(f"Coins: {len(h4)} | {closes.index[0].date()} -> {closes.index[-1].date()} | top-{UNIVERSE} liquid | OOS = 2025+")
    emit("P = X% par aadhi bech do | BE = X% ke baad stop entry par | TP = poori X% par (hawala)")
    emit("MFE = exit candle se pehle tak sab se ooncha high / entry (mohtaat; exit candle ka high shamil nahi)")

    PSIZE = {"ICHI": ("risk", 0.01), "DON": ("risk", 0.01), "DIP": ("fixed", 0.20), "CAPIT": ("fixed", 0.20)}
    bins = [-1, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 99]
    blab = ["<1%", "1-2%", "2-3%", "3-5%", "5-10%", "10-20%", ">20%"]
    for system in ("ICHI", "DON", "DIP", "CAPIT"):
        emit("\n" + "#" * 100)
        emit(f"# {system}")
        emit("#" * 100)
        res = {}
        for name, var in VARIANTS:
            tr = [t for t in gen(system, h4, daily, allowed, btc_ok, var) if pd.Timestamp(t["t_in"]) >= start]
            res[name] = tr
        base = res["BASE"]
        r = np.array([t["ret"] for t in base])
        mfe = np.array([t["mfe"] for t in base])
        risk = np.array([t["risk"] for t in base])
        los = r <= 0
        emit(f"BASE trades {len(r)} | haar {los.sum()} ({los.mean()*100:.0f}%) | median stop doori {np.median(risk)*100:.1f}%"
             f" | harne wali ka median nuqsan {np.median(r[los])*100:.1f}%")
        emit("\nHarne wali trades: nuqsan se pehle ZIADA SE ZIADA kitna ooper gayin (MFE)?")
        cnt = pd.cut(mfe[los], bins=bins, labels=blab).value_counts().reindex(blab)
        cum = 0
        for lab in blab:
            k = int(cnt[lab])
            cum += k
            emit(f"   {lab:>7}: {k:>5} ({k/los.sum()*100:>5.1f}%)  | jama: {cum/los.sum()*100:>5.1f}%")
        emit(f"   median MFE harne walon ka: {np.median(mfe[los])*100:.1f}% | jeetne walon ka: {np.median(mfe[~los])*100:.1f}%")
        for x in (0.02, 0.03, 0.05):
            emit(f"   {int(x*100)}% ooper gayi phir bhi haari: {(mfe[los] >= x).sum()} trades = kul trades ka {(los & (mfe >= x)).mean()*100:.1f}%"
                 f" | jeetne wali jo {int(x*100)}% tak kabhi nahi gayin: {(~los & (mfe < x)).sum()}")
        emit("\nVariants (trade-level + portfolio):")
        emit(f"{'Variant':>7} | {'trades':>6} | {'win%':>5} | {'PF':>5} | {'avg%':>6} | {'OOS PF':>6} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6}")
        mode, size = PSIZE[system]
        for name, _ in VARIANTS:
            s = tstats(res[name])
            eq, _t = portfolio(res[name], closes, mode, size)
            p = stats(eq)
            emit(f"{name:>7} | {s['n']:>6} | {s['win']:>5.1f} | {s['pf']:>5.2f} | {s['avg']:>+6.2f} | {s['oos']:>6.2f} | "
                 f"{p['cagr']*100:>+6.1f}% | {p['dd']*100:>6.1f}% | {p['sharpe']:>6.2f}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
