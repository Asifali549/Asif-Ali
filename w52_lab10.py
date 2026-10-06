"""
W52 LAB 10 - Search Lab 9 ka sab se dilchasp ishara (2026-10-06)
================================================================
Lab 9 mein "sal ki chouti ke qareeb pehli dafa" (A_W52_HIGH) ka 5-din ausat fark random se +4.8% tha (shor sirf 0.8%) -
sab se taqatwar entry ishara - magar chandelier 22/3 exit ne sab wapas de diya (jeet 3.7/10, PF ~1.05).
Sawal: kya ye CHHOTI MUDDAT ki taqat hai? Yahan wohi entry (P = 3 / 5 / 8%), aur sirf chhoti muddat ke exits:
  H5 / H7 / H10   : 5 / 7 / 10 din baad us din ke close par (koi SL/TP nahi)
  S3_H7           : SL signal close - 3 ATR, 7 din
  S3_T15_H10      : SL 3 ATR, TP +15%, 10 din
  S2_T10_H7       : SL 2 ATR, TP +10%, 7 din
Pehle "fark ka sach": median 5-din, top-10 / top-1% hata kar ausat - kya ye chand pump coins ka kamal hai?
Phir DEV (2025-10-01 se pehle) par PASS shartein (Lab 9 wali); sirf PASS par TAALA (aakhri 12 mahine) khulta hai.
Natija: w52_lab10_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import portfolio, stats
from winrate_lab import sim, tstats, pf_of
import search_lab9 as S9

OUT = "w52_lab10_RESULTS.txt"
EXITS = {"H5": (None, None, 5), "H7": (None, None, 7), "H10": (None, None, 10),
         "S3_H7": (3.0, None, 7), "S3_T15_H10": (3.0, 0.15, 10), "S2_T10_H7": (2.0, 0.10, 7)}
PS = [0.03, 0.05, 0.08]
RND = 10


def run(P, p, ex, period, cost=1.0, rng=None):
    k, tp, hold = EXITS[ex]
    name = f"A_W52_HIGH {p}"
    out = []
    for sym, x in P.items():
        idx = np.where(x["sig"][name])[0]
        if rng is not None:
            pool = np.where(x["pool"] & np.isfinite(x["atr"]))[0]
            if len(idx) == 0 or len(pool) == 0:
                continue
            idx = np.sort(rng.choice(pool, size=min(len(idx), len(pool)), replace=False))
        for i in idx:
            tsi = pd.Timestamp(x["ts"][i])
            if (period == "dev" and tsi >= S9.HOLDOUT) or (period == "hold" and tsi < S9.HOLDOUT):
                continue
            st = 1e-12 if k is None else x["c"][i] - k * x["atr"][i]
            if not np.isfinite(st) or st <= 0:
                continue
            t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, None, None, hold, {"tp": tp} if tp else {}, cost)
            if t:
                t.update(sym=sym, prio=0.0)
                out.append(t)
    return out


def main(daily=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    try:
        if daily is None:
            import data_fetcher
            data_fetcher.AUTO_TOP_N_COINS = S9.TOP_FETCH
            exch = data_fetcher.get_exchange()
            coins = [s for s in data_fetcher.get_coin_list(exch) if s.split("/")[0].upper() not in STABLES][:S9.TOP_FETCH]
            if "BTC/USDT" not in coins:
                coins.insert(0, "BTC/USDT")
            daily = {}
            for kk, sym in enumerate(coins, 1):
                try:
                    df = fetch_full(exch, sym, "1d", S9.DAYS)
                    if df is not None and len(df) >= 250:
                        daily[sym] = norm(df)
                        print(f"[{kk}/{len(coins)}] {sym}: {len(df)}", flush=True)
                except Exception as e:
                    print(f"[{kk}] {sym}: SKIP ({e})", flush=True)

        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        start = closes.index[0] + pd.Timedelta(days=210)
        closes = closes[closes.index >= start]
        t0, t1 = closes.index[0], closes.index[-1]
        dev_weeks = (S9.HOLDOUT - t0).days / 7
        P = S9.build(daily, start)
        emit("=" * 130)
        emit("W52 LAB 10 - sal ki chouti ke qareeb (pehli dafa) + chhoti muddat exits | DEV pehle, TAALA baad mein")
        emit("=" * 130)
        emit(f"Coins: {len(daily)} | DEV {t0.date()} -> {S9.HOLDOUT.date()} | TAALA {S9.HOLDOUT.date()} -> {t1.date()}")

        emit("\n# FARQ KA SACH (DEV, 5 din, koi SL/TP nahi) - kya nafa chand pump coins se hai?")
        for p in PS:
            r = np.array(sorted([t["ret"] for t in run(P, p, "H5", "dev")]))
            rr = np.array([np.mean([t["ret"] for t in run(P, p, "H5", "dev", rng=np.random.default_rng(40 + q))]) for q in range(RND)])
            emit(f"P {p}: n {len(r)} | ausat {r.mean()*100:+.2f}% (random {rr.mean()*100:+.2f}%) | MEDIAN {np.median(r)*100:+.2f}% | "
                 f"top-10 hata kar ausat {r[:-10].mean()*100:+.2f}% | top-1% hata kar {r[:int(len(r)*0.99)].mean()*100:+.2f}% | "
                 f"5 din baad ooper {np.mean(r > 0)*10:.1f}/10")

        emit(f"\n# DEV - har exit (jeet/10 = 10 mein se nafa wali, PF = 1 rupay nuqsan par nafa)")
        emit(f"{'P':>5} | {'exit':>11} | {'n':>4} | {'jeet/10':>7} | {'PF':>5} | {'rnd95':>5} | {'folds':>23} | {'boot5':>5} | {'-top10':>6} | "
             f"{'2xcost':>6} | {'ausat':>6}")
        rows = {}
        for ex in EXITS:
            for p in PS:
                tr = run(P, p, ex, "dev")
                if len(tr) < 25:
                    continue
                s = tstats(tr)
                rnd = [pf_of([t["ret"] for t in run(P, p, ex, "dev", rng=np.random.default_rng(60 + q))]) for q in range(RND)]
                fo = S9.folds(tr, t0, S9.HOLDOUT)
                bp = S9.boot_p5(tr)
                mt = pf_of([t["ret"] for t in sorted(tr, key=lambda t: -t["ret"])[10:]])
                c2 = pf_of([t["ret"] for t in run(P, p, ex, "dev", cost=2.0)])
                rows[(p, ex)] = dict(s=s, r95=np.percentile(rnd, 95), fo=fo, bp=bp, mt=mt, c2=c2)
                emit(f"{p:>5} | {ex:>11} | {s['n']:>4} | {s['win']/10:>7.1f} | {s['pf']:>5.2f} | {np.percentile(rnd,95):>5.2f} | "
                     f"{' '.join(f'{f:>5.2f}' for f in fo):>23} | {bp:>5.2f} | {mt:>6.2f} | {c2:>6.2f} | {s['avg']:>+5.2f}%")
            emit("")

        emit("# DEV PASS (P 0.05 beech wali; padosi P 0.03 / 0.08 mein se 1 bhi PF > random p95)")
        passed = []
        for ex in EXITS:
            key = (0.05, ex)
            if key not in rows:
                continue
            r, s = rows[key], rows[key]["s"]
            chk = {"jeet>=55": s["win"] >= 55, "PF>rnd95": s["pf"] > r["r95"], "PF>=1.4": s["pf"] >= 1.4,
                   "folds": all(np.isfinite(f) and f > 1 for f in r["fo"]), "boot>=1.15": r["bp"] >= 1.15,
                   "-top10>=1.2": r["mt"] >= 1.2, "2xcost>=1.2": r["c2"] >= 1.2,
                   "padosi": any((q, ex) in rows and rows[(q, ex)]["s"]["pf"] > rows[(q, ex)]["r95"] for q in (0.03, 0.08))}
            fails = [k for k, v in chk.items() if not v]
            if not fails:
                passed.append(ex)
            emit(f"P 0.05 {ex:>11}: {'DEV PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")

        emit("\n# TAALA (aakhri 12 mahine) - sirf DEV PASS par")
        for ex in passed:
            tr = run(P, 0.05, ex, "hold")
            if len(tr) < 5:
                emit(f"{ex}: taale mein sirf {len(tr)} trades")
                continue
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, 0.05, ex, "hold", rng=np.random.default_rng(80 + q))]) for q in range(RND)]
            ok = s["pf"] >= 1.2 and s["pf"] > np.percentile(rnd, 95)
            full = run(P, 0.05, ex, "all")
            pst = stats(portfolio(full, closes, "fixed", 0.10, max_pos=10, cap=1.0)[0])
            emit(f"P 0.05 {ex:>11}: taala {s['n']} trades | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | random p95 "
                 f"{np.percentile(rnd,95):.2f} -> {'TAALA PASS' if ok else 'TAALA FAIL'} | poora portfolio 10%/trade: CAGR "
                 f"{pst['cagr']*100:+.1f}% DD {pst['dd']*100:.1f}% Sharpe {pst['sharpe']:.2f}")
        if not passed:
            emit("koi exit DEV pass nahi hua - taala nahi khula")
    except Exception:
        emit("\n!! GHALTI (code crash):\n" + traceback.format_exc())

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
