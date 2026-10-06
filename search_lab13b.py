"""
SEARCH LAB 13b - Lab 13 ke 2 qareeb wale khayal ka taala (2026-10-06)
====================================================================
1) F_FLUSH 40 (market safai): DEV mein sirf "folds" fail - fold 2 (2022) 27 trades PF 0.88. BORDERLINE (Streak jaisa).
2) R_RVOL_UP 2 (volume 2x, +3..12% din, close ooper 20%): beech wali V3 fail (kam trades), V2 sab pass - magar V2 ko natija
   dekh kar chuna = BAAD MEIN CHUNA (post-hoc), is liye sakht shart: taale mein PF > random p95 AUR PF >= 1.5.
Taala dono ka EK dafa. Phir portfolio (10% / 20%, max 10), Dip+ aur Streak se mahana correlation, saal-war, aur ek din mein
kitne signals (jhurmut).
Natija: search_lab13b_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import portfolio, stats
from winrate_lab import tstats, pf_of
import search_lab9 as S9
import search_lab13 as L

OUT = "search_lab13b_RESULTS.txt"
CANDS = {"F_FLUSH 40": 1.2, "R_RVOL_UP 2": 1.5}
NEIGH = {"F_FLUSH 40": ["F_FLUSH 30", "F_FLUSH 50"], "R_RVOL_UP 2": ["R_RVOL_UP 3"]}


def main(daily=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    try:
        if daily is None:
            import data_fetcher
            data_fetcher.AUTO_TOP_N_COINS = S9.TOP_FETCH
            ex = data_fetcher.get_exchange()
            coins = [s for s in data_fetcher.get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:S9.TOP_FETCH]
            if "BTC/USDT" not in coins:
                coins.insert(0, "BTC/USDT")
            daily = {}
            for k, sym in enumerate(coins, 1):
                try:
                    df = fetch_full(ex, sym, "1d", S9.DAYS)
                    if df is not None and len(df) >= 250:
                        daily[sym] = norm(df)
                except Exception as e:
                    print(f"[{k}] {sym}: SKIP ({e})", flush=True)
        idx_all = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in daily.values()])))
        start = idx_all[0] + pd.Timedelta(days=210)
        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        closes = closes[closes.index >= start]
        P = L.build(daily, start)
        emit("=" * 120)
        emit(f"SEARCH LAB 13b | coins {len(daily)} | TAALA {L.HOLDOUT.date()} -> {closes.index[-1].date()}")
        emit("=" * 120)
        emit("\n# TAALA KHULA (ek dafa)")
        for nm, need in CANDS.items():
            tr = L.run(P, nm, period="hold")
            if len(tr) < 5:
                emit(f"{nm:>14}: taale mein sirf {len(tr)} trades")
                continue
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in L.run(P, nm, period="hold", rng=np.random.default_rng(990 + q))]) for q in range(20)]
            rnd = [x for x in rnd if np.isfinite(x)] or [np.inf]
            ok = s["pf"] >= need and s["pf"] > np.percentile(rnd, 95)
            days = pd.Series([str(pd.Timestamp(t["t_in"]).date()) for t in tr])
            srt = sorted(tr, key=lambda t: -t["ret"])
            emit(f"{nm:>14}: {s['n']} trades ({days.nunique()} alag din) | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | random p50 "
                 f"{np.median(rnd):.2f} p95 {np.percentile(rnd,95):.2f} | ausat {s['avg']:+.2f}% | top-5 hata kar PF "
                 f"{pf_of([t['ret'] for t in srt[5:]]):.2f} -> {'TAALA PASS' if ok else 'TAALA FAIL'}")
            bym = pd.Series([t["ret"] for t in tr], index=[pd.Timestamp(t["t_in"]).strftime("%Y-%m") for t in tr])
            emit("    mahana: " + " | ".join(f"{m}: {len(g)}tr {g.mean()*100:+.1f}%" for m, g in bym.groupby(level=0)))
            for nb in NEIGH[nm]:
                tb = L.run(P, nb, period="hold")
                if tb:
                    sb = tstats(tb)
                    emit(f"{'padosi ' + nb:>26}: {sb['n']} trades | jeet/10 {sb['win']/10:.1f} | PF {sb['pf']:.2f}")

        emit("\n# PORTFOLIO (poora 6 saal, max 10) + jhurmut")
        ref = {}
        for lab, nm in (("Dip v2", "REF_DIP"), ("Resid", "X_RESID"), ("Streak", "X_STREAK")):
            ref[lab] = portfolio(L.run(P, nm), closes, "fixed", 0.20, max_pos=10, cap=1.0)[0].resample("ME").last().pct_change()
        for nm in CANDS:
            tr = L.run(P, nm)
            st = tstats(tr)
            per_day = pd.Series([str(pd.Timestamp(t["t_in"]).date()) for t in tr]).value_counts()
            emit(f"{nm}: {st['n']} trades | jeet/10 {st['win']/10:.1f} | PF {st['pf']:.2f} | ausat {st['avg']:+.2f}% | {len(per_day)} signal din, "
                 f"ek din mein ausat {per_day.mean():.1f}, max {per_day.max()}")
            for size in (0.10, 0.20):
                eq = portfolio(tr, closes, "fixed", size, max_pos=10, cap=1.0)[0]
                p = stats(eq)
                m = eq.resample("ME").last().pct_change()
                yrs = eq.resample("YE").last().pct_change().dropna()
                emit(f"   {int(size*100)}%: CAGR {p['cagr']*100:+.1f}% | DD {p['dd']*100:.1f}% | Sharpe {p['sharpe']:.2f} | bura mahina "
                     f"{m.min()*100:.1f}% | corr " + " ".join(f"{k} {m.corr(v):.2f}" for k, v in ref.items()))
                emit(f"        saal " + " ".join(f"{d.year}:{v*100:+.0f}" for d, v in yrs.items()))
    except Exception:
        emit("\n!! GHALTI:\n" + traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
