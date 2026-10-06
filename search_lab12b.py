"""
SEARCH LAB 12b - Lab 12 ke 2 qareeb wale khayal (2026-10-06)
==========================================================
1) E_W52_DIP 15 sirf "folds" par ruka - fold 2 (2022-23) ka PF "inf" tha (koi trade haari nahi) aur shart isfinite maangti
   thi. Ye hisaab ki ghalti hai: yahan har fold ki trades/haar ginti dikhate hain; fold PASS = (trades >= 5 aur PF > 1)
   ya (trades >= 5 aur koi haar nahi). Baqi sab DEV shartein Lab 12 mein pass thin.
2) P_DOWN_STREAK 4 sirf "5d fark" par ruka (saaf 5-din nafa random se behtar nahi) - BORDERLINE; taala sirf maloomat ke
   liye, PASS tabhi jab taala bhi saaf ho.
Taala (2025-10-01 se) in dono ke liye EK dafa. Phir poora portfolio (10% / 20%, max 10) aur Dip v2 se mahana correlation.
Natija: search_lab12b_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from winrate_lab import tstats, pf_of
import search_lab9 as S9
import search_lab12 as L

OUT = "search_lab12b_RESULTS.txt"
CANDS = ["E_W52_DIP 15", "P_DOWN_STREAK 4"]
NEIGH = {"E_W52_DIP 15": ["E_W52_DIP 10", "E_W52_DIP 20"], "P_DOWN_STREAK 4": ["P_DOWN_STREAK 3", "P_DOWN_STREAK 5"]}


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
        t0 = start
        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        closes = closes[closes.index >= start]
        P = L.build(daily, start)
        emit("=" * 120)
        emit(f"SEARCH LAB 12b | coins {len(daily)} | DEV {t0.date()} -> {L.HOLDOUT.date()} | TAALA {L.HOLDOUT.date()} -> {closes.index[-1].date()}")
        emit("=" * 120)
        e = pd.date_range(t0, L.HOLDOUT, periods=5)
        for nm in ["REF_DIP"] + CANDS + sum(NEIGH.values(), []):
            tr = L.run(P, nm, period="dev")
            parts = []
            for a, b in zip(e[:-1], e[1:]):
                r = [t["ret"] for t in tr if a <= pd.Timestamp(t["t_in"]) < b]
                parts.append(f"{len(r)}tr/{sum(x <= 0 for x in r)}haar PF {pf_of(r):.2f}")
            emit(f"{nm:>18} folds: " + " | ".join(parts))

        emit("\n# TAALA KHULA (ek dafa)")
        res = {}
        for nm in ["REF_DIP"] + CANDS:
            tr = L.run(P, nm, period="hold")
            if len(tr) < 5:
                emit(f"{nm:>18}: taale mein sirf {len(tr)} trades")
                continue
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in L.run(P, nm, period="hold", rng=np.random.default_rng(990 + q))]) for q in range(20)]
            rnd = [x for x in rnd if np.isfinite(x)] or [np.inf]
            ok = s["pf"] >= 1.2 and s["pf"] > np.percentile(rnd, 95)
            res[nm] = ok
            emit(f"{nm:>18}: {s['n']} trades | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | random p50 {np.median(rnd):.2f} p95 "
                 f"{np.percentile(rnd,95):.2f} | ausat {s['avg']:+.2f}% -> {'TAALA PASS' if ok else 'TAALA FAIL'}")
            for nb in NEIGH.get(nm, []):
                tb = L.run(P, nb, period="hold")
                if tb:
                    sb = tstats(tb)
                    emit(f"{'padosi ' + nb:>30}: {sb['n']} trades | jeet/10 {sb['win']/10:.1f} | PF {sb['pf']:.2f} (sirf maloomat)")

        emit("\n# PORTFOLIO (poora 6 saal, max 10)")
        ref_m = L.daily_ret(L.run(P, "REF_DIP"), closes, 0.20).resample("ME").last().pct_change()
        for nm in ["REF_DIP"] + CANDS:
            tr = L.run(P, nm)
            st = tstats(tr)
            emit(f"{nm}: {st['n']} trades | jeet/10 {st['win']/10:.1f} | PF {st['pf']:.2f} | ausat {st['avg']:+.2f}%")
            for size in (0.10, 0.20):
                eq = L.daily_ret(tr, closes, size)
                from portfolio_lab import stats
                p = stats(eq)
                m = eq.resample("ME").last().pct_change()
                yrs = eq.resample("YE").last().pct_change().dropna()
                emit(f"   {int(size*100)}%: CAGR {p['cagr']*100:+.1f}% | DD {p['dd']*100:.1f}% | Sharpe {p['sharpe']:.2f} | corr DIP "
                     f"{m.corr(ref_m):.2f} | bura mahina {m.min()*100:.1f}% | saal " + " ".join(f"{d.year}:{v*100:+.0f}" for d, v in yrs.items()))
    except Exception:
        emit("\n!! GHALTI:\n" + traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
