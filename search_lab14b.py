"""
SEARCH LAB 14b - BTC_FEAR 35 (bull market mein BTC RSI14 < 35) ka taala + Market safai (Flush) se rishta (2026-10-06)
=================================================================================================================
DEV mein sirf "folds" fail: fold 2 (2022) mein 0 trades - BTC < EMA200 tha, shart khud bear ko bahar rakhti hai (Dip jaisa) -> BORDERLINE.
Taala EK dafa. Saath: kitne alag din, padosi 30/40, Flush ke signal dinon se overlap (wohi din ya +-3 din), portfolio, correlation.
Natija: search_lab14b_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import portfolio, stats
from winrate_lab import tstats, pf_of
import search_lab9 as S9
import search_lab13 as L13
import search_lab14 as L

OUT = "search_lab14b_RESULTS.txt"
NM = "D_BTC_FEAR 35"


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
        P13 = L13.build(daily, start)
        emit("=" * 120)
        emit(f"SEARCH LAB 14b | coins {len(daily)} | TAALA {L.HOLDOUT.date()} -> {closes.index[-1].date()}")
        emit("=" * 120)

        emit("\n# TAALA KHULA (ek dafa)")
        tr = L.run(P, NM, period="hold")
        if len(tr) >= 5:
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in L.run(P, NM, period="hold", rng=np.random.default_rng(990 + q))]) for q in range(20)]
            rnd = [x for x in rnd if np.isfinite(x)] or [np.inf]
            ok = s["pf"] >= 1.2 and s["pf"] > np.percentile(rnd, 95)
            days = sorted(set(str(pd.Timestamp(t["t_in"]).date()) for t in tr))
            srt = sorted(tr, key=lambda t: -t["ret"])
            emit(f"{NM}: {s['n']} trades ({len(days)} alag din: {', '.join(days)}) | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | "
                 f"random p50 {np.median(rnd):.2f} p95 {np.percentile(rnd,95):.2f} | ausat {s['avg']:+.2f}% | top-5 hata kar "
                 f"{pf_of([t['ret'] for t in srt[5:]]):.2f} -> {'TAALA PASS' if ok else 'TAALA FAIL'}")
        else:
            emit(f"{NM}: taale mein sirf {len(tr)} trades - faisla mumkin nahi")
        for nb in ("D_BTC_FEAR 30", "D_BTC_FEAR 40"):
            tb = L.run(P, nb, period="hold")
            if tb:
                sb = tstats(tb)
                emit(f"   padosi {nb}: {sb['n']} trades | jeet/10 {sb['win']/10:.1f} | PF {sb['pf']:.2f}")

        emit("\n# POORA 6 SAAL - signal din, Flush se overlap")
        al = L.run(P, NM)
        s = tstats(al)
        fear_days = sorted(set(pd.Timestamp(t["t_in"]).floor("1D") for t in al))
        fl = L13.run(P13, "F_FLUSH 40")
        fl_days = sorted(set(pd.Timestamp(t["t_in"]).floor("1D") for t in fl))
        near = [d for d in fear_days if any(abs((d - f).days) <= 3 for f in fl_days)]
        emit(f"{NM}: {s['n']} trades | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | ausat {s['avg']:+.2f}% | {len(fear_days)} signal din "
             f"| Flush ke +-3 din mein {len(near)} ({len(near)/max(len(fear_days),1)*100:.0f}%)")
        alag = [t for t in al if pd.Timestamp(t["t_in"]).floor("1D") not in set(near)]
        if alag:
            sa = tstats(alag)
            emit(f"   Flush se ALAG din ki trades: {sa['n']} | jeet/10 {sa['win']/10:.1f} | PF {sa['pf']:.2f}")
        yr = pd.Series([t["ret"] for t in al], index=[pd.Timestamp(t["t_in"]).year for t in al])
        emit("   saal: " + " | ".join(f"{y}: {len(g)}tr PF {pf_of(list(g)):.2f}" for y, g in yr.groupby(level=0)))
        f_m = portfolio(fl, closes, "fixed", 0.10, max_pos=10, cap=1.0)[0].resample("ME").last().pct_change()
        d_m = portfolio(L13.run(P13, "REF_DIP"), closes, "fixed", 0.20, max_pos=10, cap=1.0)[0].resample("ME").last().pct_change()
        for size in (0.10, 0.20):
            eq = portfolio(al, closes, "fixed", size, max_pos=10, cap=1.0)[0]
            p = stats(eq)
            m = eq.resample("ME").last().pct_change()
            yrs = eq.resample("YE").last().pct_change().dropna()
            emit(f"   {int(size*100)}%: CAGR {p['cagr']*100:+.1f}% | DD {p['dd']*100:.1f}% | Sharpe {p['sharpe']:.2f} | bura mahina "
                 f"{m.min()*100:.1f}% | corr Flush {m.corr(f_m):.2f} Dip {m.corr(d_m):.2f} | saal "
                 + " ".join(f"{d.year}:{v*100:+.0f}" for d, v in yrs.items()))
    except Exception:
        emit("\n!! GHALTI:\n" + traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
