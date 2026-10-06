"""
W52 HOLDOUT - Lab 10 ka ek hi umeedwar (P 5%, H10 / H7) - DEV mein sirf "jeet >= 55%" (5.1/10) aur fold-2 (2022, trades
na hone se 'inf') par ruka; user ka apna usool "10 mein se 5-6 jeet bhi theek agar jeet bari ho". Ab TAALA (aakhri
12 mahine) EK DAFA khulta hai - is ke baad is khayal par mazeed tabdeeli nahi (warna taala bekaar).
Plus: saal-war, portfolio (10% / trade, max 10), random holdout, aur DIP+RESID khate ke sath.
Natija: w52_holdout_RESULTS.txt
"""
import traceback
import numpy as np
import pandas as pd
from bot_core import fetch_full, norm, STABLES
from portfolio_lab import portfolio, stats
from winrate_lab import tstats, pf_of
import search_lab9 as S9
import w52_lab10 as W
import strategy_lab5 as L5

OUT = "w52_holdout_RESULTS.txt"


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
                except Exception as e:
                    print(f"[{kk}] {sym}: SKIP ({e})", flush=True)
        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        start = closes.index[0] + pd.Timedelta(days=210)
        closes = closes[closes.index >= start]
        P = S9.build(daily, start)
        emit("=" * 120)
        emit(f"W52 HOLDOUT - TAALA {S9.HOLDOUT.date()} -> {closes.index[-1].date()} (ek dafa) | coins {len(daily)}")
        emit("=" * 120)
        years = list(range(2021, 2027))
        L5.GRID["R1_RESID"] = [("z-2.0", -2.0)]
        btc_ok, al100 = L5.context(daily)
        P5 = L5.build(daily, btc_ok, al100, start)
        book = L5.run(P5, "REF_DIP") + L5.run(P5, "R1_RESID z-2.0")
        for p in (0.03, 0.05, 0.08):
            for ex in ("H7", "H10"):
                hold = W.run(P, p, ex, "hold")
                s = tstats(hold)
                rnd = [pf_of([t["ret"] for t in W.run(P, p, ex, "hold", rng=np.random.default_rng(80 + q))]) for q in range(20)]
                ok = s["pf"] >= 1.2 and s["pf"] > np.percentile(rnd, 95)
                emit(f"\nP {p} {ex}: TAALA {s['n']} trades | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | random p50 "
                     f"{np.median(rnd):.2f} p95 {np.percentile(rnd,95):.2f} | ausat {s['avg']:+.2f}% -> {'TAALA PASS' if ok else 'TAALA FAIL'}")
                full = W.run(P, p, ex, "all")
                for size, mx in ((0.10, 10), (0.20, 5)):
                    pst = stats(portfolio(full, closes, "fixed", size, max_pos=mx, cap=1.0)[0])
                    yr = {d.year: v for d, v in pst["yearly"].items()}
                    emit(f"   akela {int(size*100)}% x max {mx}: CAGR {pst['cagr']*100:+.1f}% | DD {pst['dd']*100:.1f}% | Sharpe "
                         f"{pst['sharpe']:.2f} | +mah {pst['pos_months']:.0f}% | bura mah {pst['worst_month']*100:.1f}% | "
                         + " ".join(f"{y}:{yr.get(y, np.nan)*100:+.0f}%" for y in years))
                seen = {(t["sym"], pd.Timestamp(t["t_in"])) for t in book}
                extra = [t for t in full if (t["sym"], pd.Timestamp(t["t_in"])) not in seen]
                for nm, trs in (("DIP+RESID", book), ("DIP+RESID+W52", book + extra)):
                    pst = stats(portfolio(trs, closes, "fixed", 0.20, max_pos=10, cap=1.0)[0])
                    yr = {d.year: v for d, v in pst["yearly"].items()}
                    emit(f"   khata {nm:>14} 20% x 10: CAGR {pst['cagr']*100:+.1f}% | DD {pst['dd']*100:.1f}% | Sharpe "
                         f"{pst['sharpe']:.2f} | +mah {pst['pos_months']:.0f}% | " + " ".join(f"{y}:{yr.get(y, np.nan)*100:+.0f}%" for y in years))
    except Exception:
        emit("\n!! GHALTI:\n" + traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
