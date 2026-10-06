"""
REGIME LAB 18 - Macro Lab 17 ka "halat filter" hamare systems par (user, 2026-10-06: "jaise aap ko acha lage check karo")
=====================================================================================================================
Filter (pehle se tay, koi tuning NAHI): signal ke din market mein tabhi jab
   STABLE30 > 0 (stablecoins 30 din mein barhe = naya paisa)  AUR  Nasdaq > apni 50-din average.
(dono 1 din peeche - sirf maloom data.)  Baqi din naya signal NAHI liya jata (khuli trade apne qaide se chalti hai).
Systems: Donchian (live jaisa: 20-din breakout, BTC>EMA50, CE22/4 trailing, maxSL 20%, 1% risk), W52 5 din (10%),
W52 + chalta SL (10%), Dip v2 (20%), Resid (20%), 4-din girawat (10%), Market safai (10%).
Har system: sab trades vs filter ON trades vs OFF trades (jeet/10, PF, ausat) - DEV aur TAALA alag; portfolio CAGR / DD / Sharpe
sab vs filter ke sath (poora + taala); aur CONTROL: wohi filter waqt mein khiska kar (circular shift 8 dafa, wohi lambai/tanasub) -
agar asal filter shifted filters se behtar nahi to faida ittefaq / sirf kam exposure hai.
Natija: regime_lab18_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema, portfolio, stats
from winrate_lab import tstats, pf_of
import strategy_lab5 as L5
import search_lab9 as S9
import search_lab13 as L13
import stop_fix_lab as SF
import trail_lab11 as T11
import w52_lab10 as W
import macro_lab17 as M

OUT = "regime_lab18_RESULTS.txt"
HOLDOUT = S9.HOLDOUT


def main(daily=None, feats=None):
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
        if feats is None:
            st = M.stable_series()
            ndq, how = M.market_series("NDQ", "^ndq", "%5EIXIC")
            ndq = ndq.asfreq("D").ffill()
            feats = {"STABLE30": (st / st.shift(30) - 1) * 100, "NDQ_TREND": (ndq / ndq.rolling(50).mean() - 1) * 100}
        feats = {k: v.sort_index().asfreq("D").ffill().shift(1) for k, v in feats.items()}

        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        start = closes.index[0] + pd.Timedelta(days=210)
        closes = closes[closes.index >= start]
        idx = closes.index
        cond = ((feats["STABLE30"].reindex(idx) > 0) & (feats["NDQ_TREND"].reindex(idx) > 0)).fillna(False)

        # systems ki trades
        btc_ok, al100 = L5.context(daily)
        Psf = SF.prep(daily, al100, btc_ok, start)
        P13 = L13.build(daily, start)
        P9 = S9.build(daily, start)
        U = T11.units_w52(daily, start)
        systems = {
            "Donchian": (SF.run(Psf, "DON", 4.0, None, 0.20), ("risk", 0.01)),
            "W52 5 din": (W.run(P9, 0.05, "H5", "all"), ("fixed", 0.10)),
            "W52 + chalta SL": (T11.run(U, dict(k=None, act=0.05, gap=0.02, hold=5), "all"), ("fixed", 0.10)),
            "Dip v2": (L13.run(P13, "REF_DIP"), ("fixed", 0.20)),
            "Resid": (L13.run(P13, "X_RESID"), ("fixed", 0.20)),
            "4-din girawat": (L13.run(P13, "X_STREAK"), ("fixed", 0.10)),
            "Market safai": (L13.run(P13, "F_FLUSH 40"), ("fixed", 0.10)),
        }
        emit("=" * 130)
        emit(f"REGIME LAB 18 - halat filter (STABLE30 > 0 AUR Nasdaq > SMA50) | coins {len(daily)} | {idx[0].date()} -> {idx[-1].date()} | "
             f"TAALA {HOLDOUT.date()} se")
        emit("=" * 130)
        emit(f"Filter ON din: poora {cond.mean()*100:.0f}% | DEV {cond[idx < HOLDOUT].mean()*100:.0f}% | TAALA {cond[idx >= HOLDOUT].mean()*100:.0f}%")
        emit("jeet/10 = 10 mein se nafa wali | PF = 1 rupay nuqsan par nafa")

        def sig_day(t):
            return pd.Timestamp(t["t_in"]).floor("1D") - pd.Timedelta(days=1)

        def pstats(tr, mode, size, cl):
            if not tr:
                return None
            eq = portfolio(tr, cl, mode, size, max_pos=10, cap=0.20 if mode == "risk" else 1.0)[0]
            return stats(eq), eq

        rng = np.random.default_rng(7)
        shifts = rng.integers(120, len(idx) - 120, size=8)
        for name, (tr, (mode, size)) in systems.items():
            emit(f"\n## {name} ({len(tr)} trades)")
            for per, lo, hi in (("DEV", idx[0], HOLDOUT), ("TAALA", HOLDOUT, idx[-1] + pd.Timedelta(days=1))):
                sub = [t for t in tr if lo <= pd.Timestamp(t["t_in"]) < hi]
                on = [t for t in sub if bool(cond.get(sig_day(t), False))]
                off = [t for t in sub if not bool(cond.get(sig_day(t), False))]
                cells = []
                for lab, g in (("sab", sub), ("ON", on), ("OFF", off)):
                    if len(g) >= 5:
                        s = tstats(g)
                        cells.append(f"{lab} {s['n']}tr jeet {s['win']/10:.1f} PF {s['pf']:.2f} ausat {s['avg']:+.2f}%")
                    else:
                        cells.append(f"{lab} {len(g)}tr -")
                emit(f"   {per:>5}: " + " | ".join(cells))
            # portfolio
            for per, cl in (("POORA", closes), ("TAALA", closes[closes.index >= HOLDOUT])):
                base_tr = [t for t in tr if pd.Timestamp(t["t_in"]) >= cl.index[0]]
                filt_tr = [t for t in base_tr if bool(cond.get(sig_day(t), False))]
                a, b = pstats(base_tr, mode, size, cl), pstats(filt_tr, mode, size, cl)
                if a is None or b is None:
                    emit(f"   {per} portfolio: data kam")
                    continue
                ctrl = []
                if per == "POORA":
                    for s_ in shifts:
                        cc = pd.Series(np.roll(cond.to_numpy(), int(s_)), index=idx)
                        ft = [t for t in base_tr if bool(cc.get(sig_day(t), False))]
                        r = pstats(ft, mode, size, cl)
                        if r:
                            ctrl.append(r[0]["sharpe"])
                yrs_a = a[1].resample("YE").last().pct_change().dropna()
                yrs_b = b[1].resample("YE").last().pct_change().dropna()
                emit(f"   {per} portfolio: SAB CAGR {a[0]['cagr']*100:+.1f}% DD {a[0]['dd']*100:.1f}% Sharpe {a[0]['sharpe']:.2f} | FILTER CAGR "
                     f"{b[0]['cagr']*100:+.1f}% DD {b[0]['dd']*100:.1f}% Sharpe {b[0]['sharpe']:.2f}"
                     + (f" | khiske filter Sharpe p50 {np.median(ctrl):.2f} max {np.max(ctrl):.2f}" if ctrl else ""))
                if per == "POORA":
                    emit("      saal SAB:    " + " ".join(f"{d.year}:{v*100:+.0f}" for d, v in yrs_a.items()))
                    emit("      saal FILTER: " + " ".join(f"{d.year}:{v*100:+.0f}" for d, v in yrs_b.items()))
    except Exception:
        emit("\n!! GHALTI:\n" + traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
