"""
DERIV LAB 15b - Lab 15 ke baad: rank (nisbat) wale qaide + mojooda systems par filter (2026-10-06)
==============================================================================================
Lab 15 sabaq: taker-buy hissa (delta) aksar 49-51% ke beech rehta hai -> pakki hadein (50/52/54%) se trades hi nahi bane.
Is liye ab har din pool coins ki aapas mein tarteeb (rank) se:
  A TOPDELTA q : aaj 3-din delta mein pool ke sab se ooper q% (q 5 / 10 / 20), pehla din. Exit H5.
  B LOWFUND q  : aaj 3-din funding mein pool ke sab se neeche q% (q 5 / 10 / 20), EMA50>EMA200, pehla din. Dip exit.
  C FILTER     : W52 (5 din) aur 4-din girawat ki trades ko funding / delta ki tarteeb se baant kar - kya kisi hisse ko
                 chhorna behtar? (sirf report; tabdeeli tabhi jab DEV aur TAALA dono mein saaf ho.)
Shartein Lab 15 jaisi. OI is dafa nahi (Lab 15 mein data chhota / shor).
Natija: deriv_lab15b_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import STABLES
from portfolio_lab import ema, portfolio, stats
from winrate_lab import sim, tstats, pf_of
import search_lab9 as S9
import search_lab13 as L13
import deriv_lab15 as D

OUT = "deriv_lab15b_RESULTS.txt"
HOLDOUT = S9.HOLDOUT
RND = 10
GRID = {"A_TOPDELTA": [5, 10, 20], "B_LOWFUND": [5, 10, 20]}
EXIT = {"A_TOPDELTA": "h5", "B_LOWFUND": "dip", "REF_DIP": "dip", "X_W52": "h5", "X_STREAK": "dip"}


def add_signals(P):
    # har din pool coins ka percentile
    rows = []
    for s, x in P.items():
        ts = pd.DatetimeIndex(x["ts"])
        rows.append(pd.DataFrame({"sym": s, "day": ts, "d3": x["tbr3"], "f3": x["fund"], "pool": x["pool"]}))
    df = pd.concat(rows)
    pdf = df[df["pool"]].copy()
    pdf["pd3"] = pdf.groupby("day")["d3"].rank(pct=True) * 100
    pdf["pf3"] = pdf.groupby("day")["f3"].rank(pct=True) * 100
    for s, x in P.items():
        sub = pdf[pdf["sym"] == s].set_index("day")
        ts = pd.DatetimeIndex(x["ts"])
        pd3 = sub["pd3"].reindex(ts).to_numpy()
        pf3 = sub["pf3"].reindex(ts).to_numpy()
        x["pd3"], x["pf3"] = pd3, pf3
        c = pd.Series(x["c"])
        golden = (ema(c, 50) > ema(c, 200)).to_numpy()
        for q in GRID["A_TOPDELTA"]:
            x["sig"][f"A_TOPDELTA {q}"] = x["pool"] & L13.fresh(pd.Series(np.nan_to_num(pd3, nan=0) >= 100 - q))
        for q in GRID["B_LOWFUND"]:
            x["sig"][f"B_LOWFUND {q}"] = x["pool"] & golden & L13.fresh(pd.Series(np.nan_to_num(pf3, nan=100) <= q))
        h = pd.Series(x["h"])
        hi365 = h.rolling(365, min_periods=250).max()
        x["sig"]["X_W52"] = x["pool"] & L13.fresh(c >= hi365 * 0.95)
        down = (c < c.shift(1)).astype(float)
        x["sig"]["X_STREAK"] = x["pool"] & golden & L13.fresh(down.rolling(4, min_periods=4).sum() >= 4)


def run(P, name, cost=1.0, rng=None, period=None, only=None):
    out = []
    ex = EXIT[name.split(" ")[0]]
    for sym, x in P.items():
        m = x["sig"][name] if only is None else (x["sig"][name] & only(x))
        idx = np.where(m)[0]
        if rng is not None:
            pool = np.where(x["pool"] & np.isfinite(x["atr"]))[0]
            if len(idx) == 0 or len(pool) == 0:
                continue
            idx = np.sort(rng.choice(pool, size=min(len(idx), len(pool)), replace=False))
        for i in idx:
            tsi = pd.Timestamp(x["ts"][i])
            if (period == "dev" and tsi >= HOLDOUT) or (period == "hold" and tsi < HOLDOUT):
                continue
            if ex == "h5":
                t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, 1e-12, None, None, 5, {}, cost)
            else:
                st = x["c"][i] - 3.0 * x["atr"][i]
                if not np.isfinite(st) or st <= 0:
                    continue
                t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, None, x["sma3"], 10, {"tp": 0.05}, cost)
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
            data_fetcher.AUTO_TOP_N_COINS = 150
            ex = data_fetcher.get_exchange()
            bases = [s.split("/")[0].upper() for s in data_fetcher.get_coin_list(ex)]
            bases = [b for b in bases if b not in STABLES][:150]
            if "BTC" not in bases:
                bases.insert(0, "BTC")
            daily = {}
            for k, b in enumerate(bases, 1):
                d = D.load_coin(f"{b}USDT")
                if d is not None and len(d) >= 250:
                    daily[b] = d
                print(f"[{k}/{len(bases)}] {b}: {0 if d is None else len(d)}", flush=True)
        end = D.END_MONTH + pd.offsets.MonthEnd(0)
        daily = {s: d[d["timestamp"] <= end].reset_index(drop=True) for s, d in daily.items()}
        idx_all = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in daily.values()])))
        start = idx_all[0] + pd.Timedelta(days=210)
        t0 = start
        dev_weeks = (HOLDOUT - t0).days / 7
        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        closes = closes[closes.index >= start]
        P = D.build(daily, start)
        add_signals(P)
        emit("=" * 130)
        emit(f"DERIV LAB 15b - rank wale qaide + filter | coins {len(daily)} | {t0.date()} -> {idx_all[-1].date()} | TAALA {HOLDOUT.date()} se")
        emit("=" * 130)
        emit(f"{'entry':>15} | {'n':>5} | {'jeet/10':>7} | {'PF':>5} | {'rnd95':>5} | {'folds (trades/PF)':>36} | {'boot5':>5} | "
             f"{'-top10':>6} | {'2xcost':>6} | {'ausat':>6}")
        rows = {}
        for nm in ["REF_DIP"] + [f"{f} {v}" for f, g in GRID.items() for v in g]:
            tr = run(P, nm, period="dev")
            if len(tr) < 25:
                emit(f"{nm:>15} | sirf {len(tr)} trades")
                continue
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, period="dev", rng=np.random.default_rng(950 + q))]) for q in range(RND)]
            fo = L13.folds_ok(tr, t0, HOLDOUT)
            bp = S9.boot_p5(tr)
            mt = pf_of([t["ret"] for t in sorted(tr, key=lambda t: -t["ret"])[10:]])
            c2 = pf_of([t["ret"] for t in run(P, nm, period="dev", cost=2.0)])
            rows[nm] = dict(s=s, r95=np.percentile(rnd, 95), fo=fo, bp=bp, mt=mt, c2=c2, wk=s["n"] / dev_weeks)
            emit(f"{nm:>15} | {s['n']:>5} | {s['win']/10:>7.1f} | {s['pf']:>5.2f} | {np.percentile(rnd,95):>5.2f} | "
                 f"{' '.join(f'{n}/{p:.2f}' for n, p, _ in fo):>36} | {bp:>5.2f} | {mt:>6.2f} | {c2:>6.2f} | {s['avg']:>+5.2f}%")
        emit("\n# DEV PASS / FAIL")
        passed = []
        for f, g in GRID.items():
            mid = f"{f} {g[1]}"
            nb = [f"{f} {v}" for k, v in enumerate(g) if k != 1]
            if mid not in rows:
                emit(f"{mid:>15}: FAIL (kam trades)")
                continue
            r, s = rows[mid], rows[mid]["s"]
            wmin = 55 if EXIT[f] == "dip" else 50
            chk = {f"jeet>={wmin}": s["win"] >= wmin, "PF>rnd95": s["pf"] > r["r95"], "PF>=1.3": s["pf"] >= 1.3,
                   "folds": all(o for _, _, o in r["fo"]), "boot>=1.15": r["bp"] >= 1.15, "-top10>=1.2": r["mt"] >= 1.2,
                   "2xcost>=1.2": r["c2"] >= 1.2, ">=0.3/hafta": r["wk"] >= 0.3,
                   "padosi": any(n in rows and rows[n]["s"]["pf"] > rows[n]["r95"] for n in nb)}
            fails = [k for k, v in chk.items() if not v]
            if not fails:
                passed.append(mid)
            emit(f"{mid:>15}: {'DEV PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
        emit("\n# TAALA - DEV pass")
        for nm in passed:
            tr = run(P, nm, period="hold")
            if len(tr) < 5:
                emit(f"{nm:>15}: taale mein sirf {len(tr)} trades")
                continue
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, period="hold", rng=np.random.default_rng(990 + q))]) for q in range(RND)]
            rnd = [x for x in rnd if np.isfinite(x)] or [np.inf]
            ok = s["pf"] >= 1.2 and s["pf"] > np.percentile(rnd, 95)
            emit(f"{nm:>15}: {s['n']} trades | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | random p95 {np.percentile(rnd,95):.2f} -> "
                 f"{'TAALA PASS' if ok else 'TAALA FAIL'}")
            if ok:
                p = stats(portfolio(run(P, nm), closes, "fixed", 0.10, max_pos=10, cap=1.0)[0])
                emit(f"{'':>17}portfolio 10%: CAGR {p['cagr']*100:+.1f}% | DD {p['dd']*100:.1f}% | Sharpe {p['sharpe']:.2f}")
        if not passed:
            emit("koi khayal DEV pass nahi hua")

        emit("\n# FILTER - mojooda systems ki trades funding / delta ki tarteeb se (DEV | TAALA)")
        splits = [("sab", None),
                  ("funding ooper 30% (bheer)", lambda x: np.nan_to_num(x["pf3"], nan=0) > 70),
                  ("funding baqi 70%", lambda x: np.nan_to_num(x["pf3"], nan=0) <= 70),
                  ("delta ooper aadha", lambda x: np.nan_to_num(x["pd3"], nan=0) > 50),
                  ("delta neeche aadha", lambda x: np.nan_to_num(x["pd3"], nan=100) <= 50)]
        for nm, lab in (("X_W52", "Saal ki chouti (5 din)"), ("X_STREAK", "4 din girawat"), ("REF_DIP", "Dip v2")):
            emit(f"{lab}:")
            for sl, fn in splits:
                a = run(P, nm, period="dev", only=fn)
                b = run(P, nm, period="hold", only=fn)
                sa = tstats(a) if a else {"n": 0, "win": 0, "pf": 0}
                sb = tstats(b) if b else {"n": 0, "win": 0, "pf": 0}
                emit(f"   {sl:>26}: DEV {sa['n']:>4}tr jeet {sa['win']/10:.1f} PF {sa['pf']:.2f} | TAALA {sb['n']:>3}tr jeet "
                     f"{sb['win']/10:.1f} PF {sb['pf']:.2f}")
    except Exception:
        emit("\n!! GHALTI:\n" + traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
