"""
POWER TEST - kya hamara imtihan ka tareeqa ghalat hai? (user, 2026-10-06)
======================================================================
Nakli (random-walk) market banao, phir JAAN BOOJH KAR ek sacha faida chhupao: kuch random dinon ke baad agle 3 din mein
qeemat G% ooper dhakel do. Phir wohi imtihan (Dip exit, random control, 4 folds, boot p5, top-10 hata kar, 2x kharcha, jeet)
chalao aur dekho:
  * G = 0 (koi faida nahi): kitni dafa ghalti se PASS? (jhoota pass - kam hona chahiye)
  * G = 1 / 2 / 3 / 5%: kitni dafa PASS? (asli faida pakarne ki taqat)
Har G par 8 alag nakli markets (seeds). Har market: 100 coins, 1500 din, ~1 signal / coin / 100 din.
Natija: power_test_RESULTS.txt
"""
import numpy as np
import pandas as pd

from portfolio_lab import ema, atr_w
from winrate_lab import sim, tstats, pf_of
import strategy_lab5 as L5
import search_lab9 as S9
import search_lab13 as L13

OUT = "power_test_RESULTS.txt"


def market(seed, G, n_coins=100, n=1500, p_sig=0.01):
    rng = np.random.default_rng(seed)
    daily, chosen = {}, {}
    mkt = rng.standard_t(4, n) * 0.025
    for k in range(n_coins):
        r = 0.7 * mkt + rng.standard_t(4, n) * 0.025 + 0.0007
        sig = rng.random(n) < p_sig
        sig[:250] = False
        sig[-10:] = False
        if G:
            for i in np.where(sig)[0]:
                r[i + 1:i + 4] += np.log(1 + G) / 3       # agle 3 din mein G% dhakka
        c = 10 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.003, n))
        h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.015, n)))
        l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.015, n)))
        sym = "BTC/USDT" if k == 0 else f"C{k}/USDT"
        daily[sym] = pd.DataFrame(dict(timestamp=pd.date_range("2020-01-01", periods=n, freq="1D"), open=o, high=h, low=l,
                                       close=c, volume=rng.lognormal(14, 0.6, n)))
        chosen[sym] = sig
    return daily, chosen


def build(daily, chosen, start):
    btc_ok, al100 = L5.context(daily)
    P = {}
    for sym, d in daily.items():
        c = d["close"]
        ts = d["timestamp"]
        ok = np.array([sym in al100.get(t, ()) for t in ts]) & (ts >= start).to_numpy()
        up = (c > ema(c, 200)).to_numpy() & (np.arange(len(d)) >= 200)
        bok = btc_ok.reindex(ts).fillna(False).to_numpy(bool)
        pool = ok & up & bok
        P[sym] = dict(o=d["open"].to_numpy(float), h=d["high"].to_numpy(float), l=d["low"].to_numpy(float), c=c.to_numpy(float),
                      ts=ts.to_numpy(), atr=atr_w(d).to_numpy(), sma3=(c > c.rolling(3).mean()).to_numpy(), pool=pool,
                      sig=pool & chosen[sym])
    return P


def run(P, cost=1.0, rng=None):
    out = []
    for sym, x in P.items():
        idx = np.where(x["sig"])[0]
        if rng is not None:
            pool = np.where(x["pool"] & np.isfinite(x["atr"]))[0]
            if len(idx) == 0 or len(pool) == 0:
                continue
            idx = np.sort(rng.choice(pool, size=min(len(idx), len(pool)), replace=False))
        for i in idx:
            st = x["c"][i] - 3 * x["atr"][i]
            if not np.isfinite(st) or st <= 0:
                continue
            t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, None, x["sma3"], 10, {"tp": 0.05}, cost)
            if t:
                t.update(sym=sym, prio=0.0)
                out.append(t)
    return out


def judge(P, t0, t1):
    tr = run(P)
    if len(tr) < 25:
        return None
    s = tstats(tr)
    rnd = [pf_of([t["ret"] for t in run(P, rng=np.random.default_rng(100 + q))]) for q in range(10)]
    r95 = np.percentile(rnd, 95)
    fo = L13.folds_ok(tr, t0, t1)
    bp = S9.boot_p5(tr, n=1000)
    mt = pf_of([t["ret"] for t in sorted(tr, key=lambda t: -t["ret"])[10:]])
    c2 = pf_of([t["ret"] for t in run(P, cost=2.0)])
    chk = {"jeet>=55": s["win"] >= 55, "PF>rnd95": s["pf"] > r95, "PF>=1.3": s["pf"] >= 1.3, "folds": all(o for _, _, o in fo),
           "boot>=1.15": bp >= 1.15, "-top10>=1.2": mt >= 1.2, "2xcost>=1.2": c2 >= 1.2}
    return s, r95, [k for k, v in chk.items() if not v]


def main():
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    emit("POWER TEST - nakli market mein chhupaya hua faida: hamara imtihan pakarta hai ya nahi?")
    emit("G = har signal ke baad agle 3 din mein chhupaya gaya ooper ka dhakka. 8 markets har G par.\n")
    for G in (0.0, 0.01, 0.02, 0.03, 0.05):
        passes, rows, fails_all = 0, [], {}
        for seed in range(8):
            daily, chosen = market(seed, G)
            start = daily["BTC/USDT"]["timestamp"].iloc[0] + pd.Timedelta(days=210)
            P = build(daily, chosen, start)
            t1 = daily["BTC/USDT"]["timestamp"].iloc[-1]
            res = judge(P, start, t1)
            if res is None:
                continue
            s, r95, fails = res
            rows.append((s["n"], s["win"], s["pf"], r95))
            passes += not fails
            for f in fails:
                fails_all[f] = fails_all.get(f, 0) + 1
        n = np.mean([r[0] for r in rows])
        w = np.mean([r[1] for r in rows]) / 10
        pf = np.mean([r[2] for r in rows])
        r95 = np.mean([r[3] for r in rows])
        emit(f"G {G*100:>3.0f}% | PASS {passes}/{len(rows)} | ausat {n:.0f} trades | jeet/10 {w:.1f} | PF {pf:.2f} | random p95 {r95:.2f} | "
             f"fail ki wajahen: " + (", ".join(f"{k} {v}" for k, v in sorted(fails_all.items(), key=lambda x: -x[1])) or "-"))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
