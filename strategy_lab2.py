"""
STRATEGY LAB 2 - naye daily khayal + tamam live systems ki ek ranking (2026-10-02)
==================================================================================
Sab Daily, spot buy-only, signal BAND din par, entry AGLE din ke OPEN par, top-100 liquid (point-in-time),
kharcha fee 0.1% + slip 0.05% har taraf + stop slip 0.25%, gap par exit = min(stop, open).
NAYE KHAYAL (har ek ka 6-cell grid = plateau):
 E1 BB_REVERSION : close > EMA200 aur close < Bollinger lower band (20, k)  | k {1.5, 2.0, 2.5} x exit {SMA5, SMA20}
                   stop 3 ATR, max 10 din (SMA20 exit par max 20 din)
 E2 FAILED_BREAKDOWN (Turtle-soup / spring): din ka low < pichle N din ka low MAGAR close us low se ooper (reclaim)
                   N {10, 20, 55} x uptrend shart {nahi, close > EMA200}; exit close > SMA5 (agle din open), stop 3 ATR, 10 din
 E3 VOLUME_THRUST : close > EMA200, din ka return >= +X, volume >= V x 20-din ausat, close din ki range ke ooper 25% mein
                   X {6, 8, 10}% x V {2, 3}; exit chandelier 22 / 3 ATR trailing, max 60 din
 E4 RS_PULLBACK  : coin ka L-din return universe ke top 20% mein AUR lagataar D din neeche close
                   L {60, 90} x D {2, 3, 4}; exit close > SMA5, stop 3 ATR, 10 din
RANDOM (20 seeds): har coin mein utni hi trades random din par, usi "pool" mein (E1/E3: uptrend din; E2 uptrend wali
   cell: uptrend din, warna top-100 din; E4: RS top-20% din) - taake sirf TRIGGER ka faida naapa jaye.
CELL PASS: n >= 100, PF >= 1.15, PF > random p95, OOS (2025+) PF > 1.   KHAYAL PLATEAU: 6 mein se 4+.
VALIDATION (har khayal ka beech wala cell, pehle se tay): bootstrap p5 PF > 1, top-10 trades hata kar PF > 1,
   4 time-folds mein 3+ musbat, kharcha 2x PF > 1.
PORTFOLIO: maujooda ICHI 60 / DIP 30 / CAPIT 10 (har trade 20%) vs ICHI 55 / DIP 25 / CAPIT 10 / NAYA 10.
   Pass: Sharpe +0.03 ya ziada, MaxDD 1.5% se ziada bura nahi.
RANKING: maujooda 4 systems + naye - akele (sleeve) CAGR / DD / Sharpe / PF / win / trades-per-hafta.
Natija: strategy_lab2_RESULTS.txt
"""
import warnings

import numpy as np
import pandas as pd

import portfolio_lab as P
from bot_core import fetch_full, norm, STABLES, chandelier

warnings.filterwarnings("ignore")
OUT = "strategy_lab2_RESULTS.txt"
N_RND = 20
OOS = pd.Timestamp("2025-01-01")
YEARS = list(range(2021, 2027))
GRID = {
    "E1": [(k, ex) for k in (1.5, 2.0, 2.5) for ex in ("SMA5", "SMA20")],
    "E2": [(n, up) for n in (10, 20, 55) for up in (False, True)],
    "E3": [(x, v) for x in (0.06, 0.08, 0.10) for v in (2.0, 3.0)],
    "E4": [(L, D) for L in (60, 90) for D in (2, 3, 4)],
}
CENTER = {"E1": (2.0, "SMA5"), "E2": (20, True), "E3": (0.08, 2.0), "E4": (90, 3)}
NAMES = {"E1": "BB REVERSION", "E2": "FAILED BREAKDOWN", "E3": "VOLUME THRUST", "E4": "RS PULLBACK"}


def pf_of(r):
    r = np.asarray(r, float)
    g, lo = r[r > 0].sum(), -r[r < 0].sum()
    return np.nan if len(r) == 0 else (np.inf if lo == 0 else g / lo)


def label(e, p):
    if e == "E1":
        return f"k{p[0]} {p[1]}"
    if e == "E2":
        return f"N{p[0]} {'uptrend' if p[1] else 'koi shart nahi'}"
    if e == "E3":
        return f"X{p[0]*100:.0f}% V{p[1]:.0f}"
    return f"L{p[0]} D{p[1]}"


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x)
        lines.append(x)

    if h4 is None:
        from data_fetcher import get_exchange, get_coin_list
        ex = get_exchange()
        coins = [s for s in get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:P.TOP_N]
        for must in ("ETH/USDT", "BTC/USDT"):
            if must not in coins:
                coins.insert(0, must)
        h4 = {}
        for k, sym in enumerate(coins, 1):
            try:
                df = fetch_full(ex, sym, "4h", P.H4_BARS)
                if df is not None and len(df) >= 6 * 250:
                    h4[sym] = norm(df)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)}")
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})")

    daily = {s: P.to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc_ok = bd > P.ema(bd, 50)
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    allowed = {day: set(row.dropna().nlargest(P.UNIVERSE).index) for day, row in dv.iterrows()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes_p = closes[closes.index >= start]
    weeks = (closes_p.index[-1] - closes_p.index[0]).days / 7

    in_uni = pd.DataFrame({s: [s in allowed.get(D, ()) for D in closes.index] for s in closes.columns}, index=closes.index)
    rs_top = {}
    for L in (60, 90):
        mom = (closes / closes.shift(L) - 1).where(in_uni)
        rs_top[L] = mom.rank(axis=1, pct=True) >= 0.8

    A = {}
    for s, d in daily.items():
        if s == "BTC/USDT" or len(d) < 260:
            continue
        ts, c, h, l = d["timestamp"], d["close"], d["high"], d["low"]
        mid, sd = c.rolling(20).mean(), c.rolling(20).std()
        a = dict(o=d["open"].to_numpy(float), h=h.to_numpy(float), l=l.to_numpy(float), c=c.to_numpy(float),
                 ts=ts.to_numpy(), stop=(c - 3.0 * P.atr_w(d)).to_numpy(), ret=c.pct_change().to_numpy(),
                 up=(c > P.ema(c, 200)).to_numpy() & (np.arange(len(c)) >= 200),
                 vrat=(d["volume"] / d["volume"].shift(1).rolling(20).mean()).to_numpy(),
                 ex5=(c > c.rolling(5).mean()).to_numpy(), ex20=(c > mid).to_numpy(),
                 mid=mid.to_numpy(), sd=sd.to_numpy(),
                 rngpos=((c - l) / (h - l).replace(0, np.nan)).to_numpy(),
                 trail=chandelier(d, 22, 3.0),
                 uni=np.array([s in allowed.get(D, ()) for D in ts]))
        for N in (10, 20, 55):
            pl = l.shift(1).rolling(N).min()
            a[f"fb{N}"] = ((l < pl) & (c > pl)).to_numpy()
        dn = (c < c.shift(1)).astype(int)
        for D in (2, 3, 4):
            a[f"dn{D}"] = (dn.rolling(D).sum() == D).to_numpy()
        for L in (60, 90):
            a[f"rs{L}"] = rs_top[L][s].reindex(ts).fillna(False).to_numpy(bool)
        a["uni"][:210] = False
        A[s] = a

    def nz(x):
        return np.nan_to_num(x, nan=0).astype(bool)

    def sig_and_pool(e, p, a):
        if e == "E1":
            k, _ = p
            m = a["uni"] & a["up"] & nz(a["c"] < a["mid"] - k * a["sd"])
            return m, a["uni"] & a["up"], -(a["c"] - a["mid"]) / a["sd"]
        if e == "E2":
            n, up = p
            base = a["uni"] & a["up"] if up else a["uni"]
            return base & nz(a[f"fb{n}"]), base, -a["ret"]
        if e == "E3":
            x, v = p
            m = a["uni"] & a["up"] & nz(a["ret"] >= x) & nz(a["vrat"] >= v) & nz(a["rngpos"] >= 0.75)
            return m, a["uni"] & a["up"], a["ret"]
        L, D = p
        base = a["uni"] & a[f"rs{L}"]
        return base & a[f"dn{D}"], base, -a["ret"]

    def run_one(e, p, a, i):
        if e == "E3":
            return P.sim_one(a["o"], a["h"], a["l"], a["c"], a["ts"], i, a["stop"][i], a["trail"], None, None, 60)
        if e == "E1" and p[1] == "SMA20":
            return P.sim_one(a["o"], a["h"], a["l"], a["c"], a["ts"], i, a["stop"][i], None, a["ex20"], None, 20)
        return P.sim_one(a["o"], a["h"], a["l"], a["c"], a["ts"], i, a["stop"][i], None, a["ex5"], None, 10)

    def sim(e, p, rng=None, cost_mult=1.0):
        old = (P.SLIP, P.FEE, P.STOP_SLIP)
        P.SLIP, P.FEE, P.STOP_SLIP = old[0] * cost_mult, old[1] * cost_mult, old[2] * cost_mult
        tr = []
        try:
            for s, a in A.items():
                m, pool, prio = sig_and_pool(e, p, a)
                idx = np.flatnonzero(m)
                if rng is not None:
                    pidx = np.flatnonzero(pool)
                    k = min(len(idx), len(pidx))
                    if k == 0:
                        continue
                    idx = np.sort(rng.choice(pidx, k, replace=False))
                busy = -1
                for i in idx:
                    if i <= busy:
                        continue
                    t = run_one(e, p, a, i)
                    if t:
                        pr = prio[i] if np.isfinite(prio[i]) else -9
                        t.update(sym=s, prio=float(pr))
                        tr.append(t)
                        busy = int(np.searchsorted(a["ts"], t["t_out"]))
        finally:
            P.SLIP, P.FEE, P.STOP_SLIP = old
        return [t for t in tr if pd.Timestamp(t["t_in"]) >= start]

    def validate(tr):
        r = np.array([t["ret"] for t in tr])
        tin = pd.to_datetime([t["t_in"] for t in tr])
        rng = np.random.default_rng(5)
        bs = np.percentile([pf_of(rng.choice(r, len(r))) for _ in range(1000)], 5)
        top = pf_of(np.sort(r)[:-10])
        order = np.argsort(tin.values)
        folds = [pf_of(r[part]) for part in np.array_split(order, 4)]
        return bs, top, folds

    emit("=" * 150)
    emit("STRATEGY LAB 2 - naye daily khayal (apna edge + validation + portfolio) aur tamam systems ki ranking")
    emit("=" * 150)
    emit(f"Coins: {len(A)} (BTC bahar) | {closes_p.index[0].date()} -> {closes_p.index[-1].date()} | top-{P.UNIVERSE} point-in-time | "
         f"random {N_RND} seeds | OOS {OOS.date()} se")

    chosen, verdict = {}, {}
    for e in GRID:
        emit(f"\n{'#' * 150}\n{e}) {NAMES[e]}\n{'#' * 150}")
        emit(f"{'Params':>22} | {'Trades':>6} | {'/hafta':>6} | {'Win%':>5} | {'Avg%':>6} | {'PF':>5} | {'RndP50':>6} | {'RndP95':>6} | "
             f"{'OOS':>5} | " + " | ".join(str(y) for y in YEARS) + " | Faisla")
        n_ok = 0
        for p in GRID[e]:
            tr = sim(e, p)
            r = np.array([t["ret"] for t in tr])
            rpf = [pf_of([t["ret"] for t in sim(e, p, np.random.default_rng(sd))]) for sd in range(N_RND)]
            p50, p95 = np.nanmedian(rpf), np.nanpercentile(rpf, 95)
            tin = pd.to_datetime([t["t_in"] for t in tr])
            oos = pf_of(r[tin >= OOS]) if len(r) else np.nan
            pf = pf_of(r)
            ok = len(r) >= 100 and pf >= 1.15 and pf > p95 and oos > 1
            n_ok += ok
            yr = {y: pf_of(r[tin.year == y]) for y in YEARS}
            emit(f"{label(e, p):>22} | {len(r):>6} | {len(r)/weeks:>6.2f} | {(r > 0).mean()*100 if len(r) else 0:>5.1f} | "
                 f"{r.mean()*100 if len(r) else 0:>+6.2f} | {pf:>5.2f} | {p50:>6.2f} | {p95:>6.2f} | {oos:>5.2f} | "
                 + " | ".join(f"{yr[y]:>4.2f}" for y in YEARS) + f" | {'PASS' if ok else '-'}")
            if p == CENTER[e]:
                chosen[e] = tr
        plateau = n_ok >= 4
        tr = chosen[e]
        if len(tr) >= 30:
            bs, top, folds = validate(tr)
            c2 = pf_of([t["ret"] for t in sim(e, CENTER[e], cost_mult=2.0)])
        else:
            bs, top, folds, c2 = np.nan, np.nan, [np.nan] * 4, np.nan
        val_ok = bs > 1 and top > 1 and sum(f > 1 for f in folds) >= 3 and c2 > 1
        verdict[e] = dict(plateau=n_ok, val=val_ok)
        emit(f"{e} plateau: {n_ok}/6 cells PASS -> {'HAAN' if plateau else 'NAHI'}")
        emit(f"{e} validation ({label(e, CENTER[e])}): bootstrap p5 {bs:.2f} | top-10 hata kar PF {top:.2f} | folds "
             + " / ".join(f"{f:.2f}" for f in folds) + f" | kharcha 2x PF {c2:.2f} -> {'PASS' if val_ok else 'FAIL'}")

    # ---------------- ranking + portfolio ----------------
    emit(f"\n{'#' * 150}\nRANKING + PORTFOLIO\n{'#' * 150}")
    T = {"ICHI": [t for t in P.trades_ichi(h4, allowed) if pd.Timestamp(t["t_in"]) >= start],
         "DIP": [t for t in P.trades_dip(daily, allowed, btc_ok) if pd.Timestamp(t["t_in"]) >= start],
         "DON": [t for t in P.trades_don(daily, allowed, btc_ok) if pd.Timestamp(t["t_in"]) >= start]}
    capit = []
    for s, a in A.items():
        m = a["uni"] & a["up"] & nz(a["ret"] <= -0.08) & nz(a["vrat"] >= 2.0)
        busy = -1
        for i in np.flatnonzero(m):
            if i <= busy:
                continue
            t = P.sim_one(a["o"], a["h"], a["l"], a["c"], a["ts"], i, a["stop"][i], None, a["ex5"], None, 10)
            if t:
                t.update(sym=s, prio=float(-a["ret"][i]))
                capit.append(t)
                busy = int(np.searchsorted(a["ts"], t["t_out"]))
    T["CAPIT"] = [t for t in capit if pd.Timestamp(t["t_in"]) >= start]
    for e, tr in chosen.items():
        T[e] = tr
    spec = {"ICHI": ("risk", 0.01), "DON": ("risk", 0.01)}
    curves = {k: P.portfolio(tr, closes_p, *spec.get(k, ("fixed", 0.20)))[0] for k, tr in T.items()}

    emit("Akele (har system apne paise par; ICHI/DON 1% risk, baqi har trade 20%):")
    emit(f"{'System':>22} | {'Trades':>6} | {'/hafta':>6} | {'Win%':>5} | {'PF':>5} | {'OOS PF':>6} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6}")
    rows = []
    for k, tr in T.items():
        r = np.array([t["ret"] for t in tr])
        tin = pd.to_datetime([t["t_in"] for t in tr])
        s = P.stats(curves[k])
        nm = {"ICHI": "Ichimoku 4H (live)", "DIP": "Dip Daily (live)", "DON": "Donchian (paper)",
              "CAPIT": "Capitulation (live)"}.get(k, f"{k} {NAMES.get(k, '')}")
        rows.append((s["sharpe"], f"{nm:>22} | {len(r):>6} | {len(r)/weeks:>6.2f} | {(r > 0).mean()*100:>5.1f} | {pf_of(r):>5.2f} | "
                     f"{pf_of(r[tin >= OOS]):>6.2f} | {s['cagr']*100:>+6.1f}% | {s['dd']*100:>6.1f}% | {s['sharpe']:>6.2f}"))
    for _, ln in sorted(rows, key=lambda x: -x[0]):
        emit(ln)

    base_w = {"ICHI": .6, "DIP": .3, "CAPIT": .1}
    mixes = {"BASE ICHI 60/DIP 30/CAPIT 10": base_w}
    for e in chosen:
        mixes[f"ICHI 55/DIP 25/CAPIT 10/{e} 10"] = {"ICHI": .55, "DIP": .25, "CAPIT": .1, e: .1}
    emit(f"\n{'Portfolio':>30} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'+mahine':>7} | {'Bura mah':>8} | "
         + " | ".join(str(y) for y in YEARS))
    port = {}
    for name, w in mixes.items():
        eq = P.mix(curves, w)
        s = P.stats(eq)
        port[name] = s
        yr = {d.year: v for d, v in s["yearly"].items()}
        emit(f"{name:>30} | {s['cagr']*100:>+6.1f}% | {s['dd']*100:>6.1f}% | {s['sharpe']:>6.2f} | {s['pos_months']:>6.0f}% | "
             f"{s['worst_month']*100:>+7.1f}% | " + " | ".join(f"{yr.get(y, np.nan)*100:>+4.0f}%" for y in YEARS))
    mo = pd.DataFrame({k: P.stats(v)["monthly"] for k, v in curves.items()})
    emit("\nMahana correlation:\n" + mo.corr().round(2).to_string())

    emit("\n" + "=" * 150 + "\nKHULASA (PASS = plateau 4/6 + validation + portfolio behtar)\n" + "=" * 150)
    b = port["BASE ICHI 60/DIP 30/CAPIT 10"]
    emit(f"BASE ICHI 60/DIP 30/CAPIT 10: CAGR {b['cagr']*100:+.1f}% | MaxDD {b['dd']*100:.1f}% | Sharpe {b['sharpe']:.2f}")
    for e in GRID:
        x = port[f"ICHI 55/DIP 25/CAPIT 10/{e} 10"]
        helps = x["sharpe"] >= b["sharpe"] + 0.03 and x["dd"] >= b["dd"] - 0.015
        ok = verdict[e]["plateau"] >= 4 and verdict[e]["val"] and helps
        emit(f"{e} {NAMES[e]:>17}: plateau {verdict[e]['plateau']}/6 | validation {'PASS' if verdict[e]['val'] else 'FAIL'} | "
             f"portfolio Sharpe {x['sharpe']:.2f} (base {b['sharpe']:.2f}), DD {x['dd']*100:.1f}% -> {'*** PASS ***' if ok else 'FAIL'}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
