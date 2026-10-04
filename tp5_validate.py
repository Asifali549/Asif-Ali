"""
TP5 VALIDATION - Winrate Lab ka sab se acha natija (Ichimoku CE 16/4 + TP 5%) asal mein mazboot hai?
=====================================================================================================
CLAUDE.md ke "pass" usool (2026-10-04):
 A) Padosi grid: CE mult 3.5/4/4.5/5 x TP 4/5/6/7% - sab acha hona chahiye (sirf ek cell nahi).
 B) Coin-universe sensitivity: 8 dafa 20% coins random hata kar - PF, OOS, CAGR, DD (Capitulation isi mein fail hua tha).
 C) Random-entry control grid ke markaz (CE4 TP5) par 20 seeds.
 D) 4 time-folds (har ek ~1.5 saal) PF > 1; bootstrap p5; top-10 hata kar; 2x/3x kharcha.
 E) Portfolio 2% risk (max 10, cap 20%) BASE (live CE5.5 + 3R) 1% aur 1.5% ke muqable; random portfolio p95 Sharpe.
 F) ICHI(CE4 TP5) 60 + DIP 40 portfolio vs live ICHI 60 + DIP 40.
PASS = A mein 80%+ cells PF>1.5 aur random p95 se behtar, B mein 8/8 PF>1.5 aur OOS>1, D sab folds >1, p5 >1.2.
Natija: tp5_validate_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema, rsi, atr_w, to_daily, portfolio, stats, mix, TOP_N, H4_BARS, UNIVERSE
from winrate_lab import sim, prep_ichi, run_ichi, tstats, pf_of

OUT = "tp5_validate_RESULTS.txt"


def dip_trades(daily, allowed, btc_ok, start):
    out = []
    for sym, d in daily.items():
        c = d["close"]
        e50, e200 = ema(c, 50), ema(c, 200)
        trend = ((c > e200) & (e50 > e200)).to_numpy() & btc_ok.reindex(d["timestamp"]).fillna(False).to_numpy(bool)
        trend[:200] = False
        r = rsi(c, 3).to_numpy()
        sig = trend & (r < 10)
        stop = (c - 3.0 * atr_w(d)).to_numpy()
        ex = (c > c.rolling(5).mean()).to_numpy()
        o, h, l, cc = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        ts = d["timestamp"].to_numpy()
        for i in np.where(sig)[0]:
            if sym not in allowed.get(d["timestamp"].iloc[i], ()) or d["timestamp"].iloc[i] < start:
                continue
            t = sim(o, h, l, cc, ts, i, stop[i], None, ex, 10, {})
            if t:
                t.update(sym=sym, prio=-r[i])
                out.append(t)
    return out


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
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

    emit("=" * 110)
    emit("TP5 VALIDATION - Ichimoku 4H: CE 16/4 + TP 5% (win-rate version) ki tasdeeq")
    emit("=" * 110)
    emit(f"Coins: {len(h4)} | {closes.index[0].date()} -> {closes.index[-1].date()} | top-{UNIVERSE} liquid | OOS = 2025+")
    emit("win% = +0.5% se ziada. Portfolio 2% risk, max 10, coin cap 20% (jab tak alag na likha ho).")

    data = prep_ichi(h4, allowed, start)
    BASE = {"tpR": 3.0}
    CAND = {"ce": 4.0, "tp": 0.05}

    # ---------------- A) grid
    emit("\n" + "#" * 110)
    emit("# A) PADOSI GRID: CE mult x TP% (har cell: win% / PF / OOS PF / CAGR@2% / MaxDD@2% / Sharpe) + random PF p95")
    emit("#" * 110)
    emit(f"{'cell':>10} | {'win%':>5} | {'PF':>5} | {'OOS':>5} | {'RND p95':>7} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6}")
    rnd_sets = [prep_ichi(h4, allowed, start, rng=np.random.default_rng(100 + s)) for s in range(8)]
    good, total = 0, 0
    for m in (3.5, 4.0, 4.5, 5.0):
        for tp in (0.04, 0.05, 0.06, 0.07):
            cfg = {"ce": m, "tp": tp}
            tr = run_ichi(data, cfg)
            s = tstats(tr)
            rp = [pf_of([t["ret"] for t in run_ichi(rd, cfg)]) for rd in rnd_sets]
            p95 = np.percentile(rp, 95)
            eq, _ = portfolio(tr, closes, "risk", 0.02)
            p = stats(eq)
            ok = s["pf"] > 1.5 and s["pf"] > p95 and s["oos"] > 1
            good += ok
            total += 1
            emit(f"CE{m:g} TP{int(tp*100):>2} | {s['win']:>5.1f} | {s['pf']:>5.2f} | {s['oos']:>5.2f} | {p95:>7.2f} | "
                 f"{p['cagr']*100:>+6.1f}% | {p['dd']*100:>6.1f}% | {p['sharpe']:>6.2f} {'OK' if ok else '--'}")
    emit(f"\nGrid: {good}/{total} cells PF>1.5 + random p95 se behtar + OOS>1  -> {'PASS' if good >= 0.8 * total else 'FAIL'}")

    # ---------------- C) random control at centre, 20 seeds
    emit("\n" + "#" * 110)
    emit("# C) RANDOM-ENTRY control (CE4 TP5, 20 seeds) - trade level aur portfolio")
    emit("#" * 110)
    cand = run_ichi(data, CAND)
    sc = tstats(cand)
    eqc, _ = portfolio(cand, closes, "risk", 0.02)
    pc = stats(eqc)
    rpf, rwin, rsh, rcg = [], [], [], []
    for s in range(20):
        rd = prep_ichi(h4, allowed, start, rng=np.random.default_rng(500 + s))
        tr = run_ichi(rd, CAND)
        r = np.array([t["ret"] for t in tr])
        rpf.append(pf_of(r))
        rwin.append((r > 0.005).mean() * 100)
        p = stats(portfolio(tr, closes, "risk", 0.02)[0])
        rsh.append(p["sharpe"])
        rcg.append(p["cagr"] * 100)
    emit(f"Asli : win {sc['win']:.1f}% | PF {sc['pf']:.2f} | Sharpe {pc['sharpe']:.2f} | CAGR {pc['cagr']*100:+.1f}%")
    emit(f"Random p50: win {np.median(rwin):.1f}% | PF {np.median(rpf):.2f} | Sharpe {np.median(rsh):.2f} | CAGR {np.median(rcg):+.1f}%")
    emit(f"Random p95: win {np.percentile(rwin,95):.1f}% | PF {np.percentile(rpf,95):.2f} | Sharpe {np.percentile(rsh,95):.2f} | CAGR {np.percentile(rcg,95):+.1f}%")
    emit(f"-> PF > random p95: {'HAAN' if sc['pf'] > np.percentile(rpf, 95) else 'NAHI'} | Sharpe > random p95: "
         f"{'HAAN' if pc['sharpe'] > np.percentile(rsh, 95) else 'NAHI'}")

    # ---------------- D) folds / bootstrap / cost
    emit("\n" + "#" * 110)
    emit("# D) 4 TIME-FOLDS, bootstrap, top-10 hata kar, kharcha 2x/3x  (BASE vs CE4 TP5)")
    emit("#" * 110)
    t0, t1 = closes.index[0], closes.index[-1]
    edges = [t0 + (t1 - t0) * k / 4 for k in range(5)]
    rng = np.random.default_rng(3)
    for name, cfg in (("BASE", BASE), ("CE4 TP5", CAND)):
        tr = run_ichi(data, cfg)
        fl = []
        for k in range(4):
            r = [t["ret"] for t in tr if edges[k] <= pd.Timestamp(t["t_in"]) < edges[k + 1]]
            fl.append(pf_of(r))
        r = np.array([t["ret"] for t in tr])
        boots = [pf_of(rng.choice(r, len(r))) for _ in range(3000)]
        c2 = tstats(run_ichi(data, cfg, cost=2.0))
        c3 = tstats(run_ichi(data, cfg, cost=3.0))
        emit(f"{name:>8} | folds PF " + " / ".join(f"{v:.2f}" for v in fl) + f" | boot p5 {np.percentile(boots,5):.2f}"
             f" | -top10 {pf_of(np.sort(r)[:-10]):.2f} | 2x PF {c2['pf']:.2f} win {c2['win']:.0f}% | 3x PF {c3['pf']:.2f} win {c3['win']:.0f}%")

    # ---------------- B) universe sensitivity
    emit("\n" + "#" * 110)
    emit("# B) COIN-UNIVERSE: 8 dafa 20% coins random hata kar (BASE 1% risk vs CE4 TP5 2% risk)")
    emit("#" * 110)
    emit(f"{'run':>4} | {'BASE PF':>7} | {'BASE OOS':>8} | {'BASE CAGR':>9} | {'BASE DD':>7} || {'TP5 win':>7} | {'TP5 PF':>6} | {'TP5 OOS':>7} | {'TP5 CAGR':>8} | {'TP5 DD':>7} | {'Sharpe':>6}")
    syms = [x["sym"] for x in data]
    passed = 0
    for k in range(8):
        keep = set(np.random.default_rng(900 + k).choice(syms, size=int(len(syms) * 0.8), replace=False)) | {"BTC/USDT", "ETH/USDT"}
        sub = [x for x in data if x["sym"] in keep]
        tb, tc = run_ichi(sub, BASE), run_ichi(sub, CAND)
        sb, sc2 = tstats(tb), tstats(tc)
        pb = stats(portfolio(tb, closes, "risk", 0.01)[0])
        pc2 = stats(portfolio(tc, closes, "risk", 0.02)[0])
        ok = sc2["pf"] > 1.5 and sc2["oos"] > 1
        passed += ok
        emit(f"{k+1:>4} | {sb['pf']:>7.2f} | {sb['oos']:>8.2f} | {pb['cagr']*100:>+8.1f}% | {pb['dd']*100:>6.1f}% || "
             f"{sc2['win']:>6.1f}% | {sc2['pf']:>6.2f} | {sc2['oos']:>7.2f} | {pc2['cagr']*100:>+7.1f}% | {pc2['dd']*100:>6.1f}% | {pc2['sharpe']:>6.2f} {'OK' if ok else '--'}")
    emit(f"Universe: {passed}/8 PASS")

    # ---------------- E/F) portfolio comparisons
    emit("\n" + "#" * 110)
    emit("# E/F) PORTFOLIO: ICHI akela aur ICHI 60 / DIP 40 (DIP 20% size) - live vs naya")
    emit("#" * 110)
    tb = run_ichi(data, BASE)
    dip = dip_trades(daily, allowed, btc_ok, start)
    curves = {
        "ICHI live 1%": portfolio(tb, closes, "risk", 0.01)[0],
        "ICHI live 1.5%": portfolio(tb, closes, "risk", 0.015)[0],
        "ICHI TP5 1.5%": portfolio(cand, closes, "risk", 0.015)[0],
        "ICHI TP5 2%": portfolio(cand, closes, "risk", 0.02)[0],
        "DIP 20%": portfolio(dip, closes, "fixed", 0.20)[0],
    }
    curves["live: ICHI60/DIP40"] = mix(curves, {"ICHI live 1%": .6, "DIP 20%": .4})
    curves["naya: TP5 2% 60/DIP40"] = mix(curves, {"ICHI TP5 2%": .6, "DIP 20%": .4})
    curves["naya: TP5 2% 80/DIP20"] = mix(curves, {"ICHI TP5 2%": .8, "DIP 20%": .2})
    emit(f"{'Portfolio':>24} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'+mah%':>5} | {'bura mah':>8} | {'lamba DD din':>12}")
    for n, eq in curves.items():
        p = stats(eq)
        emit(f"{n:>24} | {p['cagr']*100:>+6.1f}% | {p['dd']*100:>6.1f}% | {p['sharpe']:>6.2f} | {p['pos_months']:>5.0f} | "
             f"{p['worst_month']*100:>7.1f}% | {p['long_dd']:>12}")
    emit("\nSaal-war:")
    yrs = sorted({y.year for eq in curves.values() for y in stats(eq)["yearly"].index})
    emit(f"{'Portfolio':>24} | " + " | ".join(f"{y:>6}" for y in yrs))
    for n, eq in curves.items():
        yr = {y.year: v for y, v in stats(eq)["yearly"].items()}
        emit(f"{n:>24} | " + " | ".join(f"{yr.get(y, np.nan)*100:>+5.0f}%" for y in yrs))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
