"""
DIP v2 VALIDATION - Dip Daily ka win-rate version (RSI3<7, exit close>SMA3, TP +5%) mazboot hai?
=================================================================================================
Live Dip: RSI3<10, exit close>SMA5, koi TP nahi. Winrate Lab (2026-10-04) mein naya: win 81%, PF 4.04,
CAGR 10.8%, DD -17.6% (live 12.3% / -26.9%) - magar 18 variants mein se chuna, isliye poori tasdeeq:
 A) Grid: RSI 5/6/7/8/9/10 x SMA 3/5 x TP none/4/5/6 (48 cells) - PF, OOS, win, CAGR, DD, Sharpe.
 B) Random control: same coins, same uptrend+BTC din, RANDOM din par entry (RSI ke baghair), wohi exit x20.
 C) Folds (4), saal-war PF, bootstrap p5, top-10 hata kar, kharcha 2x/3x.
 D) Universe: 8 dafa 20% coins hata kar (live vs naya).
 E) Portfolio: Ichimoku TP5 (2% risk) ke sath 80/20 aur 60/40 - live Dip vs naya Dip.
PASS = grid padosi (RSI 6-8, SMA3, TP 4-6) sab PF>2 aur OOS>1.5, random p95 se behtar, universe 8/8 PF>2 + OOS>1,
       folds sab >1, boot p5 > 1.5.
Natija: dip_v2_validate_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema, rsi, atr_w, to_daily, portfolio, stats, mix, TOP_N, H4_BARS, UNIVERSE
from winrate_lab import sim, prep_ichi, run_ichi, tstats, pf_of

OUT = "dip_v2_validate_RESULTS.txt"


def prep_dip(daily, allowed, btc_ok, start):
    out = []
    for sym, d in daily.items():
        c = d["close"]
        e50, e200 = ema(c, 50), ema(c, 200)
        trend = ((c > e200) & (e50 > e200)).to_numpy() & btc_ok.reindex(d["timestamp"]).fillna(False).to_numpy(bool)
        trend[:200] = False
        ok = np.array([sym in allowed.get(t, ()) for t in d["timestamp"]]) & (d["timestamp"] >= start).to_numpy()
        out.append(dict(sym=sym, ts=d["timestamp"].to_numpy(), o=d["open"].to_numpy(float), h=d["high"].to_numpy(float),
                        l=d["low"].to_numpy(float), c=c.to_numpy(float), r=rsi(c, 3).to_numpy(),
                        stop=(c - 3.0 * atr_w(d)).to_numpy(), elig=trend & ok,
                        sma={k: (c > c.rolling(k).mean()).to_numpy() for k in (3, 5)}))
    return out


def run_dip(P, rlim, sma, tp, keep=None, cost=1.0, rng=None):
    out = []
    for x in P:
        if keep is not None and x["sym"] not in keep:
            continue
        sig = x["elig"] & (x["r"] < rlim)
        idx = np.where(sig)[0]
        if rng is not None:
            pool = np.where(x["elig"])[0]
            if len(idx) == 0 or len(pool) == 0:
                continue
            idx = np.sort(rng.choice(pool, size=min(len(idx), len(pool)), replace=False))
        cfg = {"tp": tp} if tp else {}
        for i in idx:
            t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, x["stop"][i], None, x["sma"][sma], 10, cfg, cost)
            if t:
                t.update(sym=x["sym"], prio=-x["r"][i] if np.isfinite(x["r"][i]) else -9)
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
    yrs_span = (closes.index[-1] - closes.index[0]).days / 365.25

    emit("=" * 110)
    emit("DIP v2 VALIDATION - live (RSI<10, SMA5) vs naya (RSI<7, SMA3, TP5%)")
    emit("=" * 110)
    emit(f"Coins: {len(daily)} | {closes.index[0].date()} -> {closes.index[-1].date()} | top-{UNIVERSE} | OOS = 2025+ | "
         "win = +0.5% se ziada | portfolio: har trade 20%, max 10")
    P = prep_dip(daily, allowed, btc_ok, start)
    LIVE, NEW = (10, 5, None), (7, 3, 0.05)

    # ---------------- A) grid
    emit("\n" + "#" * 110)
    emit("# A) GRID: RSI x SMA x TP")
    emit("#" * 110)
    emit(f"{'cell':>18} | {'n':>4} | {'/hafta':>6} | {'win%':>5} | {'PF':>5} | {'OOS':>5} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6}")
    neigh_ok, neigh_n = 0, 0
    for rl in (5, 6, 7, 8, 9, 10):
        for sm in (3, 5):
            for tp in (None, 0.04, 0.05, 0.06):
                tr = run_dip(P, rl, sm, tp)
                s = tstats(tr)
                p = stats(portfolio(tr, closes, "fixed", 0.20)[0])
                nb = rl in (6, 7, 8) and sm == 3 and tp in (0.04, 0.05, 0.06)
                ok = s["pf"] > 2 and s["oos"] > 1.5
                if nb:
                    neigh_n += 1
                    neigh_ok += ok
                tag = " <- LIVE" if (rl, sm, tp) == LIVE else (" <- NAYA" if (rl, sm, tp) == NEW else "")
                nm = f"RSI<{rl} SMA{sm} " + (f"TP{int(tp*100)}" if tp else "-")
                emit(f"{nm:>18} | {s['n']:>4} | {s['n']/yrs_span/52:>6.2f} | {s['win']:>5.1f} | {s['pf']:>5.2f} | {s['oos']:>5.2f} | "
                     f"{p['cagr']*100:>+6.1f}% | {p['dd']*100:>6.1f}% | {p['sharpe']:>6.2f}{' *' if nb else ''}{tag}")
    emit(f"\nPadosi (* RSI 6-8, SMA3, TP 4-6): {neigh_ok}/{neigh_n} PF>2 aur OOS>1.5 -> {'PASS' if neigh_ok == neigh_n else 'FAIL'}")

    # ---------------- B) random
    emit("\n" + "#" * 110)
    emit("# B) RANDOM control: same uptrend+BTC din, random entry (RSI ke baghair), wohi exit, 20 seeds")
    emit("#" * 110)
    for name, (rl, sm, tp) in (("LIVE", LIVE), ("NAYA", NEW)):
        s = tstats(run_dip(P, rl, sm, tp))
        pfs, wins = [], []
        for k in range(20):
            tr = run_dip(P, rl, sm, tp, rng=np.random.default_rng(700 + k))
            r = np.array([t["ret"] for t in tr])
            pfs.append(pf_of(r))
            wins.append((r > 0.005).mean() * 100)
        emit(f"{name}: asli win {s['win']:.1f}% PF {s['pf']:.2f} | random win p50 {np.median(wins):.1f}% PF p50 {np.median(pfs):.2f} "
             f"p95 {np.percentile(pfs, 95):.2f} -> {'HAAN behtar' if s['pf'] > np.percentile(pfs, 95) else 'NAHI'}")

    # ---------------- C) folds etc
    emit("\n" + "#" * 110)
    emit("# C) FOLDS, saal-war, bootstrap, top-10 hata kar, kharcha")
    emit("#" * 110)
    t0, t1 = closes.index[0], closes.index[-1]
    edges = [t0 + (t1 - t0) * k / 4 for k in range(5)]
    rng = np.random.default_rng(11)
    years = list(range(2021, 2027))
    emit(f"{'':>5} | {'fold PFs':>23} | " + " | ".join(f"{y:>5}" for y in years) + " | boot p5 | -top10 | 2x PF | 3x PF")
    for name, (rl, sm, tp) in (("LIVE", LIVE), ("NAYA", NEW)):
        tr = run_dip(P, rl, sm, tp)
        fl = [pf_of([t["ret"] for t in tr if edges[k] <= pd.Timestamp(t["t_in"]) < edges[k + 1]]) for k in range(4)]
        yr = []
        for y in years:
            r = [t["ret"] for t in tr if pd.Timestamp(t["t_in"]).year == y]
            yr.append(pf_of(r) if len(r) >= 3 else np.nan)
        r = np.array([t["ret"] for t in tr])
        boots = [pf_of(rng.choice(r, len(r))) for _ in range(3000)]
        c2 = tstats(run_dip(P, rl, sm, tp, cost=2.0))["pf"]
        c3 = tstats(run_dip(P, rl, sm, tp, cost=3.0))["pf"]
        emit(f"{name:>5} | " + " / ".join(f"{v:.2f}" for v in fl) + " | " + " | ".join(f"{v:>5.2f}" for v in yr) +
             f" | {np.percentile(boots, 5):>7.2f} | {pf_of(np.sort(r)[:-10]):>6.2f} | {c2:>5.2f} | {c3:>5.2f}")

    # ---------------- D) universe
    emit("\n" + "#" * 110)
    emit("# D) UNIVERSE: 8 dafa 20% coins hata kar")
    emit("#" * 110)
    emit(f"{'run':>3} | {'LIVE win':>8} | {'PF':>5} | {'OOS':>5} | {'CAGR':>7} | {'DD':>7} || {'NAYA win':>8} | {'PF':>5} | {'OOS':>5} | {'CAGR':>7} | {'DD':>7}")
    syms = [x["sym"] for x in P]
    passed = 0
    for k in range(8):
        keep = set(np.random.default_rng(1300 + k).choice(syms, size=int(len(syms) * 0.8), replace=False)) | {"BTC/USDT"}
        row = []
        for (rl, sm, tp) in (LIVE, NEW):
            tr = run_dip(P, rl, sm, tp, keep=keep)
            s = tstats(tr)
            p = stats(portfolio(tr, closes, "fixed", 0.20)[0])
            row.append((s, p))
        (sl, pl), (sn, pn) = row
        ok = sn["pf"] > 2 and sn["oos"] > 1
        passed += ok
        emit(f"{k+1:>3} | {sl['win']:>7.1f}% | {sl['pf']:>5.2f} | {sl['oos']:>5.2f} | {pl['cagr']*100:>+6.1f}% | {pl['dd']*100:>6.1f}% || "
             f"{sn['win']:>7.1f}% | {sn['pf']:>5.2f} | {sn['oos']:>5.2f} | {pn['cagr']*100:>+6.1f}% | {pn['dd']*100:>6.1f}% {'OK' if ok else '--'}")
    emit(f"Universe (naya): {passed}/8 PASS")

    # ---------------- E) portfolio
    emit("\n" + "#" * 110)
    emit("# E) PORTFOLIO - Ichimoku TP5 (CE4, 2% risk) ke sath")
    emit("#" * 110)
    ichi = run_ichi(prep_ichi(h4, allowed, start), {"ce": 4.0, "tp": 0.05})
    curves = {"ICHI TP5 2%": portfolio(ichi, closes, "risk", 0.02)[0],
              "DIP live": portfolio(run_dip(P, *LIVE), closes, "fixed", 0.20)[0],
              "DIP naya": portfolio(run_dip(P, *NEW), closes, "fixed", 0.20)[0]}
    for a, b in ((80, 20), (70, 30), (60, 40)):
        curves[f"TP5 {a} / DIP live {b}"] = mix(curves, {"ICHI TP5 2%": a / 100, "DIP live": b / 100})
        curves[f"TP5 {a} / DIP naya {b}"] = mix(curves, {"ICHI TP5 2%": a / 100, "DIP naya": b / 100})
    emit(f"{'Portfolio':>24} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'+mah%':>5} | {'bura mah':>8}")
    for n, eq in curves.items():
        p = stats(eq)
        emit(f"{n:>24} | {p['cagr']*100:>+6.1f}% | {p['dd']*100:>6.1f}% | {p['sharpe']:>6.2f} | {p['pos_months']:>5.0f} | {p['worst_month']*100:>7.1f}%")
    emit("\nSaal-war:")
    ys = sorted({y.year for eq in curves.values() for y in stats(eq)["yearly"].index})
    emit(f"{'Portfolio':>24} | " + " | ".join(f"{y:>6}" for y in ys))
    for n, eq in curves.items():
        yr = {y.year: v for y, v in stats(eq)["yearly"].items()}
        emit(f"{n:>24} | " + " | ".join(f"{yr.get(y, np.nan)*100:>+5.0f}%" for y in ys))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
