"""
DONCHIAN BREAKOUT VALIDATION - koi indicator nahi (user ka validation plan, 2026-10-01)
=====================================================================================
Base: DAILY close > pichle 20 din ka high -> agle din open; stop 3.0 x ATR(14), phir ATR trailing 3.0x
(Model B, pichle research jaisa). Do version: FILTER BAGHAIR aur BTC>EMA50 (sirf entry din par).
109 coins, Oct 2020 -> Sep 2026, fee 0.1% + slip 0.05% har taraf + stop slip 0.25%, gap fill.
TEST 1 coin-by-coin | TEST 2 equal-weight vs pooled (concentration) | TEST 3 regimes |
TEST 4 walk-forward (N sirf pichle data se chuna) | TEST 5 N 15/18/20/22/25/30 | TEST 6 random har jagah |
TEST 7 exits (entry N=20 fixed, stop 3xATR fixed): TP 1R/1.5R/2R/3R, ATR trailing, Donchian 10-din low exit
Natija: donchian_validate_RESULTS.txt
"""
import numpy as np
import pandas as pd
from numba import njit

import donchian_research as R
from bot_core import fetch_full, norm, STABLES, FEE, SLIP, STOP_SLIP
from portfolio_lab import ema, to_daily, portfolio, stats

OUT = "donchian_validate_RESULTS.txt"
M_STOP = 3.0
NS = [15, 18, 20, 22, 25, 30]
WF = [("2020-10-01", "2022-12-31", "2023"), ("2020-10-01", "2023-12-31", "2024"),
      ("2021-01-01", "2024-12-31", "2025"), ("2022-01-01", "2025-12-31", "2026")]


@njit(cache=True)
def sim_exit_sig(o, h, l, c, atr, sig_idx, m, exit_sig, max_hold, slip, fee, stop_slip):
    """Initial ATR stop + exit signal (close < pichle 10 din ka low) -> agle din open."""
    n = len(o)
    k = len(sig_idx)
    e_out = np.empty(k, np.int64)
    x_out = np.empty(k, np.int64)
    r_out = np.empty(k)
    rk_out = np.empty(k)
    why = np.empty(k, np.int64)
    cnt = 0
    busy = -1
    for q in range(k):
        i = sig_idx[q]
        if i <= busy or i + 1 >= n or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        e = i + 1
        entry = o[e] * (1 + slip)
        st = entry - m * atr[i]
        if st <= 0:
            continue
        last = min(e + max_hold, n - 1)
        px = -1.0
        code = 3
        j = last
        for jj in range(e, last + 1):
            j = jj
            if l[j] <= st:
                px = min(st * (1 - stop_slip), o[j])
                code = 0
                break
            if exit_sig[j] and j + 1 < n:
                j = j + 1
                px = o[j]
                code = 2
                break
        if px < 0:
            px = c[last]
        e_out[cnt] = e
        x_out[cnt] = j
        r_out[cnt] = px * (1 - slip) * (1 - fee) / (entry * (1 + fee)) - 1
        rk_out[cnt] = (entry - (entry - m * atr[i])) / entry
        why[cnt] = code
        cnt += 1
        busy = j
    return e_out[:cnt], x_out[:cnt], r_out[:cnt], rk_out[:cnt], why[:cnt]


def run_donch_exit(U, sig, rng=None, allow=None):
    rows = []
    for s in U.syms:
        a = U.arr[s]
        d = U.frames[s]
        ex = (d["close"] < d["low"].shift(1).rolling(10).min()).to_numpy()
        idx = sig[s]
        if rng is not None:
            pool = np.arange(a["warm"], len(a["o"]) - 1)
            if allow is not None:
                pool = pool[allow[s][pool]]
            k = min(len(idx), len(pool))
            if k == 0:
                continue
            idx = np.sort(rng.choice(pool, size=k, replace=False)).astype(np.int64)
        if len(idx) == 0:
            continue
        e, x, r, rk, w = sim_exit_sig(a["o"], a["h"], a["l"], a["c"], a["atr"], idx, M_STOP, ex, 365, SLIP, FEE, STOP_SLIP)
        if len(e):
            rows.append(pd.DataFrame({"sym": s, "t_in": a["ts"][e], "t_out": a["ts"][x], "ret": r, "risk": rk, "why": w,
                                      "entry_px": a["o"][e] * (1 + SLIP)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------- helpers
pf_of = R.pf_of
fmt = R.fmt


def clip_pf(x, cap=5.0):
    return cap if np.isinf(x) else x


def seq_equity(rets, risks):
    """Ek coin ki trades tarteeb se: 1% risk (max 20% position) - closed-trade equity."""
    frac = np.minimum(0.01 / np.maximum(risks, 1e-6), 0.20)
    eq = np.cumprod(1 + frac * rets)
    dd = (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min() if len(eq) else 0
    return (eq[-1] - 1 if len(eq) else 0.0), min(dd, 0.0)


def block(tr, rt):
    r, rr = tr["ret"].to_numpy(), (rt["ret"].to_numpy() if len(rt) else np.array([]))
    return {"n": len(r), "pf": pf_of(r), "rpf": pf_of(rr) if len(rr) else np.nan, "win": (r > 0).mean() * 100 if len(r) else 0,
            "exp": r.mean() * 100 if len(r) else 0, "rexp": rr.mean() * 100 if len(rr) else np.nan}


def bline(name, b, extra=""):
    imp = b["pf"] - b["rpf"] if np.isfinite(b["rpf"]) else np.nan
    return (f"{name:>30} | {b['n']:>6} | {b['win']:>5.1f} | {fmt(b['pf']):>5} | {fmt(b['rpf']):>5} | {fmt(imp):>5} | "
            f"{b['exp']:>+6.2f} | {fmt(b['rexp']):>6}{extra}")


BHEAD = f"{'':>30} | {'Trades':>6} | {'Win%':>5} | {'PF':>5} | {'RndPF':>5} | {'PF+':>5} | {'Exp%':>6} | {'RndExp':>6}"


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x)
        lines.append(x)

    if h4 is None:
        from data_fetcher import get_exchange, get_coin_list
        ex = get_exchange()
        coins = [s for s in get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:R.TOP_N]
        if "BTC/USDT" not in coins:
            coins.insert(0, "BTC/USDT")
        h4 = {}
        for k, sym in enumerate(coins, 1):
            try:
                df = fetch_full(ex, sym, "4h", R.H4_BARS)
                if df is not None and len(df) >= 6 * 250:
                    h4[sym] = norm(df).reset_index(drop=True)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)}")
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})")

    daily = {s: to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = max(pd.Timestamp("2020-10-01"), closes.index[0] + pd.Timedelta(days=30))
    closes = closes[closes.index >= start]
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    b50, b200 = ema(bd, 50), ema(bd, 200)
    btc_reg = pd.Series(np.where((bd > b200) & (b50 > b200), "bull", np.where((bd < b200) & (b50 < b200), "bear", "sideways")),
                        index=bd.index)
    bv = bd.pct_change().rolling(30).std()
    btc_vol = pd.Series(np.where(bv > bv.median(), "high-vol", "low-vol"), index=bd.index)
    btc_ok = bd > b50

    U = R.Universe(daily, warm=60)
    allow = {s: btc_ok.reindex(d["timestamp"]).fillna(False).to_numpy(bool) for s, d in daily.items()}

    def sigs(N, filt):
        s = U.signals(N)
        return {k: v[allow[k][v]] for k, v in s.items()} if filt else s

    def trades(N, filt, exit_kind="trail", tp=0.0):
        sig = sigs(N, filt)
        al = allow if filt else None
        if exit_kind == "donch":
            tr = run_donch_exit(U, sig)
            rt = run_donch_exit(U, sig, rng=np.random.default_rng(7), allow=al)
        else:
            trail = exit_kind == "trail"
            mh = 365 if trail else 120
            tr = U.run(sig, M_STOP, tp, trail, mh)
            rt = U.run(sig, M_STOP, tp, trail, mh, rng=np.random.default_rng(7), allow=al)
        tr = tr[tr["t_in"] >= start].reset_index(drop=True)
        rt = rt[rt["t_in"] >= start].reset_index(drop=True)
        return tr, rt

    emit("=" * 120)
    emit("DONCHIAN BREAKOUT VALIDATION (Daily, N=20, ATR stop 3x + ATR trailing 3x) - koi indicator nahi")
    emit("=" * 120)
    emit(f"Coins: {len(daily)} | Period: {start.date()} -> {closes.index[-1].date()} | kharcha: fee {FEE*100:.2f}% + slip "
         f"{SLIP*100:.2f}% har taraf + stop slip {STOP_SLIP*100:.2f}%")
    emit("PF+ = PF - Random PF | Random = utni hi entries, wahi coins/period/stop/exit, random din (filter version mein sirf BTC>EMA50 din)")

    base = {"BAGHAIR filter": trades(20, False), "BTC>EMA50": trades(20, True)}
    checks = {}

    # ---------------- TEST 1 + 2 ----------------
    for tag, (tr, rt) in base.items():
        emit(f"\n\n{'#' * 120}\nTEST 1 - COIN-BY-COIN | N=20 {tag}\n{'#' * 120}")
        rows = []
        for s, g in tr.groupby("sym"):
            g = g.sort_values("t_in")
            r = g["ret"].to_numpy()
            net, dd = seq_equity(r, g["risk"].to_numpy())
            gains, losses = r[r > 0].sum(), -r[r < 0].sum()
            rows.append({"coin": s.split("/")[0], "n": len(r), "win": (r > 0).mean() * 100, "pf": pf_of(r), "net": net * 100,
                         "dd": dd * 100, "exp": r.mean() * 100, "g": gains, "l": losses, "sum": r.sum()})
        cdf = pd.DataFrame(rows)
        ok = cdf[cdf["n"] >= 5].copy()
        ok["pfc"] = ok["pf"].map(clip_pf)
        emit(f"{'Coin':>10} | {'Trades':>6} | {'Win%':>5} | {'PF':>5} | {'NetRet%':>8} | {'MaxDD%':>7} | {'Exp%':>6}")
        for _, x in cdf.sort_values("pf", ascending=False).iterrows():
            emit(f"{x['coin']:>10} | {x['n']:>6} | {x['win']:>5.1f} | {fmt(x['pf']):>5} | {x['net']:>+8.1f} | {x['dd']:>7.1f} | {x['exp']:>+6.2f}")
        med, mean = ok["pfc"].median(), ok["pfc"].mean()
        n_gt1, n_gt12, n_lt1 = int((ok["pf"] > 1).sum()), int((ok["pf"] > 1.2).sum()), int((ok["pf"] < 1).sum())
        emit(f"\nKHULASA ({len(ok)} coins jin ki >=5 trades; PF inf ko 5.0 mana):")
        emit(f"   Median coin PF = {med:.2f} | Mean coin PF = {mean:.2f} | PF>1: {n_gt1} | PF>1.2: {n_gt12} | PF<1: {n_lt1}")
        emit("   Top 10: " + ", ".join(f"{x['coin']} {fmt(x['pf'])}" for _, x in ok.sort_values("pf", ascending=False).head(10).iterrows()))
        emit("   Bottom 10: " + ", ".join(f"{x['coin']} {fmt(x['pf'])}" for _, x in ok.sort_values("pf").head(10).iterrows()))

        emit(f"\n--- TEST 2 - EQUAL-WEIGHT vs POOLED | N=20 {tag} ---")
        pooled = pf_of(tr["ret"].to_numpy())
        eq_pf = (ok["g"] / ok["n"]).sum() / max((ok["l"] / ok["n"]).sum(), 1e-12)
        total = tr["ret"].sum()
        srt = cdf.sort_values("sum", ascending=False)
        top5, top10 = srt["sum"].head(5).sum() / total * 100, srt["sum"].head(10).sum() / total * 100
        kdec = max(1, int(np.ceil(0.10 * len(cdf))))               # top 10% coins
        dec_share = srt["sum"].head(kdec).sum() / total * 100
        rest = tr[~tr["sym"].str.split("/").str[0].isin(srt["coin"].head(kdec))]["ret"].to_numpy()
        recs = tr.assign(prio=0.0).to_dict("records")
        eq, _ = portfolio(recs, closes, "risk", 0.01)
        ps = stats(eq)
        emit(f"   A. POOLED: PF {fmt(pooled)} | portfolio net {(eq.iloc[-1]-1)*100:+.0f}% | CAGR {ps['cagr']*100:+.1f}% | MaxDD {ps['dd']*100:.1f}%")
        emit(f"   B. EQUAL-WEIGHT (har coin barabar): PF {fmt(eq_pf)} | median coin PF {med:.2f} | median coin net {ok['net'].median():+.1f}% | "
             f"coins net>0: {(ok['net'] > 0).sum()}/{len(ok)}")
        emit(f"   Concentration: top 5 coins = {top5:.0f}% kul nafa | top 10 = {top10:.0f}% | top 10% coins ({kdec}) = "
             f"{dec_share:.0f}% | top 10% coins hata kar PF = {fmt(pf_of(rest))}")
        conc = (med < 1.05 and pooled > 1.3) or dec_share > 50 or pf_of(rest) < 1.1
        emit(f"   {'⚠️ CONCENTRATION PROBLEM' if conc else 'OK - nafa chand coins par nahi tika'}")
        checks[f"broad_{tag}"] = (not conc) and med > 1.0 and n_gt1 >= 0.6 * len(ok)
        checks[f"pooled_{tag}"] = (pooled, ps["dd"])

    # ---------------- TEST 3 ----------------
    emit(f"\n\n{'#' * 120}\nTEST 3 - MARKET REGIMES (N=20; entry se pichle band din ka BTC regime)\n{'#' * 120}")
    for tag, (tr, rt) in base.items():
        emit(f"\nN=20 {tag}")
        emit(BHEAD + f" | {'MaxDD%':>7}")
        for col, src in (("reg", btc_reg), ("vol", btc_vol)):
            dtr = src.reindex(tr["t_in"].dt.floor("1D") - pd.Timedelta(days=1)).to_numpy()
            drt = src.reindex(rt["t_in"].dt.floor("1D") - pd.Timedelta(days=1)).to_numpy()
            for g in (["bull", "bear", "sideways"] if col == "reg" else ["high-vol", "low-vol"]):
                a, b = tr[dtr == g], rt[drt == g]
                if not len(a):
                    continue
                aa = a.sort_values("t_out")
                _, dd = seq_equity(aa["ret"].to_numpy(), np.full(len(aa), 0.05))      # har trade 20% - regime DD ka andaza
                bb = block(a, b)
                emit(bline(g, bb, f" | {dd*100:>7.1f}"))
                checks[f"reg_{tag}_{g}"] = bb["pf"]

    # ---------------- TEST 4 ----------------
    emit(f"\n\n{'#' * 120}\nTEST 4 - WALK-FORWARD (har test saal se pehle ke data par N chuna; candidates {NS})\n{'#' * 120}")
    cache = {(N, f): trades(N, f) for N in NS for f in (False, True)}
    for f, tag in ((False, "BAGHAIR filter"), (True, "BTC>EMA50")):
        emit(f"\n{tag}:")
        emit(f"{'Train':>25} | {'Test':>4} | {'Chuna N':>7} | {'Train PF':>8} | {'OOS PF':>6} | {'Rnd OOS PF':>10} | {'N=20 OOS PF':>11} | {'N=20 Rnd':>8} | {'OOS trades':>10}")
        oos_ok = 0
        for a, b, test in WF:
            best, best_pf = None, -1
            for N in NS:
                tr, _ = cache[(N, f)]
                x = tr[(tr["t_in"] >= a) & (tr["t_in"] <= b)]["ret"].to_numpy()
                pf = pf_of(x)
                if np.isfinite(pf) and pf > best_pf:
                    best, best_pf = N, pf
            tr, rt = cache[(best, f)]
            sel = lambda df: df[df["t_in"].dt.year == int(test)]
            o, ro = sel(tr)["ret"].to_numpy(), sel(rt)["ret"].to_numpy()
            t20, r20 = cache[(20, f)]
            o20, ro20 = sel(t20)["ret"].to_numpy(), sel(r20)["ret"].to_numpy()
            oos_ok += pf_of(o) > 1
            emit(f"{a[:7] + ' -> ' + b[:7]:>25} | {test:>4} | {best:>7} | {fmt(best_pf):>8} | {fmt(pf_of(o)):>6} | {fmt(pf_of(ro)):>10} | "
                 f"{fmt(pf_of(o20)):>11} | {fmt(pf_of(ro20)):>8} | {len(o):>10}")
        emit(f"   OOS PF > 1: {oos_ok}/{len(WF)} windows")
        checks[f"wf_{tag}"] = oos_ok

    # ---------------- TEST 5 + 6 ----------------
    emit(f"\n\n{'#' * 120}\nTEST 5/6 - PARAMETER STABILITY + RANDOM BASELINE (OOS = 2025-01-01 ke baad)\n{'#' * 120}")
    for f, tag in ((False, "BAGHAIR filter"), (True, "BTC>EMA50")):
        emit(f"\n{tag}:")
        emit(BHEAD + f" | {'OOS PF':>6} | {'RndOOS':>6}")
        pfs = []
        for N in NS:
            tr, rt = cache[(N, f)]
            bb = block(tr, rt)
            oo = tr[tr["t_in"] >= R.OOS_START]["ret"].to_numpy()
            ro = rt[rt["t_in"] >= R.OOS_START]["ret"].to_numpy()
            emit(bline(f"N={N}", bb, f" | {fmt(pf_of(oo)):>6} | {fmt(pf_of(ro)):>6}"))
            pfs.append((bb["pf"], pf_of(oo), bb["pf"] - bb["rpf"]))
        stable = all(p > 1.2 and o > 1.0 and imp > 0.1 for p, o, imp in pfs)
        emit(f"   Plateau: {'HAAN - sab N par PF>1.2, OOS>1, random se +0.1 behtar' if stable else 'NAHI - kuch N kamzor'} "
             f"(PF range {min(p for p,_,_ in pfs):.2f}-{max(p for p,_,_ in pfs):.2f})")
        checks[f"plateau_{tag}"] = stable

    # ---------------- TEST 7 ----------------
    emit(f"\n\n{'#' * 120}\nTEST 7 - EXIT ROBUSTNESS (entry N=20 fixed, initial stop 3xATR fixed)\n{'#' * 120}")
    for f, tag in ((False, "BAGHAIR filter"), (True, "BTC>EMA50")):
        emit(f"\n{tag}:")
        emit(BHEAD + f" | {'OOS PF':>6} | {'MaxDD%':>7}")
        good = 0
        exits = [("A. TP 1R", "tp", 1.0), ("B. TP 1.5R", "tp", 1.5), ("C. TP 2R", "tp", 2.0), ("D. TP 3R", "tp", 3.0),
                 ("E. ATR trailing 3x", "trail", 0.0), ("F. Donchian 10-din low exit", "donch", 0.0)]
        for name, kind, tp in exits:
            tr, rt = trades(20, f, kind, tp)
            bb = block(tr, rt)
            oo = pf_of(tr[tr["t_in"] >= R.OOS_START]["ret"].to_numpy())
            eq, _ = portfolio(tr.assign(prio=0.0).to_dict("records"), closes, "risk", 0.01)
            dd = stats(eq)["dd"]
            emit(bline(name, bb, f" | {fmt(oo):>6} | {dd*100:>7.1f}"))
            good += bb["pf"] > 1.1 and oo > 1.0 and bb["pf"] - bb["rpf"] > 0.05
        emit(f"   Exits jahan PF>1.1, OOS>1, random se behtar: {good}/6")
        checks[f"exits_{tag}"] = good

    # ---------------- FINAL ----------------
    emit("\n\n" + "=" * 120)
    emit("FINAL DECISION")
    emit("=" * 120)
    for tag in ("BAGHAIR filter", "BTC>EMA50"):
        pooled, dd = checks[f"pooled_{tag}"]
        crit = {
            "OOS walk-forward (>=3/4 saal PF>1)": checks[f"wf_{tag}"] >= 3,
            "Coin-level broad (median PF>1, >=60% coins PF>1, concentration nahi)": checks[f"broad_{tag}"],
            "Parameter plateau (N 15-30)": checks[f"plateau_{tag}"],
            "Exits robust (>=4/6)": checks[f"exits_{tag}"] >= 4,
            "Regimes (bull aur kam az kam ek aur regime PF>=0.95)": checks[f"reg_{tag}_bull"] >= 0.95 and
                (checks.get(f"reg_{tag}_bear", 0) >= 0.95 or checks.get(f"reg_{tag}_sideways", 0) >= 0.95),
            "Drawdown (portfolio MaxDD > -35%)": dd > -0.35,
        }
        n_ok = sum(crit.values())
        verdict = "PASS" if n_ok == len(crit) else ("PROMISING - NEEDS MORE VALIDATION" if n_ok >= len(crit) - 2 else "FAIL")
        emit(f"\nN=20 {tag}: {verdict} ({n_ok}/{len(crit)} shartein)")
        for k, v in crit.items():
            emit(f"   [{'OK' if v else 'XX'}] {k}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
