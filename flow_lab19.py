"""
FLOW LAB 19 - VWAP / volume profile / FVG / liquidity sweep / money flow / CVD - sakht imtihan (user, 2026-10-06)
==============================================================================================================
Aath cheezein (sab daily, top-100 liquid, coin close > EMA200, BTC close > EMA50; 3 settings, beech wali asal):
  A VWAP_BACK  : W-din rolling VWAP (W 10 / 20 / 50) - kal close VWAP se neeche, aaj wapas ooper band. EMA50>EMA200.
  B AVWAP      : pichle L din (L 30 / 60 / 120) ki sab se neechi jagah se "anchored VWAP" - aaj low us ko chhoo kar close ooper.
  C VPOC       : pichle L din (L 30 / 60 / 90) ka volume profile POC (sab se ziada sauda wali qeemat, daily se andaza) - qeemat
                 ooper se aa kar POC ko chhooye aur ooper band ho.
  D FVG        : bullish fair value gap (din k ka low > din k-2 ka high) pichle W din (W 5 / 10 / 20) mein bana; qeemat pehli
                 dafa gap mein wapas aaye (low <= gap ka ooper) magar close gap ke neeche wale kinare se ooper.
  E SWEEP      : liquidity sweep - aaj low pichle N din (N 10 / 20 / 30) ke sab se neeche low se neeche gaya, magar close us ke ooper
                 (stop hunt + wapsi), volume >= 1.5 x 20-din ausat.
  F MFI        : Money Flow Index(14) < X (X 15 / 20 / 25) - volume wala RSI. EMA50>EMA200.
  G CMF        : 3 din mein -5% girawat magar Chaikin Money Flow(20) >= T (T 0.05 / 0.10 / 0.15) - girawat mein paisa aa raha.
  H CVD_DIV    : (Binance archive ka taker-buy) qeemat L-din (L 5 / 10 / 20) ki sab se neechi close par, magar mujmooi delta (CVD)
                 L din mein BARHA - chupke se khareedari. EMA50>EMA200.
  REF DIP      : Dip v2 (dono data par muqable ke liye).
Exit sab ka Dip wala: TP 5% / close > SMA3 agle din open / SL signal close - 3 ATR / 10 din.
DEV PASS (beech wali): sachai | jeet >= 55% | PF > random p95 | PF >= 1.3 | 4 folds (har fold >= 5 trades, PF > 1 ya haar nahi) |
  boot p5 >= 1.15 | top-10 hata kar >= 1.2 | 2x kharcha >= 1.2 | >= 0.3 / hafta | padosi (1 bhi PF > random p95).
TAALA (2025-10-01 se): DEV PASS + BORDERLINE (sirf 1 shart fail - pehle se tay, label ke sath) ka ek dafa;
  PASS = PF >= 1.2 aur > random taala p95. Phir portfolio (10%, max 10) + Dip v2 se correlation.
Natija: flow_lab19_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema, rsi, atr_w, portfolio, stats
from winrate_lab import sim, tstats, pf_of
import strategy_lab5 as L5
import search_lab9 as S9
import search_lab13 as L13
import deriv_lab15 as D

OUT = "flow_lab19_RESULTS.txt"
HOLDOUT = S9.HOLDOUT
RND = 10
GRID = {"A_VWAP_BACK": [10, 20, 50], "B_AVWAP": [30, 60, 120], "C_VPOC": [30, 60, 90], "D_FVG": [5, 10, 20],
        "E_SWEEP": [10, 20, 30], "F_MFI": [15, 20, 25], "G_CMF": [0.05, 0.10, 0.15], "H_CVD_DIV": [5, 10, 20]}
KU = ["A_VWAP_BACK", "B_AVWAP", "C_VPOC", "D_FVG", "E_SWEEP", "F_MFI", "G_CMF"]
fresh = L13.fresh


def avwap_from_low(l, tp, v, L):
    n = len(l)
    out = np.full(n, np.nan)
    anc = np.full(n, -1)
    if n < L:
        return out, anc
    cpv = np.concatenate([[0.0], np.cumsum(tp * v)])
    cv = np.concatenate([[0.0], np.cumsum(v)])
    win = sliding_window_view(np.nan_to_num(l, nan=np.inf), L)
    am = win.argmin(axis=1)
    for j, a_rel in enumerate(am):
        i = j + L - 1
        a = j + a_rel
        den = cv[i + 1] - cv[a]
        if den > 0:
            out[i] = (cpv[i + 1] - cpv[a]) / den
            anc[i] = a
    return out, anc


def vpoc(h, l, tp, v, L, bins=40):
    n = len(tp)
    out = np.full(n, np.nan)
    for i in range(L - 1, n):
        lo, hi = np.nanmin(l[i - L + 1:i + 1]), np.nanmax(h[i - L + 1:i + 1])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        hist, edges = np.histogram(tp[i - L + 1:i + 1], bins=bins, range=(lo, hi), weights=v[i - L + 1:i + 1])
        k = int(np.argmax(hist))
        out[i] = (edges[k] + edges[k + 1]) / 2
    return out


def fvg_signal(h, l, c, W):
    n = len(c)
    sig = np.zeros(n, bool)
    gaps = []  # [created, top, bottom, alive]
    for i in range(n):
        # aaj gap mein wapsi? (sirf pehle se bane gaps)
        best = None
        for g in gaps:
            if g[3] and i - g[0] <= W and i > g[0]:
                best = g
        if best is not None and l[i] <= best[1] and c[i] > best[2]:
            sig[i] = True
        for g in gaps:
            if g[3] and i > g[0] and l[i] <= g[1]:
                g[3] = False
        # aaj naya gap bana? (din i ka low > din i-2 ka high)
        if i >= 2 and l[i] > h[i - 2]:
            gaps.append([i, l[i], h[i - 2], True])
        gaps = [g for g in gaps if g[3] and i - g[0] <= max(GRID["D_FVG"])]
    return sig


def add_common(x, c, o, h, l, v, golden):
    x.update(o=o.to_numpy(float), h=h.to_numpy(float), l=l.to_numpy(float), c=c.to_numpy(float))
    x["golden"] = golden


def build_k(daily, start):
    btc_ok, al100 = L5.context(daily)
    P = {}
    for sym, d in daily.items():
        c, o, h, l, v = d["close"], d["open"], d["high"], d["low"], d["volume"]
        ts = d["timestamp"]
        e50, e200 = ema(c, 50), ema(c, 200)
        n_ok = np.arange(len(d)) >= 200
        ok = np.array([sym in al100.get(t, ()) for t in ts]) & (ts >= start).to_numpy()
        up = (c > e200).to_numpy() & n_ok
        bok = btc_ok.reindex(ts).fillna(False).to_numpy(bool)
        pool = ok & up & bok
        golden = (e50 > e200).to_numpy()
        tp = (h + l + c) / 3
        sig = {"REF_DIP": pool & golden & (rsi(c, 3) < 7).to_numpy()}
        for W in GRID["A_VWAP_BACK"]:
            vw = (tp * v).rolling(W).sum() / v.rolling(W).sum()
            sig[f"A_VWAP_BACK {W}"] = pool & golden & ((c > vw) & (c.shift(1) < vw.shift(1))).fillna(False).to_numpy()
        tpa, va, la, ha, ca = tp.to_numpy(float), v.to_numpy(float), l.to_numpy(float), h.to_numpy(float), c.to_numpy(float)
        idx = np.arange(len(c))
        for L in GRID["B_AVWAP"]:
            av, anc = avwap_from_low(la, tpa, va, L)
            cond = (la <= av) & (ca > av) & (anc >= 0) & (idx - anc >= 3)
            sig[f"B_AVWAP {L}"] = pool & golden & fresh(pd.Series(np.nan_to_num(cond, nan=False).astype(bool)))
        for L in GRID["C_VPOC"]:
            pc = vpoc(ha, la, tpa, va, L)
            prev_above = np.r_[False, ca[:-1] > pc[:-1]]
            cond = (la <= pc) & (ca > pc) & prev_above
            sig[f"C_VPOC {L}"] = pool & golden & np.nan_to_num(cond, nan=False).astype(bool)
        for W in GRID["D_FVG"]:
            sig[f"D_FVG {W}"] = pool & golden & fvg_signal(ha, la, ca, W)
        vavg = v.rolling(20, min_periods=15).mean().shift(1)
        for N in GRID["E_SWEEP"]:
            plow = l.shift(1).rolling(N).min()
            sig[f"E_SWEEP {N}"] = pool & ((l < plow) & (c > plow) & (v >= 1.5 * vavg)).fillna(False).to_numpy()
        mf = tp * v
        pos = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
        neg = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
        mfi = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
        for X in GRID["F_MFI"]:
            sig[f"F_MFI {X}"] = pool & golden & fresh(mfi < X)
        rng_ = (h - l).replace(0, np.nan)
        clv = ((c - l) - (h - c)) / rng_
        cmf = (clv * v).rolling(20).sum() / v.rolling(20).sum()
        r3 = c / c.shift(3) - 1
        for T in GRID["G_CMF"]:
            sig[f"G_CMF {T}"] = pool & golden & fresh((r3 <= -0.05) & (cmf >= T))
        P[sym] = dict(sym=sym, ts=ts.to_numpy(), atr=atr_w(d).to_numpy(), sma3=(c > c.rolling(3).mean()).to_numpy(),
                      pool=pool, sig=sig)
        add_common(P[sym], c, o, h, l, v, golden)
    return P


def build_b(bdaily, start):
    P = D.build(bdaily, start)
    for sym, x in P.items():
        d = bdaily[sym]
        c = d["close"]
        golden = (ema(c, 50) > ema(c, 200)).to_numpy()
        cvd = (2 * d["tb"] - d["volume"]).cumsum()
        sig = {"REF_DIP": x["sig"]["REF_DIP"]}
        for L in GRID["H_CVD_DIV"]:
            low_close = c <= c.rolling(L).min()
            sig[f"H_CVD_DIV {L}"] = x["pool"] & golden & fresh(low_close & ((cvd - cvd.shift(L)) > 0))
        x["sig"] = sig
    return P


def run(P, name, cost=1.0, rng=None, period=None):
    out = []
    for sym, x in P.items():
        idx = np.where(x["sig"][name])[0]
        if rng is not None:
            pool = np.where(x["pool"] & np.isfinite(x["atr"]))[0]
            if len(idx) == 0 or len(pool) == 0:
                continue
            idx = np.sort(rng.choice(pool, size=min(len(idx), len(pool)), replace=False))
        for i in idx:
            tsi = pd.Timestamp(x["ts"][i])
            if (period == "dev" and tsi >= HOLDOUT) or (period == "hold" and tsi < HOLDOUT):
                continue
            st = x["c"][i] - 3.0 * x["atr"][i]
            if not np.isfinite(st) or st <= 0:
                continue
            t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, None, x["sma3"], 10, {"tp": 0.05}, cost)
            if t:
                t.update(sym=sym, prio=0.0)
                out.append(t)
    return out


def causality(daily, start, builder, names, emit, label, n_days=24):
    rng = np.random.default_rng(41)
    full = builder(daily, start)
    key = "BTC/USDT" if "BTC/USDT" in daily else "BTC"
    days = [d for d in daily[key]["timestamp"] if d >= start + pd.Timedelta(days=30)]
    pick = sorted(set(rng.choice(len(days), size=min(n_days, len(days)), replace=False)))
    bad = {n: 0 for n in names}
    nsig = {n: 0 for n in names}
    for k, j in enumerate(pick, 1):
        Dd = days[j]
        cut = {s: d[d["timestamp"] <= Dd].reset_index(drop=True) for s, d in daily.items()}
        cut = {s: d for s, d in cut.items() if len(d) > 0}
        part = builder(cut, start)
        for sym, x in full.items():
            if sym not in part:
                continue
            w = np.where(x["ts"] == np.datetime64(Dd))[0]
            if len(w) == 0:
                continue
            i = w[0]
            for nm in names:
                a, b = bool(x["sig"][nm][i]), bool(part[sym]["sig"][nm][-1])
                bad[nm] += a != b
                nsig[nm] += a or b
        if k % 8 == 0:
            print(f"  sachai {label} {k}/{len(pick)}", flush=True)
    emit(f"\n# SACHAI TEST ({label}) - {len(pick)} din, data kaat kar")
    for nm in names:
        emit(f"{nm:>16}: farq {bad[nm]} ({nsig[nm]} signals) -> {'SAHI' if bad[nm] == 0 else 'GHALAT - lookahead'}")
    return {nm: bad[nm] == 0 for nm in names}


def evaluate(P, names, t0, emit):
    dev_weeks = (HOLDOUT - t0).days / 7
    rows = {}
    for nm in names:
        tr = run(P, nm, period="dev")
        if len(tr) < 25:
            emit(f"{nm:>16} | sirf {len(tr)} trades")
            continue
        s = tstats(tr)
        rnd = [pf_of([t["ret"] for t in run(P, nm, period="dev", rng=np.random.default_rng(950 + q))]) for q in range(RND)]
        fo = L13.folds_ok(tr, t0, HOLDOUT)
        bp = S9.boot_p5(tr)
        mt = pf_of([t["ret"] for t in sorted(tr, key=lambda t: -t["ret"])[10:]])
        c2 = pf_of([t["ret"] for t in run(P, nm, period="dev", cost=2.0)])
        rows[nm] = dict(s=s, r95=np.percentile(rnd, 95), fo=fo, bp=bp, mt=mt, c2=c2, wk=s["n"] / dev_weeks)
        emit(f"{nm:>16} | {s['n']:>5} | {s['n']/dev_weeks:>5.2f} | {s['win']/10:>7.1f} | {s['pf']:>5.2f} | {np.percentile(rnd,95):>5.2f} | "
             f"{' '.join(f'{n}/{p:.2f}' for n, p, _ in fo):>36} | {bp:>5.2f} | {mt:>6.2f} | {c2:>6.2f} | {s['avg']:>+5.2f}%")
    return rows


def main(kdaily=None, bdaily=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    try:
        if kdaily is None:
            import data_fetcher
            data_fetcher.AUTO_TOP_N_COINS = S9.TOP_FETCH
            ex = data_fetcher.get_exchange()
            coins = [s for s in data_fetcher.get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:S9.TOP_FETCH]
            if "BTC/USDT" not in coins:
                coins.insert(0, "BTC/USDT")
            kdaily = {}
            for k, sym in enumerate(coins, 1):
                try:
                    df = fetch_full(ex, sym, "1d", S9.DAYS)
                    if df is not None and len(df) >= 250:
                        kdaily[sym] = norm(df)
                except Exception as e:
                    print(f"[{k}] {sym}: SKIP ({e})", flush=True)
            bdaily = {}
            for k, sym in enumerate(coins[:150], 1):
                b = sym.split("/")[0].upper()
                d = D.load_coin(f"{b}USDT")
                if d is not None and len(d) >= 250:
                    bdaily[b] = d
                print(f"[B {k}] {b}: {0 if d is None else len(d)}", flush=True)
            end = D.END_MONTH + pd.offsets.MonthEnd(0)
            bdaily = {s: d[d["timestamp"] <= end].reset_index(drop=True) for s, d in bdaily.items()}

        idx_k = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in kdaily.values()])))
        sk = idx_k[0] + pd.Timedelta(days=210)
        idx_b = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in bdaily.values()])))
        sb = idx_b[0] + pd.Timedelta(days=210)
        closes_k = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in kdaily.items()}).sort_index().ffill()
        closes_k = closes_k[closes_k.index >= sk]
        closes_b = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in bdaily.items()}).sort_index().ffill()
        closes_b = closes_b[closes_b.index >= sb]
        emit("=" * 140)
        emit("FLOW LAB 19 - VWAP / AVWAP / volume profile / FVG / liquidity sweep / MFI / CMF / CVD - sakht imtihan")
        emit("=" * 140)
        emit(f"KuCoin {len(kdaily)} coins {sk.date()} -> {idx_k[-1].date()} | Binance (CVD) {len(bdaily)} coins {sb.date()} -> {idx_b[-1].date()} | "
             f"TAALA {HOLDOUT.date()} se")
        emit("jeet/10 = 10 mein se nafa wali | PF = har 1 rupay nuqsan par kitna nafa | rnd95 = random entries ka 95wan hissa")

        names_k = ["REF_DIP"] + [f"{f} {v}" for f in KU for v in GRID[f]]
        names_b = ["REF_DIP"] + [f"H_CVD_DIV {v}" for v in GRID["H_CVD_DIV"]]
        ok_k = causality(kdaily, sk, build_k, names_k, emit, "KuCoin")
        ok_b = causality(bdaily, sb, build_b, names_b, emit, "Binance")
        ok_c = {**ok_k, **{k: v for k, v in ok_b.items() if k != "REF_DIP"}}
        Pk, Pb = build_k(kdaily, sk), build_b(bdaily, sb)

        head = (f"{'entry':>16} | {'n':>5} | {'/hft':>5} | {'jeet/10':>7} | {'PF':>5} | {'rnd95':>5} | {'folds (trades/PF)':>36} | "
                f"{'boot5':>5} | {'-top10':>6} | {'2xcost':>6} | {'ausat':>6}")
        emit("\n# DEV hissa - KuCoin")
        emit(head)
        rows = evaluate(Pk, names_k, sk, emit)
        emit("\n# DEV hissa - Binance (CVD)")
        emit(head)
        rows_b = evaluate(Pb, names_b, sb, emit)
        rows.update({k: v for k, v in rows_b.items() if k != "REF_DIP"})

        emit("\n# DEV PASS / FAIL (beech wali setting)")
        to_open = []
        for f, g in GRID.items():
            mid = f"{f} {g[1]}"
            nb = [f"{f} {v}" for k, v in enumerate(g) if k != 1]
            if mid not in rows:
                emit(f"{mid:>18}: FAIL (kam trades)")
                continue
            r, s = rows[mid], rows[mid]["s"]
            chk = {"sachai": all(ok_c.get(f"{f} {v}", False) for v in g), "jeet>=55": s["win"] >= 55,
                   "PF>rnd95": s["pf"] > r["r95"], "PF>=1.3": s["pf"] >= 1.3, "folds": all(o for _, _, o in r["fo"]),
                   "boot>=1.15": r["bp"] >= 1.15, "-top10>=1.2": r["mt"] >= 1.2, "2xcost>=1.2": r["c2"] >= 1.2,
                   ">=0.3/hafta": r["wk"] >= 0.3,
                   "padosi": any(n in rows and rows[n]["s"]["pf"] > rows[n]["r95"] for n in nb)}
            fails = [k for k, v in chk.items() if not v]
            if not fails:
                to_open.append((mid, "DEV PASS"))
            elif len(fails) == 1 and fails[0] != "sachai":
                to_open.append((mid, "BORDERLINE"))
            emit(f"{mid:>18}: {'DEV PASS' if not fails else 'FAIL: ' + ', '.join(fails)}" + (" -> BORDERLINE (taala khulega)" if len(fails) == 1 and fails[0] != "sachai" else ""))

        emit("\n# TAALA KHULA (aakhri 12 mahine) - sirf DEV PASS / BORDERLINE, ek dafa")
        final = []
        ref_m = portfolio(run(Pk, "REF_DIP"), closes_k, "fixed", 0.20, max_pos=10, cap=1.0)[0].resample("ME").last().pct_change()
        for nm, lab in to_open:
            P, cl = (Pb, closes_b) if nm.startswith("H_") else (Pk, closes_k)
            tr = run(P, nm, period="hold")
            if len(tr) < 5:
                emit(f"{nm:>18} [{lab}]: taale mein sirf {len(tr)} trades - faisla mumkin nahi")
                continue
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, period="hold", rng=np.random.default_rng(990 + q))]) for q in range(20)]
            rnd = [x for x in rnd if np.isfinite(x)] or [np.inf]
            ok = s["pf"] >= 1.2 and s["pf"] > np.percentile(rnd, 95)
            days = len(set(str(pd.Timestamp(t["t_in"]).date()) for t in tr))
            srt = sorted(tr, key=lambda t: -t["ret"])
            emit(f"{nm:>18} [{lab}]: {s['n']} trades ({days} din) | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | random p50 {np.median(rnd):.2f} "
                 f"p95 {np.percentile(rnd,95):.2f} | top-5 hata kar {pf_of([t['ret'] for t in srt[5:]]):.2f} -> {'TAALA PASS' if ok else 'TAALA FAIL'}")
            if ok:
                final.append((nm, lab))
                full = run(P, nm)
                for size in (0.10, 0.20):
                    eq = portfolio(full, cl, "fixed", size, max_pos=10, cap=1.0)[0]
                    p = stats(eq)
                    m = eq.resample("ME").last().pct_change()
                    yrs = eq.resample("YE").last().pct_change().dropna()
                    emit(f"{'':>20}{int(size*100)}%: CAGR {p['cagr']*100:+.1f}% | DD {p['dd']*100:.1f}% | Sharpe {p['sharpe']:.2f} | corr Dip "
                         f"{m.corr(ref_m):.2f} | saal " + " ".join(f"{d.year}:{v*100:+.0f}" for d, v in yrs.items()))
        if not to_open:
            emit("koi khayal DEV pass / borderline nahi - taala nahi khula")
        emit("\nNATIJA: " + (", ".join(f"{n} ({l})" for n, l in final) + " -> PAPER BOT ke qabil" if final else "koi naya khayal taala paar nahi kar saka"))
    except Exception:
        emit("\n!! GHALTI:\n" + traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
