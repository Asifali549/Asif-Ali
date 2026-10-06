"""
SEARCH LAB 12 - "ziada se ziada systems board par" (user, 2026-10-06): naye / hat kar khayal + purane adhoore khayal
=================================================================================================================
Wohi taala-band usool (Search Lab 9): aakhri 12 mahine (2025-10-01 se) TAALA; chunao sirf DEV par; taala sirf DEV PASS
par khulta hai (har khayal ke liye ek hi dafa). Sab daily, top-100 liquid, coin close > EMA200, BTC close > EMA50.
Har khandan ki 3 settings - beech wali asal, padosi mazbooti ke liye.

  E W52_DIP     : pichle 30 din mein "sal ki chouti" (W52, 5%) signal aaya ho + aaj RSI3 < R (R 10 / 15 / 20) -
                  taqatwar coin ki chhoti girawat (momentum + dip). Dip exit.
  G VOL_DRY     : EMA50 > EMA200, 3 din lagataar neeche close, aur teeno din volume < V x 20-din ausat (V 0.6 / 0.8 / 1.0)
                  - bechne wale thak gaye (bina shor ki girawat). Dip exit.
  H MONTH_END   : mahine ke aakhri din se K din pehle (K 0 / 1 / 2) ka close -> agle din khareed, 5 din baad becho
                  (crypto mein mahine ke shuru ki nayi khareedari). Exit H5.
  M EMA50_BACK  : coin pichle N din (N 5 / 10 / 20) lagataar EMA50 se neeche raha, aaj wapas ooper band (EMA200 se
                  ooper) - rujhaan dobara shuru. Exit H7 (7 din baad close, SL/TP nahi).
  O RESID_LONG  : L din (L 7 / 10 / 20) ka residual return (coin - beta x BTC) z < -2 (90-din std) - RESID ka lamba
                  version (Dip+ ke 3-din RESID se alag waqt). Dip exit.
  P DOWN_STREAK : N din lagataar neeche close (N 3 / 4 / 5), EMA50 > EMA200 - Research Lab 3 mein apna edge PASS tha
                  magar portfolio mein milaya nahi gaya; ab alag paper bot ke qabil hai ya nahi. Dip exit.
  REF DIP       : live Dip v2 (RSI3 < 7) - muqabla.
Dip exit = TP 5% / close > SMA3 agle din open / SL signal close - 3 ATR / 10 din.
DEV PASS (beech wali setting): jeet >= 55% (H-exit 50%) | PF > random p95 | PF >= 1.4 | 4/4 folds | boot p5 >= 1.15 |
  top-10 hata kar >= 1.2 | 2x kharcha >= 1.2 | >= 0.3 / hafta | 5-din fark > shor | padosi 1 bhi PF > random p95 | sachai.
TAALA: PF >= 1.2 aur PF > random taala p95. Phir poora portfolio (10% aur 20% / trade, max 10) aur DIP+ se correlation.
Natija: search_lab12_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema, rsi, atr_w, portfolio, stats
from winrate_lab import sim, tstats, pf_of
import strategy_lab5 as L5
import search_lab9 as S9

OUT = "search_lab12_RESULTS.txt"
RND = 10
HOLDOUT = S9.HOLDOUT
GRID = {"E_W52_DIP": [10, 15, 20], "G_VOL_DRY": [0.6, 0.8, 1.0], "H_MONTH_END": [0, 1, 2],
        "M_EMA50_BACK": [5, 10, 20], "O_RESID_LONG": [7, 10, 20], "P_DOWN_STREAK": [3, 4, 5]}
EXIT = {"E_W52_DIP": "dip", "G_VOL_DRY": "dip", "H_MONTH_END": "h5", "M_EMA50_BACK": "h7", "O_RESID_LONG": "dip",
        "P_DOWN_STREAK": "dip", "REF_DIP": "dip"}


def names():
    out = ["REF_DIP"]
    for f, g in GRID.items():
        out += [f"{f} {v}" for v in g]
    return out


def fam(nm):
    return nm.split(" ")[0]


def fresh(x):
    x = x.fillna(False).astype(bool)
    return x & ~x.shift(1, fill_value=False)


def build(daily, start):
    btc_ok, al100 = L5.context(daily)
    btc = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc_lr = np.log(btc).diff()
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
        r3 = rsi(c, 3)
        sig = {"REF_DIP": pool & golden & (r3 < 7).to_numpy()}
        # E: W52 pichle 30 din mein + RSI3 dip
        hi365 = h.rolling(365, min_periods=250).max()
        w52 = fresh(c >= hi365 * 0.95)
        recent = w52.astype(float).rolling(30, min_periods=1).max() > 0
        for R in GRID["E_W52_DIP"]:
            sig[f"E_W52_DIP {R}"] = pool & fresh(recent & (r3 < R)).to_numpy()
        # G: 3 din neeche + kam volume
        down = c < c.shift(1)
        down3 = down & down.shift(1, fill_value=False) & down.shift(2, fill_value=False)
        vavg = v.rolling(20, min_periods=15).mean().shift(3)
        vmax3 = v.rolling(3).max()
        for V in GRID["G_VOL_DRY"]:
            sig[f"G_VOL_DRY {V}"] = pool & golden & fresh(down3 & (vmax3 < V * vavg)).to_numpy()
        # H: mahine ka aakhir (calendar - pehle se maloom, lookahead nahi)
        for K in GRID["H_MONTH_END"]:
            m_end = ((ts + pd.Timedelta(days=K + 1)).dt.month != ts.dt.month) & ((ts + pd.Timedelta(days=K)).dt.month == ts.dt.month)
            sig[f"H_MONTH_END {K}"] = pool & m_end.to_numpy()
        # M: EMA50 ke neeche N din, phir wapas ooper
        below = (c < e50).astype(float)
        for N in GRID["M_EMA50_BACK"]:
            was = below.shift(1).rolling(N, min_periods=N).sum() >= N
            sig[f"M_EMA50_BACK {N}"] = pool & ((c > e50) & was).fillna(False).to_numpy()
        # O: lamba residual z
        rc = np.log(c).diff()
        rb = pd.Series(btc_lr.reindex(ts).to_numpy())
        src = pd.Series(rc.to_numpy())
        beta = (src.rolling(60, min_periods=40).cov(rb) / rb.rolling(60, min_periods=40).var()).clip(-1, 3)
        res = src - beta * rb
        sd = res.rolling(90, min_periods=60).std()
        for L in GRID["O_RESID_LONG"]:
            z = res.rolling(L).sum() / (sd * np.sqrt(L))
            sig[f"O_RESID_LONG {L}"] = pool & fresh(z < -2.0).to_numpy()
        # P: N din lagataar neeche
        for N in GRID["P_DOWN_STREAK"]:
            st = down.astype(float).rolling(N, min_periods=N).sum() >= N
            sig[f"P_DOWN_STREAK {N}"] = pool & golden & fresh(st).to_numpy()
        P[sym] = dict(sym=sym, ts=ts.to_numpy(), o=o.to_numpy(float), h=h.to_numpy(float), l=l.to_numpy(float),
                      c=c.to_numpy(float), atr=atr_w(d).to_numpy(), sma3=(c > c.rolling(3).mean()).to_numpy(),
                      pool=pool, sig=sig, ret=c.pct_change().to_numpy())
    return P


def run(P, name, mode="trade", cost=1.0, rng=None, period=None):
    out = []
    ex = EXIT[fam(name)]
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
            if mode == "h5" or ex == "h5":
                t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, 1e-12, None, None, 5, {}, cost)
            elif ex == "h7":
                t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, 1e-12, None, None, 7, {}, cost)
            else:
                st = x["c"][i] - 3.0 * x["atr"][i]
                if not np.isfinite(st) or st <= 0:
                    continue
                t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, None, x["sma3"], 10, {"tp": 0.05}, cost)
            if t:
                t.update(sym=sym, prio=0.0)
                out.append(t)
    return out


def causality(daily, start, emit, n_days=30):
    rng = np.random.default_rng(23)
    full = build(daily, start)
    days = [d for d in daily["BTC/USDT"]["timestamp"] if d >= start + pd.Timedelta(days=30)]
    ends = [k for k, d in enumerate(days) if (d + pd.Timedelta(days=3)).month != d.month]
    pick = sorted(set(rng.choice(len(days), size=min(n_days // 2, len(days)), replace=False)) |
                  set(rng.choice(ends, size=min(n_days // 2, len(ends)), replace=False)))
    bad = {n: 0 for n in names()}
    nsig = {n: 0 for n in names()}
    for k, j in enumerate(pick, 1):
        D = days[j]
        cut = {s: d[d["timestamp"] <= D].reset_index(drop=True) for s, d in daily.items()}
        cut = {s: d for s, d in cut.items() if len(d) > 0}
        part = build(cut, start)
        for sym, x in full.items():
            if sym not in part:
                continue
            w = np.where(x["ts"] == np.datetime64(D))[0]
            if len(w) == 0:
                continue
            i = w[0]
            for n in bad:
                a, b = bool(x["sig"][n][i]), bool(part[sym]["sig"][n][-1])
                bad[n] += a != b
                nsig[n] += a or b
        if k % 10 == 0:
            print(f"  sachai {k}/{len(pick)}", flush=True)
    emit(f"\n# SACHAI TEST - {len(pick)} din (aadhe mahine ke aakhir ke qareeb), data kaat kar (farq 0 hona chahiye)")
    for n in bad:
        emit(f"{n:>20}: farq {bad[n]} ({nsig[n]} signals) -> {'SAHI' if bad[n] == 0 else 'GHALAT - lookahead'}")
    return {n: bad[n] == 0 for n in bad}


def daily_ret(tr, closes, size):
    eq = portfolio(tr, closes, "fixed", size, max_pos=10, cap=1.0)[0]
    return eq


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
                        print(f"[{k}/{len(coins)}] {sym}: {len(df)}", flush=True)
                except Exception as e:
                    print(f"[{k}] {sym}: SKIP ({e})", flush=True)

        idx_all = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in daily.values()])))
        start = idx_all[0] + pd.Timedelta(days=210)
        t0, t1 = start, idx_all[-1]
        dev_weeks = (HOLDOUT - t0).days / 7
        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        closes = closes[closes.index >= start]
        emit("=" * 140)
        emit("SEARCH LAB 12 - naye / hat kar khayal + purane adhoore, aakhri 12 mahine TAALA-BAND")
        emit("=" * 140)
        emit(f"Coins: {len(daily)} | DEV {t0.date()} -> {HOLDOUT.date()} ({dev_weeks:.0f} hafte) | TAALA {HOLDOUT.date()} -> {t1.date()}")
        emit("jeet/10 = 10 mein se nafa wali | PF = har 1 rupay nuqsan par kitna nafa | 5d fark = 5-din nafa minus random")

        ok_c = causality(daily, start, emit)
        P = build(daily, start)

        rows = {}
        emit("\n# DEV hissa (taala-band data shamil NAHI)")
        emit(f"{'entry':>20} | {'n':>4} | {'/hafta':>6} | {'5d fark':>7} | {'shor95':>6} | {'jeet/10':>7} | {'PF':>5} | {'rnd95':>5} | "
             f"{'folds':>23} | {'boot5':>5} | {'-top10':>6} | {'2xcost':>6} | {'ausat':>6}")
        for nm in names():
            tr = run(P, nm, period="dev")
            if len(tr) < 25:
                emit(f"{nm:>20} | sirf {len(tr)} trades")
                continue
            h5 = run(P, nm, mode="h5", period="dev")
            r5 = np.mean([t["ret"] for t in h5])
            rnd5 = [np.mean([t["ret"] for t in run(P, nm, mode="h5", period="dev", rng=np.random.default_rng(900 + q))])
                    for q in range(RND)]
            fark, shor = r5 - np.mean(rnd5), np.percentile(np.array(rnd5) - np.mean(rnd5), 95)
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, period="dev", rng=np.random.default_rng(950 + q))]) for q in range(RND)]
            fo = S9.folds(tr, t0, HOLDOUT)
            bp = S9.boot_p5(tr)
            mt = pf_of([t["ret"] for t in sorted(tr, key=lambda t: -t["ret"])[10:]])
            c2 = pf_of([t["ret"] for t in run(P, nm, period="dev", cost=2.0)])
            rows[nm] = dict(s=s, r95=np.percentile(rnd, 95), fo=fo, bp=bp, mt=mt, c2=c2, wk=s["n"] / dev_weeks,
                            fark=fark, shor=shor)
            emit(f"{nm:>20} | {s['n']:>4} | {s['n']/dev_weeks:>6.2f} | {fark*100:>+6.2f}% | {shor*100:>+5.2f}% | {s['win']/10:>7.1f} | "
                 f"{s['pf']:>5.2f} | {np.percentile(rnd,95):>5.2f} | {' '.join(f'{f:>5.2f}' for f in fo):>23} | {bp:>5.2f} | "
                 f"{mt:>6.2f} | {c2:>6.2f} | {s['avg']:>+5.2f}%")

        emit("\n# DEV PASS / FAIL (beech wali setting) -> sirf PASS par taala khulta hai")
        passed = []
        for f, g in GRID.items():
            mid = f"{f} {g[1]}"
            nb = [f"{f} {v}" for k, v in enumerate(g) if k != 1]
            if mid not in rows:
                emit(f"{mid:>22}: FAIL (kam trades)")
                continue
            r, s = rows[mid], rows[mid]["s"]
            wmin = 50 if EXIT[f] in ("h5", "h7") else 55
            chk = {"sachai": all(ok_c.get(f"{f} {v}", False) for v in g), f"jeet>={wmin}": s["win"] >= wmin,
                   "PF>rnd95": s["pf"] > r["r95"], "PF>=1.4": s["pf"] >= 1.4,
                   "folds": all(np.isfinite(x) and x > 1 for x in r["fo"]), "boot>=1.15": r["bp"] >= 1.15,
                   "-top10>=1.2": r["mt"] >= 1.2, "2xcost>=1.2": r["c2"] >= 1.2, ">=0.3/hafta": r["wk"] >= 0.3,
                   "5d fark": r["fark"] > max(0, r["shor"]),
                   "padosi": any(n in rows and rows[n]["s"]["pf"] > rows[n]["r95"] for n in nb)}
            fails = [k for k, v in chk.items() if not v]
            if not fails:
                passed.append(mid)
            emit(f"{mid:>22}: {'DEV PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")

        emit("\n# TAALA KHULA (aakhri 12 mahine) - REF DIP sirf muqable ke liye")
        final = []
        for nm in ["REF_DIP"] + passed:
            tr = run(P, nm, period="hold")
            if len(tr) < 5:
                emit(f"{nm:>22}: taale mein sirf {len(tr)} trades - faisla mumkin nahi")
                continue
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, period="hold", rng=np.random.default_rng(990 + q))]) for q in range(RND)]
            ok = s["pf"] >= 1.2 and s["pf"] > np.percentile(rnd, 95)
            if ok and nm != "REF_DIP":
                final.append(nm)
            emit(f"{nm:>22}: {s['n']} trades | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | random p95 {np.percentile(rnd,95):.2f} | "
                 f"ausat {s['avg']:+.2f}% -> {'TAALA PASS' if ok else 'TAALA FAIL'}" + ("" if nm != "REF_DIP" else " (sirf muqabla)"))
        if not passed:
            emit("koi khayal DEV pass nahi hua - taala kisi ke liye nahi khula")

        emit("\n# PORTFOLIO (poora 6 saal, max 10) - DEV pass wale + REF DIP; corr = mahana nafa ka DIP se rishta")
        ref_eq = daily_ret(run(P, "REF_DIP"), closes, 0.20)
        ref_m = ref_eq.resample("ME").last().pct_change()
        for nm in ["REF_DIP"] + passed:
            tr = run(P, nm)
            for size in (0.10, 0.20):
                eq = daily_ret(tr, closes, size)
                p = stats(eq)
                m = eq.resample("ME").last().pct_change()
                corr = m.corr(ref_m)
                yrs = eq.resample("YE").last().pct_change().dropna()
                emit(f"{nm:>22} {int(size*100)}%: CAGR {p['cagr']*100:+.1f}% | DD {p['dd']*100:.1f}% | Sharpe {p['sharpe']:.2f} | "
                     f"corr DIP {corr:.2f} | saal " + " ".join(f"{d.year}:{v*100:+.0f}" for d, v in yrs.items()))
        emit("\nNATIJA: " + (", ".join(final) + " -> PAPER BOT ke qabil" if final else "koi naya khayal taala paar nahi kar saka"))
    except Exception:
        emit("\n!! GHALTI (code crash):\n" + traceback.format_exc())

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
