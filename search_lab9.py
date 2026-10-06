"""
SEARCH LAB 9 - talaash jaari, magar TAALA-BAND (locked holdout) ke sath (user, 2026-10-06: "behtar se behtareen ki
talaash nahi chhor sakta")
==================================================================================================================
Bohat se khayal aazmane se koi na koi ittefaq se pass ho jata hai (TP5 ka sabaq). Is ka ilaaj: aakhri 12 mahine
(2025-10-01 ke baad) ka data TAALA-BAND - sirf "dev" hisse (pehle ka data) par chunao; taala sirf unhi par khulta hai
jo dev mein PASS hon, aur wahan bhi nafa + random se behtar hona zaroori.
(Funding-rate khayal chhora: GitHub se kisi exchange ki sirf 1-3 mahine ki history milti hai - funding_probe_RESULTS.txt.)

Naye entry khandan (sab daily, top-100, coin close > EMA200, 3 settings; beech wali asal):
  A W52_HIGH   : close 365-din ke high se P% ke andar pehli dafa (P = 3 / 5 / 8), BTC > EMA50. Momentum -> exit
                 chandelier 22 / 3 ATR trailing (60 din max).
  B OBV_DIV    : 10 din mein qeemat X% giri (X = 5 / 8 / 12) magar OBV 10 din mein BARHA (chupke se khareedari),
                 BTC > EMA50. Dip exit.
  C WEEKLY_DIP : hafte ka close > 40-hafte EMA, hafte ka RSI2 < R (R = 10 / 20 / 30), sirf itwar ke band hafte par;
                 peer ke open par khareed. Exit: TP 8% / close > SMA5 / SL 3 ATR / 15 din.
  D NR_DIP     : 5 din mein X% girawat (X = 5 / 8 / 12) ke baad aaj ki candle pichle 4 din mein sab se chhoti (NR4) aur
                 andar (inside) - bechna thak gaya. EMA50 > EMA200, BTC > EMA50. Dip exit.
  REF DIP      : live Dip v2 (RSI3 < 7) - muqabla.
Dip exit = TP 5% / close > SMA3 agle din open / SL signal close - 3 ATR / 10 din.
DEV PASS (dev data, beech wali setting): jeet >= 55% | PF > random p95 | PF >= 1.4 | 4/4 folds | boot p5 >= 1.15 |
  top-10 hata kar >= 1.2 | 2x kharcha >= 1.2 | >= 0.3 / hafta | 5-din fark > shor | padosi 1 bhi PF > random p95.
TAALA (holdout) sirf DEV PASS par: PF >= 1.2 aur PF > random holdout p95.
Natija: search_lab9_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES, chandelier as ce_bot
from portfolio_lab import ema, rsi, atr_w
from winrate_lab import sim, tstats, pf_of
import strategy_lab5 as L5

OUT = "search_lab9_RESULTS.txt"
TOP_FETCH = 200
DAYS = 2400
RND = 10
HOLDOUT = pd.Timestamp("2025-10-01")
GRID = {"A_W52_HIGH": [0.03, 0.05, 0.08], "B_OBV_DIV": [0.05, 0.08, 0.12],
        "C_WEEKLY_DIP": [10, 20, 30], "D_NR_DIP": [0.05, 0.08, 0.12]}
EXIT = {"A_W52_HIGH": "trend", "B_OBV_DIV": "dip", "C_WEEKLY_DIP": "week", "D_NR_DIP": "dip", "REF_DIP": "dip"}


def names():
    out = ["REF_DIP"]
    for f, g in GRID.items():
        out += [f"{f} {v}" for v in g]
    return out


def fam(nm):
    return nm.split(" ")[0]


def build(daily, start):
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
        pool = ok & up
        sig = {"REF_DIP": pool & bok & (e50 > e200).to_numpy() & (rsi(c, 3) < 7).to_numpy()}
        # A 52-hafte high ke qareeb (pehli dafa)
        hi365 = h.rolling(365, min_periods=250).max()
        for P_ in GRID["A_W52_HIGH"]:
            near = (c >= hi365 * (1 - P_))
            fresh = near & ~near.shift(1, fill_value=False)
            sig[f"A_W52_HIGH {P_}"] = pool & bok & fresh.fillna(False).to_numpy()
        # B OBV divergence
        obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
        r10 = c / c.shift(10) - 1
        obv_up = obv > obv.shift(10)
        for X in GRID["B_OBV_DIV"]:
            cond = (r10 <= -X) & obv_up
            cond = cond & ~cond.shift(1, fill_value=False)
            sig[f"B_OBV_DIV {X}"] = pool & bok & cond.fillna(False).to_numpy()
        # C weekly dip: hafta itwar ko band (UTC) - sirf band hafton se hisaab
        wk = d.set_index("timestamp").resample("W-SUN", label="right", closed="right").agg(
            {"close": "last"}).dropna()
        last_day = ts.iloc[-1]
        wk = wk[wk.index <= last_day]        # adhoora hafta bahar
        wc = wk["close"]
        wok = (wc > ema(wc, 40)) & (np.arange(len(wc)) >= 40)
        wr = rsi(wc, 2)
        sun = (ts.dt.dayofweek == 6).to_numpy()
        for R in GRID["C_WEEKLY_DIP"]:
            wsig = (wok & (wr < R)).reindex(ts).fillna(False).to_numpy(bool)
            sig[f"C_WEEKLY_DIP {R}"] = pool & sun & wsig
        # D NR4 inside after decline
        r5 = c / c.shift(5) - 1
        rngd = h - l
        nr4 = rngd <= rngd.rolling(4).min()
        inside = (h <= h.shift(1)) & (l >= l.shift(1))
        for X in GRID["D_NR_DIP"]:
            cond = (r5 <= -X) & nr4 & inside & (e50 > e200)
            sig[f"D_NR_DIP {X}"] = pool & bok & cond.fillna(False).to_numpy()
        P[sym] = dict(sym=sym, ts=ts.to_numpy(), o=o.to_numpy(float), h=h.to_numpy(float), l=l.to_numpy(float),
                      c=c.to_numpy(float), atr=atr_w(d).to_numpy(), sma3=(c > c.rolling(3).mean()).to_numpy(),
                      sma5=(c > c.rolling(5).mean()).to_numpy(), ce=ce_bot(d, 22, 3.0), pool=pool, sig=sig, sun=sun)
    return P


def run(P, name, mode="trade", cost=1.0, rng=None, period=None):
    out = []
    ex = EXIT[fam(name)]
    for sym, x in P.items():
        idx = np.where(x["sig"][name])[0]
        if rng is not None:
            pmask = x["pool"] & np.isfinite(x["atr"])
            if ex == "week":
                pmask = pmask & x["sun"]
            pool = np.where(pmask)[0]
            if len(idx) == 0 or len(pool) == 0:
                continue
            idx = np.sort(rng.choice(pool, size=min(len(idx), len(pool)), replace=False))
        for i in idx:
            tsi = pd.Timestamp(x["ts"][i])
            if period == "dev" and tsi >= HOLDOUT:
                continue
            if period == "hold" and tsi < HOLDOUT:
                continue
            if mode == "h5":
                t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, 1e-12, None, None, 5, {}, cost)
            elif ex == "trend":
                st = x["ce"][i]
                if not np.isfinite(st) or st <= 0:
                    continue
                t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, x["ce"], None, 60, {}, cost)
            else:
                st = x["c"][i] - 3.0 * x["atr"][i]
                if not np.isfinite(st) or st <= 0:
                    continue
                if ex == "week":
                    t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, None, x["sma5"], 15, {"tp": 0.08}, cost)
                else:
                    t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, None, x["sma3"], 10, {"tp": 0.05}, cost)
            if t:
                t.update(sym=sym, prio=0.0)
                out.append(t)
    return out


def causality(daily, start, emit, n_days=30):
    rng = np.random.default_rng(17)
    full = build(daily, start)
    days = [d for d in daily["BTC/USDT"]["timestamp"] if d >= start + pd.Timedelta(days=30)]
    # itwar zaroor shamil (weekly signal)
    suns = [k for k, d in enumerate(days) if d.dayofweek == 6]
    pick = sorted(set(rng.choice(len(days), size=min(n_days // 2, len(days)), replace=False)) |
                  set(rng.choice(suns, size=min(n_days // 2, len(suns)), replace=False)))
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
    emit(f"\n# SACHAI TEST - {len(pick)} din (aadhe itwar), data kaat kar (farq 0 hona chahiye)")
    for n in bad:
        emit(f"{n:>20}: farq {bad[n]} ({nsig[n]} signals) -> {'SAHI' if bad[n] == 0 else 'GHALAT - lookahead'}")
    return {n: bad[n] == 0 for n in bad}


def folds(tr, t0, t1):
    e = pd.date_range(t0, t1, periods=5)
    return [pf_of([t["ret"] for t in tr if a <= pd.Timestamp(t["t_in"]) < b]) for a, b in zip(e[:-1], e[1:])]


def boot_p5(tr, n=2000):
    r = np.array([t["ret"] for t in tr])
    g = np.random.default_rng(3)
    return np.percentile([pf_of(g.choice(r, len(r))) for _ in range(n)], 5)


def main(daily=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    try:
        if daily is None:
            import data_fetcher
            data_fetcher.AUTO_TOP_N_COINS = TOP_FETCH
            ex = data_fetcher.get_exchange()
            coins = [s for s in data_fetcher.get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:TOP_FETCH]
            if "BTC/USDT" not in coins:
                coins.insert(0, "BTC/USDT")
            daily = {}
            for k, sym in enumerate(coins, 1):
                try:
                    df = fetch_full(ex, sym, "1d", DAYS)
                    if df is not None and len(df) >= 250:
                        daily[sym] = norm(df)
                        print(f"[{k}/{len(coins)}] {sym}: {len(df)}", flush=True)
                except Exception as e:
                    print(f"[{k}] {sym}: SKIP ({e})", flush=True)

        idx_all = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in daily.values()])))
        start = idx_all[0] + pd.Timedelta(days=210)
        t0, t1 = start, idx_all[-1]
        dev_weeks = (HOLDOUT - t0).days / 7
        hold_weeks = (t1 - HOLDOUT).days / 7
        emit("=" * 140)
        emit("SEARCH LAB 9 - naye entry khandan, aakhri 12 mahine TAALA-BAND (holdout)")
        emit("=" * 140)
        emit(f"Coins: {len(daily)} | DEV {t0.date()} -> {HOLDOUT.date()} ({dev_weeks:.0f} hafte) | TAALA {HOLDOUT.date()} -> "
             f"{t1.date()} ({hold_weeks:.0f} hafte)")
        emit("jeet/10 = 10 mein se nafa wali | PF = har 1 rupay nuqsan par kitna nafa | 5d fark = 5-din nafa minus random")

        ok_c = causality(daily, start, emit)
        P = build(daily, start)

        rows = {}
        emit(f"\n# DEV hissa (taala-band data shamil NAHI)")
        emit(f"{'entry':>20} | {'n':>4} | {'/hafta':>6} | {'5d fark':>7} | {'shor95':>6} | {'jeet/10':>7} | {'PF':>5} | {'rnd95':>5} | "
             f"{'folds':>23} | {'boot5':>5} | {'-top10':>6} | {'2xcost':>6} | {'ausat':>6}")
        for nm in names():
            h5 = run(P, nm, mode="h5", period="dev")
            tr = run(P, nm, period="dev")
            if len(tr) < 25:
                emit(f"{nm:>20} | sirf {len(tr)} trades")
                continue
            r5 = np.mean([t["ret"] for t in h5])
            rnd5 = [np.mean([t["ret"] for t in run(P, nm, mode="h5", period="dev", rng=np.random.default_rng(900 + q))])
                    for q in range(RND)]
            fark, shor = r5 - np.mean(rnd5), np.percentile(np.array(rnd5) - np.mean(rnd5), 95)
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, period="dev", rng=np.random.default_rng(950 + q))]) for q in range(RND)]
            fo = folds(tr, t0, HOLDOUT)
            bp = boot_p5(tr)
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
            chk = {"sachai": all(ok_c.get(f"{f} {v}", False) for v in g), "jeet>=55": s["win"] >= 55,
                   "PF>rnd95": s["pf"] > r["r95"], "PF>=1.4": s["pf"] >= 1.4,
                   "folds": all(np.isfinite(x) and x > 1 for x in r["fo"]), "boot>=1.15": r["bp"] >= 1.15,
                   "-top10>=1.2": r["mt"] >= 1.2, "2xcost>=1.2": r["c2"] >= 1.2, ">=0.3/hafta": r["wk"] >= 0.3,
                   "5d fark": r["fark"] > max(0, r["shor"]),
                   "padosi": any(n in rows and rows[n]["s"]["pf"] > rows[n]["r95"] for n in nb)}
            fails = [k for k, v in chk.items() if not v]
            if not fails:
                passed.append(mid)
            emit(f"{mid:>22}: {'DEV PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")

        emit("\n# TAALA KHULA (holdout, aakhri 12 mahine) - REF DIP hamesha muqable ke liye")
        for nm in ["REF_DIP"] + passed:
            tr = run(P, nm, period="hold")
            if len(tr) < 5:
                emit(f"{nm:>22}: taale mein sirf {len(tr)} trades - faisla mumkin nahi")
                continue
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, period="hold", rng=np.random.default_rng(990 + q))]) for q in range(RND)]
            ok = s["pf"] >= 1.2 and s["pf"] > np.percentile(rnd, 95)
            emit(f"{nm:>22}: {s['n']} trades | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | random p95 {np.percentile(rnd,95):.2f} | "
                 f"ausat {s['avg']:+.2f}% -> {'TAALA PASS' if ok else 'TAALA FAIL'}" + ("" if nm != "REF_DIP" else " (sirf muqabla)"))
        if not passed:
            emit("koi khayal DEV pass nahi hua - taala kisi ke liye nahi khula")
    except Exception:
        emit("\n!! GHALTI (code crash):\n" + traceback.format_exc())

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
