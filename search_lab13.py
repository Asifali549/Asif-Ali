"""
SEARCH LAB 13 - "quant trader ki tarah, quality systems" (user, 2026-10-06)
========================================================================
Taala-band usool wahi (aakhri 12 mahine 2025-10-01 se; chunao sirf DEV par; taala sirf DEV PASS par, ek dafa).
Sab daily, top-100 liquid, coin close > EMA200, BTC close > EMA50. 3 settings, beech wali asal.

  V VOL_BRK    : Larry Williams volatility breakout - din ke open se K x (kal ka high - low) ooper qeemat jaye to usi
                 qeemat par khareedo (buy-stop), agle din open par becho (1 din). K 0.4 / 0.6 / 0.8.
                 Random control: random din open par khareed, agle open par bech.
  S SQUEEZE_D  : daily Bollinger (20, 2) chaurai pichle 120 din mein sab se kam Q% mein (Q 5 / 10 / 15) aur aaj close
                 ooper wali band se ooper - lambi khamoshi ke baad dhamaka. Exit H7.
  R RVOL_UP    : aaj volume >= V x 20-din ausat (V 2 / 3 / 4), din +3% se +12%, close din ki range ke ooper 20% mein -
                 taqatwar khareedari (bina pagal pump). Exit H5.
  F FLUSH      : poori market ki safai - top-100 mein se F% coins ka 3-din nafa <= -10% (F 30 / 40 / 50), BTC close >
                 EMA200 (EMA50 nahi - safai mein BTC bhi girta hai), coin EMA50 > EMA200. Dip exit.
  REF DIP      : live Dip v2 (RSI3 < 7).
Dip exit = TP 5% / close > SMA3 agle din open / SL signal close - 3 ATR / 10 din.
DEV PASS: jeet >= 55% (dip) ya 50% (baqi) | PF > random p95 | PF >= 1.3 | 4 folds (har fold >= 5 trades aur PF > 1 ya
  koi haar nahi) | boot p5 >= 1.15 | top-10 hata kar >= 1.2 | 2x kharcha >= 1.2 | >= 0.3 / hafta | padosi | sachai.
TAALA: PF >= 1.2 aur PF > random taala p95.

KHATA TEST (alag sawal): Dip+ ka khata (Dip v2 + RESID, 20% / trade, max 10) vs us mein 4-din girawat (STREAK) milana -
  poora aur taala dono par CAGR / DD / Sharpe. (Teeno ke signals pehle se alag alag test ho chuke.)
Natija: search_lab13_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES, FEE, SLIP
from portfolio_lab import ema, rsi, atr_w, portfolio, stats
from winrate_lab import sim, tstats, pf_of
import strategy_lab5 as L5
import search_lab9 as S9

OUT = "search_lab13_RESULTS.txt"
RND = 10
HOLDOUT = S9.HOLDOUT
GRID = {"V_VOL_BRK": [0.4, 0.6, 0.8], "S_SQUEEZE_D": [5, 10, 15], "R_RVOL_UP": [2, 3, 4], "F_FLUSH": [30, 40, 50]}
EXIT = {"V_VOL_BRK": "vbrk", "S_SQUEEZE_D": "h7", "R_RVOL_UP": "h5", "F_FLUSH": "dip", "REF_DIP": "dip",
        "X_RESID": "dip", "X_STREAK": "dip"}


def names():
    out = ["REF_DIP"]
    for f, g in GRID.items():
        out += [f"{f} {v}" for v in g]
    return out


def fam(nm):
    return nm.split(" ")[0]


def fresh(x):
    x = pd.Series(x).fillna(False).astype(bool)
    return (x & ~x.shift(1, fill_value=False)).to_numpy()


def build(daily, start):
    btc_ok, al100 = L5.context(daily)
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc200 = bd > ema(bd, 200)
    btc_lr = np.log(bd).diff()
    # market flush breadth: top-100 mein 3-din nafa <= -10% ka hissa (sirf band candle)
    c_all = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index()
    r3_all = c_all / c_all.shift(3) - 1
    in100 = pd.DataFrame(False, index=c_all.index, columns=c_all.columns)
    for t, ss in al100.items():
        if t in in100.index:
            in100.loc[t, list(ss & set(in100.columns))] = True
    n100 = in100.sum(axis=1).replace(0, np.nan)
    breadth = ((r3_all <= -0.10) & in100).sum(axis=1) / n100 * 100
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
        sig = {"REF_DIP": pool & golden & (rsi(c, 3) < 7).to_numpy()}
        # V: aaj (din e) breakout; level = aaj ka open + K x kal ki range; pool kal ka (kal band candle tak maloom)
        rng_prev = (h - l).shift(1)
        pool_prev = np.r_[False, pool[:-1]]
        lv = {}
        for K in GRID["V_VOL_BRK"]:
            level = (o + K * rng_prev).to_numpy()
            lv[K] = level
            sig[f"V_VOL_BRK {K}"] = pool_prev & np.nan_to_num(h.to_numpy() >= level, nan=False) & np.isfinite(level)
        # S: squeeze
        m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
        bw = (4 * s20 / m20)
        bw_rank = bw.rolling(120, min_periods=100).rank(pct=True) * 100
        upper = m20 + 2 * s20
        for Q in GRID["S_SQUEEZE_D"]:
            sq = bw_rank.shift(1) <= Q
            sig[f"S_SQUEEZE_D {Q}"] = pool & fresh(sq & (c > upper))
        # R: relative volume up day
        vavg = v.rolling(20, min_periods=15).mean().shift(1)
        r1 = c / c.shift(1) - 1
        pos = (c - l) / (h - l).replace(0, np.nan)
        for V in GRID["R_RVOL_UP"]:
            sig[f"R_RVOL_UP {V}"] = pool & ((v >= V * vavg) & (r1 >= 0.03) & (r1 <= 0.12) & (pos >= 0.8)).fillna(False).to_numpy()
        # F: flush (BTC > EMA200, alag pool)
        b200 = btc200.reindex(ts).fillna(False).to_numpy(bool)
        br = breadth.reindex(ts).to_numpy()
        pool_f = ok & up & b200 & golden
        for F in GRID["F_FLUSH"]:
            sig[f"F_FLUSH {F}"] = pool_f & fresh(np.nan_to_num(br, nan=0) >= F)
        # khata test ke liye: RESID (book_bot jaisa) aur STREAK 4
        rc = np.log(c).diff()
        rb = pd.Series(btc_lr.reindex(ts).to_numpy())
        src = pd.Series(rc.to_numpy())
        beta = (src.rolling(60, min_periods=40).cov(rb) / rb.rolling(60, min_periods=40).var()).clip(-1, 3)
        res = src - beta * rb
        z = res.rolling(3).sum() / (res.rolling(60, min_periods=40).std() * np.sqrt(3))
        sig["X_RESID"] = pool & (np.nan_to_num(z.to_numpy(), nan=0) < -2.0)
        down = (c < c.shift(1)).astype(float)
        sig["X_STREAK"] = pool & golden & fresh(down.rolling(4, min_periods=4).sum() >= 4)
        P[sym] = dict(sym=sym, ts=ts.to_numpy(), o=o.to_numpy(float), h=h.to_numpy(float), l=l.to_numpy(float),
                      c=c.to_numpy(float), atr=atr_w(d).to_numpy(), sma3=(c > c.rolling(3).mean()).to_numpy(),
                      pool=pool, pool_f=pool_f, pool_prev=pool_prev, sig=sig, lv=lv)
    return P


def vbrk_trade(x, e, level, cost=1.0):
    """din e par buy-stop level par (gap ho to open), agle din open par bech."""
    if e + 1 >= len(x["o"]):
        return None
    slip, fee = SLIP * cost, FEE * cost
    entry = max(x["o"][e], level) * (1 + slip)
    px = x["o"][e + 1] * (1 - slip)
    return {"t_in": x["ts"][e], "t_out": x["ts"][e + 1], "entry_px": entry, "risk": 0.1,
            "ret": px * (1 - fee) / (entry * (1 + fee)) - 1}


def open_trade(x, e, cost=1.0):
    """random control: din e open par khareed, agle open par bech."""
    if e + 1 >= len(x["o"]):
        return None
    slip, fee = SLIP * cost, FEE * cost
    entry = x["o"][e] * (1 + slip)
    px = x["o"][e + 1] * (1 - slip)
    return {"t_in": x["ts"][e], "t_out": x["ts"][e + 1], "entry_px": entry, "risk": 0.1,
            "ret": px * (1 - fee) / (entry * (1 + fee)) - 1}


def run(P, name, mode="trade", cost=1.0, rng=None, period=None):
    out = []
    f = fam(name)
    ex = EXIT[f]
    for sym, x in P.items():
        idx = np.where(x["sig"][name])[0]
        if rng is not None:
            pm = x["pool_prev"] if ex == "vbrk" else (x["pool_f"] if f == "F_FLUSH" else x["pool"])
            pool = np.where(pm & np.isfinite(x["atr"]))[0]
            if len(idx) == 0 or len(pool) == 0:
                continue
            idx = np.sort(rng.choice(pool, size=min(len(idx), len(pool)), replace=False))
        for i in idx:
            tsi = pd.Timestamp(x["ts"][i])
            if (period == "dev" and tsi >= HOLDOUT) or (period == "hold" and tsi < HOLDOUT):
                continue
            if ex == "vbrk":
                if mode == "h5":
                    t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, 1e-12, None, None, 5, {}, cost)
                elif rng is not None:
                    t = open_trade(x, i, cost)
                else:
                    K = float(name.split(" ")[1])
                    t = vbrk_trade(x, i, x["lv"][K][i], cost)
            elif mode == "h5" or ex == "h5":
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
    rng = np.random.default_rng(31)
    full = build(daily, start)
    days = [d for d in daily["BTC/USDT"]["timestamp"] if d >= start + pd.Timedelta(days=30)]
    pick = sorted(set(rng.choice(len(days), size=min(n_days, len(days)), replace=False)))
    nm_all = names() + ["X_RESID", "X_STREAK"]
    bad = {n: 0 for n in nm_all}
    nsig = {n: 0 for n in nm_all}
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
    emit(f"\n# SACHAI TEST - {len(pick)} din, data kaat kar (farq 0 hona chahiye)")
    for n in bad:
        emit(f"{n:>20}: farq {bad[n]} ({nsig[n]} signals) -> {'SAHI' if bad[n] == 0 else 'GHALAT - lookahead'}")
    return {n: bad[n] == 0 for n in bad}


def folds_ok(tr, t0, t1):
    e = pd.date_range(t0, t1, periods=5)
    out = []
    for a, b in zip(e[:-1], e[1:]):
        r = [t["ret"] for t in tr if a <= pd.Timestamp(t["t_in"]) < b]
        pf = pf_of(r)
        out.append((len(r), pf, len(r) >= 5 and (pf > 1 or not any(x <= 0 for x in r))))
    return out


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
                except Exception as e:
                    print(f"[{k}] {sym}: SKIP ({e})", flush=True)

        idx_all = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in daily.values()])))
        start = idx_all[0] + pd.Timedelta(days=210)
        t0, t1 = start, idx_all[-1]
        dev_weeks = (HOLDOUT - t0).days / 7
        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        closes = closes[closes.index >= start]
        emit("=" * 140)
        emit("SEARCH LAB 13 - quant khayal (volatility breakout, squeeze, volume, market flush) + khata test | TAALA-BAND")
        emit("=" * 140)
        emit(f"Coins: {len(daily)} | DEV {t0.date()} -> {HOLDOUT.date()} ({dev_weeks:.0f} hafte) | TAALA {HOLDOUT.date()} -> {t1.date()}")
        emit("jeet/10 = 10 mein se nafa wali | PF = har 1 rupay nuqsan par kitna nafa | 5d fark = 5-din nafa minus random")

        ok_c = causality(daily, start, emit)
        P = build(daily, start)

        rows = {}
        emit("\n# DEV hissa")
        emit(f"{'entry':>18} | {'n':>5} | {'/hafta':>6} | {'5d fark':>7} | {'jeet/10':>7} | {'PF':>5} | {'rnd95':>5} | "
             f"{'folds (trades/PF)':>40} | {'boot5':>5} | {'-top10':>6} | {'2xcost':>6} | {'ausat':>6}")
        for nm in names():
            tr = run(P, nm, period="dev")
            if len(tr) < 25:
                emit(f"{nm:>18} | sirf {len(tr)} trades")
                continue
            h5 = run(P, nm, mode="h5", period="dev")
            rnd5 = [np.mean([t["ret"] for t in run(P, nm, mode="h5", period="dev", rng=np.random.default_rng(900 + q))])
                    for q in range(RND)]
            fark = np.mean([t["ret"] for t in h5]) - np.mean(rnd5)
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, period="dev", rng=np.random.default_rng(950 + q))]) for q in range(RND)]
            fo = folds_ok(tr, t0, HOLDOUT)
            bp = S9.boot_p5(tr)
            mt = pf_of([t["ret"] for t in sorted(tr, key=lambda t: -t["ret"])[10:]])
            c2 = pf_of([t["ret"] for t in run(P, nm, period="dev", cost=2.0)])
            rows[nm] = dict(s=s, r95=np.percentile(rnd, 95), fo=fo, bp=bp, mt=mt, c2=c2, wk=s["n"] / dev_weeks, fark=fark)
            emit(f"{nm:>18} | {s['n']:>5} | {s['n']/dev_weeks:>6.2f} | {fark*100:>+6.2f}% | {s['win']/10:>7.1f} | {s['pf']:>5.2f} | "
                 f"{np.percentile(rnd,95):>5.2f} | {' '.join(f'{n}/{p:.2f}' for n, p, _ in fo):>40} | {bp:>5.2f} | {mt:>6.2f} | "
                 f"{c2:>6.2f} | {s['avg']:>+5.2f}%")

        emit("\n# DEV PASS / FAIL (beech wali setting)")
        passed = []
        for f, g in GRID.items():
            mid = f"{f} {g[1]}"
            nb = [f"{f} {v}" for k, v in enumerate(g) if k != 1]
            if mid not in rows:
                emit(f"{mid:>20}: FAIL (kam trades)")
                continue
            r, s = rows[mid], rows[mid]["s"]
            wmin = 55 if EXIT[f] == "dip" else 50
            chk = {"sachai": all(ok_c.get(f"{f} {v}", False) for v in g), f"jeet>={wmin}": s["win"] >= wmin,
                   "PF>rnd95": s["pf"] > r["r95"], "PF>=1.3": s["pf"] >= 1.3, "folds": all(o for _, _, o in r["fo"]),
                   "boot>=1.15": r["bp"] >= 1.15, "-top10>=1.2": r["mt"] >= 1.2, "2xcost>=1.2": r["c2"] >= 1.2,
                   ">=0.3/hafta": r["wk"] >= 0.3,
                   "padosi": any(n in rows and rows[n]["s"]["pf"] > rows[n]["r95"] for n in nb)}
            fails = [k for k, v in chk.items() if not v]
            if not fails:
                passed.append(mid)
            emit(f"{mid:>20}: {'DEV PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")

        emit("\n# TAALA KHULA (aakhri 12 mahine)")
        final = []
        for nm in ["REF_DIP"] + passed:
            tr = run(P, nm, period="hold")
            if len(tr) < 5:
                emit(f"{nm:>20}: taale mein sirf {len(tr)} trades")
                continue
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, period="hold", rng=np.random.default_rng(990 + q))]) for q in range(RND)]
            rnd = [x for x in rnd if np.isfinite(x)] or [np.inf]
            ok = s["pf"] >= 1.2 and s["pf"] > np.percentile(rnd, 95)
            if ok and nm != "REF_DIP":
                final.append(nm)
            emit(f"{nm:>20}: {s['n']} trades | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | random p95 {np.percentile(rnd,95):.2f} | "
                 f"ausat {s['avg']:+.2f}% -> {'TAALA PASS' if ok else 'TAALA FAIL'}" + (" (sirf muqabla)" if nm == "REF_DIP" else ""))
        if not passed:
            emit("koi khayal DEV pass nahi hua - taala kisi ke liye nahi khula")
        for nm in final:
            tr = run(P, nm)
            for size in (0.10, 0.20):
                p = stats(portfolio(tr, closes, "fixed", size, max_pos=10, cap=1.0)[0])
                emit(f"  portfolio {nm} {int(size*100)}%: CAGR {p['cagr']*100:+.1f}% | DD {p['dd']*100:.1f}% | Sharpe {p['sharpe']:.2f}")

        # ---- khata test
        emit("\n# KHATA TEST - Dip+ (Dip v2 + RESID) vs Dip+ + 4-din girawat, 20% / trade, max 10")
        dip, resid, streak = run(P, "REF_DIP"), run(P, "X_RESID"), run(P, "X_STREAK")
        for t in dip:
            t["prio"] = 2.0
        for t in resid:
            t["prio"] = 1.0
        for t in streak:
            t["prio"] = 0.0
        hold_closes = closes[closes.index >= HOLDOUT]
        for lab, tr in (("Dip+ (abhi)", dip + resid), ("Dip+ + 4-din", dip + resid + streak), ("4-din akela 20%", streak)):
            ts_all = [t for t in tr]
            th = [t for t in tr if pd.Timestamp(t["t_in"]) >= HOLDOUT]
            eq, taken = portfolio(ts_all, closes, "fixed", 0.20, max_pos=10, cap=1.0)
            p = stats(eq)
            ph = stats(portfolio(th, hold_closes, "fixed", 0.20, max_pos=10, cap=1.0)[0])
            st = tstats(taken) if taken else {"n": 0, "win": 0, "pf": 0}
            yrs = eq.resample("YE").last().pct_change().dropna()
            m = eq.resample("ME").last().pct_change().dropna()
            emit(f"{lab:>16}: {st['n']} trades jeet/10 {st['win']/10:.1f} PF {st['pf']:.2f} | POORA CAGR {p['cagr']*100:+.1f}% DD "
                 f"{p['dd']*100:.1f}% Sharpe {p['sharpe']:.2f} bura mahina {m.min()*100:.1f}% | TAALA CAGR {ph['cagr']*100:+.1f}% DD "
                 f"{ph['dd']*100:.1f}% Sharpe {ph['sharpe']:.2f}")
            emit(f"{'':>16}  saal " + " ".join(f"{d.year}:{v*100:+.0f}" for d, v in yrs.items()))
        emit("\nNATIJA: " + (", ".join(final) + " -> PAPER BOT ke qabil" if final else "koi naya khayal taala paar nahi kar saka"))
    except Exception:
        emit("\n!! GHALTI (code crash):\n" + traceback.format_exc())

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
