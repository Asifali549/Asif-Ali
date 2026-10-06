"""
SEARCH LAB 14 - price action (chart dekh kar faisla) aur contrarian (hujoom ka mukhalif) ko QAIDON mein badal kar (user, 2026-10-06)
================================================================================================================================
Taala-band usool wahi (aakhri 12 mahine 2025-10-01 se; chunao sirf DEV par; taala sirf DEV PASS par, ek dafa).
Sab daily, top-100 liquid, coin close > EMA200. 3 settings, beech wali asal.

PRICE ACTION:
  A SUPPORT    : pichla pakka swing-low (pivot 5 - sirf 5 din baad pakka maana jata, lookahead nahi) support; aaj ka low support
                 se T% ke andar (T 1 / 2 / 3) aaya magar close support se ooper aur hari candle - support se uchhal. EMA50 > EMA200,
                 BTC > EMA50. Dip exit.
  B ENGULF     : D din lagataar girawat (D 2 / 3 / 4) ke baad aaj bullish engulfing (open < kal ka close, close > kal ka open,
                 hari candle). EMA50 > EMA200, BTC > EMA50. Dip exit.
  C RETEST     : pichle W din (W 5 / 10 / 15) mein 20-din breakout hua; aaj qeemat wapas us breakout level (2% ke andar) tak aayi
                 magar close level se ooper aur hari candle - "breakout ke baad retest". BTC > EMA50. Exit H7.
CONTRARIAN:
  D BTC_FEAR   : BTC RSI(14) < R (R 30 / 35 / 40, pehla din) magar BTC > EMA200 (bull market mein dar) - EMA50 > EMA200 wale coins
                 khareedo. Dip exit.
  E LOSER30    : top-100 mein pichle 30 din ka sab se bura K (K 5 / 10 / 15) coin - magar abhi bhi EMA200 se ooper (lambi muddat
                 theek, sab ne chhor diya). BTC > EMA50, pehla din. Dip exit.
  REF DIP      : live Dip v2.
Dip exit = TP 5% / close > SMA3 agle din open / SL signal close - 3 ATR / 10 din.
DEV PASS / TAALA: Lab 13 jaise (jeet >= 55% dip, 50% H7 | PF > random p95 | PF >= 1.3 | 4 folds (>= 5 trades, PF > 1 ya koi haar
nahi) | boot p5 >= 1.15 | top-10 hata kar >= 1.2 | 2x kharcha >= 1.2 | >= 0.3 / hafta | padosi | sachai). Taala: PF >= 1.2 aur > random p95.
Natija: search_lab14_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema, rsi, atr_w, portfolio, stats
from winrate_lab import sim, tstats, pf_of
import strategy_lab5 as L5
import search_lab9 as S9
import search_lab13 as L13

OUT = "search_lab14_RESULTS.txt"
RND = 10
HOLDOUT = S9.HOLDOUT
GRID = {"A_SUPPORT": [1, 2, 3], "B_ENGULF": [2, 3, 4], "C_RETEST": [5, 10, 15], "D_BTC_FEAR": [30, 35, 40],
        "E_LOSER30": [5, 10, 15]}
EXIT = {"A_SUPPORT": "dip", "B_ENGULF": "dip", "C_RETEST": "h7", "D_BTC_FEAR": "dip", "E_LOSER30": "dip", "REF_DIP": "dip"}
POOLX = {"D_BTC_FEAR": "pool_f"}


def names():
    out = ["REF_DIP"]
    for f, g in GRID.items():
        out += [f"{f} {v}" for v in g]
    return out


def fam(nm):
    return nm.split(" ")[0]


fresh = L13.fresh


def build(daily, start):
    btc_ok, al100 = L5.context(daily)
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc200 = bd > ema(bd, 200)
    btc_rsi = rsi(bd, 14)
    c_all = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index()
    r30 = c_all / c_all.shift(30) - 1
    in100 = pd.DataFrame(False, index=c_all.index, columns=c_all.columns)
    for t, ss in al100.items():
        if t in in100.index:
            in100.loc[t, list(ss & set(in100.columns))] = True
    rk30 = r30.where(in100).rank(axis=1, ascending=True)
    P = {}
    for sym, d in daily.items():
        c, o, h, l = d["close"], d["open"], d["high"], d["low"]
        ts = d["timestamp"]
        e50, e200 = ema(c, 50), ema(c, 200)
        n_ok = np.arange(len(d)) >= 200
        ok = np.array([sym in al100.get(t, ()) for t in ts]) & (ts >= start).to_numpy()
        up = (c > e200).to_numpy() & n_ok
        bok = btc_ok.reindex(ts).fillna(False).to_numpy(bool)
        pool = ok & up & bok
        golden = (e50 > e200).to_numpy()
        green = (c > o)
        sig = {"REF_DIP": pool & golden & (rsi(c, 3) < 7).to_numpy()}
        # A support bounce (pivot low 5, pakka hone ke baad hi istemal)
        n = 5
        piv = l.where(l == l.rolling(2 * n + 1, center=True).min())
        sup = piv.shift(n).ffill()
        for T in GRID["A_SUPPORT"]:
            cond = (l <= sup * (1 + T / 100)) & (c > sup) & green
            sig[f"A_SUPPORT {T}"] = pool & golden & fresh(cond)
        # B bullish engulfing after D down days
        down = (c < c.shift(1)).astype(float)
        eng = green & (o <= c.shift(1)) & (c >= o.shift(1)) & (c.shift(1) < o.shift(1))
        for D in GRID["B_ENGULF"]:
            prior = down.shift(1).rolling(D, min_periods=D).sum() >= D
            sig[f"B_ENGULF {D}"] = pool & golden & (eng & prior).fillna(False).to_numpy()
        # C breakout retest
        hh = h.rolling(20).max().shift(1)
        brk = (c > hh).fillna(False)
        lvl = hh.where(brk).ffill()
        idx = pd.Series(np.arange(len(c)))
        last_brk = idx.where(brk.to_numpy()).ffill()
        since = (idx - last_brk).to_numpy()
        for W in GRID["C_RETEST"]:
            cond = (since >= 1) & (since <= W) & (l <= lvl * 1.02).to_numpy() & (c > lvl).to_numpy() & green.to_numpy()
            sig[f"C_RETEST {W}"] = pool & fresh(pd.Series(cond))
        # D BTC fear in bull
        b200 = btc200.reindex(ts).fillna(False).to_numpy(bool)
        brsi = btc_rsi.reindex(ts).to_numpy()
        pool_f = ok & up & b200 & golden
        for R in GRID["D_BTC_FEAR"]:
            sig[f"D_BTC_FEAR {R}"] = pool_f & fresh(pd.Series(np.nan_to_num(brsi, nan=100) < R))
        # E 30-din ka sab se bura (magar EMA200 se ooper)
        rk = rk30[sym].reindex(ts).to_numpy() if sym in rk30 else np.full(len(c), np.nan)
        for K in GRID["E_LOSER30"]:
            sig[f"E_LOSER30 {K}"] = pool & fresh(pd.Series(np.nan_to_num(rk, nan=1e9) <= K))
        P[sym] = dict(sym=sym, ts=ts.to_numpy(), o=o.to_numpy(float), h=h.to_numpy(float), l=l.to_numpy(float),
                      c=c.to_numpy(float), atr=atr_w(d).to_numpy(), sma3=(c > c.rolling(3).mean()).to_numpy(),
                      pool=pool, pool_f=pool_f, sig=sig)
    return P


def run(P, name, mode="trade", cost=1.0, rng=None, period=None):
    out = []
    f = fam(name)
    ex = EXIT[f]
    for sym, x in P.items():
        idx = np.where(x["sig"][name])[0]
        if rng is not None:
            pool = np.where(x[POOLX.get(f, "pool")] & np.isfinite(x["atr"]))[0]
            if len(idx) == 0 or len(pool) == 0:
                continue
            idx = np.sort(rng.choice(pool, size=min(len(idx), len(pool)), replace=False))
        for i in idx:
            tsi = pd.Timestamp(x["ts"][i])
            if (period == "dev" and tsi >= HOLDOUT) or (period == "hold" and tsi < HOLDOUT):
                continue
            if mode == "h5":
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
    rng = np.random.default_rng(37)
    full = build(daily, start)
    days = [d for d in daily["BTC/USDT"]["timestamp"] if d >= start + pd.Timedelta(days=30)]
    pick = sorted(set(rng.choice(len(days), size=min(n_days, len(days)), replace=False)))
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
            for nm in bad:
                a, b = bool(x["sig"][nm][i]), bool(part[sym]["sig"][nm][-1])
                bad[nm] += a != b
                nsig[nm] += a or b
        if k % 10 == 0:
            print(f"  sachai {k}/{len(pick)}", flush=True)
    emit(f"\n# SACHAI TEST - {len(pick)} din, data kaat kar (farq 0 hona chahiye)")
    for nm in bad:
        emit(f"{nm:>18}: farq {bad[nm]} ({nsig[nm]} signals) -> {'SAHI' if bad[nm] == 0 else 'GHALAT - lookahead'}")
    return {nm: bad[nm] == 0 for nm in bad}


def main(daily=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    try:
        if daily is None:
            import data_fetcher
            data_fetcher.AUTO_TOP_N_COINS = S9.TOP_FETCH
            exch = data_fetcher.get_exchange()
            coins = [s for s in data_fetcher.get_coin_list(exch) if s.split("/")[0].upper() not in STABLES][:S9.TOP_FETCH]
            if "BTC/USDT" not in coins:
                coins.insert(0, "BTC/USDT")
            daily = {}
            for k, sym in enumerate(coins, 1):
                try:
                    df = fetch_full(exch, sym, "1d", S9.DAYS)
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
        emit("SEARCH LAB 14 - price action + contrarian qaidon mein | TAALA-BAND")
        emit("=" * 140)
        emit(f"Coins: {len(daily)} | DEV {t0.date()} -> {HOLDOUT.date()} ({dev_weeks:.0f} hafte) | TAALA {HOLDOUT.date()} -> {t1.date()}")
        emit("jeet/10 = 10 mein se nafa wali | PF = har 1 rupay nuqsan par kitna nafa | 5d fark = 5-din nafa minus random")

        ok_c = causality(daily, start, emit)
        P = build(daily, start)

        rows = {}
        emit("\n# DEV hissa")
        emit(f"{'entry':>16} | {'n':>5} | {'/hafta':>6} | {'5d fark':>7} | {'jeet/10':>7} | {'PF':>5} | {'rnd95':>5} | "
             f"{'folds (trades/PF)':>36} | {'boot5':>5} | {'-top10':>6} | {'2xcost':>6} | {'ausat':>6}")
        for nm in names():
            tr = run(P, nm, period="dev")
            if len(tr) < 25:
                emit(f"{nm:>16} | sirf {len(tr)} trades")
                continue
            h5 = run(P, nm, mode="h5", period="dev")
            rnd5 = [np.mean([t["ret"] for t in run(P, nm, mode="h5", period="dev", rng=np.random.default_rng(900 + q))])
                    for q in range(RND)]
            fark = np.mean([t["ret"] for t in h5]) - np.mean(rnd5)
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, period="dev", rng=np.random.default_rng(950 + q))]) for q in range(RND)]
            fo = L13.folds_ok(tr, t0, HOLDOUT)
            bp = S9.boot_p5(tr)
            mt = pf_of([t["ret"] for t in sorted(tr, key=lambda t: -t["ret"])[10:]])
            c2 = pf_of([t["ret"] for t in run(P, nm, period="dev", cost=2.0)])
            rows[nm] = dict(s=s, r95=np.percentile(rnd, 95), fo=fo, bp=bp, mt=mt, c2=c2, wk=s["n"] / dev_weeks, fark=fark)
            emit(f"{nm:>16} | {s['n']:>5} | {s['n']/dev_weeks:>6.2f} | {fark*100:>+6.2f}% | {s['win']/10:>7.1f} | {s['pf']:>5.2f} | "
                 f"{np.percentile(rnd,95):>5.2f} | {' '.join(f'{n}/{p:.2f}' for n, p, _ in fo):>36} | {bp:>5.2f} | {mt:>6.2f} | "
                 f"{c2:>6.2f} | {s['avg']:>+5.2f}%")

        emit("\n# DEV PASS / FAIL (beech wali setting)")
        passed = []
        for f, g in GRID.items():
            mid = f"{f} {g[1]}"
            nb = [f"{f} {v}" for k, v in enumerate(g) if k != 1]
            if mid not in rows:
                emit(f"{mid:>18}: FAIL (kam trades)")
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
            emit(f"{mid:>18}: {'DEV PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")

        emit("\n# TAALA KHULA (aakhri 12 mahine)")
        final = []
        for nm in ["REF_DIP"] + passed:
            tr = run(P, nm, period="hold")
            if len(tr) < 5:
                emit(f"{nm:>18}: taale mein sirf {len(tr)} trades")
                continue
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, period="hold", rng=np.random.default_rng(990 + q))]) for q in range(RND)]
            rnd = [x for x in rnd if np.isfinite(x)] or [np.inf]
            ok = s["pf"] >= 1.2 and s["pf"] > np.percentile(rnd, 95)
            days = len(set(str(pd.Timestamp(t["t_in"]).date()) for t in tr))
            if ok and nm != "REF_DIP":
                final.append(nm)
            emit(f"{nm:>18}: {s['n']} trades ({days} alag din) | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | random p95 "
                 f"{np.percentile(rnd,95):.2f} | ausat {s['avg']:+.2f}% -> {'TAALA PASS' if ok else 'TAALA FAIL'}"
                 + (" (sirf muqabla)" if nm == "REF_DIP" else ""))
        if not passed:
            emit("koi khayal DEV pass nahi hua - taala kisi ke liye nahi khula")
        for nm in final:
            tr = run(P, nm)
            for size in (0.10, 0.20):
                eq = portfolio(tr, closes, "fixed", size, max_pos=10, cap=1.0)[0]
                p = stats(eq)
                yrs = eq.resample("YE").last().pct_change().dropna()
                emit(f"  portfolio {nm} {int(size*100)}%: CAGR {p['cagr']*100:+.1f}% | DD {p['dd']*100:.1f}% | Sharpe {p['sharpe']:.2f} | saal "
                     + " ".join(f"{d.year}:{v*100:+.0f}" for d, v in yrs.items()))
        emit("\nNATIJA: " + (", ".join(final) + " -> PAPER BOT ke qabil" if final else "koi naya khayal taala paar nahi kar saka"))
    except Exception:
        emit("\n!! GHALTI (code crash):\n" + traceback.format_exc())

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
