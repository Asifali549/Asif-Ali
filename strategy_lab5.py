"""
STRATEGY LAB 5 - nayi talaash (user, 2026-10-05: "masla nayi strategy search karna hai")
=====================================================================================
Lookahead bug ke baad sirf Dip (daily, uptrend mein girawat) sach mein pass hai. Is liye usi "daily mean-reversion"
khandan mein 3 NAYE khayal (pehle kabhi test nahi), sab ke sath Dip wala tasdeeq-shuda exit:
  exit = TP +5% | close > SMA3 -> agle din open | SL signal close - 3 ATR(14) fixed | 10 din
  universe = point-in-time top-100 liquid, BTC close > EMA50, coin close > EMA200 (uptrend)

  R1 RESID   : BTC ke muqable "apni" girawat - 3 din ka residual return (coin - beta x BTC) ka z-score < Z
               (Z = -1.5 / -2.0 / -2.5). Coin market ke sath nahi, akela gira = ziada ulatne ka imkan.
  R2 LOSER   : har din eligible coins mein aaj ke sab se ziada girne wale K coins, agar girawat >= X
               (K3 X4% / K5 X4% / K3 X6%) - ziada signals.
  R3 PULLBACK: EMA20 > EMA50 > EMA200 mein close pehli dafa EMA-N se neeche (N = 10 / 20 / 30), close > EMA50.
  REF DIP    : live Dip v2 (RSI3 < 7, EMA50 > EMA200) - muqable ke liye.

SACHAI: har candidate ka signal 30 random din data kaat kar dobara - ek bhi farq = FAIL (lookahead).
PASS (pehle se tay, beech wali setting par; padosi 2 settings PF > random p95 aur OOS > 1.1):
  win >= 65 | PF > trend-random p95 | OOS (2025+) >= 1.3 | 4/4 folds PF > 1 | boot p5 >= 1.2 | -top10 >= 1.2 |
  2x kharcha >= 1.2 | >= 0.5 signals/hafta | DIP 50 / NEW 50 ka Sharpe > DIP akela | universe (80% coins x6) 5/6 PF >= 1.4
Natija: strategy_lab5_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema, rsi, atr_w, portfolio, stats, mix
from winrate_lab import sim, tstats, pf_of
import stop_fix_lab as SF

OUT = "strategy_lab5_RESULTS.txt"
TOP_FETCH = 200
DAYS = 2400
CFG = {"tp": 0.05}
RND_SEEDS = 12
SIZE = 0.20

GRID = {
    "R1_RESID": [("z-1.5", -1.5), ("z-2.0", -2.0), ("z-2.5", -2.5)],
    "R2_LOSER": [("K3 X4", (3, -0.04)), ("K5 X4", (5, -0.04)), ("K3 X6", (3, -0.06))],
    "R3_PULLBACK": [("EMA10", 10), ("EMA20", 20), ("EMA30", 30)],
}
MID = {"R1_RESID": 1, "R2_LOSER": 0, "R3_PULLBACK": 1}


# ------------------------------------------------------------------ signals (sab sirf band daily data se)
def context(daily):
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc_ok = bd > ema(bd, 50)
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    rank = dv.rank(axis=1, ascending=False)
    al100 = {day: set(row[row <= 100].index) for day, row in rank.iterrows()}
    return btc_ok, al100


def build(daily, btc_ok, al100, start):
    """Har coin: arrays + har candidate config ke signal arrays."""
    btc = daily["BTC/USDT"].set_index("timestamp")["close"]
    rb = np.log(btc).diff()
    P = {}
    ret1 = {}
    for sym, d in daily.items():
        c = d["close"]
        ts = d["timestamp"]
        e50, e200 = ema(c, 50), ema(c, 200)
        n_ok = np.arange(len(d)) >= 200
        bok = btc_ok.reindex(ts).fillna(False).to_numpy(bool)
        ok = np.array([sym in al100.get(t, ()) for t in ts]) & (ts >= start).to_numpy()
        up = ((c > e200).to_numpy() & n_ok)
        base = ok & bok & up
        # R1 residual
        rc = np.log(c).diff().to_numpy()
        rbb = rb.reindex(ts).to_numpy()
        s_rc, s_rb = pd.Series(rc), pd.Series(rbb)
        beta = (s_rc.rolling(60, min_periods=40).cov(s_rb) / s_rb.rolling(60, min_periods=40).var()).clip(-1, 3)
        res = s_rc - beta * s_rb
        z = res.rolling(3).sum() / (res.rolling(60, min_periods=40).std() * np.sqrt(3))
        r1 = (c / c.shift(1) - 1).to_numpy()
        ret1[sym] = pd.Series(np.where(base, r1, np.nan), index=ts)
        pb = {}
        for _, N in GRID["R3_PULLBACK"]:
            eN = ema(c, N)
            below = (c < eN)
            cond = below & ~below.shift(1, fill_value=True) & (eN > e50) & (e50 > e200) & (c > e50)
            if N >= 20:
                cond &= (ema(c, 20) > e50)
            pb[N] = cond.to_numpy() & base
        P[sym] = dict(sym=sym, d=d, ts=ts.to_numpy(), o=d["open"].to_numpy(float), h=d["high"].to_numpy(float),
                      l=d["low"].to_numpy(float), c=c.to_numpy(float), atr=atr_w(d).to_numpy(),
                      sma3=(c > c.rolling(3).mean()).to_numpy(), base=base, z=z.to_numpy(), r1=r1, pb=pb,
                      dip=(base & (e50 > e200).to_numpy() & (rsi(c, 3) < 7).to_numpy()), rsi3=rsi(c, 3).to_numpy())
    R = pd.DataFrame(ret1)
    rk = R.rank(axis=1, ascending=True, method="first")
    sigs = {}
    for sym, x in P.items():
        s = {"REF_DIP": x["dip"]}
        for lab, Z in GRID["R1_RESID"]:
            s[f"R1_RESID {lab}"] = x["base"] & (np.nan_to_num(x["z"], nan=0) < Z)
        rks = rk[sym].reindex(x["d"]["timestamp"]).to_numpy() if sym in rk else np.full(len(x["c"]), np.nan)
        for lab, (K, X) in GRID["R2_LOSER"]:
            s[f"R2_LOSER {lab}"] = x["base"] & (np.nan_to_num(rks, nan=1e9) <= K) & (np.nan_to_num(x["r1"], nan=0) <= X)
        for lab, N in GRID["R3_PULLBACK"]:
            s[f"R3_PULLBACK {lab}"] = x["pb"][N]
        x["sig"] = s
        x["prio"] = {"REF_DIP": -x["rsi3"], "R1": -x["z"], "R2": -x["r1"], "R3": -x["r1"]}
    return P


def names():
    out = ["REF_DIP"]
    for fam, g in GRID.items():
        out += [f"{fam} {lab}" for lab, _ in g]
    return out


def run(P, name, cost=1.0, rng=None, keep=None):
    out = []
    pk = "REF_DIP" if name == "REF_DIP" else name[:2]
    for sym, x in P.items():
        if keep is not None and sym not in keep:
            continue
        idx = np.where(x["sig"][name])[0]
        if rng is not None:
            pool = np.where(x["base"] & np.isfinite(x["atr"]))[0]
            if len(idx) == 0 or len(pool) == 0:
                continue
            idx = np.sort(rng.choice(pool, size=min(len(idx), len(pool)), replace=False))
        for i in idx:
            st = x["c"][i] - 3.0 * x["atr"][i]
            if not np.isfinite(st) or st <= 0:
                continue
            t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, None, x["sma3"], 10, CFG, cost)
            if t:
                pr = x["prio"][pk][i]
                t.update(sym=sym, prio=pr if np.isfinite(pr) else -9)
                out.append(t)
    return out


def causality(daily, start, emit, rng, n_days=30):
    btc_ok, al100 = context(daily)
    full = build(daily, btc_ok, al100, start)
    days = [d for d in daily["BTC/USDT"]["timestamp"] if d >= start + pd.Timedelta(days=30)]
    pick = sorted(rng.choice(len(days), size=min(n_days, len(days)), replace=False))
    bad = {n: 0 for n in names()}
    nsig = {n: 0 for n in names()}
    for k, j in enumerate(pick, 1):
        D = days[j]
        cut = {s: d[d["timestamp"] <= D].reset_index(drop=True) for s, d in daily.items()}
        cut = {s: d for s, d in cut.items() if len(d) > 0}
        b2, a2 = context(cut)
        part = build(cut, b2, a2, start)
        for sym, x in full.items():
            if sym not in part:
                continue
            w = np.where(x["ts"] == np.datetime64(D))[0]
            if len(w) == 0:
                continue
            i, y = w[0], part[sym]
            for n in bad:
                a, b = bool(x["sig"][n][i]), bool(y["sig"][n][-1])
                bad[n] += a != b
                nsig[n] += a or b
        if k % 10 == 0:
            print(f"  sachai {k}/{len(pick)}", flush=True)
    emit(f"\n# SACHAI TEST - {len(pick)} random din, data kaat kar dobara hisaab (farq 0 hona chahiye)")
    for n in bad:
        emit(f"{n:>22}: farq {bad[n]} ({nsig[n]} signals) -> {'SAHI' if bad[n] == 0 else 'GHALAT - lookahead'}")
    return {n: bad[n] == 0 for n in bad}


def folds(tr, t0, t1):
    edges = pd.date_range(t0, t1, periods=5)
    return [pf_of([t["ret"] for t in tr if a <= pd.Timestamp(t["t_in"]) < b]) for a, b in zip(edges[:-1], edges[1:])]


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

        btc_ok, al100 = context(daily)
        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        start = closes.index[0] + pd.Timedelta(days=210)
        closes = closes[closes.index >= start]
        t0, t1 = closes.index[0], closes.index[-1]
        weeks = (t1 - t0).days / 7
        emit("=" * 125)
        emit("STRATEGY LAB 5 - daily mean-reversion ke 3 naye khayal (Dip exit), top-100, BTC>EMA50, uptrend, 20% / trade, max 10")
        emit("=" * 125)
        emit(f"Coins: {len(daily)} | {t0.date()} -> {t1.date()} ({weeks:.0f} hafte) | win = +0.5% se ziada | OOS = 2025+")

        ok_caus = causality(daily, start, emit, np.random.default_rng(11))
        P = build(daily, btc_ok, al100, start)

        curves, rows = {}, {}
        emit(f"\n{'candidate':>22} | {'n':>4} | {'/hafta':>6} | {'win%':>5} | {'PF':>5} | {'rnd95':>5} | {'OOS':>5} | {'folds':>23} | "
             f"{'boot5':>5} | {'-top10':>6} | {'2xcost':>6} | {'CAGR':>7} | {'MaxDD':>6} | {'Sharpe':>6} | {'+mah%':>5} | {'bura mah':>8}")
        for nm in names():
            tr = run(P, nm)
            if len(tr) < 20:
                emit(f"{nm:>22} | sirf {len(tr)} trades")
                continue
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, rng=np.random.default_rng(100 + q))]) for q in range(RND_SEEDS)]
            fo = folds(tr, t0, t1)
            bp = boot_p5(tr)
            mt = pf_of([t["ret"] for t in sorted(tr, key=lambda t: -t["ret"])[10:]])
            c2 = pf_of([t["ret"] for t in run(P, nm, cost=2.0)])
            eq = portfolio(tr, closes, "fixed", SIZE)[0]
            p = stats(eq)
            curves[nm] = eq
            rows[nm] = dict(s=s, r95=np.percentile(rnd, 95), fo=fo, bp=bp, mt=mt, c2=c2, p=p, wk=s["n"] / weeks)
            emit(f"{nm:>22} | {s['n']:>4} | {s['n']/weeks:>6.2f} | {s['win']:>5.1f} | {s['pf']:>5.2f} | {np.percentile(rnd,95):>5.2f} | "
                 f"{s['oos']:>5.2f} | {' '.join(f'{f:>5.2f}' for f in fo):>23} | {bp:>5.2f} | {mt:>6.2f} | {c2:>6.2f} | "
                 f"{p['cagr']*100:>+6.1f}% | {p['dd']*100:>5.1f}% | {p['sharpe']:>6.2f} | {p['pos_months']:>5.0f} | {p['worst_month']*100:>7.1f}%")

        years = list(range(2021, 2027))
        emit("\n# SAAL-WAR NAFA (akela, 20% / trade)")
        emit(f"{'candidate':>22} | " + " | ".join(f"{y:>5}" for y in years))
        for nm, r in rows.items():
            yr = {d.year: v for d, v in r["p"]["yearly"].items()}
            emit(f"{nm:>22} | " + " | ".join(f"{yr.get(y, np.nan)*100:>+4.0f}%" for y in years))

        emit("\n# PORTFOLIO: DIP akela vs DIP 50 / NEW 50 (mahana rebalance)")
        dp = rows["REF_DIP"]["p"]
        emit(f"{'DIP akela':>22} | CAGR {dp['cagr']*100:+.1f}% | MaxDD {dp['dd']*100:.1f}% | Sharpe {dp['sharpe']:.2f}")
        dipm = curves["REF_DIP"].resample("ME").last().pct_change().dropna()
        port = {}
        for nm in rows:
            if nm == "REF_DIP":
                continue
            m = stats(mix(curves, {"REF_DIP": .5, nm: .5}))
            corr = curves[nm].resample("ME").last().pct_change().dropna().corr(dipm)
            port[nm] = m["sharpe"] > dp["sharpe"]
            emit(f"{'DIP 50 / ' + nm:>34} | corr {corr:+.2f} | CAGR {m['cagr']*100:+.1f}% | MaxDD {m['dd']*100:.1f}% | Sharpe {m['sharpe']:.2f}")

        emit("\n# PASS / FAIL (beech wali setting; padosi: PF > random p95 aur OOS > 1.1)")
        syms = list(P)
        for fam, g in GRID.items():
            mid = f"{fam} {g[MID[fam]][0]}"
            nb = [f"{fam} {lab}" for k, (lab, _) in enumerate(g) if k != MID[fam]]
            if mid not in rows:
                emit(f"{mid}: FAIL (kam trades)")
                continue
            r, s = rows[mid], rows[mid]["s"]
            chk = {"sachai": all(ok_caus.get(f"{fam} {lab}", False) for lab, _ in g),
                   "win>=65": s["win"] >= 65, "PF>rnd95": s["pf"] > r["r95"], "OOS>=1.3": s["oos"] >= 1.3,
                   "folds4/4": all(np.isfinite(f) and f > 1 for f in r["fo"]), "boot>=1.2": r["bp"] >= 1.2,
                   "-top10>=1.2": r["mt"] >= 1.2, "2xcost>=1.2": r["c2"] >= 1.2, ">=0.5/hafta": r["wk"] >= 0.5,
                   "portfolio+": port.get(mid, False),
                   "padosi": all(n in rows and rows[n]["s"]["pf"] > rows[n]["r95"] and rows[n]["s"]["oos"] > 1.1 for n in nb)}
            fails = [k for k, v in chk.items() if not v]
            uni = ""
            if not fails:
                good, det = 0, []
                for q in range(6):
                    g2 = np.random.default_rng(600 + q)
                    keep = set(g2.choice(syms, int(len(syms) * 0.8), replace=False)) | {"BTC/USDT"}
                    su = tstats(run(P, mid, keep=keep))
                    det.append(f"{su['pf']:.2f}/{su['oos']:.2f}")
                    good += su["pf"] >= 1.4 and su["oos"] >= 1.1
                uni = f" | universe {good}/6 ({' '.join(det)})"
                if good < 5:
                    fails.append("universe")
            emit(f"{mid:>22}: {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}{uni}")
    except Exception:
        emit("\n!! GHALTI (code crash):\n" + traceback.format_exc())

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
