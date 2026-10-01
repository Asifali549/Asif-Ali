"""
NAYE KHAYAL - Donchian khandan se bahar, portfolio ke liye (2026-10-02)
=====================================================================
Teen alag khayal, sab Daily, spot buy-only, entry agle din ke OPEN par, stop = signal close - 3 x ATR(14) fixed:
 A) CATCH-UP (lead-lag): BTC ne band din par >= X% chhalang lagai, coin ka usi din return BTC ke aadhe se kam
    (peeche reh gaya) -> agle din khareedo, H din baad close par becho.  Grid X {3,4,6}% x H {3,5,7}.
    Kai coins hon to sab se ziada peeche wala pehle.
 B) MARKET PANIC: top-100 coins ka MEDIAN rozana return <= Z (Z {-5,-7,-9}%) -> agle din sab se liquid coins
    khareedo, H din baad becho.  Grid Z x H {3,5,7}.
 C) VOLUME CAPITULATION: coin close > EMA200 (uptrend), din ka return <= -R (R {6,8,10}%) aur volume >= V x pichle
    20 din ka ausat (V {2,2.5,3}) -> agle din khareedo; exit close > SMA5 (agle din open) ya 10 din.
Universe: point-in-time top-100 liquidity. Kharcha: fee 0.1% + slip 0.05% har taraf + stop slip 0.25%.
RANDOM baseline (20 seeds): har coin mein utni hi trades, random din par - A/B: coin ke kisi bhi top-100 din par,
   C: sirf us coin ke uptrend (close > EMA200) top-100 din par (sirf trigger ka faida naapne ke liye).
Apna edge PASS: PF > random p95, PF >= 1.10, OOS (2025+) PF > 1, n >= 100.  Plateau: 9 mein se 6+.
PORTFOLIO (beech wala config pehle se chuna, behtareen nahi): naya sleeve har trade 10%, max 10;
   ICHI 60 / DIP(20%) 40 vs ICHI 50 / DIP 30 / NAYA 20 - Sharpe, MaxDD, correlation.
Natija: new_ideas_RESULTS.txt
"""
import warnings

import numpy as np
import pandas as pd

import portfolio_lab as P
from bot_core import fetch_full, norm, STABLES

OUT = "new_ideas_RESULTS.txt"
warnings.filterwarnings("ignore")
N_RND = 20
OOS = pd.Timestamp("2025-01-01")
YEARS = list(range(2021, 2027))
GRID = {
    "A": [(x, h) for x in (0.03, 0.04, 0.06) for h in (3, 5, 7)],
    "B": [(z, h) for z in (-0.05, -0.07, -0.09) for h in (3, 5, 7)],
    "C": [(r, v) for r in (0.06, 0.08, 0.10) for v in (2.0, 2.5, 3.0)],
}
CENTER = {"A": (0.04, 5), "B": (-0.07, 5), "C": (0.08, 2.5)}
NAMES = {"A": "CATCH-UP (BTC lead-lag)", "B": "MARKET PANIC", "C": "VOLUME CAPITULATION"}


def pf_of(r):
    r = np.asarray(r, float)
    g, lo = r[r > 0].sum(), -r[r < 0].sum()
    return np.nan if len(r) == 0 else (np.inf if lo == 0 else g / lo)


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

    # ---- din-level series ----
    btc_ret = bd.pct_change()
    rets = pd.DataFrame({s: d.set_index("timestamp")["close"].pct_change() for s, d in daily.items()}).sort_index()
    in_uni = pd.DataFrame({s: [s in allowed.get(D, ()) for D in rets.index] for s in rets.columns}, index=rets.index)
    med_ret = rets.where(in_uni).median(axis=1)

    # ---- coin arrays ----
    A = {}
    for s, d in daily.items():
        if s == "BTC/USDT" or len(d) < 260:
            continue
        ts = d["timestamp"]
        c = d["close"]
        A[s] = dict(
            o=d["open"].to_numpy(float), h=d["high"].to_numpy(float), l=d["low"].to_numpy(float), c=c.to_numpy(float),
            ts=ts.to_numpy(), day=ts,
            stop=(c - 3.0 * P.atr_w(d)).to_numpy(),
            ret=c.pct_change().to_numpy(),
            up=(c > P.ema(c, 200)).to_numpy() & (np.arange(len(c)) >= 200),
            vrat=(d["volume"] / d["volume"].shift(1).rolling(20).mean()).to_numpy(),
            sma_exit=(c > c.rolling(5).mean()).to_numpy(),
            uni=np.array([s in allowed.get(D, ()) for D in ts]),
            btc=btc_ret.reindex(ts).to_numpy(), med=med_ret.reindex(ts).to_numpy(),
            liq=dv[s].reindex(ts).to_numpy(),
        )
        A[s]["uni"][:210] = False

    def signals(idea, p):
        out = {}
        for s, a in A.items():
            if idea == "A":
                x, _ = p
                m = a["uni"] & (a["btc"] >= x) & (a["ret"] < 0.5 * a["btc"])
                prio = a["btc"] - a["ret"]
            elif idea == "B":
                z, _ = p
                m = a["uni"] & (a["med"] <= z)
                prio = a["liq"]
            else:
                r, v = p
                m = a["uni"] & a["up"] & (a["ret"] <= -r) & (a["vrat"] >= v)
                prio = -a["ret"]
            out[s] = (np.flatnonzero(np.nan_to_num(m, nan=0).astype(bool)), prio)
        return out

    def pool_of(idea, a):
        return np.flatnonzero(a["uni"] & a["up"]) if idea == "C" else np.flatnonzero(a["uni"])

    def sim(idea, p, sig, rng=None):
        tr = []
        for s, a in A.items():
            idx, prio = sig[s]
            if rng is not None:
                pool = pool_of(idea, a)
                k = min(len(idx), len(pool))
                if k == 0:
                    continue
                idx = np.sort(rng.choice(pool, k, replace=False))
            busy = -1
            for i in idx:
                if i <= busy:
                    continue
                if idea == "C":
                    t = P.sim_one(a["o"], a["h"], a["l"], a["c"], a["ts"], i, a["stop"][i], None, a["sma_exit"], None, 10)
                else:
                    t = P.sim_one(a["o"], a["h"], a["l"], a["c"], a["ts"], i, a["stop"][i], None, None, None, p[1])
                if t:
                    pr = prio[i] if np.isfinite(prio[i]) else -9
                    t.update(sym=s, prio=float(pr))
                    tr.append(t)
                    busy = int(np.searchsorted(a["ts"], t["t_out"]))
        return [t for t in tr if pd.Timestamp(t["t_in"]) >= start]

    weeks = (closes_p.index[-1] - closes_p.index[0]).days / 7
    emit("=" * 150)
    emit("NAYE KHAYAL (Donchian se bahar) - apna edge vs random, phir ICHI 60 / DIP 40 portfolio mein hissa")
    emit("=" * 150)
    emit(f"Coins: {len(A)} (BTC bahar) | {closes_p.index[0].date()} -> {closes_p.index[-1].date()} | top-{P.UNIVERSE} point-in-time | "
         f"random {N_RND} seeds | OOS {OOS.date()} se")

    chosen = {}
    summary = {}
    for idea in ("A", "B", "C"):
        emit(f"\n{'#' * 150}\n{idea}) {NAMES[idea]}\n{'#' * 150}")
        emit(f"{'Params':>16} | {'Trades':>6} | {'/hafta':>6} | {'Win%':>5} | {'Avg%':>6} | {'PF':>5} | {'RndP50':>6} | {'RndP95':>6} | "
             f"{'OOS':>5} | " + " | ".join(str(y) for y in YEARS) + " | Faisla")
        n_ok = 0
        for p in GRID[idea]:
            sig = signals(idea, p)
            tr = sim(idea, p, sig)
            r = np.array([t["ret"] for t in tr])
            rpf = [pf_of([t["ret"] for t in sim(idea, p, sig, np.random.default_rng(sd))]) for sd in range(N_RND)]
            p50, p95 = np.nanmedian(rpf), np.nanpercentile(rpf, 95)
            tin = pd.to_datetime([t["t_in"] for t in tr])
            oos = pf_of(r[tin >= OOS]) if len(r) else np.nan
            yr = {y: pf_of(r[tin.year == y]) for y in YEARS}
            pf = pf_of(r)
            ok = len(r) >= 100 and pf > p95 and pf >= 1.10 and oos > 1
            n_ok += ok
            lab = (f"X{p[0]*100:.0f}% H{p[1]}" if idea == "A" else f"Z{p[0]*100:.0f}% H{p[1]}" if idea == "B"
                   else f"R{p[0]*100:.0f}% V{p[1]}")
            emit(f"{lab:>16} | {len(r):>6} | {len(r)/weeks:>6.2f} | {(r > 0).mean()*100 if len(r) else 0:>5.1f} | "
                 f"{r.mean()*100 if len(r) else 0:>+6.2f} | {pf:>5.2f} | {p50:>6.2f} | {p95:>6.2f} | {oos:>5.2f} | "
                 + " | ".join(f"{yr[y]:>4.2f}" for y in YEARS) + f" | {'PASS' if ok else '-'}")
            if p == CENTER[idea]:
                chosen[idea] = tr
        summary[idea] = n_ok
        emit(f"{idea} apna edge: {n_ok}/{len(GRID[idea])} grid cells PASS")

    # ---- portfolio ----
    emit(f"\n{'#' * 150}\nPORTFOLIO - beech wala config (pehle se chuna): " + ", ".join(f"{k} {CENTER[k]}" for k in CENTER) + f"\n{'#' * 150}")
    T = {"ICHI": [t for t in P.trades_ichi(h4, allowed) if pd.Timestamp(t["t_in"]) >= start],
         "DIP": [t for t in P.trades_dip(daily, allowed, btc_ok) if pd.Timestamp(t["t_in"]) >= start]}
    curves = {"ICHI": P.portfolio(T["ICHI"], closes_p, "risk", 0.01)[0], "DIP": P.portfolio(T["DIP"], closes_p, "fixed", 0.20)[0]}
    for k, tr in chosen.items():
        curves[k] = P.portfolio(tr, closes_p, "fixed", 0.10)[0]
    mixes = {"BASE ICHI 60 / DIP 40": {"ICHI": .6, "DIP": .4}}
    for k in chosen:
        mixes[f"ICHI 50 / DIP 30 / {k} 20"] = {"ICHI": .5, "DIP": .3, k: .2}
    emit(f"{'Portfolio':>26} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'+mahine':>7} | {'Bura mah':>8} | "
         + " | ".join(str(y) for y in YEARS))
    port = {}
    for name, eq in list(curves.items()) + [(n, P.mix(curves, w)) for n, w in mixes.items()]:
        s = P.stats(eq)
        yr = {d.year: v for d, v in s["yearly"].items()}
        port[name] = s
        emit(f"{name:>26} | {s['cagr']*100:>+6.1f}% | {s['dd']*100:>6.1f}% | {s['sharpe']:>6.2f} | {s['pos_months']:>6.0f}% | "
             f"{s['worst_month']*100:>+7.1f}% | " + " | ".join(f"{yr.get(y, np.nan)*100:>+4.0f}%" for y in YEARS))
    mo = pd.DataFrame({k: P.stats(v)["monthly"] for k, v in curves.items()})
    emit("\nMahana correlation (0 = alag chalte, 1 = sath):")
    emit(mo.corr().round(2).to_string())

    emit("\n" + "=" * 150 + "\nKHULASA\n" + "=" * 150)
    b = port["BASE ICHI 60 / DIP 40"]
    emit(f"BASE ICHI 60 / DIP 40: CAGR {b['cagr']*100:+.1f}% | MaxDD {b['dd']*100:.1f}% | Sharpe {b['sharpe']:.2f}")
    for k in ("A", "B", "C"):
        nm = f"ICHI 50 / DIP 30 / {k} 20"
        x = port.get(nm)
        helps = x is not None and x["sharpe"] >= b["sharpe"] + 0.05 and x["dd"] >= b["dd"] - 0.02
        verdict = "PASS (edge + portfolio behtar)" if summary[k] >= 6 and helps else "FAIL"
        emit(f"{k}) {NAMES[k]}: apna edge {summary[k]}/9 | portfolio: "
             + (f"Sharpe {x['sharpe']:.2f}, DD {x['dd']*100:.1f}%" if x else "N/A") + f" -> {verdict}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
