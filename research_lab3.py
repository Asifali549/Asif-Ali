"""
RESEARCH LAB 3 - "ooncha win-rate" khandan ko barhana + sakht universe-sensitivity (2026-10-02)
==============================================================================================
Sabaq (lab 2 / capitulation): naye khandan fail; aur coin list badalne se kam-trade strategy ulat sakti hai.
Is liye: jo khandan SABIT hai (uptrend mein girawat khareedo = Dip, win ~70%) usi ke ALAG triggers + Ichimoku
ko Daily par + Dip ko 12H par. Har ek pehle akela parkha jata hai, phir Dip ke sath "ensemble".
Filters sab mein Dip jaise (muqable ke liye): coin close > EMA200 aur EMA50 > EMA200, BTC close > EMA50, top-100.
Exit (MR): close > SMA5 -> agle open; stop = signal close - 3 ATR(14) fixed; max 10 bars.
 CONTROL DIP : RSI(3) < {5, 10, 15}                       (maujooda live - framework ki jaanch)
 M1 IBS      : close din ki range ke neeche {10, 15, 20}% mein AUR close < kal ka close
 M2 STREAK   : lagataar {3, 4, 5} din neeche close
 M3 STRETCH  : close < SMA10 x (1 - {5, 8, 12}%)
 M4 DIP 12H  : 12H candle par RSI(3) < {5, 10, 15}, filters kal ke BAND din se; exit 12H close > SMA5; max 20 bars
 T1 ICHI 1D  : Ichimoku + Market Structure (live 4H wale hi usool) DAILY candle par; volume mult {1.5, 2.0, 2.5};
               exit chandelier 16 / 5.5 ATR + TP 3R, max 120
RANDOM (20 seeds): utni hi trades, usi pool mein (MR: uptrend+BTC ok din; T1: top-100 din).
CELL PASS: n >= 100, PF >= 1.3, PF > random p95, OOS (2025+) PF > 1.   PLATEAU: 3 mein se 2+.
VALIDATION (beech wala cell): bootstrap p5 > 1, top-10 hata kar PF > 1, 4 folds mein 3+, kharcha 2x PF > 1.15,
 UNIVERSE SENSITIVITY: 6 baar 20% coins random hata kar - PF > 1.2 AUR OOS > 1 kam az kam 5/6 mein.
PORTFOLIO: ICHI 60 / DIP 40 (base) vs ICHI 60 / [DIP + naya] 40 (ensemble sleeve, ek coin ek waqt) vs ICHI 50/DIP 30/NAYA 20.
Natija: research_lab3_RESULTS.txt
"""
import warnings

import numpy as np
import pandas as pd

import portfolio_lab as P
from bot_core import fetch_full, norm, STABLES, ichi_signal, ICHI_BASE, chandelier

warnings.filterwarnings("ignore")
OUT = "research_lab3_RESULTS.txt"
N_RND = 20
OOS = pd.Timestamp("2025-01-01")
YEARS = list(range(2021, 2027))
GRID = {"DIP": (5, 10, 15), "M1": (0.10, 0.15, 0.20), "M2": (3, 4, 5), "M3": (0.05, 0.08, 0.12),
        "M4": (5, 10, 15), "T1": (1.5, 2.0, 2.5)}
NAMES = {"DIP": "CONTROL DIP RSI3", "M1": "IBS low close", "M2": "DOWN STREAK", "M3": "STRETCH < SMA10",
         "M4": "DIP 12H", "T1": "ICHIMOKU DAILY"}
LAB = {"DIP": "RSI3<{}", "M1": "IBS<{:.2f}", "M2": "{} din neeche", "M3": "SMA10-{:.0%}", "M4": "12H RSI3<{}",
       "T1": "vol x{}"}


def pf_of(r):
    r = np.asarray(r, float)
    g, lo = r[r > 0].sum(), -r[r < 0].sum()
    return np.nan if len(r) == 0 else (np.inf if lo == 0 else g / lo)


def resample(d4, rule):
    x = d4.set_index("timestamp")
    out = x.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
    return out.iloc[:-1].reset_index(drop=True)


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
    weeks = (closes_p.index[-1] - closes_p.index[0]).days / 7

    # ---------- daily arrays ----------
    A = {}
    for s, d in daily.items():
        if s == "BTC/USDT" or len(d) < 260:
            continue
        ts, c, h, l = d["timestamp"], d["close"], d["high"], d["low"]
        e50, e200 = P.ema(c, 50), P.ema(c, 200)
        a = dict(o=d["open"].to_numpy(float), h=h.to_numpy(float), l=l.to_numpy(float), c=c.to_numpy(float),
                 ts=ts.to_numpy(), stop=(c - 3.0 * P.atr_w(d)).to_numpy(),
                 pool=((c > e200) & (e50 > e200)).to_numpy() & (np.arange(len(c)) >= 200)
                 & btc_ok.reindex(ts).fillna(False).to_numpy(bool),
                 uni=np.array([s in allowed.get(D, ()) for D in ts]),
                 rsi=P.rsi(c, 3).to_numpy(), ibs=((c - l) / (h - l).replace(0, np.nan)).to_numpy(),
                 dnc=(c < c.shift(1)).to_numpy(), sma10=c.rolling(10).mean().to_numpy(),
                 ex5=(c > c.rolling(5).mean()).to_numpy(), d=d)
        dn = (c < c.shift(1)).astype(int)
        for n in (3, 4, 5):
            a[f"st{n}"] = (dn.rolling(n).sum() == n).to_numpy()
        a["uni"][:210] = False
        a["pool"] &= a["uni"]
        A[s] = a

    # ---------- 12H arrays ----------
    H = {}
    for s, d4 in h4.items():
        if s not in A:
            continue
        d = resample(d4, "12h")
        if len(d) < 600:
            continue
        c = d["close"]
        day_prev = d["timestamp"].dt.floor("1D") - pd.Timedelta(days=1)
        dd = A[s]["d"].set_index("timestamp")
        e50, e200 = P.ema(dd["close"], 50), P.ema(dd["close"], 200)
        upd = ((dd["close"] > e200) & (e50 > e200)) & pd.Series(np.arange(len(dd)) >= 200, index=dd.index)
        upd &= btc_ok.reindex(dd.index).fillna(False)
        pool = upd.reindex(day_prev).fillna(False).to_numpy(bool) & np.array([s in allowed.get(D, ()) for D in d["timestamp"].dt.floor("1D")])
        pool &= (d["timestamp"] >= closes.index[0] + pd.Timedelta(days=210)).to_numpy()
        H[s] = dict(o=d["open"].to_numpy(float), h=d["high"].to_numpy(float), l=d["low"].to_numpy(float), c=c.to_numpy(float),
                    ts=d["timestamp"].to_numpy(), stop=(c - 3.0 * P.atr_w(d)).to_numpy(), pool=pool,
                    rsi=P.rsi(c, 3).to_numpy(), ex5=(c > c.rolling(5).mean()).to_numpy())

    # ---------- daily ichimoku ----------
    ICH = {}
    for s, a in A.items():
        d = a["d"]
        ICH[s] = dict(trail=chandelier(d, 16, 5.5), sig={vm: np.asarray(ichi_signal(d, {**ICHI_BASE, "vol_mult": vm}), bool)
                                                        for vm in GRID["T1"]})

    def nz(x):
        return np.nan_to_num(x, nan=0).astype(bool)

    def mask(e, p, s):
        if e == "M4":
            b = H[s]
            return b["pool"] & nz(b["rsi"] < p), b["pool"], -b["rsi"]
        a = A[s]
        if e == "T1":
            return a["uni"] & ICH[s]["sig"][p], a["uni"], np.zeros(len(a["c"]))
        if e == "DIP":
            m = nz(a["rsi"] < p)
        elif e == "M1":
            m = nz(a["ibs"] < p) & nz(a["dnc"])
        elif e == "M2":
            m = nz(a[f"st{p}"])
        else:
            m = nz(a["c"] < a["sma10"] * (1 - p))
        return a["pool"] & m, a["pool"], -a["rsi"]

    def one(e, s, i):
        if e == "M4":
            b = H[s]
            return P.sim_one(b["o"], b["h"], b["l"], b["c"], b["ts"], i, b["stop"][i], None, b["ex5"], None, 20)
        a = A[s]
        if e == "T1":
            tr = ICH[s]["trail"]
            return P.sim_one(a["o"], a["h"], a["l"], a["c"], a["ts"], i, tr[i], tr, None, 3.0, 120)
        return P.sim_one(a["o"], a["h"], a["l"], a["c"], a["ts"], i, a["stop"][i], None, a["ex5"], None, 10)

    def syms_for(e):
        return list(H) if e == "M4" else list(A)

    def sim(e, p, rng=None, cost=1.0, coins=None):
        old = (P.SLIP, P.FEE, P.STOP_SLIP)
        P.SLIP, P.FEE, P.STOP_SLIP = old[0] * cost, old[1] * cost, old[2] * cost
        out = []
        try:
            for s in (coins if coins is not None else syms_for(e)):
                if e == "M4" and s not in H:
                    continue
                m, pool, prio = mask(e, p, s)
                ts = H[s]["ts"] if e == "M4" else A[s]["ts"]
                idx = np.flatnonzero(m)
                if rng is not None:
                    pidx = np.flatnonzero(pool)
                    k = min(len(idx), len(pidx))
                    if k == 0:
                        continue
                    idx = np.sort(rng.choice(pidx, k, replace=False))
                busy = -1
                for i in idx:
                    if i <= busy:
                        continue
                    t = one(e, s, i)
                    if t:
                        t.update(sym=s, prio=float(prio[i]) if np.isfinite(prio[i]) else -9)
                        out.append(t)
                        busy = int(np.searchsorted(ts, t["t_out"]))
        finally:
            P.SLIP, P.FEE, P.STOP_SLIP = old
        return [t for t in out if pd.Timestamp(t["t_in"]) >= start]

    def stats_r(tr):
        r = np.array([t["ret"] for t in tr])
        tin = pd.to_datetime([t["t_in"] for t in tr])
        return r, tin

    emit("=" * 150)
    emit("RESEARCH LAB 3 - ooncha win-rate khandan + Ichimoku Daily + universe sensitivity")
    emit("=" * 150)
    emit(f"Coins: {len(A)} daily / {len(H)} 12H (BTC bahar) | {closes_p.index[0].date()} -> {closes_p.index[-1].date()} | "
         f"top-{P.UNIVERSE} point-in-time | random {N_RND} | OOS {OOS.date()} se")

    center, verdict = {}, {}
    rng_u = np.random.default_rng(99)
    for e, grid in GRID.items():
        emit(f"\n{'#' * 150}\n{e}) {NAMES[e]}\n{'#' * 150}")
        emit(f"{'Params':>14} | {'Trades':>6} | {'/hafta':>6} | {'Win%':>5} | {'Avg%':>6} | {'PF':>5} | {'RndP50':>6} | {'RndP95':>6} | "
             f"{'OOS':>5} | " + " | ".join(str(y) for y in YEARS) + " | Faisla")
        n_ok = 0
        for p in grid:
            tr = sim(e, p)
            r, tin = stats_r(tr)
            rp = [pf_of([t["ret"] for t in sim(e, p, np.random.default_rng(sd))]) for sd in range(N_RND)]
            p50, p95 = np.nanmedian(rp), np.nanpercentile(rp, 95)
            pf, oos = pf_of(r), pf_of(r[tin >= OOS]) if len(r) else np.nan
            ok = len(r) >= 100 and pf >= 1.3 and pf > p95 and oos > 1
            n_ok += ok
            emit(f"{LAB[e].format(p):>14} | {len(r):>6} | {len(r)/weeks:>6.2f} | {(r > 0).mean()*100 if len(r) else 0:>5.1f} | "
                 f"{r.mean()*100 if len(r) else 0:>+6.2f} | {pf:>5.2f} | {p50:>6.2f} | {p95:>6.2f} | {oos:>5.2f} | "
                 + " | ".join(f"{pf_of(r[tin.year == y]):>4.2f}" for y in YEARS) + f" | {'PASS' if ok else '-'}")
            if p == grid[1]:
                center[e] = tr
        tr = center[e]
        r, tin = stats_r(tr)
        if len(r) >= 30:
            rb = np.random.default_rng(5)
            bs = np.percentile([pf_of(rb.choice(r, len(r))) for _ in range(1000)], 5)
            top = pf_of(np.sort(r)[:-10])
            folds = [pf_of(r[part]) for part in np.array_split(np.argsort(tin.values), 4)]
            c2 = pf_of([t["ret"] for t in sim(e, grid[1], cost=2.0)])
            subs = []
            pool_syms = syms_for(e)
            for _ in range(6):
                keep = list(rng_u.choice(pool_syms, int(len(pool_syms) * 0.8), replace=False))
                rr, tt = stats_r(sim(e, grid[1], coins=keep))
                subs.append((pf_of(rr), pf_of(rr[tt >= OOS])))
            uni_ok = sum(pf > 1.2 and o > 1 for pf, o in subs)
        else:
            bs = top = c2 = np.nan
            folds, subs, uni_ok = [np.nan] * 4, [], 0
        val = bs > 1 and top > 1 and sum(f > 1 for f in folds) >= 3 and c2 > 1.15 and uni_ok >= 5
        verdict[e] = dict(plateau=n_ok, val=val)
        emit(f"{e} plateau: {n_ok}/3 | validation ({LAB[e].format(grid[1])}): bootstrap p5 {bs:.2f} | top-10 hata kar {top:.2f} | "
             f"folds " + "/".join(f"{f:.2f}" for f in folds) + f" | kharcha 2x {c2:.2f} | universe 80%: "
             + " ".join(f"{a:.2f}/{b:.2f}" for a, b in subs) + f" ({uni_ok}/6) -> {'PASS' if val else 'FAIL'}")

    # ---------- portfolio ----------
    emit(f"\n{'#' * 150}\nPORTFOLIO (beech wale cells)\n{'#' * 150}")
    ichi = [t for t in P.trades_ichi(h4, allowed) if pd.Timestamp(t["t_in"]) >= start]
    curves = {"ICHI": P.portfolio(ichi, closes_p, "risk", 0.01)[0]}
    for e, tr in center.items():
        curves[e] = P.portfolio(tr, closes_p, "risk" if e == "T1" else "fixed", 0.01 if e == "T1" else 0.20)[0]
    ens = {}
    for e in ("M1", "M2", "M3", "M4"):
        merged = sorted(center["DIP"] + center[e], key=lambda t: pd.Timestamp(t["t_in"]))
        curves[f"DIP+{e}"] = P.portfolio(merged, closes_p, "fixed", 0.20)[0]
        ens[e] = merged
    mixes = {"BASE ICHI 60 / DIP 40": {"ICHI": .6, "DIP": .4}}
    for e in ("M1", "M2", "M3", "M4"):
        mixes[f"ICHI 60 / (DIP+{e}) 40"] = {"ICHI": .6, f"DIP+{e}": .4}
    for e in GRID:
        if e != "DIP":
            mixes[f"ICHI 50 / DIP 30 / {e} 20"] = {"ICHI": .5, "DIP": .3, e: .2}
    emit("Akele sleeves:")
    for k, eq in curves.items():
        s = P.stats(eq)
        emit(f"{k:>12} | CAGR {s['cagr']*100:+6.1f}% | MaxDD {s['dd']*100:6.1f}% | Sharpe {s['sharpe']:.2f} | +mahine {s['pos_months']:.0f}%")
    emit(f"\n{'Portfolio':>30} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'+mahine':>7} | " + " | ".join(str(y) for y in YEARS))
    port = {}
    for name, w in mixes.items():
        s = P.stats(P.mix(curves, w))
        port[name] = s
        yr = {d.year: v for d, v in s["yearly"].items()}
        emit(f"{name:>30} | {s['cagr']*100:>+6.1f}% | {s['dd']*100:>6.1f}% | {s['sharpe']:>6.2f} | {s['pos_months']:>6.0f}% | "
             + " | ".join(f"{yr.get(y, np.nan)*100:>+4.0f}%" for y in YEARS))
    mo = pd.DataFrame({k: P.stats(v)["monthly"] for k, v in curves.items() if "+" not in k})
    emit("\nMahana correlation:\n" + mo.corr().round(2).to_string())

    emit("\n" + "=" * 150 + "\nKHULASA\n" + "=" * 150)
    b = port["BASE ICHI 60 / DIP 40"]
    emit(f"BASE ICHI 60 / DIP 40: CAGR {b['cagr']*100:+.1f}% | MaxDD {b['dd']*100:.1f}% | Sharpe {b['sharpe']:.2f}")
    for e in GRID:
        v = verdict[e]
        line = f"{e} {NAMES[e]:>18}: plateau {v['plateau']}/3 | validation {'PASS' if v['val'] else 'FAIL'}"
        for name, s in port.items():
            if name.endswith(f"/ {e} 20") or name.endswith(f"+{e}) 40"):
                better = s["sharpe"] >= b["sharpe"] + 0.03 and s["dd"] >= b["dd"] - 0.015
                line += f" | {name}: Sharpe {s['sharpe']:.2f} DD {s['dd']*100:.1f}% {'(behtar)' if better else ''}"
        emit(line)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
