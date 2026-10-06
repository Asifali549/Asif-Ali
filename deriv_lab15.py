"""
DERIV LAB 15 - funding rate / volume delta / open interest ka asal imtihan (user, 2026-10-06)
=========================================================================================
Data: Binance ka muft archive (data.binance.vision) - GitHub se khulta hai (deriv_probe_RESULTS.txt):
  * futures 1d klines (2020+) -> OHLC + taker_buy_volume (volume delta = khareedne walon ka aggressive hissa)
  * fundingRate monthly (2020+) -> din ka funding (3 x 8 ghante ka jor)
  * metrics daily (5-min OI) -> din ke aakhri OI ki value (bhari: sirf top-OI_COINS coins, OI_START se)
Coins: KuCoin spot top-150 jin ka Binance USDT-M perp hai. Qeemat perp ki (spot ke qareeb) - trade spot par hoga.
Taala-band usool wahi: aakhri 12 mahine (2025-10-01 se) taala; chunao DEV par.

HISSA 1 - "kya is mein koi current hai?" (decile / IC): har din pool coins ko feature se 10 hisson mein baanto, agle 5 din ka
  nafa (market ka ausat ghata kar) - sab se ooper vs sab se neeche decile ka farq, aur rank-correlation (IC) + t-stat.
  DEV aur TAALA alag. (|t| > 2 = asli ishara, warna shor.)
  Features: FUND3 (3 din ka ausat funding), DELTA3 (3 din taker-buy hissa), DELTA1, OI3 (3 din OI tabdeeli), OI_PRICE (OI 3d minus
  price 3d).
HISSA 2 - trading khayal (3 settings, beech wali asal; Lab 13 ki DEV shartein; taala ek dafa):
  F NEGFUND   : 3-din ausat funding < X (X 0 / -0.01% / -0.03% har 8 ghante) - shorts bhare hue -> squeeze. EMA50>EMA200. Dip exit.
  V ABSORB    : 3 din mein -5% ya ziada girawat magar 3-din taker-buy hissa >= T (T 50 / 52 / 54%) - girawat mein khareedar
                jazb kar rahe. EMA50>EMA200. Dip exit.
  P BUYPUSH   : aaj taker-buy hissa >= T (T 54 / 56 / 58%) aur din musbat - zor ki khareedari. Exit H5.
  O DELEV     : OI 3 din mein Y% ya ziada gira (Y 10 / 15 / 20) aur qeemat bhi giri - leverage saaf. EMA50>EMA200. Dip exit.
HISSA 3 - mojooda Dip v2 behtar ho sakta hai? Dip v2 signals ko funding (manfi / musbat) aur delta (ooper / neeche) se baant kar.
Natija: deriv_lab15_RESULTS.txt
"""
import io
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from bot_core import STABLES
from portfolio_lab import ema, rsi, atr_w, portfolio, stats
from winrate_lab import sim, tstats, pf_of
import search_lab9 as S9
import search_lab13 as L13

OUT = "deriv_lab15_RESULTS.txt"
HOLDOUT = S9.HOLDOUT
BASE = "https://data.binance.vision/data/futures/um"
START = pd.Timestamp("2020-07-01")
END_MONTH = pd.Timestamp("2026-09-01")         # aakhri poora mahina
OI_COINS = 40
OI_START = pd.Timestamp("2024-04-01")
RND = 10
GRID = {"F_NEGFUND": [0.0, -0.0001, -0.0003], "V_ABSORB": [0.50, 0.52, 0.54], "P_BUYPUSH": [0.54, 0.56, 0.58],
        "O_DELEV": [0.10, 0.15, 0.20]}
EXIT = {"F_NEGFUND": "dip", "V_ABSORB": "dip", "P_BUYPUSH": "h5", "O_DELEV": "dip", "REF_DIP": "dip"}

SES = requests.Session()


def fetch_zip(url):
    for _ in range(3):
        try:
            r = SES.get(url, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                return z.read(z.namelist()[0]).decode()
        except Exception:
            pass
    return None


def read_csv(txt, cols):
    if txt is None:
        return None
    first = txt.split("\n", 1)[0]
    has_header = any(ch.isalpha() for ch in first.replace("e-", "").replace("E-", ""))
    df = pd.read_csv(io.StringIO(txt), header=0 if has_header else None)
    if not has_header:
        df.columns = cols[:df.shape[1]]
    return df


KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume",
         "taker_buy_quote_volume", "ignore"]


def months():
    return list(pd.date_range(START, END_MONTH, freq="MS"))


def load_coin(sym):
    ms = months()
    with ThreadPoolExecutor(8) as ex:
        kl = list(ex.map(lambda m: read_csv(fetch_zip(f"{BASE}/monthly/klines/{sym}/1d/{sym}-1d-{m:%Y-%m}.zip"), KCOLS), ms))
        fu = list(ex.map(lambda m: read_csv(fetch_zip(f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{m:%Y-%m}.zip"),
                                            ["calc_time", "funding_interval_hours", "last_funding_rate"]), ms))
    kl = [k for k in kl if k is not None and len(k)]
    if not kl:
        return None
    k = pd.concat(kl)
    k["timestamp"] = pd.to_datetime(pd.to_numeric(k["open_time"]), unit="ms").dt.floor("1D").astype("datetime64[ns]")
    d = pd.DataFrame({"timestamp": k["timestamp"], "open": k["open"].astype(float), "high": k["high"].astype(float),
                      "low": k["low"].astype(float), "close": k["close"].astype(float), "volume": k["volume"].astype(float),
                      "qv": k["quote_volume"].astype(float), "tb": k["taker_buy_volume"].astype(float)})
    d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    fu = [f for f in fu if f is not None and len(f)]
    if fu:
        f = pd.concat(fu)
        f["day"] = pd.to_datetime(pd.to_numeric(f["calc_time"]), unit="ms").dt.floor("1D").astype("datetime64[ns]")
        fd = f.groupby("day")["last_funding_rate"].mean()      # din ka ausat (har 8 ghante wala rate)
        d["fund"] = d["timestamp"].map(fd)
    else:
        d["fund"] = np.nan
    d["oi"] = np.nan
    return d


def load_oi(sym, days):
    def one(day):
        t = read_csv(fetch_zip(f"{BASE}/daily/metrics/{sym}/{sym}-metrics-{day:%Y-%m-%d}.zip"), [])
        if t is None or "sum_open_interest_value" not in t:
            return day, np.nan
        v = pd.to_numeric(t["sum_open_interest_value"], errors="coerce").dropna()
        return day, (float(v.iloc[-1]) if len(v) else np.nan)
    with ThreadPoolExecutor(24) as ex:
        res = list(ex.map(one, days))
    return pd.Series({d: v for d, v in res})


def build(daily, start):
    btc = daily["BTC"].set_index("timestamp")["close"]
    btc_ok = btc > ema(btc, 50)
    dv = pd.DataFrame({s: d.set_index("timestamp")["qv"] for s, d in daily.items()}).sort_index()
    rank = dv.rolling(30, min_periods=20).mean().shift(1).rank(axis=1, ascending=False)
    P = {}
    for sym, d in daily.items():
        c, o, h, l, v = d["close"], d["open"], d["high"], d["low"], d["volume"]
        ts = d["timestamp"]
        e50, e200 = ema(c, 50), ema(c, 200)
        n_ok = np.arange(len(d)) >= 200
        rk = rank[sym].reindex(ts).to_numpy()
        ok = (np.nan_to_num(rk, nan=1e9) <= 100) & (ts >= start).to_numpy()
        up = (c > e200).to_numpy() & n_ok
        bok = btc_ok.reindex(ts).fillna(False).to_numpy(bool)
        pool = ok & up & bok
        golden = (e50 > e200).to_numpy()
        tbr1 = (d["tb"] / v.replace(0, np.nan))
        tbr3 = d["tb"].rolling(3).sum() / v.rolling(3).sum().replace(0, np.nan)
        fund3 = d["fund"].rolling(3, min_periods=2).mean()
        r1 = c / c.shift(1) - 1
        r3 = c / c.shift(3) - 1
        oi3 = d["oi"] / d["oi"].shift(3) - 1
        feat = {"FUND3": fund3.to_numpy(), "DELTA3": tbr3.to_numpy(), "DELTA1": tbr1.to_numpy(), "OI3": oi3.to_numpy(),
                "OI_PRICE": (oi3 - r3).to_numpy()}
        sig = {"REF_DIP": pool & golden & (rsi(c, 3) < 7).to_numpy()}
        for X in GRID["F_NEGFUND"]:
            sig[f"F_NEGFUND {X}"] = pool & golden & L13.fresh(fund3 < X)
        for T in GRID["V_ABSORB"]:
            sig[f"V_ABSORB {T}"] = pool & golden & L13.fresh((r3 <= -0.05) & (tbr3 >= T))
        for T in GRID["P_BUYPUSH"]:
            sig[f"P_BUYPUSH {T}"] = pool & L13.fresh((tbr1 >= T) & (r1 > 0))
        for Y in GRID["O_DELEV"]:
            sig[f"O_DELEV {Y}"] = pool & golden & L13.fresh((oi3 <= -Y) & (r3 < 0))
        o_ = o.to_numpy(float)
        fwd = np.full(len(o_), np.nan)
        fwd[:-6] = o_[6:] / o_[1:-5] - 1          # agle din open se 5 din baad open tak (sirf tajziya)
        P[sym] = dict(sym=sym, ts=ts.to_numpy(), o=o_, h=h.to_numpy(float), l=l.to_numpy(float), c=c.to_numpy(float),
                      atr=atr_w(d).to_numpy(), sma3=(c > c.rolling(3).mean()).to_numpy(), pool=pool, sig=sig, feat=feat,
                      fwd=fwd, has_oi=np.isfinite(d["oi"].to_numpy()), fund=fund3.to_numpy(), tbr3=tbr3.to_numpy())
    return P


def run(P, name, cost=1.0, rng=None, period=None, only=None):
    out = []
    f = name.split(" ")[0]
    ex = EXIT[f]
    for sym, x in P.items():
        idx = np.where(x["sig"][name])[0] if only is None else np.where(x["sig"][name] & only(x))[0]
        if rng is not None:
            pm = x["pool"] & np.isfinite(x["atr"])
            if f == "O_DELEV":
                pm = pm & x["has_oi"]
            pool = np.where(pm)[0]
            if len(idx) == 0 or len(pool) == 0:
                continue
            idx = np.sort(rng.choice(pool, size=min(len(idx), len(pool)), replace=False))
        for i in idx:
            tsi = pd.Timestamp(x["ts"][i])
            if (period == "dev" and tsi >= HOLDOUT) or (period == "hold" and tsi < HOLDOUT):
                continue
            if ex == "h5":
                t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, 1e-12, None, None, 5, {}, cost)
            else:
                st = x["c"][i] - 3.0 * x["atr"][i]
                if not np.isfinite(st) or st <= 0:
                    continue
                t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, None, x["sma3"], 10, {"tp": 0.05}, cost)
            if t:
                t.update(sym=sym, prio=0.0)
                out.append(t)
    return out


def decile_report(P, feat, period, emit):
    rows = []
    for sym, x in P.items():
        ts = pd.DatetimeIndex(x["ts"])
        m = x["pool"] & np.isfinite(x["feat"][feat]) & np.isfinite(x["fwd"])
        m &= (ts < HOLDOUT) if period == "dev" else (ts >= HOLDOUT)
        if m.any():
            rows.append(pd.DataFrame({"day": ts[m], "f": x["feat"][feat][m], "r": x["fwd"][m]}))
    if not rows:
        return emit(f"{feat:>9} {period:>5}: data nahi")
    df = pd.concat(rows)
    df = df[df.groupby("day")["f"].transform("count") >= 10]
    if len(df) < 200:
        return emit(f"{feat:>9} {period:>5}: kam data ({len(df)})")
    df["rx"] = df["r"] - df.groupby("day")["r"].transform("mean")
    df["dec"] = df.groupby("day")["f"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    ics = df.groupby("day").apply(lambda g: g["f"].rank().corr(g["r"].rank()), include_groups=False).dropna()
    t = ics.mean() / (ics.std() / np.sqrt(len(ics))) if len(ics) > 2 else np.nan
    dm = df.groupby("dec")["rx"].mean() * 100
    emit(f"{feat:>9} {period:>5}: {df['day'].nunique():>4} din | IC {ics.mean():+.3f} (t {t:+.1f}) | decile neeche->ooper (5d, market se farq %): "
         + " ".join(f"{v:+.2f}" for v in dm.values) + f" | ooper-neeche {dm.iloc[-1]-dm.iloc[0]:+.2f}%")


def main(daily=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    try:
        if daily is None:
            import data_fetcher
            data_fetcher.AUTO_TOP_N_COINS = 150
            ex = data_fetcher.get_exchange()
            bases = [s.split("/")[0].upper() for s in data_fetcher.get_coin_list(ex)]
            bases = [b for b in bases if b not in STABLES][:150]
            if "BTC" not in bases:
                bases.insert(0, "BTC")
            daily = {}
            for k, b in enumerate(bases, 1):
                d = load_coin(f"{b}USDT")
                if d is not None and len(d) >= 250:
                    daily[b] = d
                print(f"[{k}/{len(bases)}] {b}: {0 if d is None else len(d)}", flush=True)
            # OI: sab se bare OI_COINS (aakhri 90 din ka quote volume)
            big = sorted(daily, key=lambda s: -daily[s]["qv"].tail(90).mean())[:OI_COINS]
            days = list(pd.date_range(OI_START, END_MONTH + pd.offsets.MonthEnd(0), freq="D"))
            for k, b in enumerate(big, 1):
                oi = load_oi(f"{b}USDT", days)
                daily[b]["oi"] = daily[b]["timestamp"].map(oi)
                print(f"OI [{k}/{len(big)}] {b}: {int(oi.notna().sum())} din", flush=True)
        end = END_MONTH + pd.offsets.MonthEnd(0)
        daily = {s: d[d["timestamp"] <= end].reset_index(drop=True) for s, d in daily.items()}
        idx_all = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in daily.values()])))
        start = idx_all[0] + pd.Timedelta(days=210)
        t0, t1 = start, idx_all[-1]
        dev_weeks = (HOLDOUT - t0).days / 7
        n_oi = sum(1 for d in daily.values() if d["oi"].notna().any())
        nf = sum(1 for d in daily.values() if d["fund"].notna().any())
        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        closes = closes[closes.index >= start]
        emit("=" * 140)
        emit("DERIV LAB 15 - funding rate / volume delta / open interest (Binance archive) | TAALA-BAND")
        emit("=" * 140)
        emit(f"Coins {len(daily)} (funding wale {nf}, OI wale {n_oi} - OI {OI_START.date()} se) | {t0.date()} -> {t1.date()} | "
             f"TAALA {HOLDOUT.date()} se")
        emit("jeet/10 = 10 mein se nafa wali | PF = har 1 rupay nuqsan par kitna nafa")
        P = build(daily, start)

        emit("\n# HISSA 1 - KYA KOI CURRENT HAI? (har din coins ko feature se 10 hisson mein; agle 5 din ka nafa market se farq)")
        emit("IC = feature aur agle nafe ka rishta (+ = feature ooper to nafa ooper); |t| > 2 = asli, warna shor")
        for feat in ("FUND3", "DELTA3", "DELTA1", "OI3", "OI_PRICE"):
            for per in ("dev", "hold"):
                decile_report(P, feat, per, emit)

        emit("\n# HISSA 2 - TRADING KHAYAL (DEV)")
        emit(f"{'entry':>18} | {'n':>5} | {'jeet/10':>7} | {'PF':>5} | {'rnd95':>5} | {'folds (trades/PF)':>36} | {'boot5':>5} | "
             f"{'-top10':>6} | {'2xcost':>6} | {'ausat':>6}")
        rows = {}
        names = ["REF_DIP"] + [f"{f} {v}" for f, g in GRID.items() for v in g]
        for nm in names:
            tr = run(P, nm, period="dev")
            if len(tr) < 25:
                emit(f"{nm:>18} | sirf {len(tr)} trades")
                continue
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, period="dev", rng=np.random.default_rng(950 + q))]) for q in range(RND)]
            fstart = OI_START + pd.Timedelta(days=30) if nm.startswith("O_") else t0
            fo = L13.folds_ok(tr, fstart, HOLDOUT)
            bp = S9.boot_p5(tr)
            mt = pf_of([t["ret"] for t in sorted(tr, key=lambda t: -t["ret"])[10:]])
            c2 = pf_of([t["ret"] for t in run(P, nm, period="dev", cost=2.0)])
            wk = s["n"] / ((HOLDOUT - fstart).days / 7)
            rows[nm] = dict(s=s, r95=np.percentile(rnd, 95), fo=fo, bp=bp, mt=mt, c2=c2, wk=wk)
            emit(f"{nm:>18} | {s['n']:>5} | {s['win']/10:>7.1f} | {s['pf']:>5.2f} | {np.percentile(rnd,95):>5.2f} | "
                 f"{' '.join(f'{n}/{p:.2f}' for n, p, _ in fo):>36} | {bp:>5.2f} | {mt:>6.2f} | {c2:>6.2f} | {s['avg']:>+5.2f}%")
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
            chk = {f"jeet>={wmin}": s["win"] >= wmin, "PF>rnd95": s["pf"] > r["r95"], "PF>=1.3": s["pf"] >= 1.3,
                   "folds": all(o for _, _, o in r["fo"]), "boot>=1.15": r["bp"] >= 1.15, "-top10>=1.2": r["mt"] >= 1.2,
                   "2xcost>=1.2": r["c2"] >= 1.2, ">=0.3/hafta": r["wk"] >= 0.3,
                   "padosi": any(n in rows and rows[n]["s"]["pf"] > rows[n]["r95"] for n in nb)}
            fails = [k for k, v in chk.items() if not v]
            if not fails:
                passed.append(mid)
            emit(f"{mid:>20}: {'DEV PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
        emit("\n# TAALA (aakhri 12 mahine) - DEV pass + REF DIP")
        for nm in ["REF_DIP"] + passed:
            tr = run(P, nm, period="hold")
            if len(tr) < 5:
                emit(f"{nm:>20}: taale mein sirf {len(tr)} trades")
                continue
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, period="hold", rng=np.random.default_rng(990 + q))]) for q in range(RND)]
            rnd = [x for x in rnd if np.isfinite(x)] or [np.inf]
            ok = s["pf"] >= 1.2 and s["pf"] > np.percentile(rnd, 95)
            ndays = len(set(str(pd.Timestamp(t["t_in"]).date()) for t in tr))
            emit(f"{nm:>20}: {s['n']} trades ({ndays} din) | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | random p95 "
                 f"{np.percentile(rnd,95):.2f} -> {'TAALA PASS' if ok else 'TAALA FAIL'}" + (" (muqabla)" if nm == "REF_DIP" else ""))
            if nm != "REF_DIP" and ok:
                eq = portfolio(run(P, nm), closes, "fixed", 0.10, max_pos=10, cap=1.0)[0]
                p = stats(eq)
                emit(f"{'':>22}portfolio 10%: CAGR {p['cagr']*100:+.1f}% | DD {p['dd']*100:.1f}% | Sharpe {p['sharpe']:.2f}")
        if not passed:
            emit("koi khayal DEV pass nahi hua")

        emit("\n# HISSA 3 - DIP v2 ko funding / delta se baant kar (behtari ka imkan?)")
        for per in ("dev", "hold", None):
            lab = {"dev": "DEV", "hold": "TAALA", None: "POORA"}[per]
            parts = [("sab", None),
                     ("funding < 0", lambda x: np.nan_to_num(x["fund"], nan=1) < 0),
                     ("funding >= 0", lambda x: np.nan_to_num(x["fund"], nan=-1) >= 0),
                     ("delta3 >= 50%", lambda x: np.nan_to_num(x["tbr3"], nan=0) >= 0.5),
                     ("delta3 < 50%", lambda x: np.nan_to_num(x["tbr3"], nan=1) < 0.5)]
            out = []
            for name, fn in parts:
                tr = run(P, "REF_DIP", period=per, only=fn)
                s = tstats(tr) if tr else {"n": 0, "win": 0, "pf": 0}
                out.append(f"{name}: {s['n']}tr jeet {s['win']/10:.1f} PF {s['pf']:.2f}")
            emit(f"{lab:>6}: " + " | ".join(out))
    except Exception:
        emit("\n!! GHALTI:\n" + traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SAVE] {OUT}")


if __name__ == "__main__":
    main()
