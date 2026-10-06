"""
REALITY LAB 20 - teen shak door karna (user, 2026-10-06: "jo aap keh rahe ho sab kar do")
=========================================================================================
1) DOOB CHUKE COINS (survivorship): Binance spot archive (data.binance.vision) mein band ho chuke (delisted) coins bhi hain.
   Do universe: SURV = sirf aaj tak chalne wale coins | SAB = delisted samet. Dono mein point-in-time top-100 (us din ka volume).
   Har board system dono par: trades, jeet/10, PF, ausat, portfolio CAGR / DD / Sharpe, taala. Aur SAB mein delisted coins ki trades alag.
2) KHAREED KA ASAL WAQT: backtest din ke open (00:00 UTC = 5 AM PKT) par khareedta hai; bots 00:20-00:50 UTC chalte hain.
   1h data se: entry 01:00 UTC ki qeemat par (pehle ghante ka close - mohtaat) aur ~00:30 (pehle ghante ka (open+close)/2).
   Nafa ka hisaab: naya nafa = (1 + purana nafa) x open / nayi qeemat - 1 (exit wahi; TP/SL ka chhota farq nazarandaz).
3) SAB BOTS EK SATH: 8 daily systems, har ek ko barabar hissa (1/8), mahana rebalance; mushtarka CAGR / DD / Sharpe / bura mahina,
   systems ka aapas mein correlation, ek hi din kitne systems khareedte hain, ek hi coin kitne systems mein.
Natija: reality_lab20_RESULTS.txt
"""
import io
import re
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from bot_core import STABLES
from portfolio_lab import portfolio, stats, mix
from winrate_lab import tstats, pf_of
import search_lab9 as S9
import search_lab13 as L13
import stop_fix_lab as SF
import strategy_lab5 as L5
import trail_lab11 as T11
import w52_lab10 as W

OUT = "reality_lab20_RESULTS.txt"
HOLDOUT = S9.HOLDOUT
START = pd.Timestamp("2020-01-01")
END_MONTH = pd.Timestamp("2026-09-01")
ALIVE_AFTER = pd.Timestamp("2026-09-20")     # is ke baad tak data = aaj bhi chal raha
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BASE = "https://data.binance.vision/data/spot/monthly/klines"
EXTRA_STABLE = {"USDC", "BUSD", "TUSD", "PAX", "USDP", "FDUSD", "DAI", "SUSD", "EUR", "GBP", "AUD", "AEUR", "UST", "USTC",
                "USDS", "USDSB", "BKRW", "BIDR", "IDRT", "TRY", "BRL", "RUB", "ZAR", "UAH", "NGN", "VAI", "XUSD", "BFUSD",
                "EURI", "USD1", "PAXG", "WBTC", "WBETH", "BETH"}
SES = requests.Session()


# ------------------------------------------------------------------ data
def s3_list(prefix, delim=True):
    out_p, out_k, marker = [], [], ""
    for _ in range(200):
        url = f"{S3}?prefix={prefix}" + ("&delimiter=/" if delim else "") + (f"&marker={marker}" if marker else "")
        txt = None
        for _ in range(4):
            try:
                r = SES.get(url, timeout=60)
                if r.status_code == 200:
                    txt = r.text
                    break
            except Exception:
                pass
        if txt is None:
            break
        ps = re.findall(r"<Prefix>([^<]+)</Prefix>", txt)
        ks = re.findall(r"<Key>([^<]+)</Key>", txt)
        out_p += [p for p in ps if p != prefix]
        out_k += ks
        if "<IsTruncated>true</IsTruncated>" not in txt:
            break
        nm = re.findall(r"<NextMarker>([^<]+)</NextMarker>", txt)
        marker = nm[0] if nm else (ks[-1] if ks else (ps[-1] if ps else ""))
        if not marker:
            break
    return out_p, out_k


def good_symbol(sym):
    if not sym.endswith("USDT"):
        return False
    b = sym[:-4]
    if not b or not b.isalnum() or b in STABLES or b in EXTRA_STABLE:
        return False
    if any(b.endswith(x) for x in ("UP", "DOWN", "BULL", "BEAR")) and len(b) > 4:
        return False
    return True


def fetch_zip(url):
    for _ in range(3):
        try:
            r = SES.get(url, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                return z.read(z.namelist()[0]).decode()
        except Exception:
            pass
    return None


def parse_k(txt):
    if txt is None:
        return None
    first = txt.split("\n", 1)[0]
    df = pd.read_csv(io.StringIO(txt), header=0 if any(ch.isalpha() for ch in first.replace("e-", "").replace("E-", "")) else None)
    df = df.iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    t = pd.to_numeric(df["open_time"])
    t = np.where(t > 1e14, t // 1000, t)          # 2025 se microseconds
    df["ts"] = pd.to_datetime(t, unit="ms")
    return df


def months_of(sym, tf):
    _, keys = s3_list(f"data/spot/monthly/klines/{sym}/{tf}/", delim=False)
    ms = []
    for k in keys:
        m = re.search(rf"{sym}-{tf}-(\d{{4}}-\d{{2}})\.zip$", k)
        if m:
            ms.append(pd.Timestamp(m.group(1) + "-01"))
    return sorted(ms)


def load_daily(sym):
    ms = [m for m in months_of(sym, "1d") if START <= m <= END_MONTH]
    if not ms:
        return None
    parts = [parse_k(fetch_zip(f"{BASE}/{sym}/1d/{sym}-1d-{m:%Y-%m}.zip")) for m in ms]
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return None
    k = pd.concat(parts)
    d = pd.DataFrame({"timestamp": k["ts"].dt.floor("1D").astype("datetime64[ns]"), "open": k["open"].astype(float),
                      "high": k["high"].astype(float), "low": k["low"].astype(float), "close": k["close"].astype(float),
                      "volume": k["volume"].astype(float)})
    d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    d = d[(d["close"] > 0) & (d["volume"] >= 0)].reset_index(drop=True)
    return d


def load_binance(emit):
    pre, _ = s3_list("data/spot/monthly/klines/")
    syms = sorted({p.rstrip("/").split("/")[-1] for p in pre})
    syms = [s for s in syms if good_symbol(s)]
    emit(f"Binance spot archive: {len(syms)} USDT jore (stable/leveraged ke baghair)")
    daily = {}
    with ThreadPoolExecutor(24) as ex:
        for s, d in zip(syms, ex.map(load_daily, syms)):
            if d is not None and len(d) >= 60:
                daily[s[:-4] + "/USDT"] = d
    return daily


def first_hour(pairs, fake_daily=None):
    """pairs: set of (sym, month). return {(sym, day): (open, close)} pehle ghante (00:00-01:00 UTC) ka."""
    out = {}
    if fake_daily is not None:                       # local nakli jaanch
        rng = np.random.default_rng(5)
        for sym, m in pairs:
            d = fake_daily[sym]
            for _, r in d[(d["timestamp"] >= m) & (d["timestamp"] < m + pd.offsets.MonthBegin(1))].iterrows():
                c1 = r["open"] * np.exp(rng.normal(0, 0.01))
                out[(sym, r["timestamp"])] = (r["open"], c1)
        return out

    def one(p):
        sym, m = p
        b = sym.replace("/", "")
        k = parse_k(fetch_zip(f"{BASE}/{b}/1h/{b}-1h-{m:%Y-%m}.zip"))
        if k is None:
            return {}
        k = k[k["ts"].dt.hour == 0]
        return {(sym, t.floor("1D")): (float(o), float(c)) for t, o, c in zip(k["ts"], k["open"], k["close"])}
    with ThreadPoolExecutor(24) as ex:
        for dct in ex.map(one, sorted(pairs)):
            out.update(dct)
    return out


# ------------------------------------------------------------------ systems
SIZES = {"Donchian": ("risk", 0.01), "Capitulation": ("risk", 0.02), "Dip v2": ("fixed", 0.20), "Dip+": ("fixed", 0.20),
         "W52 5 din": ("fixed", 0.10), "W52 chalta SL": ("fixed", 0.10), "4-din girawat": ("fixed", 0.10), "Market safai": ("fixed", 0.10)}


def make_systems(daily, start):
    btc_ok, al100 = L5.context(daily)
    Psf = SF.prep(daily, al100, btc_ok, start)
    P13 = L13.build(daily, start)
    P9 = S9.build(daily, start)
    U = T11.units_w52(daily, start)
    dip = L13.run(P13, "REF_DIP")
    res = L13.run(P13, "X_RESID")
    return {
        "Donchian": SF.run(Psf, "DON", 4.0, None, 0.20),
        "Capitulation": SF.run(Psf, "CAPIT", 3.0, None, None),
        "Dip v2": dip,
        "Dip+": dip + [dict(t) for t in res],
        "W52 5 din": W.run(P9, 0.05, "H5", "all"),
        "W52 chalta SL": T11.run(U, dict(k=None, act=0.05, gap=0.02, hold=5), "all"),
        "4-din girawat": L13.run(P13, "X_STREAK"),
        "Market safai": L13.run(P13, "F_FLUSH 40"),
    }, al100


def tline(tr):
    if len(tr) < 3:
        return f"{len(tr):>4} tr  -"
    s = tstats(tr)
    return f"{s['n']:>4} tr jeet {s['win']/10:.1f}/10 PF {s['pf']:5.2f} ausat {s['avg']:+5.2f}%"


def pline(tr, name, cl):
    if not tr:
        return None, "data nahi"
    mode, size = SIZES[name]
    eq = portfolio(tr, cl, mode, size, max_pos=10, cap=0.20)[0]
    st = stats(eq)
    return eq, f"CAGR {st['cagr']*100:+6.1f}% DD {st['dd']*100:6.1f}% Sharpe {st['sharpe']:.2f} bura mahina {st['worst_month']*100:+.1f}%"


def main(daily_all=None, fake=False):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    try:
        if daily_all is None:
            daily_all = load_binance(emit)
        last = {s: d["timestamp"].iloc[-1] for s, d in daily_all.items()}
        alive = {s for s, t in last.items() if t >= ALIVE_AFTER}
        dead = set(daily_all) - alive
        daily_surv = {s: d for s, d in daily_all.items() if s in alive}
        if "BTC/USDT" not in daily_surv:
            raise RuntimeError("BTC data nahi")
        closes_all = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily_all.items()}).sort_index().ffill()
        start = closes_all.index[0] + pd.Timedelta(days=210)
        cl = closes_all[closes_all.index >= start]
        cl_h = cl[cl.index >= HOLDOUT]
        emit("=" * 130)
        emit(f"REALITY LAB 20 | coins: SAB {len(daily_all)} (zinda {len(alive)}, doob/band {len(dead)}) | {cl.index[0].date()} -> "
             f"{cl.index[-1].date()} | TAALA {HOLDOUT.date()} se")
        emit("=" * 130)
        emit("jeet/10 = 10 trades mein se nafa wali | PF = 1 rupay nuqsan par kitna nafa | ausat = har trade ka ausat nafa (kharche ke baad)")

        # ---------------- 1) survivorship
        emit("\n# 1) DOOB CHUKE COINS - sirf zinda coins (purane tests jaisa) vs delisted samet (asal duniya)")
        SY_s, al_s = make_systems(daily_surv, start)
        SY_a, al_a = make_systems(daily_all, start)
        top_dead = sum(1 for ss in al_a.values() for s in ss if s in dead)
        tot = sum(len(ss) for ss in al_a.values())
        emit(f"top-100 (point-in-time) mein delisted coins ka hissa: {top_dead / max(tot, 1) * 100:.1f}% (din x coin)")
        curves = {}
        for name in SIZES:
            a, s = SY_a[name], SY_s[name]
            emit(f"\n## {name}")
            emit(f"   ZINDA  poora: {tline(s)} | taala: {tline([t for t in s if pd.Timestamp(t['t_in']) >= HOLDOUT])}")
            emit(f"   SAB    poora: {tline(a)} | taala: {tline([t for t in a if pd.Timestamp(t['t_in']) >= HOLDOUT])}")
            dd_ = [t for t in a if t["sym"] in dead]
            emit(f"   sirf doobe coins ki trades: {tline(dd_)}")
            _, ps = pline(s, name, cl)
            eq, pa = pline(a, name, cl)
            curves[name] = eq
            emit(f"   portfolio ZINDA: {ps}")
            emit(f"   portfolio SAB:   {pa}")
            _, ph = pline([t for t in a if pd.Timestamp(t["t_in"]) >= HOLDOUT], name, cl_h)
            emit(f"   portfolio SAB taala: {ph}")

        # ---------------- 2) entry timing
        emit("\n# 2) KHAREED KA ASAL WAQT (SAB coins) - din ke open (5 AM PKT) vs ~5:30 AM vs 6 AM PKT")
        pairs = {(t["sym"], pd.Timestamp(t["t_in"]).floor("1D").replace(day=1)) for tr in SY_a.values() for t in tr}
        fh = first_hour(pairs, fake_daily=daily_all if fake else None)
        emit(f"1h data mila: {len(fh)} din (zaroorat {len(pairs)} mahine)")
        SY_t = {}
        for name in SIZES:
            base, mid, late, moves = [], [], [], []
            for t in SY_a[name]:
                key = (t["sym"], pd.Timestamp(t["t_in"]).floor("1D"))
                if key not in fh:
                    continue
                o1, c1 = fh[key]
                if not (o1 > 0 and c1 > 0):
                    continue
                m1 = (o1 + c1) / 2
                base.append(t)
                mid.append(dict(t, ret=(1 + t["ret"]) * o1 / m1 - 1))
                late.append(dict(t, ret=(1 + t["ret"]) * o1 / c1 - 1))
                moves.append(c1 / o1 - 1)
            SY_t[name] = late
            emit(f"\n## {name} ({len(base)} trades jin ka 1h data mila; pehle ghante mein ausat chaal {np.mean(moves)*100 if moves else 0:+.2f}%)")
            emit(f"   5 AM (backtest): {tline(base)}")
            emit(f"   5:30 AM:         {tline(mid)}")
            emit(f"   6 AM:            {tline(late)}")
            _, p0 = pline(base, name, cl)
            _, p2 = pline(late, name, cl)
            emit(f"   portfolio 5 AM: {p0}")
            emit(f"   portfolio 6 AM: {p2}")

        # ---------------- 3) all together
        emit("\n# 3) SAB 8 BOTS EK SATH (SAB coins, har system 1/8 hissa, mahana barabar) - 5 AM aur 6 AM entry")
        curves_late = {n: pline(SY_t[n], n, cl)[0] for n in SIZES if SY_t[n]}
        for lab, cv in (("5 AM", curves), ("6 AM", curves_late)):
            cv = {k: v for k, v in cv.items() if v is not None}
            w = {k: 1 / len(cv) for k in cv}
            m = mix(cv, w)
            st = stats(m)
            mh = m[m.index >= HOLDOUT]
            sh = stats(mh / mh.iloc[0])
            emit(f"   {lab}: CAGR {st['cagr']*100:+.1f}% | DD {st['dd']*100:.1f}% | Sharpe {st['sharpe']:.2f} | bura mahina "
                 f"{st['worst_month']*100:+.1f}% | +mahine {st['pos_months']:.0f}% | taala CAGR {sh['cagr']*100:+.1f}% DD {sh['dd']*100:.1f}%")
            emit("      saal: " + " ".join(f"{d.year}:{v*100:+.0f}" for d, v in st["yearly"].items()))
        cv = {k: v for k, v in curves.items() if v is not None}
        mo = pd.DataFrame({k: v.resample("ME").last().pct_change() for k, v in cv.items()}).dropna()
        emit("\n   mahana nafa ka correlation (1 = bilkul ek sath, 0 = alag):")
        names = list(mo.columns)
        emit("   " + " " * 15 + " ".join(f"{n[:7]:>8}" for n in names))
        cm = mo.corr()
        for a in names:
            emit(f"   {a[:14]:>14} " + " ".join(f"{cm.loc[a, b]:>8.2f}" for b in names))
        # crowding
        buys = {}
        for n in SIZES:
            for t in SY_a[n]:
                buys.setdefault(pd.Timestamp(t["t_in"]).floor("1D"), set()).add(n)
        cnt = pd.Series({d: len(v) for d, v in buys.items()})
        emit(f"\n   jis din koi khareed hui: {len(cnt)} din | 3+ systems ek din: {(cnt >= 3).sum()} din | 5+ systems: {(cnt >= 5).sum()} din "
             f"| sab se ziada {cnt.max()} systems ek din")
        same = {}
        for n in SIZES:
            for t in SY_a[n]:
                same.setdefault((t["sym"], pd.Timestamp(t["t_in"]).floor("1D")), set()).add(n)
        sc = pd.Series({k: len(v) for k, v in same.items()})
        emit(f"   ek coin ek din: {len(sc)} khareedain | 2+ systems ne wohi coin usi din khareeda: {(sc >= 2).mean()*100:.0f}% | 3+: {(sc >= 3).mean()*100:.0f}%")
        m = mix(cv, {k: 1 / len(cv) for k in cv})
        dr = m.pct_change().dropna()
        emit("   mushtarka sab se bure 5 din: " + ", ".join(f"{d.date()} {v*100:+.1f}%" for d, v in dr.nsmallest(5).items()))
    except Exception:
        emit("\n!! GHALTI:\n" + traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
