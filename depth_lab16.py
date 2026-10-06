"""
DEPTH LAB 16 - order book (bid / ask) ka zor agli qeemat batata hai? (user, 2026-10-06)
====================================================================================
Data: Binance archive "bookDepth" (futures, har ~30 second): qeemat se 1..5% neeche kul khareedar (bid) aur ooper kul
bechne wale (ask) - dollar mein. Qeemat: futures 1h klines (archive).
Coins: 25 bare perps (fixed list). Din: 2024-01-01 -> 2026-09-30, har 3rd din (data bhari hai).
Imbalance (IMB) = (bid - ask) / (bid + ask): +1 = sirf khareedar, -1 = sirf bechne wale.  IMB1 = 1% ke andar, IMB5 = 5% ke andar.

Sawal 1 (ghante): har ghante ke aakhir ka IMB -> agle 1 ghante / 4 ghante ka nafa.
Sawal 2 (din):    din ke aakhri 3 ghante ka ausat IMB -> agle din / agle 5 din ka nafa (hamari daily trading ke liye ahem).
Har jagah: (a) time-series: coin ka apna IMB ooncha to agla nafa ooncha? (b) coins ka aapas mein muqabla (cross-section).
IC = rank-correlation (+ = khareedar zyada -> qeemat ooper), t-stat; aur sab se ooper 20% vs sab se neeche 20% ka farq (basis points,
100 bps = 1%) - kharcha aana-jana ~30 bps (0.3%) hai, farq is se bara ho tabhi trade ke qabil.
DEV < 2025-10-01 <= TAALA.
Natija: depth_lab16_RESULTS.txt
"""
import io
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

OUT = "depth_lab16_RESULTS.txt"
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "BCH", "TRX", "NEAR", "APT", "ARB",
         "OP", "SUI", "INJ", "FIL", "ATOM", "UNI", "AAVE", "ETC", "SEI", "TIA"]
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-09-30")
STEP = 3
HOLDOUT = pd.Timestamp("2025-10-01")
B = "https://data.binance.vision/data/futures/um"
SES = requests.Session()


def fetch(url):
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


def depth_day(sym, day):
    t = fetch(f"{B}/daily/bookDepth/{sym}/{sym}-bookDepth-{day:%Y-%m-%d}.zip")
    if t is None:
        return None
    df = pd.read_csv(io.StringIO(t))
    df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce").round().astype("Int64")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    p = df.pivot_table(index="timestamp", columns="percentage", values="notional", aggfunc="last")
    need = [-5, -1, 1, 5]
    if not all(k in p.columns for k in need):
        return None
    out = pd.DataFrame({"imb1": (p[-1] - p[1]) / (p[-1] + p[1]), "imb5": (p[-5] - p[5]) / (p[-5] + p[5])})
    out["hour"] = out.index.floor("1h")
    h = out.groupby("hour")[["imb1", "imb5"]].last()        # har ghante ka aakhri snapshot
    return h


_KC = {}


def klines_1h(sym):
    if sym in _KC:
        return _KC[sym]
    months = pd.date_range(START, END, freq="MS")

    def one(m):
        t = fetch(f"{B}/monthly/klines/{sym}/1h/{sym}-1h-{m:%Y-%m}.zip")
        if t is None:
            return None
        first = t.split("\n", 1)[0]
        df = pd.read_csv(io.StringIO(t), header=0 if "open" in first else None)
        df = df.iloc[:, :5]
        df.columns = ["open_time", "open", "high", "low", "close"]
        return df
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(one, months) if p is not None]
    if not parts:
        return None
    k = pd.concat(parts)
    k["hour"] = pd.to_datetime(pd.to_numeric(k["open_time"]), unit="ms")
    _KC[sym] = k.set_index("hour")["close"].astype(float).sort_index()
    _KC[sym] = _KC[sym][~_KC[sym].index.duplicated()]
    return _KC[sym]


def ic_report(df, xcol, ycol, label, emit, by_time=True):
    """df: rows (time, sym, x, y). Cross-section IC har waqt par; time-series IC har coin par."""
    d = df.dropna(subset=[xcol, ycol])
    for per, m in (("DEV", d["t"] < HOLDOUT), ("TAALA", d["t"] >= HOLDOUT)):
        s = d[m]
        if len(s) < 200:
            emit(f"{label:>34} {per:>5}: kam data")
            continue
        # cross-section
        g = s.groupby("t")
        cs = g.apply(lambda q: q[xcol].rank().corr(q[ycol].rank()) if len(q) >= 8 else np.nan, include_groups=False).dropna()
        cs_t = cs.mean() / (cs.std() / np.sqrt(len(cs))) if len(cs) > 2 else np.nan
        # time-series (har coin apne andar)
        ts = s.groupby("sym").apply(lambda q: q[xcol].rank().corr(q[ycol].rank()), include_groups=False).dropna()
        ts_t = ts.mean() / (ts.std() / np.sqrt(len(ts))) if len(ts) > 2 else np.nan
        # top / bottom 20% (pooled, har coin ke apne percentile se)
        pr = s.groupby("sym")[xcol].rank(pct=True)
        top, bot = s.loc[pr >= 0.8, ycol].mean(), s.loc[pr <= 0.2, ycol].mean()
        emit(f"{label:>34} {per:>5}: n {len(s):>6} | cross IC {cs.mean():+.3f} (t {cs_t:+.1f}) | coin-apna IC {ts.mean():+.3f} "
             f"(t {ts_t:+.1f}) | ooper 20% {top*1e4:+.0f} bps, neeche 20% {bot*1e4:+.0f} bps, farq {(top-bot)*1e4:+.0f} bps")


def main():
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    try:
        days = list(pd.date_range(START, END, freq=f"{STEP}D"))
        H = []
        for k, c in enumerate(COINS, 1):
            sym = f"{c}USDT"
            px = klines_1h(sym)
            with ThreadPoolExecutor(16) as ex:
                parts = [p for p in ex.map(lambda d: depth_day(sym, d), days) if p is not None]
            if px is None or not parts:
                print(f"[{k}] {c}: data nahi", flush=True)
                continue
            dep = pd.concat(parts)
            # snapshot ghante ke aakhir (~hh:59) -> us ghante ka close; agla nafa us close se
            close = px  # index = ghante ka open time; close = ghante ke aakhir ki qeemat
            f1 = close.shift(-1) / close - 1
            f4 = close.shift(-4) / close - 1
            p4 = close / close.shift(4) - 1
            dep = dep.join(pd.DataFrame({"f1": f1, "f4": f4, "p4": p4}), how="left")
            dep["sym"] = c
            H.append(dep.reset_index().rename(columns={"hour": "t"}))
            print(f"[{k}/{len(COINS)}] {c}: {len(parts)} din, {len(dep)} ghante", flush=True)
        H = pd.concat(H)
        H["t"] = pd.to_datetime(H["t"])
        emit("=" * 130)
        emit(f"DEPTH LAB 16 - order book imbalance | {H['sym'].nunique()} coins | {H['t'].min().date()} -> {H['t'].max().date()} "
             f"(har {STEP}rd din) | TAALA {HOLDOUT.date()} se")
        emit("=" * 130)
        emit("IMB = (khareedar - bechne wale) / kul, qeemat ke 1% (IMB1) ya 5% (IMB5) ke andar. + IC = khareedar ziada -> qeemat ooper")
        emit("bps = 1% ka sauwan hissa (100 bps = 1%). Aana-jana kharcha ~30 bps.")
        emit(f"\nIMB ki aam halat: IMB1 ausat {H['imb1'].mean():+.3f} (beech ke 80%: {H['imb1'].quantile(.1):+.2f} .. "
             f"{H['imb1'].quantile(.9):+.2f}) | IMB5 ausat {H['imb5'].mean():+.3f} ({H['imb5'].quantile(.1):+.2f} .. {H['imb5'].quantile(.9):+.2f})")

        emit("\n# SAWAL 1 - GHANTE: ghante ke aakhir ka IMB -> agla 1 ghanta / 4 ghante")
        for x in ("imb1", "imb5"):
            for y, lab in (("f1", "agla 1 ghanta"), ("f4", "agle 4 ghante")):
                ic_report(H, x, y, f"{x.upper()} -> {lab}", emit)

        emit("\n# SAWAL 2 - DIN: din ke aakhri 3 ghante ka ausat IMB -> agla din / agle 5 din (daily trading)")
        H["day"] = H["t"].dt.floor("1D")
        last3 = H[H["t"].dt.hour >= 21].groupby(["sym", "day"])[["imb1", "imb5"]].mean().reset_index()
        D = []
        for c in last3["sym"].unique():
            px = klines_1h(f"{c}USDT")
            if px is None:
                continue
            dc = px.resample("1D").last()
            f1d = dc.shift(-1) / dc - 1
            f5d = dc.shift(-5) / dc - 1
            sub = last3[last3["sym"] == c].set_index("day")
            sub = sub.join(pd.DataFrame({"f1d": f1d, "f5d": f5d}), how="left")
            sub["sym"] = c
            D.append(sub.reset_index().rename(columns={"day": "t"}))
        D = pd.concat(D)
        for x in ("imb1", "imb5"):
            for y, lab in (("f1d", "agla din"), ("f5d", "agle 5 din")):
                ic_report(D, x, y, f"{x.upper()} (din) -> {lab}", emit)

        emit("\n# SAWAL 3 - kya IMB qeemat ke PEECHE chalta hai? (pichle 4 ghante ka nafa -> abhi ka IMB)")
        ic_report(H.rename(columns={"p4": "past4"}), "past4", "imb1", "pichla 4h nafa -> IMB1", emit)
    except Exception:
        emit("\n!! GHALTI:\n" + traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
