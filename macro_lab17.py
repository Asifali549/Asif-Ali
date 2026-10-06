"""
MACRO LAB 17 - market ko kya ooper / neeche le jata hai? (user, 2026-10-06: "aap jante ho market kaise chalti hai - usi par kaam karo")
==================================================================================================================================
Bari quwwatein (har din band hone tak maloom; khatre se bachne ke liye 1 din peeche rakhi):
  STABLE30  : stablecoins ki kul miqdar 30 din mein kitni barhi (%) - naya paisa aa raha (DefiLlama)
  FNG       : Fear & Greed index (0 = intehai dar, 100 = intehai lalach) (alternative.me)
  NDQ_TREND : Nasdaq / us ki 50-din average - 1 (US stock market ka rujhaan) (stooq / yahoo, mile to)
  DXY20     : dollar index 20 din tabdeeli % (dollar mazboot = khatre wali cheezein kamzor) (mile to)
  Y10_20    : US 10 saal sood 20 din tabdeeli (mile to)
HISSA A - kya ye agla market nafa batati hain? Alt index (top-100 liquid coins ka barabar wazan) aur BTC: agle 5 / 20 din.
  Bina overlap samples (har 5th / 20th din) se IC + t; teen hisse (neeche / beech / ooper tihai) ka ausat nafa. DEV vs TAALA.
HISSA B - hamare systems (Dip v2, Resid, 4-din girawat, Market safai, W52 5 din) ki trades ko har quwwat ki tihai se baant kar:
  kis halat mein system acha / bura? Tabdeeli ke qabil tabhi jab DEV aur TAALA dono mein ek hi rukh aur bara farq.
HISSA C - saada "halat" filter BTC / alt index par: sirf jab (STABLE30 > 0 aur NDQ_TREND > 0) tab market mein - vs hamesha.
Natija: macro_lab17_RESULTS.txt
"""
import io
import traceback

import numpy as np
import pandas as pd
import requests

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema
from winrate_lab import tstats, pf_of
import strategy_lab5 as L5
import search_lab9 as S9
import search_lab13 as L13
import w52_lab10 as W

OUT = "macro_lab17_RESULTS.txt"
HOLDOUT = S9.HOLDOUT
UA = {"User-Agent": "Mozilla/5.0"}


def get(url, **kw):
    try:
        return requests.get(url, timeout=40, headers=UA, **kw)
    except Exception:
        return None


def stable_series():
    r = get("https://stablecoins.llama.fi/stablecoincharts/all")
    js = r.json()
    s = pd.Series({pd.Timestamp(int(x["date"]), unit="s").floor("1D"): float(x["totalCirculatingUSD"].get("peggedUSD", np.nan))
                   for x in js})
    return s.sort_index()


def fng_series():
    js = get("https://api.alternative.me/fng/?limit=0").json()["data"]
    return pd.Series({pd.Timestamp(int(x["timestamp"]), unit="s").floor("1D"): float(x["value"]) for x in js}).sort_index()


def market_series(name, stooq, yahoo):
    """stooq csv, phir yahoo chart api."""
    r = get(f"https://stooq.com/q/d/l/?s={stooq}&i=d")
    if r is not None and r.status_code == 200 and r.text.startswith("Date"):
        df = pd.read_csv(io.StringIO(r.text))
        if len(df) > 500:
            return pd.Series(df["Close"].values, index=pd.to_datetime(df["Date"])).sort_index(), "stooq"
    r = get(f"https://query2.finance.yahoo.com/v8/finance/chart/{yahoo}?range=10y&interval=1d")
    try:
        js = r.json()["chart"]["result"][0]
        ts = pd.to_datetime(js["timestamp"], unit="s").floor("1D")
        cl = js["indicators"]["quote"][0]["close"]
        s = pd.Series(cl, index=ts).dropna()
        if len(s) > 500:
            return s[~s.index.duplicated()].sort_index(), "yahoo"
    except Exception:
        pass
    return None, "nahi mila"


def tercile_table(vals, rets, edges):
    lab = np.digitize(vals, edges)
    return [(np.nanmean(rets[lab == k]) if (lab == k).sum() else np.nan, int((lab == k).sum())) for k in range(3)]


def main(daily=None, feats=None):
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
        src = {}
        if feats is None:
            feats = {}
            try:
                st = stable_series()
                feats["STABLE30"] = (st / st.shift(30) - 1) * 100
                src["STABLE30"] = "DefiLlama"
            except Exception as e:
                src["STABLE30"] = f"nahi ({e})"
            try:
                feats["FNG"] = fng_series()
                src["FNG"] = "alternative.me"
            except Exception as e:
                src["FNG"] = f"nahi ({e})"
            for nm, sq, yh in (("NDQ", "^ndq", "%5EIXIC"), ("DXY", "dx.f", "DX-Y.NYB"), ("Y10", "10usy.b", "%5ETNX")):
                s, how = market_series(nm, sq, yh)
                src[nm] = how
                if s is None:
                    continue
                s = s.asfreq("D").ffill()
                if nm == "NDQ":
                    feats["NDQ_TREND"] = (s / s.rolling(50).mean() - 1) * 100
                elif nm == "DXY":
                    feats["DXY20"] = (s / s.shift(20) - 1) * 100
                else:
                    feats["Y10_20"] = s - s.shift(20)
        # 1 din peeche: aaj ke faisle mein sirf kal tak ka maloom data
        feats = {k: v.sort_index().asfreq("D").ffill().shift(1) for k, v in feats.items()}

        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index()
        opens = pd.DataFrame({s: d.set_index("timestamp")["open"] for s, d in daily.items()}).sort_index()
        btc_ok, al100 = L5.context(daily)
        start = closes.index[0] + pd.Timedelta(days=210)
        mask = pd.DataFrame(False, index=closes.index, columns=closes.columns)
        for t, ss in al100.items():
            if t in mask.index:
                mask.loc[t, list(ss & set(mask.columns))] = True
        r1 = (opens.shift(-1) / opens - 1)          # din D+1 open se D+2 open (signal D close par)
        alt_r = r1.where(mask).mean(axis=1)
        btc_r = r1["BTC/USDT"]
        idx = closes.index[closes.index >= start]
        emit("=" * 130)
        emit(f"MACRO LAB 17 - market ki bari quwwatein | coins {len(daily)} | {idx[0].date()} -> {idx[-1].date()} | TAALA {HOLDOUT.date()} se")
        emit("=" * 130)
        emit("Data: " + " | ".join(f"{k}: {v}" for k, v in src.items()))

        emit("\n# HISSA A - quwwat aaj -> agle 5 / 20 din ka market nafa (bina overlap, IC = rank rishta, + = quwwat ooper to nafa ooper)")
        for H in (5, 20):
            fa = alt_r.rolling(H).apply(lambda x: np.prod(1 + x) - 1, raw=True).shift(-(H - 1))
            fb = btc_r.rolling(H).apply(lambda x: np.prod(1 + x) - 1, raw=True).shift(-(H - 1))
            for k, f in feats.items():
                for lab, fwd in (("ALT", fa), ("BTC", fb)):
                    out = []
                    for per, m in (("DEV", idx < HOLDOUT), ("TAALA", idx >= HOLDOUT)):
                        days = idx[m][::H]
                        x = f.reindex(days).to_numpy()
                        y = fwd.reindex(days).to_numpy()
                        ok = np.isfinite(x) & np.isfinite(y)
                        if ok.sum() < 8:
                            out.append(f"{per}: kam data")
                            continue
                        ic = pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank())
                        t = ic * np.sqrt((ok.sum() - 2) / max(1 - ic ** 2, 1e-9))
                        dev_days = idx[idx < HOLDOUT]
                        edges = np.nanpercentile(f.reindex(dev_days).to_numpy(), [33.3, 66.7])
                        tt = tercile_table(x[ok], y[ok], edges)
                        out.append(f"{per}: n {ok.sum():>3} IC {ic:+.2f} (t {t:+.1f}) tihai neeche/beech/ooper " +
                                   "/".join(f"{v*100:+.1f}%" for v, _ in tt))
                    emit(f"{k:>10} -> {lab} {H:>2}d | " + " || ".join(out))

        emit("\n# HISSA B - hamare systems ki trades, quwwat ki tihai (DEV ki hadon se) ke hisaab se: jeet/10 aur PF")
        P13 = L13.build(daily, start)
        P9 = S9.build(daily, start)
        systems = {"Dip v2": L13.run(P13, "REF_DIP"), "Resid": L13.run(P13, "X_RESID"), "4-din girawat": L13.run(P13, "X_STREAK"),
                   "Market safai": L13.run(P13, "F_FLUSH 40"), "W52 5 din": W.run(P9, 0.05, "H5", "all")}
        for k, f in feats.items():
            edges = np.nanpercentile(f.reindex(idx[idx < HOLDOUT]).to_numpy(), [33.3, 66.7])
            emit(f"{k} (tihai hadein {edges[0]:.2f} / {edges[1]:.2f}):")
            for sname, tr in systems.items():
                row = []
                for per in ("DEV", "TAALA"):
                    sub = [t for t in tr if (pd.Timestamp(t["t_in"]) < HOLDOUT) == (per == "DEV")]
                    vals = np.array([f.get(pd.Timestamp(t["t_in"]).floor("1D") - pd.Timedelta(days=1), np.nan) for t in sub])
                    lab = np.digitize(vals, edges)
                    cells = []
                    for g in range(3):
                        rr = [t["ret"] for t, L_, v in zip(sub, lab, vals) if L_ == g and np.isfinite(v)]
                        if len(rr) >= 5:
                            cells.append(f"{len(rr)}tr {np.mean(np.array(rr) > 0)*10:.1f}/{pf_of(rr):.2f}")
                        else:
                            cells.append(f"{len(rr)}tr -")
                    row.append(f"{per}: " + " | ".join(cells))
                emit(f"   {sname:>14}: " + " || ".join(row))

        emit("\n# HISSA C - halat filter: sirf jab STABLE30 > 0 aur NDQ_TREND > 0 (jo mile) tab market mein")
        cond = pd.Series(True, index=idx)
        used = []
        for k in ("STABLE30", "NDQ_TREND"):
            if k in feats:
                cond &= (feats[k].reindex(idx) > 0).fillna(False)
                used.append(k)
        for lab, r in (("ALT index", alt_r), ("BTC", btc_r)):
            for per, m in (("DEV", idx < HOLDOUT), ("TAALA", idx >= HOLDOUT)):
                rr = r.reindex(idx[m]).fillna(0)
                cc = cond.reindex(idx[m]).shift(1).fillna(False).astype(bool)   # kal ki halat se aaj ka faisla
                for nm, ser in (("hamesha", rr), (f"sirf {'+'.join(used)}", rr.where(cc, 0) - 0.0015 * cc.astype(int).diff().abs().fillna(0))):
                    eq = (1 + ser).cumprod()
                    yrs = len(ser) / 365
                    cagr = eq.iloc[-1] ** (1 / max(yrs, 0.1)) - 1
                    dd = (eq / eq.cummax() - 1).min()
                    emit(f"{lab:>10} {per:>5} {nm:>22}: CAGR {cagr*100:+.1f}% | DD {dd*100:.1f}% | market mein {cc.mean()*100 if nm != 'hamesha' else 100:.0f}% din")
    except Exception:
        emit("\n!! GHALTI:\n" + traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
