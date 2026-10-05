"""
HIGH-WIN LAB - user (2026-10-05): "apne taur par mazeed test karo, koi aur achi strategy dhoond kar lao -
ooncha win rate, achi trades, mahana/salana acha nafa".
=====================================================================================================
Sabaq ab tak: ooncha win rate EXIT se aata hai (CE 16/4 stop + poori position +5% par = TP5), lekin asal faida
tab hai jab ENTRY bhi random se behtar ho. Is liye yahan wohi tasdeeq-shuda exit rakh kar 4H par NAYE entry
khandan aazmaye jate hain (sab buy-only, spot, top-250 liquid, cooldown 15 candle):

  REF_TP5     : live Ichimoku TP5 (Ichimoku AUR market structure) - muqable ke liye
  ICHI_EXTRA  : sirf Ichimoku signal jahan market structure NAHI (TP5 se bahar ke naye signals)
  MS_EXTRA    : sirf market structure (BOS) + volume 2x jahan Ichimoku NAHI
  BRK30_V3    : 30 candle high breakout + volume 3x + EMA200 + EMA50>EMA200
  BRK42_V2    : 42 candle (7 din) high breakout + volume 2x + EMA200 + EMA50>EMA200
  MOMVOL      : 10 candle ROC > 6% + volume 3x + uptrend
  EMACROSS    : EMA 9/21 cross + volume 3x + uptrend
  FVG         : bullish fair-value-gap retest + bullish candle + volume
  STREND      : Supertrend (10, 3) ooper palta + volume 1.5x + uptrend
  RSI_THRUST  : RSI14 55 ke ooper cross + volume 2x + uptrend

Har candidate par (pehle se tay PASS shartein - sab zaroori):
  1) win >= 65%   2) PF > TREND-random p95 (random entries sirf uptrend candles par, same exit, 12 seeds)
  3) OOS (2025+) PF >= 1.3   4) 4/4 time-folds PF > 1   5) bootstrap p5 PF >= 1.2
  6) top-10 trades hata kar PF >= 1.2   7) 2x kharcha PF >= 1.2   8) >= 0.5 signals/hafta
  9) PORTFOLIO: TP5 60 / DIP 25 / NEW 15 ka Sharpe > TP5 70 / DIP 30 ka Sharpe
 10) Universe: 20% coins random hata kar x6 -> kam az kam 5/6 mein PF >= 1.5 aur OOS >= 1.1
Natija: highwin_lab_RESULTS.txt
"""
import numpy as np
import pandas as pd

import config
from bot_core import fetch_full, norm, STABLES, ichi_signal, ICHI_BASE, chandelier as ce_bot
from portfolio_lab import ema, rsi, to_daily, portfolio, stats, mix, H4_BARS
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from winrate_lab import sim, tstats, pf_of, OOS_START
import stop_fix_lab as SF

OUT = "highwin_lab_RESULTS.txt"
TOP_FETCH = 330
UNIV = 250
COOL = config.SIGNAL_COOLDOWN_BARS
CFG = {"tp": 0.05}
CE_M = 4.0
RND_SEEDS = 12


# ------------------------------------------------------------------ entry khandan
def _vol(d, k):
    return (d["volume"] > k * d["volume"].rolling(20).mean()).fillna(False)


def _trend(d):
    c = d["close"]
    return ((c > ema(c, 200)) & (ema(c, 50) > ema(c, 200))).fillna(False)


def _fresh(x):
    x = x.fillna(False)
    return x & ~x.shift(1, fill_value=False)


def supertrend_up(d, n=10, m=3.0):
    h, l, c = (d[k].to_numpy(float) for k in ("high", "low", "close"))
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    mid = (h + l) / 2
    ub, lb = mid + m * atr, mid - m * atr
    fu, fl = ub.copy(), lb.copy()
    up = np.zeros(len(c), bool)
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if up[i - 1]:
            up[i] = c[i] >= fl[i]
        else:
            up[i] = c[i] > fu[i]
    return pd.Series(up, index=d.index)


def signals(d):
    """Har candidate ka bool array (cooldown ke baad)."""
    P = config.STRATEGY_PARAMS
    cd = lambda s: apply_cooldown(pd.Series(s, index=d.index).fillna(False), COOL)
    tp5 = ichi_signal(d, dict(ICHI_BASE))
    ip = dict(P["ichimoku"])
    ichi = cd(STRATEGY_FUNCTIONS["ichimoku"](d, ip)).to_numpy()
    mp = dict(P["market_structure"])
    ms = cd(STRATEGY_FUNCTIONS["market_structure"](d, mp) & _vol(d, 2.0)).to_numpy()
    tr = _trend(d)
    bp42 = dict(P["breakout"], lookback=42, volume_mult=2.0)
    out = {
        "REF_TP5": tp5,
        "ICHI_EXTRA": ichi & ~tp5,
        "MS_EXTRA": ms & ~tp5 & ~ichi,
        "BRK30_V3": cd(STRATEGY_FUNCTIONS["breakout"](d, P["breakout"])).to_numpy(),
        "BRK42_V2": cd(STRATEGY_FUNCTIONS["breakout"](d, bp42)).to_numpy(),
        "MOMVOL": cd(STRATEGY_FUNCTIONS["momentum_volume"](d, P["momentum_volume"])).to_numpy(),
        "EMACROSS": cd(STRATEGY_FUNCTIONS["ema_crossover"](d, P["ema_crossover"])).to_numpy(),
        "FVG": cd(STRATEGY_FUNCTIONS["fvg"](d, P["fvg"])).to_numpy(),
        "STREND": cd(_fresh(supertrend_up(d)) & _vol(d, 1.5) & tr).to_numpy(),
        "RSI_THRUST": cd(_fresh(rsi(d["close"], 14) > 55) & _vol(d, 2.0) & tr).to_numpy(),
    }
    return out, tr.to_numpy()


NAMES = ["REF_TP5", "ICHI_EXTRA", "MS_EXTRA", "BRK30_V3", "BRK42_V2", "MOMVOL", "EMACROSS", "FVG", "STREND", "RSI_THRUST"]


def prep(h4, allowed, start):
    data = []
    for k, (sym, d) in enumerate(h4.items(), 1):
        if len(d) < 400:
            continue
        sig, trd = signals(d)
        day = d["timestamp"].dt.floor("1D")
        ok = np.array([sym in allowed.get(x, ()) for x in day]) & (d["timestamp"] >= start).to_numpy()
        mom = (d["close"] / d["close"].shift(360) - 1).to_numpy()
        data.append(dict(sym=sym, d=d, o=d["open"].to_numpy(float), h=d["high"].to_numpy(float),
                         l=d["low"].to_numpy(float), c=d["close"].to_numpy(float), ts=d["timestamp"].to_numpy(),
                         sig={n: np.where(sig[n] & ok)[0] for n in NAMES}, ok=ok, trd=trd & ok, mom=mom,
                         stop=ce_bot(d, 16, CE_M)))
        if k % 50 == 0:
            print(f"  prep {k}/{len(h4)}", flush=True)
    return data


def run(data, name, cost=1.0, rng=None, mode=None, keep=None):
    out = []
    for x in data:
        if keep is not None and x["sym"] not in keep:
            continue
        idx = x["sig"][name]
        if rng is not None:
            pool = np.where(x["trd"] if mode == "trend" else x["ok"])[0]
            pool = pool[pool > 60]
            if len(idx) == 0 or len(pool) == 0:
                continue
            idx = np.sort(rng.choice(pool, size=min(len(idx), len(pool)), replace=False))
        for i in idx:
            t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, x["stop"][i], x["stop"], None, 500, CFG, cost)
            if t:
                t.update(sym=x["sym"], prio=x["mom"][i] if np.isfinite(x["mom"][i]) else -9)
                out.append(t)
    return out


def folds(tr, t0, t1):
    edges = pd.date_range(t0, t1, periods=5)
    res = []
    for a, b in zip(edges[:-1], edges[1:]):
        r = [t["ret"] for t in tr if a <= pd.Timestamp(t["t_in"]) < b]
        res.append(pf_of(r) if r else np.nan)
    return res


def boot_p5(tr, n=2000, seed=1):
    r = np.array([t["ret"] for t in tr])
    rng = np.random.default_rng(seed)
    return np.percentile([pf_of(rng.choice(r, len(r))) for _ in range(n)], 5)


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    if h4 is None:
        import data_fetcher
        data_fetcher.AUTO_TOP_N_COINS = TOP_FETCH
        ex = data_fetcher.get_exchange()
        coins = [s for s in data_fetcher.get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:TOP_FETCH]
        for must in ("ETH/USDT", "BTC/USDT"):
            if must not in coins:
                coins.insert(0, must)
        h4 = {}
        for k, sym in enumerate(coins, 1):
            try:
                df = fetch_full(ex, sym, "4h", H4_BARS)
                if df is not None and len(df) >= 6 * 120:
                    h4[sym] = norm(df)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)}", flush=True)
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})", flush=True)

    daily = {s: to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc_ok = bd > ema(bd, 50)
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    rank = dv.rank(axis=1, ascending=False)
    al100 = {day: set(row[row <= 100].index) for day, row in rank.iterrows()}
    alU = {day: set(row[row <= UNIV].index) for day, row in rank.iterrows()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes = closes[closes.index >= start]
    t0, t1 = closes.index[0], closes.index[-1]
    weeks = (t1 - t0).days / 7

    emit("=" * 130)
    emit("HIGH-WIN LAB - 4H naye entry khandan, wohi exit (CE 16/4 stop + TP 5%), top-250, 2% risk, cap 20%, max 10")
    emit("=" * 130)
    emit(f"Coins: {len(h4)} | {t0.date()} -> {t1.date()} ({weeks:.0f} hafte) | win = +0.5% se ziada | OOS = 2025+")

    print("prep...", flush=True)
    data = prep(h4, alU, start)
    P = SF.prep(daily, al100, btc_ok, start)
    dip_eq = portfolio(SF.run(P, "DIP", 3.0, 0.05, None), closes, "fixed", 0.20)[0]

    curves, rows = {"DIP": dip_eq}, {}
    emit(f"\n{'entry':>11} | {'n':>5} | {'/hafta':>6} | {'win%':>5} | {'PF':>5} | {'rndT95':>6} | {'rndA95':>6} | {'OOS':>5} | "
         f"{'folds':>23} | {'bootP5':>6} | {'-top10':>6} | {'2xcost':>6} | {'CAGR':>7} | {'MaxDD':>6} | {'Sharpe':>6} | {'+mah%':>5} | {'bura mah':>8}")
    for nm in NAMES:
        tr = run(data, nm)
        if len(tr) < 30 and nm != "REF_TP5":
            emit(f"{nm:>11} | sirf {len(tr)} trades - chhor diya")
            continue
        s = tstats(tr)
        rT = [pf_of([t["ret"] for t in run(data, nm, rng=np.random.default_rng(100 + q), mode="trend")]) for q in range(RND_SEEDS)]
        rA = [pf_of([t["ret"] for t in run(data, nm, rng=np.random.default_rng(200 + q), mode="all")]) for q in range(RND_SEEDS // 2)]
        fo = folds(tr, t0, t1)
        bp = boot_p5(tr)
        srt = sorted(tr, key=lambda t: -t["ret"])
        mt10 = pf_of([t["ret"] for t in srt[10:]])
        c2 = pf_of([t["ret"] for t in run(data, nm, cost=2.0)])
        eq = portfolio(tr, closes, "risk", 0.02)[0]
        p = stats(eq)
        curves[nm] = eq
        rows[nm] = dict(s=s, rT=np.percentile(rT, 95), fo=fo, bp=bp, mt10=mt10, c2=c2, p=p, wk=s["n"] / weeks, tr=tr)
        emit(f"{nm:>11} | {s['n']:>5} | {s['n']/weeks:>6.2f} | {s['win']:>5.1f} | {s['pf']:>5.2f} | {np.percentile(rT,95):>6.2f} | "
             f"{np.percentile(rA,95):>6.2f} | {s['oos']:>5.2f} | {' '.join(f'{f:>5.2f}' for f in fo):>23} | {bp:>6.2f} | {mt10:>6.2f} | "
             f"{c2:>6.2f} | {p['cagr']*100:>+6.1f}% | {p['dd']*100:>5.1f}% | {p['sharpe']:>6.2f} | {p['pos_months']:>5.0f} | {p['worst_month']*100:>7.1f}%")

    years = list(range(2021, 2027))
    emit("\n# SAAL-WAR NAFA (akela, 2% risk)")
    emit(f"{'entry':>11} | " + " | ".join(f"{y:>6}" for y in years))
    for nm, r in rows.items():
        yr = {d.year: v for d, v in r["p"]["yearly"].items()}
        emit(f"{nm:>11} | " + " | ".join(f"{yr.get(y, np.nan)*100:>+5.0f}%" for y in years))

    emit("\n# PORTFOLIO: BASE = TP5 70 / DIP 30 (mahana rebalance) vs TP5 60 / DIP 25 / NEW 15 aur TP5 50 / DIP 20 / NEW 30")
    base = stats(mix(curves, {"REF_TP5": .7, "DIP": .3}))
    emit(f"{'BASE TP5 70/DIP 30':>28} | CAGR {base['cagr']*100:+.1f}% | MaxDD {base['dd']*100:.1f}% | Sharpe {base['sharpe']:.2f} | "
         f"+mahine {base['pos_months']:.0f}% | bura mahina {base['worst_month']*100:.1f}%")
    tp5m = curves["REF_TP5"].resample("ME").last().pct_change().dropna()
    port_ok = {}
    for nm in rows:
        if nm == "REF_TP5":
            continue
        corr = curves[nm].resample("ME").last().pct_change().dropna().corr(tp5m)
        a = stats(mix(curves, {"REF_TP5": .6, "DIP": .25, nm: .15}))
        b = stats(mix(curves, {"REF_TP5": .5, "DIP": .2, nm: .3}))
        port_ok[nm] = a["sharpe"] > base["sharpe"]
        emit(f"{nm:>11} corr TP5 {corr:+.2f} | 60/25/15: CAGR {a['cagr']*100:+.1f}% DD {a['dd']*100:.1f}% Sharpe {a['sharpe']:.2f} | "
             f"50/20/30: CAGR {b['cagr']*100:+.1f}% DD {b['dd']*100:.1f}% Sharpe {b['sharpe']:.2f}")

    emit("\n# PASS/FAIL (pehle se tay 10 shartein; universe test sirf un par jo 1-9 pass karein)")
    syms = [x["sym"] for x in data]
    for nm, r in rows.items():
        s = r["s"]
        chk = {
            "win>=65": s["win"] >= 65, "PF>rndT95": s["pf"] > r["rT"], "OOS>=1.3": s["oos"] >= 1.3,
            "folds 4/4": all(np.isfinite(f) and f > 1 for f in r["fo"]), "boot>=1.2": r["bp"] >= 1.2,
            "-top10>=1.2": r["mt10"] >= 1.2, "2xcost>=1.2": r["c2"] >= 1.2, ">=0.5/hafta": r["wk"] >= 0.5,
            "portfolio+": port_ok.get(nm, True),
        }
        fails = [k for k, v in chk.items() if not v]
        uni = ""
        if not fails:
            good = 0
            det = []
            for q in range(6):
                rng = np.random.default_rng(500 + q)
                keep = set(rng.choice(syms, int(len(syms) * 0.8), replace=False)) | {"BTC/USDT"}
                su = tstats(run(data, nm, keep=keep))
                det.append(f"{su['pf']:.2f}/{su['oos']:.2f}")
                good += su["pf"] >= 1.5 and su["oos"] >= 1.1
            uni = f" | universe {good}/6 ({' '.join(det)})"
            if good < 5:
                fails.append("universe")
        verdict = "PASS" if not fails else "FAIL: " + ", ".join(fails)
        emit(f"{nm:>11}: {verdict}{uni}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
