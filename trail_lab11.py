"""
TRAIL LAB 11 - user ka "chalta SL" (2026-10-06)
==============================================
User: "coin 5% ooper chala jaye aur SL wahin entry ke neeche rahe - ye trade ke usool ke khilaf hai. SL bhi sath sath
ooper chale: coin 10% ooper ho aur faasla 2% ho to SL 8% par pohnch jaye, 2% neeche aaye to 8% nafa ban jaye."
Qaida (har din, daily candle):
  1) pehle aaj ka low pichle SL se check (gap ho to open par becho) - mohtat
  2) phir aaj ke HIGH se chouti (peak) update; jab chouti entry se A% ooper pohnche -> SL = entry x (1 + nafa% - G%)
     (misaal: chouti +10%, G 2% -> SL +8%); SL sirf ooper jata hai, AGLE din se lagu
W52 par (entry: sal ki chouti 5% andar, BTC>EMA50, top-100):  shuru SL nahi / 3 ATR  x  A 3/5/10%  x  G 2/3/5/8%  x  max 10/30 din
DIP+ par (DIP + RESID ek khata): asal (TP 5%) vs TP ki jagah chalta SL (A 3/5 x G 2/3) vs TP 5% + chalta SL (A 3, G 2)
Har cell: 10 mein se jeet, PF (1 rupay nuqsan par nafa), ausat, random p95 (wohi exit, random din), portfolio CAGR / DD.
W52 ka taala pehle istemal ho chuka -> W52 sirf DEV (2025-10-01 se pehle); DIP+ ka DEV aur TAALA dono.
Natija: trail_lab11_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES, FEE, SLIP, STOP_SLIP
from portfolio_lab import portfolio, stats
from winrate_lab import tstats, pf_of
import strategy_lab5 as L5
import search_lab9 as S9

OUT = "trail_lab11_RESULTS.txt"
RND = 6


def sim_trail(o, h, l, c, ts, i, init_stop, act, gap, hold, tp=None, exit_sig=None, cost=1.0):
    n = len(o)
    e = i + 1
    if e >= n:
        return None
    slip, fee, sslip = SLIP * cost, FEE * cost, STOP_SLIP * cost
    entry = o[e] * (1 + slip)
    st = init_stop if (init_stop is not None and np.isfinite(init_stop)) else -np.inf
    if entry <= st:
        return None
    peak = entry
    last = min(e + hold, n - 1)
    px, j = None, last
    for j in range(e, last + 1):
        if l[j] <= st:
            px = min(st * (1 - sslip), o[j]) * (1 - slip)
            break
        if tp is not None and h[j] >= entry * (1 + tp):
            px = max(entry * (1 + tp), o[j]) * (1 - slip)
            break
        if exit_sig is not None and exit_sig[j] and j + 1 < n:
            j += 1
            px = o[j] * (1 - slip)
            break
        if act is not None:
            peak = max(peak, h[j])
            gain = peak / entry - 1
            if gain >= act:
                st = max(st, entry * (1 + gain - gap))
    if px is None:
        px = c[last] * (1 - slip)
    return {"t_in": ts[e], "t_out": ts[j], "entry_px": entry, "risk": 0.1, "ret": px * (1 - fee) / (entry * (1 + fee)) - 1}


def run(units, cfg, period, cost=1.0, rng=None):
    out = []
    for x in units:
        idx = x["idx"]
        if rng is not None:
            if len(idx) == 0 or len(x["pool"]) == 0:
                continue
            idx = np.sort(rng.choice(x["pool"], size=min(len(idx), len(x["pool"])), replace=False))
        for i in idx:
            t_i = pd.Timestamp(x["ts"][i])
            if (period == "dev" and t_i >= S9.HOLDOUT) or (period == "hold" and t_i < S9.HOLDOUT):
                continue
            init = x["c"][i] - cfg["k"] * x["atr"][i] if cfg.get("k") else None
            if init is not None and (not np.isfinite(init) or init <= 0):
                continue
            t = sim_trail(x["o"], x["h"], x["l"], x["c"], x["ts"], i, init, cfg.get("act"), cfg.get("gap"), cfg["hold"],
                          cfg.get("tp"), x["sma3"] if cfg.get("sma") else None, cost)
            if t:
                t.update(sym=x["sym"], prio=0.0)
                out.append(t)
    return out


def units_w52(daily, start):
    P = S9.build(daily, start)
    return [dict(sym=s, o=x["o"], h=x["h"], l=x["l"], c=x["c"], ts=x["ts"], atr=x["atr"], sma3=x["sma3"],
                 idx=np.where(x["sig"]["A_W52_HIGH 0.05"])[0], pool=np.where(x["pool"] & np.isfinite(x["atr"]))[0])
            for s, x in P.items()]


def units_dipplus(daily, start):
    L5.GRID["R1_RESID"] = [("z-2.0", -2.0)]
    b, a = L5.context(daily)
    P = L5.build(daily, b, a, start)
    U = []
    for s, x in P.items():
        sig = x["sig"]["REF_DIP"] | x["sig"]["R1_RESID z-2.0"]
        U.append(dict(sym=s, o=x["o"], h=x["h"], l=x["l"], c=x["c"], ts=x["ts"], atr=x["atr"], sma3=x["sma"][3]
                      if "sma" in x else x["sma3"], idx=np.where(sig)[0], pool=np.where(x["base"] & np.isfinite(x["atr"]))[0]))
    return U


def main(daily=None):
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

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
        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        start = closes.index[0] + pd.Timedelta(days=210)
        closes = closes[closes.index >= start]
        dev_closes = closes[closes.index < S9.HOLDOUT]
        emit("=" * 130)
        emit("TRAIL LAB 11 - chalta SL (chouti se G% neeche, jab nafa A% ho jaye) | W52 sirf DEV, DIP+ DEV + TAALA")
        emit("=" * 130)
        emit(f"Coins {len(daily)} | {closes.index[0].date()} -> {closes.index[-1].date()} | DEV < {S9.HOLDOUT.date()}")
        emit("jeet/10 = 10 trades mein nafa wali | PF = 1 rupay nuqsan par nafa | SL = shuru ka SL | A = kab chale | G = chouti se faasla")

        # ---------------- W52
        U = units_w52(daily, start)
        cells = [("ASAL: 10 din, koi SL nahi", dict(k=None, act=None, gap=None, hold=10))]
        for k in (None, 3.0):
            for hold in (10, 30):
                for act in (0.03, 0.05, 0.10):
                    for gap in (0.02, 0.03, 0.05, 0.08):
                        if gap > act + 0.03:
                            continue
                        cells.append((f"SL {'3ATR' if k else 'nahi'} | A {int(act*100)}% G {int(gap*100)}% | {hold} din",
                                      dict(k=k, act=act, gap=gap, hold=hold)))
        emit(f"\n# W52 (DEV) - {len(cells)} tareeqe")
        emit(f"{'tareeqa':>36} | {'n':>4} | {'jeet/10':>7} | {'PF':>5} | {'rnd95':>5} | {'ausat':>7} | {'CAGR':>7} | {'MaxDD':>6} | {'Sharpe':>6}")
        res = []
        for name, cfg in cells:
            tr = run(U, cfg, "dev")
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(U, cfg, "dev", rng=np.random.default_rng(30 + q))]) for q in range(RND)]
            p = stats(portfolio(tr, dev_closes, "fixed", 0.10, max_pos=10, cap=1.0)[0])
            res.append((name, s, np.percentile(rnd, 95), p))
            emit(f"{name:>36} | {s['n']:>4} | {s['win']/10:>7.1f} | {s['pf']:>5.2f} | {np.percentile(rnd,95):>5.2f} | {s['avg']:>+6.2f}% | "
                 f"{p['cagr']*100:>+6.1f}% | {p['dd']*100:>5.1f}% | {p['sharpe']:>6.2f}")
        emit("\n# W52 behtareen 8 (Sharpe se) - asal ke muqable")
        for name, s, r95, p in sorted(res, key=lambda r: -r[3]["sharpe"])[:8]:
            emit(f"{name:>36}: jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} (rnd95 {r95:.2f}) | ausat {s['avg']:+.2f}% | "
                 f"CAGR {p['cagr']*100:+.1f}% | DD {p['dd']*100:.1f}% | Sharpe {p['sharpe']:.2f}")

        # ---------------- DIP+
        D = units_dipplus(daily, start)
        dcells = [("ASAL: TP 5% + SMA3 + SL 3ATR", dict(k=3.0, tp=0.05, sma=True, hold=10))]
        for act in (0.03, 0.05):
            for gap in (0.02, 0.03):
                dcells.append((f"TP nahi, chalta A{int(act*100)} G{int(gap*100)} +SMA3", dict(k=3.0, act=act, gap=gap, sma=True, hold=10)))
                dcells.append((f"TP nahi, chalta A{int(act*100)} G{int(gap*100)}, 20 din", dict(k=3.0, act=act, gap=gap, hold=20)))
        dcells.append(("TP 5% + chalta A3 G2 + SMA3", dict(k=3.0, tp=0.05, act=0.03, gap=0.02, sma=True, hold=10)))
        emit(f"\n# DIP+ (DIP + RESID ek khata, 20% / trade) - DEV aur TAALA")
        emit(f"{'tareeqa':>36} | {'DEV jeet/10':>11} | {'PF':>5} | {'rnd95':>5} | {'ausat':>7} | {'CAGR':>7} | {'MaxDD':>6} | "
             f"{'Sharpe':>6} || {'TAALA jeet/10':>13} | {'PF':>5} | {'ausat':>7}")
        for name, cfg in dcells:
            tr = run(D, cfg, "dev")
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(D, cfg, "dev", rng=np.random.default_rng(50 + q))]) for q in range(RND)]
            p = stats(portfolio(tr, dev_closes, "fixed", 0.20, max_pos=10, cap=1.0)[0])
            th = tstats(run(D, cfg, "hold"))
            emit(f"{name:>36} | {s['win']/10:>11.1f} | {s['pf']:>5.2f} | {np.percentile(rnd,95):>5.2f} | {s['avg']:>+6.2f}% | "
                 f"{p['cagr']*100:>+6.1f}% | {p['dd']*100:>5.1f}% | {p['sharpe']:>6.2f} || {th['win']/10:>13.1f} | {th['pf']:>5.2f} | "
                 f"{th['avg']:>+6.2f}%")
    except Exception:
        emit("\n!! GHALTI:\n" + traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
