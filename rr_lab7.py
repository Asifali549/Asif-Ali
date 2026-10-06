"""
RR LAB 7 - "SL chhota, TP bara" (user, 2026-10-06)
==================================================
User: "TP chhota aur SL bara se kaam nahi banega. SL chhota aur TP bara ho - 10 mein se 5-6 bhi jeetein to kafi nafa.
Jo strategies fail huin, shayad unki setting / kaam ki tarteeb ka masla tha - tarteeb badal kar dekho."

Har entry par WOHI fixed exit khandan (koi trailing nahi, seedha hisaab):
  SL = signal close - k x ATR(14)      k = 1.0 / 1.5 / 2.0          (chhota SL)
  TP = entry + R x (entry - SL)        R = 2 / 3 / 4                (bara TP: 2x / 3x / 4x SL)
  time stop: daily 20 din, 4H 90 candle (15 din) - us candle ke close par
  stop pehle check, phir TP (ek hi candle mein dono = haar maani jati hai - mohtat)
Entries (sab lookahead-free / sachai test shuda):
  DAILY : DIP (RSI3<7), RESID (z<-2), DONCHIAN 20 (BTC filter), CAPIT (-8% + volume 2x)
  4H    : TP5 entry (Ichimoku + market structure, FIX ke baad), MS_EXTRA (structure BOS + volume 2x, Ichimoku baghair),
          SUPERTREND (10,3) flip + volume 1.5x, BREAKOUT 42 candle + volume 2x
Har cell: 10 mein se jeet / haar, PF ("1 rupay nuqsan par kitna nafa"), random-entry p95 (wohi exit, 6 seeds),
OOS (2025+), 4 folds, 2x kharcha, boot p5, trades/hafta, aur portfolio (1% risk / trade, max 10, ek trade max 30%).
PASS (pehle se tay): PF >= 1.3 | PF > random p95 | OOS >= 1.2 | 4/4 folds > 1 | boot p5 >= 1.1 | 2x kharcha >= 1.15 |
  >= 0.5 / hafta | aur kam az kam 1 padosi cell (k ya R ek qadam) bhi PF > random p95 aur OOS > 1.1.
Natija: rr_lab7_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

import config
from bot_core import fetch_full, norm, STABLES, ichi_signal, ICHI_BASE
from portfolio_lab import ema, atr_w, to_daily, portfolio, stats, H4_BARS
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from winrate_lab import sim, tstats, pf_of
import strategy_lab5 as L5
import stop_fix_lab as SF

OUT = "rr_lab7_RESULTS.txt"
TOP_FETCH = 220
UNIV_4H = 250
KS = [1.0, 1.5, 2.0]
RS = [2.0, 3.0, 4.0]
HOLD = {"D": 20, "H": 90}
RND = 6
COOL = config.SIGNAL_COOLDOWN_BARS


# ------------------------------------------------------------------ 4H signals (sirf zaroori)
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
        up[i] = c[i] >= fl[i] if up[i - 1] else c[i] > fu[i]
    return pd.Series(up, index=d.index)


def sig4h(d):
    P = config.STRATEGY_PARAMS
    cd = lambda s: apply_cooldown(pd.Series(s, index=d.index).fillna(False), COOL).to_numpy()
    tp5 = ichi_signal(d, dict(ICHI_BASE))
    ichi = cd(STRATEGY_FUNCTIONS["ichimoku"](d, dict(P["ichimoku"])))
    ms = cd(STRATEGY_FUNCTIONS["market_structure"](d, dict(P["market_structure"])) & _vol(d, 2.0))
    tr = _trend(d)
    return {
        "4H TP5": tp5,
        "4H MS_EXTRA": ms & ~tp5 & ~ichi,
        "4H STREND": cd(_fresh(supertrend_up(d)) & _vol(d, 1.5) & tr),
        "4H BRK42": cd(STRATEGY_FUNCTIONS["breakout"](d, dict(P["breakout"], lookback=42, volume_mult=2.0))),
    }, tr.to_numpy()


# ------------------------------------------------------------------ generic run
def make_units(h4, daily, start):
    """Har (entry, coin) ke liye: arrays + signal idx + random pool."""
    btc_ok, al100 = L5.context(daily)
    rank_dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    rank = rank_dv.rolling(30, min_periods=20).mean().shift(1).rank(axis=1, ascending=False)
    alU = {day: set(row[row <= UNIV_4H].index) for day, row in rank.iterrows()}
    U = {}

    # DAILY - DIP / RESID (lab5), DON / CAPIT (stop_fix)
    P5 = L5.build(daily, btc_ok, al100, start)
    for sym, x in P5.items():
        base = dict(sym=sym, tf="D", o=x["o"], h=x["h"], l=x["l"], c=x["c"], ts=x["ts"], atr=x["atr"])
        U.setdefault("D DIP", []).append(dict(base, idx=np.where(x["sig"]["REF_DIP"])[0], pool=np.where(x["base"])[0]))
        U.setdefault("D RESID", []).append(dict(base, idx=np.where(x["sig"]["R1_RESID z-2.0"])[0], pool=np.where(x["base"])[0]))
    PS = SF.prep(daily, al100, btc_ok, start)
    for x in PS:
        base = dict(sym=x["sym"], tf="D", o=x["o"], h=x["h"], l=x["l"], c=x["c"], ts=x["ts"], atr=x["atr"])
        pool = np.where(x["ok"])[0]
        pool = pool[pool > 210]
        U.setdefault("D DONCHIAN", []).append(dict(base, idx=np.where(x["sig"]["DON"])[0], pool=pool))
        U.setdefault("D CAPIT", []).append(dict(base, idx=np.where(x["sig"]["CAPIT"])[0], pool=pool))

    # 4H
    for k, (sym, d) in enumerate(h4.items(), 1):
        if len(d) < 400:
            continue
        sg, trd = sig4h(d)
        day = d["timestamp"].dt.floor("1D")
        ok = np.array([sym in alU.get(t, ()) for t in day]) & (d["timestamp"] >= start).to_numpy()
        base = dict(sym=sym, tf="H", o=d["open"].to_numpy(float), h=d["high"].to_numpy(float), l=d["low"].to_numpy(float),
                    c=d["close"].to_numpy(float), ts=d["timestamp"].to_numpy(), atr=atr_w(d).to_numpy())
        pool = np.where(trd & ok)[0]
        pool = pool[pool > 60]
        for nm, s in sg.items():
            U.setdefault(nm, []).append(dict(base, idx=np.where(s & ok)[0], pool=pool))
        if k % 40 == 0:
            print(f"  4h signals {k}/{len(h4)}", flush=True)
    return U


def run(units, k, R, cost=1.0, rng=None):
    out = []
    for x in units:
        idx = x["idx"]
        if rng is not None:
            if len(idx) == 0 or len(x["pool"]) == 0:
                continue
            idx = np.sort(rng.choice(x["pool"], size=min(len(idx), len(x["pool"])), replace=False))
        hold = HOLD[x["tf"]]
        for i in idx:
            a = x["atr"][i]
            if not np.isfinite(a) or a <= 0:
                continue
            st = x["c"][i] - k * a
            if st <= 0:
                continue
            t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, None, None, hold, {"tpR": R}, cost)
            if t:
                t.update(sym=x["sym"], prio=0.0)
                out.append(t)
    return out


def folds(tr, t0, t1):
    edges = pd.date_range(t0, t1, periods=5)
    return [pf_of([t["ret"] for t in tr if a <= pd.Timestamp(t["t_in"]) < b]) for a, b in zip(edges[:-1], edges[1:])]


def boot_p5(tr, n=1500):
    r = np.array([t["ret"] for t in tr])
    g = np.random.default_rng(5)
    return np.percentile([pf_of(g.choice(r, len(r))) for _ in range(n)], 5)


def causality_4h(h4, emit, n_coins=4, bars=250):
    rng = np.random.default_rng(21)
    syms = [s for s, d in h4.items() if len(d) > 1500]
    syms = list(rng.choice(syms, size=min(n_coins, len(syms)), replace=False))
    bad, nsig = {}, {}
    for s in syms:
        d = h4[s]
        full, _ = sig4h(d)
        a0 = int(rng.integers(600, len(d) - bars))
        for i in range(a0, a0 + bars):
            part, _ = sig4h(d.iloc[:i + 1].reset_index(drop=True))
            for nm in full:
                bad[nm] = bad.get(nm, 0) + (bool(full[nm][i]) != bool(part[nm][-1]))
                nsig[nm] = nsig.get(nm, 0) + (bool(full[nm][i]) or bool(part[nm][-1]))
        print(f"  sachai 4h {s}", flush=True)
    emit(f"\n# SACHAI TEST 4H ({len(syms)} coins x {bars} lagataar candles, data kaat kar) - farq 0 hona chahiye")
    for nm in bad:
        emit(f"{nm:>14}: farq {bad[nm]} ({nsig[nm]} signals) -> {'SAHI' if bad[nm] == 0 else 'GHALAT'}")
    return all(v == 0 for v in bad.values())


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    try:
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
                    if df is not None and len(df) >= 6 * 250:
                        h4[sym] = norm(df)
                        print(f"[{k}/{len(coins)}] {sym}: {len(df)}", flush=True)
                except Exception as e:
                    print(f"[{k}] {sym}: SKIP ({e})", flush=True)

        daily = {s: to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        start = closes.index[0] + pd.Timedelta(days=210)
        closes = closes[closes.index >= start]
        t0, t1 = closes.index[0], closes.index[-1]
        weeks = (t1 - t0).days / 7

        emit("=" * 140)
        emit("RR LAB 7 - SL chhota (1 / 1.5 / 2 ATR), TP bara (2 / 3 / 4 guna SL), koi trailing nahi")
        emit("=" * 140)
        emit(f"Coins: {len(h4)} | {t0.date()} -> {t1.date()} ({weeks:.0f} hafte) | daily top-100, 4H top-250 | OOS = 2025+")
        emit("jeet/10 = 10 trades mein se kitni nafa (+0.5% se ziada) | PF = har 1 rupay nuqsan par kitna nafa")
        emit("Yaad rahe: TP = 2x SL par barabar (PF 1) ke liye 10 mein se ~3.4 jeet kaafi; 3x par ~2.6; 4x par ~2.1 (kharche se pehle).")

        causality_4h(h4, emit)
        U = make_units(h4, daily, start)

        rows = {}
        emit(f"\n{'entry':>12} | {'SL':>5} | {'TP':>4} | {'n':>5} | {'/hafta':>6} | {'jeet/10':>7} | {'haar/10':>7} | {'PF':>5} | {'rnd95':>5} | "
             f"{'OOS':>5} | {'folds':>23} | {'boot5':>5} | {'2xcost':>6} | {'CAGR 1%':>8} | {'MaxDD':>6} | {'Sharpe':>6}")
        for nm, units in U.items():
            for k in KS:
                for R in RS:
                    tr = run(units, k, R)
                    if len(tr) < 30:
                        emit(f"{nm:>12} | {k:>4}A | {R:>3}x | sirf {len(tr)} trades")
                        continue
                    s = tstats(tr)
                    rnd = [pf_of([t["ret"] for t in run(units, k, R, rng=np.random.default_rng(300 + q))]) for q in range(RND)]
                    fo = folds(tr, t0, t1)
                    bp = boot_p5(tr)
                    c2 = pf_of([t["ret"] for t in run(units, k, R, cost=2.0)])
                    p = stats(portfolio(tr, closes, "risk", 0.01, max_pos=10, cap=0.30)[0])
                    win10 = s["win"] / 10
                    lose10 = (100 - s["win"] - s["scr"]) / 10
                    rows[(nm, k, R)] = dict(s=s, r95=np.percentile(rnd, 95), fo=fo, bp=bp, c2=c2, p=p, wk=s["n"] / weeks, tr=tr)
                    emit(f"{nm:>12} | {k:>4}A | {R:>3}x | {s['n']:>5} | {s['n']/weeks:>6.2f} | {win10:>7.1f} | {lose10:>7.1f} | "
                         f"{s['pf']:>5.2f} | {np.percentile(rnd,95):>5.2f} | {s['oos']:>5.2f} | {' '.join(f'{f:>5.2f}' for f in fo):>23} | "
                         f"{bp:>5.2f} | {c2:>6.2f} | {p['cagr']*100:>+7.1f}% | {p['dd']*100:>5.1f}% | {p['sharpe']:>6.2f}")
            emit("")

        def good(key):
            r = rows.get(key)
            return r is not None and r["s"]["pf"] > r["r95"] and r["s"]["oos"] > 1.1

        emit("# PASS / FAIL (pehle se tay shartein)")
        passed = []
        for key, r in rows.items():
            nm, k, R = key
            s = r["s"]
            chk = {"PF>=1.3": s["pf"] >= 1.3, "PF>rnd95": s["pf"] > r["r95"], "OOS>=1.2": s["oos"] >= 1.2,
                   "folds": all(np.isfinite(f) and f > 1 for f in r["fo"]), "boot>=1.1": r["bp"] >= 1.1,
                   "2xcost>=1.15": r["c2"] >= 1.15, ">=0.5/hafta": r["wk"] >= 0.5}
            ki, ri = KS.index(k), RS.index(R)
            nb = [(nm, KS[ki + d], R) for d in (-1, 1) if 0 <= ki + d < len(KS)] + \
                 [(nm, k, RS[ri + d]) for d in (-1, 1) if 0 <= ri + d < len(RS)]
            chk["padosi"] = any(good(b) for b in nb)
            fails = [x for x, v in chk.items() if not v]
            if not fails:
                passed.append(key)
            emit(f"{nm:>12} SL {k}A TP {R:.0f}x: {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")

        emit("\n# PASS walon ka portfolio (1% / 2% risk har trade, max 10, ek trade max 30%) + saal-war")
        years = list(range(2021, 2027))
        for key in passed:
            r = rows[key]
            for risk in (0.01, 0.02):
                p = stats(portfolio(r["tr"], closes, "risk", risk, max_pos=10, cap=0.30)[0])
                yr = {d.year: v for d, v in p["yearly"].items()}
                emit(f"{key[0]:>12} SL {key[1]}A TP {key[2]:.0f}x risk {int(risk*100)}%: CAGR {p['cagr']*100:+.1f}% | DD {p['dd']*100:.1f}% | "
                     f"Sharpe {p['sharpe']:.2f} | +mah {p['pos_months']:.0f}% | ausat mah {p['monthly'].mean()*100:+.2f}% | "
                     f"bura mah {p['worst_month']*100:.1f}% | " + " ".join(f"{y}:{yr.get(y, np.nan)*100:+.0f}%" for y in years))
        if not passed:
            emit("koi cell pass nahi hua")
    except Exception:
        emit("\n!! GHALTI (code crash):\n" + traceback.format_exc())

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
