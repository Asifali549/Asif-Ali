"""
SIMPLE DONCHIAN MOMENTUM / BREAKOUT - research (user ka hypothesis, 2026-10-01)
==============================================================================
Sawal: kya sirf price breakout (koi indicator/filter nahi) crypto spot mein khud asli edge deta hai?

Entry  : 4H candle ka close > pichli N candles ka sab se ooncha high  (N = 10/20/30/40/55)
         -> agli candle ke open par (lookahead-free). Ek coin mein ek waqt ek hi trade.
         Koi RSI/MACD/ADX/EMA/volume/daily/liquidity filter NAHI.
Model A: SL = entry - ATR(14) x m (m = 1.5/2.0/2.5/3.0), fixed target = 1R/1.5R/2R/3R  (max 120 candles)
Model B: wahi initial ATR stop, phir ATR trailing (entry ke baad sab se ooncha high - m x ATR), koi TP nahi (max 500)
Random baseline (har config ke liye): utni hi entries, wahi coins/period, wahi stop/exit/sizing - random candle par.
Regimes: BTC daily (bull/bear/sideways) aur BTC 30-din volatility (ooncha/neecha) - entry ke waqt.
OOS: Training Oct 2020 -> Dec 2024 | Out-of-sample Jan 2025 -> Sep 2026.
Timeframe experiment: wahi N Daily candle par (A 2.0x/2R aur B trail 3.0).
Kharcha: fee 0.1% + slip 0.05% har taraf + stop par 0.25% extra; gap par exit = min(stop, open).
Portfolio: 1% risk/trade, max 10, coin max 20% (rozana mark-to-market).
Natija: donchian_research_RESULTS.txt
"""
import numpy as np
import pandas as pd
from numba import njit

from bot_core import fetch_full, norm, STABLES, FEE, SLIP, STOP_SLIP
from portfolio_lab import ema, to_daily, portfolio, stats

TOP_N = 150
H4_BARS = 2400 * 6
NS = [10, 20, 30, 40, 55]
MULTS = [1.5, 2.0, 2.5, 3.0]
TPS = [1.0, 1.5, 2.0, 3.0]
OOS_START = pd.Timestamp("2025-01-01")
OUT = "donchian_research_RESULTS.txt"


@njit(cache=True)
def sim_coin(o, h, l, c, atr, sig_idx, m, tp_r, trail, max_hold, slip, fee, stop_slip):
    """Wapsi: entry idx, exit idx, ret, risk/entry, exit code (0 SL,1 TP,2 TRAIL,3 TIME)."""
    n = len(o)
    k = len(sig_idx)
    e_out = np.empty(k, np.int64)
    x_out = np.empty(k, np.int64)
    r_out = np.empty(k)
    rk_out = np.empty(k)
    why = np.empty(k, np.int64)
    cnt = 0
    busy = -1
    for q in range(k):
        i = sig_idx[q]
        if i <= busy or i + 1 >= n or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        e = i + 1
        entry = o[e] * (1 + slip)
        st = entry - m * atr[i]
        if st <= 0:
            continue
        R = entry - st
        tp = entry + tp_r * R if tp_r > 0 else np.inf
        last = min(e + max_hold, n - 1)
        px = -1.0
        code = 3
        hh = entry
        j = last
        for jj in range(e, last + 1):
            j = jj
            if l[j] <= st:                                   # 1) stop pehle
                px = min(st * (1 - stop_slip), o[j])
                code = 2 if (trail and st > entry - m * atr[i] + 1e-12) else 0
                break
            if h[j] >= tp:                                   # 2) target
                px = max(tp, o[j])
                code = 1
                break
            if trail:                                        # 3) phir trail update
                if h[j] > hh:
                    hh = h[j]
                if np.isfinite(atr[j]):
                    ns = hh - m * atr[j]
                    if ns > st:
                        st = ns
        if px < 0:
            px = c[last]
            code = 3
        e_out[cnt] = e
        x_out[cnt] = j
        r_out[cnt] = px * (1 - slip) * (1 - fee) / (entry * (1 + fee)) - 1
        rk_out[cnt] = R / entry
        why[cnt] = code
        cnt += 1
        busy = j
    return e_out[:cnt], x_out[:cnt], r_out[:cnt], rk_out[:cnt], why[:cnt]


def atr14(d):
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False).mean().to_numpy()


def pf_of(r):
    r = np.asarray(r, float)
    g, lo = r[r > 0].sum(), -r[r < 0].sum()
    return np.nan if len(r) == 0 else (np.inf if lo == 0 else g / lo)


def fmt(x, d=2):
    return "N/A" if x is None or (isinstance(x, float) and not np.isfinite(x) and not np.isinf(x)) else (
        "inf" if isinstance(x, float) and np.isinf(x) else f"{x:.{d}f}")


class Universe:
    def __init__(self, frames, warm):
        self.syms = list(frames)
        self.arr = {}
        for s, d in frames.items():
            self.arr[s] = dict(o=d["open"].to_numpy(float), h=d["high"].to_numpy(float), l=d["low"].to_numpy(float),
                               c=d["close"].to_numpy(float), atr=atr14(d), ts=d["timestamp"].to_numpy(), warm=warm)
        self.frames = frames

    def signals(self, N):
        out = {}
        for s, d in self.frames.items():
            brk = (d["close"] > d["high"].shift(1).rolling(N).max()).to_numpy().copy()
            brk[: self.arr[s]["warm"]] = False
            out[s] = np.flatnonzero(brk).astype(np.int64)
        return out

    def run(self, sig, m, tp_r, trail, max_hold, rng=None, allow=None):
        rows = []
        for s in self.syms:
            a = self.arr[s]
            idx = sig[s]
            if rng is not None:
                pool = np.arange(a["warm"], len(a["o"]) - 1)
                if allow is not None:                       # random bhi sirf filter-allowed candles par
                    pool = pool[allow[s][pool]]
                k = min(len(idx), len(pool))
                if k == 0:
                    continue
                idx = np.sort(rng.choice(pool, size=k, replace=False)).astype(np.int64)
            if len(idx) == 0:
                continue
            e, x, r, rk, w = sim_coin(a["o"], a["h"], a["l"], a["c"], a["atr"], idx, m, tp_r, trail, max_hold,
                                      SLIP, FEE, STOP_SLIP)
            if len(e):
                rows.append(pd.DataFrame({"sym": s, "t_in": a["ts"][e], "t_out": a["ts"][x], "ret": r, "risk": rk,
                                          "why": w, "entry_px": a["o"][e] * (1 + SLIP)}))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate(U, sig, m, tp_r, trail, max_hold, closes, start, btc_reg, btc_vol, with_port=True, allow=None):
    tr = U.run(sig, m, tp_r, trail, max_hold)
    rt = U.run(sig, m, tp_r, trail, max_hold, rng=np.random.default_rng(7), allow=allow)
    tr = tr[tr["t_in"] >= start] if len(tr) else tr
    rt = rt[rt["t_in"] >= start] if len(rt) else rt
    if len(tr) < 10:
        return None
    r, rr = tr["ret"].to_numpy(), rt["ret"].to_numpy()
    oos = tr["t_in"] >= OOS_START
    roos = rt["t_in"] >= OOS_START
    day = tr["t_in"].dt.floor("1D") - pd.Timedelta(days=1)
    reg = btc_reg.reindex(day).to_numpy()
    vol = btc_vol.reindex(day).to_numpy()
    res = {"n": len(r), "win": (r > 0).mean() * 100, "pf": pf_of(r), "rpf": pf_of(rr), "rwin": (rr > 0).mean() * 100,
           "exp": r.mean() * 100, "avg_win": r[r > 0].mean() * 100 if (r > 0).any() else 0,
           "avg_loss": r[r <= 0].mean() * 100 if (r <= 0).any() else 0,
           "pf_tr": pf_of(r[~oos.to_numpy()]), "pf_oos": pf_of(r[oos.to_numpy()]), "n_oos": int(oos.sum()),
           "rpf_oos": pf_of(rr[roos.to_numpy()]),
           "why": dict(zip(*np.unique(tr["why"].map({0: "SL", 1: "TP", 2: "TRAIL", 3: "TIME"}), return_counts=True)))}
    for g in ("bull", "bear", "sideways"):
        x = r[reg == g]
        res[f"reg_{g}"] = (pf_of(x), len(x))
    for g in ("ooncha", "neecha"):
        x = r[vol == g]
        res[f"vol_{g}"] = (pf_of(x), len(x))
    if with_port:
        recs = tr.assign(prio=0.0).to_dict("records")
        eq, taken = portfolio(recs, closes, "risk", 0.01)
        st = stats(eq)
        res.update(cagr=st["cagr"], dd=st["dd"], total=eq.iloc[-1] / eq.iloc[0] - 1, taken=len(taken))
    res["pass"] = passes(res)
    return res


def passes(s):
    if s["n"] < 100 or s["n_oos"] < 30:
        return False
    if not (s["pf"] > 1 and s["pf_oos"] > 1 and s["exp"] > 0):
        return False
    if not (s["pf"] >= 1.10 * s["rpf"] and s["pf_oos"] >= s["rpf_oos"]):
        return False
    if "dd" in s and (s["dd"] < -0.40 or s["cagr"] <= 0):
        return False
    reg_ok = sum(1 for g in ("bull", "bear", "sideways") if s[f"reg_{g}"][1] < 30 or s[f"reg_{g}"][0] >= 0.9)
    return reg_ok >= 2


def line(name, s):
    return (f"{name:>28} | {s['n']:>6} | {s['win']:>5.1f} | {s['rwin']:>5.1f} | {fmt(s['pf']):>5} | {fmt(s['rpf']):>5} | "
            f"{s['pf'] - s['rpf']:>+5.2f} | {fmt(s['pf_tr']):>5} | {fmt(s['pf_oos']):>5} | {s['exp']:>+6.2f} | "
            f"{s['avg_win']:>+6.2f} | {s['avg_loss']:>+6.2f} | "
            + (f"{s['total']*100:>+8.0f}% | {s['dd']*100:>6.1f}% | " if "dd" in s else f"{'-':>9} | {'-':>7} | ")
            + ("PASS" if s["pass"] else "FAIL"))


HEAD = (f"{'Config':>28} | {'Trades':>6} | {'Win%':>5} | {'RndW%':>5} | {'PF':>5} | {'RndPF':>5} | {'PF+':>5} | "
        f"{'PFtr':>5} | {'PFoos':>5} | {'Exp%':>6} | {'AvgW%':>6} | {'AvgL%':>6} | {'NetRet':>9} | {'MaxDD':>7} | Faisla")


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x)
        lines.append(x)

    if h4 is None:
        from data_fetcher import get_exchange, get_coin_list
        ex = get_exchange()
        coins = [s for s in get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:TOP_N]
        if "BTC/USDT" not in coins:
            coins.insert(0, "BTC/USDT")
        h4 = {}
        for k, sym in enumerate(coins, 1):
            try:
                df = fetch_full(ex, sym, "4h", H4_BARS)
                if df is not None and len(df) >= 6 * 250:
                    h4[sym] = norm(df).reset_index(drop=True)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)}")
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})")

    daily = {s: to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = max(pd.Timestamp("2020-10-01"), closes.index[0] + pd.Timedelta(days=30))
    closes = closes[closes.index >= start]
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    b50, b200 = ema(bd, 50), ema(bd, 200)
    btc_reg = pd.Series(np.where((bd > b200) & (b50 > b200), "bull", np.where((bd < b200) & (b50 < b200), "bear", "sideways")),
                        index=bd.index)
    bv = bd.pct_change().rolling(30).std()
    btc_vol = pd.Series(np.where(bv > bv.median(), "ooncha", "neecha"), index=bd.index)

    emit("=" * 140)
    emit("SIMPLE DONCHIAN MOMENTUM / BREAKOUT - RESEARCH (koi indicator/filter nahi)")
    emit("=" * 140)
    emit(f"Coins: {len(h4)} | Period: {start.date()} -> {closes.index[-1].date()} | OOS: {OOS_START.date()} se | spot long-only")
    emit(f"Kharcha: fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% har taraf + stop slip {STOP_SLIP*100:.2f}% | portfolio 1% risk, max 10, coin max 20%")
    emit("PASS (ek config): trades>=100 (OOS>=30), PF>1, OOS PF>1, expectancy>0, PF >= 1.10 x random PF, OOS PF >= random OOS PF,")
    emit("                  portfolio MaxDD > -40% aur net return > 0, 3 trend regimes mein se kam az kam 2 mein PF >= 0.9")
    emit("Columns: PF+ = PF - Random PF | PFtr = training PF | PFoos = out-of-sample PF | NetRet = portfolio kul return")

    U4 = Universe(h4, warm=60)
    results = {}
    for N in NS:
        sig = U4.signals(N)
        emit(f"\n\n{'#' * 140}\n4H DONCHIAN {N}\n{'#' * 140}")
        emit(HEAD)
        for m in MULTS:
            for tp in TPS:
                s = evaluate(U4, sig, m, tp, False, 120, closes, start, btc_reg, btc_vol)
                if s:
                    results[("4H", N, "A", m, tp)] = s
                    emit(line(f"A: SL {m}xATR, TP {tp}R", s))
        for m in MULTS:
            s = evaluate(U4, sig, m, 0.0, True, 500, closes, start, btc_reg, btc_vol)
            if s:
                results[("4H", N, "B", m, 0)] = s
                emit(line(f"B: SL+trail {m}xATR", s))

    # ---- regime breakdown for reference exits ----
    emit("\n\n" + "=" * 140)
    emit("MARKET REGIME BREAKDOWN (reference exits: A = 2.0xATR/2R, B = trail 3.0xATR) - PF (trades)")
    emit("=" * 140)
    for N in NS:
        for key, lab in ((("4H", N, "A", 2.0, 2.0), "A 2.0x/2R"), (("4H", N, "B", 3.0, 0), "B trail 3.0x")):
            s = results.get(key)
            if not s:
                continue
            emit(f"Donchian {N:>2} {lab:>12}: " + " | ".join(
                f"{g}: {fmt(s[f'reg_{g}'][0])} ({s[f'reg_{g}'][1]})" for g in ("bull", "bear", "sideways")) + " || " +
                 " | ".join(f"vol {g}: {fmt(s[f'vol_{g}'][0])} ({s[f'vol_{g}'][1]})" for g in ("ooncha", "neecha")))

    # ---- timeframe experiment: daily ----
    emit("\n\n" + "=" * 140)
    emit("TIMEFRAME EXPERIMENT - wahi N DAILY candle par")
    emit("=" * 140)
    emit(HEAD)
    UD = Universe(daily, warm=60)
    for N in NS:
        sig = UD.signals(N)
        for m, tp, trail, mh, lab in ((2.0, 2.0, False, 60, "A 2.0x/2R"), (3.0, 0.0, True, 365, "B trail 3.0x")):
            s = evaluate(UD, sig, m, tp, trail, mh, closes, start, btc_reg, btc_vol)
            if s:
                results[("1D", N, lab)] = s
                emit(line(f"DAILY {N} {lab}", s))

    # ---- plateau ----
    emit("\n\n" + "=" * 140)
    emit("PARAMETER PLATEAU (4H) - har cell = PF (OOS PF) [PASS/FAIL]")
    emit("=" * 140)
    cfgs = [("A", m, tp) for m in MULTS for tp in TPS] + [("B", m, 0) for m in MULTS]
    emit(f"{'Exit':>16} | " + " | ".join(f"{'N=' + str(N):>18}" for N in NS))
    for mdl, m, tp in cfgs:
        cells = []
        for N in NS:
            s = results.get(("4H", N, mdl, m, tp))
            cells.append(f"{fmt(s['pf'])} ({fmt(s['pf_oos'])}) {'P' if s['pass'] else 'F'}" if s else "-")
        lab = f"A {m}x/{tp}R" if mdl == "A" else f"B trail {m}x"
        emit(f"{lab:>16} | " + " | ".join(f"{c:>18}" for c in cells))
    for N in NS:
        k = [results[(("4H", N) + c)]["pass"] for c in cfgs if ("4H", N) + c in results]
        emit(f"Donchian {N}: {sum(k)}/{len(k)} exit configs PASS")

    # ---- final table ----
    emit("\n\n" + "=" * 140)
    emit("FINAL REPORT (har N ke liye reference exits - 'best' chun kar nahi)")
    emit("=" * 140)
    emit(f"{'Test':>34} | {'Trades':>6} | {'Win%':>5} | {'PF':>5} | {'RndPF':>5} | {'OOS PF':>6} | {'MaxDD':>7} | Result")
    for tf in ("4H", "1D"):
        for N in NS:
            for key, lab in ((((tf, N, "A", 2.0, 2.0) if tf == "4H" else (tf, N, "A 2.0x/2R")), "A 2.0xATR/2R"),
                             (((tf, N, "B", 3.0, 0) if tf == "4H" else (tf, N, "B trail 3.0x")), "B trail 3.0xATR")):
                s = results.get(key)
                if s:
                    emit(f"{f'{tf} Donchian {N} {lab}':>34} | {s['n']:>6} | {s['win']:>5.1f} | {fmt(s['pf']):>5} | "
                         f"{fmt(s['rpf']):>5} | {fmt(s['pf_oos']):>6} | {s['dd']*100:>6.1f}% | {'PASS' if s['pass'] else 'FAIL'}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
