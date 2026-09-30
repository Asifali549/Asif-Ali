"""
SWING LAB - 3 nayi strategies (4H / Daily) ek hi sakht tarazu par
=================================================================
Sabaq (SMC 15m fail): chhote timeframe par kharcha stop ko kha jata hai -> sirf 4H / Daily.

  1) SMC_4H     : Daily trend (close>EMA200, EMA50>EMA200, EMA50 ooper, Higher Low)
                  -> 4H: RSI pullback (<40, pichli 6 candles) + fresh BOS (swing high toota)
                  + volume >1.5x + close candle ke ooper 40% + RSI<72
                  Stop: 4H swing low - 0.2 ATR (1-3 ATR). Exit: TP 2R (ya variant: Chandelier trailing)
  2) SQUEEZE_4H : Bollinger(20,2) width pichle 120 candles ke sab se tang 15% mein (pichli 6 candles mein)
                  -> close upper band aur pichle 20 candles ke high se ooper (fresh), volume >1.5x,
                  close > EMA200. Stop/Exit: Chandelier 16 / 5.5x ATR trailing (Ichimoku wala)
  3) DIP_DAILY  : Daily uptrend (close>EMA200, EMA50>EMA200) + BTC>EMA50 + RSI(3) < 15 (tez girawat)
                  Exit: close > SMA5 (agle din open par) | Stop 3 ATR | max 10 din

Usool (Ichimoku/Donchian wale):
  - lookahead-free: signal band candle par, entry agli candle ke open par; Daily sirf band din se
  - fee 0.1% + slip 0.05% har taraf + stop slip 0.25%; gap par exit = min(stop, open)
  - pehle stop check, phir trailing update; ek candle mein stop+target dono -> STOP
  - point-in-time liquidity: har din pichle 30 din volume se top-100
  - random-entry control (wahi trend/setup, random candle), bootstrap p5, 4 time-folds, top-10 hata kar,
    portfolio (max 10, 1% risk, max 20%) vs random-entry portfolio, parameter-neighbourhood
Natija: swing_lab_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES, FEE, SLIP, STOP_SLIP

TOP_N = 150
H4_BARS = 2400 * 6          # ~6.5 saal 4H
UNIVERSE = 100
N_FOLDS = 4
RISK_PCT, MAX_POS, MAX_POS_PCT = 0.01, 10, 0.20
OUT = "swing_lab_RESULTS.txt"


# ============================ indicators ============================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def chandelier(df, n, m):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return df["high"].rolling(n).max() - m * tr.rolling(n).mean()     # production CE jaisa (SMA ATR)


def pivots(df, k):
    """k candles baad confirm hone wale swing high/low (sirf maazi)."""
    w = 2 * k + 1
    is_ph = df["high"] == df["high"].rolling(w, center=True).max()
    is_pl = df["low"] == df["low"].rolling(w, center=True).min()
    ph = df["high"].shift(k).where(is_ph.shift(k, fill_value=False))
    pl = df["low"].shift(k).where(is_pl.shift(k, fill_value=False))
    prev_pl = pl.dropna().shift(1).reindex(df.index).ffill()
    return ph.ffill(), pl.ffill(), prev_pl


def to_daily(d4):
    x = d4.set_index("timestamp")
    return x.resample("1D", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def attach_daily(d4, feat):
    """Daily feature 4H candle par sirf tab jab wo din BAND ho chuka ho."""
    f = feat.copy()
    f["avail"] = f.index + pd.Timedelta(days=1)
    f = f.reset_index(drop=True).sort_values("avail")
    left = pd.DataFrame({"close_time": d4["timestamp"] + pd.Timedelta(hours=4)})
    m = pd.merge_asof(left, f, left_on="close_time", right_on="avail", direction="backward")
    return m.drop(columns=["close_time", "avail"])


def daily_trend(dd, p):
    e50, e200 = ema(dd["close"], 50), ema(dd["close"], 200)
    _, pl, prev_pl = pivots(dd, 3)
    t = (dd["close"] > e200) & (e50 > e200) & (e50 > e50.shift(5))
    if p.get("need_hl", True):
        t &= pl > prev_pl
    t.iloc[:200] = False
    return t


# ============================ strategies ============================
# har function: d (base-timeframe df) + columns: signal, setup, stop, trail (NaN = koi trail nahi),
# exit_sig (close par exit -> agli candle open), aur p mein tp_r / max_hold
def smc_4h(d4, p, ctx):
    d = d4.copy()
    dd = to_daily(d)
    d["trend"] = attach_daily(d, pd.DataFrame({"t": daily_trend(dd, p)}))["t"].fillna(False).astype(bool).values
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
    a = atr(d)
    r = rsi(c)
    sh, sl, _ = pivots(d, p["pivot"])
    pull = r.rolling(p["pull_bars"]).min() < p["pull_rsi"]
    bos = (c > sh) & (c.shift(1) <= sh)
    vol = v > p["vol_mult"] * v.rolling(20).mean().shift(1)
    cpos = (c - l) / (h - l).replace(0, np.nan) >= 0.6
    raw = sl - 0.2 * a
    stop = raw.where(c - raw >= a, c - a)
    d["setup"] = d["trend"]
    d["signal"] = (d["trend"] & pull & bos & vol & cpos & (r < 72) & (c - raw <= 3 * a)
                   & stop.notna()).fillna(False).astype(bool)
    d["stop"] = stop
    d["trail"] = chandelier(d, 16, 5.5) if p.get("trail") else np.nan
    d["exit_sig"] = False
    return d


def squeeze_4h(d4, p, ctx):
    d = d4.copy()
    c, h, v = d["close"], d["high"], d["volume"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    upper = mid + 2 * sd
    bbw = 4 * sd / mid
    rank = bbw.rolling(120).rank(pct=True)
    squeezed = rank.rolling(6).min() <= p["sq_pct"]
    brk = (c > upper) & (c > h.shift(1).rolling(20).max())
    fresh = brk & ~brk.shift(1, fill_value=False)
    vol = v > p["vol_mult"] * v.rolling(20).mean().shift(1)
    trend = c > ema(c, 200)
    ce = chandelier(d, p["ce_n"], p["ce_m"])
    d["setup"] = trend & (ce < c)
    d["signal"] = (squeezed & fresh & vol & trend & (ce < c)).fillna(False).astype(bool)
    d["stop"] = ce
    d["trail"] = ce
    d["exit_sig"] = False
    return d


def dip_daily(dd_in, p, ctx):
    d = dd_in.copy()
    c = d["close"]
    e50, e200 = ema(c, 50), ema(c, 200)
    btc = ctx["btc_ok"].reindex(d["timestamp"]).fillna(False).to_numpy(bool)
    trend = ((c > e200) & (e50 > e200)).to_numpy() & btc
    trend[:200] = False
    r = rsi(c, 3)
    a = atr(d)
    d["setup"] = trend
    d["signal"] = (trend & (r < p["rsi_lo"]).to_numpy())
    d["stop"] = c - p["stop_atr"] * a
    d["trail"] = np.nan
    d["exit_sig"] = (c > c.rolling(p["exit_sma"]).mean()).to_numpy()
    return d


STRATS = {
    "SMC_4H": (smc_4h, "4h", {"pivot": 3, "pull_bars": 6, "pull_rsi": 40, "vol_mult": 1.5, "tp_r": 2.0,
                              "trail": False, "max_hold": 60, "need_hl": True},
               [("TRAIL CE16/5.5 (TP nahi)", {"trail": True, "tp_r": None, "max_hold": 500}),
                ("TP 3R", {"tp_r": 3.0}), ("vol 1.2x", {"vol_mult": 1.2}), ("vol 2x", {"vol_mult": 2.0}),
                ("pivot 5", {"pivot": 5}), ("pullback RSI 45", {"pull_rsi": 45}), ("HL shart nahi", {"need_hl": False})]),
    "SQUEEZE_4H": (squeeze_4h, "4h", {"sq_pct": 0.15, "vol_mult": 1.5, "ce_n": 16, "ce_m": 5.5, "tp_r": None,
                                      "max_hold": 500},
                   [("squeeze 10%", {"sq_pct": 0.10}), ("squeeze 25%", {"sq_pct": 0.25}),
                    ("vol 1.2x", {"vol_mult": 1.2}), ("vol 2x", {"vol_mult": 2.0}),
                    ("CE 22/4", {"ce_n": 22, "ce_m": 4.0}), ("+TP 3R", {"tp_r": 3.0})]),
    "DIP_DAILY": (dip_daily, "1d", {"rsi_lo": 15, "stop_atr": 3.0, "exit_sma": 5, "tp_r": None, "max_hold": 10},
                  [("RSI3 < 10", {"rsi_lo": 10}), ("RSI3 < 25", {"rsi_lo": 25}), ("stop 2 ATR", {"stop_atr": 2.0}),
                   ("exit SMA3", {"exit_sma": 3}), ("exit SMA10", {"exit_sma": 10})]),
}


# ============================ simulation ============================
def simulate(d, idxs, p):
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    stop0 = d["stop"].to_numpy(float)
    trail = np.asarray(d["trail"], float)
    ex = np.asarray(d["exit_sig"], bool)
    ts = d["timestamp"].to_numpy()
    n, out, busy = len(d), [], -1
    for i in idxs:
        e = i + 1
        if e >= n or i <= busy:
            continue
        entry = o[e] * (1 + SLIP)
        st = stop0[i]
        if not np.isfinite(st) or entry <= st:
            continue
        R = entry - st
        tp = entry + p["tp_r"] * R if p.get("tp_r") else None
        last = min(e + p["max_hold"], n - 1)
        px, why, j = None, "time", last
        for j in range(e, last + 1):
            if l[j] <= st:                                        # 1) stop
                px, why = min(st * (1 - STOP_SLIP), o[j]) * (1 - SLIP), "stop"
                break
            if tp is not None and h[j] >= tp:                     # 2) target
                px, why = tp * (1 - SLIP), "target"
                break
            if ex[j] and j + 1 < n:                               # 3) exit signal -> agli candle open
                px, why = o[j + 1] * (1 - SLIP), "signal"
                j += 1
                break
            if np.isfinite(trail[j]) and trail[j] > st:           # 4) phir trail update
                st = trail[j]
        if px is None:
            px = c[last] * (1 - SLIP)
        out.append({"t_in": ts[e], "t_out": ts[min(j, n - 1)], "ret": px * (1 - FEE) / (entry * (1 + FEE)) - 1,
                    "risk": R / entry, "why": why})
        busy = j
    return out


def pf_of(r):
    r = np.asarray(r, float)
    g, lo = r[r > 0].sum(), -r[r < 0].sum()
    return None if len(r) == 0 else (np.inf if lo == 0 else g / lo)


def p5_of(r, n=2000):
    r = np.asarray(r, float)
    if len(r) < 20:
        return None
    s = np.random.default_rng(42).choice(r, size=(n, len(r)), replace=True)
    g, lo = np.where(s > 0, s, 0).sum(1), -np.where(s < 0, s, 0).sum(1)
    return float(np.percentile(np.where(lo > 0, g / np.maximum(lo, 1e-12), 99), 5))


def fmt(x, d=2):
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def portfolio(trades):
    if not trades:
        return None
    eq, open_pos, curve = 1.0, [], []
    for t in sorted(trades, key=lambda t: t["t_in"]):
        keep = []
        for (t_out, val, ret) in open_pos:
            if t_out <= t["t_in"]:
                eq += val * ret
                curve.append((t_out, eq))
            else:
                keep.append((t_out, val, ret))
        open_pos = keep
        if len(open_pos) >= MAX_POS:
            continue
        open_pos.append((t["t_out"], min(eq * RISK_PCT / max(t["risk"], 1e-6), MAX_POS_PCT * eq), t["ret"]))
    for (t_out, val, ret) in sorted(open_pos):
        eq += val * ret
        curve.append((t_out, eq))
    s = pd.Series([v for _, v in curve], index=pd.to_datetime([t for t, _ in curve])).sort_index()
    s = s.groupby(level=0).last().resample("1D").last().ffill()
    s = pd.concat([pd.Series([1.0], index=[s.index[0] - pd.Timedelta(days=1)]), s])
    yrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    dr = s.pct_change().dropna()
    ye = s.resample("YE").last()
    return {"cagr": s.iloc[-1] ** (1 / yrs) - 1, "dd": (s / s.cummax() - 1).min(),
            "sharpe": dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0.0,
            "yearly": pd.concat([pd.Series([ye.iloc[0] - 1], index=ye.index[:1]), ye.pct_change().iloc[1:]])}


def run(frames, allowed, fn, p, ctx, fold_bounds):
    rng = np.random.default_rng(7)
    tr, rtr = [], []
    for sym, df in frames.items():
        d = fn(df, p, ctx)
        day = d["timestamp"].dt.floor("1D")
        liq = day.map(lambda x: sym in allowed.get(x, ())).to_numpy(bool)
        sig = np.where(np.asarray(d["signal"], bool) & liq)[0]
        tr += simulate(d, sig, p)
        if len(sig):
            st = d["stop"].to_numpy(float)
            pool = np.where(np.asarray(d["setup"], bool) & liq & np.isfinite(st) & (st < d["close"].to_numpy()))[0]
            k = min(len(sig), len(pool))
            if k:
                rtr += simulate(d, np.sort(rng.choice(pool, size=k, replace=False)), p)
    if len(tr) < 5:
        return None
    rets = np.array([t["ret"] for t in tr])
    tin = np.array([t["t_in"] for t in tr], dtype="datetime64[ns]")
    folds = np.clip(np.searchsorted(fold_bounds, tin, side="right") - 1, 0, N_FOLDS - 1)
    srt = np.sort(rets)[::-1]
    return {"n": len(rets), "win": (rets > 0).mean() * 100, "pf": pf_of(rets), "p5": p5_of(rets),
            "t10": pf_of(srt[10:]) if len(srt) > 10 else None, "exp": rets.mean() * 100,
            "fp": [pf_of(rets[folds == f]) for f in range(N_FOLDS)],
            "fn": [int((folds == f).sum()) for f in range(N_FOLDS)],
            "why": pd.Series([t["why"] for t in tr]).value_counts().to_dict(),
            "rpf": pf_of([t["ret"] for t in rtr]) if rtr else None, "rn": len(rtr),
            "port": portfolio(tr), "rport": portfolio(rtr) if rtr else None}


def verdict(s):
    if s is None or s["n"] < 100 or s["p5"] is None:
        return False
    if any(pf is None or pf <= 1 or n < 10 for pf, n in zip(s["fp"], s["fn"])):
        return False
    if s["t10"] is None or s["t10"] <= 1:
        return False
    ok = s["port"]["cagr"] > 0 and (s["rport"] is None or s["port"]["sharpe"] > s["rport"]["sharpe"])
    return ok and s["p5"] > 1.2 and (s["rpf"] is None or s["p5"] > s["rpf"])


# ============================ main ============================
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
                    h4[sym] = norm(df)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)}")
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})")

    daily = {s: to_daily(d).iloc[:-1].reset_index() for s, d in h4.items()}      # aakhri (adhoora) din hatao
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    ctx = {"btc_ok": (bd > ema(bd, 50))}

    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"]
                       for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    allowed = {day: set(row.dropna().nlargest(UNIVERSE).index) for day, row in dv.iterrows()}

    start = min(d["timestamp"].iloc[0] for d in h4.values()) + pd.Timedelta(days=210)
    end = max(d["timestamp"].iloc[-1] for d in h4.values())
    fold_bounds = pd.date_range(start, end, periods=N_FOLDS + 1).values

    emit("=" * 100)
    emit("SWING LAB - SMC 4H / SQUEEZE 4H / DIP DAILY - SAKHT TEST")
    emit("=" * 100)
    emit(f"Coins: {len(h4)} | Period: {start.date()} -> {end.date()} | point-in-time top-{UNIVERSE} liquid")
    emit(f"Kharcha: fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% har taraf + stop slip {STOP_SLIP*100:.2f}%")
    emit("Muqabla: Ichimoku 4H (PF ~1.9, CAGR ~26%, MaxDD ~-11%) | Donchian Daily (PF ~1.5, CAGR 20-29%, MaxDD ~-30%)")

    summary = []
    for sname, (fn, tf, base, variants) in STRATS.items():
        frames = {s: d for s, d in (h4 if tf == "4h" else daily).items()}
        emit(f"\n\n{'#' * 100}\n{sname}\n{'#' * 100}")
        for vname, over in [("BASELINE", {})] + variants:
            p = dict(base, **over)
            s = run(frames, allowed, fn, p, ctx, fold_bounds)
            ok = verdict(s)
            summary.append((sname, vname, s, ok))
            if s is None:
                emit(f"\n[FAIL] {vname}: trades nahi")
                continue
            ps, rp = s["port"], s["rport"]
            folds = " / ".join(f"{fmt(f)}({n})" for f, n in zip(s["fp"], s["fn"]))
            emit(f"\n[{'PASS' if ok else 'FAIL'}] {vname}")
            emit(f"   Har trade: n={s['n']} win={s['win']:.1f}% PF={fmt(s['pf'])} p5={fmt(s['p5'])} "
                 f"Top10-hata={fmt(s['t10'])} Exp={s['exp']:+.2f}% | Random PF={fmt(s['rpf'])} (n={s['rn']})")
            emit(f"   Folds: {folds} | Exit: {s['why']}")
            emit(f"   Portfolio: CAGR={ps['cagr']*100:+.1f}% MaxDD={ps['dd']*100:.1f}% Sharpe={fmt(ps['sharpe'])}"
                 + (f" | Random portfolio: CAGR={rp['cagr']*100:+.1f}% Sharpe={fmt(rp['sharpe'])}" if rp else ""))
            emit("   Saal-war: " + " ".join(f"{y.year}:{v*100:+.0f}%" for y, v in ps["yearly"].items()))

    emit("\n" + "=" * 100)
    emit("KHULASA")
    emit("=" * 100)
    emit(f"{'Strategy':>11} | {'Variant':>24} | {'Trades':>6} | {'Win%':>5} | {'PF':>5} | {'p5':>5} | {'Random':>6} | "
         f"{'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | Faisla")
    for sname, vname, s, ok in summary:
        if s is None:
            emit(f"{sname:>11} | {vname:>24} | trades nahi")
            continue
        emit(f"{sname:>11} | {vname:>24} | {s['n']:>6} | {s['win']:>5.1f} | {fmt(s['pf']):>5} | {fmt(s['p5']):>5} | "
             f"{fmt(s['rpf']):>6} | {s['port']['cagr']*100:>+6.1f}% | {s['port']['dd']*100:>6.1f}% | "
             f"{fmt(s['port']['sharpe']):>6} | {'PASS' if ok else 'FAIL'}")
    for sname in STRATS:
        rows = [ok for n, _, _, ok in summary if n == sname]
        emit(f"{sname}: {sum(rows)}/{len(rows)} PASS")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
