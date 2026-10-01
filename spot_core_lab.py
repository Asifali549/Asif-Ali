"""
SPOT CORE TREND-PULLBACK PRO - sakht backtest (user ka spec, 2026-10-01)
=======================================================================
Falsafa: Daily trend -> 4H trend -> controlled pullback -> support/structure -> momentum recovery
         -> volume -> local structure reclaim -> R:R theek -> BUY   (spot, long-only)

TECHNICAL SPEC (har cheez sirf BAND candles se; pivots k candles baad confirm; koi lookahead nahi)
 1. Daily bias   : band din ka close > EMA200(D) aur EMA50(D) > EMA200(D)
 2. 4H trend     : EMA50 > EMA200, close > EMA200, ADX(14) >= adx_min (default 21)
 3. Pullback     : pichli 8 candles ka sab se neecha low, pichli 20 candles ke high se 1-5 ATR neeche
                   (controlled - na bohat chhota, na crash)
 4. Support      : us pullback mein low EMA20/EMA50 zone ko chhua (low <= EMA20 + 0.25 ATR) YA
                   pichle confirmed swing high ka retest (0.5 ATR ke andar); pullback low aakhri
                   confirmed swing low (pivot 5) se ooper (structure nahi tootna) aur EMA200 se ooper
 5. Momentum     : RSI(14) 45-65 aur RSI barh raha; bullish candle (close > open), close range ke ooper 40% mein
 6. Volume       : volume >= 1.75 x pichle 20 ka average
 7. Reclaim      : close > pichli 5 candles ka sab se ooncha high (fresh - pichli candle neeche thi)
 8. Divergence   : OPTIONAL - confirmed pivot lows par price lower-low + RSI higher-low -> sirf ranking (+1)
 9. Overextension: close - EMA20 <= 2 ATR aur candle range <= 3 ATR
10. R:R/resistance: pichle 120 candles ka high agar ooper ho to (high - close) >= 1R hona chahiye
11. Stop         : pullback low - 0.5 ATR; risk 0.5-4 ATR ke beech, warna trade nahi
12. Exits        : TP1 1.5R par 40% | TP2 3R par 30% | baqi 30% Chandelier 16/5.5 trailing (TP1 ke baad
                   stop entry par = breakeven) | max 120 candles (20 din)
13. Liquidity    : har din pichle 30 din ke volume se top-100 (point-in-time)
14. Cooldown     : ek coin par 6 candles mein ek hi signal

TESTS:
 - Baseline + parameter-neighbourhood (ADX 18/25, volume 1.5/2.0, RSI 40-65 / 50-70)
 - ABLATION (spec #21): har filter ek ek kar ke hata kar - kya wo PF/DD behtar karta hai?
 - Exit variants: TP2 par sab (2.5R) / sirf trailing
 - Random-entry control (wahi daily+4H trend, random candle, wahi stop/exit), bootstrap p5, 4 folds,
   top-10 hata kar, portfolio (1% risk, max 10, coin max 20%) vs random portfolio
 - Regimes (BTC daily: bull/bear/sideways; coin volatility high/low) aur bare coins (BTC ETH BNB SOL XRP)
 - TP1 / TP2 / SL / trail / time exits ki ginti
Natija: spot_core_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES, FEE, SLIP, STOP_SLIP
from portfolio_lab import ema, rsi, atr_w, to_daily, portfolio, stats

TOP_N = 150
H4_BARS = 2400 * 6
UNIVERSE = 100
N_FOLDS = 4
MAJORS = {"BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"}
OUT = "spot_core_RESULTS.txt"

BASE = {
    "daily": True, "adx_min": 21, "use_adx": True,
    "use_pullback": True, "use_support": True,
    "rsi_lo": 45, "rsi_hi": 65, "use_rsi": True, "use_candle": True,
    "vol_mult": 1.75, "use_vol": True,
    "use_reclaim": True, "use_ext": True, "ext_atr": 2.0, "use_rr": True, "res_rr": 1.0,
    "sl_atr": 0.5, "tp1": 1.5, "tp2": 3.0, "w1": 0.4, "w2": 0.3, "exit": "partial",
    "ce_n": 16, "ce_m": 5.5, "max_hold": 120, "cool": 6,
}
VARIANTS = [
    ("BASELINE", {}),
    ("ADX 18", {"adx_min": 18}), ("ADX 25", {"adx_min": 25}),
    ("Volume 1.5x", {"vol_mult": 1.5}), ("Volume 2.0x", {"vol_mult": 2.0}),
    ("RSI 40-65", {"rsi_lo": 40}), ("RSI 50-70", {"rsi_lo": 50, "rsi_hi": 70}),
    ("Exit: TP 2.5R sab", {"exit": "tp_all", "tp2": 2.5}),
    ("Exit: sirf trailing", {"exit": "trail"}),
    ("HATAO: daily filter", {"daily": False}),
    ("HATAO: ADX", {"use_adx": False}),
    ("HATAO: pullback+support", {"use_pullback": False, "use_support": False}),
    ("HATAO: RSI recovery", {"use_rsi": False}),
    ("HATAO: bullish candle", {"use_candle": False}),
    ("HATAO: volume", {"use_vol": False}),
    ("HATAO: reclaim", {"use_reclaim": False}),
    ("HATAO: overextension", {"use_ext": False}),
    ("HATAO: R:R/resistance", {"use_rr": False}),
]


# ============================ indicators ============================
def adx(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def pivots(df, k):
    w = 2 * k + 1
    is_ph = df["high"] == df["high"].rolling(w, center=True).max()
    is_pl = df["low"] == df["low"].rolling(w, center=True).min()
    ph = df["high"].shift(k).where(is_ph.shift(k, fill_value=False))      # bar i ka pivot, bar i+k par pata
    pl = df["low"].shift(k).where(is_pl.shift(k, fill_value=False))
    return ph, pl


def attach_daily(d4, series):
    f = pd.DataFrame({"v": series.astype(float).values, "avail": series.index + pd.Timedelta(days=1)}).sort_values("avail")
    left = pd.DataFrame({"t": d4["timestamp"] + pd.Timedelta(hours=4)})
    return pd.merge_asof(left, f, left_on="t", right_on="avail", direction="backward")["v"].to_numpy()


def chandelier(df, n, m):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return (df["high"].rolling(n).max() - m * tr.rolling(n).mean()).to_numpy()


# ============================ features (har coin ek baar) ============================
def features(d):
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]
    f = {}
    a = atr_w(d)
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    dd = to_daily(d).set_index("timestamp")
    de50, de200 = ema(dd["close"], 50), ema(dd["close"], 200)
    dok = (dd["close"] > de200) & (de50 > de200)
    dok.iloc[:200] = False
    f["daily"] = attach_daily(d, dok) > 0.5
    trend = (e50 > e200) & (c > e200)
    trend.iloc[:200] = False
    f["trend"] = trend.to_numpy()
    f["adx"] = adx(d).to_numpy()
    # pullback + support
    low8 = l.rolling(8).min()
    hh20 = h.shift(1).rolling(20).max()
    depth = (hh20 - low8) / a
    f["pullback"] = ((depth >= 1.0) & (depth <= 5.0)).to_numpy()
    touch_ema = (l <= e20 + 0.25 * a).astype(float).rolling(8).max() > 0
    ph, pl = pivots(d, 5)
    ph_prev = ph.ffill().shift(8)                     # pullback se pehle ka confirmed swing high
    retest = ((low8 - ph_prev).abs() <= 0.5 * a)
    pl_prev = pl.ffill().shift(8)                     # pullback se pehle ka confirmed swing low
    intact = (low8 > pl_prev.fillna(-np.inf)) & (low8 > e200)
    f["support"] = ((touch_ema | retest) & intact).fillna(False).to_numpy()
    r = rsi(c, 14)
    f["rsi"] = r.to_numpy()
    f["rsi_up"] = (r > r.shift(1)).to_numpy()
    rng = (h - l).replace(0, np.nan)
    f["candle"] = ((c > o) & ((c - l) / rng >= 0.6)).fillna(False).to_numpy()
    f["volr"] = (v / v.rolling(20).mean().shift(1)).to_numpy()
    hh5 = h.shift(1).rolling(5).max()
    f["reclaim"] = ((c > hh5) & (c.shift(1) <= hh5.shift(1))).fillna(False).to_numpy()
    f["ext"] = ((c - e20) / a).to_numpy()
    f["range_atr"] = (rng / a).to_numpy()
    res = h.shift(1).rolling(120).max()
    stop = low8 - BASE["sl_atr"] * a
    R = c - stop
    f["stop"] = stop.to_numpy()
    f["risk_atr"] = (R / a).to_numpy()
    f["res_room"] = ((res - c) / R).where(res > c, np.inf).to_numpy()        # R mein; ooper khula = inf
    # optional divergence (ranking): aakhri do confirmed pivot lows
    plv = pl.to_numpy()
    rv = r.to_numpy()
    div = np.zeros(len(d), bool)
    last, prev = None, None
    for i in range(len(d)):
        if not np.isnan(plv[i]):
            prev, last = last, (i - 5, plv[i], rv[i - 5] if i >= 5 else np.nan)
        if last and prev and i - last[0] <= 15 and last[1] < prev[1] and last[2] > prev[2]:
            div[i] = True
    f["div"] = div
    f["ce"] = chandelier(d, BASE["ce_n"], BASE["ce_m"])
    f["atr_pct_rank"] = (a / c).rolling(500, min_periods=100).rank(pct=True).to_numpy()
    return f


def signal_mask(f, p, liq):
    m = f["trend"] & liq
    if p["daily"]:
        m &= f["daily"]
    if p["use_adx"]:
        m &= f["adx"] >= p["adx_min"]
    if p["use_pullback"]:
        m &= f["pullback"]
    if p["use_support"]:
        m &= f["support"]
    if p["use_rsi"]:
        m &= (f["rsi"] >= p["rsi_lo"]) & (f["rsi"] <= p["rsi_hi"]) & f["rsi_up"]
    if p["use_candle"]:
        m &= f["candle"]
    if p["use_vol"]:
        m &= f["volr"] >= p["vol_mult"]
    if p["use_reclaim"]:
        m &= f["reclaim"]
    if p["use_ext"]:
        m &= (f["ext"] <= p["ext_atr"]) & (f["range_atr"] <= 3.0)
    if p["use_rr"]:
        m &= f["res_room"] >= p["res_rr"]
    m &= (f["risk_atr"] >= 0.5) & (f["risk_atr"] <= 4.0) & np.isfinite(f["stop"])
    m = np.nan_to_num(m, nan=0).astype(bool)
    # cooldown
    out, last = np.zeros_like(m), -10 ** 9
    for i in np.flatnonzero(m):
        if i - last > p["cool"]:
            out[i] = True
            last = i
    return out


# ============================ exits: TP1 / TP2 / trailing ============================
def sim_trade(o, h, l, c, ts, ce, i, stop, p):
    n = len(o)
    e = i + 1
    if e >= n:
        return None
    entry = o[e] * (1 + SLIP)
    st = stop
    if not np.isfinite(st) or entry <= st:
        return None
    R = entry - st
    if p["exit"] == "partial":
        legs = [[p["w1"], entry + p["tp1"] * R, "TP1"], [p["w2"], entry + p["tp2"] * R, "TP2"], [1 - p["w1"] - p["w2"], None, "TRAIL"]]
    elif p["exit"] == "tp_all":
        legs = [[1.0, entry + p["tp2"] * R, "TP"]]
    else:
        legs = [[1.0, None, "TRAIL"]]
    fills, trail_on = [], p["exit"] == "trail"
    last = min(e + p["max_hold"], n - 1)
    j = last
    for j in range(e, last + 1):
        open_legs = [g for g in legs if g[0] > 0]
        if not open_legs:
            break
        if l[j] <= st:                                              # 1) stop pehle
            px = min(st * (1 - STOP_SLIP), o[j])
            why = "SL" if not trail_on and st < entry else ("BE" if abs(st - entry) < 1e-12 else "TRAIL")
            for g in open_legs:
                fills.append((g[0], px, why))
                g[0] = 0
            break
        for g in open_legs:                                          # 2) targets
            if g[1] is not None and h[j] >= g[1]:
                fills.append((g[0], max(g[1], o[j]), g[2]))
                g[0] = 0
                if g[2] == "TP1":
                    st = max(st, entry)                              # breakeven
                    trail_on = True
        if trail_on and np.isfinite(ce[j]) and ce[j] > st:          # 3) phir trail update
            st = ce[j]
    rem = sum(g[0] for g in legs)
    if rem > 1e-9:
        fills.append((rem, c[j], "TIME"))
    ret = sum(w * (px * (1 - SLIP) * (1 - FEE) / (entry * (1 + FEE)) - 1) for w, px, _ in fills)
    first = fills[0][2] if fills else "TIME"
    return {"t_in": ts[e], "t_out": ts[j], "entry_px": entry, "risk": R / entry, "ret": ret,
            "exits": [x[2] for x in fills], "first": first}


# ============================ stats ============================
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


def run(h4, feats, liqs, p, closes, fold_bounds, btc_reg, with_random=True):
    rng = np.random.default_rng(7)
    tr, rtr = [], []
    for sym, d in h4.items():
        f, liq = feats[sym], liqs[sym]
        m = signal_mask(f, p, liq)
        o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        ts = d["timestamp"].to_numpy()
        idx = np.flatnonzero(m)
        for i in idx:
            t = sim_trade(o, h, l, c, ts, f["ce"], i, f["stop"][i], p)
            if t:
                t.update(sym=sym, prio=float(f["div"][i]) + f["volr"][i] / 10,
                         div=bool(f["div"][i]), volreg="ooncha" if f["atr_pct_rank"][i] >= 0.5 else "neecha")
                tr.append(t)
        if with_random and len(idx):
            pool = np.flatnonzero(f["trend"] & f["daily"] & liq & (f["risk_atr"] >= 0.5) & (f["risk_atr"] <= 4.0)
                                  & np.isfinite(f["stop"]))
            k = min(len(idx), len(pool))
            if k:
                for i in np.sort(rng.choice(pool, size=k, replace=False)):
                    t = sim_trade(o, h, l, c, ts, f["ce"], i, f["stop"][i], p)
                    if t:
                        t.update(sym=sym, prio=0)
                        rtr.append(t)
    start = closes.index[0]
    tr = [t for t in tr if pd.Timestamp(t["t_in"]) >= start]
    rtr = [t for t in rtr if pd.Timestamp(t["t_in"]) >= start]
    if len(tr) < 5:
        return None
    for t in tr:
        t["regime"] = btc_reg.asof(pd.Timestamp(t["t_in"]).floor("1D") - pd.Timedelta(days=1)) if len(btc_reg) else "?"
    rets = np.array([t["ret"] for t in tr])
    tin = np.array([t["t_in"] for t in tr], dtype="datetime64[ns]")
    folds = np.clip(np.searchsorted(fold_bounds, tin, side="right") - 1, 0, N_FOLDS - 1)
    srt = np.sort(rets)[::-1]
    eq, taken = portfolio(tr, closes, "risk", 0.01)
    req = portfolio(rtr, closes, "risk", 0.01)[0] if rtr else None
    weeks = (closes.index[-1] - closes.index[0]).days / 7
    wins, losses = rets[rets > 0], rets[rets <= 0]
    allx = pd.Series([x for t in tr for x in t["exits"]]).value_counts().to_dict()
    return {"n": len(rets), "win": (rets > 0).mean() * 100, "pf": pf_of(rets), "p5": p5_of(rets),
            "t10": pf_of(srt[10:]) if len(srt) > 10 else None, "exp": rets.mean() * 100,
            "avg_win": wins.mean() * 100 if len(wins) else 0, "avg_loss": losses.mean() * 100 if len(losses) else 0,
            "fp": [pf_of(rets[folds == f]) for f in range(N_FOLDS)], "fn": [int((folds == f).sum()) for f in range(N_FOLDS)],
            "rpf": pf_of([t["ret"] for t in rtr]) if rtr else None, "port": stats(eq),
            "rport": stats(req) if req is not None else None, "taken": len(taken), "per_week": len(taken) / weeks,
            "exits": allx, "trades": tr}


def verdict(s):
    if s is None or s["n"] < 100 or s["p5"] is None or s["t10"] is None or s["t10"] <= 1:
        return False
    if any(pf is None or pf <= 1 or n < 10 for pf, n in zip(s["fp"], s["fn"])):
        return False
    ok = s["port"]["cagr"] > 0 and (s["rport"] is None or s["port"]["sharpe"] > s["rport"]["sharpe"])
    return ok and s["p5"] > 1.2 and (s["rpf"] is None or s["p5"] > s["rpf"])


def group_line(tr, key):
    out = []
    for g in sorted({t[key] for t in tr}, key=str):
        r = [t["ret"] for t in tr if t[key] == g]
        out.append(f"{g}: n={len(r)} win={np.mean(np.array(r) > 0) * 100:.0f}% PF={fmt(pf_of(r))}")
    return " | ".join(out)


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
        for must in MAJORS:
            if must not in coins:
                coins.insert(0, must)
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
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    allowed = {day: set(row.dropna().nlargest(UNIVERSE).index) for day, row in dv.iterrows()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes = closes[closes.index >= start]
    fold_bounds = pd.date_range(start, closes.index[-1], periods=N_FOLDS + 1).values
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    b50, b200 = ema(bd, 50), ema(bd, 200)
    btc_reg = pd.Series(np.where((bd > b200) & (b50 > b200), "bull", np.where((bd < b200) & (b50 < b200), "bear", "sideways")),
                        index=bd.index)

    feats, liqs = {}, {}
    for sym, d in h4.items():
        feats[sym] = features(d)
        day = d["timestamp"].dt.floor("1D")
        liqs[sym] = day.map(lambda x: sym in allowed.get(x, ())).to_numpy(bool)

    emit("=" * 110)
    emit("SPOT CORE TREND-PULLBACK PRO - SAKHT TEST (4H, daily bias, spot long-only)")
    emit("=" * 110)
    emit(f"Coins: {len(h4)} | Period: {start.date()} -> {closes.index[-1].date()} | top-{UNIVERSE} liquid (point-in-time)")
    emit(f"Kharcha: fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% har taraf + stop slip {STOP_SLIP*100:.2f}% | "
         "portfolio: 1% risk/trade, max 10, coin max 20%")
    emit("Muqabla: Ichimoku 4H (PF ~1.9-2.4, Sharpe ~1.8) - live system")

    rows = []
    for name, over in VARIANTS:
        p = dict(BASE, **over)
        s = run(h4, feats, liqs, p, closes, fold_bounds, btc_reg)
        ok = verdict(s)
        rows.append((name, s, ok))
        if s is None:
            emit(f"\n[FAIL] {name}: trades nahi")
            continue
        ps, rp = s["port"], s["rport"]
        emit(f"\n[{'PASS' if ok else 'FAIL'}] {name}")
        emit(f"   Har trade: n={s['n']} win={s['win']:.1f}% avgWin={s['avg_win']:+.2f}% avgLoss={s['avg_loss']:+.2f}% "
             f"PF={fmt(s['pf'])} p5={fmt(s['p5'])} Top10-hata={fmt(s['t10'])} Expectancy={s['exp']:+.2f}% | Random PF={fmt(s['rpf'])}")
        emit(f"   Folds: " + " / ".join(f"{fmt(f)}({n})" for f, n in zip(s["fp"], s["fn"])) + f" | Exits: {s['exits']}")
        emit(f"   Portfolio: CAGR={ps['cagr']*100:+.1f}% MaxDD={ps['dd']*100:.1f}% Sharpe={ps['sharpe']:.2f}"
             + (f" | Random portfolio: CAGR={rp['cagr']*100:+.1f}% Sharpe={rp['sharpe']:.2f}" if rp else "")
             + f" | {s['per_week']:.1f} trades/hafta")
        emit("   Saal-war: " + " ".join(f"{y.year}:{v*100:+.0f}%" for y, v in ps["yearly"].items()))

    base = rows[0][1]
    if base:
        tr = base["trades"]
        emit("\n" + "=" * 110)
        emit("BASELINE - REGIMES, BARE COINS, DIVERGENCE")
        emit("=" * 110)
        emit("BTC regime (entry ke waqt): " + group_line(tr, "regime"))
        emit("Coin volatility: " + group_line(tr, "volreg"))
        emit("RSI divergence (optional): " + group_line(tr, "div"))
        for t in tr:
            t["major"] = "bare 5 coins" if t["sym"] in MAJORS else "baqi altcoins"
        emit("Coins: " + group_line(tr, "major"))
        per = []
        for sym in sorted(MAJORS):
            r = [t["ret"] for t in tr if t["sym"] == sym]
            if r:
                per.append(f"{sym.split('/')[0]}: n={len(r)} PF={fmt(pf_of(r))}")
        emit("Bare coins alag: " + " | ".join(per))

    emit("\n" + "=" * 110)
    emit("KHULASA (ABLATION: 'HATAO' row ka PF/CAGR BASELINE se BEHTAR ho to wo filter bekaar hai)")
    emit("=" * 110)
    emit(f"{'Variant':>26} | {'Trades':>6} | {'/hafta':>6} | {'Win%':>5} | {'PF':>5} | {'p5':>5} | {'Random':>6} | "
         f"{'Exp%':>6} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | Faisla")
    for name, s, ok in rows:
        if s is None:
            emit(f"{name:>26} | trades nahi")
            continue
        emit(f"{name:>26} | {s['n']:>6} | {s['per_week']:>6.1f} | {s['win']:>5.1f} | {fmt(s['pf']):>5} | {fmt(s['p5']):>5} | "
             f"{fmt(s['rpf']):>6} | {s['exp']:>+6.2f} | {s['port']['cagr']*100:>+6.1f}% | {s['port']['dd']*100:>6.1f}% | "
             f"{s['port']['sharpe']:>6.2f} | {'PASS' if ok else 'FAIL'}")
    core = [ok for n, _, ok in rows if not n.startswith("HATAO")]
    emit(f"\nBaseline + parosi + exits: {sum(core)}/{len(core)} PASS")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
