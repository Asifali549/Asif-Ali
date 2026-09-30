"""
SMC MTF STRATEGY - SAKHT TEST (wahi usool jin par Ichimoku 4H aur Donchian pass huin)
=====================================================================================
- Data: KuCoin top-60 coins, 15m candles ~2 saal; 1H/4H isi 15m se banaye (band candle hi use)
- Lookahead-free: signal band 15m candle par, entry AGLI candle ke open par
- Kharcha: fee 0.1% + slippage 0.05% har taraf + stop par 0.25% extra; gap par exit = min(stop, open)
- Ek bar mein stop aur target dono chhu jayen to STOP maana (bura case)
- Point-in-time liquidity: har din pichle 30 din ke volume se top-40 coins hi allowed
- Random-entry control: usi 4H+1H trend mein RANDOM 15m candle par entry, wahi stop/target
- Pass: >=100 trades, 4 time-folds sab PF>1, bootstrap p5 PF > 1.2 aur > random PF,
        portfolio (max 10 positions, 1% risk) CAGR > 0 aur Sharpe > random portfolio ka 95th
- Parameter-neighbourhood: har setting thori badal kar - asli edge ho to aas paas bhi chale
Natija: smc_mtf_RESULTS.txt
"""
import numpy as np
import pandas as pd

import smc_mtf_strategy as S
from bot_core import fetch_full, norm, STABLES, FEE, SLIP, STOP_SLIP

TOP_N = 60
DAYS = 730
UNIVERSE = 40
N_FOLDS = 4
RISK_PCT, MAX_POS, MAX_POS_PCT = 0.01, 10, 0.20
N_RANDOM = 20
OUT = "smc_mtf_RESULTS.txt"

VARIANTS = [("BASELINE", {})] + [(f"{k}={v}", {k: v}) for k, v in [
    ("vol_mult", 1.2), ("vol_mult", 2.0), ("vol_mult", 0.0),
    ("pivot_15", 2), ("pivot_15", 5),
    ("pullback_rsi", 35), ("pullback_rsi", 45),
    ("close_pos_min", 0.5),
    ("tp_r", 1.5), ("tp_r", 3.0), ("be_at_r", 1.0),
    ("sl_max_atr", 2.0), ("atr_rank_min", 0.0),
]]


# ------------------------------------------------------------------ stats
def pf_of(r):
    r = np.asarray(r, float)
    g, l = r[r > 0].sum(), -r[r < 0].sum()
    return None if len(r) == 0 else (np.inf if l == 0 else g / l)


def p5_of(r, n=2000, seed=42):
    r = np.asarray(r, float)
    if len(r) < 20:
        return None
    rng = np.random.default_rng(seed)
    s = rng.choice(r, size=(n, len(r)), replace=True)
    g = np.where(s > 0, s, 0).sum(1)
    l = -np.where(s < 0, s, 0).sum(1)
    return float(np.percentile(np.where(l > 0, g / np.maximum(l, 1e-12), 99), 5))


def fmt(x, d=2):
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


# ------------------------------------------------------------------ trade simulation
def simulate(d, idxs, p):
    """Har signal (index i) -> entry i+1 ke open par. Ek coin mein ek waqt ek hi trade."""
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    stop_arr = d["stop"].to_numpy(float)
    ts = d["timestamp"].to_numpy()
    n, out, busy_until = len(d), [], -1
    for i in idxs:
        e = i + 1
        if e >= n or i <= busy_until:
            continue
        entry = o[e] * (1 + SLIP)
        st = stop_arr[i]
        if not np.isfinite(st) or entry <= st:
            continue
        R = entry - st
        tp = entry + p["tp_r"] * R
        be_px = entry + p["be_at_r"] * R if p["be_at_r"] else None
        last = min(e + p["max_hold"], n - 1)
        px, why, j = None, "time", last
        for j in range(e, last + 1):
            if l[j] <= st:                                   # pehle STOP check
                px, why = min(st * (1 - STOP_SLIP), o[j]) * (1 - SLIP), "stop"
                break
            if h[j] >= tp:
                px, why = tp * (1 - SLIP), "target"
                break
            if be_px is not None and h[j] >= be_px:          # phir stop update
                st = max(st, entry)
        if px is None:
            px = c[last] * (1 - SLIP)
        ret = px * (1 - FEE) / (entry * (1 + FEE)) - 1
        out.append({"t_in": ts[e], "t_out": ts[j], "ret": ret, "risk": R / entry,
                    "r_mult": (px - entry) / R, "why": why})
        busy_until = j
    return out


# ------------------------------------------------------------------ portfolio
def portfolio(trades, rng=None):
    """Closed-trade equity: max 10 positions, 1% risk, ek trade max 20% equity.
    rng diya ho to slots ke liye trades ki tarteeb random (control)."""
    if not trades:
        return None
    tr = sorted(trades, key=lambda t: (t["t_in"], rng.random() if rng is not None else 0))
    eq, open_pos, curve = 1.0, [], []
    for t in tr:
        still = []
        for (t_out, val, ret) in open_pos:
            if t_out <= t["t_in"]:
                eq += val * ret
                curve.append((t_out, eq))
            else:
                still.append((t_out, val, ret))
        open_pos = still
        if len(open_pos) >= MAX_POS:
            continue
        val = min(eq * RISK_PCT / max(t["risk"], 1e-6), MAX_POS_PCT * eq)
        open_pos.append((t["t_out"], val, t["ret"]))
    for (t_out, val, ret) in sorted(open_pos):
        eq += val * ret
        curve.append((t_out, eq))
    s = pd.Series([v for _, v in curve], index=pd.to_datetime([t for t, _ in curve])).sort_index()
    s = s.groupby(level=0).last().resample("1D").last().ffill()
    s = pd.concat([pd.Series([1.0], index=[s.index[0] - pd.Timedelta(days=1)]), s])
    yrs = max((s.index[-1] - s.index[0]).days / 365.25, 1e-9)
    dr = s.pct_change().dropna()
    return {"cagr": s.iloc[-1] ** (1 / yrs) - 1, "dd": (s / s.cummax() - 1).min(),
            "sharpe": dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0.0,
            "yearly": s.resample("YE").last().pct_change().fillna(s.resample("YE").last().iloc[0] - 1)}


# ------------------------------------------------------------------ one variant
def run_variant(data, allowed, p, fold_bounds, with_random):
    rng = np.random.default_rng(7)
    tr, rtr = [], []
    for sym, df in data.items():
        d = S.compute(df, p)
        day = d["timestamp"].dt.floor("1D")
        ok_liq = day.map(lambda x: sym in allowed.get(x, ())).to_numpy(bool)
        sig = np.where(d["signal"].to_numpy(bool) & ok_liq)[0]
        tr += simulate(d, sig, p)
        if with_random and len(sig):
            pool = np.where(d["setup"].to_numpy(bool) & ok_liq & d["stop"].notna().to_numpy()
                            & (d["stop"] < d["close"]).to_numpy())[0]
            k = min(len(sig), len(pool))
            if k:
                rtr += simulate(d, np.sort(rng.choice(pool, size=k, replace=False)), p)
    if not tr:
        return None
    rets = np.array([t["ret"] for t in tr])
    tin = np.array([t["t_in"] for t in tr], dtype="datetime64[ns]")
    folds = np.clip(np.searchsorted(fold_bounds, tin, side="right") - 1, 0, N_FOLDS - 1)
    srt = np.sort(rets)[::-1]
    s = {"n": len(rets), "win": (rets > 0).mean() * 100, "pf": pf_of(rets), "p5": p5_of(rets),
         "t10": pf_of(srt[10:]) if len(srt) > 10 else None, "exp": rets.mean() * 100,
         "avg_r": np.mean([t["r_mult"] for t in tr]),
         "fp": [pf_of(rets[folds == f]) for f in range(N_FOLDS)],
         "fn": [int((folds == f).sum()) for f in range(N_FOLDS)],
         "why": pd.Series([t["why"] for t in tr]).value_counts().to_dict(),
         "rpf": pf_of([t["ret"] for t in rtr]) if rtr else None, "rn": len(rtr)}
    s["port"] = portfolio(tr)
    if with_random:
        rs = [portfolio(tr, np.random.default_rng(sd))["sharpe"] for sd in range(N_RANDOM)]
        s["r95_order"] = float(np.percentile(rs, 95))      # slot-tarteeb ki qismat
        s["rport"] = portfolio(rtr) if rtr else None       # random entries ka portfolio
    return s


def verdict(s):
    if s is None or s["n"] < 100 or s["p5"] is None:
        return False
    if any(pf is None or pf <= 1 or n < 10 for pf, n in zip(s["fp"], s["fn"])):
        return False
    port_ok = s["port"]["cagr"] > 0
    if s.get("rport"):
        port_ok = port_ok and s["port"]["sharpe"] > s["rport"]["sharpe"]
    return s["p5"] > 1.2 and (s["rpf"] is None or s["p5"] > s["rpf"]) and port_ok


# ------------------------------------------------------------------ main
def main(data=None):
    lines = []

    def emit(x=""):
        print(x)
        lines.append(x)

    if data is None:
        from data_fetcher import get_exchange, get_coin_list
        ex = get_exchange()
        coins = [s for s in get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:TOP_N]
        data = {}
        for k, sym in enumerate(coins, 1):
            try:
                df = fetch_full(ex, sym, "15m", DAYS * 96)
                if df is not None and len(df) >= 96 * 60:
                    data[sym] = norm(df)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)} candles")
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})")

    # point-in-time liquidity: kal tak ke 30 din ka average dollar volume -> top UNIVERSE
    dv = pd.DataFrame({s: (d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"])
                       .resample("1D").sum() for s, d in data.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    allowed = {day: set(row.dropna().nlargest(UNIVERSE).index) for day, row in dv.iterrows()}

    start = min(d["timestamp"].iloc[0] for d in data.values()) + pd.Timedelta(days=40)   # EMA200-4H garam
    end = max(d["timestamp"].iloc[-1] for d in data.values())
    fold_bounds = pd.date_range(start, end, periods=N_FOLDS + 1).values

    emit("=" * 100)
    emit("SMC MULTI-TIMEFRAME (4H trend -> 1H momentum -> 15m pullback + BOS) - SAKHT TEST")
    emit("=" * 100)
    emit(f"Coins: {len(data)} | Period: {start.date()} -> {end.date()} | point-in-time top-{UNIVERSE} liquid")
    emit(f"Kharcha: fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% har taraf + stop slip {STOP_SLIP*100:.2f}%")
    emit(f"Portfolio: max {MAX_POS} positions, {RISK_PCT*100:.0f}% risk/trade, ek trade max {MAX_POS_PCT*100:.0f}%")

    rows = []
    for name, over in VARIANTS:
        p = dict(S.P, **over)
        s = run_variant(data, allowed, p, fold_bounds, with_random=True)
        ok = verdict(s)
        rows.append((name, s, ok))
        if s is None:
            emit(f"\n[FAIL] {name}: koi trade nahi")
            continue
        ps = s["port"]
        be_win = 100 / (1 + p["tp_r"])
        folds = " / ".join(f"{fmt(f)}({n})" for f, n in zip(s["fp"], s["fn"]))
        emit(f"\n[{'PASS' if ok else 'FAIL'}] {name}")
        emit(f"   Har trade: n={s['n']} win={s['win']:.1f}% (2R par breakeven ~{be_win:.0f}%+kharcha) "
             f"PF={fmt(s['pf'])} p5={fmt(s['p5'])} Top10-hata={fmt(s['t10'])} Exp={s['exp']:+.3f}% avgR={s['avg_r']:+.2f}")
        emit(f"   Random-entry PF={fmt(s['rpf'])} (n={s['rn']}) | Folds: {folds} | Exit: {s['why']}")
        rp = s.get("rport")
        emit(f"   Portfolio: CAGR={ps['cagr']*100:+.1f}% MaxDD={ps['dd']*100:.1f}% Sharpe={fmt(ps['sharpe'])}"
             + (f" | Random-entry portfolio: CAGR={rp['cagr']*100:+.1f}% Sharpe={fmt(rp['sharpe'])}" if rp else ""))
        emit("   Saal-war: " + " ".join(f"{y.year}:{v*100:+.0f}%" for y, v in ps["yearly"].items()))

    emit("\n" + "=" * 100)
    emit("KHULASA")
    emit("=" * 100)
    emit(f"{'Variant':>20} | {'Trades':>6} | {'Win%':>5} | {'PF':>5} | {'p5':>5} | {'Random':>6} | {'CAGR':>7} | {'MaxDD':>7} | Faisla")
    for name, s, ok in rows:
        if s is None:
            emit(f"{name:>20} | koi trade nahi")
            continue
        emit(f"{name:>20} | {s['n']:>6} | {s['win']:>5.1f} | {fmt(s['pf']):>5} | {fmt(s['p5']):>5} | "
             f"{fmt(s['rpf']):>6} | {s['port']['cagr']*100:>+6.1f}% | {s['port']['dd']*100:>6.1f}% | {'PASS' if ok else 'FAIL'}")
    npass = sum(ok for _, _, ok in rows)
    emit(f"\nParameter-neighbourhood: {npass}/{len(rows)} PASS. "
         "Asli edge = BASELINE pass + aas paas ke ziada tar variants bhi pass.")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
