"""
FAST LAB - DIP 4H: wahi "uptrend mein girawat khareedo" edge, lekin 4H par -> rozana kai signals
================================================================================================
User ki khwahish: signals market ke sath sath milte rahein, trades taqatwar hon (hafton intezar nahi).
Daily Dip (win ~69%, PF ~2) ka edge asli tha lekin signals kam. Yahan wahi 4H candle par, 100 coins:
  Setup : coin ka DAILY close > EMA200 aur EMA50 > EMA200 (sirf band din) + BTC daily close > EMA50
  Entry : 4H RSI(3) < 10 (chand ghanton ki tez girawat) -> agli 4H candle ke open par
  Exit  : 4H close > SMA5 (4H) -> agli candle open | stop 3x ATR(14, 4H) | max 30 candles (5 din)
  Size  : har trade 20%, max 10 (Dip Daily jaisa)
Sakht usool: lookahead-free (daily filter sirf band din), fee+slip+stop slip, gap fill, random-entry
control (wahi setup, random 4H candle), bootstrap p5, 4 folds, top-10 hata kar, portfolio vs random
portfolio, parameter-neighbourhood. Saath mein: signals/hafta aur trade kitne ghante chalti hai.
Muqabla: Daily Dip (abhi live).
Natija: fast_lab_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES, FEE, SLIP, STOP_SLIP
from portfolio_lab import ema, rsi, atr_w, to_daily, sim_one, portfolio, stats, trades_dip

TOP_N = 150
H4_BARS = 2400 * 6
UNIVERSE = 100
N_FOLDS = 4
OUT = "fast_lab_RESULTS.txt"

BASE = {"rsi_lo": 10, "exit_n": 5, "stop_atr": 3.0, "max_hold": 30, "btc": True, "trend": "daily"}
VARIANTS = [("BASELINE", {})] + [(f"{k}={v}", {k: v}) for k, v in [
    ("rsi_lo", 5), ("rsi_lo", 15), ("rsi_lo", 20),
    ("exit_n", 3), ("exit_n", 8),
    ("stop_atr", 2.0), ("stop_atr", 4.0),
    ("max_hold", 12),
    ("btc", False), ("trend", "4h"),
]]


def attach_daily(d4, series):
    """Daily series (index = din) 4H candle par sirf tab jab wo din BAND ho chuka ho."""
    f = pd.DataFrame({"v": series.astype(float).values, "avail": series.index + pd.Timedelta(days=1)}).sort_values("avail")
    left = pd.DataFrame({"t": d4["timestamp"] + pd.Timedelta(hours=4)})
    return pd.merge_asof(left, f, left_on="t", right_on="avail", direction="backward")["v"].fillna(0).to_numpy() > 0.5


def dip4h_trades(h4, allowed, btc_daily_ok, p, rng=None):
    tr, rtr = [], []
    for sym, d in h4.items():
        if len(d) < 6 * 220:
            continue
        dd = to_daily(d).set_index("timestamp")
        c = d["close"]
        if p["trend"] == "daily":
            e50, e200 = ema(dd["close"], 50), ema(dd["close"], 200)
            t = (dd["close"] > e200) & (e50 > e200)
            t.iloc[:200] = False
            trend = attach_daily(d, t)
        else:
            e200 = ema(c, 200)
            trend = (c > e200).to_numpy().copy()
            trend[:200] = False
        if p["btc"]:
            trend = trend & attach_daily(d, btc_daily_ok)
        day = d["timestamp"].dt.floor("1D")
        liq = day.map(lambda x: sym in allowed.get(x, ())).to_numpy(bool)
        setup = trend & liq
        r = rsi(c, 3).to_numpy()
        sig = setup & (r < p["rsi_lo"])
        stop = (c - p["stop_atr"] * atr_w(d)).to_numpy()
        ex = (c > c.rolling(p["exit_n"]).mean()).to_numpy()
        o, h, l, cc = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        ts = d["timestamp"].to_numpy()
        idx = np.where(sig)[0]
        for i in idx:
            t_ = sim_one(o, h, l, cc, ts, i, stop[i], None, ex, None, p["max_hold"])
            if t_:
                t_.update(sym=sym, prio=-r[i])
                tr.append(t_)
        if rng is not None and len(idx):
            pool = np.where(setup & np.isfinite(stop))[0]
            k = min(len(idx), len(pool))
            for i in np.sort(rng.choice(pool, size=k, replace=False)):
                t_ = sim_one(o, h, l, cc, ts, i, stop[i], None, ex, None, p["max_hold"])
                if t_:
                    t_.update(sym=sym, prio=0)
                    rtr.append(t_)
    return tr, rtr


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


def summarize(tr, rtr, fold_bounds, closes, size, taken_only=False):
    rets = np.array([t["ret"] for t in tr])
    tin = np.array([t["t_in"] for t in tr], dtype="datetime64[ns]")
    folds = np.clip(np.searchsorted(fold_bounds, tin, side="right") - 1, 0, N_FOLDS - 1)
    srt = np.sort(rets)[::-1]
    eq, taken = portfolio(tr, closes, "fixed", size)
    req = portfolio(rtr, closes, "fixed", size)[0] if rtr else None
    hold_h = np.mean([(pd.Timestamp(t["t_out"]) - pd.Timestamp(t["t_in"])).total_seconds() / 3600 for t in taken]) if taken else 0
    weeks = (closes.index[-1] - closes.index[0]).days / 7
    return {"n": len(rets), "win": (rets > 0).mean() * 100, "pf": pf_of(rets), "p5": p5_of(rets),
            "t10": pf_of(srt[10:]) if len(srt) > 10 else None, "exp": rets.mean() * 100,
            "fp": [pf_of(rets[folds == f]) for f in range(N_FOLDS)], "fn": [int((folds == f).sum()) for f in range(N_FOLDS)],
            "rpf": pf_of([t["ret"] for t in rtr]) if rtr else None,
            "port": stats(eq), "rport": stats(req) if req is not None else None,
            "taken": len(taken), "per_week": len(taken) / weeks, "hold_h": hold_h}


def verdict(s):
    if s["n"] < 100 or s["p5"] is None or s["t10"] is None or s["t10"] <= 1:
        return False
    if any(pf is None or pf <= 1 or n < 10 for pf, n in zip(s["fp"], s["fn"])):
        return False
    ok = s["port"]["cagr"] > 0 and (s["rport"] is None or s["port"]["sharpe"] > s["rport"]["sharpe"])
    return ok and s["p5"] > 1.2 and (s["rpf"] is None or s["p5"] > s["rpf"])


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

    daily = {s: to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc_ok = bd > ema(bd, 50)
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    allowed = {day: set(row.dropna().nlargest(UNIVERSE).index) for day, row in dv.iterrows()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes = closes[closes.index >= start]
    fold_bounds = pd.date_range(start, closes.index[-1], periods=N_FOLDS + 1).values

    emit("=" * 100)
    emit("FAST LAB - DIP 4H (uptrend mein chand ghanton ki girawat khareedo)")
    emit("=" * 100)
    emit(f"Coins: {len(h4)} | Period: {start.date()} -> {closes.index[-1].date()} | top-{UNIVERSE} liquid | size 20%/trade, max 10")
    emit(f"Kharcha: fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% har taraf + stop slip {STOP_SLIP*100:.2f}%")

    rows = []
    # muqabla: Daily Dip (live)
    rng = np.random.default_rng(7)
    dtr = [t for t in trades_dip(daily, allowed, btc_ok) if pd.Timestamp(t["t_in"]) >= start]
    s = summarize(dtr, [], fold_bounds, closes, 0.20)
    rows.append(("DAILY DIP (abhi live)", s, None))

    for name, over in VARIANTS:
        p = dict(BASE, **over)
        tr, rtr = dip4h_trades(h4, allowed, btc_ok, p, rng=np.random.default_rng(7))
        tr = [t for t in tr if pd.Timestamp(t["t_in"]) >= start]
        rtr = [t for t in rtr if pd.Timestamp(t["t_in"]) >= start]
        if len(tr) < 5:
            emit(f"\n[FAIL] {name}: trades nahi")
            continue
        s = summarize(tr, rtr, fold_bounds, closes, 0.20)
        ok = verdict(s)
        rows.append((name, s, ok))

    for name, s, ok in rows:
        ps, rp = s["port"], s["rport"]
        folds = " / ".join(f"{fmt(f)}({n})" for f, n in zip(s["fp"], s["fn"]))
        tag = "MUQABLA" if ok is None else ("PASS" if ok else "FAIL")
        emit(f"\n[{tag}] {name}")
        emit(f"   Har trade: n={s['n']} win={s['win']:.1f}% PF={fmt(s['pf'])} p5={fmt(s['p5'])} Top10-hata={fmt(s['t10'])} "
             f"avg={s['exp']:+.2f}% | Random PF={fmt(s['rpf'])}")
        emit(f"   Folds: {folds}")
        emit(f"   Portfolio: CAGR={ps['cagr']*100:+.1f}% MaxDD={ps['dd']*100:.1f}% Sharpe={ps['sharpe']:.2f}"
             + (f" | Random portfolio: CAGR={rp['cagr']*100:+.1f}% Sharpe={rp['sharpe']:.2f}" if rp else ""))
        emit(f"   Li gayi trades: {s['taken']} = {s['per_week']:.1f}/hafta | trade ~{s['hold_h']:.0f} ghante | "
             "Saal-war: " + " ".join(f"{y.year}:{v*100:+.0f}%" for y, v in ps["yearly"].items()))

    emit("\n" + "=" * 100)
    emit("KHULASA")
    emit("=" * 100)
    emit(f"{'Variant':>22} | {'Trades':>6} | {'/hafta':>6} | {'Ghante':>6} | {'Win%':>5} | {'PF':>5} | {'p5':>5} | {'Random':>6} | "
         f"{'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | Faisla")
    for name, s, ok in rows:
        tag = "-" if ok is None else ("PASS" if ok else "FAIL")
        emit(f"{name:>22} | {s['n']:>6} | {s['per_week']:>6.1f} | {s['hold_h']:>6.0f} | {s['win']:>5.1f} | {fmt(s['pf']):>5} | "
             f"{fmt(s['p5']):>5} | {fmt(s['rpf']):>6} | {s['port']['cagr']*100:>+6.1f}% | {s['port']['dd']*100:>6.1f}% | "
             f"{s['port']['sharpe']:>6.2f} | {tag}")
    npass = sum(1 for _, _, ok in rows if ok)
    emit(f"\nDIP 4H: {npass}/{len(rows) - 1} PASS")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
