"""
PORTFOLIO LAB - teeno pass strategies EK SATH: kis mein kitna paisa, mil kar kitna risk?
=======================================================================================
Har system ki trades bilkul bots jaisi (entry agli candle open, fee+slip+stop slip, gap fill,
pehle stop check phir trail update), phir har system apne hisse (sleeve) mein chalta hai:
  - ICHI : Ichimoku+MS 4H, CE 16/5.5 trailing + TP 3R, 1% risk, max 10, coin max 20%, priority 60-din momentum
  - DON  : Donchian 20 daily, BTC>EMA50, CE 22/4 trailing, 1% risk, max 10, coin max 20%, priority momentum
  - DIP  : RSI3<10 uptrend + BTC>EMA50, exit close>SMA5, stop 3 ATR, max 10 din, har trade 10%, max 10
Equity rozana mark-to-market (khuli positions bhi roz ki qeemat par) - asal drawdown dikhe.
Phir mixes: har mahine shuru mein hisse wapas tay shuda % par (monthly rebalance).
Saath mein: DON 0.5% risk (kam drawdown?) aur DIP 20% size (ziada return?).
Natija: portfolio_lab_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, chandelier as ce_bot, ichi_signal, ICHI_BASE, STABLES, FEE, SLIP, STOP_SLIP

TOP_N = 150
H4_BARS = 2400 * 6
UNIVERSE = 100
OUT = "portfolio_lab_RESULTS.txt"


# ============================ helpers ============================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr_w(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def to_daily(d4):
    x = d4.set_index("timestamp")
    return x.resample("1D", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()


def sim_one(o, h, l, c, ts, i, stop, trail, exit_sig, tp_r, max_hold):
    """Ek signal (index i) ki trade - entry i+1 open. Wapsi: dict ya None."""
    n = len(o)
    e = i + 1
    if e >= n:
        return None
    entry = o[e] * (1 + SLIP)
    st = stop
    if not np.isfinite(st) or entry <= st:
        return None
    R = entry - st
    tp = entry + tp_r * R if tp_r else None
    last = min(e + max_hold, n - 1)
    px, j = None, last
    for j in range(e, last + 1):
        if l[j] <= st:
            px = min(st * (1 - STOP_SLIP), o[j]) * (1 - SLIP)
            break
        if tp is not None and h[j] >= tp:
            px = max(tp, o[j]) * (1 - SLIP)
            break
        if exit_sig is not None and exit_sig[j] and j + 1 < n:
            j += 1
            px = o[j] * (1 - SLIP)
            break
        if trail is not None and np.isfinite(trail[j]) and trail[j] > st:
            st = trail[j]
    if px is None:
        px = c[last] * (1 - SLIP)
    return {"t_in": ts[e], "t_out": ts[j], "entry_px": entry, "risk": R / entry,
            "ret": px * (1 - FEE) / (entry * (1 + FEE)) - 1}


# ============================ trade generators ============================
def trades_ichi(h4, allowed):
    out = []
    for sym, d in h4.items():
        if len(d) < 400:
            continue
        sig = ichi_signal(d, dict(ICHI_BASE))
        stop = ce_bot(d, 16, 5.5)
        mom = (d["close"] / d["close"].shift(60 * 6) - 1).to_numpy()
        o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        ts = d["timestamp"].to_numpy()
        day = d["timestamp"].dt.floor("1D")
        for i in np.where(sig)[0]:
            if sym not in allowed.get(day.iloc[i], ()):
                continue
            t = sim_one(o, h, l, c, ts, i, stop[i], stop, None, 3.0, 500)
            if t:
                t.update(sym=sym, prio=mom[i] if np.isfinite(mom[i]) else -9)
                out.append(t)
    return out


def trades_don(daily, allowed, btc_ok):
    out = []
    for sym, d in daily.items():
        c, h = d["close"], d["high"]
        brk = (c > h.shift(1).rolling(20).max()).fillna(False)
        sig = (brk & ~brk.shift(1, fill_value=False)).to_numpy() & btc_ok.reindex(d["timestamp"]).fillna(False).to_numpy(bool)
        stop = ce_bot(d, 22, 4.0)
        mom = (c / c.shift(60) - 1).to_numpy()
        o, hh, l, cc = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        ts = d["timestamp"].to_numpy()
        for i in np.where(sig)[0]:
            if sym not in allowed.get(d["timestamp"].iloc[i], ()):
                continue
            t = sim_one(o, hh, l, cc, ts, i, stop[i], stop, None, None, 365)
            if t:
                t.update(sym=sym, prio=mom[i] if np.isfinite(mom[i]) else -9)
                out.append(t)
    return out


def trades_dip(daily, allowed, btc_ok):
    out = []
    for sym, d in daily.items():
        c = d["close"]
        e50, e200 = ema(c, 50), ema(c, 200)
        trend = ((c > e200) & (e50 > e200)).to_numpy() & btc_ok.reindex(d["timestamp"]).fillna(False).to_numpy(bool)
        trend[:200] = False
        r = rsi(c, 3).to_numpy()
        sig = trend & (r < 10)
        stop = (c - 3.0 * atr_w(d)).to_numpy()
        ex = (c > c.rolling(5).mean()).to_numpy()
        o, h, l, cc = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        ts = d["timestamp"].to_numpy()
        for i in np.where(sig)[0]:
            if sym not in allowed.get(d["timestamp"].iloc[i], ()):
                continue
            t = sim_one(o, h, l, cc, ts, i, stop[i], None, ex, None, 10)
            if t:
                t.update(sym=sym, prio=-r[i])          # sab se gehri girawat pehle
                out.append(t)
    return out


# ============================ portfolio (mark-to-market) ============================
def portfolio(trades, closes, mode, size, max_pos=10, cap=0.20):
    """closes: daily close DataFrame (din x coin). mode 'risk' (size=risk%) ya 'fixed' (size=equity %)."""
    days = closes.index
    ent = sorted(trades, key=lambda t: (t["t_in"], -t["prio"]))
    by_day = {}
    for t in ent:
        by_day.setdefault(pd.Timestamp(t["t_in"]).floor("1D"), []).append(("in", t["t_in"], t))
        by_day.setdefault(pd.Timestamp(t["t_out"]).floor("1D"), []).append(("out", t["t_out"], t))
    cash, open_ = 1.0, {}                       # sym -> (trade, val)
    eq, taken = [], []
    last_px = {}
    for D in days:
        row = closes.loc[D]

        def mtm_now():                           # khuli positions ki pichle close par qeemat
            return sum(v * last_px.get(s, t["entry_px"]) / t["entry_px"] for s, (t, v) in open_.items())
        evs = sorted(by_day.get(D, []), key=lambda x: (x[1], 0 if x[0] == "out" else 1))
        for kind, _, t in evs:
            s = t["sym"]
            if kind == "out":
                if s in open_ and open_[s][0] is t:
                    tt, v = open_.pop(s)
                    cash += v * (1 + tt["ret"])
                continue
            if s in open_ or len(open_) >= max_pos:
                continue
            equity_now = cash + mtm_now()
            val = equity_now * size if mode == "fixed" else min(equity_now * size / max(t["risk"], 1e-6), cap * equity_now)
            val = min(val, cash)
            if val <= equity_now * 0.002:
                continue
            cash -= val
            open_[s] = (t, val)
            taken.append(t)
            if t["t_out"] <= t["t_in"]:                 # usi candle mein band (exit event pehle guzar chuka)
                open_.pop(s)
                cash += val * (1 + t["ret"])
        for s in open_:
            if np.isfinite(row.get(s, np.nan)):
                last_px[s] = row[s]
        eq.append(cash + sum(v * last_px.get(s, t["entry_px"]) / t["entry_px"] for s, (t, v) in open_.items()))
    return pd.Series(eq, index=days), taken


def stats(eq):
    eq = eq[eq.index >= eq.index[0]]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dr = eq.pct_change().dropna()
    ye = eq.resample("YE").last()
    yearly = pd.concat([pd.Series([ye.iloc[0] / eq.iloc[0] - 1], index=ye.index[:1]), ye.pct_change().iloc[1:]])
    mo = eq.resample("ME").last().pct_change().dropna()
    dd = eq / eq.cummax() - 1
    under = (dd < 0).astype(int)
    longest = int(under.groupby((under == 0).cumsum()).sum().max())
    return {"cagr": (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, "dd": dd.min(),
            "sharpe": dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0,
            "yearly": yearly, "pos_months": (mo > 0).mean() * 100, "worst_month": mo.min(), "long_dd": longest,
            "monthly": mo}


def mix(curves, weights):
    """Har mahine ke shuru mein sleeves wapas weights par (monthly rebalance)."""
    rets = pd.DataFrame({k: curves[k].pct_change().fillna(0) for k in weights})
    val = pd.Series(weights, dtype=float)
    out, prev_m = [], None
    for D, r in rets.iterrows():
        if prev_m is not None and D.month != prev_m:
            val = val.sum() * pd.Series(weights, dtype=float)
        prev_m = D.month
        val = val * (1 + r[val.index])
        out.append(val.sum())
    return pd.Series(out, index=rets.index)


def trade_line(tr):
    r = np.array([t["ret"] for t in tr])
    g, lo = r[r > 0].sum(), -r[r < 0].sum()
    return f"n={len(r)} win={(r > 0).mean() * 100:.1f}% PF={g / lo if lo else np.inf:.2f} avg={r.mean() * 100:+.2f}%"


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
        for must in ("ETH/USDT", "BTC/USDT"):
            if must not in coins:
                coins.insert(0, must)
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
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"]
                       for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    allowed = {day: set(row.dropna().nlargest(UNIVERSE).index) for day, row in dv.iterrows()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes = closes[closes.index >= start]

    emit("=" * 100)
    emit("PORTFOLIO LAB - Ichimoku 4H + Donchian Daily + Dip Daily EK SATH")
    emit("=" * 100)
    emit(f"Coins: {len(h4)} | Period: {closes.index[0].date()} -> {closes.index[-1].date()} | top-{UNIVERSE} liquid (point-in-time)")
    emit(f"Kharcha: fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% har taraf + stop slip {STOP_SLIP*100:.2f}% | equity rozana mark-to-market")

    T = {"ICHI": trades_ichi(h4, allowed), "DON": trades_don(daily, allowed, btc_ok), "DIP": trades_dip(daily, allowed, btc_ok)}
    for k in T:
        T[k] = [t for t in T[k] if pd.Timestamp(t["t_in"]) >= start]
    emit("\nHar signal ki trade (portfolio se pehle - tasdeeq ke liye pichle teston se milao):")
    for k, tr in T.items():
        emit(f"   {k:5}: {trade_line(tr)}")

    runs = {"ICHI": ("ICHI", "risk", 0.01), "DON": ("DON", "risk", 0.01), "DIP": ("DIP", "fixed", 0.10),
            "DON 0.5% risk": ("DON", "risk", 0.005), "DIP 20% size": ("DIP", "fixed", 0.20)}
    curves, taken = {}, {}
    for name, (k, mode, size) in runs.items():
        curves[name], taken[name] = portfolio(T[k], closes, mode, size)


    mixes = {
        "1/3 har ek (ICHI+DON+DIP)": {"ICHI": 1 / 3, "DON": 1 / 3, "DIP": 1 / 3},
        "ICHI 50 / DON 25 / DIP 25": {"ICHI": .5, "DON": .25, "DIP": .25},
        "ICHI 40 / DON 20 / DIP 40": {"ICHI": .4, "DON": .2, "DIP": .4},
        "ICHI 50 / DIP 50": {"ICHI": .5, "DIP": .5},
        "ICHI 50 / DON 50": {"ICHI": .5, "DON": .5},
        "ICHI 60 / DIP 20% 40": {"ICHI": .6, "DIP 20% size": .4},
    }
    for name, w in mixes.items():
        curves[name] = mix(curves, w)

    emit("\n" + "=" * 100)
    emit("NATIJA (har row: $1 se shuru, 6 saal)")
    emit("=" * 100)
    emit(f"{'Portfolio':>28} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'Musbat mahine':>13} | {'Bura mahina':>11} | {'Lamba DD (din)':>14} | trades")
    for name, eq in curves.items():
        s = stats(eq)
        n = len(taken[name]) if name in taken else "-"
        emit(f"{name:>28} | {s['cagr']*100:>+6.1f}% | {s['dd']*100:>6.1f}% | {s['sharpe']:>6.2f} | {s['pos_months']:>12.0f}% | "
             f"{s['worst_month']*100:>10.1f}% | {s['long_dd']:>14} | {n}")

    emit("\nSaal-war return:")
    yrs = sorted({y.year for eq in curves.values() for y in stats(eq)["yearly"].index})
    emit(f"{'Portfolio':>28} | " + " | ".join(f"{y:>6}" for y in yrs))
    for name, eq in curves.items():
        yr = {y.year: v for y, v in stats(eq)["yearly"].items()}
        emit(f"{name:>28} | " + " | ".join(f"{yr.get(y, np.nan)*100:>+5.0f}%" for y in yrs))

    mo = pd.DataFrame({k: stats(curves[k])["monthly"] for k in ("ICHI", "DON", "DIP")})
    emit("\nMahana return ka correlation (1 = bilkul sath chalte, 0 = alag, manfi = ulta):")
    emit(mo.corr().round(2).to_string())
    bad = mo.sort_values("ICHI").head(6)
    emit("\nICHI ke 6 sab se bure mahine - us waqt DON aur DIP kya kar rahe the:")
    emit((bad * 100).round(1).to_string())

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
