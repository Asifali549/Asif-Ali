"""
STRATEGY LAB - Daily / 4H spot (buy-only) strategies ki imaandaar jaanch
=======================================================================
1h par breakout/pullback/CHoCH sab mein lookahead hatane ke baad koi mazboot
edge nahi mila. Is liye ab un tareeqon ko jaancha ja raha hai jo crypto mein
aam taur par ziyada mazboot maane jate hain - bare timeframe aur momentum.

HISSA A - MOMENTUM ROTATION (portfolio, daily data, ~4 saal)
  Har hafte (7 din) band candle par: point-in-time sab se liquid coins mein se
  pichle L din mein sab se ziyada barhne wale top K coins kharido (barabar
  hissa), agle hafte dobara check. Sirf woh coin jis ka momentum > 0 ho.
  Grid: L = 30/60/90 din, K = 5/10, score = seedha return ya volatility-adjusted,
        regime = koi nahi / BTC close > BTC EMA50 (warna poora cash)
  Moqabla (benchmark): BTC buy&hold, BTC + EMA50 regime (in/out),
        equal-weight (top liquid coins, har hafte rebalance)

HISSA B - DAILY TREND FOLLOWING (har coin par trades)
  B1 Donchian breakout: close > pichle N din ka sab se ooncha high (N=20/40/55)
  B2 EMA cross: EMA fast upar se cross kare EMA slow (10/30, 20/50)
  Exits: Chandelier trailing (22 din, 3x/4x ATR), ya signal-exit
        (Donchian: close < pichle N/2 din ka low; EMA: fast neeche cross)
  Regime filter: koi nahi / BTC > EMA50 / BTC > EMA200 / coin > apna EMA200

HISSA C - 4H TREND FOLLOWING (~2 saal)
  Donchian 4H (N=20/55) aur EMA cross 4H (20/50), Chandelier exit (22, 3x/4x),
  regime: koi nahi / BTC daily > EMA50 (sirf BAND daily candle se)

IMAANDARI KE USOOL (sab parts mein):
  - Signal candle BAND hone par, entry AGLI candle ke open par (lookahead nahi)
  - 4H par daily regime sirf band hui daily candle se
  - Fee 0.1% + slippage 0.05% har taraf; stop exit gap-fill (min(stop, open))
  - Folds = WAQT ke 4 barabar hisse (sab coins ke liye ek hi calendar)
  - Data-completeness report: har timeframe ki candles aur tareekhein
  - Stop exit par extra 0.25% slippage (tez girawat mein stop neeche fill hota hai)
  - RANDOM CONTROL: har strategy ka moqabla usi exit/filter ke sath RANDOM
    entries se; rotation ka moqabla har hafte K random coins (30 seeds) se.
    (Test ne khud sabit kiya ke trailing-stop exits random entries par bhi
    PF>1 dikha sakte hain - is liye sirf PF>1 kaafi nahi.)
  - PASS (trades): har fold PF>1 (>=10 trades), kul >=100, bootstrap p5>1.2,
    AUR p5 > usi setup ka random-entry PF
  - PASS (rotation): har fold >0, Sharpe > random-pick ka 95th-percentile,
    Sharpe > BTC+EMA50, MaxDD behtar -50% se
  - NOTE survivorship bias: coins ki list AAJ ki hai (jo coins mar gaye wo
    shamil nahi) - natije thore behtar nazar aa sakte hain. Rotation mein isay
    kam karne ke liye liquidity-rank har hafte point-in-time nikalte hain.

Natija: strategy_lab_RESULTS.txt
"""

import numpy as np
import pandas as pd

import config
from backtest_engine import compute_atr

TOP_N_COINS = 150
DAILY_LIMIT = 1500          # ~4.1 saal
H4_LIMIT = 4380             # ~2 saal
N_FOLDS = 4
FEE = config.BACKTEST_PARAMS["fee_pct"] / 100
SLIP = config.BACKTEST_PARAMS["slippage_pct"] / 100
COST = FEE + SLIP           # har taraf
N_BOOTSTRAP = 5000
MIN_TOTAL_TRADES = 100
MIN_FOLD_TRADES = 10
STOP_SLIP = 0.0025          # stop-exit par extra slippage (tez girawat mein stop neeche fill hota hai)
N_RANDOM_SEEDS = 30         # rotation ka random-pick control
MAX_HOLD_DAILY = 365
MAX_HOLD_4H = 6 * 120

STABLES = {"USDC", "USDT", "DAI", "TUSD", "FDUSD", "USDD", "USDP", "PYUSD", "BUSD", "USDE", "EURC",
           "EUR", "UST", "USTC", "USD1", "RLUSD", "USDS", "PAXG", "XAUT"}


# =====================================================================
# Helpers
# =====================================================================
def norm(df):
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"]).astype("datetime64[ns]")
    return d.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def pf_of(r):
    r = np.asarray(r, float)
    if len(r) == 0:
        return None
    w, l = r[r > 0].sum(), r[r <= 0].sum()
    return w / abs(l) if l != 0 else None


def bootstrap_p5(r, seed=42):
    r = np.asarray(r, float)
    n = len(r)
    if n < 10:
        return None
    rng = np.random.default_rng(seed)
    out, done = [], 0
    chunk = max(1, min(N_BOOTSTRAP, 2_000_000 // n))
    while done < N_BOOTSTRAP:
        k = min(chunk, N_BOOTSTRAP - done)
        s = r[rng.integers(0, n, (k, n))]
        w = np.where(s > 0, s, 0).sum(1)
        l = np.abs(np.where(s <= 0, s, 0).sum(1))
        ok = l > 0
        out.append(w[ok] / l[ok])
        done += k
    out = np.concatenate(out)
    return float(np.percentile(out, 5)) if len(out) else None


def fmt(x, d=2):
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def chandelier(df, period, mult):
    return (df["high"].rolling(period).max() - mult * compute_atr(df, period)).values


# =====================================================================
# Trade simulation (per coin)
# =====================================================================
def simulate(op, hi, lo, cl, entries, stop=None, exit_sig=None, max_hold=365):
    """entries: signal-bar indexes (entry agli bar ke open par).
    stop: trailing stop array (check pehle, update baad mein) ya None.
    exit_sig: bool array - close par True ho to agli bar ke open par exit.
    Ek waqt mein ek hi position (overlap nahi)."""
    n = len(op)
    out = []
    busy_until = -1
    for i in entries:
        if i <= busy_until or i + 1 >= n:
            continue
        eb = i + 1
        entry = op[eb] * (1 + SLIP)
        trail = None
        if stop is not None:
            if np.isnan(stop[i]) or stop[i] >= entry:
                continue
            trail = float(stop[i])
        exit_px = exit_bar = None
        last = min(eb + max_hold, n)
        for j in range(eb, last):
            if trail is not None and lo[j] <= trail:
                exit_px, exit_bar = min(trail * (1 - STOP_SLIP), op[j]), j
                break
            if exit_sig is not None and exit_sig[j] and j + 1 < n:
                exit_px, exit_bar = op[j + 1], j + 1
                break
            if trail is not None and not np.isnan(stop[j]) and stop[j] > trail:
                trail = float(stop[j])
        if exit_px is None:
            exit_bar = last - 1
            exit_px = cl[exit_bar]
        exit_px *= (1 - SLIP)
        ret = ((exit_px - entry) / entry - 2 * FEE) * 100
        out.append((eb, ret, exit_bar - eb))
        busy_until = exit_bar
    return out


def fresh(cond):
    cond = cond.fillna(False).astype(bool)
    return cond & ~cond.shift(1, fill_value=False)


# =====================================================================
# Sahi data fetch (data_fetcher.fetch_ohlcv mein pagination bug hai: 1000 se
# ziada candles par beech mein bara khali soorakh reh jata hai - misal 1500
# daily mangne par ~500 din gayab). Yahan AAGE ki taraf (purane se naye)
# page karte hain aur har page ke baad agli candle se shuru karte hain.
# =====================================================================
TF_MS = {"1d": 86_400_000, "4h": 14_400_000, "1h": 3_600_000}


def fetch_full(exchange, symbol, timeframe, limit):
    import time
    step = TF_MS[timeframe]
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    since = now_ms - limit * step
    rows, empty_jumps = [], 0
    while since < now_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        time.sleep(exchange.rateLimit / 1000)
        batch = [b for b in batch if b[0] >= since] if batch else []
        if not batch:
            since += 1000 * step          # coin shayad baad mein list hua - aage jump
            empty_jumps += 1
            if empty_jumps > 20:
                break
            continue
        rows.extend(batch)
        since = batch[-1][0] + step
        if len(batch) < 5 and since >= now_ms - 2 * step:
            break
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"] + step <= now_ms]            # sirf BAND candles
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.reset_index(drop=True)


def gap_count(df, timeframe):
    d = df["timestamp"].diff().dt.total_seconds() * 1000
    return int((d > 1.5 * TF_MS[timeframe]).sum())


# =====================================================================
# Data completeness
# =====================================================================
def data_report(name, frames, full_len, tf):
    lens = [len(d) for d in frames.values()]
    if not lens:
        return [f"{name}: KOI DATA NAHI"]
    starts = [d["timestamp"].iloc[0] for d in frames.values()]
    ends = [d["timestamp"].iloc[-1] for d in frames.values()]
    full = sum(1 for x in lens if x >= full_len * 0.98)
    return [f"{name}: coins={len(lens)} | candles median={int(np.median(lens))} min={min(lens)} max={max(lens)} "
            f"| poori history (>= {int(full_len*0.98)}) wale coins={full} "
            f"| sab se purani tareekh={min(starts).date()} | aakhri tareekh={max(ends).date()}",
            f"   {name}: data mein khali soorakh (gap) wale coins = "
            f"{sum(1 for d in frames.values() if gap_count(d, tf) > 0)} / {len(lens)}"]


# =====================================================================
# HISSA A - Momentum rotation
# =====================================================================
def build_panel(daily):
    dates = sorted(set().union(*[set(d["timestamp"]) for d in daily.values()]))
    idx = pd.DatetimeIndex(dates)
    syms = list(daily.keys())
    O = pd.DataFrame({s: daily[s].set_index("timestamp")["open"] for s in syms}).reindex(idx)
    C = pd.DataFrame({s: daily[s].set_index("timestamp")["close"] for s in syms}).reindex(idx)
    V = pd.DataFrame({s: daily[s].set_index("timestamp")["volume"] for s in syms}).reindex(idx)
    return idx, O, C, V


def equity_stats(eq, idx):
    eq = pd.Series(eq, index=idx).dropna()
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan
    bounds = np.linspace(0, len(eq) - 1, N_FOLDS + 1).astype(int)
    folds = [eq.iloc[bounds[k + 1]] / eq.iloc[bounds[k]] - 1 for k in range(N_FOLDS)]
    yearly = eq.groupby(eq.index.year).agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    return {"total": eq.iloc[-1] / eq.iloc[0] - 1, "cagr": cagr, "dd": dd, "sharpe": sharpe,
            "folds": folds, "yearly": yearly}


def run_rotation(idx, O, C, V, btc_close, L, K, score_kind, regime, universe=100, rebalance=7,
                 start_i=None, seed=None):
    rng = np.random.default_rng(seed)
    T, N = C.shape
    Cv, Ov = C.values, O.values
    ret_oo = np.zeros((T, N))
    ret_oo[:-1] = Ov[1:] / Ov[:-1] - 1
    ret_oo = np.nan_to_num(ret_oo, nan=0.0, posinf=0.0, neginf=0.0)
    dvol = (C * V).rolling(30, min_periods=20).mean().values
    first_valid = np.argmax(~np.isnan(Cv), axis=0)
    dret = np.log(C / C.shift(1)).values
    btc = btc_close.reindex(idx).ffill()
    btc_ok = (btc > ema(btc, 50)).values

    vals = np.zeros(N)
    cash = 1.0
    eq = np.full(T, np.nan)
    pending = None
    for t in range(start_i, T - 1):
        if pending is not None:                         # rebalance at open t
            total = cash + vals.sum()
            target = pending * total
            turnover = np.abs(target - vals).sum()
            total -= turnover * COST
            vals = pending * total
            cash = total - vals.sum()
            pending = None
        eq[t] = cash + vals.sum()
        vals = vals * (1 + ret_oo[t])                   # open t -> open t+1
        if (t - start_i) % rebalance == 0:              # signal at close t
            w = np.zeros(N)
            if regime == "none" or btc_ok[t]:
                ok = (~np.isnan(Cv[t])) & (t - first_valid >= L + 30) & (~np.isnan(dvol[t]))
                if t - L >= 0:
                    ok &= ~np.isnan(Cv[t - L])
                cand = np.where(ok)[0]
                if len(cand):
                    liq_rank = cand[np.argsort(-dvol[t, cand])][:universe]
                    mom = Cv[t, liq_rank] / Cv[t - L, liq_rank] - 1
                    if score_kind == "all":            # benchmark: sab liquid coins barabar
                        w[liq_rank] = 1.0 / len(liq_rank)
                        pending = w
                        continue
                    if score_kind == "random":         # control: K random coins (momentum>0 wali shart ke baghair)
                        pick = rng.choice(liq_rank, size=min(K, len(liq_rank)), replace=False)
                        w[pick] = 1.0 / K
                        pending = w
                        continue
                    if score_kind == "voladj":
                        vol = np.nanstd(dret[t - L + 1:t + 1, liq_rank], axis=0)
                        score = mom / np.where(vol > 0, vol, np.nan)
                    else:
                        score = mom
                    good = (mom > 0) & ~np.isnan(score)
                    pick = liq_rank[good][np.argsort(-score[good])][:K]
                    if len(pick):
                        w[pick] = 1.0 / K                # K se kam mile to baqi cash
            pending = w
    eq[T - 1] = cash + vals.sum()
    return eq


def benchmark_btc(idx, btc_df, with_regime):
    b = btc_df.set_index("timestamp").reindex(idx).ffill()
    o = b["open"].values
    r = np.zeros(len(idx))
    r[:-1] = o[1:] / o[:-1] - 1
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    ok = (b["close"] > ema(b["close"], 50)).values
    eq = np.ones(len(idx))
    pos = not with_regime
    for t in range(1, len(idx)):
        eq[t] = eq[t - 1] * (1 + (r[t - 1] if pos else 0))
        if with_regime:
            new = bool(ok[t - 1])      # close t-1 par faisla, open t se asar
            if new != pos:
                eq[t] *= (1 - COST)
            pos = new
    return eq


# =====================================================================
# HISSA B / C - per-coin trend following
# =====================================================================
def trend_entries(df, kind, p):
    c, h, lo = df["close"], df["high"], df["low"]
    if kind == "donchian":
        sig = c > h.shift(1).rolling(p).max()
        ex = c < lo.shift(1).rolling(max(p // 2, 5)).min()
    else:  # ema cross
        f, s = p
        ef, es = ema(c, f), ema(c, s)
        sig = ef > es
        ex = (ef < es).values
        return fresh(sig).values, ex
    return fresh(sig).values, ex.fillna(False).values


def regime_mask(df, btc_d, kind, shift_daily):
    n = len(df)
    if kind == "none":
        return np.ones(n, bool)
    if kind == "coin>EMA200":
        return (df["close"] > ema(df["close"], 200)).values
    span = 50 if kind == "BTC>EMA50" else 200
    b = btc_d.copy()
    b["ok"] = b["close"] > ema(b["close"], span)
    b["timestamp"] = (b["timestamp"] + shift_daily).astype("datetime64[ns]")
    m = pd.merge_asof(df[["timestamp"]], b[["timestamp", "ok"]].sort_values("timestamp"),
                      on="timestamp", direction="backward")
    return m["ok"].astype("boolean").fillna(False).to_numpy(dtype=bool)


def trend_configs(tf):
    if tf == "1d":
        entries = [("donchian", 20), ("donchian", 40), ("donchian", 55), ("ema", (10, 30)), ("ema", (20, 50))]
        regimes = ["none", "BTC>EMA50", "BTC>EMA200", "coin>EMA200"]
    else:
        entries = [("donchian", 20), ("donchian", 55), ("ema", (20, 50))]
        regimes = ["none", "BTC>EMA50"]
    exits = ["CE22x3", "CE22x4", "signal"]
    return [(e, x, r) for e in entries for x in exits for r in regimes]


def label(cfg):
    (kind, p), x, r = cfg
    ent = f"Donchian {p}" if kind == "donchian" else f"EMA {p[0]}/{p[1]}"
    return f"{ent:13s} | exit {x:7s} | {r}"


def run_trend(frames, btc_d, tf, fold_bounds, random_entries=False, seed=0):
    rng = np.random.default_rng(seed)
    shift = pd.Timedelta(0) if tf == "1d" else pd.Timedelta(hours=20)
    max_hold = MAX_HOLD_DAILY if tf == "1d" else MAX_HOLD_4H
    cfgs = trend_configs(tf)
    res = {c: [] for c in cfgs}
    for sym, df in frames.items():
        if len(df) < 260:
            continue
        op, hi, lo, cl = (df[k].values for k in ("open", "high", "low", "close"))
        ts = df["timestamp"].values
        stops = {"CE22x3": chandelier(df, 22, 3.0), "CE22x4": chandelier(df, 22, 4.0)}
        regs = {r: regime_mask(df, btc_d, r, shift) for r in ("none", "BTC>EMA50", "BTC>EMA200", "coin>EMA200")}
        ent_cache = {}
        for cfg in cfgs:
            (kind, p), x, r = cfg
            if (kind, p) not in ent_cache:
                ent_cache[(kind, p)] = trend_entries(df, kind, p)
            sig, ex = ent_cache[(kind, p)]
            pos = np.where(sig & regs[r])[0]
            if random_entries:                       # control: utni hi entries, regime-allowed bars mein random
                allowed = np.where(regs[r][:-1])[0]
                allowed = allowed[allowed >= 200]
                k = min(len(pos), len(allowed))
                pos = np.sort(rng.choice(allowed, size=k, replace=False)) if k else pos[:0]
            if x == "signal":
                tr = simulate(op, hi, lo, cl, pos, None, ex, max_hold)
            else:
                tr = simulate(op, hi, lo, cl, pos, stops[x], None, max_hold)
            for eb, ret, bars in tr:
                f = int(np.searchsorted(fold_bounds, ts[eb], side="right") - 1)
                res[cfg].append((min(max(f, 0), N_FOLDS - 1), ret, bars))
    return res


def trade_summary(tr):
    if not tr:
        return None
    rets = np.array([t[1] for t in tr])
    folds = np.array([t[0] for t in tr])
    bars = np.array([t[2] for t in tr])
    srt = np.sort(rets)[::-1]
    fp, fn = [], []
    for f in range(N_FOLDS):
        fr = rets[folds == f]
        fp.append(pf_of(fr))
        fn.append(len(fr))
    return {"n": len(rets), "win": (rets > 0).mean() * 100, "pf": pf_of(rets), "p5": bootstrap_p5(rets),
            "t10": pf_of(srt[10:]) if len(srt) > 10 else None, "exp": rets.mean(),
            "bars": np.median(bars), "fold_pf": fp, "fold_n": fn}


def passes(s, rand_pf=None):
    if s is None or s["pf"] is None or s["p5"] is None or s["n"] < MIN_TOTAL_TRADES:
        return False
    if rand_pf is not None and s["p5"] <= rand_pf:
        return False
    if any(p is None or p <= 1 or n < MIN_FOLD_TRADES for p, n in zip(s["fold_pf"], s["fold_n"])):
        return False
    return s["p5"] > 1.2


# =====================================================================
# MAIN
# =====================================================================
def main():
    from data_fetcher import get_exchange, get_coin_list
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    ex = get_exchange()
    coins = [c for c in get_coin_list(ex) if c.split("/")[0].upper() not in STABLES][:TOP_N_COINS]
    for must in ("BTC/USDT", "ETH/USDT"):
        if must not in coins:
            coins.insert(0, must)

    daily, h4, fails = {}, {}, 0
    for k, sym in enumerate(coins, 1):
        try:
            d = fetch_full(ex, sym, "1d", DAILY_LIMIT)
            if d is not None and len(d) >= 120:
                daily[sym] = norm(d)
            h = fetch_full(ex, sym, "4h", H4_LIMIT)
            if h is not None and len(h) >= 300:
                h4[sym] = norm(h)
        except Exception as e:
            fails += 1
            print(f"[{k}] {sym}: SKIP ({e})")
        if k % 25 == 0:
            print(f"[{k}/{len(coins)}] data ...")
    if "BTC/USDT" not in daily or len(daily) < 30:
        raise SystemExit("BTC ya kaafi daily data nahi mila - test rok diya.")
    btc_d = daily["BTC/USDT"]

    emit("=" * 90)
    emit("STRATEGY LAB - DAILY / 4H SPOT STRATEGIES (lookahead-free)")
    emit("=" * 90)
    emit(f"Fee {FEE*100:.2f}% + slippage {SLIP*100:.2f}% har taraf | fetch errors: {fails}/{len(coins)}")
    emit("\n----- DATA COMPLETENESS -----")
    for l in data_report("DAILY (1d)", daily, DAILY_LIMIT, "1d") + data_report("4H", h4, H4_LIMIT, "4h"):
        emit(l)
    emit("NOTE: coin list AAJ ki top-liquid list hai (survivorship bias) - asal natija thora kamzor ho sakta hai.")

    # ---------------- HISSA A ----------------
    idx, O, C, V = build_panel(daily)
    n_elig = (C.notna().cumsum() >= 120).sum(axis=1).values
    start_i = int(np.argmax(n_elig >= 30))
    btc_first = btc_d["timestamp"].iloc[0] + pd.Timedelta(days=60)
    start_i = max(start_i, int(np.searchsorted(idx.values, btc_first.to_datetime64())))
    emit("\n\n" + "#" * 90)
    emit("HISSA A - MOMENTUM ROTATION (har hafte top-K strongest coins, point-in-time top-100 liquid)")
    emit("#" * 90)
    emit(f"Period: {idx[start_i].date()} -> {idx[-1].date()}  (4 folds = waqt ke 4 barabar hisse)")
    sub = idx[start_i:]
    bench = {
        "BTC buy&hold": benchmark_btc(idx, btc_d, False)[start_i:],
        "BTC + EMA50 regime": benchmark_btc(idx, btc_d, True)[start_i:],
    }
    ew = run_rotation(idx, O, C, V, btc_d.set_index("timestamp")["close"], L=1, K=100, score_kind="all",
                      regime="none", start_i=start_i)
    bench["Equal-weight top100 (koi selection nahi)"] = ew[start_i:]

    def row(name, eq):
        s = equity_stats(eq, sub)
        folds = " / ".join(f"{x*100:+.0f}%" for x in s["folds"])
        return (f"{name:44s} total={s['total']*100:+8.0f}% CAGR={s['cagr']*100:+6.1f}% MaxDD={s['dd']*100:6.1f}% "
                f"Sharpe={fmt(s['sharpe'])} | Folds: {folds}"), s

    emit("\n--- Benchmarks ---")
    bstats = {}
    for name, eq in bench.items():
        txt, s = row(name, eq)
        bstats[name] = s
        emit(txt)

    emit("\n--- Rotation grid ---")
    rot = []
    btc_close = btc_d.set_index("timestamp")["close"]
    for regime in ("none", "BTC>EMA50"):
        for score_kind in ("raw", "voladj"):
            for L in (30, 60, 90):
                for K in (5, 10):
                    eq = run_rotation(idx, O, C, V, btc_close, L, K, score_kind, regime, start_i=start_i)[start_i:]
                    name = f"L={L:2d} K={K:2d} {score_kind:6s} regime={regime}"
                    txt, s = row(name, eq)
                    rot.append((name, s, K, regime))
                    emit(txt)

    emit("\n--- RANDOM CONTROL: har hafte K random liquid coins (30 alag seeds) ---")
    rand_p95 = {}
    for regime in ("none", "BTC>EMA50"):
        for K in (5, 10):
            sh = []
            for sd in range(N_RANDOM_SEEDS):
                eq = run_rotation(idx, O, C, V, btc_close, 30, K, "random", regime, start_i=start_i, seed=sd)[start_i:]
                sh.append(equity_stats(eq, sub)["sharpe"])
            sh = np.array(sh, float)
            rand_p95[(K, regime)] = float(np.nanpercentile(sh, 95))
            emit(f"Random K={K:2d} regime={regime:9s}: Sharpe median={np.nanmedian(sh):.2f}  95th-pct={rand_p95[(K, regime)]:.2f}")
    emit("(Strategy ka Sharpe random ke 95th-percentile se ooper hona chahiye - warna ye qismat ho sakti hai)")

    hodl = bstats["BTC buy&hold"]
    bt_reg = bstats["BTC + EMA50 regime"]
    emit("\n--- Rotation PASS shart: har fold return > 0, Sharpe > random 95th-pct, Sharpe > BTC+EMA50, MaxDD > -50% ---")
    good = [(n, s) for n, s, K, rg in rot
            if all(f > 0 for f in s["folds"]) and s["sharpe"] > rand_p95[(K, rg)]
            and s["sharpe"] > bt_reg["sharpe"] and s["dd"] > -0.5]
    if not good:
        emit("Koi rotation combo PASS nahi hua.")
    for n, s in sorted(good, key=lambda x: -x[1]["sharpe"]):
        yr = " ".join(f"{y}:{v*100:+.0f}%" for y, v in s["yearly"].items())
        emit(f"PASS  {n}  Sharpe={fmt(s['sharpe'])} CAGR={s['cagr']*100:+.1f}% MaxDD={s['dd']*100:.1f}% | saal-war: {yr}")
    emit(f"(BTC buy&hold Sharpe={fmt(hodl['sharpe'])}, BTC+EMA50 Sharpe={fmt(bt_reg['sharpe'])})")

    # ---------------- HISSA B / C ----------------
    for tf, frames, title in (("1d", daily, "HISSA B - DAILY TREND FOLLOWING"),
                              ("4h", h4, "HISSA C - 4H TREND FOLLOWING")):
        if not frames:
            emit(f"\n{title}: data nahi")
            continue
        t0 = min(d["timestamp"].iloc[0] for d in frames.values())
        t1 = max(d["timestamp"].iloc[-1] for d in frames.values())
        # folds ko wahan se shuru karo jahan kam az kam 30 coins ka data ho
        starts = sorted(d["timestamp"].iloc[0] for d in frames.values())
        t0 = max(t0, starts[min(29, len(starts) - 1)])
        fold_bounds = pd.date_range(t0, t1, periods=N_FOLDS + 1).values
        res = run_trend(frames, btc_d, tf, fold_bounds)
        rres = run_trend(frames, btc_d, tf, fold_bounds, random_entries=True, seed=1)
        rpf = {c: pf_of([t[1] for t in v]) for c, v in rres.items()}
        emit("\n\n" + "#" * 90)
        emit(title)
        emit("#" * 90)
        emit("Folds: " + " | ".join(f"F{k+1}: {pd.Timestamp(fold_bounds[k]).date()}" for k in range(N_FOLDS))
             + f" -> {pd.Timestamp(fold_bounds[-1]).date()}")
        emit("(fold se pehle ki trades F1 mein gini gayi hain)")
        stats = {c: trade_summary(v) for c, v in res.items()}
        for cfg in trend_configs(tf):
            s = stats[cfg]
            if s is None:
                emit(f"{label(cfg)}: koi trade nahi")
                continue
            folds = " / ".join(f"{fmt(p)}({n})" for p, n in zip(s["fold_pf"], s["fold_n"]))
            emit(f"{label(cfg):42s} n={s['n']:5d} win={s['win']:4.1f}% PF={fmt(s['pf'])} p5={fmt(s['p5'])} "
                 f"Top10-hata={fmt(s['t10'])} RandomPF={fmt(rpf[cfg])} Exp={s['exp']:+.2f}% hold~{s['bars']:.0f}bars | {folds}"
                 + ("  <== PASS" if passes(s, rpf[cfg]) else ""))
        ps = sorted([(c, s) for c, s in stats.items() if passes(s, rpf[c])], key=lambda x: -x[1]["p5"])
        emit(f"\n{title} - PASS: {len(ps)}")
        for c, s in ps:
            emit(f"  {label(c)}  PF={fmt(s['pf'])} p5={fmt(s['p5'])} RandomPF={fmt(rpf[c])} Top10-hata={fmt(s['t10'])} n={s['n']}")

    with open("strategy_lab_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[SAVE] strategy_lab_RESULTS.txt")


if __name__ == "__main__":
    main()
