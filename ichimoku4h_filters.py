"""
4H ICHIMOKU + MS - FILTERS KA TEST (ek waqt mein sirf EK tabdeeli)
==================================================================
Baseline = Ichimoku 4H Bot bilkul jaisa (CE 16/5.5 + TP 3R, production entry).
Phir ek waqt mein sirf EK cheez badli:
  NAYE FILTER (shamil karna):
    + BTC>EMA50   : BTC ki BAND daily candle EMA50 se ooper (warna nayi entry nahi)
    + ETH>EMA50   : ETH ki BAND daily candle EMA50 se ooper
    + RSI 40-70   : signal candle par RSI(14) 40 aur 70 ke beech
    + RSI < 75    : signal candle par RSI(14) 75 se kam (bohot overbought nahi)
    + ADX > 20    : signal candle par ADX(14) > 20 (trend mein taqat)
    + ADX > 25
  PURANI SHARTEIN HATANA:
    - Volume      : volume > 2x avg wali shart hata di
    - EMA trend   : close>EMA200 aur EMA50>EMA200 wali shart hata di (Ichimoku + MS dono se)
    - Volume + EMA trend dono
Wahi sakht usool: ~5.5 saal 4H, lookahead-free, fee+slip+stop slip, random-entry
control, portfolio (10 slots, 1% risk) + random portfolio (baseline par 20 seeds).
Ek filter tabhi faidemand maana jaye jab: chaaron folds mein behtar ya barabar,
trades bohot kam na hon (~200+), aur Sharpe + MaxDD dono behtar.
Natija: ichimoku4h_filters_RESULTS.txt
"""
import numpy as np
import pandas as pd

import config
from strategies import apply_cooldown, _golden_cross_ok
from confluence_engine import compute_adx, compute_rsi
from strategy_lab import fetch_full, norm, ema, chandelier, data_report, fmt, STABLES, FEE, SLIP, STOP_SLIP
from unified_test import portfolio, eq_stats, per_trade_block, trade_pass, build_panel, liquid_mask, N_FOLDS
from ichimoku4h_validation import trades_and_random

TOP_N_COINS = 150
H4_LIMIT = 12000
D_LIMIT = 2400
BPD = 6
MAX_HOLD = 500
PER_YEAR = 6 * 365
N_RANDOM = 20
COOL = config.SIGNAL_COOLDOWN_BARS
CE_P, CE_M, TP_R = 16, 5.5, 3.0


def fresh(cond):
    cond = cond.fillna(False).astype(bool)
    return cond & ~cond.shift(1, fill_value=False)


def ichimoku_sig(df, use_volume=True, use_trend=True):
    p = config.STRATEGY_PARAMS["ichimoku"]
    h, l, c = df["high"], df["low"], df["close"]
    tenkan = (h.rolling(p["tenkan"]).max() + l.rolling(p["tenkan"]).min()) / 2
    kijun = (h.rolling(p["kijun"]).max() + l.rolling(p["kijun"]).min()) / 2
    sa = ((tenkan + kijun) / 2).shift(p["displacement"])
    sb = ((h.rolling(p["senkou_b"]).max() + l.rolling(p["senkou_b"]).min()) / 2).shift(p["displacement"])
    cloud_top = pd.concat([sa, sb], axis=1).max(axis=1)
    cond = (tenkan > kijun) & (c > cloud_top)
    if use_trend:
        cond &= (c > c.ewm(span=p["trend_filter"], adjust=False).mean()) & _golden_cross_ok(df, p)
    if use_volume:
        cond &= df["volume"] > df["volume"].rolling(p["volume_avg_period"]).mean() * p["volume_mult"]
    return fresh(cond)


def ms_sig(df, use_trend=True):
    p = config.STRATEGY_PARAMS["market_structure"]
    n, min_swing = p["pivot_lookback"], p["min_swing_pct"] / 100
    high, close = df["high"].values, df["close"].values
    length = len(df)
    is_ph = np.zeros(length, bool)
    for i in range(n, length - n):
        if high[i] == high[i - n:i + n + 1].max():
            is_ph[i] = True
    sig = np.zeros(length, bool)
    last_val = prev_val = None
    bull = False
    for i in range(length):
        if is_ph[i]:
            val = high[i]
            if prev_val is not None and abs(val - prev_val) / prev_val >= min_swing:
                bull = val > prev_val
            prev_val, last_val = last_val, val
        if last_val is not None and bull and close[i] > last_val and (i == 0 or close[i - 1] <= last_val):
            sig[i] = True
    s = pd.Series(sig, index=df.index)
    if use_trend:
        s &= (df["close"] > df["close"].ewm(span=p["trend_filter"], adjust=False).mean()) & _golden_cross_ok(df, p)
    return s


def combo_sig(df, use_volume=True, use_trend=True):
    a = apply_cooldown(ichimoku_sig(df, use_volume, use_trend), COOL)
    b = apply_cooldown(ms_sig(df, use_trend), COOL)
    return (a & b).values


VARIANTS = [
    ("BASELINE (Ichimoku 4H Bot jaisa)", {}),
    ("+ BTC>EMA50", {"regime": "BTC"}),
    ("+ ETH>EMA50", {"regime": "ETH"}),
    ("+ RSI 40-70", {"rsi": (40, 70)}),
    ("+ RSI < 75", {"rsi": (0, 75)}),
    ("+ ADX > 20", {"adx": 20}),
    ("+ ADX > 25", {"adx": 25}),
    ("- Volume shart hatai", {"volume": False}),
    ("- EMA trend shart hatai", {"trend": False}),
    ("- Volume + EMA trend dono hatai", {"volume": False, "trend": False}),
]


def main():
    from data_fetcher import get_exchange, get_coin_list
    lines = []

    def emit(x=""):
        print(x)
        lines.append(x)

    ex = get_exchange()
    coins = [c for c in get_coin_list(ex) if c.split("/")[0].upper() not in STABLES][:TOP_N_COINS]
    h4, fails = {}, 0
    for k, sym in enumerate(coins, 1):
        try:
            h = fetch_full(ex, sym, "4h", H4_LIMIT)
            if h is not None and len(h) >= 600:
                h4[sym] = norm(h)
        except Exception as e:
            fails += 1
            print(f"[{k}] {sym}: SKIP ({e})")
        if k % 25 == 0:
            print(f"[{k}/{len(coins)}] data...")
    daily = {s: norm(fetch_full(ex, s, "1d", D_LIMIT)) for s in ("BTC/USDT", "ETH/USDT")}

    emit("=" * 100)
    emit("4H ICHIMOKU + MS - FILTERS KA TEST (ek waqt mein ek tabdeeli)")
    emit("=" * 100)
    emit(f"fetch errors: {fails}/{len(coins)} | fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% har taraf + stop slip {STOP_SLIP*100:.2f}%")
    for l in data_report("4H", h4, H4_LIMIT, "4h"):
        emit(l)
    emit(f"Exit sab mein: CE {CE_P}/{CE_M} + TP {TP_R}R | 1% risk, max 10 positions")

    idx = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in h4.values()])))
    syms, O, H, L, C, V = build_panel(h4, idx)
    T, N = O.shape
    liquid = liquid_mask(C, V, hist_min=60 * BPD, window=30 * BPD, min_periods=20 * BPD)
    mom = pd.DataFrame(C).pct_change(60 * BPD, fill_method=None).values
    start = int(max(np.argmax(liquid.sum(1) >= 30), 60 * BPD))
    fold_bounds = pd.date_range(idx[start], idx[-1], periods=N_FOLDS + 1).values
    emit(f"Period: {idx[start].date()} -> {idx[-1].date()}  (folds: "
         + " | ".join(str(pd.Timestamp(b).date()) for b in fold_bounds) + ")")

    stop = np.full((T, N), np.nan)
    rsi = np.full((T, N), np.nan)
    adx = np.full((T, N), np.nan)
    sig_cache = {}
    for j, s in enumerate(syms):
        pos = idx.get_indexer(h4[s]["timestamp"])
        stop[pos, j] = chandelier(h4[s], CE_P, CE_M)
        rsi[pos, j] = compute_rsi(h4[s], 14).values
        adx[pos, j] = compute_adx(h4[s], 14).values

    def sig_for(use_volume, use_trend):
        key = (use_volume, use_trend)
        if key not in sig_cache:
            sg = np.zeros((T, N), bool)
            for j, s in enumerate(syms):
                sg[idx.get_indexer(h4[s]["timestamp"]), j] = combo_sig(h4[s], use_volume, use_trend)
            sig_cache[key] = sg
        return sig_cache[key]

    def regime(sym):
        d = daily[sym].copy()
        d["ok"] = d["close"] > ema(d["close"], 50)
        d["timestamp"] = (d["timestamp"] + pd.Timedelta(hours=20)).astype("datetime64[ns]")   # sirf BAND daily candle
        m = pd.merge_asof(pd.DataFrame({"timestamp": idx.values}), d[["timestamp", "ok"]].sort_values("timestamp"),
                          on="timestamp", direction="backward")
        return m["ok"].astype("boolean").fillna(False).to_numpy(dtype=bool)

    reg = {"BTC": regime("BTC/USDT"), "ETH": regime("ETH/USDT")}

    rows = []
    for vname, ch in VARIANTS:
        sig = sig_for(ch.get("volume", True), ch.get("trend", True)).copy()
        if "rsi" in ch:
            lo, hi = ch["rsi"]
            sig &= (rsi >= lo) & (rsi <= hi)
        if "adx" in ch:
            sig &= adx > ch["adx"]
        allow = np.ones((T, N), bool)
        if "regime" in ch:
            allow = np.repeat(reg[ch["regime"]][:, None], N, axis=1)
        tr, rtr = trades_and_random(O, H, L, C, sig, allow, stop, TP_R, liquid, start)
        s = per_trade_block(tr, rtr, fold_bounds, idx.values)
        eq, ntr = portfolio(O, H, L, C, sig, allow, stop, stop, TP_R, liquid, mom, start, MAX_HOLD)
        ps = eq_stats(eq[start:], idx[start:], PER_YEAR)
        extra = ""
        if not ch:
            rsh = []
            for sd in range(N_RANDOM):
                req, _ = portfolio(O, H, L, C, sig, allow, stop, stop, TP_R, liquid, mom, start, MAX_HOLD,
                                   rng=np.random.default_rng(sd))
                rsh.append(eq_stats(req[start:], idx[start:], PER_YEAR)["sharpe"])
            extra = f" | random portfolio Sharpe 95th={np.nanpercentile(rsh, 95):.2f}"
        ok = s is not None and trade_pass(s)
        rows.append((vname, s, ps))
        if s is None:
            emit(f"\n{vname}: koi trade nahi")
            continue
        folds = " / ".join(f"{fmt(p)}({c})" for p, c in zip(s["fp"], s["fn"]))
        yr = " ".join(f"{y}:{v*100:+.0f}%" for y, v in ps["yearly"].items())
        emit(f"\n[{'theek' if ok else 'FAIL '}] {vname}")
        emit(f"   Har trade: n={s['n']} win={s['win']:.1f}% PF={fmt(s['pf'])} p5={fmt(s['p5'])} "
             f"Top10-hata={fmt(s['t10'])} RandomPF={fmt(s['rpf'])} Exp={s['exp']:+.2f}% | Folds: {folds}")
        emit(f"   Portfolio: CAGR={ps['cagr']*100:+.1f}% MaxDD={ps['dd']*100:.1f}% Sharpe={fmt(ps['sharpe'])} trades={ntr}{extra}")
        emit(f"   Saal-war: {yr}")

    base_s, base_p = rows[0][1], rows[0][2]
    emit("\n" + "=" * 100)
    emit("KHULASA - baseline ke muqable mein (+ = behtar)")
    emit("=" * 100)
    emit(f"{'Variant':34s} | {'Trades':>6} | {'PF':>5} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | Folds behtar/barabar | Faisla")
    for vname, s, ps in rows:
        if s is None:
            emit(f"{vname:34s} | koi trade nahi")
            continue
        fb = sum(1 for a, b in zip(s["fp"], base_s["fp"]) if a is not None and b is not None and a >= b * 0.97)
        better = (ps["sharpe"] > base_p["sharpe"] and ps["dd"] >= base_p["dd"] and fb >= 3 and s["n"] >= 200)
        verdict = "BASELINE" if s is base_s else ("FAIDEMAND" if better else "faida nahi")
        emit(f"{vname:34s} | {s['n']:>6} | {fmt(s['pf']):>5} | {ps['cagr']*100:>+6.1f}% | {ps['dd']*100:>6.1f}% | "
             f"{fmt(ps['sharpe']):>6} | {fb}/4 | {verdict}")

    with open("ichimoku4h_filters_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[SAVE] ichimoku4h_filters_RESULTS.txt")


if __name__ == "__main__":
    main()
