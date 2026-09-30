"""
UNIFIED TEST - screener ki SAARI strategies + Daily Donchian, EK HI SAKHT TARAZU par
===================================================================================
Pehle screener ki 1h strategies halke usoolon par jaanchi gayi thin (sirf har trade
alag, 1 saal, random-control nahi, stop slippage nahi). Daily Donchian sakht usoolon
par. Is test mein SAB ko bilkul EK jaise usool par jaancha jata hai:

  1) Lookahead-free: signal band candle par, entry agli candle ke open par;
     daily ETH/RS filters sirf BAND daily candle se; confluence ka 4H/BTC bhi band candle se
  2) Fee 0.1% + slippage 0.05% har taraf, stop exit par extra 0.25% slippage,
     gap-fill (stop se neeche khule to open par), check-pehle-update-baad
  3) Point-in-time top-100 liquid coins (30 din ka dollar volume)
  4) Ek coin par ek waqt mein ek hi trade
  5) HAR TRADE (per-trade): 4 waqt ke folds, bootstrap p5, top-10 hata kar PF,
     RANDOM ENTRIES (usi exit/filter ke sath) ka PF
  6) PORTFOLIO: max 10 positions, har trade 1% risk, ek coin max 20%, leverage nahi;
     RANDOM portfolio (20 seeds) se moqabla
  7) Data-completeness + gap report

PASS = per-trade [n>=100, har fold PF>1 (>=10 trades), p5>1.2, p5>random PF]
       AUR portfolio [Sharpe > random 95th-pct, CAGR>0, MaxDD behtar -50% se]

Strategies (har ek ka PRODUCTION exit):
  1h Union AB Ichimoku+MS   exit CE 16/4.5 + TP 2R
  1h Union AB EMA+Breakout  exit CE 12/4.5 + TP 2R
  1h NEW AdvancedConfluence (CHoCH + Score>=7)  exit CE 16/3.0 + TP 2R
  1h CE Buy-Only            entry CE(11/4.5) cross, exit CE 16/3.0 (trail)
  1h Pullback-in-Uptrend    exit CE 16/4.5 (trail)
  1h Donchian Breakout      exit CE 16/4.5 (trail)
  Har 1h strategy 3 filter tiers par: No filter / RS only (Backup) / ETH+RS (Baseline)
  DAILY Donchian 20 (reference) exit CE 22/4x (trail), regime BTC>EMA50

TIMEFRAMES: har screener strategy 1H, 4H aur DAILY teeno par (bilkul wahi
parameters - bars ke hisab se). Daily/4H filters aur confluence ka BTC/4H
hamesha sirf BAND candle se (shift = 24h - timeframe).

Natija: unified_test_RESULTS.txt
"""

import numpy as np
import pandas as pd

import config
import scheduled_dashboard_scan as live
import confluence_engine as ce_mod
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import compute_chandelier_long_stop
from strategy_lab import (fetch_full, norm, ema, chandelier, data_report, pf_of, bootstrap_p5, fmt,
                          STABLES, FEE, SLIP, STOP_SLIP)
import choch_robustness as cr
from choch_robustness import daily_filter_arrays, shift_daily, _resample_to_4h_closed, _orig_resample_to_4h

TOP_N_COINS = 150
H1_LIMIT = 17520            # 2 saal 1h
H4_LIMIT = 6570             # 3 saal 4H
D_FILTER_LIMIT = 900        # 1h filters ke liye daily
D_REF_LIMIT = 2400          # Daily Donchian reference (~6.5 saal)
UNIVERSE = 100
N_FOLDS = 4
MAX_POS, RISK, MAX_POS_PCT = 10, 0.01, 0.20
N_RANDOM = 20
COOL = config.SIGNAL_COOLDOWN_BARS
RR = live.RR_MULTIPLE

STRATS_1H = [
    # name, init-stop CE, trail CE, tp R (None = sirf trail)
    ("Union AB Ichimoku+MS", (16, 4.5), (16, 4.5), RR),
    ("Union AB EMA+Breakout", (12, 4.5), (12, 4.5), RR),
    ("NEW AdvancedConfluence (CHoCH)", (16, 3.0), (16, 3.0), RR),
    ("CE Buy-Only", (11, 4.5), (16, 3.0), None),
    ("Pullback-in-Uptrend", (16, 4.5), (16, 4.5), None),
    ("Donchian Breakout 1h", (16, 4.5), (16, 4.5), None),
]
TIERS_1H = ["No filter", "RS only", "ETH+RS"]
PROD_TIER = {"Union AB Ichimoku+MS": "ETH+RS", "Union AB EMA+Breakout": "ETH+RS",
             "NEW AdvancedConfluence (CHoCH)": "No filter", "CE Buy-Only": "No filter",
             "Pullback-in-Uptrend": "ETH+RS", "Donchian Breakout 1h": "ETH+RS"}


def signals_1h(name, df, btc_daily):
    if name == "Union AB Ichimoku+MS":
        a = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), COOL)
        b = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), COOL)
        return (a & b).values
    if name == "Union AB EMA+Breakout":
        a = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), COOL)
        b = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), COOL)
        return (a & b).values
    if name == "NEW AdvancedConfluence (CHoCH)":
        ce_mod.resample_to_4h = _resample_to_4h_closed
        try:
            r = ce_mod.compute_confluence(df, shift_daily(btc_daily), ce_mod.DEFAULT_PARAMS, usdt_d_weak=None)
        finally:
            ce_mod.resample_to_4h = _orig_resample_to_4h
        s = r["choch"].fillna(False).astype(bool) & (r["score"] >= ce_mod.DEFAULT_PARAMS["score_threshold"])
        return apply_cooldown(pd.Series(s.values, index=df.index), COOL).values
    if name == "CE Buy-Only":
        st = compute_chandelier_long_stop(df, 11, 4.5)
        c = df["close"]
        cross = ((c > st) & (c.shift(1) <= st.shift(1))).fillna(False)
        return apply_cooldown(cross, COOL).values
    if name == "Pullback-in-Uptrend":
        return apply_cooldown(live.pullback_uptrend_entry(df).fillna(False).astype(bool), COOL).values
    if name == "Donchian Breakout 1h":
        return apply_cooldown(live.donchian_channel_breakout(df).fillna(False).astype(bool), COOL).values
    raise ValueError(name)


# ---------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------
def sim_trades(op, hi, lo, cl, entries, init_stop, trail_stop, tp_r, max_hold):
    n = len(op)
    out, busy = [], -1
    for i in entries:
        if i <= busy or i + 1 >= n or np.isnan(op[i + 1]):
            continue
        eb = i + 1
        entry = op[eb] * (1 + SLIP)
        st0 = init_stop[i]
        if np.isnan(st0) or st0 >= entry:
            continue
        trail = float(st0)
        tp = entry + (entry - st0) * tp_r if tp_r else None
        px = xb = None
        last = min(eb + max_hold, n)
        for j in range(eb, last):
            if np.isnan(lo[j]):
                continue
            if lo[j] <= trail:
                px, xb = min(trail * (1 - STOP_SLIP), op[j]), j
                break
            if tp is not None and hi[j] >= tp:
                px, xb = max(tp, op[j]), j
                break
            s = trail_stop[j]
            if not np.isnan(s) and s > trail:
                trail = float(s)
        if px is None:
            xb = last - 1
            while xb > eb and np.isnan(cl[xb]):
                xb -= 1
            px = cl[xb]
        px *= (1 - SLIP)
        out.append((eb, ((px - entry) / entry - 2 * FEE) * 100))
        busy = xb
    return out


def portfolio(O, H, L, C, sig, allow, init_stop, trail_stop, tp_r, liquid, mom, start, max_hold, rng=None):
    T, N = O.shape
    cash, pos, pending = 1.0, {}, []
    eq = np.full(T, np.nan)
    last_px = np.full(N, np.nan)
    ntr = 0
    for t in range(start, T):
        if pending:
            eo = cash + sum(p[0] * (O[t, j] if not np.isnan(O[t, j]) else last_px[j]) for j, p in pos.items())
            for j in pending:
                if len(pos) >= MAX_POS:
                    break
                if j in pos or np.isnan(O[t, j]):
                    continue
                entry = O[t, j] * (1 + SLIP)
                st0 = init_stop[t - 1, j]
                if np.isnan(st0) or st0 >= entry:
                    continue
                val = min(eo * RISK * entry / (entry - st0), MAX_POS_PCT * eo, cash / (1 + FEE))
                if val < 0.005 * eo:
                    continue
                cash -= val * (1 + FEE)
                tp = entry + (entry - st0) * tp_r if tp_r else None
                pos[j] = [val / entry, float(st0), t, tp]
                ntr += 1
            pending = []
        for j in list(pos):
            q, trail, t0, tp = pos[j]
            if np.isnan(L[t, j]):
                continue
            px = None
            if L[t, j] <= trail:
                px = min(trail * (1 - STOP_SLIP), O[t, j])
            elif tp is not None and H[t, j] >= tp:
                px = max(tp, O[t, j])
            elif t - t0 >= max_hold:
                px = C[t, j]
            if px is not None:
                cash += q * px * (1 - SLIP) * (1 - FEE)
                del pos[j]
                continue
            s = trail_stop[t, j]
            if not np.isnan(s) and s > trail:
                pos[j][1] = float(s)
        ok = ~np.isnan(C[t])
        last_px[ok] = C[t, ok]
        eq[t] = cash + sum(p[0] * last_px[j] for j, p in pos.items())
        if len(pos) < MAX_POS:
            cand = [j for j in np.where(sig[t] & allow[t] & liquid[t])[0] if j not in pos]
            if rng is not None:
                pool = [j for j in np.where(allow[t] & liquid[t] & ok)[0] if j not in pos]
                k = min(len(cand), len(pool))
                cand = list(rng.choice(pool, size=k, replace=False)) if k else []
            else:
                cand.sort(key=lambda j: -(mom[t, j] if not np.isnan(mom[t, j]) else -9))
            pending = cand
    return eq, ntr


def eq_stats(eq, idx, per_year):
    s = pd.Series(eq, index=idx).dropna()
    r = s.pct_change().dropna()
    yrs = (s.index[-1] - s.index[0]).total_seconds() / (365.25 * 86400)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1 if s.iloc[-1] > 0 and yrs > 0 else -1
    return {"cagr": cagr, "dd": (s / s.cummax() - 1).min(),
            "sharpe": r.mean() / r.std() * np.sqrt(per_year) if r.std() > 0 else np.nan,
            "yearly": s.groupby(s.index.year).agg(lambda x: x.iloc[-1] / x.iloc[0] - 1)}


def per_trade_block(trades, rtrades, fold_bounds, ts):
    if not trades:
        return None
    rets = np.array([r for _, r in trades])
    folds = np.clip(np.searchsorted(fold_bounds, ts[[eb for eb, _ in trades]], side="right") - 1, 0, N_FOLDS - 1)
    fp = [pf_of(rets[folds == f]) for f in range(N_FOLDS)]
    fn = [int((folds == f).sum()) for f in range(N_FOLDS)]
    srt = np.sort(rets)[::-1]
    return {"n": len(rets), "win": (rets > 0).mean() * 100, "pf": pf_of(rets), "p5": bootstrap_p5(rets),
            "t10": pf_of(srt[10:]) if len(srt) > 10 else None, "exp": rets.mean(),
            "rpf": pf_of([r for _, r in rtrades]) if rtrades else None, "fp": fp, "fn": fn}


def trade_pass(s):
    if s is None or s["pf"] is None or s["p5"] is None or s["n"] < 100:
        return False
    if any(p is None or p <= 1 or n < 10 for p, n in zip(s["fp"], s["fn"])):
        return False
    return s["p5"] > 1.2 and (s["rpf"] is None or s["p5"] > s["rpf"])


# ---------------------------------------------------------------------
def evaluate(emit, title, idx, O, H, L, C, sig, allow, init_stop, trail_stop, tp_r, liquid, mom, start,
             max_hold, per_year, is_prod):
    T, N = O.shape
    fold_bounds = pd.date_range(idx[start], idx[-1], periods=N_FOLDS + 1).values
    ts = idx.values
    rng = np.random.default_rng(7)
    tr, rtr = [], []
    for j in range(N):
        ok = sig[:, j] & allow[:, j] & liquid[:, j]
        ok[:start] = False
        ent = np.where(ok)[0]
        tr += sim_trades(O[:, j], H[:, j], L[:, j], C[:, j], ent, init_stop[:, j], trail_stop[:, j], tp_r, max_hold)
        pool = np.where(allow[:, j] & liquid[:, j] & ~np.isnan(O[:, j]))[0]
        pool = pool[pool >= start]
        k = min(len(ent), len(pool))
        if k:
            rent = np.sort(rng.choice(pool, size=k, replace=False))
            rtr += sim_trades(O[:, j], H[:, j], L[:, j], C[:, j], rent, init_stop[:, j], trail_stop[:, j], tp_r, max_hold)
    s = per_trade_block(tr, rtr, fold_bounds, ts)

    eq, ntr = portfolio(O, H, L, C, sig, allow, init_stop, trail_stop, tp_r, liquid, mom, start, max_hold)
    ps = eq_stats(eq[start:], idx[start:], per_year)
    rsh = []
    for sd in range(N_RANDOM):
        req, _ = portfolio(O, H, L, C, sig, allow, init_stop, trail_stop, tp_r, liquid, mom, start, max_hold,
                           rng=np.random.default_rng(sd))
        rsh.append(eq_stats(req[start:], idx[start:], per_year)["sharpe"])
    r95 = float(np.nanpercentile(rsh, 95))
    port_ok = ps["sharpe"] > r95 and ps["cagr"] > 0 and ps["dd"] > -0.5
    tp_ok = trade_pass(s)
    verdict = "PASS" if (tp_ok and port_ok) else "FAIL"

    emit(f"\n[{verdict}] {title}" + ("   <-- screener par yehi chal raha hai" if is_prod else ""))
    if s is None:
        emit("   koi trade nahi")
        return verdict, None
    folds = " / ".join(f"{fmt(p)}({n})" for p, n in zip(s["fp"], s["fn"]))
    emit(f"   Har trade: n={s['n']} win={s['win']:.1f}% PF={fmt(s['pf'])} p5={fmt(s['p5'])} "
         f"Top10-hata={fmt(s['t10'])} RandomPF={fmt(s['rpf'])} Exp={s['exp']:+.2f}% | Folds: {folds} "
         f"-> {'theek' if tp_ok else 'FAIL'}")
    yr = " ".join(f"{y}:{v*100:+.0f}%" for y, v in ps["yearly"].items())
    emit(f"   Portfolio (10 slots, 1% risk): CAGR={ps['cagr']*100:+.1f}% MaxDD={ps['dd']*100:.1f}% "
         f"Sharpe={fmt(ps['sharpe'])} (random 95th={r95:.2f}) trades={ntr} -> {'theek' if port_ok else 'FAIL'}")
    emit(f"   Saal-war: {yr}")
    return verdict, (s, ps, r95)


def build_panel(frames, idx):
    syms = list(frames)
    get = lambda k: pd.DataFrame({s: frames[s].set_index("timestamp")[k] for s in syms}).reindex(idx).values
    return syms, get("open"), get("high"), get("low"), get("close"), get("volume")


def liquid_mask(C, V, hist_min, window, min_periods):
    dv = pd.DataFrame(C * V).rolling(window, min_periods=min_periods).mean().values
    hist = np.cumsum(~np.isnan(C), axis=0)
    T, N = C.shape
    liq = np.zeros((T, N), bool)
    for t in range(T):
        ok = np.where((hist[t] >= hist_min) & ~np.isnan(dv[t]))[0]
        if len(ok):
            liq[t, ok[np.argsort(-dv[t, ok])][:UNIVERSE]] = True
    return liq


TF_CFG = {
    #       bars/din, max_hold, per_year, daily-filter shift, 4H-confluence shift
    "1h": (24, 500, 24 * 365, pd.Timedelta(hours=23), pd.Timedelta(hours=3)),
    "4h": (6, 500, 6 * 365, pd.Timedelta(hours=20), pd.Timedelta(0)),
    "1d": (1, 365, 365, pd.Timedelta(0), pd.Timedelta(0)),
}


def run_tf(tf, frames, dd, emit, summary):
    bpd, max_hold, per_year, dshift, h4shift = TF_CFG[tf]
    cr.DAILY_CLOSE_SHIFT, cr.H4_CLOSE_SHIFT = dshift, h4shift      # band-candle shifts is timeframe ke liye
    btc_d, eth_d = dd["BTC/USDT"], dd["ETH/USDT"]
    idx = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in frames.values()])))
    syms, O, H, L, C, V = build_panel(frames, idx)
    T, N = O.shape
    liquid = liquid_mask(C, V, hist_min=60 * bpd, window=30 * bpd, min_periods=20 * bpd)
    mom = pd.DataFrame(C).pct_change(60 * bpd, fill_method=None).values
    start = int(max(np.argmax(liquid.sum(1) >= 30), 60 * bpd))

    tiers = {t: np.zeros((T, N), bool) for t in TIERS_1H}
    tiers["No filter"][:] = True
    sigs = {n: np.zeros((T, N), bool) for n, *_ in STRATS_1H}
    stops = {}
    for j, s in enumerate(syms):
        df = frames[s]
        pos = idx.get_indexer(df["timestamp"])
        if s in dd:
            try:
                eth_ok, rs_ok = daily_filter_arrays(df["timestamp"], eth_d, dd[s], btc_d, shift=True)
                tiers["RS only"][pos, j] = rs_ok
                tiers["ETH+RS"][pos, j] = rs_ok & eth_ok
            except Exception as e:
                print(f"{s}: filter fail ({e})")
        for name, ice, tce, _ in STRATS_1H:
            try:
                sigs[name][pos, j] = signals_1h(name, df, btc_d)
            except Exception as e:
                print(f"{tf} {s} {name}: signal fail ({e})")
            for ce in (ice, tce):
                if ce not in stops:
                    stops[ce] = np.full((T, N), np.nan)
                stops[ce][pos, j] = chandelier(df, *ce)

    emit(f"\n\n{'#'*100}\n{tf.upper()} - screener strategies - period {idx[start].date()} -> {idx[-1].date()}\n{'#'*100}")
    for name, ice, tce, tp_r in STRATS_1H:
        for tier in TIERS_1H:
            live_now = tf == "1h" and PROD_TIER[name] == tier
            v, _ = evaluate(emit, f"[{tf}] {name} | {tier}", idx, O, H, L, C, sigs[name], tiers[tier],
                            stops[ice], stops[tce], tp_r, liquid, mom, start, max_hold, per_year, live_now)
            summary.append((f"[{tf}] {name} | {tier}", v, live_now))

    if tf == "1d":
        bi = syms.index("BTC/USDT")
        bt = pd.Series(C[:, bi], index=idx).ffill()
        reg = np.repeat((bt > ema(bt, 50)).values[:, None], N, axis=1)
        dsig = np.zeros((T, N), bool)
        dstop = np.full((T, N), np.nan)
        for j, s in enumerate(syms):
            d = frames[s]
            pos = idx.get_indexer(d["timestamp"])
            brk = (d["close"] > d["high"].shift(1).rolling(20).max()).fillna(False)
            dsig[pos, j] = (brk & ~brk.shift(1, fill_value=False)).values
            dstop[pos, j] = chandelier(d, 22, 4.0)
        emit(f"\n--- Reference: Daily Donchian 20 (Donchian Daily Bot wali) ---")
        v, _ = evaluate(emit, "[1d] Daily Donchian 20 | BTC>EMA50", idx, O, H, L, C, dsig, reg, dstop, dstop, None,
                        liquid, mom, start, max_hold, per_year, False)
        summary.append(("[1d] Daily Donchian 20 | BTC>EMA50  (Donchian Daily Bot)", v, False))


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

    data = {"1h": {}, "4h": {}, "1d": {}}
    fails = 0
    for k, sym in enumerate(coins, 1):
        try:
            d = fetch_full(ex, sym, "1d", D_REF_LIMIT)
            if d is not None and len(d) >= 150:
                data["1d"][sym] = norm(d)
            for tf, lim, mn in (("4h", H4_LIMIT, 600), ("1h", H1_LIMIT, 1500)):
                h = fetch_full(ex, sym, tf, lim)
                if h is not None and len(h) >= mn:
                    data[tf][sym] = norm(h)
        except Exception as e:
            fails += 1
            print(f"[{k}] {sym}: SKIP ({e})")
        if k % 25 == 0:
            print(f"[{k}/{len(coins)}] data...")
    dd = data["1d"]

    emit("=" * 100)
    emit("UNIFIED TEST - SAB STRATEGIES, 1H + 4H + DAILY, EK HI SAKHT TARAZU PAR")
    emit("=" * 100)
    emit(f"fetch errors: {fails}/{len(coins)} | fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% har taraf + stop slip {STOP_SLIP*100:.2f}%")
    for l in (data_report("1H", data["1h"], H1_LIMIT, "1h") + data_report("4H", data["4h"], H4_LIMIT, "4h")
              + data_report("DAILY", dd, D_REF_LIMIT, "1d")):
        emit(l)
    emit("PASS = har trade [n>=100, har fold PF>1, p5>1.2, p5>random] AUR portfolio [Sharpe>random 95th, CAGR>0, MaxDD>-50%]")
    emit("NOTE: parameters (EMA, Ichimoku, cooldown, CE) bars mein hain - 4H/daily par wahi numbers istemal hue.")
    emit("NOTE: coin list aaj ki hai (survivorship bias) - sab par barabar asar.")

    summary = []
    for tf in ("1h", "4h", "1d"):
        if data[tf]:
            run_tf(tf, data[tf], dd, emit, summary)

    emit(f"\n\n{'='*100}\nKHULASA\n{'='*100}")
    for name, v, prod in summary:
        emit(f"{v:4s}  {name}" + ("   (screener par live)" if prod else ""))

    with open("unified_test_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[SAVE] unified_test_RESULTS.txt")


if __name__ == "__main__":
    main()
