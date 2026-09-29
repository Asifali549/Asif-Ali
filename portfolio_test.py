"""
PORTFOLIO TEST - asal trading jaisi jaanch (daily, ~6.5 saal: 2021 bull + 2022 crash)
====================================================================================
Strategy Lab mein 2 cheezein PASS huin. Wahan har trade alag se gini gayi thi.
Asal zindagi mein paisa mehdood hai aur ek din mein kai signals aa sakte hain,
is liye yahan POORA PORTFOLIO chalaya jata hai:

STRATEGY A - Daily Trend Following (portfolio)
  Entry : Donchian N breakout (close > pichle N din ka high) ya EMA 10/30 cross,
          sirf jab BTC close > BTC EMA50 (warna nayi entry nahi)
  Exit  : Chandelier trailing stop (22 din, 4x ATR) - check pehle, update baad mein
  Size  : har trade par equity ka R% risk (entry se stop tak), ek position
          max 20% equity, max M positions ek sath, leverage nahi (cash se ziada nahi)
  Jab slots se ziada signals hon: pichle 60 din ke sab se mazboot coin pehle
  Grid  : entry (Donchian 20 / Donchian 55 / EMA 10/30) x M (5/10) x R (1%/2%)
  Control: wahi portfolio lekin RANDOM entries (usi din utne hi signals, 20 seeds)

STRATEGY B - Momentum Rotation (har hafte top-K, 60 din, vol-adjusted)
  K=10 regime none (PASS wala) + K=5 + regime BTC>EMA50 wale variants

COMBINED - 50% A (Donchian 20, M=10, R=1%) + 50% B (L60 K10) - rozana barabar

Imaandari: signal band candle par, entry agle din ke open par, fee 0.1% +
slippage 0.05% har taraf, stop par extra 0.25% slippage, point-in-time top-100
liquidity, data-gap report. Survivorship bias abhi bhi hai (aaj ki coin list).

Natija: portfolio_test_RESULTS.txt
"""

import numpy as np
import pandas as pd

from strategy_lab import (fetch_full, norm, ema, chandelier, data_report, run_rotation,
                          benchmark_btc, STABLES, FEE, SLIP, STOP_SLIP)

TOP_N_COINS = 150
DAILY_LIMIT = 2400           # ~6.5 saal
UNIVERSE = 100
MAX_POS_PCT = 0.20
MAX_HOLD = 365
N_RANDOM = 20


# ---------------------------------------------------------------------
def panels(daily):
    idx = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in daily.values()])))
    syms = list(daily.keys())
    P = {k: pd.DataFrame({s: daily[s].set_index("timestamp")[k] for s in syms}).reindex(idx)
         for k in ("open", "high", "low", "close", "volume")}
    return idx, syms, P


def stats(eq, idx):
    eq = pd.Series(eq, index=idx).dropna()
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
    dd = eq / eq.cummax() - 1
    # sab se lamba drawdown (din)
    under = (dd < 0).astype(int).values
    longest, cur = 0, 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    monthly = eq.resample("ME").last().pct_change().dropna()
    yearly = eq.groupby(eq.index.year).agg(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    return {"total": eq.iloc[-1] / eq.iloc[0] - 1, "cagr": cagr, "dd": dd.min(),
            "sharpe": r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else np.nan,
            "worst_m": monthly.min() if len(monthly) else np.nan,
            "pos_m": (monthly > 0).mean() * 100 if len(monthly) else np.nan,
            "longest_dd": longest, "yearly": yearly}


def line(name, s, extra=""):
    yr = " ".join(f"{y}:{v*100:+.0f}%" for y, v in s["yearly"].items())
    return (f"{name:46s} CAGR={s['cagr']*100:+6.1f}% MaxDD={s['dd']*100:6.1f}% Sharpe={s['sharpe']:.2f} "
            f"BadtareenMahina={s['worst_m']*100:+.0f}% MusbatMahine={s['pos_m']:.0f}% "
            f"LambaDD={s['longest_dd']}d {extra}\n{'':46s} saal-war: {yr}")


# ---------------------------------------------------------------------
def portfolio_trend(idx, P, sig, stop, regime, liquid, mom60, start_i, M, R, rng=None):
    """Event-driven daily portfolio. sig/stop/liquid: T x N arrays; regime: T."""
    O, H, Lo, C = (P[k].values for k in ("open", "high", "low", "close"))
    T, N = O.shape
    cash = 1.0
    pos = {}          # j -> [qty, trail, entry_t]
    eq = np.full(T, np.nan)
    last_px = np.full(N, np.nan)
    pending = []
    n_trades, rets, expo = 0, [], []
    for t in range(start_i, T):
        # ---- open t: nayi entries ----
        if pending:
            eq_open = cash + sum(q * (O[t, j] if not np.isnan(O[t, j]) else last_px[j]) for j, (q, _, _) in pos.items())
            for j in pending:
                if len(pos) >= M:
                    break
                if j in pos or np.isnan(O[t, j]):
                    continue
                entry = O[t, j] * (1 + SLIP)
                st = stop[t - 1, j]
                if np.isnan(st) or st >= entry:
                    continue
                val = eq_open * R * entry / (entry - st)
                val = min(val, MAX_POS_PCT * eq_open, cash / (1 + FEE))
                if val < 0.005 * eq_open:
                    continue
                cash -= val * (1 + FEE)
                pos[j] = [val / entry, float(st), t]
                n_trades += 1
            pending = []
        # ---- din t: stop check, phir trail update ----
        for j in list(pos):
            q, trail, t0 = pos[j]
            if np.isnan(Lo[t, j]):
                continue
            exit_px = None
            if Lo[t, j] <= trail:
                exit_px = min(trail * (1 - STOP_SLIP), O[t, j])
            elif t - t0 >= MAX_HOLD:
                exit_px = C[t, j]
            if exit_px is not None:
                proceeds = q * exit_px * (1 - SLIP)
                cash += proceeds * (1 - FEE)
                rets.append(proceeds * (1 - FEE))
                del pos[j]
                continue
            if not np.isnan(stop[t, j]) and stop[t, j] > trail:
                pos[j][1] = float(stop[t, j])
        valid = ~np.isnan(C[t])
        last_px[valid] = C[t, valid]
        inv = sum(q * last_px[j] for j, (q, _, _) in pos.items())
        eq[t] = cash + inv
        expo.append(inv / eq[t] if eq[t] > 0 else 0)
        # ---- close t: signals ----
        if regime[t] and len(pos) < M:
            cand = np.where(sig[t] & liquid[t])[0]
            cand = [j for j in cand if j not in pos]
            if rng is not None:                               # random control
                pool = np.where(liquid[t] & valid)[0]
                pool = [j for j in pool if j not in pos]
                k = min(len(cand), len(pool))
                cand = list(rng.choice(pool, size=k, replace=False)) if k else []
            else:
                cand.sort(key=lambda j: -(mom60[t, j] if not np.isnan(mom60[t, j]) else -9))
            pending = cand
    return eq, n_trades, float(np.mean(expo)) if expo else 0.0


# ---------------------------------------------------------------------
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
    daily, fails = {}, 0
    for k, sym in enumerate(coins, 1):
        try:
            d = fetch_full(ex, sym, "1d", DAILY_LIMIT)
            if d is not None and len(d) >= 150:
                daily[sym] = norm(d)
        except Exception as e:
            fails += 1
            print(f"[{k}] {sym}: SKIP ({e})")
    if "BTC/USDT" not in daily:
        raise SystemExit("BTC data nahi mila")
    btc_d = daily["BTC/USDT"]

    emit("=" * 100)
    emit("PORTFOLIO TEST - DAILY (asal trading jaisa: mehdood paisa, max positions, risk sizing)")
    emit("=" * 100)
    emit(f"fetch errors: {fails}/{len(coins)} | fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% har taraf + stop slip {STOP_SLIP*100:.2f}%")
    for l in data_report("DAILY", daily, DAILY_LIMIT, "1d"):
        emit(l)
    emit("NOTE: coin list aaj ki hai (survivorship bias) - asal natija isse kamzor hoga, khaas kar rotation ka.")

    idx, syms, P = panels(daily)
    T, N = len(idx), len(syms)
    C = P["close"]
    dvol = (C * P["volume"]).rolling(30, min_periods=20).mean().values
    hist = C.notna().cumsum().values
    liquid = np.zeros((T, N), bool)
    for t in range(T):
        ok = np.where((hist[t] >= 60) & ~np.isnan(dvol[t]))[0]
        if len(ok):
            liquid[t, ok[np.argsort(-dvol[t, ok])][:UNIVERSE]] = True
    mom60 = (C / C.shift(60) - 1).values
    btc = btc_d.set_index("timestamp")["close"].reindex(idx).ffill()
    regime = (btc > ema(btc, 50)).values

    # per-coin signals + stops (har coin ki apni series par, phir panel mein)
    def per_coin(fn):
        out = np.zeros((T, N), bool) if fn != "stop" else np.full((T, N), np.nan)
        for j, s in enumerate(syms):
            d = daily[s]
            pos_i = idx.get_indexer(d["timestamp"])
            c, h = d["close"], d["high"]
            if fn == "stop":
                out[pos_i, j] = chandelier(d, 22, 4.0)
                continue
            if fn.startswith("don"):
                n = int(fn[3:])
                cond = c > h.shift(1).rolling(n).max()
            else:
                cond = ema(c, 10) > ema(c, 30)
            cond = cond.fillna(False).astype(bool)
            out[pos_i, j] = (cond & ~cond.shift(1, fill_value=False)).values
        return out

    stop = per_coin("stop")
    sigs = {"Donchian 20": per_coin("don20"), "Donchian 55": per_coin("don55"), "EMA 10/30": per_coin("ema")}

    n_liq = liquid.sum(1)
    start_i = int(max(np.argmax(n_liq >= 30), 60))
    sub = idx[start_i:]
    emit(f"\nPeriod: {sub[0].date()} -> {sub[-1].date()}  ({len(sub)} din)")

    emit("\n----- BENCHMARKS -----")
    b1 = benchmark_btc(idx, btc_d, False)[start_i:]
    b2 = benchmark_btc(idx, btc_d, True)[start_i:]
    emit(line("BTC buy&hold", stats(b1, sub)))
    emit(line("BTC + EMA50 regime", stats(b2, sub)))

    emit("\n\n" + "#" * 100)
    emit("STRATEGY A - DAILY TREND FOLLOWING PORTFOLIO (exit CE 22/4x, entry sirf BTC>EMA50)")
    emit("#" * 100)
    eqA = {}
    for name, sg in sigs.items():
        for M in (5, 10):
            for R in (0.01, 0.02):
                eq, nt, ex_ = portfolio_trend(idx, P, sg, stop, regime, liquid, mom60, start_i, M, R)
                s = stats(eq[start_i:], sub)
                eqA[(name, M, R)] = eq[start_i:]
                emit(line(f"{name} M={M} risk={R*100:.0f}%", s, f"trades={nt} avg-invested={ex_*100:.0f}%"))

    emit("\n--- RANDOM CONTROL (Donchian 20, M=10, risk=1%, random coins usi din utne hi signals, 20 seeds) ---")
    rs = []
    for sd in range(N_RANDOM):
        eq, _, _ = portfolio_trend(idx, P, sigs["Donchian 20"], stop, regime, liquid, mom60, start_i, 10, 0.01,
                                   rng=np.random.default_rng(sd))
        rs.append(stats(eq[start_i:], sub))
    sh = np.array([r["sharpe"] for r in rs])
    cg = np.array([r["cagr"] for r in rs])
    emit(f"Random: Sharpe median={np.nanmedian(sh):.2f} 95th-pct={np.nanpercentile(sh, 95):.2f} | "
         f"CAGR median={np.nanmedian(cg)*100:+.1f}% 95th-pct={np.nanpercentile(cg, 95)*100:+.1f}%")
    real = stats(eqA[("Donchian 20", 10, 0.01)], sub)
    emit(f"Asal (Donchian 20 M=10 1%): Sharpe={real['sharpe']:.2f} CAGR={real['cagr']*100:+.1f}%  -> "
         + ("RANDOM SE BEHTAR (95th-pct se ooper)" if real["sharpe"] > np.nanpercentile(sh, 95) else "random se saaf behtar NAHI"))

    emit("\n\n" + "#" * 100)
    emit("STRATEGY B - MOMENTUM ROTATION (har hafte, 60 din, vol-adjusted)")
    emit("#" * 100)
    btc_close = btc_d.set_index("timestamp")["close"]
    eqB = {}
    for K in (10, 5):
        for rg in ("none", "BTC>EMA50"):
            eq = run_rotation(idx, P["open"], P["close"], P["volume"], btc_close, 60, K, "voladj", rg,
                              universe=UNIVERSE, start_i=start_i)[start_i:]
            eqB[(K, rg)] = eq
            emit(line(f"Rotation L=60 K={K} regime={rg}", stats(eq, sub)))

    rsh = []
    for sd in range(N_RANDOM):
        eq = run_rotation(idx, P["open"], P["close"], P["volume"], btc_close, 60, 10, "random", "none",
                          universe=UNIVERSE, start_i=start_i, seed=sd)[start_i:]
        rsh.append(stats(eq, sub)["sharpe"])
    rsh = np.array(rsh, float)
    realB = stats(eqB[(10, "none")], sub)["sharpe"]
    emit(f"\n--- RANDOM CONTROL (har hafte 10 random liquid coins, {N_RANDOM} seeds): Sharpe median={np.nanmedian(rsh):.2f} "
         f"95th-pct={np.nanpercentile(rsh, 95):.2f} | Asal K=10 Sharpe={realB:.2f} -> "
         + ("RANDOM SE BEHTAR" if realB > np.nanpercentile(rsh, 95) else "random se saaf behtar NAHI"))

    emit("\n\n" + "#" * 100)
    emit("COMBINED - 50% Strategy A (Donchian 20, M=10, 1%) + 50% Strategy B (K=10, regime none)")
    emit("#" * 100)
    a = pd.Series(eqA[("Donchian 20", 10, 0.01)], index=sub).pct_change().fillna(0)
    b = pd.Series(eqB[(10, "none")], index=sub).pct_change().fillna(0)
    comb = (1 + 0.5 * a + 0.5 * b).cumprod().values
    emit(line("Combined 50/50", stats(comb, sub)))
    emit(f"A aur B ke rozana returns ka correlation: {a.corr(b):.2f} (kam = behtar diversification)")

    with open("portfolio_test_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[SAVE] portfolio_test_RESULTS.txt")


if __name__ == "__main__":
    main()
