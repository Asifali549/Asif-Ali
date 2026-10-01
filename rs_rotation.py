"""
WEEKLY RELATIVE-STRENGTH ROTATION (2026-10-01)
==============================================
Khayal: koi entry signal nahi - har hafte (itwar ka BAND din) tamam eligible coins ko pichle L din ke return se
tarteeb do, sab se aage wale K coins rakho. Agle din (peer) ke OPEN par: jo top-K se nikal gaya wo becho, naya
aaya wo khareedo (bechne se aaya cash naye coins mein barabar). Pehle se rakhe coins ko dobara barabar nahi kiya jata.
Eligible (point-in-time): kam az kam 90 din history, us din data maujood, 30-din ausat $volume mein top-100.
Kharcha: fee 0.1% + slippage 0.05% har taraf (2x stress bhi). Koi stop nahi (khalis rotation).
Grid (plateau): L = 14/30/60/90 din x K = 3/5/10.  EK filter alag se: BTC close > EMA50 warna sab cash.
Random baseline: wahi mechanics aur WAHI turnover (rakha coin keep_p imkaan se rehta hai, baqi RANDOM eligible), 100 seeds -> p50/p95.
Benchmarks: sab eligible coins barabar (EW), BTC buy & hold.
Pass: Sharpe > random p95, CAGR > EW benchmark, OOS (2025+) return > random OOS median, 2x kharche par CAGR > 0.
Natija: rs_rotation_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES, FEE, SLIP
from portfolio_lab import ema, to_daily, stats

OUT = "rs_rotation_RESULTS.txt"
TOP_N, H4_BARS = 150, 2400 * 6
LOOKS, KS = (14, 30, 60, 90), (3, 5, 10)
N_RND = 100
MIN_HIST, LIQ_TOP = 90, 100
START = pd.Timestamp("2020-10-01")
OOS_START = pd.Timestamp("2025-01-01")
YEARS = list(range(2021, 2027))


def simulate(O, C, Cf, last_idx, reb, pick, cost):
    """pick(t) -> target column indices (decided on BAND din t, chalaya t+1 open par)."""
    T, N = C.shape
    units = np.zeros(N)
    entry = np.full(N, np.nan)
    ent_t = np.zeros(N, int)
    cash = 1.0
    eq = np.empty(T)
    trades = []          # (ret, hold_days)
    pending = None
    for t in range(T):
        if pending is not None:
            held = set(np.nonzero(units)[0])
            for j in held - pending:
                px = O[t, j] if not np.isnan(O[t, j]) else Cf[t, j]
                cash += units[j] * px * (1 - cost)
                trades.append((px * (1 - cost) / entry[j] - 1, t - ent_t[j]))
                units[j] = 0.0
            new = [j for j in pending - held if not np.isnan(O[t, j])]
            if new and cash > 0:
                amt = cash / len(new)
                for j in new:
                    units[j] = amt / (O[t, j] * (1 + cost))
                    entry[j] = O[t, j] * (1 + cost)
                    ent_t[j] = t
                cash = 0.0
            pending = None
        for j in np.nonzero(units)[0]:                  # coin ka data khatam -> aakhri close par becho
            if t > last_idx[j]:
                cash += units[j] * Cf[t, j] * (1 - cost)
                trades.append((Cf[t, j] * (1 - cost) / entry[j] - 1, t - ent_t[j]))
                units[j] = 0.0
        eq[t] = cash + np.nansum(units * Cf[t])
        if reb[t] and t + 1 < T:
            pending = set(pick(t, set(np.nonzero(units)[0])))
    return eq, trades


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

    daily = {s: to_daily(d).iloc[:-1].set_index("timestamp") for s, d in h4.items()}
    syms = list(daily)
    Cd = pd.DataFrame({s: d["close"] for s, d in daily.items()}).sort_index()
    Od = pd.DataFrame({s: d["open"] for s, d in daily.items()}).reindex(Cd.index)
    Vd = pd.DataFrame({s: d["close"] * d["volume"] for s, d in daily.items()}).reindex(Cd.index)
    Cd = Cd[syms]; Od = Od[syms]; Vd = Vd[syms]
    idx = Cd.index
    hist = Cd.notna().cumsum()
    liq = Vd.rolling(30, min_periods=20).mean()
    liq_rank = liq.where(Cd.notna()).rank(axis=1, ascending=False)
    elig = (Cd.notna() & (hist >= MIN_HIST) & (liq_rank <= LIQ_TOP)).to_numpy()
    score = {L: (Cd / Cd.shift(L) - 1).to_numpy() for L in LOOKS}
    bd = Cd["BTC/USDT"]
    btc_on = (bd > ema(bd.dropna(), 50).reindex(idx)).to_numpy()

    keep = idx >= START
    C = Cd.to_numpy()[keep]; O = Od.to_numpy()[keep]
    Cf = Cd.ffill().to_numpy()[keep]
    elig = elig[keep]; btc_on = btc_on[keep]
    score = {L: v[keep] for L, v in score.items()}
    days = idx[keep]
    T = len(days)
    last_idx = np.array([np.max(np.nonzero(~np.isnan(C[:, j]))[0]) if (~np.isnan(C[:, j])).any() else -1 for j in range(C.shape[1])])
    reb = np.asarray(days.dayofweek == 6)
    reb[0] = True

    def top_pick(L, K, filt):
        def f(t, held):
            if filt and not btc_on[t]:
                return []
            s = np.where(elig[t] & ~np.isnan(score[L][t]), score[L][t], -np.inf)
            order = np.argsort(-s)[:K]
            return [j for j in order if np.isfinite(s[j])]
        return f

    def rnd_pick(K, filt, rng, keep_p):
        # turnover-matched: har rakha coin keep_p imkaan se rehta hai (agar eligible), baqi jagah random
        def f(t, held):
            if filt and not btc_on[t]:
                return []
            e = np.nonzero(elig[t])[0]
            kept = [j for j in held if elig[t, j] and rng.random() < keep_p][:K]
            pool = np.setdiff1d(e, kept)
            need = min(K - len(kept), len(pool))
            return kept + (list(rng.choice(pool, need, replace=False)) if need > 0 else [])
        return f

    def ew_pick(t, held):
        return list(np.nonzero(elig[t])[0])

    cost1 = FEE + SLIP

    def run(pick, cost=cost1):
        eq, tr = simulate(O, C, Cf, last_idx, reb, pick, cost)
        return pd.Series(eq, index=days), tr

    def oos_ret(eq):
        e = eq[eq.index >= OOS_START]
        return e.iloc[-1] / e.iloc[0] - 1 if len(e) > 1 else np.nan

    def yr_ret(eq):
        st = stats(eq)
        y = st["yearly"]
        return {yy: (y[y.index.year == yy].iloc[0] if (y.index.year == yy).any() else np.nan) for yy in YEARS}

    emit("=" * 160)
    emit("WEEKLY RELATIVE-STRENGTH ROTATION (top-K coins by L-din return, har itwar band -> peer open)")
    emit("=" * 160)
    emit(f"Coins: {len(syms)} | Period: {days[0].date()} -> {days[-1].date()} | OOS {OOS_START.date()} se | "
         f"kharcha fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% har taraf | eligible: {MIN_HIST}+ din history, top-{LIQ_TOP} liquidity")
    emit(f"Rozana ausat eligible coins: {elig.sum(axis=1)[elig.sum(axis=1) > 0].mean():.0f} | rebalance hafte: {int(reb.sum())}")

    # ---- benchmarks ----
    ew_eq, _ = run(ew_pick)
    btc = pd.Series(Cf[:, syms.index("BTC/USDT")], index=days)
    bench = {}
    for name, eq in (("EW sab coins", ew_eq), ("BTC buy&hold", btc / btc.iloc[0])):
        st = stats(eq)
        bench[name] = st
        yr = yr_ret(eq)
        emit(f"BENCH {name:>14}: CAGR {st['cagr']*100:+.1f}% | MaxDD {st['dd']*100:.1f}% | Sharpe {st['sharpe']:.2f} | OOS {oos_ret(eq)*100:+.1f}% | "
             + " ".join(f"{y}:{yr[y]*100:+.0f}%" for y in YEARS))

    def rnd_stats(K, filt, keep_p):
        sh, cg, oo, nt = [], [], [], []
        for sd in range(N_RND):
            eq, tr = run(rnd_pick(K, filt, np.random.default_rng(sd), keep_p))
            st = stats(eq)
            sh.append(st["sharpe"]); cg.append(st["cagr"]); oo.append(oos_ret(eq)); nt.append(len(tr))
        return dict(sh50=np.median(sh), sh95=np.percentile(sh, 95), cg50=np.median(cg),
                    oo50=np.nanmedian(oo), tpw=np.mean(nt) / (T / 7))

    HEAD = (f"{'Version':>16} | {'CAGR':>7} | {'MaxDD':>6} | {'Sharpe':>6} | {'RndSh50':>7} | {'RndSh95':>7} | {'RndCAGR':>7} | "
            f"{'OOS':>7} | {'RndOOS':>7} | {'2xCost':>7} | {'Tr/hfta':>7} | {'RndTr':>5} | {'Win%':>5} | {'PF':>5} | {'Hold':>4} | "
            + " | ".join(str(y) for y in YEARS) + " | PASS")
    res = {}
    for filt in (False, True):
        emit(f"\n{'#' * 160}\n{'BTC > EMA50 FILTER (warna cash)' if filt else 'BAGHAIR FILTER'}\n{'#' * 160}")
        emit(HEAD)
        for L in LOOKS:
            for K in KS:
                name = f"L{L} K{K}"
                eq, tr = run(top_pick(L, K, filt))
                eq2, _ = run(top_pick(L, K, filt), cost=2 * cost1)
                st, st2 = stats(eq), stats(eq2)
                inv_weeks = int((reb & (btc_on if filt else True)).sum())
                keep_p = float(np.clip(1 - len(tr) / max(inv_weeks * K, 1), 0, 1))
                rb = rnd_stats(K, filt, keep_p)
                print(f"{name} random done (keep_p {keep_p:.2f})")
                r = np.array([x[0] for x in tr]) if tr else np.array([0.0])
                hold = np.mean([x[1] for x in tr]) if tr else 0
                gp, gl = r[r > 0].sum(), -r[r < 0].sum()
                pf = gp / gl if gl > 0 else np.inf
                oo = oos_ret(eq)
                yr = yr_ret(eq)
                ok = (st["sharpe"] > rb["sh95"] and st["cagr"] > bench["EW sab coins"]["cagr"]
                      and oo > rb["oo50"] and st2["cagr"] > 0)
                res[(filt, L, K)] = dict(st=st, ok=ok, oo=oo, yr=yr, pf=pf)
                emit(f"{name:>16} | {st['cagr']*100:>+6.1f}% | {st['dd']*100:>5.1f}% | {st['sharpe']:>6.2f} | {rb['sh50']:>7.2f} | "
                     f"{rb['sh95']:>7.2f} | {rb['cg50']*100:>+6.1f}% | {oo*100:>+6.1f}% | {rb['oo50']*100:>+6.1f}% | "
                     f"{st2['cagr']*100:>+6.1f}% | {len(tr)/(T/7):>7.2f} | {rb['tpw']:>5.2f} | {(r > 0).mean()*100:>5.1f} | {pf:>5.2f} | {hold:>4.0f} | "
                     + " | ".join(f"{yr[y]*100:>+4.0f}%" for y in YEARS) + f" | {'PASS' if ok else '-'}")

    emit("\n" + "=" * 160)
    emit("KHULASA")
    emit("=" * 160)
    for filt in (False, True):
        lab = "BTC>EMA50 filter" if filt else "Baghair filter"
        n_ok = sum(res[(filt, L, K)]["ok"] for L in LOOKS for K in KS)
        emit(f"{lab}: {n_ok}/{len(LOOKS)*len(KS)} grid cells PASS")
        for K in KS:
            emit(f"   K{K}: " + " | ".join(f"L{L} {'PASS' if res[(filt, L, K)]['ok'] else '-'} (Sh {res[(filt, L, K)]['st']['sharpe']:.2f})" for L in LOOKS))
    emit("Faisla ka usool: plateau (zyada tar cells PASS, padosi cells bhi) ho tabhi aage validation; akela acha cell = itefaq.")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
