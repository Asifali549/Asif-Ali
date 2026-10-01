"""
WEEKLY RS ROTATION - VALIDATION (2026-10-01)
============================================
Base: rs_rotation.py jaisa (har itwar band -> top-K by L-din return, peer open par sirf nikle/naye coin; kharcha
fee 0.1% + slip 0.05% har taraf; random = turnover-matched, 50 seeds -> p95). Plateau wale 6 configs.
A) CONCENTRATION: kaun se coins nafa banate hain (top-5 ka hissa); top-5 nafa wale coins universe se NIKAAL kar
   dobara chalao - kya ab bhi random p95 se behtar?
B) 2022 SE SHURU: 2021 ka bubble nikaal kar CAGR / MaxDD / Sharpe vs random, EW aur BTC (usi period mein).
C) SURVIVORSHIP PROXY (har ek alag): (1) sirf purane coins (365+ din history) (2) sirf top-50 liquidity.
   Naye/chhote coins mein pump-and-die ka khatra ziada - agar edge sirf wahan hai to shak.
D) DRAWDOWN CONTROL (sirf EK cheez): strategy ki apni equity < apni SMA(N) -> sab becho, cash (N = 30/50/100);
   wapas ooper -> agle itwar dobara top-K. Muqabla: baghair control.
Natija: rs_validate_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES, FEE, SLIP
from portfolio_lab import to_daily, stats

OUT = "rs_validate_RESULTS.txt"
TOP_N, H4_BARS = 150, 2400 * 6
CONFIGS = [(14, 5), (14, 10), (30, 10), (60, 5), (60, 10), (90, 10)]
N_RND = 50
START = pd.Timestamp("2020-10-01")
START22 = pd.Timestamp("2022-01-01")
OOS_START = pd.Timestamp("2025-01-01")
YEARS = list(range(2021, 2027))
COST = FEE + SLIP


def simulate(O, C, Cf, last_idx, reb, pick, cost):
    T, N = C.shape
    units = np.zeros(N)
    entry = np.full(N, np.nan)
    basis = np.zeros(N)
    ent_t = np.zeros(N, int)
    cash = 1.0
    eq = np.empty(T)
    trades = []          # (coin, ret, hold, pnl)
    pending = None

    def close(j, px, t):
        nonlocal cash
        val = units[j] * px * (1 - cost)
        cash += val
        trades.append((j, px * (1 - cost) / entry[j] - 1, t - ent_t[j], val - basis[j]))
        units[j] = 0.0

    for t in range(T):
        if pending is not None:
            held = set(np.nonzero(units)[0])
            for j in held - pending:
                close(j, O[t, j] if not np.isnan(O[t, j]) else Cf[t, j], t)
            new = [j for j in pending - held if not np.isnan(O[t, j])]
            if new and cash > 0:
                amt = cash / len(new)
                for j in new:
                    units[j] = amt / (O[t, j] * (1 + cost))
                    entry[j] = O[t, j] * (1 + cost)
                    basis[j] = amt
                    ent_t[j] = t
                cash = 0.0
            pending = None
        for j in np.nonzero(units)[0]:
            if t > last_idx[j]:
                close(j, Cf[t, j], t)
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
    Cd = pd.DataFrame({s: d["close"] for s, d in daily.items()}).sort_index()[syms]
    Od = pd.DataFrame({s: d["open"] for s, d in daily.items()}).reindex(Cd.index)[syms]
    Vd = pd.DataFrame({s: d["close"] * d["volume"] for s, d in daily.items()}).reindex(Cd.index)[syms]
    idx = Cd.index
    hist = Cd.notna().cumsum()
    liq_rank = Vd.rolling(30, min_periods=20).mean().where(Cd.notna()).rank(axis=1, ascending=False)
    scores = {L: (Cd / Cd.shift(L) - 1).to_numpy() for L in sorted({c[0] for c in CONFIGS})}
    Cn, On, Cfn = Cd.to_numpy(), Od.to_numpy(), Cd.ffill().to_numpy()

    def ctx(start=START, min_hist=90, liq_top=100, excl=()):
        keep = idx >= start
        elig = (Cd.notna() & (hist >= min_hist) & (liq_rank <= liq_top)).to_numpy()[keep]
        for j in excl:
            elig[:, j] = False
        C = Cn[keep]
        days = idx[keep]
        reb = np.asarray(days.dayofweek == 6)
        reb[0] = True
        last = np.array([np.max(np.nonzero(~np.isnan(C[:, j]))[0]) if (~np.isnan(C[:, j])).any() else -1
                         for j in range(C.shape[1])])
        return dict(C=C, O=On[keep], Cf=Cfn[keep], elig=elig, sc={L: v[keep] for L, v in scores.items()},
                    days=days, reb=reb, last=last, T=len(days))

    def top_pick(x, L, K, gate=None):
        def f(t, held):
            if gate is not None and not gate[t]:
                return []
            s = np.where(x["elig"][t] & ~np.isnan(x["sc"][L][t]), x["sc"][L][t], -np.inf)
            order = np.argsort(-s)[:K]
            return [j for j in order if np.isfinite(s[j])]
        return f

    def rnd_pick(x, K, rng, keep_p):
        def f(t, held):
            e = np.nonzero(x["elig"][t])[0]
            kept = [j for j in held if x["elig"][t, j] and rng.random() < keep_p][:K]
            pool = np.setdiff1d(e, kept)
            need = min(K - len(kept), len(pool))
            return kept + (list(rng.choice(pool, need, replace=False)) if need > 0 else [])
        return f

    def run(x, pick):
        eq, tr = simulate(x["O"], x["C"], x["Cf"], x["last"], x["reb"], pick, COST)
        return pd.Series(eq, index=x["days"]), tr

    def evaluate(x, L, K, with_rnd=True):
        eq, tr = run(x, top_pick(x, L, K))
        st = stats(eq)
        out = dict(eq=eq, tr=tr, st=st)
        if with_rnd:
            keep_p = float(np.clip(1 - len(tr) / max(int(x["reb"].sum()) * K, 1), 0, 1))
            sh = [stats(run(x, rnd_pick(x, K, np.random.default_rng(sd), keep_p))[0])["sharpe"] for sd in range(N_RND)]
            out["r50"], out["r95"] = np.median(sh), np.percentile(sh, 95)
        return out

    def oos(eq):
        e = eq[eq.index >= OOS_START]
        return e.iloc[-1] / e.iloc[0] - 1 if len(e) > 1 else np.nan

    def yrs(eq):
        y = stats(eq)["yearly"]
        return " ".join(f"{yy}:{y[y.index.year == yy].iloc[0]*100:+.0f}%" for yy in YEARS if (y.index.year == yy).any())

    def row(name, o):
        st = o["st"]
        rnd = f" | rnd p50 {o['r50']:.2f} p95 {o['r95']:.2f} -> {'PASS' if st['sharpe'] > o['r95'] else 'FAIL'}" if "r95" in o else ""
        return (f"{name:>12} | CAGR {st['cagr']*100:+7.1f}% | MaxDD {st['dd']*100:6.1f}% | Sharpe {st['sharpe']:.2f} | "
                f"+mahine {st['pos_months']:.0f}% | bura mahina {st['worst_month']*100:+.0f}% | DD din {st['long_dd']}{rnd}")

    base = ctx()
    emit("=" * 150)
    emit("WEEKLY RS ROTATION - VALIDATION")
    emit("=" * 150)
    emit(f"Coins: {len(syms)} | {base['days'][0].date()} -> {base['days'][-1].date()} | kharcha {COST*100:.2f}% har taraf | random {N_RND} seeds (turnover-matched)")

    # ---------------- A) concentration ----------------
    emit("\n" + "#" * 150 + "\nA) CONCENTRATION - nafa kitne coins par tika hai\n" + "#" * 150)
    passA = 0
    for L, K in CONFIGS:
        o = evaluate(base, L, K, with_rnd=False)
        pnl = pd.Series({syms[j]: 0.0 for j in set(t[0] for t in o["tr"])})
        for j, r, h, p in o["tr"]:
            pnl[syms[j]] += p
        tot = pnl.sum()
        pos = pnl[pnl > 0].sort_values(ascending=False)
        top5 = pos.head(5)
        share = top5.sum() / tot if tot > 0 else np.nan
        excl = [syms.index(s) for s in top5.index]
        ox = evaluate(ctx(excl=excl), L, K)
        ok = ox["st"]["sharpe"] > ox["r95"] and ox["st"]["cagr"] > 0
        passA += ok
        emit(f"L{L} K{K}: {len(pnl)} coins trade hue, nafa wale {int((pnl > 0).sum())} / nuqsan wale {int((pnl < 0).sum())} | "
             f"top-5 coins = kul net nafa ka {share*100:.0f}% ({', '.join(s.split('/')[0] for s in top5.index)})")
        emit(row("  baghair top5", ox) + f" -> {'MAZBOOT' if ok else 'KAMZOR'}")
    emit(f"A natija: {passA}/{len(CONFIGS)} configs top-5 coins nikaal kar bhi random p95 se behtar")

    # ---------------- B) 2022 se ----------------
    emit("\n" + "#" * 150 + "\nB) 2022-01-01 SE SHURU (2021 bubble bahar)\n" + "#" * 150)
    x22 = ctx(start=START22)
    btc = pd.Series(x22["Cf"][:, syms.index("BTC/USDT")], index=x22["days"])
    ew, _ = run(x22, lambda t, held: list(np.nonzero(x22["elig"][t])[0]))
    for nm, e in (("EW sab coins", ew), ("BTC B&H", btc / btc.iloc[0])):
        st = stats(e)
        emit(f"BENCH {nm:>12}: CAGR {st['cagr']*100:+.1f}% | MaxDD {st['dd']*100:.1f}% | Sharpe {st['sharpe']:.2f} | {yrs(e)}")
    ew_c, btc_c = stats(ew)["cagr"], stats(btc / btc.iloc[0])["cagr"]
    passB = 0
    for L, K in CONFIGS:
        o = evaluate(x22, L, K)
        ok = o["st"]["sharpe"] > o["r95"] and o["st"]["cagr"] > max(ew_c, btc_c)
        passB += ok
        emit(row(f"L{L} K{K}", o) + f" | {yrs(o['eq'])} | Tr/hfta {len(o['tr'])/(x22['T']/7):.1f} -> {'PASS' if ok else '-'}")
    emit(f"B natija: {passB}/{len(CONFIGS)} configs 2022 se bhi random p95 + EW + BTC se behtar")

    # ---------------- C) survivorship proxy ----------------
    emit("\n" + "#" * 150 + "\nC) SURVIVORSHIP PROXY - sirf purane / bare coins\n" + "#" * 150)
    passC = {}
    for nm, x in (("365+ din history", ctx(min_hist=365)), ("top-50 liquidity", ctx(liq_top=50))):
        emit(f"-- {nm} (rozana ausat eligible {x['elig'].sum(axis=1)[x['elig'].sum(axis=1) > 0].mean():.0f} coins)")
        n = 0
        for L, K in CONFIGS:
            o = evaluate(x, L, K)
            ok = o["st"]["sharpe"] > o["r95"]
            n += ok
            emit(row(f"L{L} K{K}", o) + f" | OOS {oos(o['eq'])*100:+.0f}%")
        passC[nm] = n
        emit(f"   {nm}: {n}/{len(CONFIGS)} PASS")

    # ---------------- D) drawdown control ----------------
    emit("\n" + "#" * 150 + "\nD) DRAWDOWN CONTROL - apni equity < apni SMA(N) -> cash\n" + "#" * 150)
    dd_ok = {}
    for L, K in CONFIGS:
        o = evaluate(base, L, K, with_rnd=False)
        shadow = o["eq"]
        emit(row(f"L{L} K{K} base", o) + f" | {yrs(o['eq'])}")
        for N in (30, 50, 100):
            gate = (shadow >= shadow.rolling(N, min_periods=N).mean()).to_numpy() | np.isnan(shadow.rolling(N, min_periods=N).mean().to_numpy())
            eqg, trg = run(base, top_pick(base, L, K, gate))
            og = dict(eq=eqg, tr=trg, st=stats(eqg))
            better = og["st"]["dd"] > -0.45 and og["st"]["sharpe"] >= o["st"]["sharpe"] - 0.10 and og["st"]["cagr"] > 0
            dd_ok[(L, K, N)] = better
            emit(row(f"   SMA{N}", og) + f" | {yrs(eqg)} | cash {100 - gate.mean()*100:.0f}% din -> {'BEHTAR' if better else '-'}")
    for N in (30, 50, 100):
        emit(f"SMA{N}: {sum(dd_ok[(L, K, N)] for L, K in CONFIGS)}/{len(CONFIGS)} configs mein DD > -45%, Sharpe 0.10 se ziada kam nahi, CAGR > 0")

    emit("\n" + "=" * 150 + "\nKHULASA\n" + "=" * 150)
    emit(f"A) Concentration (top-5 coins nikaal kar): {passA}/{len(CONFIGS)}")
    emit(f"B) 2022 se: {passB}/{len(CONFIGS)}")
    emit("C) Survivorship: " + " | ".join(f"{k}: {v}/{len(CONFIGS)}" for k, v in passC.items()))
    emit("D) Drawdown control: " + " | ".join(f"SMA{N}: {sum(dd_ok[(L, K, N)] for L, K in CONFIGS)}/{len(CONFIGS)}" for N in (30, 50, 100)))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
