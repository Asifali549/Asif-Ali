"""
DONCHIAN N=20 - AGLI VALIDATION (koi indicator nahi) - 2026-10-01
================================================================
Base: Daily close > pichle 20 din ka high -> agle din open; initial stop 3 x ATR(14). Do version: filter baghair / BTC>EMA50.
TEST A - COST STRESS: kharcha 1x (fee 0.1% + slip 0.05% har taraf + stop slip 0.25%), 2x aur 3x
         exits: ATR trailing 3x aur Donchian 10-din low exit
TEST B - DONCHIAN EXIT PLATEAU: close < pichle M din ka low -> agle din open; M = 5 / 10 / 15 / 20 / 30
TEST C - 2025 TAJZIYA: mahana trades/PF/win (strategy vs random), BTC regime, market breadth
         (coins ka % jo EMA50 se ooper), sab se bure coins, initial-stop ki sharah
Natija: donchian_stress_RESULTS.txt
"""
import numpy as np
import pandas as pd

import donchian_research as R
import donchian_validate as V
from bot_core import fetch_full, norm, STABLES, FEE, SLIP, STOP_SLIP
from portfolio_lab import ema, to_daily, portfolio, stats

OUT = "donchian_stress_RESULTS.txt"
M_STOP = 3.0
pf_of, fmt = R.pf_of, R.fmt


def set_costs(mult):
    for mod in (R, V):
        mod.FEE, mod.SLIP, mod.STOP_SLIP = FEE * mult, SLIP * mult, STOP_SLIP * mult


def run_donch(U, sig, M, rng=None, allow=None):
    rows = []
    for s in U.syms:
        a, d = U.arr[s], U.frames[s]
        ex = (d["close"] < d["low"].shift(1).rolling(M).min()).to_numpy()
        idx = sig[s]
        if rng is not None:
            pool = np.arange(a["warm"], len(a["o"]) - 1)
            if allow is not None:
                pool = pool[allow[s][pool]]
            k = min(len(idx), len(pool))
            if k == 0:
                continue
            idx = np.sort(rng.choice(pool, size=k, replace=False)).astype(np.int64)
        if len(idx) == 0:
            continue
        e, x, r, rk, w = V.sim_exit_sig(a["o"], a["h"], a["l"], a["c"], a["atr"], idx, M_STOP, ex, 365,
                                        V.SLIP, V.FEE, V.STOP_SLIP)
        if len(e):
            rows.append(pd.DataFrame({"sym": s, "t_in": a["ts"][e], "t_out": a["ts"][x], "ret": r, "risk": rk, "why": w,
                                      "entry_px": a["o"][e] * (1 + V.SLIP)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x)
        lines.append(x)

    if h4 is None:
        from data_fetcher import get_exchange, get_coin_list
        ex = get_exchange()
        coins = [s for s in get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:R.TOP_N]
        if "BTC/USDT" not in coins:
            coins.insert(0, "BTC/USDT")
        h4 = {}
        for k, sym in enumerate(coins, 1):
            try:
                df = fetch_full(ex, sym, "4h", R.H4_BARS)
                if df is not None and len(df) >= 6 * 250:
                    h4[sym] = norm(df).reset_index(drop=True)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)}")
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})")

    daily = {s: to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = max(pd.Timestamp("2020-10-01"), closes.index[0] + pd.Timedelta(days=30))
    closes = closes[closes.index >= start]
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    b50, b200 = ema(bd, 50), ema(bd, 200)
    btc_reg = pd.Series(np.where((bd > b200) & (b50 > b200), "bull", np.where((bd < b200) & (b50 < b200), "bear", "sideways")),
                        index=bd.index)
    btc_ok = bd > b50
    U = R.Universe(daily, warm=60)
    allow = {s: btc_ok.reindex(d["timestamp"]).fillna(False).to_numpy(bool) for s, d in daily.items()}
    sig_all = U.signals(20)
    sig_f = {k: v[allow[k][v]] for k, v in sig_all.items()}
    VERS = (("BAGHAIR filter", sig_all, None), ("BTC>EMA50", sig_f, allow))

    def cut(df):
        return df[df["t_in"] >= start].reset_index(drop=True) if len(df) else df

    def run(kind, sig, al, M=10):
        if kind == "trail":
            tr = U.run(sig, M_STOP, 0.0, True, 365)
            rt = U.run(sig, M_STOP, 0.0, True, 365, rng=np.random.default_rng(7), allow=al)
        else:
            tr = run_donch(U, sig, M)
            rt = run_donch(U, sig, M, rng=np.random.default_rng(7), allow=al)
        return cut(tr), cut(rt)

    def summary(tr, rt):
        r, rr = tr["ret"].to_numpy(), rt["ret"].to_numpy()
        oo = tr[tr["t_in"] >= R.OOS_START]["ret"].to_numpy()
        ro = rt[rt["t_in"] >= R.OOS_START]["ret"].to_numpy()
        eq, _ = portfolio(tr.assign(prio=0.0).to_dict("records"), closes, "risk", 0.01)
        st = stats(eq)
        return (f"{len(r):>6} | {(r > 0).mean()*100:>5.1f} | {fmt(pf_of(r)):>5} | {fmt(pf_of(rr)):>5} | "
                f"{pf_of(r) - pf_of(rr):>+5.2f} | {r.mean()*100:>+6.2f} | {fmt(pf_of(oo)):>6} | {fmt(pf_of(ro)):>6} | "
                f"{st['cagr']*100:>+6.1f}% | {st['dd']*100:>6.1f}%"), pf_of(r), pf_of(oo), pf_of(r) - pf_of(rr)

    HEAD = (f"{'':>34} | {'Trades':>6} | {'Win%':>5} | {'PF':>5} | {'RndPF':>5} | {'PF+':>5} | {'Exp%':>6} | {'OOS PF':>6} | "
            f"{'RndOOS':>6} | {'CAGR':>7} | {'MaxDD':>7}")

    emit("=" * 130)
    emit("DONCHIAN N=20 - COST STRESS / EXIT PLATEAU / 2025 TAJZIYA (koi indicator nahi)")
    emit("=" * 130)
    emit(f"Coins: {len(daily)} | Period: {start.date()} -> {closes.index[-1].date()} | OOS {R.OOS_START.date()} se | "
         "portfolio 1% risk, max 10, coin max 20%")

    # ---------------- TEST A ----------------
    emit(f"\n\n{'#' * 130}\nTEST A - COST STRESS (1x = fee 0.10% + slip 0.05% har taraf + stop slip 0.25%)\n{'#' * 130}")
    emit(HEAD)
    stress = {}
    for mult in (1, 2, 3):
        set_costs(mult)
        for tag, sig, al in VERS:
            for kind, lab in (("trail", "ATR trail 3x"), ("donch", "Donchian 10 exit")):
                txt, pf, oo, imp = summary(*run(kind, sig, al))
                stress[(mult, tag, kind)] = (pf, oo, imp)
                emit(f"{f'{mult}x kharcha | {tag} | {lab}':>34} | {txt}")
    set_costs(1)

    # ---------------- TEST B ----------------
    emit(f"\n\n{'#' * 130}\nTEST B - DONCHIAN EXIT PLATEAU (close < pichle M din ka low -> agle din open; stop 3xATR)\n{'#' * 130}")
    emit(HEAD)
    plat = {}
    for tag, sig, al in VERS:
        for M in (5, 10, 15, 20, 30):
            txt, pf, oo, imp = summary(*run("donch", sig, al, M))
            plat[(tag, M)] = (pf, oo, imp)
            emit(f"{f'{tag} | exit M={M}':>34} | {txt}")
        emit("")

    # ---------------- TEST C ----------------
    emit(f"\n\n{'#' * 130}\nTEST C - 2025 TAJZIYA (N=20, filter baghair, ATR trail 3x)\n{'#' * 130}")
    tr, rt = run("trail", sig_all, None)
    y = lambda df: df[df["t_in"].dt.year == 2025]
    t25, r25 = y(tr), y(rt)
    above = (closes > closes.ewm(span=50, adjust=False).mean()).mean(axis=1)
    b25 = bd[bd.index.year == 2025]
    emit(f"BTC 2025: {b25.iloc[0]:,.0f} -> {b25.iloc[-1]:,.0f} ({(b25.iloc[-1]/b25.iloc[0]-1)*100:+.0f}%) | BTC regime din: "
         + ", ".join(f"{k} {v*100:.0f}%" for k, v in btc_reg[btc_reg.index.year == 2025].value_counts(normalize=True).items()))
    emit(f"{'Mahina':>8} | {'Trades':>6} | {'Win%':>5} | {'PF':>5} | {'Rnd PF':>6} | {'Avg%':>6} | {'Breadth':>7} | {'BTC mahana':>10}")
    for m in range(1, 13):
        a = t25[t25["t_in"].dt.month == m]["ret"].to_numpy()
        b = r25[r25["t_in"].dt.month == m]["ret"].to_numpy()
        br = above[(above.index.year == 2025) & (above.index.month == m)].mean()
        bm = b25[b25.index.month == m]
        if not len(a):
            continue
        emit(f"{f'2025-{m:02d}':>8} | {len(a):>6} | {(a>0).mean()*100:>5.1f} | {fmt(pf_of(a)):>5} | {fmt(pf_of(b)):>6} | "
             f"{a.mean()*100:>+6.2f} | {br*100:>6.0f}% | {(bm.iloc[-1]/bm.iloc[0]-1)*100 if len(bm) else 0:>+9.0f}%")
    emit(f"\nMarket breadth (coins ka % EMA50 se ooper), saal-war ausat: " + " | ".join(
        f"{yy}: {above[above.index.year == yy].mean()*100:.0f}%" for yy in sorted(set(above.index.year))))
    emit("Har saal: " + " | ".join(
        f"{yy}: n={len(g)} PF {fmt(pf_of(g['ret'].to_numpy()))} win {(g['ret']>0).mean()*100:.0f}% stop-out "
        f"{(g['why']==0).mean()*100:.0f}%" for yy, g in tr.groupby(tr["t_in"].dt.year)))
    by = t25.groupby("sym")["ret"].agg(["count", "sum"]).sort_values("sum")
    emit("2025 sab se bura nuqsan (coins): " + ", ".join(f"{s.split('/')[0]} {v*100:+.0f}% ({int(n)})" for s, (n, v) in by.head(10).iterrows()))
    emit("2025 sab se acha (coins): " + ", ".join(f"{s.split('/')[0]} {v*100:+.0f}% ({int(n)})" for s, (n, v) in by.tail(5).iloc[::-1].iterrows()))
    loss_share = by["sum"].clip(upper=0)
    emit(f"2025 nuqsan kitna phaila: {int((by['sum'] < 0).sum())}/{len(by)} coins manfi; sab se bure 10 coins = "
         f"{loss_share.head(10).sum() / loss_share.sum() * 100:.0f}% kul nuqsan")
    reg = btc_reg.reindex(t25["t_in"].dt.floor("1D") - pd.Timedelta(days=1)).to_numpy()
    emit("2025 trades BTC regime ke hisab se: " + " | ".join(
        f"{g}: n={int((reg == g).sum())} PF {fmt(pf_of(t25['ret'].to_numpy()[reg == g]))}" for g in ("bull", "bear", "sideways")))

    # ---------------- KHULASA ----------------
    emit("\n\n" + "=" * 130)
    emit("KHULASA")
    emit("=" * 130)
    for tag, _, _ in VERS:
        for kind, lab in (("trail", "ATR trail"), ("donch", "Donchian 10 exit")):
            p1, p2, p3 = (stress[(m, tag, kind)] for m in (1, 2, 3))
            emit(f"Cost stress {tag:>14} {lab:>17}: PF {fmt(p1[0])} -> {fmt(p2[0])} (2x) -> {fmt(p3[0])} (3x) | "
                 f"OOS {fmt(p1[1])} -> {fmt(p2[1])} -> {fmt(p3[1])} | random par faida {p1[2]:+.2f} -> {p2[2]:+.2f} -> {p3[2]:+.2f}")
    for tag, _, _ in VERS:
        ok = sum(1 for M in (5, 10, 15, 20, 30) if plat[(tag, M)][0] > 1.2 and plat[(tag, M)][1] > 1.0 and plat[(tag, M)][2] > 0.1)
        emit(f"Exit plateau {tag}: {ok}/5 M values PF>1.2, OOS>1, random se +0.1 behtar | PF: "
             + " / ".join(f"M{M} {fmt(plat[(tag, M)][0])}" for M in (5, 10, 15, 20, 30)))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
