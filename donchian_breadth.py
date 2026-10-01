"""
DAILY DONCHIAN N=20 + SIRF EK FILTER: ALTCOIN BREADTH (2026-10-01)
==================================================================
Tashkhees (stress test): 2022 aur 2025 mein nuqsan - jab altcoin market kamzor thi; BTC regime ise nahi pakarta.
Filter: signal wale BAND din par "breadth" = un coins ka % jo apni EMA50 se ooper hain
        (sirf wo coins gine jaate hain jo us din maujood the aur jin ki kam az kam 50 din history thi).
        breadth >= had -> nayi entry allowed. Had = 30% / 40% / 50% / 60% (plateau dekhne ke liye).
Muqabla: filter baghair aur BTC>EMA50 (pichla filter) - reference.
Exits: ATR stop 3x + ATR trailing 3x (main) aur Donchian 10-din low exit.
Random baseline: filter wali row mein random bhi sirf allowed din par. Kharcha 1x. 109 coins, Oct 2020 -> Sep 2026.
Report: PF, random PF, faida, OOS (2025+), saal-war PF (2022/2025 khaas), BTC regimes, coin-level median PF,
walk-forward saal (2023-2026), portfolio CAGR/MaxDD.
Natija: donchian_breadth_RESULTS.txt
"""
import numpy as np
import pandas as pd

import donchian_research as R
import donchian_stress as S
from bot_core import fetch_full, norm, STABLES, FEE, SLIP, STOP_SLIP
from portfolio_lab import ema, to_daily, portfolio, stats

OUT = "donchian_breadth_RESULTS.txt"
M_STOP = 3.0
THRESH = [0.30, 0.40, 0.50, 0.60]
pf_of, fmt = R.pf_of, R.fmt


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

    # ---- breadth: sahi denominator ----
    above, valid = {}, {}
    for s, d in daily.items():
        x = d.set_index("timestamp")["close"]
        e = ema(x, 50)
        ok = pd.Series(np.arange(len(x)) >= 50, index=x.index)
        above[s] = ((x > e) & ok).astype(float)
        valid[s] = ok.astype(float)
    A = pd.DataFrame(above).sort_index()
    Vd = pd.DataFrame(valid).sort_index()
    breadth = (A.fillna(0).sum(axis=1) / Vd.fillna(0).sum(axis=1).replace(0, np.nan)).fillna(0)

    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    b50, b200 = ema(bd, 50), ema(bd, 200)
    btc_reg = pd.Series(np.where((bd > b200) & (b50 > b200), "bull", np.where((bd < b200) & (b50 < b200), "bear", "sideways")),
                        index=bd.index)
    U = R.Universe(daily, warm=60)
    sig_all = U.signals(20)

    def mk_allow(day_ok):
        return {s: day_ok.reindex(d["timestamp"]).fillna(False).to_numpy(bool) for s, d in daily.items()}

    versions = [("BAGHAIR filter", None)]
    versions.append(("BTC>EMA50 (ref)", mk_allow(bd > b50)))
    for t in THRESH:
        versions.append((f"Breadth >= {int(t*100)}%", mk_allow(breadth >= t)))

    emit("=" * 140)
    emit("DAILY DONCHIAN N=20 + EK FILTER: ALTCOIN BREADTH (% coins EMA50 se ooper, sirf maujood coins)")
    emit("=" * 140)
    emit(f"Coins: {len(daily)} | Period: {start.date()} -> {closes.index[-1].date()} | OOS {R.OOS_START.date()} se | "
         f"kharcha fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% + stop slip {STOP_SLIP*100:.2f}%")
    br = breadth[breadth.index >= start]
    emit("Breadth saal-war ausat: " + " | ".join(f"{y}: {br[br.index.year == y].mean()*100:.0f}%" for y in sorted(set(br.index.year))))
    emit("Allowed din (%): " + " | ".join(f"{n}: {(br >= t).mean()*100:.0f}%" for n, t in [(f'>={int(t*100)}%', t) for t in THRESH])
         + f" | BTC>EMA50: {(bd[bd.index >= start] > b50[b50.index >= start]).mean()*100:.0f}%")

    def cut(df):
        return df[df["t_in"] >= start].reset_index(drop=True) if len(df) else df

    YEARS = list(range(2021, 2027))
    HEAD = (f"{'Version':>22} | {'Trades':>6} | {'Win%':>5} | {'PF':>5} | {'RndPF':>5} | {'PF+':>5} | {'OOS':>5} | {'RndOOS':>6} | "
            f"{'CAGR':>7} | {'MaxDD':>7} | {'MedCoinPF':>9} | {'Coins>1':>7} | " + " | ".join(f"{y}" for y in YEARS))
    results = {}
    for kind, lab in (("trail", "ATR TRAILING 3x"), ("donch", "DONCHIAN 10-DIN LOW EXIT")):
        emit(f"\n\n{'#' * 140}\nEXIT: {lab}\n{'#' * 140}")
        emit(HEAD)
        for name, al in versions:
            sig = sig_all if al is None else {k: v[al[k][v]] for k, v in sig_all.items()}
            if kind == "trail":
                tr = U.run(sig, M_STOP, 0.0, True, 365)
                rt = U.run(sig, M_STOP, 0.0, True, 365, rng=np.random.default_rng(7), allow=al)
            else:
                tr = S.run_donch(U, sig, 10)
                rt = S.run_donch(U, sig, 10, rng=np.random.default_rng(7), allow=al)
            tr, rt = cut(tr), cut(rt)
            r, rr = tr["ret"].to_numpy(), rt["ret"].to_numpy()
            oo = tr[tr["t_in"] >= R.OOS_START]["ret"].to_numpy()
            ro = rt[rt["t_in"] >= R.OOS_START]["ret"].to_numpy()
            eq, _ = portfolio(tr.assign(prio=0.0).to_dict("records"), closes, "risk", 0.01)
            st = stats(eq)
            cpf = tr.groupby("sym")["ret"].agg(lambda x: pf_of(x.to_numpy()) if len(x) >= 5 else np.nan).dropna()
            cpf = cpf.replace(np.inf, 5.0)
            yr = {y: pf_of(tr[tr["t_in"].dt.year == y]["ret"].to_numpy()) for y in YEARS}
            ryr = {y: pf_of(rt[rt["t_in"].dt.year == y]["ret"].to_numpy()) for y in YEARS}
            reg = btc_reg.reindex(tr["t_in"].dt.floor("1D") - pd.Timedelta(days=1)).to_numpy()
            regs = {g: (pf_of(r[reg == g]), int((reg == g).sum())) for g in ("bull", "bear", "sideways")}
            res = dict(n=len(r), pf=pf_of(r), rpf=pf_of(rr), oos=pf_of(oo), roos=pf_of(ro), cagr=st["cagr"], dd=st["dd"],
                       med=cpf.median(), share=(cpf > 1).mean(), yr=yr, ryr=ryr, regs=regs)
            results[(kind, name)] = res
            emit(f"{name:>22} | {res['n']:>6} | {(r > 0).mean()*100:>5.1f} | {fmt(res['pf']):>5} | {fmt(res['rpf']):>5} | "
                 f"{res['pf'] - res['rpf']:>+5.2f} | {fmt(res['oos']):>5} | {fmt(res['roos']):>6} | {res['cagr']*100:>+6.1f}% | "
                 f"{res['dd']*100:>6.1f}% | {res['med']:>9.2f} | {res['share']*100:>6.0f}% | "
                 + " | ".join(f"{fmt(yr[y]):>4}" for y in YEARS))
        emit("\nRandom PF saal-war (har version):")
        for name, _ in versions:
            ryr = results[(kind, name)]["ryr"]
            emit(f"{name:>22} | " + " | ".join(f"{y}: {fmt(ryr[y])}" for y in YEARS))
        emit("\nBTC regimes (PF, trades):")
        for name, _ in versions:
            rg = results[(kind, name)]["regs"]
            emit(f"{name:>22} | " + " | ".join(f"{g}: {fmt(rg[g][0])} ({rg[g][1]})" for g in ("bull", "bear", "sideways")))

    emit("\n" + "=" * 140)
    emit("KHULASA - kya breadth filter 2022/2025 ka masla theek karta hai, aur plateau hai?")
    emit("=" * 140)
    for kind, lab in (("trail", "ATR trail"), ("donch", "Donchian 10 exit")):
        b = results[(kind, "BAGHAIR filter")]
        emit(f"\n{lab}: baghair filter -> PF {fmt(b['pf'])}, OOS {fmt(b['oos'])}, DD {b['dd']*100:.0f}%, 2022 {fmt(b['yr'][2022])}, 2025 {fmt(b['yr'][2025])}")
        good = 0
        for t in THRESH:
            name = f"Breadth >= {int(t*100)}%"
            x = results[(kind, name)]
            better = (x["oos"] >= b["oos"] and x["dd"] >= b["dd"] and x["yr"][2022] > b["yr"][2022]
                      and x["yr"][2025] > b["yr"][2025] and x["pf"] - x["rpf"] >= 0.15)
            good += better
            emit(f"   {name}: PF {fmt(x['pf'])} (random {fmt(x['rpf'])}, {x['pf']-x['rpf']:+.2f}) | OOS {fmt(x['oos'])} | DD {x['dd']*100:.0f}% | "
                 f"2022 {fmt(x['yr'][2022])} | 2025 {fmt(x['yr'][2025])} | trades {x['n']} -> {'BEHTAR (har cheez)' if better else 'har cheez behtar nahi'}")
        emit(f"   Plateau: {good}/{len(THRESH)} had values har lihaz se behtar")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
