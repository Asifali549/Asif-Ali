"""
ALT/BTC BREAKOUT (2026-10-01)
=============================
Khayal: USDT chart par breakout aksar sirf BTC ke sath chalne ka nateeja hota hai (2022/2025 ke jhoote rally).
ALT/BTC ratio (coin close / BTC close) par breakout = coin BTC se bhi aage nikal raha hai (apni taqat).
Trade hamesha spot USDT mein; sirf SIGNAL ka chart badla. BTC khud trade nahi (sab versions mein bahar).
VERSIONS (har group mein sirf EK cheez badli):
  BASE          : USDT Donchian N=20 (close > pichle 20 din ka high) - pehle se sabit.
  RATIO BRK N   : ALT/BTC ratio close > pichle N din ka sab se ooncha ratio close (N = 10/20/30/55) - naya entry.
  BASE + RATIO>EMA L : USDT N20 breakout sirf jab ratio > apni EMA(L) (L = 20/50/100) - ek filter.
Exits (dono): ATR stop 3x + ATR trailing 3x | Donchian 10-din low exit.  Kharcha 1x.
Random baseline: utni hi entries, random din (filter wali row mein sirf allowed din).
Pass (base ke muqable): random se +0.15 PF, OOS > 1 aur >= base, MaxDD base se bura nahi (3% gunjaish),
2022 aur 2025 PF base se behtar, n >= 150.  Plateau: group ke ziada tar N/L pass.
Natija: alt_btc_RESULTS.txt
"""
import numpy as np
import pandas as pd

import donchian_research as R
import donchian_stress as S
from bot_core import fetch_full, norm, STABLES, FEE, SLIP, STOP_SLIP
from portfolio_lab import ema, to_daily, portfolio, stats

OUT = "alt_btc_RESULTS.txt"
M_STOP = 3.0
RATIO_NS = [10, 20, 30, 55]
EMA_LS = [20, 50, 100]
YEARS = list(range(2021, 2027))
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

    daily_all = {s: to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    bd = daily_all["BTC/USDT"].set_index("timestamp")["close"]
    daily = {s: d for s, d in daily_all.items() if s != "BTC/USDT"}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = max(pd.Timestamp("2020-10-01"), closes.index[0] + pd.Timedelta(days=30))
    closes = closes[closes.index >= start]

    b50, b200 = ema(bd, 50), ema(bd, 200)
    btc_reg = pd.Series(np.where((bd > b200) & (b50 > b200), "bull", np.where((bd < b200) & (b50 < b200), "bear", "sideways")),
                        index=bd.index)

    # ---- ratio series (har coin ke apne din par; BTC ka us din ka close) ----
    ratio = {}
    for s, d in daily.items():
        btc_c = bd.reindex(d["timestamp"]).to_numpy()
        ratio[s] = pd.Series(d["close"].to_numpy() / btc_c)

    U = R.Universe(daily, warm=60)
    base_sig = U.signals(20)

    def ratio_sig(N):
        out = {}
        for s, r in ratio.items():
            brk = (r > r.shift(1).rolling(N).max()).to_numpy().copy()
            brk[:60] = False
            out[s] = np.flatnonzero(brk).astype(np.int64)
        return out

    def ratio_allow(L):
        return {s: (r > ema(r, L)).fillna(False).to_numpy(bool) for s, r in ratio.items()}

    versions = [("BASE USDT N20", base_sig, None)]
    for N in RATIO_NS:
        versions.append((f"RATIO BRK N{N}", ratio_sig(N), None))
    for L in EMA_LS:
        al = ratio_allow(L)
        versions.append((f"BASE + RATIO>EMA{L}", {k: v[al[k][v]] for k, v in base_sig.items()}, al))

    emit("=" * 150)
    emit("ALT/BTC BREAKOUT - kya coin ki BTC ke muqable taqat behtar signal hai?")
    emit("=" * 150)
    emit(f"Coins: {len(daily)} (BTC bahar) | Period: {start.date()} -> {closes.index[-1].date()} | OOS {R.OOS_START.date()} se | "
         f"kharcha fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% + stop slip {STOP_SLIP*100:.2f}%")

    def cut(df):
        return df[df["t_in"] >= start].reset_index(drop=True) if len(df) else df

    HEAD = (f"{'Version':>22} | {'Trades':>6} | {'Win%':>5} | {'PF':>5} | {'RndPF':>5} | {'PF+':>5} | {'OOS':>5} | {'RndOOS':>6} | "
            f"{'CAGR':>7} | {'MaxDD':>7} | {'MedCoin':>7} | {'Coins>1':>7} | " + " | ".join(str(y) for y in YEARS))
    results = {}
    for kind, lab in (("trail", "ATR TRAILING 3x"), ("donch", "DONCHIAN 10-DIN LOW EXIT")):
        emit(f"\n\n{'#' * 150}\nEXIT: {lab}\n{'#' * 150}")
        emit(HEAD)
        for name, sig, al in versions:
            if kind == "trail":
                tr = U.run(sig, M_STOP, 0.0, True, 365)
                rt = U.run(sig, M_STOP, 0.0, True, 365, rng=np.random.default_rng(7), allow=al)
            else:
                tr = S.run_donch(U, sig, 10)
                rt = S.run_donch(U, sig, 10, rng=np.random.default_rng(7), allow=al)
            tr, rt = cut(tr), cut(rt)
            if len(tr) < 10:
                emit(f"{name:>22} | trades nahi")
                continue
            r, rr = tr["ret"].to_numpy(), rt["ret"].to_numpy()
            oo = tr[tr["t_in"] >= R.OOS_START]["ret"].to_numpy()
            ro = rt[rt["t_in"] >= R.OOS_START]["ret"].to_numpy()
            eq, _ = portfolio(tr.assign(prio=0.0).to_dict("records"), closes, "risk", 0.01)
            st = stats(eq)
            cpf = tr.groupby("sym")["ret"].agg(lambda x: pf_of(x.to_numpy()) if len(x) >= 5 else np.nan).dropna().replace(np.inf, 5.0)
            yr = {y: pf_of(tr[tr["t_in"].dt.year == y]["ret"].to_numpy()) for y in YEARS}
            reg = btc_reg.reindex(tr["t_in"].dt.floor("1D") - pd.Timedelta(days=1)).to_numpy()
            res = dict(n=len(r), pf=pf_of(r), rpf=pf_of(rr), oos=pf_of(oo), roos=pf_of(ro), cagr=st["cagr"], dd=st["dd"],
                       yr=yr, regs={g: (pf_of(r[reg == g]), int((reg == g).sum())) for g in ("bull", "bear", "sideways")})
            results[(kind, name)] = res
            emit(f"{name:>22} | {res['n']:>6} | {(r > 0).mean()*100:>5.1f} | {fmt(res['pf']):>5} | {fmt(res['rpf']):>5} | "
                 f"{res['pf'] - res['rpf']:>+5.2f} | {fmt(res['oos']):>5} | {fmt(res['roos']):>6} | {res['cagr']*100:>+6.1f}% | "
                 f"{res['dd']*100:>6.1f}% | {cpf.median():>7.2f} | {(cpf > 1).mean()*100:>6.0f}% | "
                 + " | ".join(f"{fmt(yr[y]):>4}" for y in YEARS))
        emit("\nBTC regimes (PF, trades):")
        for name, _, _ in versions:
            if (kind, name) in results:
                rg = results[(kind, name)]["regs"]
                emit(f"{name:>22} | " + " | ".join(f"{g}: {fmt(rg[g][0])} ({rg[g][1]})" for g in ("bull", "bear", "sideways")))

    emit("\n" + "=" * 150)
    emit("KHULASA - kya ALT/BTC base se har lihaz se behtar hai, aur plateau hai?")
    emit("=" * 150)
    for kind, lab in (("trail", "ATR trail"), ("donch", "Donchian 10 exit")):
        b = results[(kind, "BASE USDT N20")]
        emit(f"\n{lab}: BASE -> PF {fmt(b['pf'])} (random {fmt(b['rpf'])}), OOS {fmt(b['oos'])}, DD {b['dd']*100:.0f}%, "
             f"2022 {fmt(b['yr'][2022])}, 2025 {fmt(b['yr'][2025])}, n {b['n']}")
        for grp, names in (("RATIO BRK", [f"RATIO BRK N{N}" for N in RATIO_NS]),
                           ("RATIO>EMA filter", [f"BASE + RATIO>EMA{L}" for L in EMA_LS])):
            good = 0
            for nm in names:
                x = results.get((kind, nm))
                if x is None:
                    continue
                ok = (x["pf"] - x["rpf"] >= 0.15 and x["oos"] > 1.0 and x["oos"] >= b["oos"] and x["dd"] >= b["dd"] - 0.03
                      and x["yr"][2022] > b["yr"][2022] and x["yr"][2025] > b["yr"][2025] and x["n"] >= 150)
                good += ok
                emit(f"   {nm:>20}: PF {fmt(x['pf'])} vs random {fmt(x['rpf'])} ({x['pf']-x['rpf']:+.2f}) | OOS {fmt(x['oos'])} | "
                     f"DD {x['dd']*100:.0f}% | 2022 {fmt(x['yr'][2022])} | 2025 {fmt(x['yr'][2025])} | n {x['n']} -> {'BEHTAR' if ok else '-'}")
            emit(f"   {grp}: {good}/{len(names)} har shart par base se behtar")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
