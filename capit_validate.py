"""
VOLUME CAPITULATION - VALIDATION (2026-10-02)
=============================================
Rule (new_ideas C): coin close > EMA200, din ka return <= -R, volume >= V x pichle 20 din ka ausat, top-100 liquid
-> agle din OPEN par khareed; stop = signal close - 3 ATR(14) fixed; exit close > SMA5 (agle din open) ya 10 din.
Pehla test: 7/9 grid PASS, PF 1.55-1.89 vs random p50 ~1.0, OOS 1.5-2.1, ICHI/DIP se correlation ~0.
Yahan (R8% V2.0 aur R8% V2.5 - plateau ke beech):
 1) Bootstrap p5 PF (2000 baar)   2) Top-10 trades hata kar PF   3) 4 time-folds PF
 4) Coin concentration (top 10% coins ka nafa mein hissa) + median coin PF
 5) Kharcha 2x / 3x   6) BTC regime (bull/bear/sideways)   7) DIP se overlap (usi coin mein +-3 din DIP signal)
 8) Portfolio: C ka size (10/20%) aur hissa - ICHI/DIP/C mixes vs ICHI 60 / DIP 40
Natija: capit_validate_RESULTS.txt
"""
import warnings

import numpy as np
import pandas as pd

import portfolio_lab as P
from bot_core import fetch_full, norm, STABLES

warnings.filterwarnings("ignore")
OUT = "capit_validate_RESULTS.txt"
OOS = pd.Timestamp("2025-01-01")
CONFIGS = [(0.08, 2.0), (0.08, 2.5)]


def pf_of(r):
    r = np.asarray(r, float)
    g, lo = r[r > 0].sum(), -r[r < 0].sum()
    return np.nan if len(r) == 0 else (np.inf if lo == 0 else g / lo)


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x)
        lines.append(x)

    if h4 is None:
        from data_fetcher import get_exchange, get_coin_list
        ex = get_exchange()
        coins = [s for s in get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:P.TOP_N]
        for must in ("ETH/USDT", "BTC/USDT"):
            if must not in coins:
                coins.insert(0, must)
        h4 = {}
        for k, sym in enumerate(coins, 1):
            try:
                df = fetch_full(ex, sym, "4h", P.H4_BARS)
                if df is not None and len(df) >= 6 * 250:
                    h4[sym] = norm(df)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)}")
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})")

    daily = {s: P.to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc_ok = bd > P.ema(bd, 50)
    b50, b200 = P.ema(bd, 50), P.ema(bd, 200)
    btc_reg = pd.Series(np.where((bd > b200) & (b50 > b200), "bull", np.where((bd < b200) & (b50 < b200), "bear", "sideways")), index=bd.index)
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    allowed = {day: set(row.dropna().nlargest(P.UNIVERSE).index) for day, row in dv.iterrows()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes_p = closes[closes.index >= start]
    weeks = (closes_p.index[-1] - closes_p.index[0]).days / 7

    A = {}
    for s, d in daily.items():
        if s == "BTC/USDT" or len(d) < 260:
            continue
        ts, c = d["timestamp"], d["close"]
        a = dict(o=d["open"].to_numpy(float), h=d["high"].to_numpy(float), l=d["low"].to_numpy(float), c=c.to_numpy(float),
                 ts=ts.to_numpy(), stop=(c - 3.0 * P.atr_w(d)).to_numpy(), ret=c.pct_change().to_numpy(),
                 up=(c > P.ema(c, 200)).to_numpy() & (np.arange(len(c)) >= 200),
                 vrat=(d["volume"] / d["volume"].shift(1).rolling(20).mean()).to_numpy(),
                 ex=(c > c.rolling(5).mean()).to_numpy(), uni=np.array([s in allowed.get(D, ()) for D in ts]))
        a["uni"][:210] = False
        A[s] = a

    def trades(R, V, slip_mult=1.0):
        old = (P.SLIP, P.FEE, P.STOP_SLIP)
        P.SLIP, P.FEE, P.STOP_SLIP = old[0] * slip_mult, old[1] * slip_mult, old[2] * slip_mult
        out = []
        try:
            for s, a in A.items():
                m = a["uni"] & a["up"] & (np.nan_to_num(a["ret"]) <= -R) & (np.nan_to_num(a["vrat"]) >= V)
                busy = -1
                for i in np.flatnonzero(m):
                    if i <= busy:
                        continue
                    t = P.sim_one(a["o"], a["h"], a["l"], a["c"], a["ts"], i, a["stop"][i], None, a["ex"], None, 10)
                    if t:
                        t.update(sym=s, prio=float(-a["ret"][i]), sig_day=pd.Timestamp(a["ts"][i]))
                        out.append(t)
                        busy = int(np.searchsorted(a["ts"], t["t_out"]))
        finally:
            P.SLIP, P.FEE, P.STOP_SLIP = old
        return [t for t in out if pd.Timestamp(t["t_in"]) >= start]

    emit("=" * 130)
    emit("VOLUME CAPITULATION - VALIDATION")
    emit("=" * 130)
    emit(f"Coins: {len(A)} | {closes_p.index[0].date()} -> {closes_p.index[-1].date()} | top-{P.UNIVERSE} point-in-time")

    dip_tr = [t for t in P.trades_dip(daily, allowed, btc_ok) if pd.Timestamp(t["t_in"]) >= start]
    dip_days = {}
    for t in dip_tr:
        dip_days.setdefault(t["sym"], []).append(pd.Timestamp(t["t_in"]))

    rng = np.random.default_rng(11)
    for R, V in CONFIGS:
        tr = trades(R, V)
        r = np.array([t["ret"] for t in tr])
        tin = pd.to_datetime([t["t_in"] for t in tr])
        emit(f"\n{'#' * 130}\nR{R*100:.0f}% V{V}: {len(r)} trades ({len(r)/weeks:.2f}/hafta) | win {(r > 0).mean()*100:.1f}% | "
             f"avg {r.mean()*100:+.2f}% | PF {pf_of(r):.2f} | OOS PF {pf_of(r[tin >= OOS]):.2f}\n{'#' * 130}")
        # 1 bootstrap
        bs = [pf_of(rng.choice(r, len(r), replace=True)) for _ in range(2000)]
        emit(f"1) Bootstrap PF: p5 {np.percentile(bs, 5):.2f} | p50 {np.median(bs):.2f} -> {'PASS' if np.percentile(bs, 5) > 1 else 'FAIL'}")
        # 2 top-10 removed
        rs = np.sort(r)
        emit(f"2) Top-10 trades hata kar PF: {pf_of(rs[:-10]):.2f} | top-20 hata kar: {pf_of(rs[:-20]):.2f} -> "
             f"{'PASS' if pf_of(rs[:-10]) > 1 else 'FAIL'}")
        # 3 folds
        order = np.argsort(tin.values)
        folds = []
        for part in np.array_split(order, 4):
            folds.append((pd.Timestamp(tin.values[part].min()).date(), pf_of(r[part]), len(part)))
        emit("3) 4 time-folds: " + " | ".join(f"{d}: PF {p:.2f} ({n})" for d, p, n in folds)
             + f" -> {sum(p > 1 for _, p, _ in folds)}/4 musbat")
        # 4 concentration
        df = pd.DataFrame({"sym": [t["sym"] for t in tr], "ret": r})
        per = df.groupby("sym")["ret"].agg(["sum", "count"])
        cpf = df.groupby("sym")["ret"].agg(lambda x: pf_of(x.to_numpy()) if len(x) >= 3 else np.nan).dropna().replace(np.inf, 5)
        pos = per["sum"][per["sum"] > 0].sort_values(ascending=False)
        k10 = max(1, int(round(len(per) * 0.10)))
        share = pos.head(k10).sum() / per["sum"].sum() if per["sum"].sum() > 0 else np.nan
        emit(f"4) Coins: {len(per)} | nafa wale {int((per['sum'] > 0).sum())}, nuqsan wale {int((per['sum'] < 0).sum())} | "
             f"top 10% ({k10}) coins = net nafa ka {share*100:.0f}% | median coin PF (3+ trades) {cpf.median():.2f}")
        # 5 costs
        emit("5) Kharcha: " + " | ".join(f"{m}x PF {pf_of([t['ret'] for t in trades(R, V, m)]):.2f}" for m in (1, 2, 3)))
        # 6 regime
        reg = btc_reg.reindex(tin.floor("1D") - pd.Timedelta(days=1)).to_numpy()
        emit("6) BTC regime: " + " | ".join(f"{g}: PF {pf_of(r[reg == g]):.2f} ({int((reg == g).sum())})" for g in ("bull", "bear", "sideways")))
        # 7 overlap with DIP
        ov = 0
        for t in tr:
            ds = dip_days.get(t["sym"], [])
            if any(abs((d - pd.Timestamp(t["t_in"])).days) <= 3 for d in ds):
                ov += 1
        emit(f"7) DIP ke sath overlap (usi coin mein +-3 din DIP trade): {ov}/{len(tr)} = {ov/len(tr)*100:.0f}%")
        # yearly
        emit("   Saal-war PF: " + " | ".join(f"{y}: {pf_of(r[tin.year == y]):.2f} ({int((tin.year == y).sum())})" for y in range(2021, 2027)))
        # OOS tajziya: kaun se coins / mahine
        oo = df.assign(t_in=tin)[tin >= OOS]
        emit(f"   OOS trades {len(oo)} | PF {pf_of(oo['ret'].to_numpy()):.2f} | mahana PF: " + " | ".join(
            f"{k}: {pf_of(g['ret'].to_numpy()):.2f}({len(g)})" for k, g in oo.groupby(oo['t_in'].dt.strftime('%y-%m'))))
        worst = oo.sort_values("ret").head(8)
        emit("   OOS sab se bure: " + " | ".join(f"{a.sym.split('/')[0]} {a.t_in.date()} {a.ret*100:+.0f}%" for a in worst.itertuples()))
        emit(f"   Coin list (aaj ki top-150) mein coins: {len(A)} -> " + ",".join(sorted(x.split('/')[0] for x in A))[:900])

    # 8 portfolio
    emit(f"\n{'#' * 130}\n8) PORTFOLIO (R8% V2.0) - C ka size aur hissa\n{'#' * 130}")
    T = {"ICHI": [t for t in P.trades_ichi(h4, allowed) if pd.Timestamp(t["t_in"]) >= start], "DIP": dip_tr}
    trc = trades(0.08, 2.0)
    curves = {"ICHI": P.portfolio(T["ICHI"], closes_p, "risk", 0.01)[0], "DIP": P.portfolio(T["DIP"], closes_p, "fixed", 0.20)[0],
              "C10": P.portfolio(trc, closes_p, "fixed", 0.10)[0], "C20": P.portfolio(trc, closes_p, "fixed", 0.20)[0]}
    mixes = {"BASE ICHI 60 / DIP 40": {"ICHI": .6, "DIP": .4},
             "ICHI 60 / DIP 30 / C20 10": {"ICHI": .6, "DIP": .3, "C20": .1},
             "ICHI 60 / DIP 20 / C20 20": {"ICHI": .6, "DIP": .2, "C20": .2},
             "ICHI 50 / DIP 30 / C20 20": {"ICHI": .5, "DIP": .3, "C20": .2},
             "ICHI 50 / DIP 25 / C20 25": {"ICHI": .5, "DIP": .25, "C20": .25},
             "ICHI 50 / DIP 30 / C10 20": {"ICHI": .5, "DIP": .3, "C10": .2}}
    emit(f"{'Portfolio':>28} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'+mahine':>7} | {'Bura mah':>8} | "
         + " | ".join(str(y) for y in range(2021, 2027)))
    for name, eq in list(curves.items()) + [(n, P.mix(curves, w)) for n, w in mixes.items()]:
        s = P.stats(eq)
        yr = {d.year: v for d, v in s["yearly"].items()}
        emit(f"{name:>28} | {s['cagr']*100:>+6.1f}% | {s['dd']*100:>6.1f}% | {s['sharpe']:>6.2f} | {s['pos_months']:>6.0f}% | "
             f"{s['worst_month']*100:>+7.1f}% | " + " | ".join(f"{yr.get(y, np.nan)*100:>+4.0f}%" for y in range(2021, 2027)))
    mo = pd.DataFrame({k: P.stats(v)["monthly"] for k, v in curves.items() if k != "C10"})
    emit("\nMahana correlation:\n" + mo.corr().round(2).to_string())

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
