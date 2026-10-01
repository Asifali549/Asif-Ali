"""
SECTOR ROTATION + DONCHIAN BREAKOUT (user ka idea, 2026-10-01)
==============================================================
Khayal: jo shoba (sector) is waqt sab se mazboot ho, sirf usi ke coins mein structure (Daily Donchian N=20
breakout) par trade lo. Base entry/exit pehle se sabit: close > pichle 20 din ka high -> agle din open;
stop 3 x ATR(14) + ATR trailing 3x. Kharcha 1x. 109 coins, Oct 2020 -> Sep 2026.

SECTORS (dasti label, aaj ki pehchaan - note: kuch narratives jaise AI baad mein bane):
  L1, PAY_POW (BTC/LTC/XRP/privacy/payments), L2_INFRA (L2 + interop/oracle), DEFI, AI_DEPIN, MEME.
  Jin coins ka shoba wazeh nahi (exchange tokens, RWA, gaming, naamaloom) -> is test se bahar.
Sector ki taqat (har BAND din, sirf maujood coins; shoba tabhi gina jab kam az kam 4 coins ka data ho):
  - momentum = shobe ke coins ka MEDIAN L-din return
  - breadth  = shobe ke coins ka % jo apni EMA50 se ooper
VERSIONS (har ek mein sirf EK cheez):
  BASE (sector coins, koi filter nahi)
  ROTATION: sirf top-K shobe (K = 1/2/3) by 30-din momentum; aur K=2 by 14 / 60 din
  SECTOR BREADTH: coin ke shobe ka breadth >= 50/60/70/80%  (user ka "70%+ coins up")
Random baseline: wahi allowed (coin, din) jodon mein random din -> kya breakout garam shobe ke andar bhi kuch deta hai?
Natija: sector_rotation_RESULTS.txt
"""
import numpy as np
import pandas as pd

import donchian_research as R
from bot_core import fetch_full, norm, STABLES, FEE, SLIP, STOP_SLIP
from portfolio_lab import ema, to_daily, portfolio, stats

OUT = "sector_rotation_RESULTS.txt"
M_STOP = 3.0
MIN_COINS = 4
pf_of, fmt = R.pf_of, R.fmt

SECTORS = {
    "L1": "ETH SOL AVAX ADA ALGO SUI NEAR APT DOT HBAR TRX ONE SEI ICP ATOM S MON XPL MOVR INJ SOMI CC NIGHT EGLD TON",
    "PAY_POW": "BTC LTC BCH BSV ETC XRP XLM DASH ZEC XMR KAS XPR BDX TEL ULTIMA ZBCN VELO",
    "L2_INFRA": "OP ARB STRK POL STX SOON LINK QNT ZRO AXL MNT IMX",
    "DEFI": "UNI AAVE CRV PENDLE SYN ENA RUNE AERO HYPE ASTER LIT MET APR SPK HOME JUP LDO MKR COMP SNX DYDX",
    "AI_DEPIN": "FET RENDER TAO VIRTUAL WLD GRASS AKT AR FIL KITE VVV TRAC NIL BIO LYN ALLO DATA JASMY PEAQ",
    "MEME": "PENGU PUMP USELESS FARTCOIN TRUMP COQ DOGE SHIB PEPE WIF BONK FLOKI",
}
SECTOR_OF = {c: s for s, cs in SECTORS.items() for c in cs.split()}


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
    sect = {s: SECTOR_OF.get(s.split("/")[0].upper()) for s in daily_all}
    daily = {s: d for s, d in daily_all.items() if sect[s]}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index()
    start = max(pd.Timestamp("2020-10-01"), closes.index[0] + pd.Timedelta(days=30))
    closes_ff = closes.ffill()
    port_closes = closes_ff[closes_ff.index >= start]

    # ---- sector taqat (band din; sirf maujood coins) ----
    hist = closes.notna().cumsum()
    above = (closes > closes.apply(lambda x: ema(x.dropna(), 50).reindex(x.index))) & (hist >= 50)
    valid50 = hist >= 50
    mom = {L: closes / closes.shift(L) - 1 for L in (14, 30, 60)}
    names = list(SECTORS)
    sec_cols = {n: [s for s in closes.columns if sect[s] == n] for n in names}
    breadth = pd.DataFrame({n: above[c].sum(axis=1) / valid50[c].sum(axis=1).replace(0, np.nan) for n, c in sec_cols.items()})
    sec_mom = {}
    for L, mtx in mom.items():
        df = pd.DataFrame({n: mtx[c].median(axis=1) for n, c in sec_cols.items()})
        cnt = pd.DataFrame({n: mtx[c].notna().sum(axis=1) for n, c in sec_cols.items()})
        sec_mom[L] = df.where(cnt >= MIN_COINS)
    breadth = breadth.where(pd.DataFrame({n: valid50[c].sum(axis=1) for n, c in sec_cols.items()}) >= MIN_COINS)

    def topk_allowed(L, K):
        rk = sec_mom[L].rank(axis=1, ascending=False)
        return rk <= K                                            # din x shoba (NaN -> False)

    U = R.Universe(daily, warm=60)
    sig_all = U.signals(20)

    def mk_allow(day_sector_ok):
        out = {}
        for s, d in daily.items():
            col = day_sector_ok[sect[s]] if sect[s] in day_sector_ok else pd.Series(False, index=day_sector_ok.index)
            out[s] = col.reindex(d["timestamp"]).fillna(False).to_numpy(bool)
        return out

    versions = [("BASE (koi filter nahi)", None)]
    for K in (1, 2, 3):
        versions.append((f"ROTATION top-{K} (30d)", mk_allow(topk_allowed(30, K))))
    for L in (14, 60):
        versions.append((f"ROTATION top-2 ({L}d)", mk_allow(topk_allowed(L, 2))))
    for t in (0.5, 0.6, 0.7, 0.8):
        versions.append((f"SECTOR BREADTH >= {int(t*100)}%", mk_allow(breadth >= t)))

    bd = daily_all["BTC/USDT"].set_index("timestamp")["close"]
    b50, b200 = ema(bd, 50), ema(bd, 200)
    btc_reg = pd.Series(np.where((bd > b200) & (b50 > b200), "bull", np.where((bd < b200) & (b50 < b200), "bear", "sideways")),
                        index=bd.index)

    emit("=" * 150)
    emit("SECTOR ROTATION + DAILY DONCHIAN N=20 (ATR stop 3x + ATR trailing 3x)")
    emit("=" * 150)
    emit(f"Sector coins: {len(daily)} / {len(daily_all)} | Period: {start.date()} -> {port_closes.index[-1].date()} | "
         f"kharcha fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% + stop slip {STOP_SLIP*100:.2f}% | OOS {R.OOS_START.date()} se")
    emit("Shobe (coins jo test mein aaye): " + " | ".join(f"{n} {len(c)}" for n, c in sec_cols.items()))
    sm = sec_mom[30][sec_mom[30].index >= start].dropna(how="all")
    switches = sm.idxmax(axis=1)
    top1 = switches.value_counts(normalize=True)
    emit("Kitne % din kaun sa shoba #1 (30d momentum): " + " | ".join(f"{k} {v*100:.0f}%" for k, v in top1.items()))
    emit(f"#1 shoba kitni baar badla: {int((switches != switches.shift()).sum())} baar (ausat {len(switches)/max((switches != switches.shift()).sum(),1):.0f} din tak tikta)")

    def cut(df):
        return df[df["t_in"] >= start].reset_index(drop=True) if len(df) else df

    YEARS = list(range(2021, 2027))
    emit("\n" + f"{'Version':>26} | {'Trades':>6} | {'Win%':>5} | {'PF':>5} | {'RndPF':>5} | {'PF+':>5} | {'OOS':>5} | {'RndOOS':>6} | "
         f"{'CAGR':>7} | {'MaxDD':>7} | {'MedCoin':>7} | {'Coins>1':>7} | " + " | ".join(str(y) for y in YEARS))
    res = {}
    for name, al in versions:
        sig = sig_all if al is None else {k: v[al[k][v]] for k, v in sig_all.items()}
        tr = cut(U.run(sig, M_STOP, 0.0, True, 365))
        rt = cut(U.run(sig, M_STOP, 0.0, True, 365, rng=np.random.default_rng(7), allow=al))
        if len(tr) < 10:
            emit(f"{name:>26} | trades nahi")
            continue
        r, rr = tr["ret"].to_numpy(), rt["ret"].to_numpy()
        oo = tr[tr["t_in"] >= R.OOS_START]["ret"].to_numpy()
        ro = rt[rt["t_in"] >= R.OOS_START]["ret"].to_numpy()
        eq, _ = portfolio(tr.assign(prio=0.0).to_dict("records"), port_closes, "risk", 0.01)
        st = stats(eq)
        cpf = tr.groupby("sym")["ret"].agg(lambda x: pf_of(x.to_numpy()) if len(x) >= 5 else np.nan).dropna().replace(np.inf, 5.0)
        yr = {y: pf_of(tr[tr["t_in"].dt.year == y]["ret"].to_numpy()) for y in YEARS}
        reg = btc_reg.reindex(tr["t_in"].dt.floor("1D") - pd.Timedelta(days=1)).to_numpy()
        s_ = tr["sym"].map(sect).to_numpy()
        res[name] = dict(n=len(r), pf=pf_of(r), rpf=pf_of(rr), oos=pf_of(oo), roos=pf_of(ro), cagr=st["cagr"], dd=st["dd"], yr=yr,
                         regs={g: (pf_of(r[reg == g]), int((reg == g).sum())) for g in ("bull", "bear", "sideways")},
                         secs={n: (pf_of(r[s_ == n]), int((s_ == n).sum())) for n in names})
        emit(f"{name:>26} | {len(r):>6} | {(r>0).mean()*100:>5.1f} | {fmt(pf_of(r)):>5} | {fmt(pf_of(rr)):>5} | "
             f"{pf_of(r)-pf_of(rr):>+5.2f} | {fmt(pf_of(oo)):>5} | {fmt(pf_of(ro)):>6} | {st['cagr']*100:>+6.1f}% | {st['dd']*100:>6.1f}% | "
             f"{cpf.median():>7.2f} | {(cpf > 1).mean()*100:>6.0f}% | " + " | ".join(f"{fmt(yr[y]):>4}" for y in YEARS))

    emit("\nBTC regimes (PF, trades):")
    for name in res:
        emit(f"{name:>26} | " + " | ".join(f"{g}: {fmt(res[name]['regs'][g][0])} ({res[name]['regs'][g][1]})" for g in ("bull", "bear", "sideways")))
    emit("\nShoba-war PF (trades):")
    for name in res:
        emit(f"{name:>26} | " + " | ".join(f"{n}: {fmt(res[name]['secs'][n][0])} ({res[name]['secs'][n][1]})" for n in names))

    emit("\n" + "=" * 150)
    emit("KHULASA - kya rotation / sector breadth base se behtar hai, aur plateau hai?")
    emit("=" * 150)
    b = res["BASE (koi filter nahi)"]
    emit(f"BASE: PF {fmt(b['pf'])} (random {fmt(b['rpf'])}) | OOS {fmt(b['oos'])} | DD {b['dd']*100:.0f}% | 2022 {fmt(b['yr'][2022])} | 2025 {fmt(b['yr'][2025])}")
    for grp, keys in (("ROTATION", [k for k in res if k.startswith("ROTATION")]),
                      ("SECTOR BREADTH", [k for k in res if k.startswith("SECTOR")])):
        good = 0
        for k in keys:
            x = res[k]
            ok = (x["pf"] - x["rpf"] >= 0.15 and x["oos"] > 1.0 and x["oos"] >= b["oos"] and x["dd"] >= b["dd"] - 0.03
                  and x["yr"][2022] >= b["yr"][2022] and x["yr"][2025] >= b["yr"][2025] and x["n"] >= 150)
            good += ok
            emit(f"   {k:>26}: PF {fmt(x['pf'])} vs random {fmt(x['rpf'])} ({x['pf']-x['rpf']:+.2f}) | OOS {fmt(x['oos'])} | "
                 f"DD {x['dd']*100:.0f}% | 2022 {fmt(x['yr'][2022])} | 2025 {fmt(x['yr'][2025])} | n {x['n']} -> {'BEHTAR' if ok else '-'}")
        emit(f"   {grp}: {good}/{len(keys)} versions har shart par base se behtar\n")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
