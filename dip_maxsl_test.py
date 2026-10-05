"""
DIP MAX-SL TEST - Dip v2 ka SL bohat door ho to kya karein? (user, 2026-10-05)
==============================================================================
Live misaal: BR/USDT entry 0.5256, SL 0.0146 (97% neeche) - 3 x ATR naye volatile coin par.
Dip v2 = RSI3<7 uptrend, exit close>SMA3 / TP +5% / 10 din, stop signal close - 3 ATR, top-100, 20% size.
Do tareeqe:
  A) maxSL filter: SL entry se X% se ziada door ho to signal CHHOR do (X = 30/40/50/60/70%)
  B) size chhota: size = min(R / SL-faasla, 20%) - door SL par chhoti position (R = 4/6/8% account)
Har cell: trades, win, PF, OOS, 1 trade ka sab se bara account nuqsan, CAGR, MaxDD, Sharpe, bura mahina, saal-war.
Pehle: Dip trades mein SL faasle ki taqseem. Akhir mein behtareen ke liye random control + TP5 70 / Dip 30 portfolio.
Natija: dip_maxsl_test_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema, to_daily, portfolio, stats, mix, H4_BARS
from winrate_lab import prep_ichi, run_ichi, tstats, pf_of
import stop_fix_lab as SF

OUT = "dip_maxsl_test_RESULTS.txt"
TOP_FETCH = 330


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    if h4 is None:
        import data_fetcher
        ex = data_fetcher.get_exchange()
        coins = [s for s in data_fetcher.get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:TOP_FETCH]
        for must in ("ETH/USDT", "BTC/USDT"):
            if must not in coins:
                coins.insert(0, must)
        h4 = {}
        for k, sym in enumerate(coins, 1):
            try:
                df = fetch_full(ex, sym, "4h", H4_BARS)
                if df is not None and len(df) >= 6 * 120:
                    h4[sym] = norm(df)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)}")
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})")

    daily = {s: to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc_ok = bd > ema(bd, 50)
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    rank = dv.rank(axis=1, ascending=False)
    al100 = {day: set(row[row <= 100].index) for day, row in rank.iterrows()}
    al250 = {day: set(row[row <= 250].index) for day, row in rank.iterrows()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes = closes[closes.index >= start]
    P = SF.prep(daily, al100, btc_ok, start)

    emit("=" * 125)
    emit("DIP MAX-SL TEST - Dip v2 (RSI3<7, SMA3, TP5, stop 3 ATR, top-100): door SL ka ilaaj")
    emit("=" * 125)
    emit(f"Coins: {len(daily)} | {closes.index[0].date()} -> {closes.index[-1].date()} | win = +0.5% se ziada | OOS = 2025+")

    base = SF.run(P, "DIP", 3.0, 0.05, None)
    sl = np.array([t["risk"] for t in base]) * 100
    emit(f"\n# SL KA FAASLA (entry se, %) - {len(base)} Dip trades")
    emit("percentile: " + " | ".join(f"p{q} {np.percentile(sl, q):.0f}%" for q in (10, 25, 50, 75, 90, 95, 99)) + f" | max {sl.max():.0f}%")
    for lo, hi in ((0, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 101)):
        sub = [t for t in base if lo <= t["risk"] * 100 < hi]
        if sub:
            s = tstats(sub)
            emit(f"  SL {lo:>2}-{hi:<3}%: {s['n']:>3} trades | win {s['win']:>5.1f}% | PF {s['pf']:>5.2f} | avg {s['avg']:+.2f}% | "
                 f"sab se buri trade {min(t['ret'] for t in sub)*100:+.1f}%")

    years = list(range(2021, 2027))
    emit(f"\n{'tareeqa':>22} | {'n':>4} | {'win%':>5} | {'PF':>5} | {'OOS':>5} | {'1 trade max':>11} | {'CAGR':>7} | {'MaxDD':>7} | "
         f"{'Sharpe':>6} | {'bura mah':>8} | " + " | ".join(f"{y:>5}" for y in years))
    cells = [("LIVE: koi had nahi", None, ("fixed", 0.20))]
    cells += [(f"A: maxSL {int(x*100)}%", x, ("fixed", 0.20)) for x in (0.30, 0.40, 0.50, 0.60, 0.70)]
    cells += [(f"B: size R{int(r*100)}% cap20", None, ("risk", r)) for r in (0.04, 0.06, 0.08)]
    cells += [("A50 + B R6%", 0.50, ("risk", 0.06))]
    curves, res = {}, []
    for name, mx, (mode, size) in cells:
        tr = SF.run(P, "DIP", 3.0, 0.05, mx)
        s = tstats(tr)
        eq = portfolio(tr, closes, mode, size)[0]
        p = stats(eq)
        wa = SF.worst_acct(tr, mode, size)
        yr = {d.year: v for d, v in p["yearly"].items()}
        curves[name] = eq
        res.append((name, mx, mode, size, s, p))
        emit(f"{name:>22} | {s['n']:>4} | {s['win']:>5.1f} | {s['pf']:>5.2f} | {s['oos']:>5.2f} | {wa:>10.1f}% | {p['cagr']*100:>+6.1f}% | "
             f"{p['dd']*100:>6.1f}% | {p['sharpe']:>6.2f} | {p['worst_month']*100:>7.1f}% | "
             + " | ".join(f"{yr.get(y, np.nan)*100:>+4.0f}%" for y in years))

    emit("\n# RANDOM CONTROL (8 seeds, wohi stop/TP/filter) - sirf entry ka edge")
    for name, mx, mode, size, s, p in res:
        pfs = [pf_of([t["ret"] for t in SF.run(P, "DIP", 3.0, 0.05, mx, rng=np.random.default_rng(70 + q))]) for q in range(8)]
        emit(f"{name:>22}: PF {s['pf']:.2f} vs random p95 {np.percentile(pfs, 95):.2f} -> {'behtar' if s['pf'] > np.percentile(pfs, 95) else 'NAHI'}")

    emit("\n# PORTFOLIO: TP5 (top-250, 2% risk) 70% + Dip 30%")
    curves["TP5"] = portfolio(run_ichi(prep_ichi(h4, al250, start), {"ce": 4.0, "tp": 0.05}), closes, "risk", 0.02)[0]
    for name, *_ in res:
        eq = mix(curves, {"TP5": .7, name: .3})
        p = stats(eq)
        emit(f"{'TP5 70 / ' + name:>32} | CAGR {p['cagr']*100:+.1f}% | MaxDD {p['dd']*100:.1f}% | Sharpe {p['sharpe']:.2f} | "
             f"bura mahina {p['worst_month']*100:.1f}%")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
