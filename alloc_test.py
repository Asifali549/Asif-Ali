"""
ALLOC TEST - naye systems ke sath sarmaye ki taqseem (user, 2026-10-04)
=========================================================================
Naye usool:  Ichimoku TP5 (CE 16/4 + TP 5%, 2% risk, top-200)  +  Dip v2 (RSI3<7, SMA3, TP5, 20% fixed, top-100)
Sath mein (sirf paper wale, naye usool): Donchian maxSL 20% (1% risk), Capitulation 2% risk; purana Ichimoku (muqable ke liye).
Mixes: TP5/DIP 100/0 .. 50/50, teen-system mixes, purana ICHI60/DIP40 (hawala). Har mahine rebalance.
Natija: alloc_test_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema, to_daily, portfolio, stats, mix, H4_BARS
from winrate_lab import prep_ichi, run_ichi, tstats
from dip_v2_validate import prep_dip, run_dip
import stop_fix_lab as SF

OUT = "alloc_test_RESULTS.txt"
TOP_FETCH = 260


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    if h4 is None:
        from data_fetcher import get_exchange, get_coin_list
        ex = get_exchange()
        coins = [s for s in get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:TOP_FETCH]
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
    al200 = {day: set(row[row <= 200].index) for day, row in rank.iterrows()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes = closes[closes.index >= start]

    emit("=" * 110)
    emit("ALLOC TEST - naye systems (Ichimoku TP5 + Dip v2) ke sath sarmaye ki taqseem")
    emit("=" * 110)
    emit(f"Coins: {len(daily)} | {closes.index[0].date()} -> {closes.index[-1].date()} | har mahine rebalance | win = +0.5% se ziada")

    tr = {}
    tr["TP5"] = (run_ichi(prep_ichi(h4, al200, start), {"ce": 4.0, "tp": 0.05}), ("risk", 0.02))
    tr["ICHI purana"] = (run_ichi(prep_ichi(h4, al100, start), {"tpR": 3.0}), ("risk", 0.01))
    tr["DIP v2"] = (run_dip(prep_dip(daily, al100, btc_ok, start), 7, 3, 0.05), ("fixed", 0.20))
    P = SF.prep(daily, al100, btc_ok, start)
    tr["DON maxSL20"] = (SF.run(P, "DON", 4.0, None, 0.20), ("risk", 0.01))
    tr["CAPIT 2%"] = (SF.run(P, "CAPIT", 3.0, None, None), ("risk", 0.02))

    curves = {}
    emit("\n# AKELE SYSTEMS")
    emit(f"{'System':>14} | {'trades':>6} | {'win%':>5} | {'PF':>5} | {'OOS':>5}")
    for k, (t, (mode, size)) in tr.items():
        s = tstats(t)
        curves[k] = portfolio(t, closes, mode, size)[0]
        emit(f"{k:>14} | {s['n']:>6} | {s['win']:>5.1f} | {s['pf']:>5.2f} | {s['oos']:>5.2f}")

    mo = pd.DataFrame({k: c.resample("ME").last().pct_change() for k, c in curves.items()}).dropna()
    emit("\n# MAHANA CORRELATION (kam = behtar diversification)")
    emit(mo.corr().round(2).to_string())

    MIX = {"TP5 100": {"TP5": 1.0}}
    for a in (90, 80, 70, 60, 50):
        MIX[f"TP5 {a} / DIP {100 - a}"] = {"TP5": a / 100, "DIP v2": (100 - a) / 100}
    MIX["TP5 70 / DIP 20 / CAPIT 10"] = {"TP5": .7, "DIP v2": .2, "CAPIT 2%": .1}
    MIX["TP5 70 / DIP 20 / DON 10"] = {"TP5": .7, "DIP v2": .2, "DON maxSL20": .1}
    MIX["TP5 60 / DIP 20 / CAPIT 10 / DON 10"] = {"TP5": .6, "DIP v2": .2, "CAPIT 2%": .1, "DON maxSL20": .1}
    MIX["HAWALA: purana ICHI 60 / DIP v2 40"] = {"ICHI purana": .6, "DIP v2": .4}
    for k in ("DIP v2", "DON maxSL20", "CAPIT 2%", "ICHI purana"):
        MIX[f"akela: {k}"] = {k: 1.0}

    years = list(range(2021, 2027))
    emit("\n# PORTFOLIO ($1000 se)")
    emit(f"{'Mix':>36} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'+mah%':>5} | {'bura mah':>8} | {'$1000 ->':>9} | "
         + " | ".join(f"{y:>6}" for y in years))
    res = []
    for name, w in MIX.items():
        eq = mix(curves, w)
        p = stats(eq)
        yr = {d.year: v for d, v in p["yearly"].items()}
        res.append((name, p))
        emit(f"{name:>36} | {p['cagr']*100:>+6.1f}% | {p['dd']*100:>6.1f}% | {p['sharpe']:>6.2f} | {p['pos_months']:>5.0f} | "
             f"{p['worst_month']*100:>7.1f}% | {1000*eq.iloc[-1]/eq.iloc[0]:>9,.0f} | "
             + " | ".join(f"{yr.get(y, np.nan)*100:>+5.0f}%" for y in years))

    best = max(res, key=lambda r: r[1]["sharpe"])
    emit(f"\nSab se ooncha Sharpe: {best[0]} ({best[1]['sharpe']:.2f})")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
