"""
TP MARGIN TEST - TP5 ka nafa fi trade kam lagta hai: bara TP ya bara size behtar? (user, 2026-10-04)
=====================================================================================================
Ichimoku (CE 16/4 stop), top-200 coins. TP 4/5/6/7/8/10% aur trailing (TP nahi) x risk 2%/3% x coin cap 20%/30%.
Har cell: win%, PF, OOS, SL tak ausat faasla, ausat position size, $1000 par ek jeet ka ausat $, CAGR, MaxDD, Sharpe,
bura mahina, saal-war. Phir behtareen ke sath Dip v2 30%.
Natija: tp_margin_test_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema, to_daily, portfolio, stats, mix, H4_BARS
from winrate_lab import prep_ichi, run_ichi, tstats
from dip_v2_validate import prep_dip, run_dip

OUT = "tp_margin_test_RESULTS.txt"
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

    emit("=" * 130)
    emit("TP MARGIN TEST - Ichimoku (CE 16/4), top-200: TP kitna? size kitna?")
    emit("=" * 130)
    emit(f"Coins: {len(daily)} | {closes.index[0].date()} -> {closes.index[-1].date()} | win = +0.5% se ziada | OOS = 2025+")
    emit("'SL door' = entry se SL ka ausat faasla | 'size' = ausat position (account ka %) | '$/jeet' = $1000 account par ek jeetne wali trade ka ausat nafa")

    idata = prep_ichi(h4, al200, start)
    years = list(range(2021, 2027))
    emit(f"\n{'TP':>6} | {'risk':>4} | {'cap':>3} | {'n':>4} | {'win%':>5} | {'PF':>5} | {'OOS':>5} | {'SL door':>7} | {'size':>5} | "
         f"{'$/jeet':>6} | {'$/haar':>6} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'bura mah':>8} | " + " | ".join(f"{y:>5}" for y in years))
    curves, res = {}, []
    for tp in (0.04, 0.05, 0.06, 0.07, 0.08, 0.10, None):
        cfg = {"ce": 4.0, "tp": tp} if tp else {"ce": 4.0}
        tr = run_ichi(idata, cfg)
        s = tstats(tr)
        sl = np.mean([t["risk"] for t in tr]) * 100
        for risk in (0.02, 0.03):
            for cap in (0.20, 0.30):
                eq, taken = portfolio(tr, closes, "risk", risk, cap=cap)
                p = stats(eq)
                sz = np.array([min(risk / max(t["risk"], 1e-6), cap) for t in taken])
                rr = np.array([t["ret"] for t in taken])
                w = sz[rr > 0.005] * rr[rr > 0.005] * 1000
                l = sz[rr < 0] * rr[rr < 0] * 1000
                yr = {d.year: v for d, v in p["yearly"].items()}
                name = f"TP{int(tp*100)}" if tp else "trail"
                key = f"{name} r{int(risk*100)} c{int(cap*100)}"
                curves[key] = eq
                res.append((key, p))
                emit(f"{name:>6} | {risk*100:>3.0f}% | {cap*100:>2.0f}% | {s['n']:>4} | {s['win']:>5.1f} | {s['pf']:>5.2f} | {s['oos']:>5.2f} | "
                     f"{sl:>6.1f}% | {sz.mean()*100:>4.1f}% | {w.mean() if len(w) else 0:>6.1f} | {l.mean() if len(l) else 0:>6.1f} | "
                     f"{p['cagr']*100:>+6.1f}% | {p['dd']*100:>6.1f}% | {p['sharpe']:>6.2f} | {p['worst_month']*100:>7.1f}% | "
                     + " | ".join(f"{yr.get(y, np.nan)*100:>+4.0f}%" for y in years))

    curves["DIP v2"] = portfolio(run_dip(prep_dip(daily, al100, btc_ok, start), 7, 3, 0.05), closes, "fixed", 0.20)[0]
    emit("\n# DIP v2 30% ke sath (har mahine rebalance)")
    for key in ("TP5 r2 c20", "TP5 r3 c20", "TP5 r3 c30", "TP7 r2 c20", "TP8 r2 c20", "TP10 r2 c20", "trail r2 c20"):
        eq = mix(curves, {key: .7, "DIP v2": .3})
        p = stats(eq)
        emit(f"{key:>14} 70 / DIP 30 | CAGR {p['cagr']*100:+.1f}% | MaxDD {p['dd']*100:.1f}% | Sharpe {p['sharpe']:.2f} | "
             f"+mahine {p['pos_months']:.0f}% | bura mahina {p['worst_month']*100:.1f}% | $1000 -> {1000*eq.iloc[-1]/eq.iloc[0]:,.0f}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
