"""
W52 COMBO TEST (2026-10-06) - user: "5 din par band karna aur chalta SL - dono mein faida hai to in ka combination?"
Wohi W52 entry. Exit: X din (5/7/10) baad close, AUR/YA chalta SL (nafa A% hone par SL = entry x (1 + chouti nafa - G%)).
Portfolio 10% x max 10 (slots ki tangi shamil). DEV aur POORA dono.
Natija: w52_combo_test_RESULTS.txt
"""
import traceback
import numpy as np
import pandas as pd
from bot_core import fetch_full, norm, STABLES
from portfolio_lab import portfolio, stats
from winrate_lab import tstats, pf_of
import search_lab9 as S9
import trail_lab11 as T

OUT = "w52_combo_test_RESULTS.txt"
CELLS = [("10 din (purana)", dict(hold=10)), ("5 din", dict(hold=5)), ("7 din", dict(hold=7)),
         ("5 din + chalta A3 G2", dict(hold=5, act=0.03, gap=0.02)),
         ("5 din + chalta A5 G2", dict(hold=5, act=0.05, gap=0.02)),
         ("5 din + chalta A5 G3", dict(hold=5, act=0.05, gap=0.03)),
         ("5 din + chalta A8 G3", dict(hold=5, act=0.08, gap=0.03)),
         ("5 din + chalta A10 G5", dict(hold=5, act=0.10, gap=0.05)),
         ("7 din + chalta A5 G2", dict(hold=7, act=0.05, gap=0.02)),
         ("10 din + chalta A5 G2", dict(hold=10, act=0.05, gap=0.02))]


def main():
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)
    try:
        import data_fetcher
        data_fetcher.AUTO_TOP_N_COINS = S9.TOP_FETCH
        ex = data_fetcher.get_exchange()
        coins = [s for s in data_fetcher.get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:S9.TOP_FETCH]
        if "BTC/USDT" not in coins:
            coins.insert(0, "BTC/USDT")
        daily = {}
        for sym in coins:
            try:
                df = fetch_full(ex, sym, "1d", S9.DAYS)
                if df is not None and len(df) >= 250:
                    daily[sym] = norm(df)
            except Exception:
                pass
        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        start = closes.index[0] + pd.Timedelta(days=210)
        closes = closes[closes.index >= start]
        U = T.units_w52(daily, start)
        emit(f"W52 COMBO TEST | coins {len(daily)} | {closes.index[0].date()} -> {closes.index[-1].date()} | 10% x max 10")
        for period, cl in (("dev", closes[closes.index < S9.HOLDOUT]), ("all", closes)):
            emit(f"\n# {'DEV (2025-10 se pehle)' if period == 'dev' else 'POORA (aakhri 12 mahine samet)'}")
            emit(f"{'tareeqa':>24} | {'li':>4} | {'jeet/10':>7} | {'PF':>5} | {'rnd95':>5} | {'ausat':>7} | {'CAGR':>7} | {'MaxDD':>6} | "
                 f"{'Sharpe':>6} | {'+mah%':>5} | {'bura mah':>8}")
            for name, cfg in CELLS:
                cfg = dict(cfg)
                tr = T.run(U, cfg, period)
                eq, taken = portfolio(tr, cl, "fixed", 0.10, max_pos=10, cap=1.0)
                p = stats(eq)
                s = tstats(taken)
                rnd = [pf_of([t["ret"] for t in T.run(U, cfg, period, rng=np.random.default_rng(70 + q))]) for q in range(6)]
                emit(f"{name:>24} | {len(taken):>4} | {s['win']/10:>7.1f} | {s['pf']:>5.2f} | {np.percentile(rnd,95):>5.2f} | "
                     f"{s['avg']:>+6.2f}% | {p['cagr']*100:>+6.1f}% | {p['dd']*100:>5.1f}% | {p['sharpe']:>6.2f} | "
                     f"{p['pos_months']:>5.0f} | {p['worst_month']*100:>7.1f}%")
    except Exception:
        emit(traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
