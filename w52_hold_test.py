"""
W52 HOLD TEST - user ka sawal (2026-10-06): "5 din mein bechein to paisa jaldi azad - 10 din mein 2 trades, nafa double?"
Asal sawal: kya sarmaya (slots) kam parta hai? Agar signals itne kam hain ke paisa pehle hi khali betha hai, to jaldi
bechne se trades nahi barhti. Yahan H5 / H7 / H10 portfolio mein: 10% x max 10 aur 20% x max 5 (tang sarmaya),
kitne signals slot na hone se chhoot gaye, ausat kitni trades khuli, ausat nafa / trade, CAGR, DD, Sharpe - DEV aur poora.
Natija: w52_hold_test_RESULTS.txt
"""
import traceback
import numpy as np
import pandas as pd
from bot_core import fetch_full, norm, STABLES
from portfolio_lab import portfolio, stats
from winrate_lab import tstats
import search_lab9 as S9
import w52_lab10 as W

OUT = "w52_hold_test_RESULTS.txt"


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
        P = S9.build(daily, start)
        emit("W52 HOLD TEST - 5 / 7 / 10 din, sarmaya kam parta hai ya nahi")
        emit(f"coins {len(daily)} | {closes.index[0].date()} -> {closes.index[-1].date()}")
        for period, cl in (("dev", closes[closes.index < S9.HOLDOUT]), ("all", closes)):
            emit(f"\n# {'DEV (2025-10 se pehle)' if period == 'dev' else 'POORA (taala samet)'}")
            emit(f"{'exit':>4} | {'size':>10} | {'signals':>7} | {'li gayin':>8} | {'chhooti':>7} | {'ausat khuli':>11} | {'jeet/10':>7} | "
                 f"{'ausat/trade':>11} | {'CAGR':>7} | {'MaxDD':>6} | {'Sharpe':>6}")
            for ex_ in ("H5", "H7", "H10"):
                tr = W.run(P, 0.05, ex_, period)
                for size, mx in ((0.10, 10), (0.20, 5)):
                    eq, taken = portfolio(tr, cl, "fixed", size, max_pos=mx, cap=1.0)
                    p = stats(eq)
                    s = tstats(taken)
                    cnt = pd.Series(0.0, index=cl.index)
                    for t in taken:
                        a, b = pd.Timestamp(t["t_in"]).floor("1D"), pd.Timestamp(t["t_out"]).floor("1D")
                        cnt[(cnt.index >= a) & (cnt.index <= b)] += 1
                    emit(f"{ex_:>4} | {int(size*100):>3}% x {mx:<3} | {len(tr):>7} | {len(taken):>8} | {len(tr)-len(taken):>7} | "
                         f"{cnt.mean():>11.2f} | {s['win']/10:>7.1f} | {s['avg']:>+10.2f}% | {p['cagr']*100:>+6.1f}% | "
                         f"{p['dd']*100:>5.1f}% | {p['sharpe']:>6.2f}")
    except Exception:
        emit(traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
