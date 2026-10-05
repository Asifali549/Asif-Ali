"""
COMBO LAB 6 - "is se kaam nahi banega, kuch best karo" (user, 2026-10-05)
=========================================================================
Masla: Dip (0.6/hafta) aur Residual Dip (1.8/hafta) dono asli edge wale hain, magar paisa aksar khali bethta hai
(sirf ~43% mahine musbat, baqi khali). Nafa barhane ka sab se sacha tareeqa naya jadoo nahi, balke DONO ko EK
khate (ek hi sarmaya) mein chalana aur khali paise ko kaam par lagana: ziada signals + bara size.

Yahan:
  BOOK = DIP v2 (RSI3<7) + RESID (z < Z) ek sath, ek sarmaya, ek coin par ek waqt mein ek trade.
  Z = -1.75 / -2.0 / -2.25 (padosi) x size per trade 20 / 25 / 33 / 50% x max khuli trades 10 / 4 / 3 / 2.
  Har cell: trades/hafta, win, PF, OOS, CAGR, MaxDD, Sharpe, musbat mahine %, ausat mahana, bura mahina,
  ausat khuli trades (paisa kitna kaam par), saal-war, 2x kharcha CAGR.
  Usool: MaxDD 25% se ziada wala cell mashware mein nahi (insaan jhel nahi pata).
Signals strategy_lab5.build se (andar sachai test pass). Natija: combo_lab6_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import portfolio, stats
from winrate_lab import tstats, pf_of
import strategy_lab5 as L5

OUT = "combo_lab6_RESULTS.txt"
ZS = [-1.75, -2.0, -2.25]
SIZES = [(0.20, 10), (0.25, 4), (0.33, 3), (0.50, 2)]


def open_count(taken, days):
    cnt = pd.Series(0.0, index=days)
    for t in taken:
        a, b = pd.Timestamp(t["t_in"]).floor("1D"), pd.Timestamp(t["t_out"]).floor("1D")
        cnt[(cnt.index >= a) & (cnt.index <= b)] += 1
    return cnt.mean()


def main(daily=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    try:
        if daily is None:
            import data_fetcher
            data_fetcher.AUTO_TOP_N_COINS = L5.TOP_FETCH
            ex = data_fetcher.get_exchange()
            coins = [s for s in data_fetcher.get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:L5.TOP_FETCH]
            if "BTC/USDT" not in coins:
                coins.insert(0, "BTC/USDT")
            daily = {}
            for k, sym in enumerate(coins, 1):
                try:
                    df = fetch_full(ex, sym, "1d", L5.DAYS)
                    if df is not None and len(df) >= 250:
                        daily[sym] = norm(df)
                        print(f"[{k}/{len(coins)}] {sym}: {len(df)}", flush=True)
                except Exception as e:
                    print(f"[{k}] {sym}: SKIP ({e})", flush=True)

        L5.GRID["R1_RESID"] = [(f"z{z}", z) for z in ZS]
        btc_ok, al100 = L5.context(daily)
        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        start = closes.index[0] + pd.Timedelta(days=210)
        closes = closes[closes.index >= start]
        weeks = (closes.index[-1] - closes.index[0]).days / 7
        P = L5.build(daily, btc_ok, al100, start)

        emit("=" * 125)
        emit("COMBO LAB 6 - DIP + RESIDUAL DIP ek khate mein, size / max trades grid")
        emit("=" * 125)
        emit(f"Coins: {len(daily)} | {closes.index[0].date()} -> {closes.index[-1].date()} | OOS = 2025+")

        dip = L5.run(P, "REF_DIP")
        dip2 = L5.run(P, "REF_DIP", cost=2.0)
        years = list(range(2021, 2027))
        emit(f"\n{'book':>14} | {'size/max':>8} | {'/hafta':>6} | {'win%':>5} | {'PF':>5} | {'OOS':>5} | {'CAGR':>7} | {'MaxDD':>6} | "
             f"{'Sharpe':>6} | {'+mah%':>5} | {'ausat mah':>9} | {'bura mah':>8} | {'khuli':>5} | {'2x CAGR':>7} | "
             + " | ".join(f"{y:>5}" for y in years))

        def line(name, tr, tr2, size, mx):
            s = tstats(tr)
            eq, taken = portfolio(tr, closes, "fixed", size, max_pos=mx, cap=1.0)
            p = stats(eq)
            p2 = stats(portfolio(tr2, closes, "fixed", size, max_pos=mx, cap=1.0)[0])
            yr = {d.year: v for d, v in p["yearly"].items()}
            emit(f"{name:>14} | {int(size*100):>3}% / {mx:<2} | {s['n']/weeks:>6.2f} | {s['win']:>5.1f} | {s['pf']:>5.2f} | {s['oos']:>5.2f} | "
                 f"{p['cagr']*100:>+6.1f}% | {p['dd']*100:>5.1f}% | {p['sharpe']:>6.2f} | {p['pos_months']:>5.0f} | "
                 f"{p['monthly'].mean()*100:>+8.2f}% | {p['worst_month']*100:>7.1f}% | {open_count(taken, closes.index):>5.2f} | "
                 f"{p2['cagr']*100:>+6.1f}% | " + " | ".join(f"{yr.get(y, np.nan)*100:>+4.0f}%" for y in years))
            return p

        best = []
        for size, mx in SIZES:
            line("DIP akela", dip, dip2, size, mx)
        for lab, z in L5.GRID["R1_RESID"]:
            nm = f"R1_RESID {lab}"
            r1, r12 = L5.run(P, nm), L5.run(P, nm, cost=2.0)
            emit("")
            for size, mx in SIZES:
                line(f"RESID {z}", r1, r12, size, mx)
            for size, mx in SIZES:
                p = line(f"DIP+RESID {z}", dip + r1, dip2 + r12, size, mx)
                best.append((p["sharpe"], p["cagr"], p["dd"], z, size, mx))

        emit("\n# MASHWARA KE QABIL (MaxDD 25% se kam), Sharpe se tarteeb")
        for sh, cg, dd, z, size, mx in sorted([b for b in best if b[2] > -0.25], reverse=True)[:6]:
            emit(f"DIP+RESID z{z} | {int(size*100)}% x max {mx} | Sharpe {sh:.2f} | CAGR {cg*100:+.1f}% | MaxDD {dd*100:.1f}%")
    except Exception:
        emit("\n!! GHALTI (code crash):\n" + traceback.format_exc())

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
