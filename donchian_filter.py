"""
DAILY DONCHIAN + SIRF EK FILTER (BTC > EMA50) - user ki research methodology ka agla qadam (2026-10-01)
=====================================================================================================
Pichla natija: simple Donchian 4H FAIL, Daily par asli magar mamooli edge (N 20-55 + ATR trail 3.0x).
Ab sirf EK cheez badli: BTC ka daily close > BTC EMA50 (signal wale band din par) - warna nayi entry nahi.
Har config do baar: FILTER BAGHAIR vs FILTER KE SATH (baqi sab bilkul same).
Random baseline: filter wali row mein random entries bhi sirf BTC>EMA50 wale din (wahi shart, random din).
N = 10/20/30/40/55 | Exit: ATR stop + ATR trailing (2.5x aur 3.0x), max 365 din | Daily | 109 coins | OOS 2025+
Regimes (BTC bull/bear/sideways, BTC volatility) har row ke liye. Portfolio 1% risk, max 10, coin max 20%.
Natija: donchian_filter_RESULTS.txt
"""
import numpy as np
import pandas as pd

import donchian_research as R
from bot_core import fetch_full, norm, STABLES, FEE, SLIP, STOP_SLIP
from portfolio_lab import ema, to_daily

NS = [10, 20, 30, 40, 55]
TRAILS = [2.5, 3.0]
OUT = "donchian_filter_RESULTS.txt"


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
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    b50, b200 = ema(bd, 50), ema(bd, 200)
    btc_reg = pd.Series(np.where((bd > b200) & (b50 > b200), "bull", np.where((bd < b200) & (b50 < b200), "bear", "sideways")),
                        index=bd.index)
    bv = bd.pct_change().rolling(30).std()
    btc_vol = pd.Series(np.where(bv > bv.median(), "ooncha", "neecha"), index=bd.index)
    btc_ok = bd > b50                                                   # band din ka close vs EMA50

    U = R.Universe(daily, warm=60)
    allow = {s: btc_ok.reindex(d["timestamp"]).fillna(False).to_numpy(bool) for s, d in daily.items()}

    emit("=" * 140)
    emit("DAILY DONCHIAN + SIRF EK FILTER: BTC daily close > BTC EMA50")
    emit("=" * 140)
    emit(f"Coins: {len(daily)} | Period: {start.date()} -> {closes.index[-1].date()} | OOS: {R.OOS_START.date()} se | "
         f"BTC>EMA50 din: {btc_ok[btc_ok.index >= start].mean()*100:.0f}%")
    emit(f"Kharcha: fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% har taraf + stop slip {STOP_SLIP*100:.2f}% | portfolio 1% risk, max 10, coin max 20%")
    emit("PASS (ek config): trades>=100 (OOS>=30), PF>1, OOS PF>1, expectancy>0, PF >= 1.10 x random PF, OOS PF >= random OOS PF,")
    emit("                  portfolio MaxDD > -40% aur net return > 0, 3 trend regimes mein se kam az kam 2 mein PF >= 0.9")

    res = {}
    for N in NS:
        sig_all = U.signals(N)
        sig_f = {s: idx[allow[s][idx]] for s, idx in sig_all.items()}
        emit(f"\n\n{'#' * 140}\nDAILY DONCHIAN {N}\n{'#' * 140}")
        emit(R.HEAD)
        for m in TRAILS:
            for tag, sig, al in (("BAGHAIR filter", sig_all, None), ("BTC>EMA50", sig_f, allow)):
                s = R.evaluate(U, sig, m, 0.0, True, 365, closes, start, btc_reg, btc_vol, allow=al)
                if s:
                    res[(N, m, tag)] = s
                    emit(R.line(f"trail {m}x | {tag}", s))
                    emit(f"{'':>28}   regimes: " + " | ".join(
                        f"{g} {R.fmt(s[f'reg_{g}'][0])} ({s[f'reg_{g}'][1]})" for g in ("bull", "bear", "sideways"))
                         + " || " + " | ".join(f"vol {g} {R.fmt(s[f'vol_{g}'][0])} ({s[f'vol_{g}'][1]})" for g in ("ooncha", "neecha")))

    emit("\n\n" + "=" * 140)
    emit("FILTER KA ASAR (trail 3.0x) - ek hi tabdeeli")
    emit("=" * 140)
    emit(f"{'Test':>26} | {'Trades':>6} | {'Win%':>5} | {'PF':>5} | {'RndPF':>5} | {'OOS PF':>6} | {'NetRet':>8} | {'MaxDD':>7} | "
         f"{'Bear PF':>7} | Result")
    for N in NS:
        for tag in ("BAGHAIR filter", "BTC>EMA50"):
            s = res.get((N, 3.0, tag))
            if s:
                emit(f"{f'Daily D{N} {tag}':>26} | {s['n']:>6} | {s['win']:>5.1f} | {R.fmt(s['pf']):>5} | {R.fmt(s['rpf']):>5} | "
                     f"{R.fmt(s['pf_oos']):>6} | {s['total']*100:>+7.0f}% | {s['dd']*100:>6.1f}% | {R.fmt(s['reg_bear'][0]):>7} | "
                     f"{'PASS' if s['pass'] else 'FAIL'}")
    for m in TRAILS:
        for tag in ("BAGHAIR filter", "BTC>EMA50"):
            k = [res[(N, m, tag)]["pass"] for N in NS if (N, m, tag) in res]
            emit(f"trail {m}x | {tag:>14}: {sum(k)}/{len(k)} N PASS")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
