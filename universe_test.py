"""
UNIVERSE TEST - coins 100 se 150 / 200 karne se signals barhte hain? natija wahi rehta hai? (user, 2026-10-04)
==============================================================================================================
KuCoin ke top ~260 coins (volume se) ka 4H data. Har din "top-U liquid" (30 din ka dollar volume) U = 50/70/100/150/200.
Systems: Ichimoku TP5 (CE 16/4 + TP 5%, 2% risk), Ichimoku live (CE 5.5 + 3R, 1% risk),
         Dip naya (RSI<7, SMA3, TP5, 20% size), Dip live (RSI<10, SMA5, 20% size).
Har U: rozana ausat kitne coins asal mein eligible, signals/hafta, win%, PF, OOS PF (2025+), CAGR, MaxDD, Sharpe.
Phir: sirf NAYE coins (rank 101-200) wali trades alag - kya woh bhi utni hi achi hain?
Natija: universe_test_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema, to_daily, portfolio, stats, H4_BARS
from winrate_lab import prep_ichi, run_ichi, tstats
from dip_v2_validate import prep_dip, run_dip

OUT = "universe_test_RESULTS.txt"
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
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes = closes[closes.index >= start]
    yrs = (closes.index[-1] - closes.index[0]).days / 365.25
    have = dv[dv.index >= start].notna().sum(axis=1)

    emit("=" * 110)
    emit("UNIVERSE TEST - top 50 / 70 / 100 / 150 / 200 liquid coins")
    emit("=" * 110)
    emit(f"Data mila: {len(h4)} coins (KuCoin top {TOP_FETCH} mein se, kam az kam 120 din) | {closes.index[0].date()} -> {closes.index[-1].date()}")
    emit(f"Rozana kitne coins ke paas volume data tha: ausat {have.mean():.0f}, kam se kam {have.min()}, ziada se ziada {have.max()}, "
         f"2025+ ausat {have[have.index >= '2025-01-01'].mean():.0f}")
    emit("Note: jis din kisi coin ka data hi nahi (listing se pehle), woh us din kisi universe mein nahi ginta.")
    emit("win = +0.5% se ziada | OOS = 2025+")

    SYS = {
        "ICHI TP5": ("ichi", {"ce": 4.0, "tp": 0.05}, ("risk", 0.02)),
        "ICHI live": ("ichi", {"tpR": 3.0}, ("risk", 0.01)),
        "DIP naya": ("dip", (7, 3, 0.05), ("fixed", 0.20)),
        "DIP live": ("dip", (10, 5, None), ("fixed", 0.20)),
    }
    res = {}
    for U in (50, 70, 100, 150, 200):
        allowed = {day: set(row[row <= U].index) for day, row in rank.iterrows()}
        eff = np.mean([len(v) for d, v in allowed.items() if d >= start])
        emit("\n" + "#" * 110)
        emit(f"# TOP-{U}  (asal mein rozana ausat {eff:.0f} coins eligible)")
        emit("#" * 110)
        emit(f"{'System':>10} | {'trades':>6} | {'/hafta':>6} | {'win%':>5} | {'PF':>5} | {'OOS':>5} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'bura mah':>8}")
        idata = prep_ichi(h4, allowed, start)
        P = prep_dip(daily, allowed, btc_ok, start)
        for name, (kind, cfg, (mode, size)) in SYS.items():
            tr = run_ichi(idata, cfg) if kind == "ichi" else run_dip(P, *cfg)
            s = tstats(tr)
            p = stats(portfolio(tr, closes, mode, size)[0])
            res[(name, U)] = tr
            emit(f"{name:>10} | {s['n']:>6} | {s['n']/yrs/52:>6.2f} | {s['win']:>5.1f} | {s['pf']:>5.2f} | {s['oos']:>5.2f} | "
                 f"{p['cagr']*100:>+6.1f}% | {p['dd']*100:>6.1f}% | {p['sharpe']:>6.2f} | {p['worst_month']*100:>7.1f}%")

    emit("\n" + "#" * 110)
    emit("# SIRF NAYE COINS: jo trades top-200 mein hain magar top-100 mein NAHI thin (rank 101-200 coins)")
    emit("#" * 110)
    for name in SYS:
        base = {(t["sym"], str(t["t_in"])) for t in res[(name, 100)]}
        extra = [t for t in res[(name, 200)] if (t["sym"], str(t["t_in"])) not in base]
        s = tstats(extra)
        s100 = tstats(res[(name, 100)])
        emit(f"{name:>10}: naye coins ki {s['n']} trades | win {s['win']:.1f}% (top-100: {s100['win']:.1f}%) | PF {s['pf']:.2f} "
             f"(top-100: {s100['pf']:.2f}) | OOS {s['oos']:.2f} | avg {s['avg']:+.2f}%")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
