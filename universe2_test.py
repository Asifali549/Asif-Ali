"""
UNIVERSE 2 TEST - Ichimoku TP5: 200 se aage 250 / 300 / 350 / 400 coins (user, 2026-10-05)
==========================================================================================
KuCoin ke top ~600 coins (24h volume) ka 4H data. Har din "top-U liquid" (30 din dollar volume), U = 100..400.
TP5 = CE 16/4 stop + TP 5%, 2% risk, coin cap 20%, max 10.
Har U: rozana asal eligible coins, signals/hafta, win%, PF, OOS (2025+), CAGR, MaxDD, Sharpe, bura mahina, saal-war.
Phir rank-band (1-100, 101-200, 201-300, 301-400) ki trades alag: win, PF, OOS, 2x kharcha PF,
aur us band ke coins ka ausat rozana dollar volume (chhote coins = slippage khatra).
Natija: universe2_test_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import to_daily, portfolio, stats, H4_BARS
from winrate_lab import prep_ichi, run_ichi, tstats, pf_of

OUT = "universe2_test_RESULTS.txt"
TOP_FETCH = 600


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    if h4 is None:
        import data_fetcher
        data_fetcher.AUTO_TOP_N_COINS = TOP_FETCH
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
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    rank = dv.rank(axis=1, ascending=False)
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes = closes[closes.index >= start]
    yrs = (closes.index[-1] - closes.index[0]).days / 365.25
    have = dv[dv.index >= start].notna().sum(axis=1)

    emit("=" * 120)
    emit("UNIVERSE 2 TEST - Ichimoku TP5 (CE 16/4, TP 5%, 2% risk): top 100 / 200 / 250 / 300 / 350 / 400 coins")
    emit("=" * 120)
    emit(f"Data: {len(h4)} coins (KuCoin top {TOP_FETCH} mein se, kam az kam 120 din) | {closes.index[0].date()} -> {closes.index[-1].date()}")
    emit(f"Rozana volume-data wale coins: ausat {have.mean():.0f}, 2025+ ausat {have[have.index >= '2025-01-01'].mean():.0f}, "
         f"aakhri din {have.iloc[-1]}")
    emit("win = +0.5% se ziada | OOS = 2025+ | '2025+ elig' = 2025 se rozana asal mein kitne coins us universe mein the")

    years = list(range(2021, 2027))
    emit(f"\n{'U':>4} | {'elig':>4} | {'2025+ elig':>10} | {'trades':>6} | {'/hafta':>6} | {'win%':>5} | {'PF':>5} | {'OOS':>5} | "
         f"{'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'bura mah':>8} | " + " | ".join(f"{y:>5}" for y in years))
    res = {}
    for U in (100, 200, 250, 300, 350, 400):
        allowed = {day: set(row[row <= U].index) for day, row in rank.iterrows()}
        el = pd.Series({d: len(v) for d, v in allowed.items() if d >= start})
        tr = run_ichi(prep_ichi(h4, allowed, start), {"ce": 4.0, "tp": 0.05})
        s = tstats(tr)
        p = stats(portfolio(tr, closes, "risk", 0.02)[0])
        yr = {d.year: v for d, v in p["yearly"].items()}
        res[U] = tr
        emit(f"{U:>4} | {el.mean():>4.0f} | {el[el.index >= '2025-01-01'].mean():>10.0f} | {s['n']:>6} | {s['n']/yrs/52:>6.2f} | "
             f"{s['win']:>5.1f} | {s['pf']:>5.2f} | {s['oos']:>5.2f} | {p['cagr']*100:>+6.1f}% | {p['dd']*100:>6.1f}% | "
             f"{p['sharpe']:>6.2f} | {p['worst_month']*100:>7.1f}% | " + " | ".join(f"{yr.get(y, np.nan)*100:>+4.0f}%" for y in years))

    emit("\n# RANK-BAND: top-400 run ki trades, entry ke din coin ka rank kis band mein tha")
    emit(f"{'band':>9} | {'trades':>6} | {'/hafta':>6} | {'win%':>5} | {'PF':>5} | {'OOS':>5} | {'2x kharcha PF':>13} | {'avg %':>6} | {'ausat $ volume/din':>18}")
    for lo, hi in ((1, 100), (101, 200), (201, 250), (251, 300), (301, 400)):
        band = []
        vols = []
        for t in res[400]:
            day = pd.Timestamp(t["t_in"]).floor("1D")
            if day in rank.index:
                r = rank.at[day, t["sym"]] if t["sym"] in rank.columns else np.nan
                if lo <= r <= hi:
                    band.append(t)
                    vols.append(dv.at[day, t["sym"]])
        s = tstats(band)
        pf2 = pf_of([t["ret"] - 0.003 for t in band])          # har trade par ~0.3% extra kharcha (fee+slippage dugna)
        emit(f"{lo:>4}-{hi:<4} | {s['n']:>6} | {s['n']/yrs/52:>6.2f} | {s['win']:>5.1f} | {s['pf']:>5.2f} | {s['oos']:>5.2f} | "
             f"{pf2:>13.2f} | {s['avg']:>+5.2f}% | {np.nanmedian(vols) if vols else 0:>18,.0f}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
