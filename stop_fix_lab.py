"""
STOP FIX LAB - bara nuqsan (SL bohat door) theek karna: Donchian, Capitulation, Dip v2 (user, 2026-10-04)
==========================================================================================================
Masla: live mein CARDS (Donchian) SL 42% neeche, US (Capitulation) SL 89% neeche (3 x ATR naye/volatile coin par).
Capitulation/Dip har trade 20% fixed lagate hain -> door SL = ek trade mein account ka bara hissa.
Har system par ye tareeqe:
  stop  : DON CE22 x {4 (live), 3, 2.5, 2} | CAPIT/DIP signal close - k x ATR, k {3 (live), 2, 1.5, 1}
  TP    : none / +5% / (DON +8%)
  maxSL : SL entry se X% se ziada door ho to signal CHHOR do: none / 15% / 20%
  size  : fixed 20% (CAPIT/DIP live) ya RISK 2% (cap 20%) ; DON risk 1% (live) ya 2%
Har cell: trades, win%, PF, OOS (2025+), ek trade ka sab se bara account-nuqsan %, CAGR, MaxDD, Sharpe, bura mahina.
Phir har system ke top-3 (Sharpe) par random-entry control (8 seeds) aur saal-war PF.
Natija: stop_fix_lab_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, chandelier as ce_bot, STABLES
from portfolio_lab import ema, rsi, atr_w, to_daily, portfolio, stats, TOP_N, H4_BARS, UNIVERSE
from winrate_lab import sim, tstats, pf_of

OUT = "stop_fix_lab_RESULTS.txt"


def prep(daily, allowed, btc_ok, start):
    P = []
    for sym, d in daily.items():
        c = d["close"]
        e50, e200 = ema(c, 50), ema(c, 200)
        bok = btc_ok.reindex(d["timestamp"]).fillna(False).to_numpy(bool)
        ok = np.array([sym in allowed.get(t, ()) for t in d["timestamp"]]) & (d["timestamp"] >= start).to_numpy()
        n = np.arange(len(d)) >= 200
        brk = (c > d["high"].shift(1).rolling(20).max()).fillna(False)
        ret1 = c / c.shift(1) - 1
        vavg = d["volume"].rolling(20).mean().shift(1)
        sig = {
            "DON": ((brk & ~brk.shift(1, fill_value=False)).to_numpy() & bok & ok),
            "CAPIT": (((c > e200).to_numpy() & n) & (ret1 <= -0.08).to_numpy() & (d["volume"] >= 2 * vavg).fillna(False).to_numpy() & ok),
            "DIP": (((c > e200) & (e50 > e200)).to_numpy() & n & bok & (rsi(c, 3) < 7).to_numpy() & ok),
        }
        prio = {"DON": (c / c.shift(60) - 1).to_numpy(), "CAPIT": -ret1.to_numpy(), "DIP": -rsi(c, 3).to_numpy()}
        P.append(dict(sym=sym, d=d, ts=d["timestamp"].to_numpy(), o=d["open"].to_numpy(float), h=d["high"].to_numpy(float),
                      l=d["low"].to_numpy(float), c=c.to_numpy(float), atr=atr_w(d).to_numpy(), sig=sig, prio=prio,
                      sma={k: (c > c.rolling(k).mean()).to_numpy() for k in (3, 5)}, ce={}, ok=ok))
    return P


def run(P, system, k, tp, maxsl, rng=None):
    out = []
    for x in P:
        idx = np.where(x["sig"][system])[0]
        if rng is not None:
            base = np.where(x["ok"] & np.isfinite(x["atr"]))[0]
            base = base[base > 210]
            if len(idx) == 0 or len(base) == 0:
                continue
            idx = np.sort(rng.choice(base, size=min(len(idx), len(base)), replace=False))
        if system == "DON":
            if k not in x["ce"]:
                x["ce"][k] = ce_bot(x["d"], 22, k)
            stop_arr, trail, ex, hold = x["ce"][k], x["ce"][k], None, 365
        else:
            stop_arr = x["c"] - k * x["atr"]
            trail = None
            ex = x["sma"][3 if system == "DIP" else 5]
            hold = 10
        cfg = {"tp": tp} if tp else {}
        for i in idx:
            st = stop_arr[i]
            if not np.isfinite(st):
                continue
            if maxsl and (x["c"][i] - st) / x["c"][i] > maxsl:
                continue
            t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, trail, ex, hold, cfg)
            if t:
                pr = x["prio"][system][i]
                t.update(sym=x["sym"], prio=pr if np.isfinite(pr) else -9)
                out.append(t)
    return out


def worst_acct(tr, mode, size, cap=0.20):
    w = 0.0
    for t in tr:
        frac = size if mode == "fixed" else min(size / max(t["risk"], 1e-6), cap)
        w = min(w, frac * t["ret"])
    return w * 100


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    if h4 is None:
        from data_fetcher import get_exchange, get_coin_list
        ex = get_exchange()
        coins = [s for s in get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:TOP_N]
        for must in ("ETH/USDT", "BTC/USDT"):
            if must not in coins:
                coins.insert(0, must)
        h4 = {}
        for k, sym in enumerate(coins, 1):
            try:
                df = fetch_full(ex, sym, "4h", H4_BARS)
                if df is not None and len(df) >= 6 * 250:
                    h4[sym] = norm(df)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)}")
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})")

    daily = {s: to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc_ok = bd > ema(bd, 50)
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    allowed = {day: set(row.dropna().nlargest(UNIVERSE).index) for day, row in dv.iterrows()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes = closes[closes.index >= start]
    P = prep(daily, allowed, btc_ok, start)

    emit("=" * 120)
    emit("STOP FIX LAB - Donchian / Capitulation / Dip v2: SL ki doori, TP, max-SL filter, size")
    emit("=" * 120)
    emit(f"Coins: {len(daily)} | {closes.index[0].date()} -> {closes.index[-1].date()} | top-{UNIVERSE} | OOS = 2025+ | win = +0.5% se ziada")
    emit("'1 trade max' = backtest ki sab se buri EK trade ne poore account ka kitna % khoya (size ke hisab se).")

    GRID = {
        "DON": dict(ks=(4.0, 3.0, 2.5, 2.0), tps=(None, 0.05, 0.08), sizes=(("risk", 0.01), ("risk", 0.02)), live=(4.0, None, None, ("risk", 0.01))),
        "CAPIT": dict(ks=(3.0, 2.0, 1.5, 1.0), tps=(None, 0.05), sizes=(("fixed", 0.20), ("risk", 0.02)), live=(3.0, None, None, ("fixed", 0.20))),
        "DIP": dict(ks=(3.0, 2.0, 1.5, 1.0), tps=(0.05,), sizes=(("fixed", 0.20), ("risk", 0.02)), live=(3.0, 0.05, None, ("fixed", 0.20))),
    }
    best = {}
    for system, g in GRID.items():
        emit("\n" + "#" * 120)
        emit(f"# {system}" + ("  (Dip v2: RSI<7, SMA3, TP5 - sirf stop/filter/size badle)" if system == "DIP" else ""))
        emit("#" * 120)
        emit(f"{'stop':>6} | {'TP':>4} | {'maxSL':>5} | {'size':>8} | {'n':>4} | {'win%':>5} | {'PF':>5} | {'OOS':>5} | {'1 trade max':>11} | "
             f"{'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'bura mah':>8}")
        rows = []
        for k in g["ks"]:
            for tp in g["tps"]:
                for maxsl in (None, 0.15, 0.20):
                    tr = run(P, system, k, tp, maxsl)
                    s = tstats(tr)
                    for mode, size in g["sizes"]:
                        p = stats(portfolio(tr, closes, mode, size)[0])
                        wa = worst_acct(tr, mode, size)
                        live = (k, tp, maxsl, (mode, size)) == g["live"]
                        rows.append(((k, tp, maxsl, mode, size), s, p, wa))
                        emit(f"{('CE' if system == 'DON' else '') + f'{k:g}x':>6} | {('-' if not tp else f'{int(tp*100)}%'):>4} | "
                             f"{('-' if not maxsl else f'{int(maxsl*100)}%'):>5} | {mode[:4] + f' {size*100:g}%':>8} | {s['n']:>4} | {s['win']:>5.1f} | "
                             f"{s['pf']:>5.2f} | {s['oos']:>5.2f} | {wa:>10.1f}% | {p['cagr']*100:>+6.1f}% | {p['dd']*100:>6.1f}% | "
                             f"{p['sharpe']:>6.2f} | {p['worst_month']*100:>7.1f}%{'  <- LIVE' if live else ''}")
        rows.sort(key=lambda r: -r[2]["sharpe"])
        best[system] = rows[:3]

    emit("\n" + "#" * 120)
    emit("# TOP-3 har system (Sharpe se): random-entry control (8 seeds, wohi exit/stop/filter) + saal-war PF")
    emit("#" * 120)
    years = list(range(2021, 2027))
    for system, rows in best.items():
        for (k, tp, maxsl, mode, size), s, p, wa in rows:
            pfs = [pf_of([t["ret"] for t in run(P, system, k, tp, maxsl, rng=np.random.default_rng(40 + q))]) for q in range(8)]
            tr = run(P, system, k, tp, maxsl)
            yr = []
            for y in years:
                r = [t["ret"] for t in tr if pd.Timestamp(t["t_in"]).year == y]
                yr.append(pf_of(r) if len(r) >= 3 else np.nan)
            emit(f"{system:>5} stop {k:g} TP {tp} maxSL {maxsl} {mode} {size*100:g}% | win {s['win']:.1f}% PF {s['pf']:.2f} vs random p95 "
                 f"{np.percentile(pfs, 95):.2f} ({'behtar' if s['pf'] > np.percentile(pfs, 95) else 'NAHI'}) | CAGR {p['cagr']*100:+.1f}% DD "
                 f"{p['dd']*100:.1f}% Sharpe {p['sharpe']:.2f} | 1 trade max {wa:.1f}% | saal PF " + " ".join(f"{v:.2f}" for v in yr))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
