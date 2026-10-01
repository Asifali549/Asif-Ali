"""
VOLATILITY TARGETING - ICHI 60% + DIP 40% portfolio par (2026-10-01)
====================================================================
Khayal: entry/exit bilkul wahi (live bots jaise). Sirf NAYI trade ka size ek "multiplier" se:
   mult = clip( normal volatility / haaliya volatility , 0.25 , CAP )
   - haaliya vol = pichle L din ke rozana returns ka std (sirf kal tak ka data -> lookahead nahi)
   - normal vol  = ab tak ki tamam haaliya-vol values ka median (expanding, sirf maazi)
   Ziada hal-chal -> chhota size; pur-sukoon market -> poora (ya CAP 1.5 tak bara) size.
   Khuli trades ko chheda nahi jata (bot mein asaani se lagu ho sake).
Signal 2 qism: BTC ki volatility  |  portfolio ki apni (shadow) equity ki volatility.
Grid (plateau): signal {BTC, APNI} x L {20, 60} x CAP {1.0, 1.5}.
Base: ICHI (1% risk, max 10, coin 20%) 60% + DIP (har trade 20%, max 10) 40%, mahana rebalance.
Pass: Sharpe base se kam az kam +0.10, MaxDD base se 2% behtar, bura-tareen saal base se bura nahi.
Natija: vol_target_RESULTS.txt
"""
import numpy as np
import pandas as pd

import portfolio_lab as P
from bot_core import fetch_full, norm, STABLES

OUT = "vol_target_RESULTS.txt"
FLOOR = 0.25
LS, CAPS = (20, 60), (1.0, 1.5)
W = {"ICHI": 0.6, "DIP": 0.4}


def portfolio_m(trades, closes, mode, size, mult, max_pos=10, cap=0.20):
    """portfolio_lab.portfolio jaisa, bas nayi trade ka size x mult[us din] (mult pehle se shift shuda)."""
    days = closes.index
    ent = sorted(trades, key=lambda t: (t["t_in"], -t["prio"]))
    by_day = {}
    for t in ent:
        by_day.setdefault(pd.Timestamp(t["t_in"]).floor("1D"), []).append(("in", t["t_in"], t))
        by_day.setdefault(pd.Timestamp(t["t_out"]).floor("1D"), []).append(("out", t["t_out"], t))
    cash, open_ = 1.0, {}
    eq, last_px = [], {}
    for D in days:
        row = closes.loc[D]
        m = float(mult.get(D, 1.0)) if np.isfinite(mult.get(D, np.nan)) else 1.0

        def mtm_now():
            return sum(v * last_px.get(s, t["entry_px"]) / t["entry_px"] for s, (t, v) in open_.items())
        evs = sorted(by_day.get(D, []), key=lambda x: (x[1], 0 if x[0] == "out" else 1))
        for kind, _, t in evs:
            s = t["sym"]
            if kind == "out":
                if s in open_ and open_[s][0] is t:
                    tt, v = open_.pop(s)
                    cash += v * (1 + tt["ret"])
                continue
            if s in open_ or len(open_) >= max_pos:
                continue
            equity_now = cash + mtm_now()
            val = equity_now * size * m if mode == "fixed" else min(equity_now * size * m / max(t["risk"], 1e-6), cap * equity_now)
            val = min(val, cash)
            if val <= equity_now * 0.002:
                continue
            cash -= val
            open_[s] = (t, val)
            if t["t_out"] <= t["t_in"]:
                open_.pop(s)
                cash += val * (1 + t["ret"])
        for s in open_:
            if np.isfinite(row.get(s, np.nan)):
                last_px[s] = row[s]
        eq.append(cash + sum(v * last_px.get(s, t["entry_px"]) / t["entry_px"] for s, (t, v) in open_.items()))
    return pd.Series(eq, index=days)


def make_mult(ret, L, cap):
    vol = ret.rolling(L, min_periods=L).std()
    ref = vol.expanding(min_periods=120).median()
    m = (ref / vol).clip(lower=FLOOR, upper=cap)
    return m.shift(1)            # din D ka multiplier sirf D-1 tak ke data se


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x)
        lines.append(x)

    if h4 is None:
        from data_fetcher import get_exchange, get_coin_list
        ex = get_exchange()
        coins = [s for s in get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:P.TOP_N]
        for must in ("ETH/USDT", "BTC/USDT"):
            if must not in coins:
                coins.insert(0, must)
        h4 = {}
        for k, sym in enumerate(coins, 1):
            try:
                df = fetch_full(ex, sym, "4h", P.H4_BARS)
                if df is not None and len(df) >= 6 * 250:
                    h4[sym] = norm(df)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)}")
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})")

    daily = {s: P.to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc_ok = bd > P.ema(bd, 50)
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    allowed = {day: set(row.dropna().nlargest(P.UNIVERSE).index) for day, row in dv.iterrows()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes = closes[closes.index >= start]

    T = {"ICHI": P.trades_ichi(h4, allowed), "DIP": P.trades_dip(daily, allowed, btc_ok)}
    for k in T:
        T[k] = [t for t in T[k] if pd.Timestamp(t["t_in"]) >= start]
    SPEC = {"ICHI": ("risk", 0.01), "DIP": ("fixed", 0.20)}

    def build(mult):
        cur = {k: portfolio_m(T[k], closes, SPEC[k][0], SPEC[k][1], mult) for k in W}
        return P.mix(cur, W)

    ones = pd.Series(1.0, index=closes.index)
    base = build(ones)
    btc_ret = bd.pct_change().reindex(closes.index)
    own_ret = base.pct_change()

    emit("=" * 140)
    emit("VOLATILITY TARGETING - ICHI 60% + DIP(20%) 40% (entry/exit wahi, sirf nayi trade ka size)")
    emit("=" * 140)
    emit(f"Coins: {len(h4)} | {closes.index[0].date()} -> {closes.index[-1].date()} | mult = clip(normal vol / haaliya vol, {FLOOR}, CAP)")
    emit("ICHI trades: " + P.trade_line(T["ICHI"]) + " | DIP trades: " + P.trade_line(T["DIP"]))

    def yearly(eq):
        y = P.stats(eq)["yearly"]
        return {d.year: v for d, v in y.items()}

    bs = P.stats(base)
    by = yearly(base)
    years = sorted(by)
    emit(f"\n{'Version':>24} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'+mahine':>7} | {'Bura mah':>8} | {'DD din':>6} | "
         f"{'Ausat mult':>10} | {'mult<1 din':>10} | " + " | ".join(f"{y:>5}" for y in years) + " | Faisla")

    def show(name, eq, mult=None, verdict=""):
        s = P.stats(eq)
        yr = yearly(eq)
        mm = mult.reindex(eq.index).dropna() if mult is not None else None
        emit(f"{name:>24} | {s['cagr']*100:>+6.1f}% | {s['dd']*100:>6.1f}% | {s['sharpe']:>6.2f} | {s['pos_months']:>6.0f}% | "
             f"{s['worst_month']*100:>+7.1f}% | {s['long_dd']:>6} | "
             + (f"{mm.mean():>10.2f} | {(mm < 1).mean()*100:>9.0f}% | " if mm is not None else f"{'1.00':>10} | {'0%':>10} | ")
             + " | ".join(f"{yr.get(y, np.nan)*100:>+4.0f}%" for y in years) + f" | {verdict}")
        return s, yr

    show("BASE (koi scaling nahi)", base)
    worst_base = min(by.values())
    res = {}
    for sig_name, ret in (("BTC", btc_ret), ("APNI", own_ret)):
        for L in LS:
            for cap in CAPS:
                mult = make_mult(ret, L, cap)
                eq = build(mult)
                s = P.stats(eq)
                yr = yearly(eq)
                ok = (s["sharpe"] >= bs["sharpe"] + 0.10 and s["dd"] >= bs["dd"] + 0.02 and min(yr.values()) >= worst_base)
                res[(sig_name, L, cap)] = ok
                show(f"{sig_name} L{L} CAP{cap}", eq, mult, "PASS" if ok else "-")

    emit("\n" + "=" * 140 + "\nKHULASA\n" + "=" * 140)
    emit(f"BASE: CAGR {bs['cagr']*100:+.1f}% | MaxDD {bs['dd']*100:.1f}% | Sharpe {bs['sharpe']:.2f} | bura-tareen saal {worst_base*100:+.0f}%")
    for sig_name in ("BTC", "APNI"):
        n = sum(res[(sig_name, L, c)] for L in LS for c in CAPS)
        emit(f"{sig_name} volatility: {n}/{len(LS)*len(CAPS)} PASS  (" + " | ".join(
            f"L{L} CAP{c}: {'PASS' if res[(sig_name, L, c)] else '-'}" for L in LS for c in CAPS) + ")")
    emit("Plateau: ek signal ke ziada tar (3/4+) cells PASS hon tabhi bot mein lagane ka sochen.")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
