"""
DAILY DONCHIAN - CHHOTE PERIODS KA TEST (3 / 5 / 7 / 10 / 15 din, moqable ke liye 20)
=====================================================================================
Sirf Donchian period badla - baqi sab Donchian Daily Bot jaisa:
  Entry : daily close > pichle N din ka sab se ooncha high (fresh) + BTC close > BTC EMA50
          + coin top-100 liquid -> agle din ke open par
  Exit  : Chandelier trailing 22 din, 4x ATR (sirf ooper)
  Size  : 1% risk, max 10 positions, ek coin max 20%
Wahi sakht usool: ~6.5 saal daily (2022 crash samet), lookahead-free, fee+slip+stop slip,
random-entry control, portfolio + random portfolio (20 seeds), saal-war natija.
Natija: donchian_periods_RESULTS.txt
"""
import numpy as np
import pandas as pd

from strategy_lab import fetch_full, norm, ema, chandelier, data_report, fmt, STABLES, FEE, SLIP, STOP_SLIP
from unified_test import (sim_trades, portfolio, eq_stats, per_trade_block, trade_pass, build_panel,
                          liquid_mask, N_FOLDS)

TOP_N_COINS = 150
D_LIMIT = 2400
PERIODS = [3, 5, 7, 10, 15, 20]
N_RANDOM = 20
MAX_HOLD = 365


def main():
    from data_fetcher import get_exchange, get_coin_list
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    ex = get_exchange()
    coins = [c for c in get_coin_list(ex) if c.split("/")[0].upper() not in STABLES][:TOP_N_COINS]
    if "BTC/USDT" not in coins:
        coins.insert(0, "BTC/USDT")
    d1, fails = {}, 0
    for k, sym in enumerate(coins, 1):
        try:
            d = fetch_full(ex, sym, "1d", D_LIMIT)
            if d is not None and len(d) >= 150:
                d1[sym] = norm(d)
        except Exception as e:
            fails += 1
            print(f"[{k}] {sym}: SKIP ({e})")

    emit("=" * 100)
    emit("DAILY DONCHIAN - CHHOTE PERIODS (3/5/7/10/15) vs 20")
    emit("=" * 100)
    emit(f"fetch errors: {fails}/{len(coins)} | fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% har taraf + stop slip {STOP_SLIP*100:.2f}%")
    for l in data_report("DAILY", d1, D_LIMIT, "1d"):
        emit(l)
    emit("Exit: CE 22/4x | Regime: BTC>EMA50 | 1% risk, max 10 positions")

    idx = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in d1.values()])))
    syms, O, H, L, C, V = build_panel(d1, idx)
    T, N = O.shape
    liquid = liquid_mask(C, V, hist_min=60, window=30, min_periods=20)
    mom = pd.DataFrame(C).pct_change(60, fill_method=None).values
    start = int(max(np.argmax(liquid.sum(1) >= 30), 60))
    bt = pd.Series(C[:, syms.index("BTC/USDT")], index=idx).ffill()
    reg = np.repeat((bt > ema(bt, 50)).values[:, None], N, axis=1)
    stop = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        stop[idx.get_indexer(d1[s]["timestamp"]), j] = chandelier(d1[s], 22, 4.0)
    fold_bounds = pd.date_range(idx[start], idx[-1], periods=N_FOLDS + 1).values
    emit(f"Period: {idx[start].date()} -> {idx[-1].date()}")

    rows = []
    for n in PERIODS:
        sig = np.zeros((T, N), bool)
        for j, s in enumerate(syms):
            d = d1[s]
            brk = (d["close"] > d["high"].shift(1).rolling(n).max()).fillna(False)
            sig[idx.get_indexer(d["timestamp"]), j] = (brk & ~brk.shift(1, fill_value=False)).values
        rng = np.random.default_rng(7)
        tr, rtr = [], []
        for j in range(N):
            ok = sig[:, j] & reg[:, j] & liquid[:, j]
            ok[:start] = False
            ent = np.where(ok)[0]
            tr += sim_trades(O[:, j], H[:, j], L[:, j], C[:, j], ent, stop[:, j], stop[:, j], None, MAX_HOLD)
            pool = np.where(reg[:, j] & liquid[:, j] & ~np.isnan(O[:, j]))[0]
            pool = pool[pool >= start]
            k = min(len(ent), len(pool))
            if k:
                rent = np.sort(rng.choice(pool, size=k, replace=False))
                rtr += sim_trades(O[:, j], H[:, j], L[:, j], C[:, j], rent, stop[:, j], stop[:, j], None, MAX_HOLD)
        s = per_trade_block(tr, rtr, fold_bounds, idx.values)
        eq, ntr = portfolio(O, H, L, C, sig, reg, stop, stop, None, liquid, mom, start, MAX_HOLD)
        ps = eq_stats(eq[start:], idx[start:], 365)
        rsh = []
        for sd in range(N_RANDOM):
            req, _ = portfolio(O, H, L, C, sig, reg, stop, stop, None, liquid, mom, start, MAX_HOLD,
                               rng=np.random.default_rng(sd))
            rsh.append(eq_stats(req[start:], idx[start:], 365)["sharpe"])
        r95 = float(np.nanpercentile(rsh, 95))
        ok = trade_pass(s) and ps["sharpe"] > r95 and ps["cagr"] > 0 and ps["dd"] > -0.5
        rows.append((n, s, ps, r95, ok, ntr))
        folds = " / ".join(f"{fmt(p)}({c})" for p, c in zip(s["fp"], s["fn"]))
        yr = " ".join(f"{y}:{v*100:+.0f}%" for y, v in ps["yearly"].items())
        emit(f"\n[{'PASS' if ok else 'FAIL'}] Donchian {n} din")
        emit(f"   Har trade: n={s['n']} win={s['win']:.1f}% PF={fmt(s['pf'])} p5={fmt(s['p5'])} "
             f"Top10-hata={fmt(s['t10'])} RandomPF={fmt(s['rpf'])} Exp={s['exp']:+.2f}% | Folds: {folds}")
        emit(f"   Portfolio: CAGR={ps['cagr']*100:+.1f}% MaxDD={ps['dd']*100:.1f}% Sharpe={fmt(ps['sharpe'])} "
             f"(random 95th={r95:.2f}) trades={ntr}")
        emit(f"   Saal-war: {yr}")

    emit("\n" + "=" * 100)
    emit("KHULASA")
    emit("=" * 100)
    emit(f"{'Period':>8} | {'Trades':>6} | {'Win%':>5} | {'PF':>5} | {'p5':>5} | {'Random':>6} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | Faisla")
    for n, s, ps, r95, ok, ntr in rows:
        emit(f"{n:>6} d | {s['n']:>6} | {s['win']:>5.1f} | {fmt(s['pf']):>5} | {fmt(s['p5']):>5} | {fmt(s['rpf']):>6} | "
             f"{ps['cagr']*100:>+6.1f}% | {ps['dd']*100:>6.1f}% | {fmt(ps['sharpe']):>6} | {'PASS' if ok else 'FAIL'}")

    with open("donchian_periods_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[SAVE] donchian_periods_RESULTS.txt")


if __name__ == "__main__":
    main()
