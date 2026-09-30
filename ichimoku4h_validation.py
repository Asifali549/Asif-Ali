"""
4H ICHIMOKU + MARKET STRUCTURE - PAKKI TASDEEQ (validation)
============================================================
Unified test mein 54 combos mein se sirf "4H Union AB Ichimoku+MS (No filter)"
PASS hua (PF 1.88, portfolio CAGR +31%, MaxDD -11%) - lekin:
  (a) 54 combos mein se 1 ka pass hona ittefaq bhi ho sakta hai
  (b) 4H data sirf Oct-2023 se tha - 2022 ka crash shamil nahi tha
Is liye:
  1) LAMBA 4H data (~5.5 saal, 2021 se - 2022 crash samet), saal-war natija
  2) PARAMETER PADOS: har setting ko thora upar/neeche kar ke dekhna (baqi wahi) -
     asli edge ho to pados ki settings bhi achhi rahen; sirf ek jagah achha ho to ittefaq
  3) Wahi sakht usool: lookahead-free, fee+slip+stop slip, top-100 liquid,
     random-entry control, portfolio (10 slots, 1% risk), random portfolio (20 seeds)
  4) Daily Donchian 20 ke sath MILAAP: dono ke rozana returns ka correlation, aur
     50/50 portfolio (dono alag alag aur mila kar) - kya milane se drawdown kam hota hai

Natija: ichimoku4h_validation_RESULTS.txt
"""

import copy

import numpy as np
import pandas as pd

import config
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from strategy_lab import fetch_full, norm, ema, chandelier, data_report, fmt, STABLES, FEE, SLIP, STOP_SLIP
from unified_test import (sim_trades, portfolio, eq_stats, per_trade_block, trade_pass, build_panel,
                          liquid_mask, N_FOLDS)

TOP_N_COINS = 150
H4_LIMIT = 12000            # ~5.5 saal
D_LIMIT = 2400
BPD = 6                     # 4H bars per din
MAX_HOLD = 500
PER_YEAR = 6 * 365
N_RANDOM = 20
COOL = config.SIGNAL_COOLDOWN_BARS

BASE = {"tenkan": 9, "kijun": 26, "senkou_b": 52, "vol_mult": 2.0, "pivot": 5, "swing": 1.5,
        "ce_p": 16, "ce_m": 4.5, "tp": 2.0, "cool": COOL}

VARIANTS = [("BASELINE (production settings)", {})]
for k, vals in [("tenkan", [7, 12]), ("kijun", [22, 30]), ("vol_mult", [1.5, 2.5]), ("pivot", [3, 7]),
                ("swing", [1.0, 2.0]), ("ce_p", [12, 20]), ("ce_m", [3.5, 5.5]), ("tp", [1.5, 3.0, None]),
                ("cool", [8, 24])]:
    for v in vals:
        VARIANTS.append((f"{k}={v}", {k: v}))


def signal(df, p):
    ip = copy.deepcopy(config.STRATEGY_PARAMS["ichimoku"])
    ip.update({"tenkan": p["tenkan"], "kijun": p["kijun"], "senkou_b": p["senkou_b"], "volume_mult": p["vol_mult"]})
    mp = copy.deepcopy(config.STRATEGY_PARAMS["market_structure"])
    mp.update({"pivot_lookback": p["pivot"], "min_swing_pct": p["swing"]})
    a = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, ip), p["cool"])
    b = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, mp), p["cool"])
    return (a & b).values


def trades_and_random(O, H, L, C, sig, allow, stop, tp, liquid, start, seed=7):
    T, N = O.shape
    rng = np.random.default_rng(seed)
    tr, rtr = [], []
    for j in range(N):
        ok = sig[:, j] & allow[:, j] & liquid[:, j]
        ok[:start] = False
        ent = np.where(ok)[0]
        tr += sim_trades(O[:, j], H[:, j], L[:, j], C[:, j], ent, stop[:, j], stop[:, j], tp, MAX_HOLD)
        pool = np.where(allow[:, j] & liquid[:, j] & ~np.isnan(O[:, j]))[0]
        pool = pool[pool >= start]
        k = min(len(ent), len(pool))
        if k:
            rent = np.sort(rng.choice(pool, size=k, replace=False))
            rtr += sim_trades(O[:, j], H[:, j], L[:, j], C[:, j], rent, stop[:, j], stop[:, j], tp, MAX_HOLD)
    return tr, rtr


def main():
    from data_fetcher import get_exchange, get_coin_list
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    ex = get_exchange()
    coins = [c for c in get_coin_list(ex) if c.split("/")[0].upper() not in STABLES][:TOP_N_COINS]
    for must in ("BTC/USDT", "ETH/USDT"):
        if must not in coins:
            coins.insert(0, must)
    h4, d1, fails = {}, {}, 0
    for k, sym in enumerate(coins, 1):
        try:
            h = fetch_full(ex, sym, "4h", H4_LIMIT)
            if h is not None and len(h) >= 600:
                h4[sym] = norm(h)
            d = fetch_full(ex, sym, "1d", D_LIMIT)
            if d is not None and len(d) >= 150:
                d1[sym] = norm(d)
        except Exception as e:
            fails += 1
            print(f"[{k}] {sym}: SKIP ({e})")
        if k % 25 == 0:
            print(f"[{k}/{len(coins)}] data...")

    emit("=" * 100)
    emit("4H ICHIMOKU + MARKET STRUCTURE - PAKKI TASDEEQ")
    emit("=" * 100)
    emit(f"fetch errors: {fails}/{len(coins)} | fee {FEE*100:.2f}% + slip {SLIP*100:.2f}% har taraf + stop slip {STOP_SLIP*100:.2f}%")
    for l in data_report("4H", h4, H4_LIMIT, "4h") + data_report("DAILY", d1, D_LIMIT, "1d"):
        emit(l)

    # ---------------- 4H panel ----------------
    idx = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in h4.values()])))
    syms, O, H, L, C, V = build_panel(h4, idx)
    T, N = O.shape
    liquid = liquid_mask(C, V, hist_min=60 * BPD, window=30 * BPD, min_periods=20 * BPD)
    mom = pd.DataFrame(C).pct_change(60 * BPD, fill_method=None).values
    start = int(max(np.argmax(liquid.sum(1) >= 30), 60 * BPD))
    allow = np.ones((T, N), bool)
    fold_bounds = pd.date_range(idx[start], idx[-1], periods=N_FOLDS + 1).values
    emit(f"\n4H period: {idx[start].date()} -> {idx[-1].date()}  (4 folds: "
         + " | ".join(str(pd.Timestamp(b).date()) for b in fold_bounds) + ")")

    stop_cache = {}

    def stops_for(p, m):
        if (p, m) not in stop_cache:
            st = np.full((T, N), np.nan)
            for j, s in enumerate(syms):
                st[idx.get_indexer(h4[s]["timestamp"]), j] = chandelier(h4[s], p, m)
            stop_cache[(p, m)] = st
        return stop_cache[(p, m)]

    sig_cache = {}

    def sig_for(p):
        key = (p["tenkan"], p["kijun"], p["vol_mult"], p["pivot"], p["swing"], p["cool"])
        if key not in sig_cache:
            sg = np.zeros((T, N), bool)
            for j, s in enumerate(syms):
                try:
                    sg[idx.get_indexer(h4[s]["timestamp"]), j] = signal(h4[s], p)
                except Exception as e:
                    print(f"{s}: signal fail ({e})")
            sig_cache[key] = sg
        return sig_cache[key]

    emit("\n" + "#" * 100)
    emit("PARAMETER PADOS (har line mein sirf EK setting badli, baqi production jaisi)")
    emit("#" * 100)
    base_eq = None
    rows = []
    for vname, change in VARIANTS:
        p = dict(BASE, **change)
        sig = sig_for(p)
        stop = stops_for(p["ce_p"], p["ce_m"])
        tr, rtr = trades_and_random(O, H, L, C, sig, allow, stop, p["tp"], liquid, start)
        s = per_trade_block(tr, rtr, fold_bounds, idx.values)
        eq, ntr = portfolio(O, H, L, C, sig, allow, stop, stop, p["tp"], liquid, mom, start, MAX_HOLD)
        ps = eq_stats(eq[start:], idx[start:], PER_YEAR)
        extra = ""
        if vname.startswith("BASELINE"):
            base_eq = pd.Series(eq[start:], index=idx[start:])
            rsh = []
            for sd in range(N_RANDOM):
                req, _ = portfolio(O, H, L, C, sig, allow, stop, stop, p["tp"], liquid, mom, start, MAX_HOLD,
                                   rng=np.random.default_rng(sd))
                rsh.append(eq_stats(req[start:], idx[start:], PER_YEAR)["sharpe"])
            r95 = float(np.nanpercentile(rsh, 95))
            extra = f" | random portfolio Sharpe 95th={r95:.2f} -> " + ("BEHTAR" if ps["sharpe"] > r95 else "behtar NAHI")
        ok = trade_pass(s)
        rows.append((vname, s, ps, ok))
        if s is None:
            emit(f"\n{vname}: koi trade nahi")
            continue
        folds = " / ".join(f"{fmt(fp_)}({fn})" for fp_, fn in zip(s["fp"], s["fn"]))
        yr = " ".join(f"{y}:{v*100:+.0f}%" for y, v in ps["yearly"].items())
        emit(f"\n{'[theek]' if ok else '[FAIL] '} {vname}")
        emit(f"   Har trade: n={s['n']} win={s['win']:.1f}% PF={fmt(s['pf'])} p5={fmt(s['p5'])} "
             f"Top10-hata={fmt(s['t10'])} RandomPF={fmt(s['rpf'])} Exp={s['exp']:+.2f}% | Folds: {folds}")
        emit(f"   Portfolio: CAGR={ps['cagr']*100:+.1f}% MaxDD={ps['dd']*100:.1f}% Sharpe={fmt(ps['sharpe'])} trades={ntr}{extra}")
        emit(f"   Saal-war: {yr}")

    n_ok = sum(1 for r in rows if r[3])
    pfs = [r[1]["pf"] for r in rows if r[1] is not None and r[1]["pf"] is not None]
    if not pfs:
        pfs = [float("nan")]
    emit(f"\n--- PADOS KA KHULASA: {n_ok}/{len(rows)} settings theek | PF range {min(pfs):.2f} - {max(pfs):.2f} "
         f"| PF>1.3 wali settings: {sum(1 for x in pfs if x > 1.3)}/{len(pfs)} ---")
    emit("(Agar zyada tar pados wali settings bhi theek hain to edge asli hai; agar sirf baseline theek hai to ittefaq)")

    # ---------------- Daily Donchian + milaap ----------------
    didx = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in d1.values()])))
    dsyms, dO, dH, dL, dC, dV = build_panel(d1, didx)
    dT, dN = dO.shape
    dliq = liquid_mask(dC, dV, hist_min=60, window=30, min_periods=20)
    dmom = pd.DataFrame(dC).pct_change(60, fill_method=None).values
    dstart = int(max(np.argmax(dliq.sum(1) >= 30), 60))
    bt = pd.Series(dC[:, dsyms.index("BTC/USDT")], index=didx).ffill()
    reg = np.repeat((bt > ema(bt, 50)).values[:, None], dN, axis=1)
    dsig = np.zeros((dT, dN), bool)
    dstop = np.full((dT, dN), np.nan)
    for j, s in enumerate(dsyms):
        d = d1[s]
        pos = didx.get_indexer(d["timestamp"])
        brk = (d["close"] > d["high"].shift(1).rolling(20).max()).fillna(False)
        dsig[pos, j] = (brk & ~brk.shift(1, fill_value=False)).values
        dstop[pos, j] = chandelier(d, 22, 4.0)
    deq, _ = portfolio(dO, dH, dL, dC, dsig, reg, dstop, dstop, None, dliq, dmom, dstart, 365)
    don = pd.Series(deq[dstart:], index=didx[dstart:])

    # 4H equity ko din ke aakhir (band candle 20:00 = din ka close) par le aao
    ich_d = base_eq.resample("1D").last().dropna()
    common = ich_d.index.intersection(don.index)
    a = don.loc[common].pct_change().fillna(0)
    b = ich_d.loc[common].pct_change().fillna(0)
    comb = (1 + 0.5 * a + 0.5 * b).cumprod()
    cidx = common

    emit("\n" + "#" * 100)
    emit(f"MILAAP: Daily Donchian 20 + 4H Ichimoku+MS  (mushtarka period {cidx[0].date()} -> {cidx[-1].date()})")
    emit("#" * 100)
    emit(f"Dono ke rozana returns ka correlation: {a.corr(b):.2f}  (0 ke qareeb = alag alag chalti hain = milane ka faida)")
    for name, eqs in (("Sirf Daily Donchian 20", (1 + a).cumprod()), ("Sirf 4H Ichimoku+MS", (1 + b).cumprod()),
                      ("50% Donchian + 50% Ichimoku", comb)):
        st = eq_stats(eqs.values, cidx, 365)
        yr = " ".join(f"{y}:{v*100:+.0f}%" for y, v in st["yearly"].items())
        emit(f"{name:30s} CAGR={st['cagr']*100:+6.1f}% MaxDD={st['dd']*100:6.1f}% Sharpe={fmt(st['sharpe'])} | {yr}")

    with open("ichimoku4h_validation_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[SAVE] ichimoku4h_validation_RESULTS.txt")


if __name__ == "__main__":
    main()
