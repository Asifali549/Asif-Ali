"""
CE Buy-Only v2 (FILTERED redesign) + Robustness Framework.

WAJAH: v1 (khaali Chandelier-cross, koi filter nahi) audit mein PF 0.40
nikla - koi asal edge nahi, sirf bug ka natija tha. Donchian/Pullback
dono mein ETH Regime + RS Trend + RS Percentile>=95 filters hain, aur
wahi 2 systems audit mein mustaqil (consistent) nikle. Is liye v2 mein
CE Buy-Only ke Chandelier-cross entry par WAHI SABIT-SHUDA filters lagate
hain - andaza nahi, khud ke data se saabit formula.

ROBUSTNESS FRAMEWORK (is script ka doosra hissa, har system ke liye
dobara istemal ho sakta hai):

  1) PARAMETER SENSITIVITY GRID: sirf EK (period, multiplier) par
     bharosa nahi - aas paas ki values bhi test karte hain. Agar result
     sirf ek khaas number par acha ho aur paas wali value par toot
     jaye, to wo ittefaq (overfitting) hai, asal edge nahi.

  2) BOOTSTRAP CONFIDENCE INTERVAL: sirf ek PF number kaafi nahi. Trades
     ko hazaron baar dobara (with replacement) sample kar ke dekhte hain
     PF kis range mein yaqeen se rehta hai. Agar NICHLI hadd (5th
     percentile) bhi >1 ho, tabhi haqeeqi edge maante hain.

ACCEPTANCE CRITERIA (Phase C, pehle se tay):
  - Har 4 folds mein PF > 1
  - Bootstrap 5th percentile PF > 1.0 (behtar: > 1.2)
  - Parameter thoda idhar-udhar karne par result na toote
  (Agar ye teeno pooray hon, tabhi "trustworthy" mana jayega)

Result 'ce_v2_and_robustness_RESULTS.txt' mein save hota hai.
"""

import numpy as np
import pandas as pd

import config
import scheduled_dashboard_scan as live
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown

TOP_N_COINS = 150
TF = "1h"
CANDLE_LIMIT = 8760
DAILY_LIMIT = 800
N_FOLDS = 4
AUDIT_MAX_HOLD_BARS = 500
FEE = config.BACKTEST_PARAMS["fee_pct"] / 100
SLIP = config.BACKTEST_PARAMS["slippage_pct"] / 100

N_BOOTSTRAP = 5000

# Parameter grid - entry period/multiplier ke aas-paas ki values
ENTRY_PERIOD_GRID = [9, 11, 13]
ENTRY_MULT_GRID = [3.5, 4.5, 5.5]
EXIT_PARAMS = {"period": 16, "multiplier": 3.0}   # sabit rakha (Donchian/Pullback jaisa exit hai)


def chandelier(df, period, mult):
    atr = live.compute_atr(df, period)
    return df["high"].rolling(period).max() - mult * atr


def simulate(df, positions, entry_p, entry_m, exit_p, exit_m):
    """CE Buy-Only ka apna simulate: trailing-only exit (jaisa production/audit mein hai),
    GAP-FILL fix (exit = min(stop, bar open)) aur invalid-setup skip dono shamil."""
    entry_stop = chandelier(df, entry_p, entry_m).values
    exit_stop = entry_stop if (entry_p, entry_m) == (exit_p, exit_m) else chandelier(df, exit_p, exit_m).values
    op, lo, cl = df["open"].values, df["low"].values, df["close"].values
    n = len(df)
    out = []
    for i in positions:
        if i + 1 >= n or np.isnan(entry_stop[i]):
            continue
        eb = i + 1
        entry = op[eb] * (1 + SLIP)
        init = float(entry_stop[i])
        if init >= entry:
            continue
        trail = init
        exit_px, exit_bar = None, None
        last = min(eb + AUDIT_MAX_HOLD_BARS, n)
        for j in range(eb, last):
            if lo[j] <= trail:
                exit_px, exit_bar = min(trail, op[j]), j   # GAP FILL
                break
            if not np.isnan(exit_stop[j]) and exit_stop[j] > trail:
                trail = float(exit_stop[j])
        if exit_px is None:
            exit_bar = last - 1
            exit_px = cl[exit_bar]
        exit_px *= (1 - SLIP)
        ret = ((exit_px - entry) / entry - 2 * FEE) * 100
        out.append({"i": i, "ret": ret, "bars": exit_bar - eb})
    return out


def fold_of(i, n):
    bounds = [int(n * k / N_FOLDS) for k in range(N_FOLDS + 1)]
    for f in range(N_FOLDS):
        if bounds[f] <= i < bounds[f + 1]:
            return f
    return N_FOLDS - 1


def pf_stats(rets):
    rets = np.asarray(rets, dtype=float)
    if len(rets) == 0:
        return None, 0, None
    wins, losses = rets[rets > 0], rets[rets <= 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else None
    return pf, len(rets), len(wins) / len(rets) * 100


def bootstrap_pf_ci(rets, n_boot=N_BOOTSTRAP, seed=42):
    """Trades ko baar baar (with replacement) sample kar ke PF ki distribution
    nikalta hai. Returns (p5, p50, p95) - 90% confidence interval."""
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 10:
        return None, None, None
    rng = np.random.default_rng(seed)
    n = len(rets)
    pfs = []
    for _ in range(n_boot):
        sample = rets[rng.integers(0, n, n)]
        wins, losses = sample[sample > 0], sample[sample <= 0]
        if len(losses) == 0 or losses.sum() == 0:
            continue   # is sample mein koi loss nahi - PF undefined, skip
        pfs.append(wins.sum() / abs(losses.sum()))
    if len(pfs) < n_boot * 0.5:
        return None, None, None
    pfs = np.array(pfs)
    return round(np.percentile(pfs, 5), 3), round(np.percentile(pfs, 50), 3), round(np.percentile(pfs, 95), 3)


def main():
    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"{len(coins)} coins, {TF}, {CANDLE_LIMIT} candles\n")

    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DAILY_LIMIT)
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=DAILY_LIMIT)
    eth_regime = live.compute_eth_regime(eth_daily, ema_period=live.ETH_EMA_PERIOD)

    # Grid ke har combo ke liye: filtered (RS%95+ETH) aur unfiltered dono record karte hain,
    # taake fark bhi dikhe ke filter kitna faida deta hai.
    grid_results = {(ep, em): {"filtered": [], "unfiltered": []}
                     for ep in ENTRY_PERIOD_GRID for em in ENTRY_MULT_GRID}

    for k, symbol in enumerate(coins, 1):
        try:
            df = fetch_ohlcv(exchange, symbol, TF, limit=CANDLE_LIMIT)
        except Exception as e:
            print(f"[{k}/{len(coins)}] {symbol}: fetch fail ({e})")
            continue
        if df is None or len(df) < 500:
            continue
        n = len(df)
        ts_col = df["timestamp"]

        rs_bullish = rs_ratio = None
        try:
            coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=DAILY_LIMIT)
            rs_bullish, rs_ratio = live.compute_rs_trend_and_ratio(coin_daily, btc_daily, ema_period=live.RS_EMA_PERIOD)
        except Exception as e:
            print(f"  [RS-FAIL] {symbol}: {e}")

        for ep in ENTRY_PERIOD_GRID:
            for em in ENTRY_MULT_GRID:
                try:
                    ent = chandelier(df, ep, em)
                    cx = ((df["close"] > ent) & (df["close"].shift(1) <= ent.shift(1))).fillna(False)
                    sig = apply_cooldown(cx, config.SIGNAL_COOLDOWN_BARS)
                    pos_all = list(np.where(sig.values)[0])

                    trades_unf = simulate(df, pos_all, ep, em, EXIT_PARAMS["period"], EXIT_PARAMS["multiplier"])
                    for t in trades_unf:
                        t["fold"] = fold_of(t["i"], n)
                    grid_results[(ep, em)]["unfiltered"].extend(trades_unf)

                    if rs_bullish is not None:
                        filt_pos = [i for i in pos_all
                                    if live.passes_full_filters(pd.Timestamp(ts_col.iloc[i]), eth_regime, rs_bullish, rs_ratio)[0]]
                        trades_f = simulate(df, filt_pos, ep, em, EXIT_PARAMS["period"], EXIT_PARAMS["multiplier"])
                        for t in trades_f:
                            t["fold"] = fold_of(t["i"], n)
                        grid_results[(ep, em)]["filtered"].extend(trades_f)
                except Exception as e:
                    print(f"  [SKIP-{ep}/{em}] {symbol}: {e}")

        if k % 10 == 0 or k == len(coins):
            print(f"[{k}/{len(coins)}] ... done")

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("\n\n########## CE BUY-ONLY v2 (FILTERED) + PARAMETER SENSITIVITY + BOOTSTRAP CI ##########")
    emit(f"Filter: ETH Regime + RS Trend + RS Percentile>=95 (Donchian/Pullback jaisa)")
    emit(f"Exit (trailing): period={EXIT_PARAMS['period']}, mult={EXIT_PARAMS['multiplier']} (sabit)")
    emit(f"({len(coins)} coins, {TF}, {CANDLE_LIMIT} candles, fee {FEE*100:.2f}%/side, slip {SLIP*100:.2f}%)\n")

    best_key, best_p5 = None, -999
    for ep in ENTRY_PERIOD_GRID:
        for em in ENTRY_MULT_GRID:
            emit(f"\n=== Entry period={ep}, multiplier={em} ===")
            for label in ("unfiltered", "filtered"):
                trades = grid_results[(ep, em)][label]
                rets = [t["ret"] for t in trades]
                pf, cnt, wr = pf_stats(rets)
                if pf is None:
                    emit(f"  [{label:10s}] koi trade nahi")
                    continue
                p5, p50, p95 = bootstrap_pf_ci(rets)
                ci_txt = f"90% CI: [{p5}, {p95}]" if p5 is not None else "CI: N/A (kam trades)"
                emit(f"  [{label:10s}] trades={cnt}, win={wr:.1f}%, PF={pf:.3f}, {ci_txt}")
                fold_cells = []
                all_folds_positive = True
                for f in range(N_FOLDS):
                    fr = [t["ret"] for t in trades if t["fold"] == f]
                    fpf, fc, fw = pf_stats(fr)
                    if fpf is None or fpf <= 1:
                        all_folds_positive = False
                    fold_cells.append(f"F{f+1}: {'N/A' if fpf is None else f'{fpf:.2f}'} ({fc})")
                emit("    Folds: " + " | ".join(fold_cells))
                if label == "filtered" and p5 is not None and all_folds_positive and p5 > best_p5:
                    best_p5, best_key = p5, (ep, em)

    emit("\n\n----- ACCEPTANCE CHECK (Phase C criteria) -----")
    if best_key is not None:
        emit(f"Sab se mazboot filtered combo: Entry period={best_key[0]}, multiplier={best_key[1]} "
             f"(bootstrap 5th percentile PF = {best_p5})")
        if best_p5 > 1.2:
            emit("-> ACCEPTANCE CRITERIA POORE: har fold PF>1 aur bootstrap ki nichli hadd bhi >1.2 hai.")
            emit("   Ye combo aage manual/auto bot mein CE Buy-Only v2 ke tor par azmaya ja sakta hai.")
        elif best_p5 > 1.0:
            emit("-> Border-line: nichli hadd >1 hai lekin 1.2 se kam - ehtiyat se, chhote size se hi azmayein.")
        else:
            emit("-> Koi combo criteria poore nahi karta - filter lagane ke baad bhi CE Buy-Only ka "
                 "haqeeqi edge sabit nahi hota. Is design ko yahin chhor dena behtar hoga.")
    else:
        emit("Koi filtered combo mein sab 4 folds PF>1 nahi mile - CE Buy-Only (chahe filter ke sath) "
             "abhi tak koi mustaqil edge nahi dikha raha.")

    with open("ce_v2_and_robustness_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[SAVE] ce_v2_and_robustness_RESULTS.txt")


if __name__ == "__main__":
    main()
