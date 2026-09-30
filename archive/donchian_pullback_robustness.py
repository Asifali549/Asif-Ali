"""
Donchian Breakout + Pullback-in-Uptrend - Robustness Hardening.

WAJAH: CE Buy-Only v2 (filtered redesign) bhi FAIL ho gaya - filter lagane
ke baad bhi koi (period, multiplier) combo sab 4 folds mein PF>1 nahi de
saka, aur filtered trades ka sample itna chhota (7-300) tha ke fold-wise
numbers (PF 0.00 se 16.53 tak) khud hi noise sabit karte hain. CE Buy-Only
ka chandelier-cross entry mechanism khud structurally bekaar hai - filter
lagana bhi usay theek nahi kar saka. Is idea ko yahan permanently chhorte
hain.

Naya, ZYADA HONESHMAND raasta: jo 2 systems audit mein WAAQAI mustaqil edge
dikha chuke hain (Donchian Breakout PF~2.9, Pullback-in-Uptrend PF~1.9,
dono top-10-outlier-removed hone ke baad bhi PF>1), UNHEIN isi bootstrap +
parameter-grid framework se hard-test karte hain - taake:
  1) pata chale unka asal parameter (Donchian channel_period=20,
     Pullback ema_period=20/tolerance=1.0%) kitna "robust" hai (paas wali
     values par bhi PF>1 rehta hai, ya sirf ittefaq hai)
  2) bootstrap confidence interval se maloom ho unka PF number statistically
     kitna bharosemand hai (sirf ek point-estimate nahi)
  3) Donchian ke liye khaas taur par flag ki gayi "survivorship bias" ka
     khadsha aur Pullback ke "zyada tar trades sirf 1 fold mein" wala
     masla is baar per-fold breakdown mein saaf nazar aayega

Entry/exit chandelier trailing stop (period=16, multiplier=4.5 - CE_PB,
production mein dono systems isi par tasdeeq-shuda) FIXED rakha gaya hai -
sirf har strategy ka apna entry-parameter (Donchian: channel_period,
Pullback: ema_period + tolerance_pct) grid mein test hota hai. Filter
hamesha ETH Regime + RS Trend + RS Percentile>=95 (production jaisa).

Result 'donchian_pullback_robustness_RESULTS.txt' mein save hota hai.
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

# Trailing exit - production-tasdeeq-shuda, sabit (dono systems isi par hain)
EXIT_PARAMS = {"period": 16, "multiplier": 4.5}

# Har strategy ka apna parameter grid (production value darmiyan mein hai)
DONCHIAN_GRID = [15, 20, 25]                              # channel_period
PULLBACK_EMA_GRID = [15, 20, 25]                          # pullback_ema_period
PULLBACK_TOL_GRID = [0.5, 1.0, 1.5]                       # tolerance_pct


def chandelier(df, period, mult):
    atr = live.compute_atr(df, period)
    return df["high"].rolling(period).max() - mult * atr


def simulate(df, positions, ce_period, ce_mult):
    """Saanjha simulate: gap-fill fix (exit=min(stop, bar open)) + invalid-setup
    skip, exactly wahi jo production/backtest_engine mein hai."""
    stop_series = chandelier(df, ce_period, ce_mult).values
    op, lo, cl = df["open"].values, df["low"].values, df["close"].values
    n = len(df)
    out = []
    for i in positions:
        if i + 1 >= n or np.isnan(stop_series[i]):
            continue
        eb = i + 1
        entry = op[eb] * (1 + SLIP)
        init = float(stop_series[i])
        if init >= entry:
            continue
        trail = init
        exit_px, exit_bar = None, None
        last = min(eb + AUDIT_MAX_HOLD_BARS, n)
        for j in range(eb, last):
            if lo[j] <= trail:
                exit_px, exit_bar = min(trail, op[j]), j   # GAP FILL
                break
            if not np.isnan(stop_series[j]) and stop_series[j] > trail:
                trail = float(stop_series[j])
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
            continue
        pfs.append(wins.sum() / abs(losses.sum()))
    if len(pfs) < n_boot * 0.5:
        return None, None, None
    pfs = np.array(pfs)
    return round(np.percentile(pfs, 5), 3), round(np.percentile(pfs, 50), 3), round(np.percentile(pfs, 95), 3)


def top10_removed_pf(rets):
    """Top-10 sab se zyada munafa wali trades nikal kar PF - taake pata chale
    edge chand outlier trades par to nahi khara (jaisa pehle audit mein CE
    Buy-Only aur AdvancedConfluence ke liye check kiya gaya tha)."""
    rets = sorted(rets, reverse=True)
    trimmed = rets[10:] if len(rets) > 10 else []
    pf, cnt, wr = pf_stats(trimmed)
    return pf, cnt


def emit_report(lines, label, grid_results, grid_keys, param_names):
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"\n\n########## {label} — PARAMETER SENSITIVITY + BOOTSTRAP CI ##########")
    emit(f"Exit (trailing, fixed): period={EXIT_PARAMS['period']}, mult={EXIT_PARAMS['multiplier']}")
    emit("Filter: ETH Regime + RS Trend + RS Percentile>=95 (production jaisa)\n")

    best_key, best_p5 = None, -999
    for key in grid_keys:
        param_txt = ", ".join(f"{n}={v}" for n, v in zip(param_names, key if isinstance(key, tuple) else (key,)))
        emit(f"\n=== {param_txt} ===")
        trades = grid_results[key]
        rets = [t["ret"] for t in trades]
        pf, cnt, wr = pf_stats(rets)
        if pf is None:
            emit("  koi trade nahi")
            continue
        p5, p50, p95 = bootstrap_pf_ci(rets)
        ci_txt = f"90% CI: [{p5}, {p95}]" if p5 is not None else "CI: N/A (kam trades)"
        emit(f"  trades={cnt}, win={wr:.1f}%, PF={pf:.3f}, {ci_txt}")
        t10_pf, t10_cnt = top10_removed_pf(rets)
        emit(f"  Top-10 nikal kar: PF={t10_pf:.3f} ({t10_cnt} trades)" if t10_pf is not None else "  Top-10 nikal kar: N/A")
        fold_cells = []
        all_folds_positive = True
        for f in range(N_FOLDS):
            fr = [t["ret"] for t in trades if t["fold"] == f]
            fpf, fc, fw = pf_stats(fr)
            if fpf is None or fpf <= 1:
                all_folds_positive = False
            fold_cells.append(f"F{f+1}: {'N/A' if fpf is None else f'{fpf:.2f}'} ({fc})")
        emit("  Folds: " + " | ".join(fold_cells))
        if p5 is not None and all_folds_positive and p5 > best_p5:
            best_p5, best_key = p5, key

    emit(f"\n----- ACCEPTANCE CHECK ({label}) -----")
    if best_key is not None:
        param_txt = ", ".join(f"{n}={v}" for n, v in zip(param_names, best_key if isinstance(best_key, tuple) else (best_key,)))
        emit(f"Sab se mazboot combo: {param_txt} (bootstrap 5th percentile PF = {best_p5})")
        if best_p5 > 1.2:
            emit("-> ACCEPTANCE CRITERIA POORE: har fold PF>1 aur bootstrap ki nichli hadd bhi >1.2 hai.")
        elif best_p5 > 1.0:
            emit("-> Border-line: nichli hadd >1 hai lekin 1.2 se kam - ehtiyat se azmayein.")
        else:
            emit("-> Koi combo criteria poore nahi karta.")
    else:
        emit("Koi combo mein sab 4 folds PF>1 nahi mile.")

    return lines


def main():
    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"{len(coins)} coins, {TF}, {CANDLE_LIMIT} candles\n")

    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DAILY_LIMIT)
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=DAILY_LIMIT)
    eth_regime = live.compute_eth_regime(eth_daily, ema_period=live.ETH_EMA_PERIOD)

    donchian_results = {cp: [] for cp in DONCHIAN_GRID}
    pullback_results = {(pe, tol): [] for pe in PULLBACK_EMA_GRID for tol in PULLBACK_TOL_GRID}

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
            continue
        if rs_bullish is None:
            continue

        def filtered_positions(sig):
            pos_all = list(np.where(apply_cooldown(sig, config.SIGNAL_COOLDOWN_BARS).values)[0])
            return [i for i in pos_all
                    if live.passes_full_filters(pd.Timestamp(ts_col.iloc[i]), eth_regime, rs_bullish, rs_ratio)[0]]

        for cp in DONCHIAN_GRID:
            try:
                sig = live.donchian_channel_breakout(df, channel_period=cp)
                pos = filtered_positions(sig)
                trades = simulate(df, pos, EXIT_PARAMS["period"], EXIT_PARAMS["multiplier"])
                for t in trades:
                    t["fold"] = fold_of(t["i"], n)
                donchian_results[cp].extend(trades)
            except Exception as e:
                print(f"  [SKIP-Donchian-{cp}] {symbol}: {e}")

        for pe in PULLBACK_EMA_GRID:
            for tol in PULLBACK_TOL_GRID:
                try:
                    sig = live.pullback_uptrend_entry(df, pullback_ema_period=pe, tolerance_pct=tol)
                    pos = filtered_positions(sig)
                    trades = simulate(df, pos, EXIT_PARAMS["period"], EXIT_PARAMS["multiplier"])
                    for t in trades:
                        t["fold"] = fold_of(t["i"], n)
                    pullback_results[(pe, tol)].extend(trades)
                except Exception as e:
                    print(f"  [SKIP-Pullback-{pe}/{tol}] {symbol}: {e}")

        if k % 10 == 0 or k == len(coins):
            print(f"[{k}/{len(coins)}] ... done")

    lines = []
    emit_report(lines, "DONCHIAN BREAKOUT", donchian_results, DONCHIAN_GRID, ["channel_period"])
    emit_report(lines, "PULLBACK-IN-UPTREND", pullback_results, list(pullback_results.keys()),
                ["ema_period", "tolerance_pct"])

    with open("donchian_pullback_robustness_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[SAVE] donchian_pullback_robustness_RESULTS.txt")


if __name__ == "__main__":
    main()
