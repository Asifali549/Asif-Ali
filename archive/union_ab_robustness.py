"""
Union AB + Union AB Backup Tier - Robustness Hardening (same framework as
Donchian/Pullback, ab in do "confluence" systems par).

Union AB do combos se bana hai (dono production mein aik sath chalte hain):
  - Ichimoku+MS  (ichimoku signal AND market_structure signal, exit CE_A)
  - EMA+Breakout (ema_crossover signal AND breakout signal, exit CE_B)

Har combo do filter-tier ke sath test hota hai (jaisa production mein hai):
  - "Baseline"    -> ETH Regime + RS Trend + RS Percentile>=95 (ziyada mehfooz)
  - "Backup Tier" -> sirf RS Trend + RS Percentile>=95 (ETH check NAHI - halka)

(52-week-high wala "Baseline+52W" sub-tier yahan test nahi kiya - wo Baseline
ka hi ek tang (tighter) upar wala tabqa hai, is script ka maqsad khud combo +
tier ki bunyadi mazbooti dekhna hai.)

Exit mode Union AB mein Donchian/Pullback se ALAG hai: sirf trailing nahi,
balkay FIXED Take-Profit (Entry + Risk*RR_MULTIPLE) bhi hai - jo bhi pehle
lage (production compute_trade_progress jaisa, use_fixed_tp=True).

Parameter grid: har combo ke apne trailing-exit chandelier (period,
multiplier) ke aas-paas ki values test karte hain - agar production ka
number (CE_A=16/4.5, CE_B=12/4.5) sirf ittefaq nahi to paas wali values
par bhi PF>1 rehna chahiye.

Result 'union_ab_robustness_RESULTS.txt' mein save hota hai.
"""

import numpy as np
import pandas as pd

import config
import scheduled_dashboard_scan as live
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown

TOP_N_COINS = 150
TF = "1h"
CANDLE_LIMIT = 8760
DAILY_LIMIT = 800
N_FOLDS = 4
AUDIT_MAX_HOLD_BARS = 500
FEE = config.BACKTEST_PARAMS["fee_pct"] / 100
SLIP = config.BACKTEST_PARAMS["slippage_pct"] / 100
RR_MULTIPLE = live.RR_MULTIPLE   # 2.0, production jaisa

N_BOOTSTRAP = 5000

COMBOS = {
    # combo_name: (grid of (period, multiplier), production default)
    "Ichimoku+MS": {"period_grid": [12, 16, 20], "mult_grid": [3.5, 4.5, 5.5], "default": (16, 4.5)},
    "EMA+Breakout": {"period_grid": [8, 12, 16], "mult_grid": [3.5, 4.5, 5.5], "default": (12, 4.5)},
}


def chandelier(df, period, mult):
    atr = live.compute_atr(df, period)
    return df["high"].rolling(period).max() - mult * atr


def simulate_union(df, positions, ce_period, ce_mult, rr_multiple=RR_MULTIPLE):
    """Union AB ka apna simulate: trailing stop OR fixed TP (Entry+Risk*RR),
    jo pehle lage (SL ko tie mein pehle mana jata hai - production jaisa),
    GAP-FILL fix aur invalid-setup skip dono shamil."""
    stop_series = chandelier(df, ce_period, ce_mult).values
    op, hi, lo, cl = df["open"].values, df["high"].values, df["low"].values, df["close"].values
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
        risk = entry - init
        tp_price = entry + risk * rr_multiple
        trail = init
        exit_px, exit_bar = None, None
        last = min(eb + AUDIT_MAX_HOLD_BARS, n)
        for j in range(eb, last):
            stop_hit = lo[j] <= trail
            tp_hit = hi[j] >= tp_price
            if stop_hit:
                exit_px, exit_bar = min(trail, op[j]), j   # GAP FILL
                break
            if tp_hit:
                exit_px, exit_bar = tp_price, j
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
    rets = sorted(rets, reverse=True)
    trimmed = rets[10:] if len(rets) > 10 else []
    pf, cnt, wr = pf_stats(trimmed)
    return pf, cnt


def emit_grid_report(lines, label, results, param_grid):
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"\n\n########## {label} ##########")
    best_key, best_p5 = None, -999
    for (period, mult) in param_grid:
        emit(f"\n=== period={period}, multiplier={mult} ===")
        trades = results[(period, mult)]
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
            best_p5, best_key = p5, (period, mult)

    emit(f"\n----- ACCEPTANCE CHECK ({label}) -----")
    if best_key is not None:
        emit(f"Sab se mazboot combo: period={best_key[0]}, multiplier={best_key[1]} (bootstrap 5th percentile PF = {best_p5})")
        if best_p5 > 1.2:
            emit("-> ACCEPTANCE CRITERIA POORE: har fold PF>1 aur bootstrap ki nichli hadd bhi >1.2 hai.")
        elif best_p5 > 1.0:
            emit("-> Border-line: nichli hadd >1 hai lekin 1.2 se kam - ehtiyat se azmayein.")
        else:
            emit("-> Koi combo criteria poore nahi karta.")
    else:
        emit("Koi combo mein sab 4 folds PF>1 nahi mile.")


def main():
    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"{len(coins)} coins, {TF}, {CANDLE_LIMIT} candles\n")

    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DAILY_LIMIT)
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=DAILY_LIMIT)
    eth_regime = live.compute_eth_regime(eth_daily, ema_period=live.ETH_EMA_PERIOD)

    # results[combo_name][tier][(period,mult)] = trades list
    results = {
        combo_name: {
            "Baseline (ETH+RS)": {(p, m): [] for p in cfg["period_grid"] for m in cfg["mult_grid"]},
            "Backup Tier (RS only)": {(p, m): [] for p in cfg["period_grid"] for m in cfg["mult_grid"]},
        }
        for combo_name, cfg in COMBOS.items()
    }

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

        try:
            ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
            ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
            ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
            breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)
        except Exception as e:
            print(f"  [SIG-FAIL] {symbol}: {e}")
            continue

        combo_signals = {
            "Ichimoku+MS": ichi_sig & ms_sig,
            "EMA+Breakout": ema_sig & breakout_sig,
        }

        for combo_name, cfg in COMBOS.items():
            combo_sig = combo_signals[combo_name]
            pos_all = list(np.where(combo_sig.values)[0])
            if not pos_all:
                continue

            baseline_pos = [i for i in pos_all
                            if live.passes_full_filters(pd.Timestamp(ts_col.iloc[i]), eth_regime, rs_bullish, rs_ratio)[0]]
            backup_pos = [i for i in pos_all
                          if live.passes_rs_filters(pd.Timestamp(ts_col.iloc[i]), rs_bullish, rs_ratio)[0]]

            for period in cfg["period_grid"]:
                for mult in cfg["mult_grid"]:
                    try:
                        tb = simulate_union(df, baseline_pos, period, mult)
                        for t in tb:
                            t["fold"] = fold_of(t["i"], n)
                        results[combo_name]["Baseline (ETH+RS)"][(period, mult)].extend(tb)

                        tr = simulate_union(df, backup_pos, period, mult)
                        for t in tr:
                            t["fold"] = fold_of(t["i"], n)
                        results[combo_name]["Backup Tier (RS only)"][(period, mult)].extend(tr)
                    except Exception as e:
                        print(f"  [SKIP-{combo_name}-{period}/{mult}] {symbol}: {e}")

        if k % 10 == 0 or k == len(coins):
            print(f"[{k}/{len(coins)}] ... done")

    lines = []
    for combo_name, cfg in COMBOS.items():
        param_grid = [(p, m) for p in cfg["period_grid"] for m in cfg["mult_grid"]]
        for tier_name in ("Baseline (ETH+RS)", "Backup Tier (RS only)"):
            emit_grid_report(lines, f"UNION AB — {combo_name} — {tier_name}", results[combo_name][tier_name], param_grid)

    with open("union_ab_robustness_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[SAVE] union_ab_robustness_RESULTS.txt")


if __name__ == "__main__":
    main()
