"""
AUDIT: SAB live systems ka dobara walk-forward backtest - DARUST (order-fixed)
engine ke sath, aur PRODUCTION ke apne functions (scheduled_dashboard_scan) se
hi signals/filters - taake backtest aur live ek hi logic par hon.

Kyun: Pullback / Donchian / Union AB Backup Tier ke purane nambar (PF 1.83 /
2.17 / 3.04) usi order-bug wale engine se bane the jis ne CE Buy-Only ko
Win93%/PF45 dikhaya tha. Yahan har system dobara napa jata hai.

Systems: Union AB (ETH+RS%95), Union AB Backup Tier (RS%95, ETH nahi),
CE Buy-Only (baseline aur +ETH), Pullback-in-Uptrend, Donchian Breakout,
NEW AdvancedConfluence.

Simulation (sab ke liye ek jaisi):
  - signal bar ke agle bar ke OPEN par entry (+slippage), fee dono taraf
  - initial stop = ENTRY chandelier (signal bar), agar entry se upar to skip
  - har bar: PEHLE low (aur TP) PICHLI maloom stop se check, PHIR stop update
  - trailing = EXIT chandelier; Union AB / NEW mein fixed TP (RR 2.0), baqi trailing-only
  - max hold = 500 bars (live ki lookback jitni; purana 50-bar cap live se mel nahi khata tha)
  - GAP FILL: agar bar ka open stop se neeche ho to exit open par (stop ki qeemat par nahi).
    Pehle version yahan ghalat tha: CE Buy-Only ke Win58%/PF5.3 aur 93% trades 1 bar mein
    band hona isi ki alamat thi (zero-edge random data par bhi Win61%/PF7 aata hai).

Result 'audit_all_systems_RESULTS.txt' mein save hota hai (aur log mein bhi).
"""

import time

import numpy as np
import pandas as pd

import config
import scheduled_dashboard_scan as live
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from confluence_engine import compute_confluence, DEFAULT_PARAMS as CONF_PARAMS

TOP_N_COINS = 150
TF = "1h"
CANDLE_LIMIT = 8760          # ~1 saal
DAILY_LIMIT = 800            # RS percentile (180 din) ki poori history ke liye
N_FOLDS = 4
TRIM_N = 10
AUDIT_MAX_HOLD_BARS = 500
INCLUDE_NEW_CONFLUENCE = True

FEE = config.BACKTEST_PARAMS["fee_pct"] / 100
SLIP = config.BACKTEST_PARAMS["slippage_pct"] / 100

# Purane (order-bug wale) nambar - sirf moazne ke liye print hote hain
OLD_CLAIMS = {
    "Pullback-in-Uptrend": "purana: 799 trades, PF 1.831",
    "Donchian Breakout": "purana: 552 trades, PF 2.171",
    "Union AB Backup Tier": "purana: PF 3.04",
}

SYSTEMS = [
    "Union AB", "Union AB Backup Tier", "CE Buy-Only (baseline)", "CE Buy-Only (+ETH bullish)",
    "Pullback-in-Uptrend", "Donchian Breakout", "NEW AdvancedConfluence",
]


def chandelier(df, period, mult):
    atr = live.compute_atr(df, period)
    return df["high"].rolling(period).max() - mult * atr


def simulate(df, positions, entry_p, entry_m, exit_p, exit_m, use_tp, rr):
    entry_stop = chandelier(df, entry_p, entry_m).values
    if (entry_p, entry_m) == (exit_p, exit_m):
        exit_stop = entry_stop
    else:
        exit_stop = chandelier(df, exit_p, exit_m).values
    op, lo, hi, cl = df["open"].values, df["low"].values, df["high"].values, df["close"].values
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
        tp = entry + rr * (entry - init) if use_tp else None
        trail = init
        exit_px, exit_bar = None, None
        last = min(eb + AUDIT_MAX_HOLD_BARS, n)
        gapped = False
        for j in range(eb, last):
            if lo[j] <= trail:
                # GAP FILL: bar stop se neeche khule to fill open par (stop ki bulandar qeemat par nahi)
                gapped = op[j] < trail
                exit_px, exit_bar = min(trail, op[j]), j
                break
            if use_tp and hi[j] >= tp:
                exit_px, exit_bar = tp, j
                break
            if not np.isnan(exit_stop[j]) and exit_stop[j] > trail:
                trail = float(exit_stop[j])
        if exit_px is None:
            exit_bar = last - 1
            exit_px = cl[exit_bar]
        exit_px *= (1 - SLIP)
        ret = ((exit_px - entry) / entry - 2 * FEE) * 100
        out.append({"i": i, "ret": ret, "bars": exit_bar - eb, "gap": gapped})
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


def main():
    t0 = time.time()
    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"{len(coins)} coins, {TF}, {CANDLE_LIMIT} candles, {N_FOLDS} folds\n")

    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DAILY_LIMIT)
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=DAILY_LIMIT)
    eth_regime = live.compute_eth_regime(eth_daily, ema_period=live.ETH_EMA_PERIOD)

    res = {s: [] for s in SYSTEMS}   # har trade: {ret, bars, fold, symbol}

    def add(system, trades, symbol, n):
        for t in trades:
            res[system].append({"ret": t["ret"], "bars": t["bars"], "fold": fold_of(t["i"], n), "symbol": symbol, "gap": t["gap"]})

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

        rs_bullish = rs_ratio = coin_daily = None
        try:
            coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=DAILY_LIMIT)
            rs_bullish, rs_ratio = live.compute_rs_trend_and_ratio(coin_daily, btc_daily, ema_period=live.RS_EMA_PERIOD)
        except Exception as e:
            print(f"  [RS-FAIL] {symbol}: {e}")

        # ---- Union AB + Backup Tier ----
        try:
            cd = config.SIGNAL_COOLDOWN_BARS
            ichi = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), cd)
            ms = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), cd)
            ema = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), cd)
            brk = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), cd)
            if rs_bullish is not None:
                for combo_sig, ce in ((ichi & ms, live.CE_A), (ema & brk, live.CE_B)):
                    pos_all = np.where(combo_sig.values)[0]
                    ab_pos, bk_pos = [], []
                    for i in pos_all:
                        ts = pd.Timestamp(ts_col.iloc[i])
                        if live.passes_rs_filters(ts, rs_bullish, rs_ratio)[0]:
                            bk_pos.append(i)
                            if live.evaluate_union_ab_tier(symbol, df, i, eth_regime, rs_bullish, rs_ratio, coin_daily)[0] is not None:
                                ab_pos.append(i)
                    add("Union AB", simulate(df, ab_pos, ce["period"], ce["multiplier"], ce["period"], ce["multiplier"], True, live.RR_MULTIPLE), symbol, n)
                    add("Union AB Backup Tier", simulate(df, bk_pos, ce["period"], ce["multiplier"], ce["period"], ce["multiplier"], True, live.RR_MULTIPLE), symbol, n)
        except Exception as e:
            print(f"  [SKIP-UnionAB] {symbol}: {e}")

        # ---- CE Buy-Only ----
        try:
            ent = chandelier(df, live.CE_BUYONLY_ENTRY["period"], live.CE_BUYONLY_ENTRY["multiplier"])
            cx = ((df["close"] > ent) & (df["close"].shift(1) <= ent.shift(1))).fillna(False)
            sig = apply_cooldown(cx, config.SIGNAL_COOLDOWN_BARS)
            pos_all = list(np.where(sig.values)[0])
            args = (live.CE_BUYONLY_ENTRY["period"], live.CE_BUYONLY_ENTRY["multiplier"],
                    live.CE_BUYONLY_EXIT["period"], live.CE_BUYONLY_EXIT["multiplier"], False, None)
            add("CE Buy-Only (baseline)", simulate(df, pos_all, *args), symbol, n)
            eth_pos = [i for i in pos_all if live.is_bullish_at(eth_regime, pd.Timestamp(ts_col.iloc[i]))]
            add("CE Buy-Only (+ETH bullish)", simulate(df, eth_pos, *args), symbol, n)
        except Exception as e:
            print(f"  [SKIP-CE] {symbol}: {e}")

        # ---- Pullback + Donchian ----
        if rs_bullish is not None:
            for name, fn in (("Pullback-in-Uptrend", live.pullback_uptrend_entry),
                             ("Donchian Breakout", live.donchian_channel_breakout)):
                try:
                    sig = apply_cooldown(fn(df), config.SIGNAL_COOLDOWN_BARS)
                    pos = [i for i in np.where(sig.values)[0]
                           if live.passes_full_filters(pd.Timestamp(ts_col.iloc[i]), eth_regime, rs_bullish, rs_ratio)[0]]
                    ce = live.CE_PB
                    add(name, simulate(df, pos, ce["period"], ce["multiplier"], ce["period"], ce["multiplier"], False, None), symbol, n)
                except Exception as e:
                    print(f"  [SKIP-{name}] {symbol}: {e}")

        # ---- NEW AdvancedConfluence ----
        if INCLUDE_NEW_CONFLUENCE:
            try:
                r = compute_confluence(df, btc_daily, CONF_PARAMS, usdt_d_weak=None)
                sig = apply_cooldown(r["choch"] & (r["score"] >= CONF_PARAMS["score_threshold"]), config.SIGNAL_COOLDOWN_BARS)
                ce = live.CE_D
                add("NEW AdvancedConfluence", simulate(df, list(np.where(sig.values)[0]),
                    ce["period"], ce["multiplier"], ce["period"], ce["multiplier"], True, live.RR_MULTIPLE), symbol, n)
            except Exception as e:
                print(f"  [SKIP-NEW] {symbol}: {e}")

        if k % 10 == 0 or k == len(coins):
            print(f"[{k}/{len(coins)}] ... {(time.time() - t0) / 60:.1f} min")

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("\n\n########## AUDIT - SAB SYSTEMS (order-fixed engine, production filters) ##########")
    emit(f"({len(coins)} coins pool, {TF}, {CANDLE_LIMIT} candles, fee {FEE*100:.2f}%/side, slip {SLIP*100:.2f}%, max hold {AUDIT_MAX_HOLD_BARS} bars)\n")

    for s in SYSTEMS:
        trades = res[s]
        emit(f"\n=== {s} ===" + (f"   [{OLD_CLAIMS[s]}]" if s in OLD_CLAIMS else ""))
        if not trades:
            emit("  koi trade nahi")
            continue
        rets = np.array([t["ret"] for t in trades])
        pf, cnt, wr = pf_stats(rets)
        pf_txt = f"{pf:.3f}" if pf is not None else "N/A"
        emit(f"  POORA: trades={cnt}, win={wr:.1f}%, PF={pf_txt}, expectancy={rets.mean():+.3f}%/trade, "
             f"avg bars={np.mean([t['bars'] for t in trades]):.1f}")

        srt = np.sort(rets)[::-1]
        trimmed = srt[TRIM_N:]
        pf_t, cnt_t, wr_t = pf_stats(trimmed)
        gross_win = rets[rets > 0].sum()
        share = srt[:TRIM_N][srt[:TRIM_N] > 0].sum() / gross_win * 100 if gross_win > 0 else 0
        if pf_t is not None:
            emit(f"  Top-{TRIM_N} trades nikal kar: PF={pf_t:.3f} (in {TRIM_N} trades ka gross-profit mein hissa {share:.1f}%)")

        fold_cells = []
        for f in range(N_FOLDS):
            fr = [t["ret"] for t in trades if t["fold"] == f]
            fpf, fc, fw = pf_stats(fr)
            fold_cells.append(f"F{f+1}: {'N/A' if fpf is None else f'{fpf:.2f}'} ({fc}, {0 if fw is None else fw:.0f}%)")
        emit("  Folds: " + " | ".join(fold_cells))

        quick = np.mean([t["bars"] <= 1 for t in trades]) * 100
        emit(f"  1 bar ke andar band hone wali trades: {quick:.1f}%")
        gap_share = np.mean([t["gap"] for t in trades]) * 100
        emit(f"  Gap-through exits (bar stop se neeche khula, fill open par): {gap_share:.1f}%")

    with open("audit_all_systems_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] audit_all_systems_RESULTS.txt ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
