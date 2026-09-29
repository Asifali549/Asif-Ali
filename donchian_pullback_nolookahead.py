"""
Donchian Breakout + Pullback-in-Uptrend - LOOKAHEAD-FREE dobara jaanch
======================================================================
CHoCH test ne sabit kiya ke purane daily filters (ETH Regime, RS Trend,
RS Percentile>=95) mein lookahead tha: daily candle ka timestamp din ke
SHURU (00:00) ka hota hai, lekin purana code usi din ke 1h signals par
din ke AAKHIR ka close istemal kar leta tha. CHoCH par iska asar:
RS-only PF 1.81 -> 1.15 (sahi karne ke baad).

Donchian (PF ~2.8) aur Pullback (PF ~2.3) ke purane robustness natije
BILKUL ISI filter par bane the - is liye ab dono ko sahi (band-candle)
filter ke sath dobara jaancha ja raha hai.

FILTER TIERS (sab lookahead-free, siwaye [OLD] ke):
  - "No filter"
  - "ETH only"            - sirf ETH > daily EMA200
  - "RS trend"            - sirf coin/BTC ratio > EMA50 (percentile NAHI)
  - "ETH+RS trend"        - ETH + RS trend (percentile NAHI)
  - "RS trend+%95"        - RS trend + RS Percentile>=95  (Backup Tier)
  - "ETH+RS+%95"          - production ka poora filter
  - "ETH+RS+%95 [OLD]"    - production filter PURANE (lookahead) tareeqe se -
                            ye purane natije (~2.8 / ~2.3) dobara bana kar
                            tasdeeq karta hai ke farq sirf lookahead ka hai

Exit: production CE_PB trailing (period=16, mult=4.5), fixed.
Entry grid: Donchian channel 15/20/25; Pullback EMA 15/20/25 x tolerance 0.5/1.0/1.5.
Natija: donchian_pullback_nolookahead_RESULTS.txt
"""

import numpy as np
import pandas as pd

import config
import scheduled_dashboard_scan as live
from strategies import apply_cooldown
from choch_robustness import (
    _norm, _ns, chandelier_arr, simulate, summarize, passes, fmt,
    DAILY_CLOSE_SHIFT, N_FOLDS, MIN_TOTAL_TRADES, MIN_FOLD_TRADES,
)

TOP_N_COINS = 150
TF = "1h"
CANDLE_LIMIT = 8760
DAILY_LIMIT = 800
COOLDOWN = config.SIGNAL_COOLDOWN_BARS
EXIT = (live.CE_PB["period"], live.CE_PB["multiplier"])

ENTRIES = [f"Donchian ch={cp}" for cp in (15, 20, 25)] + \
          [f"Pullback ema={pe} tol={tol}" for pe in (15, 20, 25) for tol in (0.5, 1.0, 1.5)]
TIERS = ["No filter", "ETH only", "RS trend", "ETH+RS trend", "RS trend+%95", "ETH+RS+%95", "ETH+RS+%95 [OLD]"]


def entry_signal(df, name):
    if name.startswith("Donchian"):
        cp = int(name.split("=")[1])
        return live.donchian_channel_breakout(df, channel_period=cp)
    parts = name.split()
    pe = int(parts[1].split("=")[1])
    tol = float(parts[2].split("=")[1])
    return live.pullback_uptrend_entry(df, pullback_ema_period=pe, tolerance_pct=tol)


def daily_parts(ts_1h, eth_daily, coin_daily, btc_daily, shift):
    """Har 1h bar ke liye (eth_ok, rs_trend_ok, rs_pct_ok). shift=True -> sirf
    band daily candle (lookahead-free); shift=False -> purana production tareeqa."""
    sh = DAILY_CLOSE_SHIFT if shift else pd.Timedelta(0)

    eth_ema = eth_daily["close"].ewm(span=live.ETH_EMA_PERIOD, adjust=False).mean()
    eth_df = pd.DataFrame({"timestamp": (_ns(eth_daily["timestamp"]) + sh).astype("datetime64[ns]"),
                           "eth_ok": (eth_daily["close"] > eth_ema).values}).sort_values("timestamp")

    c = coin_daily[["timestamp", "close"]].rename(columns={"close": "coin_close"})
    b = btc_daily[["timestamp", "close"]].rename(columns={"close": "btc_close"})
    m = pd.merge_asof(c.sort_values("timestamp"), b.sort_values("timestamp"), on="timestamp", direction="backward")
    ratio = m["coin_close"] / m["btc_close"].replace(0, np.nan)
    rs_bull = ratio > ratio.ewm(span=live.RS_EMA_PERIOD, adjust=False).mean()
    look = live.RS_PERCENTILE_LOOKBACK_DAYS
    rs_pct = ratio.rolling(look, min_periods=2).apply(lambda w: (w <= w[-1]).sum() / len(w) * 100, raw=True)
    rs_df = pd.DataFrame({"timestamp": (_ns(m["timestamp"]) + sh).astype("datetime64[ns]"),
                          "rs_trend": rs_bull.values,
                          "rs_pct": (rs_pct >= live.RS_PERCENTILE_CUTOFF).values}).sort_values("timestamp")

    base = pd.DataFrame({"timestamp": _ns(pd.Series(ts_1h)).values})
    e = pd.merge_asof(base, eth_df, on="timestamp", direction="backward")
    r = pd.merge_asof(base, rs_df, on="timestamp", direction="backward")
    eth_ok = e["eth_ok"].astype("boolean").fillna(True).to_numpy(dtype=bool)
    rs_trend = r["rs_trend"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    rs_pct_ok = r["rs_pct"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return eth_ok, rs_trend, rs_pct_ok


def process_coin(df, coin_daily, btc_daily, eth_daily, results):
    df, coin_daily, btc_daily, eth_daily = _norm(df), _norm(coin_daily), _norm(btc_daily), _norm(eth_daily)
    n = len(df)
    ts = df["timestamp"]

    eth, rst, rsp = daily_parts(ts, eth_daily, coin_daily, btc_daily, shift=True)
    eth_o, rst_o, rsp_o = daily_parts(ts, eth_daily, coin_daily, btc_daily, shift=False)
    masks = {
        "No filter": np.ones(n, dtype=bool),
        "ETH only": eth,
        "RS trend": rst,
        "ETH+RS trend": eth & rst,
        "RS trend+%95": rst & rsp,
        "ETH+RS+%95": eth & rst & rsp,
        "ETH+RS+%95 [OLD]": eth_o & rst_o & rsp_o,
    }

    arrs = (df["open"].values, df["high"].values, df["low"].values, df["close"].values)
    stop = chandelier_arr(df, *EXIT)
    bounds = [int(n * k / N_FOLDS) for k in range(N_FOLDS + 1)]

    def fold_of(i):
        for f in range(N_FOLDS):
            if bounds[f] <= i < bounds[f + 1]:
                return f
        return N_FOLDS - 1

    for entry in ENTRIES:
        # production jaisa: pehle cooldown, phir filter
        sig = apply_cooldown(entry_signal(df, entry).fillna(False).astype(bool), COOLDOWN).values
        for tier in TIERS:
            pos = np.where(sig & masks[tier])[0]
            for i, ret in simulate(arrs, stop, pos, None):
                results[(entry, tier)].append((fold_of(i), ret))


def build_report(results):
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    stats = {k: summarize(v) for k, v in results.items()}
    emit("=" * 78)
    emit("DONCHIAN + PULLBACK - LOOKAHEAD-FREE DOBARA JAANCH")
    emit("=" * 78)
    emit(f"Exit: CE trailing period={EXIT[0]}, mult={EXIT[1]} (production)")
    emit(f"PASS shart: har fold PF>1 (>= {MIN_FOLD_TRADES} trades/fold), kul >= {MIN_TOTAL_TRADES} trades, bootstrap p5 > 1.2")

    emit("\n----- KHULASA (PF | p5 | Top10-hata | trades) -----")
    header = f"{'Entry':26s}" + "".join(f"| {t:18s}" for t in TIERS)
    emit(header)
    for entry in ENTRIES:
        row = f"{entry:26s}"
        for tier in TIERS:
            s = stats[(entry, tier)]
            if s is None or s["pf"] is None:
                cell = "-"
            else:
                cell = f"{s['pf']:.2f}|{fmt(s['p5'])}|{fmt(s['t10'])}|{s['n']}" + ("*" if passes(s) else "")
            row += f"| {cell:18s}"
        emit(row)
    emit("(* = PASS)")

    emit("\n----- TAMAM PASS (sahi / lookahead-free tiers) -----")
    passed = [(k, s) for k, s in stats.items() if "[OLD]" not in k[1] and passes(s)]
    passed.sort(key=lambda ks: -ks[1]["p5"])
    if not passed:
        emit("Koi combo PASS nahi hua.")
    for k, s in passed:
        emit(f"[{k[0]} | {k[1]}] n={s['n']} win={s['win']:.1f}% PF={fmt(s['pf'],3)} p5={fmt(s['p5'],3)} "
             f"Top10-hata={fmt(s['t10'],3)} Exp={s['exp']:+.2f}%/trade")

    emit("\n\n" + "=" * 78)
    emit("POORI DETAIL")
    emit("=" * 78)
    for entry in ENTRIES:
        emit(f"\n### {entry}")
        for tier in TIERS:
            s = stats[(entry, tier)]
            if s is None:
                emit(f"  {tier:18s}: koi trade nahi")
                continue
            folds = " / ".join(f"{fmt(fp)}({fn})" for fp, fn in zip(s["fold_pf"], s["fold_n"]))
            emit(f"  {tier:18s}: n={s['n']} win={s['win']:.1f}% PF={fmt(s['pf'],3)} p5={fmt(s['p5'],3)} "
                 f"Top10-hata={fmt(s['t10'],3)} Exp={s['exp']:+.2f}% | Folds: {folds}"
                 + ("  <== PASS" if passes(s) else ""))
    return lines


def main():
    from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"{len(coins)} coins, {TF}, {CANDLE_LIMIT} candles\n")

    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DAILY_LIMIT)
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=DAILY_LIMIT)
    results = {(e, t): [] for e in ENTRIES for t in TIERS}

    n_fail = 0
    for k, symbol in enumerate(coins, 1):
        try:
            df = fetch_ohlcv(exchange, symbol, TF, limit=CANDLE_LIMIT)
            if df is None or len(df) < 500:
                continue
            coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=DAILY_LIMIT)
            process_coin(df, coin_daily, btc_daily, eth_daily, results)
        except Exception as e:
            n_fail += 1
            print(f"[{k}/{len(coins)}] {symbol}: SKIP ({e})")
            if n_fail == 1:
                import traceback
                traceback.print_exc()
            continue
        if k % 10 == 0 or k == len(coins):
            print(f"[{k}/{len(coins)}] ... done")

    if sum(len(v) for v in results.values()) == 0:
        raise SystemExit(f"KOI TRADE NAHI BANA - {n_fail} coins error se skip hue.")
    lines = build_report(results)
    lines.insert(0, f"(Coins error se skip hue: {n_fail}/{len(coins)})")
    with open("donchian_pullback_nolookahead_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[SAVE] donchian_pullback_nolookahead_RESULTS.txt")


if __name__ == "__main__":
    main()
