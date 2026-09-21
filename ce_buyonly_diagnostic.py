"""
CE Buy-Only Diagnostic - sirf ek dafa chalti hai (workflow_dispatch),
kisi bhi cheez ko badalti nahi, sirf report deti hai.

Maqsad: CE Buy-Only system 150 coins mein se HAR ek par pichle ~30 din
(720 x 1h bars) mein kitne "raw" chandelier(11,4.5) cross-above events
huay, cooldown ke baad kitne bache, aur SAB SE AAKHRI event ki status
kya hai (abhi bhi OPEN hai ya SL/TP pe CLOSED ho chuka) - taake pata
chale ke dashboard par "0 signals" genuine market behaviour hai ya
kahin aur koi masla hai.

Result 'ce_buyonly_diagnostic_RESULTS.txt' mein save hoti hai.
"""

import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown
from backtest_engine import compute_chandelier_long_stop
from scheduled_dashboard_scan import (
    compute_trade_progress, CE_BUYONLY_ENTRY, CE_BUYONLY_EXIT,
    OPEN_TRADE_LOOKBACK_BARS, NEW_SIGNAL_WINDOW_BARS,
)

TOP_N_COINS = 150
SIGNAL_TIMEFRAME = "1h"
LOOKBACK_BARS_FOR_DIAG = 720   # ~30 din


def main():
    lines = []
    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TOP_N_COINS]
    lines.append(f"CE Buy-Only Diagnostic - {len(coins)} coins, pichle {LOOKBACK_BARS_FOR_DIAG} bars (~30 din)\n")
    lines.append("=" * 70)

    total_raw_crosses = 0
    total_post_cooldown = 0
    coins_with_any_event = 0
    coins_with_currently_open = 0
    most_recent_overall = None  # (bars_ago, symbol, status)

    per_coin_details = []

    for symbol in coins:
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=max(config.CANDLE_LIMITS.get(SIGNAL_TIMEFRAME, 500), 300))
        except Exception as e:
            per_coin_details.append(f"{symbol}: FETCH FAILED ({e})")
            continue
        if df is None or len(df) < 220:
            per_coin_details.append(f"{symbol}: SKIPPED (not enough data, len={0 if df is None else len(df)})")
            continue

        try:
            entry_stop = compute_chandelier_long_stop(df, CE_BUYONLY_ENTRY["period"], CE_BUYONLY_ENTRY["multiplier"])
            close = df["close"]
            cross_above = (close > entry_stop) & (close.shift(1) <= entry_stop.shift(1))
            raw_series = cross_above.fillna(False)

            n = len(raw_series)
            start = max(0, n - LOOKBACK_BARS_FOR_DIAG)
            raw_recent = raw_series.iloc[start:]
            raw_count = int(raw_recent.sum())
            total_raw_crosses += raw_count
            if raw_count > 0:
                coins_with_any_event += 1

            ce_sig = apply_cooldown(raw_series, config.SIGNAL_COOLDOWN_BARS)
            post_recent = ce_sig.iloc[start:]
            post_count = int(post_recent.sum())
            total_post_cooldown += post_count

            # sab se aakhri (poori history mein, sirf abhi wale window tak mehdood nahi) True index
            true_idxs = ce_sig[ce_sig].index
            if len(true_idxs) == 0:
                per_coin_details.append(f"{symbol}: raw={raw_count}, post-cooldown={post_count}, koi bhi signal nahi mila")
                continue

            last_idx = true_idxs[-1]
            bars_ago_signal = n - 1 - last_idx
            progress = compute_trade_progress(
                df, last_idx, CE_BUYONLY_EXIT["period"], CE_BUYONLY_EXIT["multiplier"],
                entry_ce_period=CE_BUYONLY_ENTRY["period"], entry_ce_multiplier=CE_BUYONLY_ENTRY["multiplier"],
            )
            if progress is None:
                status_str = "INVALID (stop>=entry ya NaN us waqt)"
            else:
                status_str = progress["status"]
                if status_str == "CLOSED":
                    status_str += f" ({progress['exit_reason']}, exit {int(n-1-progress['exit_idx'])} bars pehle)"
                if progress["status"] == "OPEN" and bars_ago_signal <= OPEN_TRADE_LOOKBACK_BARS:
                    coins_with_currently_open += 1

            per_coin_details.append(
                f"{symbol}: raw={raw_count}, post-cooldown={post_count}, "
                f"aakhri signal {bars_ago_signal} bars ({bars_ago_signal/24:.1f} din) pehle -> {status_str}"
            )

            if most_recent_overall is None or bars_ago_signal < most_recent_overall[0]:
                most_recent_overall = (bars_ago_signal, symbol, status_str)

        except Exception as e:
            per_coin_details.append(f"{symbol}: ERROR - {e}")

    lines.append(f"\nTotal RAW cross events (pichle {LOOKBACK_BARS_FOR_DIAG} bars, sab {len(coins)} coins mila ke): {total_raw_crosses}")
    lines.append(f"Total POST-COOLDOWN signals: {total_post_cooldown}")
    lines.append(f"Kitne coins par kam-az-kam 1 raw event hua: {coins_with_any_event}/{len(coins)}")
    lines.append(f"Kitne coins par ABHI (is waqt) ek OPEN trade hai (jo dashboard dikhata): {coins_with_currently_open}/{len(coins)}")
    if most_recent_overall is not None:
        ba, sym, st = most_recent_overall
        lines.append(f"\nSab se RECENT signal (kisi bhi coin par): {sym}, {ba} bars ({ba/24:.1f} din) pehle -> {st}")
    lines.append("\n" + "=" * 70)
    lines.append("Har coin ki detail:\n")
    lines.extend(per_coin_details)

    report = "\n".join(lines)
    with open("ce_buyonly_diagnostic_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
