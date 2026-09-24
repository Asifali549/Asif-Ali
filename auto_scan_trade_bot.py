"""
Auto-Scan Paper-Trading Bot - "dummy" auto mode, manual_trade_bot.py ka
sāthi. Manual bot SIRF unhi coins par kaam karta hai jo aap khud
manual_watchlist.json mein daalein. Yeh (auto) bot iske bar-aks khud hi
150 coins scan karta hai aur JAB BHI kisi bhi live system (Union AB, NEW
AdvancedConfluence, CE Buy-Only, Pullback-in-Uptrend, Donchian Breakout)
ka fresh signal bane, khud wahi trade (paper/virtual) le leta hai - koi
manual feed ki zaroorat nahi.

YE BHI ASAL PAISON SE TRADE NAHI KARTA (paper/virtual hai) - isliye
"dummy auto" rakhna bhi mehfooz hai, kyunke koi asal nuqsan ka khatra
nahi (jaisa aap ne khud kaha: "result hi lena hai").

Dono bots (manual + auto) ki ledger/capital BILKUL ALAG hain
(auto_bot_state.json / auto_bot_closed_trades.csv vs
manual_bot_state.json / manual_bot_closed_trades.csv) - taake dono ka
Win Rate/PF alag alag, saaf taur par compare ho sake.

NOTE: "Union AB Backup Tier" is auto-scan mein SHAAMIL NAHI ki gayi -
kyunke wo Union AB ke bilkul usi signal (baseline) par bhi ban sakti
hai, sirf filter halka hai (ETH check nahi) - dono ko auto-open karne se
aksar EK HI candle par do overlapping virtual trades ban jatin, jo
capital ka galat/dohra hisaab deta. Sirf Union AB (Baseline tier) ko
"primary" version ke taur par auto-trade kiya gaya hai.
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd

import config
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from confluence_engine import compute_confluence, DEFAULT_PARAMS as CONF_PARAMS

import scheduled_dashboard_scan as live
import manual_trade_bot as mbot

STATE_FILE = "auto_bot_state.json"
CLOSED_TRADES_FILE = "auto_bot_closed_trades.csv"

STARTING_CAPITAL = 2000.0        # manual bot se alag, mehez isliye zyada diya gaya kyunke 5 systems ek sath scan hote hain
POSITION_SIZE_USD = 100.0
MAX_CONCURRENT_POSITIONS = 15


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"cash": STARTING_CAPITAL, "positions": {}, "last_updated": None}


def save_state(state):
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def append_closed_trade(row):
    df_row = pd.DataFrame([row])
    if os.path.exists(CLOSED_TRADES_FILE):
        df_row.to_csv(CLOSED_TRADES_FILE, mode="a", header=False, index=False)
    else:
        df_row.to_csv(CLOSED_TRADES_FILE, mode="w", header=True, index=False)


def position_key(symbol, system, combo):
    return f"{symbol}|{system}|{combo or ''}"


def main():
    exchange = live.get_exchange()
    state = load_state()
    cash = state.get("cash", STARTING_CAPITAL)
    positions = state.get("positions", {})

    print(f"Auto-scan bot shuru: cash=${cash:.2f}, open positions={len(positions)}")

    # ---- STEP 1: khuli positions update/close (manual bot ka hi generic function reuse) ----
    still_open = {}
    for key, pos in positions.items():
        symbol = pos["symbol"]
        try:
            is_open, result, closed_row = mbot.update_open_position(symbol, pos, exchange)
        except Exception as e:
            print(f"  [SKIP-UPDATE] {key}: {e}")
            still_open[key] = pos
            continue

        if is_open:
            still_open[key] = result
        else:
            cash += result["capital_allocated"] + result["realized_pnl"]
            append_closed_trade(closed_row)
            print(f"  [CLOSED] {key}: {closed_row['result']} ${closed_row['realized_pnl_usd']:.2f}")

    positions = still_open

    # ---- STEP 2: 150 coins scan karo, jis bhi system ka FRESH signal bane wahi (agar jagah/cash ho) khud trade lo ----
    btc_daily = live.fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=800)
    eth_daily = live.fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=800)
    eth_regime = live.compute_eth_regime(eth_daily, ema_period=live.ETH_EMA_PERIOD)

    coins = live.get_coin_list(exchange)[:live.TOP_N_COINS]
    print(f"Scanning {len(coins)} coins (5 systems: Union AB, NEW AdvancedConfluence, CE Buy-Only, "
          f"Pullback-in-Uptrend, Donchian Breakout)...")

    def try_open(symbol, system, combo=None):
        nonlocal cash
        key = position_key(symbol, system, combo)
        if key in positions:
            return
        if len(positions) >= MAX_CONCURRENT_POSITIONS or cash < POSITION_SIZE_USD:
            return
        try:
            pos, reason = mbot.open_manual_position(symbol, exchange, cash, len(positions),
                                                       amount=POSITION_SIZE_USD, system=system, combo=combo)
        except Exception as e:
            print(f"  [ERROR] {symbol}/{system}: {e}")
            return
        if pos is not None:
            pos["symbol"] = symbol
            positions[key] = pos
            cash -= pos["capital_allocated"]
            print(f"  [AUTO-OPENED] {symbol} ({system}{'/' + combo if combo else ''}): "
                  f"entry={pos['entry_price']}, SL={pos['initial_stop']}, TP={pos['tp_price']}")

    for symbol in coins:
        try:
            df = live.fetch_ohlcv(exchange, symbol, live.SIGNAL_TIMEFRAME,
                                   limit=max(config.CANDLE_LIMITS.get(live.SIGNAL_TIMEFRAME, 500), 300))
        except Exception:
            df = None
        if df is None or len(df) < 220:
            continue

        rs_bullish, rs_ratio_series, coin_daily = None, None, None
        try:
            coin_daily = live.fetch_ohlcv(exchange, symbol, "1d", limit=live.DAILY_FETCH_LIMIT)
            rs_bullish, rs_ratio_series = live.compute_rs_trend_and_ratio(coin_daily, btc_daily, ema_period=live.RS_EMA_PERIOD)
        except Exception:
            pass

        # ---- Union AB (Baseline tier) - dono combos ----
        try:
            ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
            ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
            ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
            breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)
            combo_a = ichi_sig & ms_sig
            combo_b = ema_sig & breakout_sig

            for combo_sig, combo_name, ce in [(combo_a, "Ichimoku+MS", live.CE_A), (combo_b, "EMA+Breakout", live.CE_B)]:
                found = live.find_latest_open_or_new(df, combo_sig, ce["period"], ce["multiplier"])
                if found is not None and found["category"] == "New Signal" and rs_bullish is not None:
                    idx = found["signal_idx"]
                    tier, _ = live.evaluate_union_ab_tier(symbol, df, idx, eth_regime, rs_bullish, rs_ratio_series, coin_daily)
                    if tier is not None:
                        try_open(symbol, "Union AB", combo_name)
        except Exception as e:
            print(f"  [SKIP-UnionAB] {symbol}: {e}")

        # ---- NEW AdvancedConfluence ----
        try:
            result_new = compute_confluence(df, btc_daily, CONF_PARAMS, usdt_d_weak=None)
            choch_signal = result_new["choch"]
            new_sig = apply_cooldown(choch_signal & (result_new["score"] >= CONF_PARAMS["score_threshold"]), config.SIGNAL_COOLDOWN_BARS)
            found = live.find_latest_open_or_new(df, new_sig, live.CE_D["period"], live.CE_D["multiplier"])
            if found is not None and found["category"] == "New Signal":
                try_open(symbol, "NEW AdvancedConfluence")
        except Exception as e:
            print(f"  [SKIP-NEWConf] {symbol}: {e}")

        # ---- CE Buy-Only (koi filter nahi) ----
        try:
            entry_stop = live.compute_chandelier_long_stop(df, live.CE_BUYONLY_ENTRY["period"], live.CE_BUYONLY_ENTRY["multiplier"])
            close = df["close"]
            cross_above = (close > entry_stop) & (close.shift(1) <= entry_stop.shift(1))
            ce_sig = apply_cooldown(cross_above.fillna(False), config.SIGNAL_COOLDOWN_BARS)
            found = live.find_latest_open_or_new(
                df, ce_sig, live.CE_BUYONLY_EXIT["period"], live.CE_BUYONLY_EXIT["multiplier"],
                entry_ce_period=live.CE_BUYONLY_ENTRY["period"], entry_ce_multiplier=live.CE_BUYONLY_ENTRY["multiplier"],
                use_fixed_tp=False,
            )
            if found is not None and found["category"] == "New Signal":
                try_open(symbol, "CE Buy-Only")
        except Exception as e:
            print(f"  [SKIP-CEBuyOnly] {symbol}: {e}")

        # ---- Pullback-in-Uptrend + Donchian Breakout (ETH + RS + RS%95 filters) ----
        if rs_bullish is not None:
            try:
                pb_sig = apply_cooldown(live.pullback_uptrend_entry(df), config.SIGNAL_COOLDOWN_BARS)
                found = live.find_latest_open_or_new(df, pb_sig, live.CE_PB["period"], live.CE_PB["multiplier"], use_fixed_tp=False)
                if found is not None and found["category"] == "New Signal":
                    sig_ts = pd.Timestamp(df["timestamp"].iloc[found["signal_idx"]])
                    ok, _ = live.passes_full_filters(sig_ts, eth_regime, rs_bullish, rs_ratio_series)
                    if ok:
                        try_open(symbol, "Pullback-in-Uptrend")
            except Exception as e:
                print(f"  [SKIP-Pullback] {symbol}: {e}")

            try:
                dc_sig = apply_cooldown(live.donchian_channel_breakout(df), config.SIGNAL_COOLDOWN_BARS)
                found = live.find_latest_open_or_new(df, dc_sig, live.CE_PB["period"], live.CE_PB["multiplier"], use_fixed_tp=False)
                if found is not None and found["category"] == "New Signal":
                    sig_ts = pd.Timestamp(df["timestamp"].iloc[found["signal_idx"]])
                    ok, _ = live.passes_full_filters(sig_ts, eth_regime, rs_bullish, rs_ratio_series)
                    if ok:
                        try_open(symbol, "Donchian Breakout")
            except Exception as e:
                print(f"  [SKIP-Donchian] {symbol}: {e}")

        if len(positions) >= MAX_CONCURRENT_POSITIONS or cash < POSITION_SIZE_USD:
            print("  Capital/positions full - baaqi coins ka scan agli baar.")
            break

    # ---- STEP 3: state save ----
    total_equity = cash + sum(p["capital_allocated"] for p in positions.values())
    state = {
        "cash": round(cash, 4),
        "positions": positions,
        "total_equity_usd": round(total_equity, 4),
        "starting_capital_usd": STARTING_CAPITAL,
        "open_position_count": len(positions),
        "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
        "position_size_usd": POSITION_SIZE_USD,
    }
    save_state(state)

    print(f"Auto-scan bot khatam: cash=${cash:.2f}, open positions={len(positions)}, total equity=${total_equity:.2f}")


if __name__ == "__main__":
    main()
