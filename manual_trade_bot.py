"""
Manual Paper-Trading Bot - sirf UNHI coins par kaam karta hai jo AAP khud
"manual_watchlist.json" mein daalein. Koi auto-scan nahi karta, koi khud
se coin nahi chunta - bilkul "manual" jaisa maanga gaya tha.

AB YE SIRF CE Buy-Only TAK MEHDOOD NAHI - jis bhi LIVE SYSTEM (CE
Buy-Only, Union AB, NEW AdvancedConfluence, Pullback-in-Uptrend, Donchian
Breakout) ka signal aap dashboard par dekhein, wahi "system" watchlist
mein bata sakte hain - bot us system ke apne SL/TP rules (Chandelier
period/multiplier + fixed-TP ya trailing-only) ke sath hoobahoo trade
manage karega, jaisa us system ka backtest/live-tracking mein hai.

TAREEQA-E-ISTEMAL:
    1) GitHub par "manual_watchlist.json" file kholein, coin symbol +
       (agar CE Buy-Only se alag system hai to) "system" naam daal dein:

           ["MCAT/USDT"]                                    -> CE Buy-Only (default)
           [{"symbol": "BTC/USDT", "system": "NEW AdvancedConfluence"}]
           [{"symbol": "ETH/USDT", "system": "Union AB", "combo": "Ichimoku+MS"}]
           [{"symbol": "SOL/USDT", "system": "Pullback-in-Uptrend", "amount": 200}]

       "combo" sirf "Union AB" / "Union AB Backup Tier" ke liye zaroori
       hai (dashboard par "Combo" column mein "Ichimoku+MS" ya
       "EMA+Breakout" likha hota hai - wahi yahan likh dein). Baqi sab
       systems ke liye "combo" ki zaroorat nahi.

    2) Bot ka agla run (max 5 min) us coin ko uthata hai:
         - Entry = us waqt ka current (aakhri band hui candle ka) price.
         - SL = us SYSTEM ke apne Chandelier params se (misaal: CE
           Buy-Only = 16/3.0, Union AB Ichimoku+MS = 16/4.5, Union AB
           EMA+Breakout = 12/4.5, NEW AdvancedConfluence = 16/3.0,
           Pullback/Donchian = 16/4.5).
         - TP: CE Buy-Only/Pullback/Donchian mein koi fixed TP nahi
           (khalis trailing-stop, jaisa in ka tasdeeq-shuda tareeqa
           hai). Union AB aur NEW AdvancedConfluence mein fixed TP hai
           (Risk x 2.0 RR), jaisa un ka apna tasdeeq-shuda tareeqa hai.
       Watchlist se wo coin turant hata diya jata hai.
    3) Dashboard ke "Manual Trade Bot" section mein progress + final
       result dikhta hai (System column bhi dikhega).

YE ASAL PAISON SE TRADE NAHI KARTA - paper/virtual hai (koi API
trade-key nahi chahiye).
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd

import config
from data_fetcher import get_exchange, fetch_ohlcv
from backtest_engine import compute_atr, compute_chandelier_long_stop

try:
    from telegram_alert import send_telegram_alert
except Exception:
    def send_telegram_alert(msg):
        pass


# ============================= SETTINGS =============================
SIGNAL_TIMEFRAME = "1h"

# Har live system ke apne SL/TP rules - dashboard/scheduled_dashboard_scan.py
# mein jo constants tasdeeq-shuda hain, wahi hoobahoo yahan bhi:
SYSTEM_PRESETS = {
    "CE Buy-Only":              {"ce_period": 16, "ce_multiplier": 3.0, "use_fixed_tp": False, "rr_multiple": None},
    "NEW AdvancedConfluence":   {"ce_period": 16, "ce_multiplier": 3.0, "use_fixed_tp": True,  "rr_multiple": 2.0},
    "Pullback-in-Uptrend":      {"ce_period": 16, "ce_multiplier": 4.5, "use_fixed_tp": False, "rr_multiple": None},
    "Donchian Breakout":        {"ce_period": 16, "ce_multiplier": 4.5, "use_fixed_tp": False, "rr_multiple": None},
    # Union AB / Backup Tier: combo ke hisaab se params alag hain
    ("Union AB", "Ichimoku+MS"):        {"ce_period": 16, "ce_multiplier": 4.5, "use_fixed_tp": True, "rr_multiple": 2.0},
    ("Union AB", "EMA+Breakout"):       {"ce_period": 12, "ce_multiplier": 4.5, "use_fixed_tp": True, "rr_multiple": 2.0},
    ("Union AB Backup Tier", "Ichimoku+MS"):  {"ce_period": 16, "ce_multiplier": 4.5, "use_fixed_tp": True, "rr_multiple": 2.0},
    ("Union AB Backup Tier", "EMA+Breakout"): {"ce_period": 12, "ce_multiplier": 4.5, "use_fixed_tp": True, "rr_multiple": 2.0},
}
DEFAULT_SYSTEM = "CE Buy-Only"
DEFAULT_COMBO = "Ichimoku+MS"   # sirf Union AB variants ke liye, jab combo na diya jaye

STARTING_CAPITAL = 1000.0
POSITION_SIZE_USD = 100.0      # har manually-feed ki gayi coin ke liye default $100 (ya watchlist mein "amount")
MAX_CONCURRENT_POSITIONS = 8
FEE_PCT = config.BACKTEST_PARAMS["fee_pct"] / 100   # 0.1% per side, backtest/live jaisa hoobahoo

POSITION_DATA_LIMIT = 700

STATE_FILE = "manual_bot_state.json"
CLOSED_TRADES_FILE = "manual_bot_closed_trades.csv"
WATCHLIST_FILE = "manual_watchlist.json"


def resolve_preset(system, combo):
    system = system or DEFAULT_SYSTEM
    if system in ("Union AB", "Union AB Backup Tier"):
        combo = combo or DEFAULT_COMBO
        key = (system, combo)
        if key not in SYSTEM_PRESETS:
            return None, f"Combo '{combo}' pehchana nahi gaya (Ichimoku+MS ya EMA+Breakout likhein)"
        return SYSTEM_PRESETS[key], None
    if system not in SYSTEM_PRESETS:
        return None, f"System '{system}' pehchana nahi gaya"
    return SYSTEM_PRESETS[system], None


# ============================= STATE / WATCHLIST I/O =============================
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


def load_watchlist():
    """
    manual_watchlist.json - entries plain symbol ya dict ho sakte hain:

        ["MCAT/USDT", {"symbol": "BTC/USDT", "system": "Union AB",
                        "combo": "Ichimoku+MS", "amount": 200}]

    Return: list of dicts {symbol, amount, system, combo}.
    """
    if not os.path.exists(WATCHLIST_FILE):
        return []
    try:
        with open(WATCHLIST_FILE) as f:
            raw = json.load(f)
    except Exception:
        return []

    cleaned = []
    for entry in raw:
        if isinstance(entry, dict):
            s = str(entry.get("symbol", "")).strip().upper()
            amount = entry.get("amount")
            try:
                amount = float(amount) if amount is not None else None
            except (TypeError, ValueError):
                amount = None
            system = entry.get("system") or DEFAULT_SYSTEM
            combo = entry.get("combo")
        else:
            s = str(entry).strip().upper()
            amount, system, combo = None, DEFAULT_SYSTEM, None

        if not s:
            continue
        if "/" not in s:
            s = f"{s}/USDT"
        cleaned.append({"symbol": s, "amount": amount, "system": system, "combo": combo})
    return cleaned


def save_watchlist(entries):
    """entries: list of dicts {symbol, amount, system, combo} - wapis watchlist format mein likhte hain."""
    raw = []
    for e in entries:
        if e["amount"] is None and e["system"] == DEFAULT_SYSTEM and not e.get("combo"):
            raw.append(e["symbol"])
        else:
            item = {"symbol": e["symbol"], "system": e["system"]}
            if e.get("combo"):
                item["combo"] = e["combo"]
            if e["amount"] is not None:
                item["amount"] = e["amount"]
            raw.append(item)
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(raw, f, indent=2)


def append_closed_trade(row):
    df_row = pd.DataFrame([row])
    if os.path.exists(CLOSED_TRADES_FILE):
        df_row.to_csv(CLOSED_TRADES_FILE, mode="a", header=False, index=False)
    else:
        df_row.to_csv(CLOSED_TRADES_FILE, mode="w", header=True, index=False)


def to_pkt_str(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("Asia/Karachi").strftime("%Y-%m-%d %I:%M %p PKT")


# ============================= OPEN POSITION UPDATE (close if SL/TP hit) =============================
def update_open_position(symbol, pos, exchange):
    ce_period = pos.get("ce_period", 16)
    ce_multiplier = pos.get("ce_multiplier", 3.0)
    use_fixed_tp = pos.get("use_fixed_tp", False)
    tp_price = pos.get("tp_price")

    df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=POSITION_DATA_LIMIT)
    if df is None or len(df) < ce_period + 5:
        return True, pos, None

    entry_time = pd.Timestamp(pos["entry_time"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize("UTC")
    ts_series = df["timestamp"]
    if ts_series.dt.tz is None:
        ts_series = ts_series.dt.tz_localize("UTC")

    matches = df.index[ts_series == entry_time]
    signal_idx = matches[0] if len(matches) else 0

    atr = compute_atr(df, ce_period)
    highest_high = df["high"].rolling(ce_period).max()
    chandelier_series = highest_high - ce_multiplier * atr

    running_stop = float(pos["initial_stop"])
    status = "OPEN"
    exit_price = None
    exit_time = None
    exit_reason = None

    for i in range(signal_idx + 1, len(df)):
        bar_stop = chandelier_series.iloc[i]
        if not pd.isna(bar_stop) and bar_stop > running_stop:
            running_stop = float(bar_stop)

        low_i = df["low"].iloc[i]
        high_i = df["high"].iloc[i]
        stop_hit = low_i <= running_stop
        tp_hit = use_fixed_tp and tp_price is not None and high_i >= tp_price

        if stop_hit:
            status, exit_price, exit_time, exit_reason = "CLOSED", running_stop, df["timestamp"].iloc[i], "SL"
            break
        elif tp_hit:
            status, exit_price, exit_time, exit_reason = "CLOSED", tp_price, df["timestamp"].iloc[i], "TP"
            break

    if status == "OPEN":
        current_price = float(df["close"].iloc[-1])
        pos["trail_stop"] = round(running_stop, 8)
        pos["current_price"] = round(current_price, 8)
        pos["unrealized_pnl_pct"] = round((current_price - pos["entry_price"]) / pos["entry_price"] * 100, 3)
        return True, pos, None

    entry_price = pos["entry_price"]
    gross_return = (exit_price - entry_price) / entry_price
    net_return = gross_return - (2 * FEE_PCT)
    realized_pnl = pos["capital_allocated"] * net_return

    closed_row = {
        "symbol": symbol,
        "system": pos.get("system", DEFAULT_SYSTEM),
        "entry_time_pkt": to_pkt_str(pos["entry_time"]),
        "exit_time_pkt": to_pkt_str(exit_time),
        "entry_price": round(entry_price, 8),
        "exit_price": round(exit_price, 8),
        "exit_reason": exit_reason,
        "capital_allocated_usd": pos["capital_allocated"],
        "gross_return_pct": round(gross_return * 100, 3),
        "net_return_pct": round(net_return * 100, 3),
        "realized_pnl_usd": round(realized_pnl, 4),
        "result": "WIN" if net_return > 0 else "LOSS",
    }

    send_telegram_alert(
        f"🤖 Manual Bot Trade CLOSED ({closed_row['system']}): {symbol}\n"
        f"Entry: {entry_price:.6f} -> Exit: {exit_price:.6f} ({exit_reason})\n"
        f"Net Return: {net_return*100:.2f}% | P&L: ${realized_pnl:.2f} ({closed_row['result']})"
    )

    return False, {"realized_pnl": realized_pnl, "capital_allocated": pos["capital_allocated"]}, closed_row


# ============================= MANUAL ENTRY (jo watchlist mein daala gaya) =============================
def open_manual_position(symbol, exchange, cash, open_count, amount=None, system=DEFAULT_SYSTEM, combo=None):
    """
    amount: agar watchlist mein us coin ke sath khud ki amount di gayi ho
    to wo istemal hoti hai, warna default POSITION_SIZE_USD ($100).
    system/combo: kaunse live system ke SL/TP rules follow karne hain.
    """
    position_size = amount if amount is not None else POSITION_SIZE_USD

    if open_count >= MAX_CONCURRENT_POSITIONS:
        return None, "MAX_POSITIONS"
    if cash < position_size:
        return None, "NO_CASH"
    if position_size <= 0:
        return None, "INVALID_AMOUNT"

    preset, err = resolve_preset(system, combo)
    if preset is None:
        return None, f"UNKNOWN_SYSTEM: {err}"

    ce_period, ce_multiplier = preset["ce_period"], preset["ce_multiplier"]
    use_fixed_tp, rr_multiple = preset["use_fixed_tp"], preset["rr_multiple"]

    df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=max(config.CANDLE_LIMITS.get(SIGNAL_TIMEFRAME, 500), 300))
    if df is None or len(df) < ce_period + 5:
        return None, "NO_DATA"

    exit_stop_series = compute_chandelier_long_stop(df, ce_period, ce_multiplier)
    entry_price = float(df["close"].iloc[-1])
    initial_stop = exit_stop_series.iloc[-1]

    if pd.isna(initial_stop) or initial_stop >= entry_price:
        # ATR-based stop is waqt entry price se upar/bohot qareeb hai - matlab
        # abhi coin bohot ziada volatile/overextended hai, is waqt trade lena
        # khud CE indicator ke apne rule ke khilaf hoga - isliye safe taur par skip.
        return None, "INVALID_STOP"

    risk = entry_price - float(initial_stop)
    tp_price = (entry_price + risk * rr_multiple) if use_fixed_tp else None

    pos = {
        "entry_time": df["timestamp"].iloc[-1].isoformat(),
        "entry_price": round(entry_price, 8),
        "initial_stop": round(float(initial_stop), 8),
        "trail_stop": round(float(initial_stop), 8),
        "tp_price": round(tp_price, 8) if tp_price is not None else None,
        "capital_allocated": position_size,
        "current_price": round(entry_price, 8),
        "unrealized_pnl_pct": 0.0,
        "source": "manual",
        "system": system,
        "combo": combo,
        "ce_period": ce_period,
        "ce_multiplier": ce_multiplier,
        "use_fixed_tp": use_fixed_tp,
    }

    tp_str = f"{tp_price:.6f}" if tp_price is not None else "N/A (trailing-stop hi exit hai)"
    send_telegram_alert(
        f"🤖 Manual Bot Trade OPENED ({system}{' - ' + combo if combo else ''}): {symbol}\n"
        f"Entry: {entry_price:.6f} | SL (CE {ce_period},{ce_multiplier}): {float(initial_stop):.6f} | TP: {tp_str}\n"
        f"Capital Allocated: ${position_size:.2f} (virtual)"
    )

    return pos, "OPENED"


# ============================= MAIN =============================
def main():
    exchange = get_exchange()
    state = load_state()
    cash = state.get("cash", STARTING_CAPITAL)
    positions = state.get("positions", {})

    print(f"Manual bot shuru: cash=${cash:.2f}, open positions={len(positions)}")

    # ---- STEP 1: khuli positions update/close ----
    still_open = {}
    for symbol, pos in positions.items():
        try:
            is_open, result, closed_row = update_open_position(symbol, pos, exchange)
        except Exception as e:
            print(f"  [SKIP-UPDATE] {symbol}: {e}")
            still_open[symbol] = pos
            continue

        if is_open:
            still_open[symbol] = result
        else:
            cash += result["capital_allocated"] + result["realized_pnl"]
            append_closed_trade(closed_row)
            print(f"  [CLOSED] {symbol}: {closed_row['result']} ${closed_row['realized_pnl_usd']:.2f} ({closed_row['exit_reason']})")

    positions = still_open

    # ---- STEP 2: watchlist mein jo coins hain unhe process karo (SIRF yehi, koi auto-scan nahi) ----
    watchlist = load_watchlist()
    remaining_watchlist = []

    for entry in watchlist:
        symbol, amount, system, combo = entry["symbol"], entry["amount"], entry["system"], entry["combo"]
        if symbol in positions:
            print(f"  [SKIP] {symbol}: pehle se hi ek open position mojood hai")
            continue
        try:
            pos, reason = open_manual_position(symbol, exchange, cash, len(positions), amount=amount, system=system, combo=combo)
        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            remaining_watchlist.append(entry)   # error par dobara try karne ke liye rakh lo
            continue

        if pos is not None:
            positions[symbol] = pos
            cash -= pos["capital_allocated"]
            print(f"  [OPENED] {symbol} ({system}): entry={pos['entry_price']}, SL={pos['initial_stop']}, "
                  f"TP={pos['tp_price']}, amount=${pos['capital_allocated']}")
        elif reason in ("MAX_POSITIONS", "NO_CASH", "NO_DATA"):
            print(f"  [QUEUED] {symbol}: abhi nahi ({reason}), agli baar phir koshish hogi")
            remaining_watchlist.append(entry)
        else:
            print(f"  [REJECTED] {symbol}: {reason} - watchlist se hata diya")

    save_watchlist(remaining_watchlist)

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

    print(f"Manual bot khatam: cash=${cash:.2f}, open positions={len(positions)}, total equity=${total_equity:.2f}")


if __name__ == "__main__":
    main()
