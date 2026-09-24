"""
Manual Paper-Trading Bot (CE Buy-Only) - sirf UNHI coins par kaam karta
hai jo AAP khud "manual_watchlist.json" mein daalein. Koi auto-scan nahi
karta, koi khud se coin nahi chunta - bilkul "manual" jaisa maanga gaya
tha.

TAREEQA-E-ISTEMAL (bohot simple):
    1) GitHub par "manual_watchlist.json" file kholein, aur us coin ka
       symbol daal dein jo aap ne dashboard par CE Buy-Only signal mein
       dekha (misaal: ["MCAT/USDT"]).
    2) Bot ka agla run (max 5 minute ke andar, self-loop ki wajah se) us
       coin ko utha kar "professional tareeqe se" ek asal (virtual) trade
       le lega:
         - Entry = us waqt ka current (aakhri band hui candle ka) price.
         - Stop/SL = CE Buy-Only ka wahi exit Chandelier (period=16,
           multiplier=3.0), jaisa poore live system mein hai - koi fixed
           % nahi, ATR-based dynamic stop.
         - Koi fixed Take Profit NAHI (jaisa CE Buy-Only ka tasdeeq-shuda
           tareeqa hai) - trade sirf apni trailing stop hit hone par band
           hogi, jab tak price upar jati rahegi stop bhi upar trail hoga.
       Watchlist se wo coin turant hata diya jata hai (ek baar process ho
       gaya, dobara khud nahi khulega jab tak aap dobara na daalein).
    3) Dashboard par "Manual Trade Bot" section mein aapko us coin ki
       LIVE progress (current price, trail stop, unrealized P/L%) aur
       phir close hone par uska final result dikhta rahega.

YE ASAL PAISON SE TRADE NAHI KARTA - paper/virtual hai (koi API trade-key
nahi chahiye). Jab aap dono (manual selection + CE-based SL) ka tareeqa
kaafi dinon tak dekh kar mutmain ho jayein, isi file ka "entry/exit"
hissa asal KuCoin order calls (ccxt create_order) mein badla ja sakta
hai.
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
CE_BUYONLY_EXIT = {"period": 16, "multiplier": 3.0}   # SL/trailing-stop isi se calculate hota hai

STARTING_CAPITAL = 1000.0
POSITION_SIZE_USD = 100.0      # har manually-feed ki gayi coin ke liye fixed $100
MAX_CONCURRENT_POSITIONS = 8
FEE_PCT = config.BACKTEST_PARAMS["fee_pct"] / 100   # 0.1% per side, backtest/live jaisa hoobahoo

POSITION_DATA_LIMIT = 700

STATE_FILE = "manual_bot_state.json"
CLOSED_TRADES_FILE = "manual_bot_closed_trades.csv"
WATCHLIST_FILE = "manual_watchlist.json"


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
    manual_watchlist.json: simple list of coin symbols, misaal:
        ["MCAT/USDT", "DOGE/USDT"]
    Agar file mojood nahi to khali list. Har symbol ko normalize karte
    hain (upper-case, "/USDT" agar na diya ho to khud laga dete hain).
    """
    if not os.path.exists(WATCHLIST_FILE):
        return []
    try:
        with open(WATCHLIST_FILE) as f:
            raw = json.load(f)
    except Exception:
        return []

    cleaned = []
    for s in raw:
        s = str(s).strip().upper()
        if not s:
            continue
        if "/" not in s:
            s = f"{s}/USDT"
        cleaned.append(s)
    return cleaned


def save_watchlist(symbols):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(symbols, f, indent=2)


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


# ============================= OPEN POSITION UPDATE (close if SL hit) =============================
def update_open_position(symbol, pos, exchange):
    df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=POSITION_DATA_LIMIT)
    if df is None or len(df) < CE_BUYONLY_EXIT["period"] + 5:
        return True, pos, None

    entry_time = pd.Timestamp(pos["entry_time"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize("UTC")
    ts_series = df["timestamp"]
    if ts_series.dt.tz is None:
        ts_series = ts_series.dt.tz_localize("UTC")

    matches = df.index[ts_series == entry_time]
    signal_idx = matches[0] if len(matches) else 0

    atr = compute_atr(df, CE_BUYONLY_EXIT["period"])
    highest_high = df["high"].rolling(CE_BUYONLY_EXIT["period"]).max()
    chandelier_series = highest_high - CE_BUYONLY_EXIT["multiplier"] * atr

    running_stop = float(pos["initial_stop"])
    status = "OPEN"
    exit_price = None
    exit_time = None

    for i in range(signal_idx + 1, len(df)):
        bar_stop = chandelier_series.iloc[i]
        if not pd.isna(bar_stop) and bar_stop > running_stop:
            running_stop = float(bar_stop)
        if df["low"].iloc[i] <= running_stop:
            status = "CLOSED"
            exit_price = running_stop
            exit_time = df["timestamp"].iloc[i]
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
        "entry_time_pkt": to_pkt_str(pos["entry_time"]),
        "exit_time_pkt": to_pkt_str(exit_time),
        "entry_price": round(entry_price, 8),
        "exit_price": round(exit_price, 8),
        "capital_allocated_usd": pos["capital_allocated"],
        "gross_return_pct": round(gross_return * 100, 3),
        "net_return_pct": round(net_return * 100, 3),
        "realized_pnl_usd": round(realized_pnl, 4),
        "result": "WIN" if net_return > 0 else "LOSS",
    }

    send_telegram_alert(
        f"🤖 Manual Bot Trade CLOSED: {symbol}\n"
        f"Entry: {entry_price:.6f} -> Exit: {exit_price:.6f}\n"
        f"Net Return: {net_return*100:.2f}% | P&L: ${realized_pnl:.2f} ({closed_row['result']})"
    )

    return False, {"realized_pnl": realized_pnl, "capital_allocated": pos["capital_allocated"]}, closed_row


# ============================= MANUAL ENTRY (jo watchlist mein daala gaya) =============================
def open_manual_position(symbol, exchange, cash, open_count):
    if open_count >= MAX_CONCURRENT_POSITIONS:
        return None, "MAX_POSITIONS"
    if cash < POSITION_SIZE_USD:
        return None, "NO_CASH"

    df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=max(config.CANDLE_LIMITS.get(SIGNAL_TIMEFRAME, 500), 300))
    if df is None or len(df) < CE_BUYONLY_EXIT["period"] + 5:
        return None, "NO_DATA"

    exit_stop_series = compute_chandelier_long_stop(df, CE_BUYONLY_EXIT["period"], CE_BUYONLY_EXIT["multiplier"])
    entry_price = float(df["close"].iloc[-1])
    initial_stop = exit_stop_series.iloc[-1]

    if pd.isna(initial_stop) or initial_stop >= entry_price:
        # ATR-based stop is waqt entry price se upar/bohot qareeb hai - matlab
        # abhi coin bohot ziada volatile/overextended hai, is waqt trade lena
        # khud CE indicator ke apne rule ke khilaf hoga - isliye safe taur par skip.
        return None, "INVALID_STOP"

    pos = {
        "entry_time": df["timestamp"].iloc[-1].isoformat(),
        "entry_price": round(entry_price, 8),
        "initial_stop": round(float(initial_stop), 8),
        "trail_stop": round(float(initial_stop), 8),
        "capital_allocated": POSITION_SIZE_USD,
        "current_price": round(entry_price, 8),
        "unrealized_pnl_pct": 0.0,
        "source": "manual",
    }

    send_telegram_alert(
        f"🤖 Manual Bot Trade OPENED: {symbol}\n"
        f"Entry: {entry_price:.6f} | Initial SL (CE 16,3.0): {float(initial_stop):.6f}\n"
        f"Capital Allocated: ${POSITION_SIZE_USD:.2f} (virtual)"
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
            print(f"  [CLOSED] {symbol}: {closed_row['result']} ${closed_row['realized_pnl_usd']:.2f}")

    positions = still_open

    # ---- STEP 2: watchlist mein jo coins hain unhe process karo (SIRF yehi, koi auto-scan nahi) ----
    watchlist = load_watchlist()
    remaining_watchlist = []

    for symbol in watchlist:
        if symbol in positions:
            print(f"  [SKIP] {symbol}: pehle se hi ek open position mojood hai")
            continue
        try:
            pos, reason = open_manual_position(symbol, exchange, cash, len(positions))
        except Exception as e:
            print(f"  [ERROR] {symbol}: {e}")
            remaining_watchlist.append(symbol)   # error par dobara try karne ke liye rakh lo
            continue

        if pos is not None:
            positions[symbol] = pos
            cash -= pos["capital_allocated"]
            print(f"  [OPENED] {symbol}: entry={pos['entry_price']}, SL={pos['initial_stop']}")
        elif reason in ("MAX_POSITIONS", "NO_CASH", "NO_DATA"):
            print(f"  [QUEUED] {symbol}: abhi nahi ({reason}), agli baar phir koshish hogi")
            remaining_watchlist.append(symbol)
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
