"""
Auto-Scan Paper-Trading Bot - "dummy" auto mode, manual_trade_bot.py ka
sāthi. Khud 150 coins scan karta hai aur live systems (Union AB, NEW
AdvancedConfluence, CE Buy-Only, Pullback-in-Uptrend, Donchian Breakout)
ke signal par khud paper/virtual trade le leta hai.

YE ASAL PAISON SE TRADE NAHI KARTA (paper/virtual hai).

Dono bots (manual + auto) ki ledger/capital BILKUL ALAG hain
(auto_bot_state.json / auto_bot_closed_trades.csv).

=====================================================================
NAYE QAWAID (nuqsan kam karne ke liye) - sab neeche SETTINGS mein
ON/OFF ho sakte hain:

 1) SIRF STRONG SIGNAL:
      - Union AB / NEW AdvancedConfluence: sirf wahi signal jis ka
        Verdict live dashboard (dashboard_signals.json) par "Strong" ho
        (STRONG_VERDICTS mein "Good" bhi shamil kar sakte hain).
      - CE Buy-Only: sirf jab ETH bullish ho ("Base+ETH" tier).
      - Pullback-in-Uptrend / Donchian Breakout: in par pehle se hi
        ETH + RS + RS%95 ke sakht filter lage hain (in ka alag Verdict
        nahi banta) - is liye ye waise hi chalte hain. Band karne hon
        to SYSTEMS_ENABLED mein False kar dein.

 2) FORAN ENTRY (late nahi): signal sirf BAND ho chuki candle par
    maana jata hai (chalti candle ka signal candle band hone par
    ghayab ho sakta hai), aur entry sirf tab jab signal sab se aakhri
    band candle par bana ho (MAX_SIGNAL_AGE_BARS). Pehle 3 candle
    purane signal par bhi entry ho jati thi.

 3) EK COIN, EK TRADE: ek hi coin par kai systems ki ek sath trades
    nahi (misaal: SOL par 3 trades = 3 guna nuqsan ka khatra).

 4) EK SIGNAL, EK DAFA: jis signal par trade li ja chuki, us par
    dobara trade nahi (chahe pehli trade SL par band ho gayi ho).

 5) LOSS-STREAK BREAK: agar thori der mein kai trades SL par band hon
    (misaal 24 ghante mein 3), to kuch ghante NAYI entry band - taake
    market ki ek hi giravat mein bar bar nuqsan na ho. Khuli trades
    ka SL/TP is dauran bhi chalta rehta hai.

 6) MARKET CRASH GUARD (BTC / ETH par nazar):
      - EHTIYAT: BTC ya ETH apni pichle 4 ghante ki bulandi se 2% gir
        jaye -> sirf NAYI entry band (khuli trades chalti rahengi).
      - CRASH: 4 ghante mein 5% ya 24 ghante mein 8% giravat -> 6
        ghante tak koi nayi entry nahi + Telegram par ittela.
        (CRASH_CLOSE_ALL = True karne par saari khuli trades bhi foran
        band hoti hain - lekin default OFF hai: SL check pehle hota hai,
        is liye girti trades CE trailing SL par khud nikal chuki hoti
        hain; close-all sirf mazboot/nafa wali trades ko kaat-ta.)

 7) ⏸️ ROKNE KA BUTTON: dashboard ke Auto Bot hisse mein "Nayi Entry
    Rokein" button auto_bot_control.json ({"paused": true}) GitHub par
    likhta hai. Roka hua bot koi NAYI trade nahi leta, lekin khuli
    trades ka SL/TP barabar check karta rehta hai. Bot ye file GitHub se
    taaza parhta hai (run ke shuru mein aur scan ke dauran bhi).

NOTE: "Union AB Backup Tier" is auto-scan mein SHAAMIL NAHI - wo Union
AB ke usi signal par ban sakti hai (dohri trade ka khatra).
=====================================================================
"""

import json
import os
from datetime import datetime, timezone, timedelta

import pandas as pd

import config
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from confluence_engine import compute_confluence, DEFAULT_PARAMS as CONF_PARAMS

import scheduled_dashboard_scan as live
import manual_trade_bot as mbot

STATE_FILE = "auto_bot_state.json"
CLOSED_TRADES_FILE = "auto_bot_closed_trades.csv"
DASHBOARD_FILE = "dashboard_signals.json"
CONTROL_FILE = "auto_bot_control.json"      # dashboard ka "Rokein / Chalayein" button isay likhta hai
CONTROL_RECHECK_EVERY_COINS = 30            # scan ke dauran har itne coins baad button dobara check

STARTING_CAPITAL = 2000.0
POSITION_SIZE_USD = 100.0
MAX_CONCURRENT_POSITIONS = 15

# ============================= NAYI SETTINGS =============================
STRONG_ONLY = True                       # False = purana tareeqa (har signal par entry)
STRONG_VERDICTS = ("Strong",)            # "Good" bhi chahiye to: ("Strong", "Good")
MAX_SIGNAL_AGE_BARS = 1                  # 0 = sirf aakhri band candle; 1 = ek candle ki gunjaish (run late ho to)
ONE_POSITION_PER_COIN = True
LOSS_STREAK_LOSSES = 3                   # itni SL ...
LOSS_STREAK_WINDOW_HOURS = 24            # ... itne ghanton mein ho jayen to
LOSS_STREAK_PAUSE_HOURS = 12             # itne ghante nayi entry band

# ---- Market crash guard (BTC + ETH, 1h candles) ----
MARKET_GUARD_ON = True
CAUTION_DROP_4H_PCT = 2.0        # BTC/ETH 4 ghante ki bulandi se itna % girein: nayi entry band
CRASH_DROP_4H_PCT = 5.0          # 4 ghante mein itna % -> saari trades band
CRASH_DROP_24H_PCT = 8.0         # ya 24 ghante mein itna % -> saari trades band
CRASH_CLOSE_ALL = False          # True = crash par saari khuli trades foran band. False (behtar): CE trailing SL khud nikalta hai, sirf nayi entry band
CRASH_COOLDOWN_HOURS = 6         # crash ke baad itne ghante nayi entry band

SYSTEMS_ENABLED = {
    "Union AB": True,
    "NEW AdvancedConfluence": True,
    "CE Buy-Only": True,
    "Pullback-in-Uptrend": True,
    "Donchian Breakout": True,
}


# ============================= STATE =============================
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


# ============================= HELPERS =============================
def _utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def closed_candles_only(df, now_utc, tf_hours=1):
    """Aakhri candle abhi chal rahi ho (band nahi hui) to use hata deta hai -
    chalti candle ka signal band hone se pehle ghayab ho sakta hai."""
    if df is None or len(df) == 0:
        return df
    last_open = _utc(df["timestamp"].iloc[-1])
    if last_open + pd.Timedelta(hours=tf_hours) > now_utc:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def load_dashboard_verdicts():
    """dashboard_signals.json se (Coin, System, Combo, Signal Time) -> Verdict."""
    lookup = {}
    try:
        with open(DASHBOARD_FILE) as f:
            data = json.load(f)
        for r in data.get("signals", []):
            k = (r.get("Coin"), r.get("System"), r.get("Combo"), r.get("Signal Time (PKT)"))
            lookup[k] = str(r.get("Verdict", ""))
    except Exception as e:
        print(f"  [WARN] {DASHBOARD_FILE} parha nahi ja saka ({e}) - Union AB/NEW strong check skip hoga")
    return lookup


def is_strong_verdict(verdict):
    return any(v.lower() in str(verdict).lower() for v in STRONG_VERDICTS)


def signal_key(symbol, system, combo, sig_ts):
    return f"{symbol}|{system}|{combo or ''}|{_utc(sig_ts).isoformat()}"


def is_manually_paused():
    """
    Dashboard ke button ki halat. GitHub se TAAZA parhta hai (taake bot ke
    chalte hue button dabaya jaye to bhi foran pata chale); GitHub na mile
    to repo ki local copy. File na ho = chal raha hai.
    """
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        try:
            import base64
            import requests
            headers = {"Accept": "application/vnd.github.v3+json"}
            token = os.environ.get("GITHUB_TOKEN")
            if token:
                headers["Authorization"] = f"token {token}"
            r = requests.get(f"https://api.github.com/repos/{repo}/contents/{CONTROL_FILE}",
                             headers=headers, params={"ref": "main"}, timeout=10)
            if r.status_code == 200:
                data = json.loads(base64.b64decode(r.json().get("content", "")).decode("utf-8") or "{}")
                return bool(data.get("paused"))
            if r.status_code == 404:
                return False
            print(f"  [CONTROL] GitHub se button ki halat nahi mili ({r.status_code}) - local copy dekh rahe hain")
        except Exception as e:
            print(f"  [CONTROL] GitHub check fail ({e}) - local copy dekh rahe hain")
    try:
        with open(CONTROL_FILE) as f:
            return bool(json.load(f).get("paused"))
    except Exception:
        return False


def check_market(exchange):
    """
    BTC aur ETH ki 1h candles se giravat naapta hai (chalti candle samet,
    yani abhi ki qeemat). Returns (level, details):
      level = "OK" / "CAUTION" / "CRASH"
    """
    worst = "OK"
    details = []
    rank = {"OK": 0, "CAUTION": 1, "CRASH": 2}
    for sym in ("BTC/USDT", "ETH/USDT"):
        try:
            df = live.fetch_ohlcv(exchange, sym, "1h", limit=30)
            if df is None or len(df) < 26:
                continue
            last = float(df["close"].iloc[-1])
            hi_4h = float(df["high"].iloc[-5:].max())
            hi_24h = float(df["high"].iloc[-25:].max())
            dd_4h = (last - hi_4h) / hi_4h * 100
            dd_24h = (last - hi_24h) / hi_24h * 100
            level = "OK"
            if dd_4h <= -CRASH_DROP_4H_PCT or dd_24h <= -CRASH_DROP_24H_PCT:
                level = "CRASH"
            elif dd_4h <= -CAUTION_DROP_4H_PCT:
                level = "CAUTION"
            details.append(f"{sym.split('/')[0]} 4h:{dd_4h:+.1f}% 24h:{dd_24h:+.1f}%")
            if rank[level] > rank[worst]:
                worst = level
        except Exception as e:
            print(f"  [MARKET-CHECK-FAIL] {sym}: {e}")
    return worst, ", ".join(details)


def force_close_position(symbol, pos, exchange, reason="CRASH_EXIT"):
    """Trade ko abhi ki qeemat par band karta hai (chahe nafa ho ya nuqsan)."""
    df = live.fetch_ohlcv(exchange, symbol, "1h", limit=5)
    exit_price = float(df["close"].iloc[-1])
    entry_price = pos["entry_price"]
    gross_return = (exit_price - entry_price) / entry_price
    net_return = gross_return - (2 * mbot.FEE_PCT)
    realized_pnl = pos["capital_allocated"] * net_return
    return {
        "symbol": symbol,
        "system": pos.get("system", "CE Buy-Only"),
        "entry_time_pkt": mbot.to_pkt_str(pos["entry_time"]),
        "exit_time_pkt": mbot.to_pkt_str(pd.Timestamp.now(tz="UTC")),
        "entry_price": round(entry_price, 8),
        "exit_price": round(exit_price, 8),
        "exit_reason": reason,
        "capital_allocated_usd": pos["capital_allocated"],
        "gross_return_pct": round(gross_return * 100, 3),
        "net_return_pct": round(net_return * 100, 3),
        "realized_pnl_usd": round(realized_pnl, 4),
        "result": "WIN" if net_return > 0 else "LOSS",
    }, realized_pnl


# ============================= MAIN =============================
def main():
    now_utc = pd.Timestamp.now(tz="UTC")
    exchange = live.get_exchange()
    state = load_state()
    cash = state.get("cash", STARTING_CAPITAL)
    positions = state.get("positions", {})
    taken_signals = state.get("taken_signals", {})
    recent_losses = state.get("recent_losses", [])
    pause_until = state.get("pause_until")
    crash_until = state.get("crash_until")
    market_level, market_info = "OK", ""

    print(f"Auto-scan bot shuru: cash=${cash:.2f}, open positions={len(positions)}")

    # ---- STEP 1: khuli positions update/close ----
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
            if closed_row.get("result") == "LOSS":
                recent_losses.append(now_utc.isoformat())
            print(f"  [CLOSED] {key}: {closed_row['result']} ${closed_row['realized_pnl_usd']:.2f}")

    positions = still_open

    # ---- Market crash guard (BTC / ETH) ----
    if MARKET_GUARD_ON:
        market_level, market_info = check_market(exchange)
        print(f"  [MARKET] {market_level} ({market_info})")
        if market_level == "CRASH":
            crash_until = (now_utc + pd.Timedelta(hours=CRASH_COOLDOWN_HOURS)).isoformat()
            closed_lines = []
            if CRASH_CLOSE_ALL and positions:
                for key, pos in list(positions.items()):
                    try:
                        row, pnl = force_close_position(pos["symbol"], pos, exchange)
                    except Exception as e:
                        print(f"  [CRASH-CLOSE-FAIL] {key}: {e} (agli baar dobara koshish)")
                        continue
                    cash += pos["capital_allocated"] + pnl
                    append_closed_trade(row)
                    del positions[key]
                    closed_lines.append(f"{row['symbol']} {row['net_return_pct']:+.2f}%")
                    print(f"  [CRASH-EXIT] {key}: {row['result']} ${row['realized_pnl_usd']:.2f}")
            try:
                live.send_telegram_alert(
                    f"🚨 Auto Bot MARKET CRASH ({market_info})\n"
                    + (f"Band ki gayi trades: {', '.join(closed_lines)}\n" if closed_lines else "")
                    + f"Nayi entries {CRASH_COOLDOWN_HOURS} ghante ke liye band."
                )
            except Exception:
                pass
    if crash_until is not None and _utc(crash_until) <= now_utc:
        crash_until = None

    # ---- Loss-streak break ----
    window_start = now_utc - pd.Timedelta(hours=LOSS_STREAK_WINDOW_HOURS)
    recent_losses = [t for t in recent_losses if _utc(t) > window_start]
    paused = pause_until is not None and _utc(pause_until) > now_utc
    if not paused and LOSS_STREAK_LOSSES > 0 and len(recent_losses) >= LOSS_STREAK_LOSSES:
        pause_until = (now_utc + pd.Timedelta(hours=LOSS_STREAK_PAUSE_HOURS)).isoformat()
        recent_losses = []   # break ke baad ginti naye sire se
        paused = True
        print(f"  [PAUSE] {LOSS_STREAK_LOSSES} SL {LOSS_STREAK_WINDOW_HOURS} ghanton mein - "
              f"nayi entry {LOSS_STREAK_PAUSE_HOURS} ghante band")
        try:
            live.send_telegram_alert(
                f"⏸️ Auto Bot: {LOSS_STREAK_LOSSES} trades {LOSS_STREAK_WINDOW_HOURS} ghanton mein SL - "
                f"nayi entries {LOSS_STREAK_PAUSE_HOURS} ghante ke liye band. Khuli trades chalti rahengi."
            )
        except Exception:
            pass
    if not paused:
        pause_until = None

    # purani taken-signal keys saaf (7 din se purani)
    cutoff = now_utc - pd.Timedelta(days=7)
    taken_signals = {k: v for k, v in taken_signals.items() if _utc(v) > cutoff}

    manual_pause = is_manually_paused()

    def finish():
        total_equity = cash + sum(p["capital_allocated"] for p in positions.values())
        save_state({
            "cash": round(cash, 4),
            "positions": positions,
            "total_equity_usd": round(total_equity, 4),
            "starting_capital_usd": STARTING_CAPITAL,
            "open_position_count": len(positions),
            "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
            "position_size_usd": POSITION_SIZE_USD,
            "taken_signals": taken_signals,
            "recent_losses": recent_losses,
            "pause_until": pause_until,
            "crash_until": crash_until,
            "market_level": market_level,
            "market_info": market_info,
            "manual_pause": manual_pause,
            "rules": {
                "strong_only": STRONG_ONLY, "strong_verdicts": list(STRONG_VERDICTS),
                "max_signal_age_bars": MAX_SIGNAL_AGE_BARS, "one_position_per_coin": ONE_POSITION_PER_COIN,
            },
        })
        print(f"Auto-scan bot khatam: cash=${cash:.2f}, open positions={len(positions)}, total equity=${total_equity:.2f}")

    if manual_pause:
        print("  [ROKA HUA] Dashboard ke button se bot roka gaya hai - koi nayi trade nahi, sirf khuli trades update hui.")
        finish()
        return
    if paused:
        print(f"  [PAUSED] Nayi entry band hai {pause_until} (UTC) tak - sirf khuli trades update hui.")
        finish()
        return
    if crash_until is not None:
        print(f"  [CRASH-BLOCK] Market crash ki wajah se nayi entry band hai {crash_until} (UTC) tak.")
        finish()
        return
    if market_level == "CAUTION":
        print(f"  [CAUTION] BTC/ETH tezi se gir rahe hain - is run mein nayi entry nahi.")
        finish()
        return

    # ---- STEP 2: scan + entry ----
    verdicts = load_dashboard_verdicts() if STRONG_ONLY else {}

    btc_daily = live.fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=800)
    eth_daily = live.fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=800)
    eth_regime = live.compute_eth_regime(eth_daily, ema_period=live.ETH_EMA_PERIOD)

    coins = live.get_coin_list(exchange)[:live.TOP_N_COINS]
    print(f"Scanning {len(coins)} coins (strong_only={STRONG_ONLY}, max_age={MAX_SIGNAL_AGE_BARS} candle, "
          f"one_per_coin={ONE_POSITION_PER_COIN})...")

    skipped = {"weak": 0, "old": 0, "coin_busy": 0, "already_taken": 0}

    def fresh(found):
        if found is None:
            return False
        if found["bars_since_entry"] > MAX_SIGNAL_AGE_BARS:
            skipped["old"] += 1
            return False
        return True

    def try_open(symbol, system, sig_ts, combo=None):
        nonlocal cash
        if not SYSTEMS_ENABLED.get(system, False):
            return
        key = position_key(symbol, system, combo)
        if key in positions:
            return
        if ONE_POSITION_PER_COIN and any(p.get("symbol") == symbol for p in positions.values()):
            skipped["coin_busy"] += 1
            return
        skey = signal_key(symbol, system, combo, sig_ts)
        if skey in taken_signals:
            skipped["already_taken"] += 1
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
            pos["signal_time"] = _utc(sig_ts).isoformat()
            positions[key] = pos
            cash -= pos["capital_allocated"]
            taken_signals[skey] = now_utc.isoformat()
            print(f"  [AUTO-OPENED] {symbol} ({system}{'/' + combo if combo else ''}): "
                  f"entry={pos['entry_price']}, SL={pos['initial_stop']}, TP={pos['tp_price']}")

    for coin_i, symbol in enumerate(coins):
        if coin_i > 0 and coin_i % CONTROL_RECHECK_EVERY_COINS == 0 and is_manually_paused():
            manual_pause = True
            print(f"  [ROKA HUA] Scan ke dauran button se roka gaya - {coin_i} coins ke baad scan band.")
            break
        try:
            df_raw = live.fetch_ohlcv(exchange, symbol, live.SIGNAL_TIMEFRAME,
                                      limit=max(config.CANDLE_LIMITS.get(live.SIGNAL_TIMEFRAME, 500), 300))
        except Exception:
            df_raw = None
        df = closed_candles_only(df_raw, now_utc)
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
                if fresh(found) and rs_bullish is not None:
                    idx = found["signal_idx"]
                    tier, _ = live.evaluate_union_ab_tier(symbol, df, idx, eth_regime, rs_bullish, rs_ratio_series, coin_daily)
                    if tier is not None:
                        sig_ts = df["timestamp"].iloc[idx]
                        if STRONG_ONLY:
                            v = verdicts.get((symbol, "Union AB", combo_name, live.to_pkt_str(sig_ts)))
                            if not is_strong_verdict(v):
                                skipped["weak"] += 1
                                continue
                        try_open(symbol, "Union AB", sig_ts, combo_name)
        except Exception as e:
            print(f"  [SKIP-UnionAB] {symbol}: {e}")

        # ---- NEW AdvancedConfluence ----
        try:
            result_new = compute_confluence(df, btc_daily, CONF_PARAMS, usdt_d_weak=None)
            choch_signal = result_new["choch"]
            new_sig = apply_cooldown(choch_signal & (result_new["score"] >= CONF_PARAMS["score_threshold"]), config.SIGNAL_COOLDOWN_BARS)
            found = live.find_latest_open_or_new(df, new_sig, live.CE_D["period"], live.CE_D["multiplier"])
            if fresh(found):
                sig_ts = df["timestamp"].iloc[found["signal_idx"]]
                strong_ok = True
                if STRONG_ONLY:
                    v = verdicts.get((symbol, "NEW AdvancedConfluence", "CHoCH", live.to_pkt_str(sig_ts)))
                    strong_ok = is_strong_verdict(v)
                if strong_ok:
                    try_open(symbol, "NEW AdvancedConfluence", sig_ts)
                else:
                    skipped["weak"] += 1
        except Exception as e:
            print(f"  [SKIP-NEWConf] {symbol}: {e}")

        # ---- CE Buy-Only (strong = ETH bullish) ----
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
            if fresh(found):
                sig_ts = df["timestamp"].iloc[found["signal_idx"]]
                if STRONG_ONLY and not live.is_bullish_at(eth_regime, pd.Timestamp(sig_ts)):
                    skipped["weak"] += 1
                else:
                    try_open(symbol, "CE Buy-Only", sig_ts)
        except Exception as e:
            print(f"  [SKIP-CEBuyOnly] {symbol}: {e}")

        # ---- Pullback-in-Uptrend + Donchian Breakout (ETH + RS + RS%95 filters pehle se) ----
        if rs_bullish is not None:
            for sys_name, sig_fn in (("Pullback-in-Uptrend", live.pullback_uptrend_entry),
                                     ("Donchian Breakout", live.donchian_channel_breakout)):
                try:
                    sig = apply_cooldown(sig_fn(df), config.SIGNAL_COOLDOWN_BARS)
                    found = live.find_latest_open_or_new(df, sig, live.CE_PB["period"], live.CE_PB["multiplier"], use_fixed_tp=False)
                    if fresh(found):
                        sig_ts = pd.Timestamp(df["timestamp"].iloc[found["signal_idx"]])
                        ok, _ = live.passes_full_filters(sig_ts, eth_regime, rs_bullish, rs_ratio_series)
                        if ok:
                            try_open(symbol, sys_name, sig_ts)
                except Exception as e:
                    print(f"  [SKIP-{sys_name}] {symbol}: {e}")

        if len(positions) >= MAX_CONCURRENT_POSITIONS or cash < POSITION_SIZE_USD:
            print("  Capital/positions full - baaqi coins ka scan agli baar.")
            break

    print(f"  Chhor diye: kamzor(weak)={skipped['weak']}, purane(late)={skipped['old']}, "
          f"coin pehle se khula={skipped['coin_busy']}, signal pehle le chuke={skipped['already_taken']}")
    finish()


if __name__ == "__main__":
    main()
