"""
NEW AdvancedConfluence Score Table - 4 tajweez shuda behtariyan ko
WALK-FORWARD (4 sequential folds, ~1 saal) tareeqe se test karte hain,
taake koi bhi tabdeeli sirf "lagta hai acha hoga" ki bajaye asal
sabit-shuda evidence ke sath adopt ho.

Test hone wale candidates (har ek baseline ke khilaf ALAG/standalone
A/B - koi combo test nahi, taake overfitting na ho aur har candidate
ka apna asar saaf pata chale):

  BASELINE = production confluence_engine.py ka wahi system jo abhi
             dashboard mein live hai (10-point score, fixed 1% swing,
             score>=6 + fresh CHoCH, exit Chandelier 16/3.0).

  A) Dynamic Swing Threshold - fixed 1% min-swing ki jagah ATR%-based
     dynamic threshold (2 multiplier settings: 1.0x aur 1.5x ATR%).

  B) Higher-Timeframe (4H) Trend Filter - ek NAYA 2-point component
     (4H close > 4H EMA50), 10-point score 12-point ho jata hai.
     4H data alag se fetch NAHI hota - jo 1h data pehle se fetch ho
     chuka hai, usi se resample hota hai (extra API calls nahi).
     Threshold=7/12 ke sath test.

  C) Overextension Filter - agar price apni EMA20 se bohot door ja
     chuki ho (blow-off top ka khatra) to signal SKIP kar dete hain.
     2 threshold settings: 6% aur 10%.

  D) USDT.Dominance bonus - IS SCRIPT MEIN SHAMIL NAHI. Wajah: mufat
     historical USDT.D data kahin available nahi mila (jaisa
     confluence_engine.py ki docstring mein pehle se likha hai, aur
     isi wajah se production scan bhi ye component hamesha bandh
     rakhta hai - usdt_d_weak=None). Agar koi paid/alternate data
     source mile to alag se test kar lenge.

Exit hamesha fixed (Chandelier period=16, multiplier=3.0) - jaisa
production mein hai - sirf ENTRY (score/filter) side test ho raha hai.

Chalayen (apne PC par):
    pip install -r requirements.txt
    python confirm_confluence_improvements.py

NOTE: 70 coins ka 1 saal ka data fetch karne mein 20-40+ minute lag
sakte hain (KuCoin rate-limit ki wajah se).
"""

import numpy as np
import pandas as pd

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import apply_cooldown
from backtest_engine import simulate_trades
from confluence_engine import compute_atr, compute_adx, compute_rsi, DEFAULT_PARAMS as CONF_PARAMS

TOP_N_COINS = 70
SIGNAL_TIMEFRAME = "1h"
CANDLE_LIMIT = 8760          # ~1 saal
N_FOLDS = 4                  # har fold ~3 maheene

CE_EXIT = {"period": 16, "multiplier": 3.0}   # production jaisa fixed exit
SCORE_THRESHOLD_BASE = 6
SCORE_THRESHOLD_HTF = 7       # 12-point scale par thoda proportional-tight

DYNAMIC_SWING_MULTS = [1.0, 1.5]     # ATR% multiplier candidates
OVEREXTENSION_PCTS = [6.0, 10.0]     # EMA20 se kitna % door = "overextended"


# ============================================================
# MARKET STRUCTURE (parametrized: fixed % ya per-bar dynamic %)
# ============================================================
def detect_market_structure_v2(df, pivot_lookback, swing_threshold):
    """
    swing_threshold: ek fixed float (%, jaise 1.0) YA ek pandas Series
    (per-bar dynamic %, jaise ATR-based threshold). Jis bar par doosra
    pivot confirm hota hai, usi bar ka threshold value istemal hota hai.
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    length = len(df)

    is_pivot_high = np.zeros(length, dtype=bool)
    is_pivot_low = np.zeros(length, dtype=bool)
    for i in range(pivot_lookback, length - pivot_lookback):
        window_h = high[i - pivot_lookback:i + pivot_lookback + 1]
        window_l = low[i - pivot_lookback:i + pivot_lookback + 1]
        if high[i] == window_h.max():
            is_pivot_high[i] = True
        if low[i] == window_l.min():
            is_pivot_low[i] = True

    if isinstance(swing_threshold, pd.Series):
        thresh_arr = swing_threshold.values / 100.0
        fixed_thresh = None
    else:
        thresh_arr = None
        fixed_thresh = swing_threshold / 100.0

    bos_signal = np.zeros(length, dtype=bool)
    choch_signal = np.zeros(length, dtype=bool)
    swing_low_series = np.full(length, np.nan)

    last_pivot_high_val = None
    prev_pivot_high_val = None
    last_pivot_low_val = None
    was_bearish_structure = False
    structure_bullish = False

    for i in range(length):
        if is_pivot_low[i]:
            last_pivot_low_val = low[i]

        if is_pivot_high[i]:
            val = high[i]
            if prev_pivot_high_val is not None:
                swing_pct = abs(val - prev_pivot_high_val) / prev_pivot_high_val
                min_swing = thresh_arr[i] if thresh_arr is not None else fixed_thresh
                if min_swing is not None and not np.isnan(min_swing) and swing_pct >= min_swing:
                    was_bearish_structure = not structure_bullish
                    structure_bullish = val > prev_pivot_high_val
            prev_pivot_high_val = last_pivot_high_val
            last_pivot_high_val = val

        swing_low_series[i] = last_pivot_low_val if last_pivot_low_val is not None else np.nan

        if last_pivot_high_val is not None and structure_bullish:
            fresh_break = close[i] > last_pivot_high_val and (i == 0 or close[i - 1] <= last_pivot_high_val)
            if fresh_break:
                if was_bearish_structure:
                    choch_signal[i] = True
                    was_bearish_structure = False
                else:
                    bos_signal[i] = True

    return (
        pd.Series(bos_signal, index=df.index),
        pd.Series(choch_signal, index=df.index),
        pd.Series(swing_low_series, index=df.index),
    )


def resample_4h(df):
    d = df.copy()
    d.index = pd.to_datetime(d["timestamp"])
    d.index.name = "timestamp"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    r = d.resample("4h").agg(agg).dropna(subset=["open", "high", "low", "close"])
    r = r.reset_index()
    return r


# ============================================================
# CONFLUENCE SCORE (variant-aware: dynamic swing / HTF / overextension)
# ============================================================
def compute_confluence_variant(df, btc_daily, params, df_4h=None,
                                swing_threshold=1.0, dynamic_swing_mult=None,
                                use_htf=False, score_threshold=6, overext_pct=None):
    close = df["close"]
    high = df["high"]
    volume = df["volume"]

    atr14 = compute_atr(df, params["atr_period"])

    if dynamic_swing_mult is not None:
        swing_thresh_series = dynamic_swing_mult * (atr14 / close * 100)
        bos, choch, swing_low = detect_market_structure_v2(df, params["pivot_lookback"], swing_thresh_series)
    else:
        bos, choch, swing_low = detect_market_structure_v2(df, params["pivot_lookback"], swing_threshold)

    structure_signal = bos | choch
    structure_points = structure_signal.astype(int) * 2

    vol_ma = volume.rolling(params["vol_avg_period"]).mean()
    rvol = volume / vol_ma.replace(0, np.nan)
    bullish_candle = close > df["open"]
    rvol_ok = (rvol > params["rvol_threshold"]) & bullish_candle
    rvol_points = rvol_ok.fillna(False).astype(int) * 2

    btc_ema50 = btc_daily["close"].ewm(span=50, adjust=False).mean()
    btc_bullish_daily = (btc_daily["close"] > btc_ema50).rename("btc_bullish")
    btc_rsi_daily = compute_rsi(btc_daily, 14)
    btc_not_overbought = (btc_rsi_daily <= 70).rename("btc_not_ob")
    btc_merge = pd.merge_asof(
        df[["timestamp"]].sort_values("timestamp"),
        pd.DataFrame({
            "timestamp": btc_daily["timestamp"],
            "btc_bullish": btc_bullish_daily,
            "btc_not_ob": btc_not_overbought,
        }).sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    btc_bullish = btc_merge["btc_bullish"].fillna(False).values
    btc_not_ob = btc_merge["btc_not_ob"].fillna(True).values
    btc_macro_ok = btc_bullish & btc_not_ob
    btc_points = pd.Series(btc_macro_ok, index=df.index).astype(int) * 2

    ema50 = close.ewm(span=50, adjust=False).mean()
    near_ema = (close - ema50).abs() / close <= (params["demand_zone_tolerance_pct"] / 100)
    near_swing_low = (close - swing_low).abs() / close <= (params["demand_zone_tolerance_pct"] / 100)
    near_demand = (near_ema | near_swing_low).fillna(False)
    recent_high = high.rolling(params["resistance_lookback"]).max()
    not_near_resistance = (recent_high - close) / close > (params["resistance_buffer_pct"] / 100)
    demand_zone_ok = near_demand & not_near_resistance.fillna(True)
    demand_points = demand_zone_ok.astype(int) * 2

    adx = compute_adx(df, params["adx_period"])
    rsi = compute_rsi(df, params["rsi_period"])
    trend_momentum_ok = (adx > params["adx_threshold"]) & (rsi >= params["rsi_zone_low"]) & (rsi <= params["rsi_zone_high"])
    momentum_points = trend_momentum_ok.fillna(False).astype(int) * 2

    score = structure_points + rvol_points + btc_points + demand_points + momentum_points

    if use_htf and df_4h is not None and len(df_4h) > 60:
        ema50_4h = df_4h["close"].ewm(span=50, adjust=False).mean()
        htf_bullish_4h = df_4h["close"] > ema50_4h
        htf_merge = pd.merge_asof(
            df[["timestamp"]].sort_values("timestamp"),
            pd.DataFrame({"timestamp": df_4h["timestamp"], "htf_ok": htf_bullish_4h.values}).sort_values("timestamp"),
            on="timestamp", direction="backward",
        )
        htf_ok = htf_merge["htf_ok"].fillna(False).values
        htf_points = pd.Series(htf_ok, index=df.index).astype(int) * 2
        score = score + htf_points

    buy_signal = choch & (score >= score_threshold)

    if overext_pct is not None:
        ema20 = close.ewm(span=20, adjust=False).mean()
        overextended = ((close - ema20) / ema20) > (overext_pct / 100)
        buy_signal = buy_signal & (~overextended.fillna(False))

    return buy_signal


def fold_masked_signal(sig_series, n_folds, fold_idx):
    n = len(sig_series)
    bounds = [int(n * i / n_folds) for i in range(n_folds + 1)]
    start, end = bounds[fold_idx], bounds[fold_idx + 1]
    masked = sig_series.copy()
    masked.iloc[:start] = False
    masked.iloc[end:] = False
    return masked


# ============================================================
# CANDIDATE DEFINITIONS
# ============================================================
def build_candidates():
    cands = {"Baseline (10pt, fixed 1% swing)": dict(swing_threshold=1.0, dynamic_swing_mult=None, use_htf=False, score_threshold=SCORE_THRESHOLD_BASE, overext_pct=None)}
    for mult in DYNAMIC_SWING_MULTS:
        cands[f"Dynamic Swing ({mult}x ATR%)"] = dict(swing_threshold=1.0, dynamic_swing_mult=mult, use_htf=False, score_threshold=SCORE_THRESHOLD_BASE, overext_pct=None)
    cands[f"HTF 4H Filter Added (12pt, threshold={SCORE_THRESHOLD_HTF})"] = dict(swing_threshold=1.0, dynamic_swing_mult=None, use_htf=True, score_threshold=SCORE_THRESHOLD_HTF, overext_pct=None)
    for pct in OVEREXTENSION_PCTS:
        cands[f"Overextension Filter ({pct}%)"] = dict(swing_threshold=1.0, dynamic_swing_mult=None, use_htf=False, score_threshold=SCORE_THRESHOLD_BASE, overext_pct=pct)
    return cands


def main():
    exchange = get_exchange()
    candidates = build_candidates()

    print("BTC daily benchmark data fetch kar rahe hain...")
    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=800)

    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"Top {len(coins)} coins, {SIGNAL_TIMEFRAME}, {CANDLE_LIMIT} candles, {N_FOLDS} folds, {len(candidates)} candidates...\n")

    results = {name: {f: [] for f in range(N_FOLDS)} for name in candidates}
    done = 0

    for symbol in coins:
        done += 1
        try:
            df = fetch_ohlcv(exchange, symbol, SIGNAL_TIMEFRAME, limit=CANDLE_LIMIT)
        except Exception as e:
            print(f"[{done}/{len(coins)}] {symbol}: fetch fail ({e})")
            continue
        if df is None or len(df) < 500:
            continue

        df_4h = None
        if any(c["use_htf"] for c in candidates.values()):
            try:
                df_4h = resample_4h(df)
            except Exception as e:
                print(f"[{done}/{len(coins)}] {symbol}: 4H resample fail ({e})")

        for name, params_variant in candidates.items():
            try:
                buy_signal = compute_confluence_variant(
                    df, btc_daily, CONF_PARAMS, df_4h=df_4h, **params_variant
                )
                sig = apply_cooldown(buy_signal, config.SIGNAL_COOLDOWN_BARS)
            except Exception as e:
                print(f"[{done}/{len(coins)}] {symbol}: candidate={name} calc fail ({e})")
                continue

            for fold_idx in range(N_FOLDS):
                masked = fold_masked_signal(sig, N_FOLDS, fold_idx)
                try:
                    trades = simulate_trades(df, masked, config.BACKTEST_PARAMS, CE_EXIT)
                    results[name][fold_idx].extend(trades)
                except Exception as e:
                    print(f"[{done}/{len(coins)}] {symbol}: candidate={name} fold={fold_idx} sim fail ({e})")

        if done % 10 == 0 or done == len(coins):
            print(f"[{done}/{len(coins)}] ... done")

    output_lines = []

    def emit(line=""):
        print(line)
        output_lines.append(line)

    def pf_of(trades):
        if not trades:
            return None, 0, None
        df_t = pd.DataFrame(trades)
        wins = df_t[df_t["return_pct"] > 0]
        losses = df_t[df_t["return_pct"] <= 0]
        win_rate = len(wins) / len(df_t) * 100
        pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
        return pf, len(df_t), win_rate

    emit("\n\n########## CONFLUENCE SCORE IMPROVEMENTS - WALK-FORWARD RESULTS ##########")
    emit(f"({N_FOLDS} sequential folds, ~1 saal, exit fixed Chandelier 16/3.0)\n")
    emit("NOTE: USDT.Dominance candidate is script mein shamil NAHI (mufat historical data available nahi).\n")

    for name in candidates:
        emit(f"\n----- {name} -----")
        header = "  " + " | ".join([f"Fold{i+1} PF (trades, win%)" for i in range(N_FOLDS)])
        emit(header)
        cells = []
        for fold_idx in range(N_FOLDS):
            pf, n, wr = pf_of(results[name][fold_idx])
            if pf is None:
                cells.append("N/A")
            else:
                cells.append(f"{pf:.2f} ({n}, {wr:.0f}%)")
        emit("  " + " | ".join(cells))

        all_trades = []
        for fold_idx in range(N_FOLDS):
            all_trades.extend(results[name][fold_idx])
        pf, n, wr = pf_of(all_trades)
        if pf is None:
            emit("  Overall (4 folds combined): koi trades nahi")
        else:
            emit(f"  Overall (4 folds combined): Trades={n}, Win Rate={wr:.2f}%, PF={pf:.3f}")

    result_file = "confirm_confluence_improvements_RESULTS.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\n[SAVE] Results file mein bhi save ho gaye: {result_file}")


if __name__ == "__main__":
    main()
