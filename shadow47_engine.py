"""
SHADOW47 AI Structure Engine PRO v2.1 - Pine Script se Python mein
FAITHFUL conversion, taake hamare historical data par backtest ho sake.

User ne diya hua poora Pine Script (structure/BOS-CHOCH, liquidity sweep,
FVG, Premium/Discount, 4x Multi-Timeframe trend, BTC correlation/RS,
weighted 0-100 scoring, apna dynamic SL + multi-TP plan) isi tarteeb
mein yahan dobara likha gaya hai.

Kuch chhoti approximations (kyunke Pine Script ke kuch built-ins ka
1:1 Python barabar nahi hota):
  - Pine ka ta.atr() Wilder/RMA smoothing use karta hai; yahan hum
    (poore project ki tarah) simple rolling-mean ATR use kar rahe
    hain, jaisa confluence_engine.py mein hai - taake consistency
    bani rahe. Farq chhota hota hai.
  - Multi-Timeframe (request.security) ko yahan har HTF ka apna
    OHLCV fetch kar ke, "aakhri band hui HTF candle" ko current bar
    par merge_asof (backward) se align kiya gaya hai - jitna mumkin
    ho utna Pine ke lookahead_off jaisa (koi future data nahi).
  - EQH/EQL sirf visual label ke liye thi (scoring mein shamil nahi
    thi Pine mein bhi), is liye yahan bhi skip ki gayi hai.

Ye module sirf ENGINE hai (koi backtest nahi) - test_shadow47.py isay
istemal kar ke buy signals + apna SL/TP plan nikalta hai.
"""

import numpy as np
import pandas as pd

# ---------------- Defaults (Pine script ke inputs jaisa) ----------------
DEFAULT_PARAMS = {
    "pivot_len": 3,
    "ema_fast_len": 50,
    "ema_slow_len": 200,
    "rsi_len": 14,
    "vol_len": 20,
    "min_vol_ratio": 1.00,
    "vol_spike_ratio": 1.60,
    "atr_len": 14,
    "atr_regime_len": 50,
    "fvg_min_atr": 0.10,
    "corr_len": 50,
    "rs_len": 20,
    "signal_threshold": 72.0,
    "min_score_edge": 12.0,
    "recent_break_bars": 8,
    "sl_atr_mult": 1.30,
    "structure_buffer": 0.15,
    "stop_mode": "Hybrid",     # "ATR" | "Structure" | "Hybrid"
    "rr1": 1.0,
    "rr2": 2.0,
    "rr3": 3.0,
}


# ============================================================
# BASIC INDICATORS
# ============================================================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def compute_atr_simple(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd_hist(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    macd_signal = ema(macd_line, signal)
    return macd_line - macd_signal


def compute_roc(close, length):
    return (close - close.shift(length)) / close.shift(length) * 100.0


# ============================================================
# PIVOTS (ta.pivothigh / ta.pivotlow jaisa, causal - koi lookahead nahi)
# ============================================================
def compute_pivots(high, low, pivot_len):
    n = len(high)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    for i in range(2 * pivot_len, n):
        center = i - pivot_len
        lo = center - pivot_len
        hi = center + pivot_len + 1
        window_h = high[lo:hi]
        window_l = low[lo:hi]
        ch = high[center]
        cl = low[center]
        if ch == window_h.max() and (window_h == ch).sum() == 1:
            ph[i] = ch
        if cl == window_l.min() and (window_l == cl).sum() == 1:
            pl[i] = cl
    return ph, pl


# ============================================================
# STRUCTURE: HH/HL/LH/LL, BOS/CHOCH, Liquidity Sweeps
# ============================================================
def compute_structure(high, low, close, pivot_len):
    n = len(high)
    ph_arr, pl_arr = compute_pivots(high, low, pivot_len)

    lastPH = np.nan
    prevPH = np.nan
    lastPL = np.nan
    prevPL = np.nan
    lastHighWasHH = False
    lastHighWasLH = False
    lastLowWasHL = False
    lastLowWasLL = False
    structureBias = 0

    out = {
        "lastPH": np.full(n, np.nan), "lastPL": np.full(n, np.nan),
        "structureBias": np.zeros(n, dtype=int),
        "bullBreak": np.zeros(n, dtype=bool), "bearBreak": np.zeros(n, dtype=bool),
        "bullCHOCH": np.zeros(n, dtype=bool), "bullBOS": np.zeros(n, dtype=bool),
        "bearCHOCH": np.zeros(n, dtype=bool), "bearBOS": np.zeros(n, dtype=bool),
        "bullSweep": np.zeros(n, dtype=bool), "bearSweep": np.zeros(n, dtype=bool),
        "lastHighWasHH": np.zeros(n, dtype=bool), "lastHighWasLH": np.zeros(n, dtype=bool),
        "lastLowWasHL": np.zeros(n, dtype=bool), "lastLowWasLL": np.zeros(n, dtype=bool),
    }

    prev_close = np.nan

    for i in range(n):
        if not np.isnan(ph_arr[i]):
            prevPH = lastPH
            lastPH = ph_arr[i]
            if not np.isnan(prevPH):
                lastHighWasHH = lastPH > prevPH
                lastHighWasLH = lastPH <= prevPH
        if not np.isnan(pl_arr[i]):
            prevPL = lastPL
            lastPL = pl_arr[i]
            if not np.isnan(prevPL):
                lastLowWasHL = lastPL > prevPL
                lastLowWasLL = lastPL <= prevPL

        c = close[i]
        bullBreak = False
        bearBreak = False
        if not np.isnan(lastPH) and not np.isnan(prev_close):
            if prev_close <= lastPH and c > lastPH:
                bullBreak = True
        if not np.isnan(lastPL) and not np.isnan(prev_close):
            if prev_close >= lastPL and c < lastPL:
                bearBreak = True

        bullCHOCH = bullBreak and structureBias == -1
        bullBOS = bullBreak and structureBias != -1
        bearCHOCH = bearBreak and structureBias == 1
        bearBOS = bearBreak and structureBias != 1

        if bullBreak:
            structureBias = 1
        if bearBreak:
            structureBias = -1

        bullSweep = (not np.isnan(lastPL)) and (low[i] < lastPL) and (c > lastPL)
        bearSweep = (not np.isnan(lastPH)) and (high[i] > lastPH) and (c < lastPH)

        out["lastPH"][i] = lastPH
        out["lastPL"][i] = lastPL
        out["structureBias"][i] = structureBias
        out["bullBreak"][i] = bullBreak
        out["bearBreak"][i] = bearBreak
        out["bullCHOCH"][i] = bullCHOCH
        out["bullBOS"][i] = bullBOS
        out["bearCHOCH"][i] = bearCHOCH
        out["bearBOS"][i] = bearBOS
        out["bullSweep"][i] = bullSweep
        out["bearSweep"][i] = bearSweep
        out["lastHighWasHH"][i] = lastHighWasHH
        out["lastHighWasLH"][i] = lastHighWasLH
        out["lastLowWasHL"][i] = lastLowWasHL
        out["lastLowWasLL"][i] = lastLowWasLL

        prev_close = c

    return out


def bars_since(bool_arr):
    n = len(bool_arr)
    out = np.full(n, np.nan)
    last_idx = None
    for i in range(n):
        if bool_arr[i]:
            last_idx = i
        if last_idx is not None:
            out[i] = i - last_idx
    return out


def compute_fvg(df, atr_vals, fvg_min_atr):
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    bullFVG = np.zeros(n, dtype=bool)
    bearFVG = np.zeros(n, dtype=bool)
    for i in range(2, n):
        a = atr_vals[i]
        if np.isnan(a):
            continue
        if low[i] > high[i - 2] and (low[i] - high[i - 2]) >= a * fvg_min_atr:
            bullFVG[i] = True
        if high[i] < low[i - 2] and (low[i - 2] - high[i]) >= a * fvg_min_atr:
            bearFVG[i] = True
    return bullFVG, bearFVG


def htf_bull_bear(htf_df, ema_fast_len, ema_slow_len):
    """HTF (higher timeframe) ke har band hue bar ke liye bull/bear boolean."""
    ef = ema(htf_df["close"], ema_fast_len)
    es = ema(htf_df["close"], ema_slow_len)
    bull = (htf_df["close"] > ef) & (ef > es)
    bear = (htf_df["close"] < ef) & (ef < es)
    out = pd.DataFrame({"timestamp": htf_df["timestamp"], "bull": bull, "bear": bear})
    return out


def align_htf_to_base(base_ts, htf_bullbear_df):
    """Base timeframe ke har bar ko, us waqt tak ki AAKHRI band hui HTF
    candle ke bull/bear se align karta hai (merge_asof backward - koi
    lookahead nahi)."""
    base = pd.DataFrame({"timestamp": base_ts}).sort_values("timestamp")
    htf = htf_bullbear_df.sort_values("timestamp")
    merged = pd.merge_asof(base, htf, on="timestamp", direction="backward")
    bull = merged["bull"].fillna(False).values
    bear = merged["bear"].fillna(False).values
    return bull, bear


def align_btc_close(base_ts, btc_df):
    base = pd.DataFrame({"timestamp": base_ts}).sort_values("timestamp")
    btc = btc_df[["timestamp", "close"]].rename(columns={"close": "btc_close"}).sort_values("timestamp")
    merged = pd.merge_asof(base, btc, on="timestamp", direction="backward")
    return merged["btc_close"].values


# ============================================================
# MAIN: SCORE + SIGNAL COMPUTE
# ============================================================
def compute_shadow47(df, htf_dfs, btc_df, params=None):
    """
    df: base timeframe (1h) OHLCV dataframe (coin)
    htf_dfs: dict {"tf1": df15m, "tf2": df1h, "tf3": df4h, "tf4": df1d}
             (tf2 hamesha base df ke barabar bhi ho sakta hai)
    btc_df: BTC OHLCV (isi base timeframe/1h par)
    Returns: dict of pandas Series (sab df ke index ke sath aligned)
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    n = len(df)
    close = df["close"]
    high = df["high"].values
    low = df["low"].values

    emaFast = ema(close, p["ema_fast_len"])
    emaSlow = ema(close, p["ema_slow_len"])
    rsi = compute_rsi(df, p["rsi_len"])
    macdHist = compute_macd_hist(close, 12, 26, 9)

    atr = compute_atr_simple(df, p["atr_len"])
    atr_vals = atr.values

    volAvg = df["volume"].rolling(p["vol_len"]).mean()
    volRatio = (df["volume"] / volAvg.replace(0, np.nan)).fillna(0.0)
    volumeSpike = volRatio >= p["vol_spike_ratio"]

    struct = compute_structure(high, low, close.values, p["pivot_len"])

    bullFVG, bearFVG = compute_fvg(df, atr_vals, p["fvg_min_atr"])

    lastPH = struct["lastPH"]
    lastPL = struct["lastPL"]
    rangeValid = (~np.isnan(lastPH)) & (~np.isnan(lastPL)) & (lastPH > lastPL)
    rangeMid = np.where(rangeValid, (lastPH + lastPL) / 2, np.nan)
    inDiscount = rangeValid & (close.values < rangeMid)
    inPremium = rangeValid & (close.values > rangeMid)

    # ---- Multi-Timeframe ----
    bull_counts = np.zeros(n)
    bear_counts = np.zeros(n)
    for key in ["tf1", "tf2", "tf3", "tf4"]:
        htf_df = htf_dfs.get(key)
        if htf_df is None or len(htf_df) < 30:
            continue
        hb = htf_bull_bear(htf_df, p["ema_fast_len"], p["ema_slow_len"])
        bull_arr, bear_arr = align_htf_to_base(df["timestamp"], hb)
        bull_counts += bull_arr.astype(int)
        bear_counts += bear_arr.astype(int)

    # ---- BTC correlation / relative strength ----
    btc_close_aligned = align_btc_close(df["timestamp"], btc_df)
    btc_close_series = pd.Series(btc_close_aligned, index=df.index)

    coinRet = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    btcRet = np.log(btc_close_series / btc_close_series.shift(1)).replace([np.inf, -np.inf], np.nan)
    btcCorr = coinRet.rolling(p["corr_len"]).corr(btcRet)

    coinROC = compute_roc(close, p["rs_len"])
    btcROC = compute_roc(btc_close_series, p["rs_len"])
    relativeStrength = coinROC - btcROC

    bullAlignment = (coinROC > 0) & (btcROC > 0)
    bearAlignment = (coinROC < 0) & (btcROC < 0)
    bullDivergence = (btcROC < 0) & (coinROC > 0)
    bearDivergence = (btcROC > 0) & (coinROC < 0)

    # ---- Recent break context ----
    barsSinceBull = bars_since(struct["bullBreak"])
    barsSinceBear = bars_since(struct["bearBreak"])
    recentBullBreak = (~np.isnan(barsSinceBull)) & (barsSinceBull <= p["recent_break_bars"])
    recentBearBreak = (~np.isnan(barsSinceBear)) & (barsSinceBear <= p["recent_break_bars"])

    # ---- Scoring ----
    longMTF = bull_counts * 6.25
    shortMTF = bear_counts * 6.25

    longStructure = ((struct["structureBias"] == 1).astype(float) * 12.0
                      + struct["lastHighWasHH"].astype(float) * 4.0
                      + struct["lastLowWasHL"].astype(float) * 4.0
                      + recentBullBreak.astype(float) * 5.0)
    shortStructure = ((struct["structureBias"] == -1).astype(float) * 12.0
                       + struct["lastHighWasLH"].astype(float) * 4.0
                       + struct["lastLowWasLL"].astype(float) * 4.0
                       + recentBearBreak.astype(float) * 5.0)

    longMomentum = ((emaFast > emaSlow).astype(float) * 5.0
                     + (rsi >= 52).astype(float) * 5.0
                     + (macdHist > 0).astype(float) * 5.0)
    shortMomentum = ((emaFast < emaSlow).astype(float) * 5.0
                      + (rsi <= 48).astype(float) * 5.0
                      + (macdHist < 0).astype(float) * 5.0)

    volumePoints = ((volRatio - 0.75) / 1.25 * 10.0).clip(lower=0.0, upper=10.0)
    longVolume = volumePoints
    shortVolume = volumePoints

    longLiquidity = struct["bullSweep"].astype(float) * 6.0 + bullFVG.astype(float) * 4.0
    shortLiquidity = struct["bearSweep"].astype(float) * 6.0 + bearFVG.astype(float) * 4.0

    longCorr = (relativeStrength > 0).astype(float) * 5.0 + ((bullAlignment | bullDivergence).astype(float)) * 5.0
    shortCorr = (relativeStrength < 0).astype(float) * 5.0 + ((bearAlignment | bearDivergence).astype(float)) * 5.0

    longPD = pd.Series(inDiscount.astype(float) * 5.0, index=df.index)
    shortPD = pd.Series(inPremium.astype(float) * 5.0, index=df.index)

    longScore = (pd.Series(longMTF, index=df.index) + pd.Series(longStructure, index=df.index)
                 + longMomentum + longVolume + pd.Series(longLiquidity, index=df.index)
                 + longCorr + longPD).clip(upper=100.0)
    shortScore = (pd.Series(shortMTF, index=df.index) + pd.Series(shortStructure, index=df.index)
                  + shortMomentum + shortVolume + pd.Series(shortLiquidity, index=df.index)
                  + shortCorr + shortPD).clip(upper=100.0)

    longPenalty = (bearDivergence.astype(float) * 6.0 + (rsi > 78).astype(float) * 4.0
                   + pd.Series(struct["bearSweep"], index=df.index).astype(float) * 4.0)
    shortPenalty = (bullDivergence.astype(float) * 6.0 + (rsi < 22).astype(float) * 4.0
                    + pd.Series(struct["bullSweep"], index=df.index).astype(float) * 4.0)

    longScoreFinal = (longScore - longPenalty).clip(lower=0.0)
    shortScoreFinal = (shortScore - shortPenalty).clip(lower=0.0)

    # ---- Signal ----
    longSetupRaw = ((longScoreFinal >= p["signal_threshold"])
                     & ((longScoreFinal - shortScoreFinal) >= p["min_score_edge"])
                     & (volRatio >= p["min_vol_ratio"]))
    buySignal = longSetupRaw & (~longSetupRaw.shift(1).fillna(False))

    # ---- SL / TP plan (Hybrid default) ----
    longATRStop = close - atr * p["sl_atr_mult"]
    lastPL_series = pd.Series(lastPL, index=df.index)
    longStructureStop = pd.Series(np.where(
        (~np.isnan(lastPL)) & (lastPL < close.values),
        lastPL - atr.values * p["structure_buffer"], np.nan,
    ), index=df.index)

    if p["stop_mode"] == "ATR":
        longSL = longATRStop
    elif p["stop_mode"] == "Structure":
        longSL = longStructureStop.where(longStructureStop.notna(), longATRStop)
    else:  # Hybrid
        longSL = np.maximum(longATRStop, longStructureStop.fillna(-np.inf))
        longSL = longSL.where(longStructureStop.notna(), longATRStop)

    longRisk = (close - longSL).clip(lower=1e-9)
    longTP1 = close + longRisk * p["rr1"]
    longTP2 = close + longRisk * p["rr2"]
    longTP3 = close + longRisk * p["rr3"]

    return {
        "longScoreFinal": longScoreFinal, "shortScoreFinal": shortScoreFinal,
        "buySignal": buySignal.fillna(False),
        "longSL": longSL, "longTP1": longTP1, "longTP2": longTP2, "longTP3": longTP3,
        "structureBias": pd.Series(struct["structureBias"], index=df.index),
        "inDiscount": pd.Series(inDiscount, index=df.index),
        "bullMTFCount": pd.Series(bull_counts, index=df.index),
        "volRatio": volRatio,
    }


def build_signal_with_threshold(longScoreFinal, shortScoreFinal, volRatio, threshold, min_score_edge, min_vol_ratio):
    """SHADOW47 ke signal ko alag threshold/edge ke sath dobara banane ke
    liye - poora engine dobara chalaye bina (score pehle se ready hai)."""
    raw = ((longScoreFinal >= threshold)
           & ((longScoreFinal - shortScoreFinal) >= min_score_edge)
           & (volRatio >= min_vol_ratio))
    buy = raw & (~raw.shift(1).fillna(False))
    return buy.fillna(False)
