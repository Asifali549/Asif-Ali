"""
Strategies - Har strategy df (OHLCV) leti hai aur ek boolean 'signal' column
return karti hai jo sirf us candle par True hota hai jahan "fresh buy"
condition pehli baar true hui ho (pichli candle par false thi).

Har function ka output: pandas Series (same index as df), dtype=bool
"""

import numpy as np
import pandas as pd


def _fresh(cond: pd.Series) -> pd.Series:
    """Sirf wahi bar True rakho jahan condition newly true hui ho."""
    cond = cond.fillna("""
Strategies - Har strategy df (OHLCV) leti hai aur ek boolean 'signal' column
return karti hai jo sirf us candle par True hota hai jahan "fresh buy"
condition pehli baar true hui ho (pichli candle par false thi).

Har function ka output: pandas Series (same index as df), dtype=bool
"""

import numpy as np
import pandas as pd


def _fresh(cond: pd.Series) -> pd.Series:
    """Sirf wahi bar True rakho jahan condition newly true hui ho."""
    cond = cond.fillna(False)
    return cond & (~cond.shift(1).fillna(False))


def apply_cooldown(signal: pd.Series, cooldown_bars: int) -> pd.Series:
    """
    Ek signal ke baad agle N bars tak koi naya signal allow nahi karta
    (spam/clutter rokne ke liye). Har coin/strategy par independently chalta hai.
    """
    if cooldown_bars <= 0:
        return signal
    sig = signal.values.copy()
    last_signal_idx = -cooldown_bars - 1
    for i in range(len(sig)):
        if sig[i]:
            if i - last_signal_idx <= cooldown_bars:
                sig[i] = False
            else:
                last_signal_idx = i
    return pd.Series(sig, index=signal.index)


def _golden_cross_ok(df, params):
    """EMA(golden_cross_filter) > EMA(trend_filter) -> extra tight trend confirmation."""
    if "golden_cross_filter" not in params:
        return pd.Series(True, index=df.index)
    fast_trend = df["close"].ewm(span=params["golden_cross_filter"], adjust=False).mean()
    slow_trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    return fast_trend > slow_trend


# ------------------------------------------------------------------
# 1. EMA CROSSOVER
# ------------------------------------------------------------------
def ema_crossover(df, params):
    fast = df["close"].ewm(span=params["fast"], adjust=False).mean()
    slow = df["close"].ewm(span=params["slow"], adjust=False).mean()
    trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    avg_vol = df["volume"].rolling(params["volume_avg_period"]).mean()

    cross_up = (fast > slow)
    above_trend = df["close"] > trend
    volume_ok = df["volume"] > avg_vol * params["volume_mult"]
    golden = _golden_cross_ok(df, params)

    cond = cross_up & above_trend & volume_ok & golden
    return _fresh(cond)


# ------------------------------------------------------------------
# 2. BREAKOUT (price + volume confirmation + 200 EMA trend filter)
# ------------------------------------------------------------------
def breakout(df, params):
    lookback = params["lookback"]
    prior_high = df["high"].shift(1).rolling(lookback).max()
    avg_vol = df["volume"].rolling(params["volume_avg_period"]).mean()
    trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    golden = _golden_cross_ok(df, params)

    cond = (
        (df["close"] > prior_high)
        & (df["volume"] > avg_vol * params["volume_mult"])
        & (df["close"] > trend)
        & golden
    )
    return _fresh(cond)


# ------------------------------------------------------------------
# 3. MOMENTUM + VOLUME UP (+ 200 EMA trend filter)
# ------------------------------------------------------------------
def momentum_volume(df, params):
    roc = df["close"].pct_change(params["roc_period"]) * 100
    avg_vol = df["volume"].rolling(params["volume_avg_period"]).mean()
    trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    golden = _golden_cross_ok(df, params)

    cond = (
        (roc > params["roc_threshold"])
        & (df["volume"] > avg_vol * params["volume_mult"])
        & (df["close"] > trend)
        & golden
    )
    return _fresh(cond)


# ------------------------------------------------------------------
# 4. ICHIMOKU CLOUD (fresh bullish + 200 EMA trend filter + volume)
# ------------------------------------------------------------------
def ichimoku(df, params):
    high, low, close = df["high"], df["low"], df["close"]

    tenkan = (high.rolling(params["tenkan"]).max() + low.rolling(params["tenkan"]).min()) / 2
    kijun = (high.rolling(params["kijun"]).max() + low.rolling(params["kijun"]).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(params["displacement"])
    senkou_b = ((high.rolling(params["senkou_b"]).max() + low.rolling(params["senkou_b"]).min()) / 2).shift(params["displacement"])

    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    trend = close.ewm(span=params["trend_filter"], adjust=False).mean()
    avg_vol = df["volume"].rolling(params["volume_avg_period"]).mean()
    golden = _golden_cross_ok(df, params)

    tk_cross_up = tenkan > kijun
    above_cloud = close > cloud_top
    above_trend = close > trend
    volume_ok = df["volume"] > avg_vol * params["volume_mult"]

    cond = tk_cross_up & above_cloud & above_trend & volume_ok & golden
    return _fresh(cond)


# ------------------------------------------------------------------
# 5. MARKET STRUCTURE (HH/HL + Break of Structure)
# ------------------------------------------------------------------
def market_structure(df, params):
    n = params["pivot_lookback"]
    min_swing = params["min_swing_pct"] / 100

    high, low, close = df["high"].values, df["low"].values, df["close"].values
    length = len(df)

    is_pivot_high = np.zeros(length, dtype=bool)
    is_pivot_low = np.zeros(length, dtype=bool)

    for i in range(n, length - n):
        window_h = high[i - n:i + n + 1]
        window_l = low[i - n:i + n + 1]
        if high[i] == window_h.max():
            is_pivot_high[i] = True
        if low[i] == window_l.min():
            is_pivot_low[i] = True

    signal = np.zeros(length, dtype=bool)

    last_pivot_high = None
    last_pivot_high_val = None
    prev_pivot_high_val = None
    structure_bullish = False

    for i in range(length):
        if is_pivot_high[i]:
            val = high[i]
            if prev_pivot_high_val is not None:
                swing_pct = abs(val - prev_pivot_high_val) / prev_pivot_high_val
                if swing_pct >= min_swing:
                    structure_bullish = val > prev_pivot_high_val
            prev_pivot_high_val = last_pivot_high_val
            last_pivot_high_val = val
            last_pivot_high = i

        # Break of Structure: close crosses above the last confirmed pivot high
        if last_pivot_high_val is not None and structure_bullish:
            if close[i] > last_pivot_high_val and (i == 0 or close[i - 1] <= last_pivot_high_val):
                signal[i] = True

    signal_series = pd.Series(signal, index=df.index)

    # NEW: trend filter + golden cross (extra tight, structure ke sath double confirmation)
    trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    above_trend = df["close"] > trend
    golden = _golden_cross_ok(df, params)

    return signal_series & above_trend & golden


# ------------------------------------------------------------------
# 6. VOLUME PROFILE (POC / Value Area bounce or breakout)
# ------------------------------------------------------------------
def volume_profile(df, params):
    lookback = params["lookback_period"]
    num_bins = params["num_bins"]
    va_pct = params["value_area_pct"]
    tol = params["bounce_tolerance_pct"] / 100

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    length = len(df)

    signal = np.zeros(length, dtype=bool)
    prev_cond = False

    for i in range(lookback, length):
        window_high = high[i - lookback:i]
        window_low = low[i - lookback:i]
        window_vol = volume[i - lookback:i]
        window_close = close[i - lookback:i]

        price_min, price_max = window_low.min(), window_high.max()
        if price_max <= price_min:
            continue

        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_vol = np.zeros(num_bins)
        bin_idx = np.clip(np.digitize(window_close, bins) - 1, 0, num_bins - 1)
        for b, v in zip(bin_idx, window_vol):
            bin_vol[b] += v

        poc_bin = bin_vol.argmax()
        poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2

        # Value area: POC se shuru kar ke va_pct volume tak bins add karo
        total_vol = bin_vol.sum()
        target_vol = total_vol * va_pct
        included = {poc_bin}
        acc_vol = bin_vol[poc_bin]
        lo, hi = poc_bin, poc_bin
        while acc_vol < target_vol and (lo > 0 or hi < num_bins - 1):
            left_vol = bin_vol[lo - 1] if lo > 0 else -1
            right_vol = bin_vol[hi + 1] if hi < num_bins - 1 else -1
            if right_vol >= left_vol:
                hi += 1
                acc_vol += bin_vol[hi]
                included.add(hi)
            else:
                lo -= 1
                acc_vol += bin_vol[lo]
                included.add(lo)

        val_price = bins[lo]      # Value Area Low
        vah_price = bins[hi + 1]  # Value Area High

        current_price = close[i]

        near_val = abs(current_price - val_price) / val_price <= tol
        bounced_up = current_price > low[i] and close[i] > df["open"].values[i]  # bullish candle
        poc_breakout = (close[i - 1] <= poc_price) and (close[i] > poc_price)

        cond = (near_val and bounced_up) or poc_breakout

        signal[i] = cond and not prev_cond
        prev_cond = cond

    signal_series = pd.Series(signal, index=df.index)

    # NEW: 200 EMA trend filter + golden cross + volume confirmation (tight)
    trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    avg_vol = df["volume"].rolling(params["volume_avg_period"]).mean()
    above_trend = df["close"] > trend
    volume_ok = df["volume"] > avg_vol * params["volume_mult"]
    golden = _golden_cross_ok(df, params)

    return signal_series & above_trend & volume_ok & golden


# ------------------------------------------------------------------
# 7. FAIR VALUE GAP (FVG) - Retracement Entry
# ------------------------------------------------------------------
def fvg(df, params):
    """
    Bullish FVG: 3-candle pattern jahan candle[i-2] ki high, candle[i] ki
    low se kam ho (beech mein ek "khaali" price zone chhoot jata hai jahan
    trading nahi hui).

    Entry logic (RETRACEMENT): FVG bante hi buy NAHI karte. Zone ko "active/
    unfilled" track karte hain. Jab baad mein price wapis is zone ke andar
    aaye AUR us bar ki candle bullish ho (close > open) -> tabhi fresh buy.
    Agar price zone ke neeche band ho jaye (fully fill + break) to zone ko
    invalid maan kar hata dete hain.
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_ = df["open"].values
    volume = df["volume"].values
    length = len(df)

    min_gap_pct = params["min_gap_pct"] / 100
    max_age = params["max_zone_age_bars"]
    min_body_pct = params.get("min_candle_body_pct", 0) / 100
    avg_vol = df["volume"].rolling(params.get("volume_avg_period", 20)).mean().values
    vol_mult = params.get("volume_mult", 0)

    signal = np.zeros(length, dtype=bool)

    # Active unfilled bullish FVG zones: list of dicts {top, bottom, created_at}
    active_zones = []

    for i in range(2, length):
        # ---- Naya FVG detect karo (candle i-2, i-1, i se banta hai) ----
        zone_bottom = high[i - 2]
        zone_top = low[i]
        if zone_top > zone_bottom:
            gap_pct = (zone_top - zone_bottom) / zone_bottom

            # NEW: beech wali (impulse) candle[i-1] ka body bara/asli hona chahiye
            impulse_body = abs(close[i - 1] - open_[i - 1])
            impulse_body_pct = impulse_body / open_[i - 1] if open_[i - 1] > 0 else 0

            if gap_pct >= min_gap_pct and impulse_body_pct >= min_body_pct:
                active_zones.append({"top": zone_top, "bottom": zone_bottom, "created_at": i})

        # ---- Purani/expired zones nikal do ----
        active_zones = [z for z in active_zones if (i - z["created_at"]) <= max_age]

        # ---- Har active zone check karo: price retrace hui aur bounce hui? ----
        remaining_zones = []
        touched_this_bar = False
        for z in active_zones:
            price_in_zone = low[i] <= z["top"] and high[i] >= z["bottom"]
            fully_broken = close[i] < z["bottom"]

            if fully_broken:
                continue  # zone invalid ho gayi, hata do

            # NEW: bounce candle par volume confirmation bhi chahiye
            vol_ok = True
            if vol_mult > 0 and not np.isnan(avg_vol[i]):
                vol_ok = volume[i] > avg_vol[i] * vol_mult

            if price_in_zone and close[i] > open_[i] and vol_ok and not touched_this_bar:
                signal[i] = True
                touched_this_bar = True  # ek bar mein sirf ek signal
                continue  # is zone ko "consumed" maan kar hata dete hain

            remaining_zones.append(z)

        active_zones = remaining_zones

    signal_series = pd.Series(signal, index=df.index)

    # Trend filter + golden cross (sirf uptrend mein FVG bounce par buy)
    trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    above_trend = df["close"] > trend
    golden = _golden_cross_ok(df, params)

    return signal_series & above_trend & golden


# ------------------------------------------------------------------
STRATEGY_FUNCTIONS = {
    "ema_crossover": ema_crossover,
    "breakout": breakout,
    "momentum_volume": momentum_volume,
    "ichimoku": ichimoku,
    "market_structure": market_structure,
    "volume_profile": volume_profile,
    "fvg": fvg,
})
    return cond & (~cond.shift(1).fillna(False))


def apply_cooldown(signal: pd.Series, cooldown_bars: int) -> pd.Series:
    """
    Ek signal ke baad agle N bars tak koi naya signal allow nahi karta
    (spam/clutter rokne ke liye). Har coin/strategy par independently chalta hai.
    """
    if cooldown_bars <= 0:
        return signal
    sig = signal.values.copy()
    last_signal_idx = -cooldown_bars - 1
    for i in range(len(sig)):
        if sig[i]:
            if i - last_signal_idx <= cooldown_bars:
                sig[i] = False
            else:
                last_signal_idx = i
    return pd.Series(sig, index=signal.index)


def _golden_cross_ok(df, params):
    """EMA(golden_cross_filter) > EMA(trend_filter) -> extra tight trend confirmation."""
    if "golden_cross_filter" not in params:
        return pd.Series(True, index=df.index)
    fast_trend = df["close"].ewm(span=params["golden_cross_filter"], adjust=False).mean()
    slow_trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    return fast_trend > slow_trend


# ------------------------------------------------------------------
# 1. EMA CROSSOVER
# ------------------------------------------------------------------
def ema_crossover(df, params):
    fast = df["close"].ewm(span=params["fast"], adjust=False).mean()
    slow = df["close"].ewm(span=params["slow"], adjust=False).mean()
    trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    avg_vol = df["volume"].rolling(params["volume_avg_period"]).mean()

    cross_up = (fast > slow)
    above_trend = df["close"] > trend
    volume_ok = df["volume"] > avg_vol * params["volume_mult"]
    golden = _golden_cross_ok(df, params)

    cond = cross_up & above_trend & volume_ok & golden
    return _fresh(cond)


# ------------------------------------------------------------------
# 2. BREAKOUT (price + volume confirmation + 200 EMA trend filter)
# ------------------------------------------------------------------
def breakout(df, params):
    lookback = params["lookback"]
    prior_high = df["high"].shift(1).rolling(lookback).max()
    avg_vol = df["volume"].rolling(params["volume_avg_period"]).mean()
    trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    golden = _golden_cross_ok(df, params)

    cond = (
        (df["close"] > prior_high)
        & (df["volume"] > avg_vol * params["volume_mult"])
        & (df["close"] > trend)
        & golden
    )
    return _fresh(cond)


# ------------------------------------------------------------------
# 3. MOMENTUM + VOLUME UP (+ 200 EMA trend filter)
# ------------------------------------------------------------------
def momentum_volume(df, params):
    roc = df["close"].pct_change(params["roc_period"]) * 100
    avg_vol = df["volume"].rolling(params["volume_avg_period"]).mean()
    trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    golden = _golden_cross_ok(df, params)

    cond = (
        (roc > params["roc_threshold"])
        & (df["volume"] > avg_vol * params["volume_mult"])
        & (df["close"] > trend)
        & golden
    )
    return _fresh(cond)


# ------------------------------------------------------------------
# 4. ICHIMOKU CLOUD (fresh bullish + 200 EMA trend filter + volume)
# ------------------------------------------------------------------
def ichimoku(df, params):
    high, low, close = df["high"], df["low"], df["close"]

    tenkan = (high.rolling(params["tenkan"]).max() + low.rolling(params["tenkan"]).min()) / 2
    kijun = (high.rolling(params["kijun"]).max() + low.rolling(params["kijun"]).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(params["displacement"])
    senkou_b = ((high.rolling(params["senkou_b"]).max() + low.rolling(params["senkou_b"]).min()) / 2).shift(params["displacement"])

    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    trend = close.ewm(span=params["trend_filter"], adjust=False).mean()
    avg_vol = df["volume"].rolling(params["volume_avg_period"]).mean()
    golden = _golden_cross_ok(df, params)

    tk_cross_up = tenkan > kijun
    above_cloud = close > cloud_top
    above_trend = close > trend
    volume_ok = df["volume"] > avg_vol * params["volume_mult"]

    cond = tk_cross_up & above_cloud & above_trend & volume_ok & golden
    return _fresh(cond)


# ------------------------------------------------------------------
# 5. MARKET STRUCTURE (HH/HL + Break of Structure)
# ------------------------------------------------------------------
def market_structure(df, params):
    n = params["pivot_lookback"]
    min_swing = params["min_swing_pct"] / 100

    high, low, close = df["high"].values, df["low"].values, df["close"].values
    length = len(df)

    is_pivot_high = np.zeros(length, dtype=bool)
    is_pivot_low = np.zeros(length, dtype=bool)

    for i in range(n, length - n):
        window_h = high[i - n:i + n + 1]
        window_l = low[i - n:i + n + 1]
        if high[i] == window_h.max():
            is_pivot_high[i] = True
        if low[i] == window_l.min():
            is_pivot_low[i] = True

    signal = np.zeros(length, dtype=bool)

    last_pivot_high = None
    last_pivot_high_val = None
    prev_pivot_high_val = None
    structure_bullish = False

    for i in range(length):
        if is_pivot_high[i]:
            val = high[i]
            if prev_pivot_high_val is not None:
                swing_pct = abs(val - prev_pivot_high_val) / prev_pivot_high_val
                if swing_pct >= min_swing:
                    structure_bullish = val > prev_pivot_high_val
            prev_pivot_high_val = last_pivot_high_val
            last_pivot_high_val = val
            last_pivot_high = i

        # Break of Structure: close crosses above the last confirmed pivot high
        if last_pivot_high_val is not None and structure_bullish:
            if close[i] > last_pivot_high_val and (i == 0 or close[i - 1] <= last_pivot_high_val):
                signal[i] = True

    signal_series = pd.Series(signal, index=df.index)

    # NEW: trend filter + golden cross (extra tight, structure ke sath double confirmation)
    trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    above_trend = df["close"] > trend
    golden = _golden_cross_ok(df, params)

    return signal_series & above_trend & golden


# ------------------------------------------------------------------
# 6. VOLUME PROFILE (POC / Value Area bounce or breakout)
# ------------------------------------------------------------------
def volume_profile(df, params):
    lookback = params["lookback_period"]
    num_bins = params["num_bins"]
    va_pct = params["value_area_pct"]
    tol = params["bounce_tolerance_pct"] / 100

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    length = len(df)

    signal = np.zeros(length, dtype=bool)
    prev_cond = False

    for i in range(lookback, length):
        window_high = high[i - lookback:i]
        window_low = low[i - lookback:i]
        window_vol = volume[i - lookback:i]
        window_close = close[i - lookback:i]

        price_min, price_max = window_low.min(), window_high.max()
        if price_max <= price_min:
            continue

        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_vol = np.zeros(num_bins)
        bin_idx = np.clip(np.digitize(window_close, bins) - 1, 0, num_bins - 1)
        for b, v in zip(bin_idx, window_vol):
            bin_vol[b] += v

        poc_bin = bin_vol.argmax()
        poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2

        # Value area: POC se shuru kar ke va_pct volume tak bins add karo
        total_vol = bin_vol.sum()
        target_vol = total_vol * va_pct
        included = {poc_bin}
        acc_vol = bin_vol[poc_bin]
        lo, hi = poc_bin, poc_bin
        while acc_vol < target_vol and (lo > 0 or hi < num_bins - 1):
            left_vol = bin_vol[lo - 1] if lo > 0 else -1
            right_vol = bin_vol[hi + 1] if hi < num_bins - 1 else -1
            if right_vol >= left_vol:
                hi += 1
                acc_vol += bin_vol[hi]
                included.add(hi)
            else:
                lo -= 1
                acc_vol += bin_vol[lo]
                included.add(lo)

        val_price = bins[lo]      # Value Area Low
        vah_price = bins[hi + 1]  # Value Area High

        current_price = close[i]

        near_val = abs(current_price - val_price) / val_price <= tol
        bounced_up = current_price > low[i] and close[i] > df["open"].values[i]  # bullish candle
        poc_breakout = (close[i - 1] <= poc_price) and (close[i] > poc_price)

        cond = (near_val and bounced_up) or poc_breakout

        signal[i] = cond and not prev_cond
        prev_cond = cond

    signal_series = pd.Series(signal, index=df.index)

    # NEW: 200 EMA trend filter + golden cross + volume confirmation (tight)
    trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    avg_vol = df["volume"].rolling(params["volume_avg_period"]).mean()
    above_trend = df["close"] > trend
    volume_ok = df["volume"] > avg_vol * params["volume_mult"]
    golden = _golden_cross_ok(df, params)

    return signal_series & above_trend & volume_ok & golden


# ------------------------------------------------------------------
# 7. FAIR VALUE GAP (FVG) - Retracement Entry
# ------------------------------------------------------------------
def fvg(df, params):
    """
    Bullish FVG: 3-candle pattern jahan candle[i-2] ki high, candle[i] ki
    low se kam ho (beech mein ek "khaali" price zone chhoot jata hai jahan
    trading nahi hui).

    Entry logic (RETRACEMENT): FVG bante hi buy NAHI karte. Zone ko "active/
    unfilled" track karte hain. Jab baad mein price wapis is zone ke andar
    aaye AUR us bar ki candle bullish ho (close > open) -> tabhi fresh buy.
    Agar price zone ke neeche band ho jaye (fully fill + break) to zone ko
    invalid maan kar hata dete hain.
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_ = df["open"].values
    volume = df["volume"].values
    length = len(df)

    min_gap_pct = params["min_gap_pct"] / 100
    max_age = params["max_zone_age_bars"]
    min_body_pct = params.get("min_candle_body_pct", 0) / 100
    avg_vol = df["volume"].rolling(params.get("volume_avg_period", 20)).mean().values
    vol_mult = params.get("volume_mult", 0)

    signal = np.zeros(length, dtype=bool)

    # Active unfilled bullish FVG zones: list of dicts {top, bottom, created_at}
    active_zones = []

    for i in range(2, length):
        # ---- Naya FVG detect karo (candle i-2, i-1, i se banta hai) ----
        zone_bottom = high[i - 2]
        zone_top = low[i]
        if zone_top > zone_bottom:
            gap_pct = (zone_top - zone_bottom) / zone_bottom

            # NEW: beech wali (impulse) candle[i-1] ka body bara/asli hona chahiye
            impulse_body = abs(close[i - 1] - open_[i - 1])
            impulse_body_pct = impulse_body / open_[i - 1] if open_[i - 1] > 0 else 0

            if gap_pct >= min_gap_pct and impulse_body_pct >= min_body_pct:
                active_zones.append({"top": zone_top, "bottom": zone_bottom, "created_at": i})

        # ---- Purani/expired zones nikal do ----
        active_zones = [z for z in active_zones if (i - z["created_at"]) <= max_age]

        # ---- Har active zone check karo: price retrace hui aur bounce hui? ----
        remaining_zones = []
        touched_this_bar = False
        for z in active_zones:
            price_in_zone = low[i] <= z["top"] and high[i] >= z["bottom"]
            fully_broken = close[i] < z["bottom"]

            if fully_broken:
                continue  # zone invalid ho gayi, hata do

            # NEW: bounce candle par volume confirmation bhi chahiye
            vol_ok = True
            if vol_mult > 0 and not np.isnan(avg_vol[i]):
                vol_ok = volume[i] > avg_vol[i] * vol_mult

            if price_in_zone and close[i] > open_[i] and vol_ok and not touched_this_bar:
                signal[i] = True
                touched_this_bar = True  # ek bar mein sirf ek signal
                continue  # is zone ko "consumed" maan kar hata dete hain

            remaining_zones.append(z)

        active_zones = remaining_zones

    signal_series = pd.Series(signal, index=df.index)

    # Trend filter + golden cross (sirf uptrend mein FVG bounce par buy)
    trend = df["close"].ewm(span=params["trend_filter"], adjust=False).mean()
    above_trend = df["close"] > trend
    golden = _golden_cross_ok(df, params)

    return signal_series & above_trend & golden


# ------------------------------------------------------------------
STRATEGY_FUNCTIONS = {
    "ema_crossover": ema_crossover,
    "breakout": breakout,
    "momentum_volume": momentum_volume,
    "ichimoku": ichimoku,
    "market_structure": market_structure,
    "volume_profile": volume_profile,
    "fvg": fvg,
}
