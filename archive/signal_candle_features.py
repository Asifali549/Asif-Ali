"""
Signal Candle Feature Extractor - Core Logic
================================================
Har signal ke ird gird candles ki khasoosiyat (features) nikalta hai:
  - Signal candle khud kaisi thi (body size, wicks, volume)
  - Entry ke foran baad wali candle(s) kaisi nikleen (bullish/bearish,
    kitni badi, ATR ke muqable)

In features ko baad mein "acha result" (TP/2%+ gain) aur "fauri nuksan"
(entry ke turant baad bara red candle -> quick SL) groups ke darmiyan
compare kiya jayega.
"""


def candle_body_pct(o, h, l, c):
    """Candle ke total range mein body ka % hissa."""
    total_range = h - l
    if total_range <= 0:
        return 0.0
    return abs(c - o) / total_range * 100


def candle_upper_wick_pct(o, h, l, c):
    total_range = h - l
    if total_range <= 0:
        return 0.0
    top = max(o, c)
    return (h - top) / total_range * 100


def candle_lower_wick_pct(o, h, l, c):
    total_range = h - l
    if total_range <= 0:
        return 0.0
    bottom = min(o, c)
    return (bottom - l) / total_range * 100


def is_bearish(o, h, l, c):
    return c < o


def candle_move_pct(o, c):
    """(close - open) / open * 100 - musbat ya manfi move."""
    if o == 0:
        return 0.0
    return (c - o) / o * 100


def extract_signal_features(df_values, signal_idx, entry_idx, atr_at_signal, lookahead=2):
    """
    df_values: dict with numpy arrays "open","high","low","close","volume"
    signal_idx: signal candle ka index
    entry_idx: entry candle ka index (usually signal_idx + 1)
    atr_at_signal: ATR value us waqt (normalize karne ke liye)
    lookahead: entry ke baad kitni candles ka data lena hai

    Returns dict of features.
    """
    o, h, l, c, v = (df_values["open"], df_values["high"], df_values["low"],
                      df_values["close"], df_values["volume"])

    sig_o, sig_h, sig_l, sig_c = o[signal_idx], h[signal_idx], l[signal_idx], c[signal_idx]

    # signal candle ke pichle 20 candles ka average volume (khud ko chhor kar)
    vol_window = v[max(0, signal_idx - 20):signal_idx]
    avg_vol = vol_window.mean() if len(vol_window) > 0 else v[signal_idx]
    vol_ratio = v[signal_idx] / avg_vol if avg_vol > 0 else 1.0

    features = {
        "signal_body_pct": candle_body_pct(sig_o, sig_h, sig_l, sig_c),
        "signal_upper_wick_pct": candle_upper_wick_pct(sig_o, sig_h, sig_l, sig_c),
        "signal_volume_ratio": round(vol_ratio, 3),
    }

    # Entry ke baad wali candles (foran baad wala reaction)
    n = len(o)
    biggest_red_pct = 0.0  # sab se badi (ATR ke muqable) red candle ka size, in-lookahead candles mein
    first_candle_bearish = False

    for k in range(lookahead):
        idx = entry_idx + k
        if idx >= n:
            break
        move_pct = candle_move_pct(o[idx], c[idx])
        if atr_at_signal and atr_at_signal > 0:
            move_atr_units = (c[idx] - o[idx]) / atr_at_signal
        else:
            move_atr_units = None

        if k == 0:
            first_candle_bearish = is_bearish(o[idx], h[idx], l[idx], c[idx])
            features["next_candle_move_pct"] = round(move_pct, 3)
            features["next_candle_move_atr"] = round(move_atr_units, 3) if move_atr_units is not None else None
            features["next_candle_bearish"] = first_candle_bearish

        if move_pct < biggest_red_pct:
            biggest_red_pct = move_pct

    features["worst_move_pct_in_lookahead"] = round(biggest_red_pct, 3)

    return features