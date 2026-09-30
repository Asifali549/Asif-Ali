"""
Confirmation Candle Filter - Core Logic
===========================================
Idea: Signal milte hi turant entry lene ke bajaye, ek candle aur
intezar karo (yehi "confirmation candle" hai - signal ke theek baad
wali candle). Agar wo candle bullish (close >= open) band ho, to
agli candle ke open par entry lo. Agar wo candle bearish band ho,
to poora signal chhor do (trade na lo).
"""


def passes_confirmation(o, h, l, c):
    """
    o,h,l,c: confirmation candle ke OHLC (signal ke theek baad wali candle)
    Returns True agar ye candle bullish band hui (entry lene ki ijazat),
    False agar bearish (signal reject).
    """
    return c >= o


def get_confirmed_entry_bar(signal_idx, n):
    """
    signal_idx: jahan signal bana (original system mein entry = signal_idx+1)
    n: total candles ki tadaad

    Confirmation candle = signal_idx + 1
    Agar wo pass ho jaye, entry = signal_idx + 2 ke open par

    Returns confirmation_candle_idx, entry_idx (dono None agar data na ho)
    """
    confirmation_idx = signal_idx + 1
    entry_idx = signal_idx + 2
    if confirmation_idx >= n or entry_idx >= n:
        return None, None
    return confirmation_idx, entry_idx