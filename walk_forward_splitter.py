"""
Walk-Forward Period Splitter - Core Logic
=============================================
Poore dorania ko barabar hisson (periods) mein baant kar, har trade ko
uske entry_time ke hisab se sahi period mein daal deta hai. Isse pata
chalta hai ke system har dor (period) mein kitna consistent raha -
sirf ek overall number kafi nahi, kyunke wo kisi ek acche/bure
hafte ke asar mein chhup sakta hai.
"""

import pandas as pd


def make_period_boundaries(start_ts, end_ts, n_periods):
    """
    start_ts, end_ts: pandas Timestamp
    n_periods: kitne barabar hisson mein baantna hai

    Returns list of (period_start, period_end) tuples, n_periods lambi.
    """
    total_span = end_ts - start_ts
    step = total_span / n_periods
    boundaries = []
    for i in range(n_periods):
        p_start = start_ts + step * i
        p_end = start_ts + step * (i + 1)
        boundaries.append((p_start, p_end))
    return boundaries


def assign_trades_to_periods(trades, boundaries):
    """
    trades: list of dicts, har ek mein 'entry_time' (pandas Timestamp ya
            usko convert kiya ja sakta ho) hona chahiye
    boundaries: make_period_boundaries() ka result

    Returns: list of lists - har period ke liye us mein aane wale trades
    """
    periods = [[] for _ in boundaries]
    for t in trades:
        entry_time = pd.Timestamp(t["entry_time"])
        for idx, (p_start, p_end) in enumerate(boundaries):
            if p_start <= entry_time < p_end:
                periods[idx].append(t)
                break
            elif idx == len(boundaries) - 1 and entry_time >= p_end:
                periods[idx].append(t)
    return periods