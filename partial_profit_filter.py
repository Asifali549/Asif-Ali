"""
Partial Profit Filter - Core Logic
======================================
Idea: Poori position ek sath rakhne ke bajaye:
  1. Jab price +PARTIAL_TARGET_PCT tak pahunch jaye, AADHI position
     wahan band kar do (munafa lock)
  2. Baqi AADHI position ka stop turant BREAKEVEN (entry price) par
     le aao, phir wahan se trailing (chandelier) jaisa pehle chalta
     rahega - sirf ab neeche breakeven se niche nahi ja sakta

Isse:
  - Win rate badhni chahiye (chhote winners bhi ab "win" ban jayenge)
  - Bade winners ka thoda hissa chhoot sakta hai (aadhi position jaldi
    nikal jaati hai)
  - Losers ka nuksan kam hona chahiye (jo trade pehle "loss" tha, agar
    wo partial target tak pahunch kar wapas aaya, to ab breakeven+chhota
    profit milega, poora loss nahi)
"""


def compute_partial_target(entry_price, partial_target_pct):
    """Entry se partial_target_pct % upar ka price level."""
    return entry_price * (1 + partial_target_pct / 100)


def simulate_partial_profit_trade(o, h, l, c, entry_bar, entry_price,
                                    initial_stop, ce_stops, max_hold,
                                    partial_target_pct, fee, slip):
    """
    Ek trade ko bar-by-bar simulate karta hai partial-profit logic ke sath.
    """
    n = len(o)
    partial_target = compute_partial_target(entry_price, partial_target_pct)

    trail_stop = initial_stop
    partial_taken = False
    leg1_return = None
    final_exit_price = None
    final_exit_bar = None
    final_exit_reason = None

    for j in range(entry_bar, min(entry_bar + max_hold, n)):
        if not (ce_stops[j] != ce_stops[j]):  # not NaN
            trail_stop = max(trail_stop, ce_stops[j])

        if l[j] <= trail_stop:
            final_exit_price = trail_stop * (1 - slip)
            final_exit_bar = j
            final_exit_reason = "CE_STOP" if partial_taken else "CE_STOP_FULL"
            break

        if not partial_taken and h[j] >= partial_target:
            partial_taken = True
            leg1_exit_price = partial_target * (1 - slip)
            leg1_return = (leg1_exit_price - entry_price) / entry_price
            trail_stop = max(trail_stop, entry_price)

    if final_exit_price is None:
        last_bar = min(entry_bar + max_hold - 1, n - 1)
        final_exit_price = c[last_bar] * (1 - slip)
        final_exit_bar = last_bar
        final_exit_reason = "TIME"

    leg2_return = (final_exit_price - entry_price) / entry_price

    if partial_taken:
        blended_gross = 0.5 * leg1_return + 0.5 * leg2_return
    else:
        blended_gross = leg2_return

    blended_net_pct = (blended_gross - 2 * fee) * 100

    return {
        "return_pct": blended_net_pct,
        "exit_reason": final_exit_reason,
        "partial_taken": partial_taken,
        "bars_held": final_exit_bar - entry_bar,
    }