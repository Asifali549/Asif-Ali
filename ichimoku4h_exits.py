"""
4H ICHIMOKU + MS - EXIT KA MOQABLA (TP ke sath vs sirf trailing stop)
====================================================================
Entry bilkul wahi (production). Sirf exit badla:
  - Trailing stop (Chandelier) period 16 / 22 / 30 (4H candles) x multiplier 4.0 / 5.5
  - TP: koi nahi (Donchian ki tarah trend ke sath chalti rahe) ya 3R
  - Baseline: CE 16 / 4.5 + TP 2R (production)
Wahi sakht usool (ichimoku4h_validation.py ke sath shared code): ~5.5 saal 4H,
lookahead-free, fees + slippage + stop slippage, random-entry control, portfolio
(10 slots, 1% risk), Daily Donchian ke sath milaap.
Natija: ichimoku4h_exits_RESULTS.txt
"""
import ichimoku4h_validation as V

V.OUT_FILE = "ichimoku4h_exits_RESULTS.txt"
V.TITLE = "4H ICHIMOKU + MS - EXIT KA MOQABLA (TP vs sirf trailing stop)"
V.VARIANTS = [("BASELINE (CE 16/4.5 + TP 2R, production)", {})]
for tp in (None, 3.0):
    for ce_p in (16, 22, 30):
        for ce_m in (4.0, 5.5):
            V.VARIANTS.append((f"CE {ce_p}/{ce_m} + {'TP NAHI (sirf trailing)' if tp is None else 'TP 3R'}",
                               {"ce_p": ce_p, "ce_m": ce_m, "tp": tp}))

if __name__ == "__main__":
    V.main()
