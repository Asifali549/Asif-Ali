"""
W52 + CHALTA SL BOT - SIRF PAPER (2026-10-06 se), W52 bot ke SATH muqable ke liye (user: "chalta SL bhi sath test ke liye")
=====================================================================================================================
Entry: W52 jaisa (close pichle 365 din ke high se 5% ke andar PEHLI dafa; close > EMA200; top-100; BTC > EMA50). Agle din open.
Exit : (1) chalta SL: din ka high entry se 5% ooper pohnche -> SL = entry x (1 + chouti nafa - 2%), sirf ooper, AGLE din se lagu
       (misaal: chouti +10% -> SL +8%); shuru mein koi SL nahi.  (2) warna 5 din baad us din ke close par.
Size : har trade khate ka 10%, max 10.
Backtest (W52 Combo Test, 2020-26): jeet 6.2/10, PF 2.30, ausat +3.7%/trade, CAGR 42%, MaxDD -19.6%, Sharpe 1.92
(sada 5 din: jeet 5.5/10, PF 2.77, CAGR 66%, DD -21.6%, Sharpe 1.88).
"""
import book_bot as B

B.CFG.update(name="W52 + chalta SL", kind="w52", emoji="🧗", w52_p=0.05, stop_atr=None, tp=None, hold=5,
             trail_act=0.05, trail_gap=0.02,
             sma_exit=False, pos_pct=0.10, max_pos=10, history_days=700,
             state="w52trail_paper_state.json", trades="w52trail_paper_trades.csv", signals="w52trail_signals.json",
             howto="Becho: jab coin 5% ooper chala jaye to SL chouti se 2% neeche chalta rahe; warna 5 din baad close par.")

if __name__ == "__main__":
    B.main()
