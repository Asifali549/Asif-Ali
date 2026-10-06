"""
W52 BOT - SIRF PAPER (2026-10-06 se): sal ki chouti ke qareeb (W52 Lab 10, "promising, na-sabit")
================================================================================================
Entry: coin ka daily close pichle 365 din ke sab se oonche high se 5% ke andar PEHLI dafa band ho; coin close > EMA200,
       top-100 liquid, BTC close > EMA50. Agle din open par.
Exit : 5 din baad us din ke close par (2026-10-06 se; pehle 10 - W52 Combo Test: 5 din wohi nafa, behtar Sharpe). Koi SL / TP NAHI (backtest mein SL/TP lagane se nateeja bigra).
Size : har trade khate ka 10%, max 10.
Backtest 5 din (poora 2020-26): jeet 5.5/10, PF 2.77, ausat +7.1% / trade, CAGR 66%, DD -21.6%; taala (aakhri 12 mahine, 37 trades) PF 1.94 magar
random se saaf behtar sabit nahi -> asal imtihan yahi live paper hai. 2025-26 kamzor; survivorship khatra.
"""
import book_bot as B

B.CFG.update(name="W52 (sal ki chouti)", kind="w52", emoji="🏔️", w52_p=0.05, stop_atr=None, tp=None, hold=5,
             sma_exit=False, pos_pct=0.10, max_pos=10, history_days=700,
             state="w52_paper_state.json", trades="w52_paper_trades.csv", signals="w52_signals.json",
             howto="Becho: khareed ke 5 din baad us din ke close par (koi SL/TP nahi - sirf paper par parkh rahe hain).")

if __name__ == "__main__":
    B.main()
