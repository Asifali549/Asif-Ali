"""
DIP+ BOT - SIRF PAPER (2026-10-06 se): Dip v2 + Residual Dip EK khate mein (Combo Lab 6 ka behtareen)
====================================================================================================
Entry (har din daily candle band hone ke baad, BTC close > EMA50, coin top-100 liquid, close > EMA200):
  DIP   : EMA50 > EMA200 aur RSI(3) < 7
  RESID : coin ki 3-din "apni" girawat (BTC ka hissa nikaal kar) apne aam utaar-charhao ka 2 guna (z < -2)
Exit : TP +5% | close > SMA3 -> agle din open | SL signal close - 3 ATR (fixed) | 10 din baad close
Size : har trade khate ka 20%, max 10.
Backtest (6 saal, 148 coins): jeet 7/10, PF 2.30, CAGR ~37-40%, MaxDD ~-14.5%, Sharpe ~1.3; nafa 2024 mein ziada.
"""
import book_bot as B

B.CFG.update(name="Dip+ (Dip + Resid)", kind="dipplus", emoji="🪂", stop_atr=3.0, tp=0.05, hold=10, sma_exit=True,
             pos_pct=0.20, max_pos=10, history_days=420,
             state="dipplus_paper_state.json", trades="dipplus_paper_trades.csv", signals="dipplus_signals.json",
             howto="Becho: +5% par (limit order), ya jis din close 3-din average se ooper band ho to agle din open par "
                   "(max 10 din). SL order zaroor lagayein.")

if __name__ == "__main__":
    B.main()
