"""
STREAK BOT (4 din ki girawat) - SIRF PAPER (2026-10-06 se) - Search Lab 12 / 12b
==============================================================================
Entry: coin ka daily close 4 din LAGATAAR pichle din se neeche (sirf 4th din - 5th par dobara nahi); coin close > EMA200
       aur EMA50 > EMA200 (uptrend); top-100 liquid; BTC close > EMA50. Agle din open par.
Exit : Dip jaisa - TP +5% | close 3-din average se ooper -> agle din open | SL signal close - 3 ATR | 10 din.
Size : har trade khate ka 10%, max 10.
Backtest (148 coins, 2020-26): 588 trades, jeet 6.9/10, PF 1.67, ausat +1.1%/trade; 10%: CAGR 10.5%, DD -16%, Sharpe 0.80,
saal 2021 +22 / 2022 0 / 2023 +10 / 2024 +27 / 2025 +5 / 2026 +10. Taala (aakhri 12 mahine, 46 trades): jeet 7.2/10,
PF 3.42 > random p95 2.15 = PASS. Kamzori: DEV mein saaf 5-din nafa random se behtar nahi tha (BORDERLINE), padosi
3 din taale mein kamzor -> asal imtihan yahi paper.
"""
import book_bot as B

B.CFG.update(name="4 din ki girawat", kind="streak", emoji="🪜", stop_atr=3.0, tp=0.05, hold=10, sma_exit=True,
             pos_pct=0.10, max_pos=10, history_days=420,
             state="streak_paper_state.json", trades="streak_paper_trades.csv", signals="streak_signals.json",
             howto="Becho: +5% par (TP), ya close 3-din average se ooper band ho to agle din open par, ya SL, ya 10 din baad.")

if __name__ == "__main__":
    B.main()
