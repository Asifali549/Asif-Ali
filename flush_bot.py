"""
FLUSH BOT (market ki safai) - SIRF PAPER (2026-10-06 se) - Search Lab 13 / 13b
============================================================================
Entry: jis din top-100 liquid coins mein se 40%+ ka 3-din nafa -10% ya kam ho (poori market ki safai - sirf pehla din),
       BTC close > EMA200 (bara rujhaan theek), coin close > EMA200 aur EMA50 > EMA200. Agle din open par.
       Ek din mein bohat signals aate hain -> sab se liquid coins pehle, max 10.
Exit : Dip jaisa - TP +5% | close 3-din average se ooper -> agle din open | SL signal close - 3 ATR | 10 din.
Size : har trade khate ka 10%, max 10.
Backtest (149 coins, 2020-26): 1376 trades magar sirf 54 safai din (~9 / saal); jeet 7/10, PF 1.83; 10%: CAGR 24%, DD -24%,
Sharpe 1.08; saal 2021 +137 / 2022 0 / 2023 -6 / 2024 +3 / 2025 +43 / 2026 0 (LUMPY). Taala: 25 trades jeet 6.4/10 PF 7.66 >
random p95 1.27 = PASS - MAGAR sirf 2 din (Oct 2025 crash) = KAMZOR saboot. Corr Dip v2 0.30, Resid 0.06, Streak 0.12.
"""
import book_bot as B

B.CFG.update(name="Market safai", kind="flush", emoji="🌊", stop_atr=3.0, tp=0.05, hold=10, sma_exit=True,
             pos_pct=0.10, max_pos=10, history_days=420, btc_ema=200, flush_pct=40,
             state="flush_paper_state.json", trades="flush_paper_trades.csv", signals="flush_signals.json",
             howto="Becho: +5% par (TP), ya close 3-din average se ooper band ho to agle din open par, ya SL, ya 10 din baad.")

if __name__ == "__main__":
    B.main()
