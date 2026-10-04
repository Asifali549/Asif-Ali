"""
ICHIMOKU TP5 BOT - SIRF PAPER (win-rate version), live Ichimoku 4H ke SATH muqable ke liye
=========================================================================================
Entry bilkul live Ichimoku 4H jaisi. Farq sirf exit/size mein (winrate_lab + tp5_validate, 2026-10-04):
  Stop : Chandelier 16 / 4.0 x ATR (live 5.5) - trailing, sirf ooper
  TP   : poori position entry se +5% par (live 3R)
  Size : har trade 2% risk, max 10, ek coin max 20%
  Coins: top-200 liquid (live Ichimoku top-100)
Backtest: win ~78% (live ~42%), PF 2.07, 2% risk par CAGR ~33%, MaxDD ~-7% (live 1%: 35% / -12.5%).
Asli paisa NAHI - 2-3 mahine live Ichimoku se muqabla, phir faisla.
State: ichi_tp5_paper_state.json | trades: ichi_tp5_paper_trades.csv | signals: ichi_tp5_signals.json
"""
import ichimoku4h_bot as B

B.CE_M = 4.0
B.TP_PCT = 0.05
B.RISK_PCT = 0.02
B.ALLOC = 0.0
B.PAPER_ONLY = True
B.SYSTEM_NAME = "Ichimoku TP5"
B.STATE_FILE = "ichi_tp5_paper_state.json"
B.TRADES_CSV = "ichi_tp5_paper_trades.csv"
B.SIGNALS_FILE = "ichi_tp5_signals.json"
B.UNIVERSE = 200          # universe_test (2026-10-04): top-200 par win 78%, Sharpe 2.81 (top-100: 2.48)
B.TOP_N_COINS = 260

if __name__ == "__main__":
    B.main()
