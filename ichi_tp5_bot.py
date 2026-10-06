"""
ICHIMOKU TP5 BOT - SIRF PAPER (win-rate version), live Ichimoku 4H ke SATH muqable ke liye
=========================================================================================
Entry bilkul live Ichimoku 4H jaisi. Farq sirf exit/size mein (winrate_lab + tp5_validate, 2026-10-04):
  Stop : Chandelier 16 / 4.0 x ATR (live 5.5) - trailing, sirf ooper
  TP   : poori position entry se +5% par (live 3R)
  Size : har trade 2% risk, max 10, ek coin max 20%
  Coins: top-250 liquid (2026-10-05 se; pehle 200) (live Ichimoku top-100)
Backtest (PURANA, lookahead bug wala - JHOOTA): win ~78%, PF 2.07. Fix ke baad (2026-10-05): win ~66%, PF ~1.01 = random jaisa.
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
B.UNIVERSE = 250          # universe2_test (2026-10-05): top-250 Sharpe 2.99, 4.8 signals/hafta (top-200: 2.88, 4.4); 300+ ke coins kamzor
B.TOP_N_COINS = 330       # 250 liquid chunne ke liye thore ziada coins fetch
B.TELEGRAM_ON = False     # 2026-10-06: lookahead fix ke baad backtest mein edge NAHI (PF ~1.0) -> Telegram khamosh, sirf paper record

if __name__ == "__main__":
    B.main()
