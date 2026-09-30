# Crypto Spot Bots (buy-only)

Sirf 2 tasdeeq-shuda systems chal rahe hain (paper trading + Telegram alerts):

| System | Entry | Exit | Backtest (lookahead-free, fee+slippage) |
|---|---|---|---|
| **Ichimoku + Market Structure (4H)** | Ichimoku + MS dono ek candle par, volume > 2x | Chandelier 16 / 5.5x ATR + TP 3R | PF ~1.9, CAGR ~26%, MaxDD ~-11% |
| **Donchian 20 (Daily)** | Close > pichle 20 din ka high, BTC > EMA50 | Chandelier 22 / 4x ATR | PF ~1.5, CAGR 20-29%, MaxDD ~-30% |

Dono: 1% risk har trade, max 10 positions, ek coin max 20%.

## Files
- `ichimoku4h_bot.py`, `donchian_daily_bot.py` - bots (GitHub Actions par khud chalte hain)
- `bot_core.py`, `strategies.py`, `config.py`, `data_fetcher.py` - bots ke helpers
- `live_colorful_dashboard.py` - Streamlit dashboard
- `loop_watchdog.py` - bot ruk jaye to dobara chalata hai
- `telegram_alert.py` - Telegram (token GitHub secrets mein)

Purane research/test scripts aur unke natije git history mein mehfooz hain (commit se pehle ka version).
