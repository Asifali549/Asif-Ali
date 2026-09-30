# CLAUDE.md — Asif ka Crypto Bots Project (nayi chat yahan se shuru kare)

> Nayi chat mein Claude ye file PEHLE parhe. Is mein project ka poora haal, faislay aur usool hain.
> Har bara kaam khatam hone par is file ka "Haal / Aglay kaam" hissa update karo aur push karo.

## User
- Asif, Pakistan (PKT = UTC+5). **Jawab hamesha Urdu mein.** Waqt PKT mein batao.
- Trading: sirf **spot, buy-only**, daily/swing. Maqsad: ek sach mein qabil-e-aitmaad strategy.
- Mobile se kaam karta hai (GitHub app + Streamlit dashboard). Coding khud nahi karta - Claude repo mein
  seedha commit/push karta hai (Claude GitHub App installed).
- Waqt ke andaazay (estimates) mat do - tests minutes mein khatam hote hain.
- Commit messages ke aakhir mein attribution lines (system reminder wali) lagao.

## Repo: github.com/Asifali549/Asif-Ali (branch main, public)
| File | Kaam |
|---|---|
| `ichimoku4h_bot.py` | Ichimoku 4H bot - signals + paper trading + Telegram |
| `donchian_daily_bot.py` | Donchian Daily bot - signals + paper trading + Telegram |
| `bot_core.py` | dono bots ke helpers: fetch_full, norm, ema, chandelier, ichi_signal, FEE/SLIP/STOP_SLIP, STABLES |
| `strategies.py`, `config.py` | ichimoku + market_structure signal functions aur unke params (bot_core inhein use karta hai) |
| `data_fetcher.py` | KuCoin (ccxt) exchange + top coins list |
| `live_colorful_dashboard.py` | Streamlit Cloud dashboard (sirf ye 2 systems) |
| `loop_watchdog.py` | bot ruk jaye (state ka `last_updated` purana) to workflow dobara chalata hai + Telegram |
| `telegram_alert.py` | Telegram - token/chat id env ya Streamlit secrets se |
| `*_paper_state.json`, `*_paper_trades.csv`, `*_signals.json` | bots ka data (bots khud commit karte hain) - haath mat lagao |

Workflows (.github/workflows): `ichimoku4h_bot.yml` (cron `10 */4 * * *`), `donchian_daily_bot.yml`
(cron `15 0 * * *` = 5:15 AM PKT), `watchdog.yml` (har 30 min), `telegram_test.yml` (sirf manual).

Secrets:
- GitHub Actions secrets: `GH_TOKEN` (watchdog ke liye), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
- Streamlit secrets: `GITHUB_TOKEN` (fine-grained PAT, Actions + Contents = Read and write) - dashboard ke "▶️ Abhi chala do" buttons.
- Token kabhi code mein mat likho.

## Live strategies (dono paper trading par, $1000 start, 1% risk/trade, max 10 positions, ek coin max 20%)
**1) Ichimoku + Market Structure, 4H**
- Entry: ek hi 4H candle par Ichimoku (tenkan 9 / kijun 26 / senkou_b 52, volume > 2x) AUR market structure
  (pivot 5, swing 1.5%), cooldown `config.SIGNAL_COOLDOWN_BARS`. Agli candle ke open par.
- Exit: Chandelier 16 / 5.5x ATR trailing + TP 3R. Koi BTC/ETH market filter NAHI.
- Backtest (5.3 saal): PF ~1.9, CAGR ~26%, MaxDD ~-11%, Sharpe ~1.6, 2022 ~-5%. 19/20 parameter variants pass.

**2) Donchian 20, Daily**
- Entry: daily close > pichle 20 din ka highest high (fresh), BTC close > BTC EMA50, coin top-100 liquid.
  Agle din ke open par. Slots kam hon to 60-din momentum wala pehle.
- Exit: Chandelier 22 / 4x ATR trailing (sirf ooper jata hai).
- Backtest (~6.5 saal): PF ~1.5, CAGR 20-29%, MaxDD -28 se -33%.

## Testing ke usool (har nayi strategy/tabdeeli par lazmi)
- Lookahead-free: signal band candle par, entry agli candle ke open par; daily filter sirf band daily candle se.
- Kharcha: fee 0.1% + slippage 0.05% har taraf + stop par 0.25% extra; gap par exit = min(stop, open).
- Pehle stop check, phir trailing stop update (ulta karne se jhoote PF 45 aaye the).
- Pass hone ke liye: random-entry control se behtar, bootstrap p5 PF > 1, 4 time-folds, top-10 trades hata kar
  bhi PF > 1, parameter-neighbourhood mazboot, portfolio Sharpe > random portfolio ka 95th percentile.
- Data: KuCoin top-150, point-in-time top-100 liquidity. Khatra: aaj ki coin list = survivorship bias.
- pandas 3: timestamps ko ns mein normalize karo (`norm`).

## Jo FAIL ho chuka (dobara waqt zaya na karo)
- 1h ki sab screener strategies (Union AB, CHoCH/AdvancedConfluence, CE Buy-Only, Pullback, 1h Donchian).
- Donchian 3/5/7/10 din fail; 15 aur 20 pass. ETH EMA50 filter: thora ziada return lekin gehra drawdown - BTC rakha.
- Ichimoku 4H par extra filters (BTC, ETH, RSI, ADX) se faida nahi hua; volume shart zaroori hai.
- Purane research scripts/natije git history mein hain: commit `ec7962f` (cleanup se pehle) par
  `git show ec7962f:<file>` se wapas mil sakte hain (unified_test.py, ichimoku4h_validation.py, strategy_lab.py,
  portfolio_test.py, donchian_regime_test.py, archive/...).

## Haal (2026-09-30)
- Dono bots live aur chal rahe; Telegram test kamyab; tino GitHub secrets lag gaye.
- Repo saaf kiya: sirf upar wali files. Dashboard naya (2 systems, TradingView link, manual run button,
  per-system closed-trade performance, PKT session analysis).
- Purana Telegram token public git history mein hai - user ko BotFather `/revoke` ka mashwara diya.

## Aglay kaam
- User dashboard review kar ke mazeed tabdeeliyan batayega.
- 2-3 mahine paper trading ke natije backtest se milao, phir asli paisa.
