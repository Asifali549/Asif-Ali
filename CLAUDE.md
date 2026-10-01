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
| `dip_daily_bot.py` | Dip Daily bot - signals + paper trading + Telegram |
| `donchian_daily_bot.py` | Donchian Daily bot - signals + paper trading + Telegram |
| `bot_core.py` | dono bots ke helpers: fetch_full, norm, ema, chandelier, ichi_signal, FEE/SLIP/STOP_SLIP, STABLES |
| `strategies.py`, `config.py` | ichimoku + market_structure signal functions aur unke params (bot_core inhein use karta hai) |
| `data_fetcher.py` | KuCoin (ccxt) exchange + top coins list |
| `live_colorful_dashboard.py` | Streamlit Cloud dashboard (teeno systems) |
| `loop_watchdog.py` | bot ruk jaye (state ka `last_updated` purana) to workflow dobara chalata hai + Telegram |
| `telegram_alert.py` | Telegram - token/chat id env ya Streamlit secrets se |
| `*_paper_state.json`, `*_paper_trades.csv`, `*_signals.json` | bots ka data (bots khud commit karte hain) - haath mat lagao |

Workflows (.github/workflows): `ichimoku4h_bot.yml` (cron `10 */4 * * *`), `donchian_daily_bot.yml`
(cron `15 0 * * *` = 5:15 AM PKT), `dip_daily_bot.yml` (cron `20 0 * * *`), `watchdog.yml` (har 30 min), `telegram_test.yml` (sirf manual).

Secrets:
- GitHub Actions secrets: `GH_TOKEN` (watchdog ke liye), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
- Streamlit secrets: `GITHUB_TOKEN` (fine-grained PAT, Actions + Contents = Read and write) - dashboard ke "▶️ Abhi chala do" buttons.
- Token kabhi code mein mat likho.

## Live strategies (teeno paper trading par, $1000 start; Ichimoku/Donchian: 1% risk/trade, max 10, ek coin max 20%)
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

**3) Dip Daily** (`dip_daily_bot.py`, workflow `dip_daily_bot.yml` cron `20 0 * * *` = 5:20 AM PKT)
- Entry: coin daily close > EMA200 aur EMA50 > EMA200, BTC close > EMA50, RSI(3) < 10, top-100 liquid.
  Agle din open par. Kai signals hon to sab se kam RSI pehle.
- Exit: close > SMA5 -> agle din open par; stop = signal close - 3x ATR(14) fixed; 10 din baad close par.
  Jis din coin band ho us din usi coin mein nayi entry nahi (backtest jaisa).
- Size: har trade apne hisse ka 20% (2026-10-01 se; pehle 10%), max 10. Files dip_paper_state.json / dip_paper_trades.csv / dip_signals.json.
- Backtest (2020-10 -> 2026-09): n=254, win 69%, PF 2.07, p5 1.48, random PF 0.82, CAGR ~6.5%, MaxDD ~-17%.
  Grid RSI 5-12 x exit SMA 3/5/7 sab PF 1.7-4 (random se behtar); fail sirf kam trades / 2022 fold ~1.0 ki wajah se;
  RSI<15 kamzor (PF 1.4). BTC filter hatane se kamzor. Bot ne fake data par 46/46 trades backtest se hubahu milayin.
- Kam return, lekin breakout bots ka ulta (girawat par khareed) - diversification.

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
- Swing Lab / Dip Focus test code: `git show 5d223f5:swing_lab.py` / `dip_focus.py` (+ `_RESULTS.txt`).
- Donchian 3/5/7/10 din fail; 15 aur 20 pass. ETH EMA50 filter: thora ziada return lekin gehra drawdown - BTC rakha.
- Ichimoku 4H par extra filters (BTC, ETH, RSI, ADX) se faida nahi hua; volume shart zaroori hai.
- **SMC MTF (4H trend -> 1H -> 15m pullback+BOS, TP 2R) - FAIL (2026-09-30):** 39 coins, Nov 2024-Sep 2026:
  PF 0.58, win 31%, CAGR -6%; 14/14 variants fail, random entries jaisi ya un se bhi buri. 15m par kharcha
  chhote stops ko kha jata hai. Files `git show 2645cca:smc_mtf_strategy.py` / `smc_mtf_test.py` / `smc_mtf_RESULTS.txt`.
  Sabaq: 15m/1h intraday entries is setup (spot, taker fees) mein kaam nahi karti - 4H/Daily par raho.
- **Swing Lab (2026-09-30, 109 coins, 2020-10 -> 2026-09):** SMC_4H FAIL (PF 0.82, 0/8; sirf 67 trades,
  random se bura). SQUEEZE_4H FAIL (PF 1.34 lekin random PF 1.47 - edge entry ka nahi trailing/trend ka,
  MaxDD -49%, 0/7). Natija `swing_lab_RESULTS.txt`.
- Purane research scripts/natije git history mein hain: commit `ec7962f` (cleanup se pehle) par
  `git show ec7962f:<file>` se wapas mil sakte hain (unified_test.py, ichimoku4h_validation.py, strategy_lab.py,
  portfolio_test.py, donchian_regime_test.py, archive/...).

## Haal (2026-09-30)
- Teesra bot Dip Daily shamil (dashboard tab + watchdog). Ichimoku/Donchian bots live aur chal rahe; Telegram test kamyab; tino GitHub secrets lag gaye.
- Repo saaf kiya: sirf upar wali files. Dashboard naya (2 systems, TradingView link, manual run button,
  per-system closed-trade performance, PKT session analysis).
- Purana Telegram token public git history mein hai - user ko BotFather `/revoke` ka mashwara diya.

## Portfolio Lab natija (2026-09-30, 109 coins, 2020-10 -> 2026-09, rozana mark-to-market)
`portfolio_lab.py` / workflow "Portfolio Lab Test" (dobara chalane ke liye rakha) / `portfolio_lab_RESULTS.txt`.
| Portfolio | CAGR | MaxDD | Sharpe | 2022 |
|---|---|---|---|---|
| ICHI akela | +34.6% | -11.5% | 1.81 | +1% |
| DON akela | +31.2% | -34.6% | 1.02 | -24% |
| DIP akela (10%) / (20%) | +6.6% / +9.7% | -18% / -28% | 0.53 / 0.58 | 0% / -1% |
| ICHI 50 / DON 25 / DIP 25 | +27.5% | -14.0% | 1.62 | -6% |
| ICHI 60 / DIP(20%) 40 | +24.9% | -12.2% | 1.74 | 0% (koi saal manfi nahi) |
| ICHI 50 / DON 50 | +33.8% | -20.1% | 1.43 | -12% |
- Mahana correlation: ICHI-DON 0.72 (dono breakout - sath girte hain), DIP sirf 0.17-0.19 (asli diversifier).
- Is engine mein ICHI ke number pehle (26.6%) se ziada aaye - absolute number se ziada tarteeb (ranking) par bharosa karo;
  survivorship bias ki wajah se asal mein kam hon ge.
- **FAISLA (2026-10-01, user ne mana): ICHI 60% + DIP 40% (DIP har trade apne hisse ka 20% = kul ka 8%),
  DON 0% - sirf paper par nazar.** Lagu: dip_daily_bot POS_PCT 0.20 + ALLOC 0.40, ichimoku4h_bot ALLOC 0.60
  (Telegram mein "hisse ka X% = kul capital ka Y%"), Donchian Telegram mein "Sirf PAPER", dashboard sidebar
  "Sarmaye ki taqseem" (kul capital -> har system ka $) aur har signal ka size system ke hisse se.

## Aglay kaam
- Dip Daily bot naya (2026-09-30) - pehle run ke baad dashboard/Telegram check karo.
- User dashboard review kar ke mazeed tabdeeliyan batayega.
- 2-3 mahine paper trading ke natije backtest se milao, phir asli paisa.
