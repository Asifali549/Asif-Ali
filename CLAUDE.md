# CLAUDE.md — Asif ka Crypto Bots Project (nayi chat yahan se shuru kare)

> Nayi chat mein Claude ye file PEHLE parhe. Is mein project ka poora haal, faislay aur usool hain.
> Har bara kaam khatam hone par is file ka "Haal / Aglay kaam" hissa update karo aur push karo.

## User
- Strategy pasand: signals market ke sath sath milte rahein (rozana), taqatwar trades - saal mein chand trades wali
  (BTC/ETH trend) strategies NAHI chahiye.
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
| `telegram_alert.py` | Telegram (token/chat id env ya Streamlit secrets) + har alert ntfy.sh par bhi (topic `asifali549-strong-signals-9k3m7x`, ya secret `NTFY_TOPIC`) |
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
- **Majors Trend (BTC/ETH SMA/EMA/Donchian, 2026-10-01):** sab rules B&H se behtar (DD -78% -> -35/-50%) lekin
  random-timing se sirf thora behtar, 3/20 pass (best SMA150+vol: CAGR 41%, DD -35%, Sharpe 1.21). **User ne
  REJECT kiya** - saal mein chand trades nahi chahiye; use ziada signals wali taqatwar strategy chahiye.
  Files `git show 9e0f740:majors_lab.py` / `majors_lab_RESULTS.txt`.
- **DIP 4H (2026-10-01, top 70/100/150/200 coins) - FAIL 0/16:** win ~61% lekin PF 0.87-1.04 (random PF ~0.80 -
  entry mein thora edge, magar 7-8 trades/hafta ka kharcha sab kha jata hai), portfolio CAGR -6 se -49%, DD -59 se -99%.
  Coins ziada/kam karne se farq nahi. Sabaq (3rd baar): ziada signals = ziada kharcha; 4H par mean-reversion nahi chalti.
  Universe note: sirf 175 coins ke paas kaafi history; rozana ausat ~94 coins volume-data wale -> top150/200 = top100 jaisa.
  Daily Dip isi test mein (20% size, 175 coins): top70 PF 2.55 win 73% CAGR +17.6% DD -26%; top100 PF 2.41 CAGR +15.4%
  DD -36% -> top70 thora behtar (sirf ek test; live bot abhi top100). Files `git show f289b1a:dip4h_universe.py` / `_RESULTS.txt`.
- **SPOT CORE TREND-PULLBACK PRO (user ka spec, 2026-10-01) - FAIL 0/9:** 4H daily+4H trend, ADX, pullback/support,
  RSI 45-65, candle, volume 1.75x, reclaim, overext, R:R; TP1/TP2/trail. 109 coins 2020-10->2026-09: sirf 41 trades
  (0.1/hafta!), win 24%, PF 0.48, random PF 0.75 se bhi bura; bull regime mein PF 0.29. ABLATION: filters ulta nuqsan
  dete hain - volume hatao PF 1.23 (387 trades) magar random 1.14, p5 0.99 -> edge nahi; daily/reclaim hatao PF 1.10.
  Sabaq: bohat se filters AND karne se "perfect setup" late entry ban jata hai; 4H pullback-reclaim ka crypto mein edge
  nahi (SMC_4H jaisa). Files `git show 2c353bc:spot_core_lab.py` / `spot_core_RESULTS.txt`; Pine `git show 05f2b7f:spot_core_trend_pullback_pro.pine`.
- **SECTOR ROTATION (user ka idea, 2026-10-01) - FAIL:** 6 dasti shobe (89 coins), Daily Donchian N20 + ATR 3x.
  BASE PF 1.41 (rnd 1.12), OOS 1.16, CAGR 23.8%, DD -33%, 2022 0.52, 2025 0.98. ROTATION top-K by momentum: 1/5
  (sirf top-1 30d: PF 1.55, OOS 1.26 magar random OOS bhi 1.21, CAGR 14.5%, median coin PF 1.17 - akela cell, plateau
  nahi); top-2/3 aur 14d/60d mein 2022/2025 aur bure. #1 shoba har ~6 din badalta hai (noise). SECTOR BREADTH 50-80%:
  0/4 - PF barhta hai magar random bhi (+0.12..+0.18 sirf), 2022 0.52 -> 0.11-0.31, 2025 0.98 -> 0.46-0.83 (market
  breadth jaisa hi bull-trap masla). Sabaq: shobe ki taqat Donchian ki 2022/2025 kamzori theek nahi karti.
  Files `git show f985de7:sector_rotation.py` / natija commit ke baad `sector_rotation_RESULTS.txt` history mein.
- **WEEKLY RS ROTATION (top-K coins by L-din return, har itwar, 2026-10-01) - FAIL (tradeable nahi):**
  Pehla test: 9/12 grid random (turnover-matched) p95 se behtar, OOS 2025+ +3..+728% vs random -40..-56%.
  VALIDATION (6 plateau configs): edge hai (2022 se 5/6, 365+ din coins 6/6, top-50 coins 6/6 random p95 se behtar)
  LEKIN (1) nafa lottery jaisa: top-5 coins = kul net nafa ka 80-161% (ZEC, VVV, USELESS, AKE, TEL, PUMP...),
  unhein nikaal kar sirf 2/6 bache; nafa wale coins ~45%, nuqsan wale ~55%. (2) MaxDD -80 se -95% HAR version mein
  (purane/bare coins mein bhi), 2022 -76..-89%, 1000-1900 din paani ke neeche, sirf ~50% mahine musbat.
  (3) Equity-curve filter (apni equity < SMA 30/50/100 -> cash): 0/18 - DD sirf -68..-91% tak aaya, nafa aadha.
  Sabaq: crypto mein cross-sectional momentum ka edge right-tail (chand pump coins) se aata hai aur survivorship
  bias isi ko phulata hai; crash risk itna hai ke koi insaan nahi jhel sakta. Files `git show c8faed3:rs_rotation.py`,
  `git show c7fa63a:rs_validate.py`, natije `git show cd81dde:rs_rotation_RESULTS.txt` / `git show 2005af0:rs_validate_RESULTS.txt`.
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

## Simple Donchian Research natija (2026-10-01, 109 coins, 2020-10 -> 2026-09, OOS 2025+)
`donchian_research.py` (numba) / workflow "Donchian Research Test" / `donchian_research_RESULTS.txt` (agle qadam ke liye rakha).
- **4H: FAIL 0/100** - PF 0.77-1.15, OOS PF har config < 1 (0.80-0.99), random jaisa; portfolio DD -72 se -98%.
  Bear (PF 0.82-0.93) aur sideways mein kharab. Tight stop (1.5x) sab se bura -> 4H par noise + kharcha.
- **Daily: 5/10 PASS** - PF 1.09-1.54, OOS 0.98-1.16; trailing (B 3.0x) > fixed TP. Random PF bhi 1.01-1.35
  (trailing + crypto uptrend khud kuch deta hai); entry ka faida N>=20 par +0.15 se +0.35 PF, N=10 par ~0 (random jaisa).
  Plateau: N 20-55 + trail 3.0x sab PF 1.47-1.54. DD -25 se -50% (koi market filter nahi).
- Masla-shinakht (user ka rule): asal masla TIMEFRAME (4H) tha; entry hypothesis Daily par sach hai; exit: trailing behtar.
  Ye live Donchian Daily (N20 + CE22/4 + BTC>EMA50) ki tasdeeq hai.
- **EK FILTER (BTC>EMA50) natija** (`donchian_filter.py`, workflow "Donchian Filter Test", `donchian_filter_RESULTS.txt`):
  trail 3.0x: BAGHAIR filter 4/5 N PASS, BTC>EMA50 sirf 1/5 (N20). Filter se PF barhta hai (1.49->1.61 N20) lekin
  RANDOM PF bhi barhta hai (1.12->1.31; N55 1.35->1.55) - yani filter market-timing hai, entry edge nahi; breakout ka
  random par faida ghatta hai (+0.36->+0.30, N40 +0.35->+0.22, N55 +0.23->+0.10). OOS PF thora behtar (N20 1.19->1.28,
  N30 1.08->1.13, N40 1.15->1.19) magar N55 kam (1.17->1.13); MaxDD sirf N20 behtar (-32.5->-29.7), baqi bura (-25->-32).
  Bear PF thora behtar (0.83-1.05 -> 0.89-1.15). FAISLA: filter ka faida N par munhasir, plateau nahi -> universal
  taur par SABIT NAHI. Daily Donchian (filter baghair, N20-55, trail 3x) khud mazboot. Live bot (N20+BTC filter) wohi
  ek cell hai jahan filter sab cheez behtar karta hai - badalne ki zaroorat nahi.

## Donchian N=20 Validation natija (2026-10-01) - PROMISING, NEEDS MORE VALIDATION (dono versions 5/6)
`donchian_validate.py` / workflow "Donchian Validation Test" / `donchian_validate_RESULTS.txt`.
- Coin-level (baghair filter): 101 coins, median coin PF 1.30, mean 1.69, PF>1: 67 (66%), PF<1: 34; median coin net +2.2%.
  Top 10% coins (11) = 60% kul nafa (FLAG) - lekin unhein hata kar bhi PF 1.22. BTC filter: median 1.39, 62/86 PF>1, top10% = 54%.
- Regimes: bull PF 1.61 vs random 1.10 (asli edge); BEAR 1.03 vs random 1.18 (edge NAHI); sideways 2.21 vs 1.61 (216 trades);
  high-vol +0.12 (BTC filter -0.01, edge nahi); low-vol +0.41. (Regime MaxDD column bekaar - har trade 20% sequential, ignore.)
- Walk-forward OOS: 2023 2.15 (rnd 1.98), 2024 1.65 (1.31), 2025 0.87 (0.66 - nuqsan magar random se behtar),
  2026 1.38 (random 1.41 - random se KAM). Haaliya edge kamzor ho raha hai.
- Plateau N 15-30: PF 1.36-1.55, random se +0.22..+0.40, OOS 1.04-1.20 -> mazboot.
- Exits 6/6 random se behtar; Donchian 10-din low exit: PF 2.42 (rnd 1.86, +0.56, OOS 1.49) sab se behtar; ATR trail sab se kam DD (-32%).
- Faisla: PROMISING - kyunke (1) nafa right-tail coins par kaafi had tak, (2) bear/high-vol mein edge nahi, (3) 2025-26 kamzor.

## Donchian Stress natija (2026-10-01) - `donchian_stress.py` / `donchian_stress_RESULTS.txt`
- COST STRESS: mazboot. Baghair filter ATR trail PF 1.45 -> 1.36 (2x) -> 1.28 (3x), OOS 1.18 -> 1.04, random par faida
  +0.28 -> +0.25 barqarar. Donchian 10 exit PF 2.39 -> 2.25, OOS 1.49 -> 1.40, faida +0.69 -> +0.66.
- EXIT PLATEAU (Donchian M): baghair filter 5/5 M (5-30) random se +0.57..+0.84, OOS 1.37-2.01; lekin M>=15 par
  MaxDD -62 se -76% (M5 -37%, M10 -40%, ATR trail -32%). BTC filter ke sath M>=15 random jaisa (2/5).
- 2025: BTC -7% saal, magar BTC regime 73% din "bull" - altcoins kamzor (breadth kam). 56/90 coins manfi, bure 10
  coins = 40% nuqsan -> nuqsan PHAILA hua (chand coins nahi). Jan-Mar aur May bhari nuqsan; Apr/Jul nafa.
  2025 bull-regime trades PF 0.89. Saal-war PF: 2020 1.93, 2021 2.95, 2022 0.49, 2023 2.07, 2024 1.33, 2025 0.91, 2026 1.39.
  (Breadth % ka denominator listing se pehle wale coins bhi ginta hai - absolute % kam; sirf relative dekho.)
- Tashkhees: edge kharche aur exit par mazboot; kamzori ALTCOIN market ki halat (2022, 2025) - BTC regime ise nahi pakarta.
- **ALTCOIN BREADTH FILTER - FAIL (0/4 had, dono exits)** (`donchian_breadth.py`, `donchian_breadth_RESULTS.txt`):
  ATR trail: PF 1.44 -> 1.51-1.62 lekin random bhi 1.32-1.43 (faida +0.12..+0.28); OOS 1.17 -> 1.18-1.23; DD -33 -> -28..-36%.
  Asal maqsad NAKAM: 2022 PF 0.49 -> 0.22-0.26 aur 2025 0.89 -> 0.66-0.87 (aur BURA). Donchian exit: 2022 0.37 -> 0.11-0.16.
  Matlab: bure saalon ka nuqsan tab hota hai jab breadth ooncha ho (bear market ke jhoote rally / bull traps) -
  market-level filters (BTC, breadth) is kamzori ko theek nahi karte. FAISLA: Donchian par filter-tahqeeq BAND.

## Zer-e-test
- **ALT/BTC BREAKOUT** (`alt_btc.py`, workflow "ALT BTC Breakout Test", natija `alt_btc_RESULTS.txt`): trade USDT
  mein, signal ALT/BTC ratio se (BTC khud bahar). BASE USDT Donchian N20; RATIO BRK (ratio close > pichle N din ka
  max ratio close, N 10/20/30/55); BASE + filter ratio > EMA 20/50/100. Exits ATR trail 3x aur Donchian 10 low.
  Pass vs base: random +0.15, OOS >= base, DD 3% se ziada bura nahi, 2022 aur 2025 PF behtar, n >= 150; plateau.

## Aglay kaam
- Dip Daily bot naya (2026-09-30) - pehle run ke baad dashboard/Telegram check karo.
- User dashboard review kar ke mazeed tabdeeliyan batayega.
- 2-3 mahine paper trading ke natije backtest se milao, phir asli paisa.
