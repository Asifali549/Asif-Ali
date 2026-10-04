# CLAUDE.md — Asif ka Crypto Bots Project (nayi chat yahan se shuru kare)

> Nayi chat mein Claude ye file PEHLE parhe. Is mein project ka poora haal, faislay aur usool hain.
> Har bara kaam khatam hone par is file ka "Haal / Aglay kaam" hissa update karo aur push karo.

## User
- Strategy pasand: signals market ke sath sath milte rahein (rozana), taqatwar trades - saal mein chand trades wali
  (BTC/ETH trend) strategies NAHI chahiye.
- Asif, Pakistan (PKT = UTC+5). **Jawab hamesha Urdu mein.** Waqt PKT mein batao.
- **Urdu likhai (2026-10-04, user ki shikayat):** Urdu jumlon mein English alfaaz, minus/% wale numbers mix karne se mobile par
  alfaaz aage peeche ho jate hain. Is liye: naam Urdu rasm-ul-khat mein (اچیموکو، ڈپ، ڈونچین، کیپیچولیشن، ٹی پی، ایس ایل، پی ایف)،
  "%" ki jagah "فیصد"، minus ki jagah "نقصان/کمی" lafz; bullet/line English lafz ya number se shuru na ho.
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
| `capit_daily_bot.py` | Volume Capitulation bot - signals + paper trading + Telegram (2026-10-02 se) |
| `ichi_tp5_bot.py` | **Ichimoku TP5 - SIRF PAPER** (2026-10-04 se): `ichimoku4h_bot` import kar ke CE_M 4, TP_PCT 5%, RISK 2%, ALLOC 0, **UNIVERSE 200 / TOP_N 260**, apni files `ichi_tp5_*` |
| `bot_core.py` | dono bots ke helpers: fetch_full, norm, ema, chandelier, ichi_signal, FEE/SLIP/STOP_SLIP, STABLES |
| `strategies.py`, `config.py` | ichimoku + market_structure signal functions aur unke params (bot_core inhein use karta hai) |
| `data_fetcher.py` | KuCoin (ccxt) exchange + top coins list |
| `live_colorful_dashboard.py` | Streamlit dashboard - tabs: 📊 Aaj ka Scoreboard (tamam signals + khuli trades, nafa/nuqsan) / 🏁 Muqabla (5 systems ek table: equity, win/PF/DD live vs test, asli paise ki 4 shartein - GOLIVE dict, mushtarka equity chart; 🔔 Telegram / 🔕 khamosh) / 🤖 Auto Trading (bots, har system ki tab) / ✋ Manual Trading (user ki apni trades, GitHub `manual_trades.json` mein Contents API se) / 🕐 Session |
| `scheduler.py` | GitHub cron ka mutabadil: ~5h40m chalta, theek waqt par bots dispatch karta (GH_TOKEN), phir khud ko dobara chalata (zanjeer); push par bhi shuru |
| `loop_watchdog.py` | bot ruk jaye (state ka `last_updated` purana) to workflow dobara chalata hai + Telegram |
| `telegram_alert.py` | Telegram (token/chat id env ya Streamlit secrets) + har alert ntfy.sh par bhi (topic `asifali549-strong-signals-9k3m7x`, ya secret `NTFY_TOPIC`) |
| `*_paper_state.json`, `*_paper_trades.csv`, `*_signals.json` | bots ka data (bots khud commit karte hain) - haath mat lagao |
| `manual_trades.json` | user ki manual trades (dashboard likhta hai) - haath mat lagao |

Workflows (.github/workflows): `ichimoku4h_bot.yml` (cron `10 */4 * * *`), `donchian_daily_bot.yml`
(cron `15 0 * * *` = 5:15 AM PKT), `dip_daily_bot.yml` (cron `20 0 * * *`), `capit_daily_bot.yml` (cron `25 0 * * *`), `ichi_tp5_bot.yml` (cron `12 */4 * * *`, scheduler :12), `watchdog.yml` (har 30 min), `scheduler.yml` (lagataar; asal waqt-paband trigger - cron ab sirf backup), `telegram_test.yml` (sirf manual).

Secrets:
- GitHub Actions secrets: `GH_TOKEN` (watchdog ke liye), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
- Streamlit secrets: `GITHUB_TOKEN` (fine-grained PAT, Actions + Contents = Read and write) - dashboard ke "▶️ Abhi chala do" buttons.
- Token kabhi code mein mat likho.

## Live strategies (chaaron paper trading par, $1000 start; Ichimoku/Donchian: 1% risk/trade, max 10, ek coin max 20%)
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
- **v2 (2026-10-04 se, user ne kaha "badal do"): RSI3 < 7, exit close > SMA3, TP +5% (stop TP se pehle check), baqi wahi.
  Naya record shuru; v1 ka record `dip_v1_paper_state.json` / `dip_v1_signals.json` mein (git mv). Neeche v1 ki tafseel.**
- Entry: coin daily close > EMA200 aur EMA50 > EMA200, BTC close > EMA50, RSI(3) < 10, top-100 liquid.
  Agle din open par. Kai signals hon to sab se kam RSI pehle.
- Exit: close > SMA5 -> agle din open par; stop = signal close - 3x ATR(14) fixed; 10 din baad close par.
  Jis din coin band ho us din usi coin mein nayi entry nahi (backtest jaisa).
- Size: har trade apne hisse ka 20% (2026-10-01 se; pehle 10%), max 10. Hissa 40% (10-02 ko kuch ghante 30% raha, wapas 40%). Files dip_paper_state.json / dip_paper_trades.csv / dip_signals.json.
- Backtest (2020-10 -> 2026-09): n=254, win 69%, PF 2.07, p5 1.48, random PF 0.82, CAGR ~6.5%, MaxDD ~-17%.
  Grid RSI 5-12 x exit SMA 3/5/7 sab PF 1.7-4 (random se behtar); fail sirf kam trades / 2022 fold ~1.0 ki wajah se;
  RSI<15 kamzor (PF 1.4). BTC filter hatane se kamzor. Bot ne fake data par 46/46 trades backtest se hubahu milayin.
- Kam return, lekin breakout bots ka ulta (girawat par khareed) - diversification.

**4) Volume Capitulation Daily** (`capit_daily_bot.py`, workflow `capit_daily_bot.yml` cron `25 0 * * *` = 5:25 AM PKT)
- Entry: coin daily close > EMA200, us din return <= -8%, volume >= 2 x pichle 20 din ka ausat (shift 1), top-100
  liquid; koi BTC filter nahi. Agle din open par. Kai signals hon to sab se gehri girawat pehle.
- Exit/stop/time: Dip jaisa (close > SMA5 -> agle din open; stop signal close - 3 ATR fixed; 10 din). HISTORY_DAYS 1000
  (EMA200 pakne ke liye). Size: paper mein 20%/trade, max 10; **hissa 0% - SIRF PAPER (2026-10-02 se)**. Files capit_paper_state.json /
  capit_paper_trades.csv / capit_signals.json.
- Backtest (new_ideas + capit_validate): 313 trades, win 65%, PF 1.87, OOS 1.59, p5 1.40, 4/4 folds, 3x kharcha 1.66.
- Bot ne fake data par din-ba-din replay mein 45/45 backtest trades hubahu (entry, exit, return) milayin; 1 extra trade.

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
- **ALT/BTC BREAKOUT (2026-10-01) - FAIL 0/14:** 104 coins (BTC bahar). RATIO BRK (ALT/BTC ratio close > pichle N din
  max, N 10/20/30/55) apne aap mein bhi random jaisa: ATR trail PF 1.13-1.25 vs random 1.09-1.22 (-0.05..+0.15),
  Donchian exit 1.61-1.67 vs random 1.64-1.86 (-0.20..+0.02); DD -40..-56%; 2022 0.39-0.63, 2025 0.58-0.91.
  Filter "ratio > EMA 20/50/100" base (USDT N20) ko kamzor karta hai (PF 1.43 -> 1.25-1.34, 2025 0.87 -> 0.58-0.85).
  Sabaq: BTC ke muqable taqat ka koi alag entry edge nahi; USDT breakout behtar. Files `git show 2873a0c:alt_btc.py`,
  natija `git show 2ed9b41:alt_btc_RESULTS.txt`. User ne kaha: har cheez ko Donchian par mat parkho - aage naye
  khayal portfolio ke liye parkhne hain (random se behtar + ICHI/DIP se kam correlation + portfolio Sharpe/DD behtar).
- **VOLATILITY TARGETING ICHI60/DIP40 (2026-10-01) - FAIL 0/8:** nayi trade ka size x clip(normal vol/haaliya vol,
  0.25, CAP); BTC vol ya portfolio ki apni vol, L 20/60, CAP 1.0/1.5. BASE CAGR 23.7%, DD -11.1%, Sharpe 1.65.
  Sharpe sirf 1.53-1.70 (-0.12..+0.05); DD kam hua to nafa bhi utna hi kam (APNI L20: DD -7.5% magar CAGR 18.1%) =
  bas chhota size. Sabaq: portfolio pehle se mutawazin; vol scaling se muft faida nahi. Files `git show 70675d7:vol_target.py`,
  natija `git show 6ab0786:vol_target_RESULTS.txt`.
- **STRATEGY LAB 2 (2026-10-02) - 4/4 FAIL** (`git show bfd832f:strategy_lab2.py` / natija commit `strategy_lab2_RESULTS.txt`
  history mein): E1 BB reversion (uptrend, close < BB lower k, exit SMA5/SMA20) plateau 1/6, validation FAIL (p5 0.83),
  akela DD -49%; E2 failed breakdown / turtle soup (N 10/20/55, uptrend ya nahi) 0/6, aksar random se bura; E3 volume
  thrust (+X% din, volume Vx, close upper 25%, chandelier 22/3) 0/6, random jaisa, DD -70%; E4 RS pullback (top-20% L-din
  RS + D din neeche) 0/6, PF ~1.0-1.1 random p95 ke barabar. Portfolio mein har ek ne Sharpe 1.74 -> 1.39-1.55 giraya.
  RANKING akele (Sharpe): Ichimoku 1.69 (PF 2.34, OOS 1.82) > Donchian 1.10 (DD -37%) > Dip 0.72 (PF 2.36, OOS 3.92)
  > Capitulation 0.41 (OOS 0.83).
- **RESEARCH LAB 3 (2026-10-02) - koi naya portfolio behtar nahi** (`git show c4e14d9:research_lab3.py`, natija
  `git show d8fc454:research_lab3_RESULTS.txt`). CONTROL DIP (RSI3<10) har jaanch PASS: PF 1.94 vs random p95 1.20, OOS 5.70,
  bootstrap p5 1.35, top-10 hata kar 1.48, 2x kharcha 1.77, UNIVERSE 80% 6/6 (PF 1.89-2.25) -> Dip waqai mazboot.
  RSI3<5: win 79%, PF 3.18 magar sirf 43 trades. M1 IBS: FAIL (random se bura). M2 down-streak 4 din: apna edge PASS
  (PF 1.35, universe 6/6) LEKIN portfolio kharab (ICHI 50/DIP 30/M2 20 Sharpe 1.44 vs base 1.67, DD -15.5 vs -11.0;
  DIP+M2 ensemble 1.26). M3 stretch: universe 3/6 FAIL. M4 Dip 12H: plateau 1/3, folds 2/4 FAIL, akela DD -67%.
  T1 Ichimoku Daily: sirf 52 trades, random p95 ke barabar, top-10 hata kar 0.45 -> FAIL (portfolio mein 1.70/-8.9% magar
  bharosa nahi). NATIJA: ICHI 60 / DIP 40 (CAGR ~24%, DD -11%, Sharpe ~1.67-1.73) ab tak sab se behtar; 20+ khayal ke
  baad aur talaash se false-discovery ka khatra barhta hai.
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

## Naye khayal / Capitulation natija
- **NAYE KHAYAL natija (2026-10-02, `new_ideas.py` / `new_ideas_RESULTS.txt`):**
  A) BTC catch-up: 5/9 (X>=4% chalta, PF 1.42-1.84 vs rnd p95 1.12-1.42) LEKIN ICHI se corr 0.50, akela DD -58%,
     2022 PF 0.16-0.49, 2025 0.36-0.68; portfolio mein Sharpe 1.68 -> 1.47 -> FAIL (portfolio ke liye bekaar).
  B) Market panic: 3/9 (sirf Z-9%), Z-5/-7% random se bura; akela CAGR -5%, DD -71% -> FAIL.
  C) **VOLUME CAPITULATION: 7/9 PASS** - PF 1.55-1.89 vs random p50 ~1.0 (p95 1.2-1.5), win 57-66%, OOS 1.5-2.1,
     ~0.4-1.3 trades/hafta; 2022 kamzor (0.21-0.75). Corr ICHI 0.08, DIP -0.02 (asli diversifier).
     ICHI 50 / DIP 30 / C(10%) 20: Sharpe 1.73 vs 1.68, DD -9.5% vs -11.0%, +mahine 62% vs 56%, CAGR 21.5 vs 24.2.
     -> PROMISING. Validation: `capit_validate.py` (workflow "Capitulation Validation Test", push par chalta) -
     bootstrap, top-10 hata kar, 4 folds, coin concentration, kharcha 2x/3x, regime, DIP overlap, size/hissa mixes.
  **VALIDATION NATIJA (2026-10-02, `capit_validate_RESULTS.txt`) - PASS, PAPER ke qabil:** R8% V2.0: 313 trades
  (1/hafta), win 65%, PF 1.87, OOS 1.59; bootstrap p5 1.40; top-10 hata kar 1.38 (top-20: 1.19); 4/4 folds musbat
  (1.55-2.61); kharcha 3x PF 1.66; 48/73 coins nafa mein, median coin PF 2.76 (lekin top 10% coins = 70% nafa - flag);
  bull PF 2.21 (255), bear 1.08 (30), sideways 1.39 (28); 2022 sirf 7 trades PF 0.55; DIP se overlap sirf 7%.
  R8% V2.5 bhi pass (PF 1.79, p5 1.22, top-20 hata kar 0.89). Portfolio (C har trade 20%): ICHI 60/DIP 30/C 10 ->
  CAGR 25.9%, DD -9.6%, Sharpe 1.81; ICHI 60/DIP 20/C 20 -> 26.8%, DD -8.4%, Sharpe 1.79, +mahine 66% (BASE 24.9%,
  -11.0%, 1.71, 55%). Corr C: ICHI 0.03, DIP 0.06. Khatre: 3 khayalon mein se chuna (multiple testing), survivorship.
  **LAGU (2026-10-02, user ne kaha "bot mein shamil karo"):** capit_daily_bot + dashboard tab + watchdog;
  taqseem ICHI 60 / DIP 30 / CAPIT 10 (sab se ooncha Sharpe 1.81) - dip_daily_bot ALLOC 0.30, dashboard alloc.
  **DOBARA CHALANE PAR (2026-10-02 dopahar) KAMZOR:** sirf 1 din baad, aaj ki top-150 coin list badalne se (US, UAI,
  ALLO, AKE, GTC jaise naye coins) R8 V2.0: PF 1.87 -> 1.47, OOS 1.59 -> **0.83**, 2026 PF 0.62, bear PF 0.37, top10%
  coins = 77% nafa; V2.5 bootstrap p5 0.92 FAIL. C20 akela 2026 -22%. Portfolio ICHI 60/DIP 30/C 10 Sharpe 1.74 vs
  ICHI 60/DIP 40 1.73 (koi farq nahi). FAISLA: **Capitulation SIRF PAPER (ALLOC 0), taqseem wapas ICHI 60 / DIP 40.**
  SABAQ (bohat ahem): `get_coin_list` har din aaj ke 24h volume se top-150 chunta hai -> backtest universe roz badalta
  hai; kam trades wali strategy (300) ka OOS isi se ulat sakta hai. Aage har validation mein (a) 2 alag din ke coin
  list / ya coins ka 10-20% random hata kar sensitivity, (b) OOS > 1 dono mein - tabhi PASS.

## MFE Study natija (2026-10-04) - `mfe_study.py` / workflow "MFE Study Test" / `mfe_study_RESULTS.txt`
User ka sawal: harne wali trades pehle 2-4% ooper ja kar wapas SL hit karti hain? Chhota TP / breakeven madad karega?
- ICHI: haan - harne walon ka median MFE 7.8%, 91% pehle >2% ooper gayin (jeetne walon ka median MFE 32%).
  BASE CAGR 35.3% DD -12.5% Sharpe 1.82. P2-P8 (aadhi bechna): CAGR 21-25%, Sharpe 1.30-1.53 (BURA).
  BE2/BE3: win 75%, CAGR 16-19%, DD -6/-8%, Sharpe 1.90. **TP5 (poori 5% par): win 83%, CAGR 15.6%, DD -4.6%,
  Sharpe 2.60, OOS PF 2.57 (base 1.99)**; TP3 Sharpe 2.28 CAGR 9%. Matlab: nafa kam magar bohat hamwar ->
  aage test: TP4-7 plateau + 1.5-2% risk (size barha kar CAGR wapas?), folds, kharcha. Live bot NAHI badla.
- DON: losers median MFE 9.4%, lekin har variant Sharpe/CAGR girata (bare winners median 62% kat jate); sirf TP5 DD -40 -> -29.
- DIP: harne wali aksar foran ulat jati (median MFE 2.4%, 30% <1%) - user ka khayal yahan sahi nahi; variants ~barabar/bure
  (BE2 thora behtar: CAGR 13.3 vs 12.3, DD -23.8 vs -26.9 - kamzor faida). CAPIT: har variant bura.

## Winrate Lab natija (2026-10-04) - `winrate_lab.py` / workflow "Winrate Lab Test" / `winrate_lab_RESULTS.txt`
User: "win rate ziada chahiye, jo mumkin hai test karo". 31 ICHI exit variants + random control + mazbooti + size.
(win = +0.5% se ziada; BE exits "brbr" alag - MFE study ke BE "75% win" asal mein ~11% jeet + 68% barabar thay.)
- **CE4_TP5 (stop CE 16 / 4x ATR + poori position +5% par): win 78%, PF 2.07, OOS 2.40, RND PF p95 0.92 (entry edge
  asli), saal-war PF 2021-26 sab >1 (2022 1.31, BASE 0.85), 2x kharcha 1.80, boot p5 1.75, -top10 2.03.**
  Portfolio: 1% risk CAGR 19.5% DD -4.4% Sharpe 2.67; **2% risk CAGR 32.6% DD -6.6% Sharpe 2.53**; 3%/15/30% 49% DD -10%.
  BASE 1%: 35% / -12.5% / 1.81; BASE 1.5%: 51% / -16%. Matlab: same nafa adhe drawdown par, win 42% -> 78%.
- Plateau: TP3-TP6 aur TPR0.25-0.5 sab win 74-90%, Sharpe 2.2-2.6, sab random p95 se behtar. TP5: win 83%, 2% risk 28.5% / -9%.
- BE akela: asli jeet sirf 10-17% (baqi barabar). TS (time stop) = BASE jaisa. P{x}BE Sharpe gira.
- DIP: RSI<7 SMA3 TP5 win 81%, CAGR 10.8%, DD -17.6%, Sharpe 0.93 (live RSI<10 SMA5: 12.3% / -26.9% / 0.67); RSI<5 SMA3 win 81%
  PF 6.4 magar sirf 68 trades (CAGR 5%).
- Khatra: 31 variants mein se chuna (multiple testing) - lekin poora padosi khandan acha hai. Live NAHI badla.
  Aglay: CE 3.5/4/4.5 x TP 4/5/6 grid + coin-universe sensitivity (20% coins hata kar) + paper bot "ICHI TP5" live ke sath.

## TP5 Validation natija (2026-10-04) - `tp5_validate.py` / workflow "TP5 Validation Test" / `tp5_validate_RESULTS.txt` - **PASS (sab)**
- Grid CE 3.5/4/4.5/5 x TP 4/5/6/7: **16/16** cells PF>1.5, OOS>1, random p95 se behtar (win 61-85%, 2% risk CAGR 22-33%, DD -6..-11%).
- Random 20 seeds (CE4 TP5): asli win 77.8% / PF 2.04 / Sharpe 2.51 vs random p95 win 47.5% / PF 0.99 / Sharpe -0.11.
- 4 folds PF 2.10/1.72/1.97/2.30; boot p5 1.70; -top10 2.00; 3x kharcha PF 1.57 (win 77% barqarar).
- Universe (20% coins hata kar x8): **8/8 PASS** (PF 1.90-2.15, OOS 1.92-2.39, CAGR 23-27%, DD -6..-10%).
- Portfolio: TP5 2% 31.8% / -6.6% / Sharpe 2.51 / +mahine 74%; live ICHI 1% 35.0% / -14.0% / 1.80 / 58%.
  Saal-war TP5 2%: 2021 +41, 2022 +3, 2023 +41, 2024 +41, **2025 +23 (live +4)**, 2026 +41 - koi saal manfi nahi.
  Mix: TP5 80/DIP20 28.3% / -5.5% / 2.54; TP5 60/DIP40 24.6% / -9.8% / 2.11; live ICHI60/DIP40 26.7% / -11.3% / 1.80.
- LAGU: paper bot `ichi_tp5_bot.py` live Ichimoku ke SATH (pehla run 2026-10-04 07:55 UTC kamyab). Live bot/taqseem NAHI badli.
  Faisla 2-3 mahine paper muqable ke baad (user se poochh kar): live Ichimoku ko TP5 se badalna + taqseem (80/20?).

## Dip v2 Validation natija (2026-10-04) - `dip_v2_validate.py` / workflow "Dip v2 Validation Test" / `dip_v2_validate_RESULTS.txt` - **PASS (sab)**
Naya Dip = RSI3<7, exit close>SMA3, TP +5% (stop/10-din wahi). Live = RSI3<10, SMA5, TP nahi.
- Padosi grid (RSI 6-8, SMA3, TP 4-6): 9/9 PF>2 + OOS>1.5. Naya: n 167 (0.54/hafta, live 1.29), win 80.8% (live 69.7),
  PF 3.99 (2.35), OOS 4.04 (2.99), CAGR 10.7% (13.6), DD -18.3% (-27.2), Sharpe 0.91 (0.70), **bura mahina -1.4% (live -18.4%)**.
- Random (same uptrend din, random entry) x20: naya PF 3.99 vs p95 1.29; live 2.35 vs 1.22 - dono asli edge.
- Folds naya 14.7/inf/3.45/2.81 (live 2.53/0.88/2.61/2.00); boot p5 2.51 (1.80); -top10 3.64; 3x kharcha 3.02.
- Universe 8/8 PASS (naya PF 3.65-7.36, DD -12..-19%; live DD -16..-28%).
- Portfolio ICHI TP5 2% ke sath: 80/naya20 28.3% / -5.3% / 2.69 / +mahine 78%; 80/live20 29.3% / -5.5% / 2.60;
  70/naya30 26.1% / -5.9% / 2.67; 60/naya40 23.8% / -7.8% / 2.57 (60/live40 25.6% / -10.7% / 2.12).
  Kamzori: naya kam signals -> 2025 +10% / 2026 +3% (live +21/+22).
- Live Dip bot NAHI badla - user ka faisla baqi.

## Universe Test natija (2026-10-04) - `universe_test.py` / workflow "Universe Test" / `universe_test_RESULTS.txt`
KuCoin top-260 fetch (221 coins, min 120 din) - is liye numbers pichle teston (111 coins, min 250 din) se thore alag.
Rozana asal eligible coins: top-100 -> 86, top-150 -> 103, top-200 -> 112 (purane coins kam, 2025+ ausat 183 data wale).
- ICHI TP5: top-100 754 trades (2.4/hafta) win 77% CAGR 38.6% DD -9.3% Sharpe 2.48 | top-150 2.7/hafta 47.5% / -9.4% / 2.71 |
  **top-200 2.9/hafta win 78% PF 1.94 OOS 1.80 CAGR 50.7% DD -9.4% Sharpe 2.81**. Naye coins (rank 101-200) ki 143 trades:
  win 81%, PF 2.50, OOS 2.22 - top-100 se BEHTAR. -> ICHI ke liye bara universe faida mand.
- ICHI live bhi top-200 par 48.8% / -15.7% / 1.86. TP5 har universe par Sharpe 2.4-2.8 vs live 1.7-1.9.
- DIP naya: top-100 0.69/hafta win 80% CAGR 14.2% DD -19.4% Sharpe 1.01; top-200 sirf 0.77/hafta, OOS 3.62 -> 1.93 gira;
  naye coins ki 25 trades PF 1.23, OOS 0.39 (KAMZOR). DIP live naye coins bhi kamzor (PF 1.43, OOS 0.67). -> Dip top-100 par raho.
- Is bare data mein DIP live DD -36.6% / bura mahina -19.5% vs DIP naya -19.4% / -3.1%.

## Stop Fix Lab natija (2026-10-04) - `stop_fix_lab.py` / workflow "Stop Fix Lab Test" / `stop_fix_lab_RESULTS.txt`
User: live CARDS (Donchian) -11.9% (SL 42% door), US (Capitulation) -24.8% (SL 89% door, $-49.8 = 5% account) - "account wash".
- **SL tang karna har system ko BURA karta hai** (DON CE3/2.5/2: PF 1.37/1.23/1.19; CAPIT 2x/1.5x/1x PF 1.60/1.34/1.21;
  DIP v2 2x/1.5x/1x PF 2.74/2.14/1.05). Door SL hi strategy ka hissa hai; masla SIZE hai, stop nahi.
- TP5/TP8 Donchian/Capitulation ko kharab (PF ~1.0-1.3). DIP v2 live (3x, TP5, fixed 20%) best: Sharpe 0.88, 1 trade max -5.8%
  (portfolio 80/20 mein kul account ka ~-1.2%); risk 2% size: 1 trade max -2.1% magar CAGR 10.4 -> 3.3%. Dip v2 NAHI badla.
- CAPIT: risk 2% size -> 1 trade max -12.6% -> -2.0%, DD -31 -> -14%, Sharpe 0.77 -> 0.86 (CAGR 24.8 -> 9.3%). **LAGU** (RISK_PCT 0.02).
- DON: maxSL 20% filter -> PF 1.75 -> 2.30, OOS 1.12 -> 1.48, DD -40 -> -33%, Sharpe 1.00 -> 1.18, CAGR 31.5 -> 36.6% (1% risk);
  2022 PF 0.11, 2025 0.59 phir bhi kamzor. **LAGU** (MAX_SL_PCT 0.20). Dono abhi bhi SIRF PAPER + Telegram khamosh.

## Aglay kaam
- **GitHub cron masla - HAL (2026-10-02):** 30 Sep se cron runs ghanton der se / gayab (Watchdog 48 ki jagah ~4/din).
  `scheduler.py` + `scheduler.yml`: ek workflow ~5h40m lagataar chalta, har minute: Ichimoku har 4h :10, Donchian 00:15,
  Dip 00:20, Capit 00:25, Watchdog :05/:35 UTC ko workflow_dispatch (foran chalta, der nahi); aakhir mein khud ko dobara
  dispatch (concurrency group = ek waqt mein ek). Watchdog `ensure_scheduler()` - scheduler band ho to chala deta hai.
  Purane cron backup; bots idempotent (last_bar/last_day). Public repo -> Actions minutes muft.
- Naye khayal (Donchian khandan se bahar), user ko diye: (1) BTC lead-lag catch-up (2) volume capitulation dip
  (3) market-wide panic ke baad khareed (4) VOLATILITY TARGETING ICHI+DIP portfolio par (meri pehli tarjeeh)
  (5) funding rate (6) seasonality. Pass ka paimana: portfolio_lab mein ICHI60/DIP40 ke sath.
- Dip Daily bot naya (2026-09-30) - pehle run ke baad dashboard/Telegram check karo.
- User dashboard review kar ke mazeed tabdeeliyan batayega.
- 2-3 mahine paper trading ke natije backtest se milao, phir asli paisa.
- Ichimoku TP5 (paper) vs live Ichimoku: dono ka win%, PF, DD dashboard se milao; TP5 behtar rahe to user se poochh kar live par lao.
- 2026-10-04: Dip v2 + TP5 top-200 dono pehle run kamyab (Dip v2 pehla signal BR/USDT).
- **FAISLA (2026-10-04, user): sab 6 bots chalte rahein (har ek ka alag record/dashboard tab), lekin TELEGRAM par sirf
  Ichimoku TP5 + Dip v2.** Lagu: `TELEGRAM_ON = False` purane Ichimoku (`ichimoku4h_bot.py`), Donchian, Capitulation mein
  (send() sirf print karta hai); `ichi_tp5_bot.py` mein `B.TELEGRAM_ON = True`; Dip bot ka send() pehle jaisa. Watchdog
  alerts chalu (system sehat). Kisi ko wapas chalu karna ho to us file mein TELEGRAM_ON = True.
- **NAYI SHURUAT (2026-10-04 shaam, user):** purana Ichimoku, Donchian, Capitulation ka record bhi naye usoolon ke liye saaf:
  purani files `ichimoku4h_v1_*`, `donchian_v1_*`, `capit_v1_*` (git mv). Ab paanchon systems (TP5, Dip v2, purana Ichimoku,
  Donchian maxSL20, Capitulation 2% risk) ka record 2026-10-04 se ek sath shuru -> barabar muqabla.
- **Dashboard 🏁 Muqabla tab (2026-10-04):** go-live shartein har system: 20 band trades, 60 din, win hadaf (TP5 70, Dip 60), DD test ke 2x se kam. Dip badge ab 🪂.
  Mashwara diya (user ki ijazat baqi): sidebar taqseem TP5 80 / Dip 20 (abhi purana Ichimoku 60 / Dip 40 dikhata hai).
- **ALLOC TEST (2026-10-04, `alloc_test.py` / `alloc_test_RESULTS.txt`, 219 coins):** naye systems ki taqseem. TP5 akela 51.2% / -13.5% / Sharpe 2.78;
  TP5 90/DIP10 2.86; **80/20 43.6% / -10.6% / 2.91; 70/30 39.8% / -9.1% / 2.92**; 60/40 36.0% / -9.0% / 2.85; 50/50 2.67 (plateau 90-60).
  CAPIT 10 ya DON 10 milane se koi faida nahi (2.85-2.93). Purana ICHI60/DIPv2 40: 28.6% / -9.5% / 1.80. Corr TP5-DIP 0.18.
  CAPIT 2% akela OOS 1.00, CAGR 7% - kamzor. Mashwara: TP5 70 / DIP 30 (ya 80/20) - user ki ijazat baqi, sidebar abhi nahi badla.
- **TP MARGIN TEST (2026-10-04, `tp_margin_test.py` / `_RESULTS.txt`, top-200, 888 trades):** user: "TP5 margin kam". SL ausat 12.1% door ->
  2% risk = size ~16%; $1000 par ek jeet ~$7.6. Bara TP: TP4 win 82% CAGR 44.5% Sharpe 3.03 | TP5 77.5% 51.5% -14.4% 2.78 | TP6 72% 54% -12.9% 2.60 |
  TP7 67% 56.7% -13.5% 2.48 | TP8 63% 56% 2.32 | TP10 56% 58.6% -18.6% 2.16 | trail 38% 48% -28.9% 1.13. -> TP 4-7 sab ek jaise, TP barhane se
  nafa thora, jhatke ziada. Asal lever SIZE: TP5 r2 cap30 59.8% / -15.1% / 2.84 (cap20 51.5 / -14.4); r3 cap30 80% / -21%.
  Mix 70 TP5 r3c30 / 30 DIP 59% / -14.9% / 2.97 (r2c20 70/30: 40% / -9.8% / 2.92). Mashwara: TP5 rakho, cap 30% (paper) - user ki ijazat baqi.
