"""
CAUSALITY (SACHAI) TEST - har backtest ke liye LAZMI (2026-10-05, market-structure lookahead bug ke baad)
=====================================================================================================
Sawal: kya backtest kisi din ka signal / SL / exit wohi nikalta hai jo us din LIVE (sirf us din tak ka data) nikalta?
Tareeqa: kai random din D chuno. Har D par sab coins ka data D tak kaat do (aage ka data mitao), saari
hisaab-kitaab (BTC filter, liquidity rank, signals, SL, exit flags) dobara karo, aur D wale din ki value ko
poore data wale hisaab se milao. Ek bhi farq = backtest ko mustaqbil ka pata hai = natija jhoota.

Jaanch:
  DAILY (stop_fix_lab.prep - Dip v2 / Donchian / Capitulation backtest yahi istemal karte hain):
    signals DIP / DON / CAPIT, liquidity top-100 membership, BTC filter, ATR stop, SMA3/SMA5 exit flag,
    Donchian chandelier 22/4
  4H (bot_core.ichi_signal - Ichimoku/TP5, FIX ke baad): signal + chandelier 16/4
  RANDOM-WALK: bina kisi edge ke nakli data par har system ka PF ~1 hona chahiye (zyada aaya = engine mein bias).
Natija: causality_test_RESULTS.txt
"""
import numpy as np
import pandas as pd

import config
from bot_core import fetch_full, norm, STABLES, ichi_signal, ICHI_BASE, chandelier as ce_bot
from strategies import STRATEGY_FUNCTIONS
from portfolio_lab import ema, to_daily, H4_BARS
from winrate_lab import pf_of
import stop_fix_lab as SF

OUT = "causality_test_RESULTS.txt"
TOP_FETCH = 200
N_DAYS = 100         # daily: kitne random din
N_4H_COINS = 12      # 4H: kitne coins
N_4H_BARS = 600      # 4H: har coin ke lagataar bars


def daily_ctx(daily):
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc_ok = bd > ema(bd, 50)
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    rank = dv.rank(axis=1, ascending=False)
    al100 = {day: set(row[row <= 100].index) for day, row in rank.iterrows()}
    return btc_ok, al100


def snapshot(P, D):
    """Har coin ki D din wali values."""
    out = {}
    for x in P:
        w = np.where(x["ts"] == np.datetime64(D))[0]
        if len(w) == 0:
            continue
        i = w[0]
        if "DON" not in x["ce"]:
            x["ce"]["DON"] = ce_bot(x["d"], 22, 4.0)
        out[x["sym"]] = dict(DIP=bool(x["sig"]["DIP"][i]), DON=bool(x["sig"]["DON"][i]), CAPIT=bool(x["sig"]["CAPIT"][i]),
                             ok=bool(x["ok"][i]), atr=x["atr"][i], sma3=bool(x["sma"][3][i]), sma5=bool(x["sma"][5][i]),
                             ce=x["ce"]["DON"][i])
    return out


def same(a, b):
    if isinstance(a, bool):
        return a == b
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= 1e-9 * max(1.0, abs(a))


def daily_check(daily, emit, rng):
    start = min(d["timestamp"].iloc[0] for d in daily.values()) + pd.Timedelta(days=210)
    btc_ok, al100 = daily_ctx(daily)
    full = SF.prep(daily, al100, btc_ok, start)
    days = sorted(set(daily["BTC/USDT"]["timestamp"]))
    days = [d for d in days if d >= start + pd.Timedelta(days=30)]
    pick = sorted(rng.choice(len(days), size=min(N_DAYS, len(days)), replace=False))
    keys = ["DIP", "DON", "CAPIT", "ok", "atr", "sma3", "sma5", "ce"]
    bad = {k: 0 for k in keys}
    tot = {k: 0 for k in keys}
    nsig = {k: 0 for k in ("DIP", "DON", "CAPIT")}
    examples = []
    for k, j in enumerate(pick, 1):
        D = days[j]
        cut = {s: d[d["timestamp"] <= D].reset_index(drop=True) for s, d in daily.items()}
        cut = {s: d for s, d in cut.items() if len(d) > 0}
        b2, a2 = daily_ctx(cut)
        part = SF.prep(cut, a2, b2, start)
        A, B = snapshot(full, D), snapshot(part, D)
        for s in A:
            if s not in B:
                continue
            for key in keys:
                tot[key] += 1
                if not same(A[s][key], B[s][key]):
                    bad[key] += 1
                    if len(examples) < 10:
                        examples.append(f"{D.date()} {s} {key}: poora data {A[s][key]} | live {B[s][key]}")
            for key in nsig:
                nsig[key] += A[s][key]
        if k % 10 == 0:
            print(f"  daily {k}/{len(pick)}", flush=True)
    emit(f"\n# DAILY - {len(pick)} random din x tamam coins (live jaisa data kaat kar dobara hisaab)")
    names = {"DIP": "Dip signal", "DON": "Donchian signal", "CAPIT": "Capitulation signal", "ok": "top-100 liquidity",
             "atr": "ATR (SL)", "sma3": "SMA3 exit (Dip)", "sma5": "SMA5 exit (Capit)", "ce": "Chandelier 22/4 (Donchian SL)"}
    for key in keys:
        extra = f" | in dinon ke signals: {nsig[key]}" if key in nsig else ""
        emit(f"{names[key]:>30}: {tot[key]:>6} jaanch | farq {bad[key]:>4} -> {'SAHI' if bad[key] == 0 else 'GHALAT (lookahead)'}{extra}")
    for e in examples:
        emit("   misaal: " + e)
    return sum(bad.values()) == 0


def h4_check(h4, emit, rng):
    syms = [s for s, d in h4.items() if len(d) > 1500]
    syms = list(rng.choice(syms, size=min(N_4H_COINS, len(syms)), replace=False))
    bad_s = bad_c = bad_m = tot = nsig = nms = 0
    MP = config.STRATEGY_PARAMS["market_structure"]
    for k, s in enumerate(syms, 1):
        d = h4[s]
        fs = ichi_signal(d, dict(ICHI_BASE))
        fc = ce_bot(d, 16, 4.0)
        fm = STRATEGY_FUNCTIONS["market_structure"](d, MP).to_numpy()
        # LAGATAAR bars ka ek block (random jagah) - taake woh signals bhi pakre jayen jo live mein aate
        # magar backtest mein gayab hon (pichla bug yahi tha); random akele bars yeh nahi pakarte
        a0 = int(rng.integers(400, len(d) - N_4H_BARS))
        idx = range(a0, a0 + N_4H_BARS)
        for i in idx:
            c = d.iloc[:i + 1].reset_index(drop=True)
            ps = ichi_signal(c, dict(ICHI_BASE))[-1]
            pc = ce_bot(c, 16, 4.0)[-1]
            pm = bool(STRATEGY_FUNCTIONS["market_structure"](c, MP).to_numpy()[-1])
            nms += bool(fm[i]) or pm
            bad_m += pm != bool(fm[i])
            tot += 1
            nsig += bool(fs[i]) or bool(ps)
            bad_s += bool(ps) != bool(fs[i])
            bad_c += not same(float(pc), float(fc[i]))
        print(f"  4h {k}/{len(syms)}", flush=True)
    emit(f"\n# 4H ICHIMOKU / TP5 (market structure fix ke baad) - {len(syms)} coins, {tot} bars ({nsig} signal bars)")
    emit(f"{'Ichimoku+MS signal':>30}: farq {bad_s} -> {'SAHI' if bad_s == 0 else 'GHALAT'}")
    emit(f"{'Market structure (akela)':>30}: farq {bad_m} ({nms} signal bars) -> {'SAHI' if bad_m == 0 else 'GHALAT'}")
    emit(f"{'Chandelier 16/4 (TP5 SL)':>30}: farq {bad_c} -> {'SAHI' if bad_c == 0 else 'GHALAT'}")
    return bad_s == 0 and bad_c == 0 and bad_m == 0


def random_walk_check(emit, seeds=5):
    """Nakli data (koi edge nahi): PF ~1 aana chahiye. Har seed par 120 coins x 2000 din."""
    emit("\n# RANDOM-WALK (nakli data, koi edge nahi) - engine bias check: PF ~1.0 hona chahiye (koi bhi > 1.3 = khatra)")
    res = {"DIP": [], "DON": [], "CAPIT": []}
    for sd in range(seeds):
        rng = np.random.default_rng(900 + sd)
        daily = {}
        for k in range(120):
            n = 2000
            r = rng.standard_t(4, n) * 0.025 + 0.0004
            c = 10 * np.exp(np.cumsum(r))
            o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.003, n))
            h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.02, n)))
            l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.02, n)))
            v = rng.lognormal(12, 0.7, n) * (1 + 3 * (np.abs(r) > 0.06))
            ts = pd.date_range("2020-01-01", periods=n, freq="1D")
            daily["BTC/USDT" if k == 0 else f"R{k}/USDT"] = pd.DataFrame(
                dict(timestamp=ts, open=o, high=h, low=l, close=c, volume=v))
        btc_ok, al100 = daily_ctx(daily)
        start = daily["BTC/USDT"]["timestamp"].iloc[0] + pd.Timedelta(days=210)
        P = SF.prep(daily, al100, btc_ok, start)
        for name, k_, tp in (("DIP", 3.0, 0.05), ("DON", 4.0, None), ("CAPIT", 3.0, None)):
            res[name].append(pf_of([t["ret"] for t in SF.run(P, name, k_, tp, 0.20 if name == "DON" else None)]))
        print(f"  random-walk seed {sd + 1}/{seeds}", flush=True)
    ok = True
    for name, v in res.items():
        m = float(np.mean(v))
        ok &= m < 1.3
        emit(f"{name:>30}: PF har seed {' '.join(f'{x:.2f}' for x in v)} | ausat {m:.2f} -> {'SAHI' if m < 1.3 else 'KHATRA'}")
    return ok


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    if h4 is None:
        import data_fetcher
        data_fetcher.AUTO_TOP_N_COINS = TOP_FETCH
        ex = data_fetcher.get_exchange()
        coins = [s for s in data_fetcher.get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:TOP_FETCH]
        for must in ("ETH/USDT", "BTC/USDT"):
            if must not in coins:
                coins.insert(0, must)
        h4 = {}
        for k, sym in enumerate(coins, 1):
            try:
                df = fetch_full(ex, sym, "4h", H4_BARS)
                if df is not None and len(df) >= 6 * 120:
                    h4[sym] = norm(df)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)}", flush=True)
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})", flush=True)

    daily = {s: to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    rng = np.random.default_rng(7)
    emit("=" * 110)
    emit("CAUSALITY (SACHAI) TEST - backtest ko mustaqbil ka pata to nahi?")
    emit("=" * 110)
    emit(f"Coins: {len(h4)} | daily {min(d['timestamp'].iloc[0] for d in daily.values()).date()} -> "
         f"{max(d['timestamp'].iloc[-1] for d in daily.values()).date()}")
    a = daily_check(daily, emit, rng)
    b = h4_check(h4, emit, rng)
    c = random_walk_check(emit)
    emit("\n# KHULASA")
    emit(f"Dip / Donchian / Capitulation backtest (daily): {'SAHI - koi lookahead nahi' if a else 'GHALAT'}")
    emit(f"Ichimoku / TP5 backtest (4H, fix ke baad): {'SAHI - koi lookahead nahi' if b else 'GHALAT'}")
    emit(f"Engine random-walk par: {'SAHI - bina edge PF ~1' if c else 'KHATRA'}")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
