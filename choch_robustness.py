"""
CHoCH (Change of Character) - Robustness Test
==============================================
Wohi framework jo Union AB / Donchian / Pullback par chala (parameter grid +
4-fold walk-forward + bootstrap CI + Top-10 nikal kar PF), ab CHoCH entry par.

ENTRIES (3 qisam):
  1. "CHoCH khaalis"      - sirf fresh CHoCH (bearish -> bullish pehla break)
  2. "CHoCH + Trend"      - CHoCH + close > EMA200 + EMA50 > EMA200
                            (wohi trend-shart jo Market Structure strategy mein hai)
  3. "CHoCH + Score"      - CHoCH + confluence score >= threshold
                            (NEW AdvancedConfluence wala tareeqa)

FILTER TIERS:
  - "No filter"           - koi bahar ka filter nahi (moqabla ke liye)
  - "RS only"             - RS Trend (vs BTC) + RS Percentile>=95  (Backup Tier)
  - "ETH+RS"              - ETH Regime + RS Trend + RS%>=95        (Baseline)
  - "RS only [OLD]"       - wohi, lekin PURANE tareeqe se (lookahead ke sath)
  - "ETH+RS [OLD]"        - wohi, lekin PURANE tareeqe se (lookahead ke sath)

*** ZAROORI DARYAFT - DAILY FILTER MEIN LOOKAHEAD ***
ccxt ki daily candle ka timestamp din ke SHURU (00:00) ka hota hai, lekin
us ka close din ke AAKHIR mein banta hai. Purana code (is_bullish_at /
merge_asof backward) 1h signal (misal 10:00) par USI din ki daily candle
utha leta tha - yani backtest ko din ke aakhir ka close (14 ghante aage ka)
pehle se pata hota tha. RS Percentile>=95 par iska asar sab se ziyada hai:
jis din coin zor se upar band hua, us din ke saare signals "pass" ho jate the.
Live mein aisa mumkin nahi (wahan din adhoora hota hai).
Is script mein fix: daily data sirf tab istemal hota hai jab wo candle BAND
ho chuki ho (timestamp + 23h, yani 23:00 wali 1h candle ke close par).
Isi tarah confluence score ka 4H trend aur BTC daily bhi band candle se.
[OLD] tiers sirf ye dikhane ke liye hain ke lookahead ne natija kitna
barhaya tha.

EXITS (2 qisam):
  - "CE+2R"   - Chandelier trailing + fixed TP (Entry + 2x Risk), jo pehle lage
                (Union AB jaisa)
  - "CE only" - sirf Chandelier trailing (Donchian/Pullback jaisa - yehi dono
                systems sab se mazboot nikle the)

Grid: chandelier period 12/16/20 x multiplier 3.5/4.5/5.5
Fee 0.1% + slippage 0.05% dono taraf, gap-fill fix, check-before-update.

Natija: choch_robustness_RESULTS.txt
"""

import numpy as np
import pandas as pd

import config
import scheduled_dashboard_scan as live
import confluence_engine as ce_mod
from confluence_engine import detect_market_structure, compute_confluence, DEFAULT_PARAMS as CONF_PARAMS
from strategies import apply_cooldown

TOP_N_COINS = 150
TF = "1h"
CANDLE_LIMIT = 8760
DAILY_LIMIT = 800
N_FOLDS = 4
MAX_HOLD_BARS = 500
FEE = config.BACKTEST_PARAMS["fee_pct"] / 100
SLIP = config.BACKTEST_PARAMS["slippage_pct"] / 100
RR_MULTIPLE = live.RR_MULTIPLE      # 2.0
N_BOOTSTRAP = 5000
COOLDOWN = config.SIGNAL_COOLDOWN_BARS

PERIOD_GRID = [12, 16, 20]
MULT_GRID = [3.5, 4.5, 5.5]
GRID = [(p, m) for p in PERIOD_GRID for m in MULT_GRID]

ENTRIES = ["CHoCH khaalis", "CHoCH + Trend", "CHoCH + Score"]
TIERS = ["No filter", "RS only", "ETH+RS", "RS only [OLD]", "ETH+RS [OLD]"]
EXITS = ["CE+2R", "CE only"]

# Acceptance (pehle se tay): har fold PF>1 (kam az kam 10 trades har fold),
# kul >= 80 trades, bootstrap 5th-percentile PF > 1.2
MIN_TOTAL_TRADES = 80
MIN_FOLD_TRADES = 10

DAILY_CLOSE_SHIFT = pd.Timedelta(hours=23)   # daily candle 23:00 wali 1h candle ke close par "band"
H4_CLOSE_SHIFT = pd.Timedelta(hours=3)       # 4H candle 3 ghante baad wali 1h candle ke close par "band"


# ---------------------------------------------------------------
# Lookahead-free versions (sirf is test ke andar, production file nahi chheri)
# ---------------------------------------------------------------
_orig_resample_to_4h = ce_mod.resample_to_4h


def _resample_to_4h_closed(df):
    r = _orig_resample_to_4h(df)
    r["timestamp"] = pd.to_datetime(r["timestamp"]) + H4_CLOSE_SHIFT
    return r


def shift_daily(daily_df):
    d = daily_df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"]) + DAILY_CLOSE_SHIFT
    return d


def _ns(x):
    return pd.to_datetime(x).astype("datetime64[ns]")


def daily_filter_arrays(ts_1h, eth_daily, coin_daily, btc_daily, shift):
    """Har 1h bar ke liye (eth_ok, rs_ok) arrays - vectorized, production
    passes_rs_filters / passes_full_filters jaisi hi logic."""
    sh = DAILY_CLOSE_SHIFT if shift else pd.Timedelta(0)

    eth_ema = eth_daily["close"].ewm(span=live.ETH_EMA_PERIOD, adjust=False).mean()
    eth_df = pd.DataFrame({"timestamp": _ns(eth_daily["timestamp"]) + sh,
                           "eth_ok": (eth_daily["close"] > eth_ema).values}).sort_values("timestamp")

    c = coin_daily[["timestamp", "close"]].rename(columns={"close": "coin_close"})
    b = btc_daily[["timestamp", "close"]].rename(columns={"close": "btc_close"})
    m = pd.merge_asof(c.sort_values("timestamp"), b.sort_values("timestamp"), on="timestamp", direction="backward")
    ratio = m["coin_close"] / m["btc_close"].replace(0, np.nan)
    rs_ema = ratio.ewm(span=live.RS_EMA_PERIOD, adjust=False).mean()
    rs_bull = ratio > rs_ema
    look = live.RS_PERCENTILE_LOOKBACK_DAYS
    rs_pct = ratio.rolling(look, min_periods=2).apply(lambda w: (w <= w[-1]).sum() / len(w) * 100, raw=True)
    rs_df = pd.DataFrame({"timestamp": _ns(m["timestamp"]) + sh,
                          "rs_ok": (rs_bull & (rs_pct >= live.RS_PERCENTILE_CUTOFF)).values}).sort_values("timestamp")

    base = pd.DataFrame({"timestamp": _ns(pd.Series(ts_1h)).values})
    e = pd.merge_asof(base, eth_df, on="timestamp", direction="backward")
    r = pd.merge_asof(base, rs_df, on="timestamp", direction="backward")
    eth_ok = e["eth_ok"].astype("boolean").fillna(True).to_numpy(dtype=bool)   # is_bullish_at: data na ho to True
    rs_ok = r["rs_ok"].astype("boolean").fillna(False).to_numpy(dtype=bool)    # percentile < 2 values -> False
    return eth_ok, rs_ok


def chandelier_arr(df, period, mult):
    atr = live.compute_atr(df, period)
    return (df["high"].rolling(period).max() - mult * atr).values


def simulate(arrs, stop_series, positions, rr_multiple):
    op, hi, lo, cl = arrs
    n = len(op)
    out = []
    for i in positions:
        if i + 1 >= n or np.isnan(stop_series[i]):
            continue
        eb = i + 1
        entry = op[eb] * (1 + SLIP)
        init = float(stop_series[i])
        if init >= entry:
            continue
        risk = entry - init
        tp_price = entry + risk * rr_multiple if rr_multiple else None
        trail = init
        exit_px = exit_bar = None
        last = min(eb + MAX_HOLD_BARS, n)
        for j in range(eb, last):
            if lo[j] <= trail:
                exit_px, exit_bar = min(trail, op[j]), j      # gap fill
                break
            if tp_price is not None and hi[j] >= tp_price:
                exit_px, exit_bar = max(tp_price, op[j]), j   # gap-up par open par fill
                break
            s = stop_series[j]
            if not np.isnan(s) and s > trail:
                trail = float(s)                              # check ke BAAD update
        if exit_px is None:
            exit_bar = last - 1
            exit_px = cl[exit_bar]
        exit_px *= (1 - SLIP)
        ret = ((exit_px - entry) / entry - 2 * FEE) * 100
        out.append((i, ret))
    return out


# ---------------------------------------------------------------
# Stats
# ---------------------------------------------------------------
def pf_of(rets):
    rets = np.asarray(rets, dtype=float)
    if len(rets) == 0:
        return None
    w, l = rets[rets > 0].sum(), rets[rets <= 0].sum()
    return w / abs(l) if l != 0 else None


def bootstrap_p5(rets, seed=42):
    rets = np.asarray(rets, dtype=float)
    n = len(rets)
    if n < 10:
        return None
    rng = np.random.default_rng(seed)
    pfs = []
    chunk = max(1, min(N_BOOTSTRAP, 2_000_000 // n))
    done = 0
    while done < N_BOOTSTRAP:
        k = min(chunk, N_BOOTSTRAP - done)
        s = rets[rng.integers(0, n, (k, n))]
        w = np.where(s > 0, s, 0).sum(axis=1)
        l = np.abs(np.where(s <= 0, s, 0).sum(axis=1))
        ok = l > 0
        pfs.append(w[ok] / l[ok])
        done += k
    pfs = np.concatenate(pfs)
    return float(np.percentile(pfs, 5)) if len(pfs) else None


def summarize(trades):
    """trades: list of (fold, ret)"""
    if not trades:
        return None
    rets = np.array([r for _, r in trades])
    folds = np.array([f for f, _ in trades])
    pf = pf_of(rets)
    srt = np.sort(rets)[::-1]
    t10 = pf_of(srt[10:]) if len(srt) > 10 else None
    fold_pf, fold_n = [], []
    for f in range(N_FOLDS):
        fr = rets[folds == f]
        fold_pf.append(pf_of(fr))
        fold_n.append(len(fr))
    p5 = bootstrap_p5(rets)
    # max drawdown (trade sequence, fixed-fraction ke bajaye simple cumulative %)
    return {
        "n": len(rets), "win": (rets > 0).mean() * 100, "pf": pf, "p5": p5, "t10": t10,
        "exp": rets.mean(), "fold_pf": fold_pf, "fold_n": fold_n,
    }


def passes(s):
    if s is None or s["pf"] is None or s["p5"] is None:
        return False
    if s["n"] < MIN_TOTAL_TRADES:
        return False
    for fpf, fn in zip(s["fold_pf"], s["fold_n"]):
        if fpf is None or fpf <= 1 or fn < MIN_FOLD_TRADES:
            return False
    return s["p5"] > 1.2


def fmt(x, d=2):
    return "N/A" if x is None else f"{x:.{d}f}"


def line_for(key, s):
    entry, tier, exitm, (p, m) = key
    if s is None:
        return f"  p{p}/m{m}: koi trade nahi"
    folds = " / ".join(f"{fmt(fp)}({fn})" for fp, fn in zip(s["fold_pf"], s["fold_n"]))
    tag = "  <== PASS" if passes(s) else ""
    return (f"  p{p}/m{m}: n={s['n']} win={s['win']:.1f}% PF={fmt(s['pf'],3)} p5={fmt(s['p5'],3)} "
            f"Top10-hata={fmt(s['t10'],3)} Exp={s['exp']:+.2f}%/trade | Folds: {folds}{tag}")


# ---------------------------------------------------------------
# Per-coin processing (alag function taake test ho sake)
# ---------------------------------------------------------------
def process_coin(df, coin_daily, btc_daily, eth_daily, results):
    n = len(df)
    ts = pd.to_datetime(df["timestamp"])
    close = df["close"]

    # --- CHoCH ---
    _, choch, _ = detect_market_structure(df, CONF_PARAMS["pivot_lookback"], CONF_PARAMS["min_swing_pct"])
    choch = choch.fillna(False).astype(bool)

    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    trend_ok = (close > ema200) & (ema50 > ema200)

    ce_mod.resample_to_4h = _resample_to_4h_closed
    try:
        conf = compute_confluence(df, shift_daily(btc_daily), CONF_PARAMS, usdt_d_weak=None)
    finally:
        ce_mod.resample_to_4h = _orig_resample_to_4h
    score_ok = (conf["score"] >= CONF_PARAMS["score_threshold"]).values

    entry_sigs = {
        "CHoCH khaalis": apply_cooldown(choch, COOLDOWN).values,
        "CHoCH + Trend": apply_cooldown(choch & trend_ok, COOLDOWN).values,
        "CHoCH + Score": apply_cooldown(pd.Series(choch.values & score_ok, index=df.index), COOLDOWN).values,
    }

    eth_new, rs_new = daily_filter_arrays(ts, eth_daily, coin_daily, btc_daily, shift=True)
    eth_old, rs_old = daily_filter_arrays(ts, eth_daily, coin_daily, btc_daily, shift=False)
    tier_masks = {
        "No filter": np.ones(n, dtype=bool),
        "RS only": rs_new,
        "ETH+RS": rs_new & eth_new,
        "RS only [OLD]": rs_old,
        "ETH+RS [OLD]": rs_old & eth_old,
    }

    arrs = (df["open"].values, df["high"].values, df["low"].values, df["close"].values)
    stops = {(p, m): chandelier_arr(df, p, m) for (p, m) in GRID}
    bounds = [int(n * k / N_FOLDS) for k in range(N_FOLDS + 1)]

    def fold_of(i):
        for f in range(N_FOLDS):
            if bounds[f] <= i < bounds[f + 1]:
                return f
        return N_FOLDS - 1

    for entry in ENTRIES:
        sig = entry_sigs[entry]
        for tier in TIERS:
            pos = np.where(sig & tier_masks[tier])[0]
            if len(pos) == 0:
                continue
            for exitm in EXITS:
                rr = RR_MULTIPLE if exitm == "CE+2R" else None
                for pm in GRID:
                    for i, ret in simulate(arrs, stops[pm], pos, rr):
                        results[(entry, tier, exitm, pm)].append((fold_of(i), ret))


def build_report(results):
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    stats = {k: summarize(v) for k, v in results.items()}

    emit("=" * 78)
    emit("CHoCH ROBUSTNESS TEST - NATIJA")
    emit("=" * 78)
    emit(f"PASS shart: har fold PF>1 (har fold >= {MIN_FOLD_TRADES} trades), kul >= {MIN_TOTAL_TRADES} trades, "
         f"bootstrap 5th-percentile PF > 1.2")

    # --- 1. Khulasa: har entry/tier/exit ka behtareen combo ---
    emit("\n----- KHULASA: har (Entry | Tier | Exit) ka sab se mazboot combo (p5 ke hisab se) -----")
    for entry in ENTRIES:
        for tier in TIERS:
            for exitm in EXITS:
                best, best_key = None, None
                n_pass = 0
                for pm in GRID:
                    s = stats[(entry, tier, exitm, pm)]
                    if passes(s):
                        n_pass += 1
                    if s and s["p5"] is not None and s["n"] >= 30 and (best is None or s["p5"] > best["p5"]):
                        best, best_key = s, pm
                label = f"{entry:14s} | {tier:14s} | {exitm:7s}"
                if best is None:
                    emit(f"{label}: kaafi trades nahi")
                else:
                    emit(f"{label}: best p{best_key[0]}/m{best_key[1]} n={best['n']} PF={fmt(best['pf'])} "
                         f"p5={fmt(best['p5'])} Top10-hata={fmt(best['t10'])} | PASS {n_pass}/9 grid")

    # --- 2. Lookahead ka asar ---
    emit("\n----- LOOKAHEAD KA ASAR (production grid p16/m4.5, CE+2R) -----")
    for entry in ENTRIES:
        for new_t, old_t in (("RS only", "RS only [OLD]"), ("ETH+RS", "ETH+RS [OLD]")):
            sn = stats[(entry, new_t, "CE+2R", (16, 4.5))]
            so = stats[(entry, old_t, "CE+2R", (16, 4.5))]
            emit(f"{entry:14s} | {new_t:8s}: PURANA (lookahead) n={so['n'] if so else 0} PF={fmt(so['pf'] if so else None)}  "
                 f"-->  SAHI n={sn['n'] if sn else 0} PF={fmt(sn['pf'] if sn else None)}")

    # --- 3. Tamam PASS ---
    emit("\n----- TAMAM PASS COMBOS (sirf SAHI / lookahead-free tiers) -----")
    passed = [(k, s) for k, s in stats.items() if "[OLD]" not in k[1] and passes(s)]
    passed.sort(key=lambda ks: -ks[1]["p5"])
    if not passed:
        emit("Koi combo PASS nahi hua.")
    for k, s in passed:
        emit(f"[{k[0]} | {k[1]} | {k[2]}]" + line_for(k, s))

    # --- 4. Poori detail ---
    emit("\n\n" + "=" * 78)
    emit("POORI DETAIL")
    emit("=" * 78)
    for entry in ENTRIES:
        for tier in TIERS:
            for exitm in EXITS:
                emit(f"\n### {entry} | {tier} | {exitm}")
                for pm in GRID:
                    emit(line_for((entry, tier, exitm, pm), stats[(entry, tier, exitm, pm)]))
    return lines


def main():
    from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
    exchange = get_exchange()
    coins = get_coin_list(exchange)[:TOP_N_COINS]
    print(f"{len(coins)} coins, {TF}, {CANDLE_LIMIT} candles\n")

    btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=DAILY_LIMIT)
    eth_daily = fetch_ohlcv(exchange, "ETH/USDT", "1d", limit=DAILY_LIMIT)

    results = {(e, t, x, pm): [] for e in ENTRIES for t in TIERS for x in EXITS for pm in GRID}

    for k, symbol in enumerate(coins, 1):
        try:
            df = fetch_ohlcv(exchange, symbol, TF, limit=CANDLE_LIMIT)
            if df is None or len(df) < 500:
                continue
            coin_daily = fetch_ohlcv(exchange, symbol, "1d", limit=DAILY_LIMIT)
            process_coin(df, coin_daily, btc_daily, eth_daily, results)
        except Exception as e:
            print(f"[{k}/{len(coins)}] {symbol}: SKIP ({e})")
            continue
        if k % 10 == 0 or k == len(coins):
            print(f"[{k}/{len(coins)}] ... done")

    lines = build_report(results)
    with open("choch_robustness_RESULTS.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n[SAVE] choch_robustness_RESULTS.txt")


if __name__ == "__main__":
    main()
