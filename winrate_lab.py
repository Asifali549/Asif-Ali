"""
WINRATE LAB - user: "jo bhi mumkin hai test karo, hamein WIN RATE ziada chahiye" (2026-10-04)
==============================================================================================
MFE study ne dikhaya: Ichimoku ki harne wali trades ka median pehle +7.8% ooper jata hai. Yahan har mumkin
exit/size tareeqa Ichimoku 4H par (aur Dip Daily par chand) aazmaya jata hai - lekin sirf win rate nahi:
win rate ke sath PF, OOS (2025+), CAGR, MaxDD, Sharpe, RANDOM-ENTRY control (kya ooncha win rate sirf exit
ki wajah se hai jo kisi bhi entry par mil jata?), saal-war PF, 2x kharcha, bootstrap p5, top-10 hata kar.

ICHI exit khandan:
  BASE          : CE 16/5.5 trailing + TP 3R (live)
  TP{x}         : poori position +x% par (3,4,5,6,7,8,10,12)
  TPR{r}        : poori position +r*R par (R = entry - stop) (0.25,0.33,0.5,0.75,1.0)
  BE{x}         : +x% ke baad stop entry par, baqi BASE (1.5,2,3,4)
  BE{x}_TP{y}   : dono (BE x%, TP y%)
  P{x}BE        : +x% par aadhi becho + baqi ka stop entry par, baqi BASE
  CE{m}_TP{y}   : tang stop (CE 16 / m x ATR) + TP y%
  TS{n}_{x}     : n candle (4H) mein +x% na pohnchi to agli candle open par bahar, baqi BASE
Phir sab se behtar 6 (Sharpe se) par: risk 1/1.5/2/3% x max positions 10/15 (coin cap 20% / 30%).
DIP: RSI<5/7/10 x exit close>SMA3/SMA5 x TP3/5 (win rate ke liye).
Natija: winrate_lab_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, chandelier as ce_bot, ichi_signal, ICHI_BASE, STABLES, FEE, SLIP, STOP_SLIP
from portfolio_lab import ema, rsi, atr_w, to_daily, portfolio, stats, TOP_N, H4_BARS, UNIVERSE

OUT = "winrate_lab_RESULTS.txt"
OOS_START = pd.Timestamp("2025-01-01")
BE_MULT = (1 + FEE) * (1 + SLIP) / ((1 - FEE) * (1 - SLIP) * (1 - STOP_SLIP))
RND_SEEDS = 12
WIN_MIN = 0.005   # jeet = +0.5% se ziada; |ret| <= 0.5% = "barabar" (breakeven exit asli jeet nahi)


def sim(o, h, l, c, ts, i, stop, trail, exit_sig, max_hold, cfg, cost=1.0):
    """cfg: tpR (BASE 3R), tp (pct), be (pct), part (pct, frac) + be, ts (n, pct)."""
    n = len(o)
    e = i + 1
    if e >= n:
        return None
    slip, fee, sslip = SLIP * cost, FEE * cost, STOP_SLIP * cost
    entry = o[e] * (1 + slip)
    st = stop
    if not np.isfinite(st) or entry <= st:
        return None
    R = entry - st
    tps = []
    if cfg.get("tpR"):
        tps.append(entry + cfg["tpR"] * R)
    if cfg.get("tp"):
        tps.append(entry * (1 + cfg["tp"]))
    tp = min(tps) if tps else None
    be = cfg.get("be")
    part = cfg.get("part")
    tstop = cfg.get("ts")
    part_done, part_ret, frac = False, 0.0, (part[1] if part else 0.0)
    be_on = False
    reached_ts = False
    last = min(e + max_hold, n - 1)
    px, j = None, last
    for j in range(e, last + 1):
        if l[j] <= st:
            px = min(st * (1 - sslip), o[j]) * (1 - slip)
            break
        if tp is not None and h[j] >= tp:
            px = max(tp, o[j]) * (1 - slip)
            break
        if part and not part_done and h[j] >= entry * (1 + part[0]):
            part_done = True
            part_ret = max(entry * (1 + part[0]), o[j]) * (1 - slip) * (1 - fee) / (entry * (1 + fee)) - 1
            st = max(st, entry * BE_MULT)
        if exit_sig is not None and exit_sig[j] and j + 1 < n:
            j += 1
            px = o[j] * (1 - slip)
            break
        if tstop and not reached_ts:
            if h[j] >= entry * (1 + tstop[1]):
                reached_ts = True
            elif j - e + 1 >= tstop[0] and j + 1 < n:
                j += 1
                px = o[j] * (1 - slip)
                break
        if be and not be_on and h[j] >= entry * (1 + be):
            be_on = True
            st = max(st, entry * BE_MULT)
        if trail is not None and np.isfinite(trail[j]) and trail[j] > st:
            st = trail[j]
    if px is None:
        px = c[last] * (1 - slip)
    rest = px * (1 - fee) / (entry * (1 + fee)) - 1
    ret = frac * part_ret + (1 - frac) * rest if part_done else rest
    return {"t_in": ts[e], "t_out": ts[j], "entry_px": entry, "risk": R / entry, "ret": ret}


# ---------------------------------------------------------------- ICHI prep
def prep_ichi(h4, allowed, start, rng=None):
    """Har coin ke arrays + signal indices. rng diya ho to RANDOM entries (har coin mein utne hi, allowed din)."""
    data = []
    for sym, d in h4.items():
        if len(d) < 400:
            continue
        sig = ichi_signal(d, dict(ICHI_BASE))
        mom = (d["close"] / d["close"].shift(60 * 6) - 1).to_numpy()
        o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        ts = d["timestamp"].to_numpy()
        day = d["timestamp"].dt.floor("1D")
        ok = np.array([sym in allowed.get(x, ()) for x in day]) & (d["timestamp"] >= start).to_numpy()
        idx = np.where(sig & ok)[0]
        if rng is not None:
            pool = np.where(ok)[0]
            pool = pool[pool > 60]
            if len(idx) == 0 or len(pool) == 0:
                continue
            idx = np.sort(rng.choice(pool, size=min(len(idx), len(pool)), replace=False))
        data.append(dict(sym=sym, d=d, o=o, h=h, l=l, c=c, ts=ts, idx=idx, mom=mom, ce={}))
    return data


def run_ichi(data, cfg, cost=1.0):
    out = []
    m = cfg.get("ce", 5.5)
    for x in data:
        if m not in x["ce"]:
            x["ce"][m] = ce_bot(x["d"], 16, m)
        stop = x["ce"][m]
        for i in x["idx"]:
            t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, stop[i], stop, None, 500, cfg, cost)
            if t:
                t.update(sym=x["sym"], prio=x["mom"][i] if np.isfinite(x["mom"][i]) else -9)
                out.append(t)
    return out


def tstats(tr):
    r = np.array([t["ret"] for t in tr])
    if len(r) == 0:
        return dict(n=0, win=0, scr=0, pf=0, avg=0, oos=0)
    g, lo = r[r > 0].sum(), -r[r <= 0].sum()
    ro = np.array([t["ret"] for t in tr if pd.Timestamp(t["t_in"]) >= OOS_START])
    go, loo = (ro[ro > 0].sum(), -ro[ro <= 0].sum()) if len(ro) else (0, 0)
    return dict(n=len(r), win=(r > WIN_MIN).mean() * 100, scr=(np.abs(r) <= WIN_MIN).mean() * 100,
                pf=g / lo if lo else np.inf, avg=r.mean() * 100, oos=go / loo if loo else np.inf)


def pf_of(r):
    r = np.asarray(r)
    g, lo = r[r > 0].sum(), -r[r <= 0].sum()
    return g / lo if lo else np.inf


def main(h4=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    if h4 is None:
        from data_fetcher import get_exchange, get_coin_list
        ex = get_exchange()
        coins = [s for s in get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:TOP_N]
        for must in ("ETH/USDT", "BTC/USDT"):
            if must not in coins:
                coins.insert(0, must)
        h4 = {}
        for k, sym in enumerate(coins, 1):
            try:
                df = fetch_full(ex, sym, "4h", H4_BARS)
                if df is not None and len(df) >= 6 * 250:
                    h4[sym] = norm(df)
                    print(f"[{k}/{len(coins)}] {sym}: {len(df)}")
            except Exception as e:
                print(f"[{k}] {sym}: SKIP ({e})")

    daily = {s: to_daily(d).iloc[:-1].reset_index(drop=True) for s, d in h4.items()}
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc_ok = bd > ema(bd, 50)
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean().shift(1)
    allowed = {day: set(row.dropna().nlargest(UNIVERSE).index) for day, row in dv.iterrows()}
    closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
    start = closes.index[0] + pd.Timedelta(days=210)
    closes = closes[closes.index >= start]

    emit("=" * 110)
    emit("WINRATE LAB - Ichimoku 4H (aur Dip) ke har exit/size tareeqe: win rate + nafa + risk + random control")
    emit("=" * 110)
    emit(f"Coins: {len(h4)} | {closes.index[0].date()} -> {closes.index[-1].date()} | top-{UNIVERSE} liquid | OOS = 2025+")
    emit("win% = +0.5% se ziada nafa wali trades; brbr% = -0.5..+0.5% (breakeven par nikli - na jeet na haar).")
    emit("Portfolio: ICHI 1% risk/trade, max 10, coin cap 20% (live jaisa) jab tak alag na likha ho.")
    emit("RND = wohi exit, magar RANDOM entries (har coin mein utni hi, same din) x" + str(RND_SEEDS) +
         " - agar RND win/PF bhi utna hi ho to faida entry ka nahi, sirf exit ka hai.")

    V = {"BASE": {"tpR": 3.0}}
    for x in (0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12):
        V[f"TP{int(x*100)}"] = {"tp": x}
    for r in (0.25, 0.33, 0.5, 0.75, 1.0):
        V[f"TPR{r}"] = {"tpR": r}
    for b in (0.015, 0.02, 0.03, 0.04):
        V[f"BE{b*100:g}"] = {"tpR": 3.0, "be": b}
    for b, y in ((0.02, 0.05), (0.02, 0.08), (0.03, 0.08), (0.03, 0.10), (0.04, 0.12)):
        V[f"BE{b*100:g}_TP{int(y*100)}"] = {"be": b, "tp": y}
    for x in (0.03, 0.05, 0.08):
        V[f"P{int(x*100)}BE"] = {"tpR": 3.0, "part": (x, 0.5)}
    for m in (3.0, 4.0):
        for y in (0.05, 0.08):
            V[f"CE{m:g}_TP{int(y*100)}"] = {"ce": m, "tp": y}
    for nb, x in ((6, 0.02), (12, 0.03), (18, 0.05)):
        V[f"TS{nb}_{int(x*100)}"] = {"tpR": 3.0, "ts": (nb, x)}

    data = prep_ichi(h4, allowed, start)
    emit(f"\nICHI signals: {sum(len(x['idx']) for x in data)}")
    emit("\n" + "#" * 110)
    emit("# 1) ICHI - har variant (portfolio 1% risk, max 10)")
    emit("#" * 110)
    emit(f"{'Variant':>12} | {'n':>4} | {'win%':>5} | {'brbr%':>5} | {'PF':>5} | {'avg%':>6} | {'OOS':>5} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'+mah%':>5}")
    res, port = {}, {}
    for name, cfg in V.items():
        tr = run_ichi(data, cfg)
        res[name] = tr
        s = tstats(tr)
        eq, _ = portfolio(tr, closes, "risk", 0.01)
        p = stats(eq)
        port[name] = p
        emit(f"{name:>12} | {s['n']:>4} | {s['win']:>5.1f} | {s['scr']:>5.1f} | {s['pf']:>5.2f} | {s['avg']:>+6.2f} | {s['oos']:>5.2f} | "
             f"{p['cagr']*100:>+6.1f}% | {p['dd']*100:>6.1f}% | {p['sharpe']:>6.2f} | {p['pos_months']:>5.0f}")

    # ---- shortlist: base + top Sharpe + top win rate with PF>1.5
    by_sh = sorted([k for k in V if k != "BASE"], key=lambda k: -port[k]["sharpe"])[:6]
    by_win = sorted([k for k in V if k != "BASE" and tstats(res[k])["pf"] > 1.5], key=lambda k: -tstats(res[k])["win"])[:3]
    short = ["BASE"] + list(dict.fromkeys(by_sh + by_win))
    emit("\nShortlist (Sharpe top-6 + win-rate top-3 jin ka PF > 1.5): " + ", ".join(short))

    # ---- 2) random control
    emit("\n" + "#" * 110)
    emit("# 2) RANDOM-ENTRY control (wohi exit, random entries) - asli edge entry ka hai ya exit ka?")
    emit("#" * 110)
    emit(f"{'Variant':>12} | {'win%':>5} | {'RND win p50':>11} | {'PF':>5} | {'RND PF p50':>10} | {'RND PF p95':>10} | {'PF > p95?':>9}")
    rnd = {k: [] for k in short}
    for seed in range(RND_SEEDS):
        rd = prep_ichi(h4, allowed, start, rng=np.random.default_rng(seed))
        for k in short:
            tr = run_ichi(rd, V[k])
            r = [t["ret"] for t in tr]
            rnd[k].append((pf_of(r), np.mean(np.array(r) > WIN_MIN) * 100))
    for k in short:
        s = tstats(res[k])
        pfs = np.array([a for a, _ in rnd[k]])
        wins = np.array([b for _, b in rnd[k]])
        emit(f"{k:>12} | {s['win']:>5.1f} | {np.median(wins):>11.1f} | {s['pf']:>5.2f} | {np.median(pfs):>10.2f} | "
             f"{np.percentile(pfs, 95):>10.2f} | {'HAAN' if s['pf'] > np.percentile(pfs, 95) else 'NAHI':>9}")

    # ---- 3) robustness
    emit("\n" + "#" * 110)
    emit("# 3) MAZBOOTI - saal-war PF, 2x kharcha, bootstrap p5, top-10 trades hata kar")
    emit("#" * 110)
    yrs = list(range(2021, 2027))
    emit(f"{'Variant':>12} | " + " | ".join(f"{y:>5}" for y in yrs) + f" | {'2x cost':>7} | {'boot p5':>7} | {'-top10':>6}")
    rng = np.random.default_rng(7)
    for k in short:
        tr = res[k]
        yr = []
        for y in yrs:
            r = [t["ret"] for t in tr if pd.Timestamp(t["t_in"]).year == y]
            yr.append(pf_of(r) if len(r) >= 5 else np.nan)
        c2 = tstats(run_ichi(data, V[k], cost=2.0))["pf"]
        r = np.array([t["ret"] for t in tr])
        boots = [pf_of(rng.choice(r, len(r))) for _ in range(2000)]
        top = np.sort(r)[:-10]
        emit(f"{k:>12} | " + " | ".join(f"{v:>5.2f}" for v in yr) + f" | {c2:>7.2f} | {np.percentile(boots, 5):>7.2f} | {pf_of(top):>6.2f}")

    # ---- 4) sizing
    emit("\n" + "#" * 110)
    emit("# 4) SIZE - chhote TP ka nafa kam hota hai: kya risk barha kar (same win rate) CAGR wapas aata hai?")
    emit("#" * 110)
    emit(f"{'Variant':>12} | {'risk':>5} | {'max':>3} | {'cap':>4} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'bura mahina':>11} | {'+mah%':>5}")
    for k in short:
        for risk, mp, cap in ((0.01, 10, 0.2), (0.015, 10, 0.2), (0.02, 10, 0.2), (0.02, 15, 0.3), (0.03, 15, 0.3)):
            eq, _ = portfolio(res[k], closes, "risk", risk, max_pos=mp, cap=cap)
            p = stats(eq)
            emit(f"{k:>12} | {risk*100:>4.1f}% | {mp:>3} | {cap*100:>3.0f}% | {p['cagr']*100:>+6.1f}% | {p['dd']*100:>6.1f}% | "
                 f"{p['sharpe']:>6.2f} | {p['worst_month']*100:>10.1f}% | {p['pos_months']:>5.0f}")

    # ---- 5) DIP win-rate variants
    emit("\n" + "#" * 110)
    emit("# 5) DIP Daily - win rate ke tareeqe (portfolio 20% size, max 10)")
    emit("#" * 110)
    emit(f"{'Variant':>18} | {'n':>4} | {'win%':>5} | {'brbr%':>5} | {'PF':>5} | {'avg%':>6} | {'OOS':>5} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6}")
    for rlim in (5, 7, 10):
        for sma in (3, 5):
            for tpx in (None, 0.03, 0.05):
                out = []
                for sym, d in daily.items():
                    c = d["close"]
                    e50, e200 = ema(c, 50), ema(c, 200)
                    trend = ((c > e200) & (e50 > e200)).to_numpy() & btc_ok.reindex(d["timestamp"]).fillna(False).to_numpy(bool)
                    trend[:200] = False
                    r = rsi(c, 3).to_numpy()
                    sig = trend & (r < rlim)
                    stop = (c - 3.0 * atr_w(d)).to_numpy()
                    exs = (c > c.rolling(sma).mean()).to_numpy()
                    o, hh, l, cc = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
                    ts = d["timestamp"].to_numpy()
                    for i in np.where(sig)[0]:
                        if sym not in allowed.get(d["timestamp"].iloc[i], ()) or d["timestamp"].iloc[i] < start:
                            continue
                        t = sim(o, hh, l, cc, ts, i, stop[i], None, exs, 10, {"tp": tpx} if tpx else {})
                        if t:
                            t.update(sym=sym, prio=-r[i])
                            out.append(t)
                s = tstats(out)
                eq, _ = portfolio(out, closes, "fixed", 0.20)
                p = stats(eq)
                nm = f"RSI<{rlim} SMA{sma}" + (f" TP{int(tpx*100)}" if tpx else "")
                emit(f"{nm:>18} | {s['n']:>4} | {s['win']:>5.1f} | {s['scr']:>5.1f} | {s['pf']:>5.2f} | {s['avg']:>+6.2f} | {s['oos']:>5.2f} | "
                     f"{p['cagr']*100:>+6.1f}% | {p['dd']*100:>6.1f}% | {p['sharpe']:>6.2f}")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
