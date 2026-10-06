"""
TIMING LAB 8 - "kab khareedna hai" ko tarjeeh (user, 2026-10-06: "mazeed naye system dhoondo jo kab khareeda ko tarjeeh de")
==========================================================================================================================
Sirf ENTRY ka edge naapna asal maqsad hai. Is liye har naye entry khayal ko do tarah parkha jata hai:
  (1) SAAF WAQT-PAIMANA: khareed ke baad 5 din (fixed, koi SL/TP nahi) ka ausat nafa vs RANDOM din (wohi coins, wohi
      shart-pool) ka ausat -> "entry ka asal faida" (fark) aur 10 mein se kitni baar 5 din baad ooper.
  (2) DIP EXIT ke sath poori trade (TP 5% / close > SMA3 / SL 3 ATR / 10 din) - jo pehle se kaamyab exit hai.
Sab daily, point-in-time top-100, coin uptrend (close > EMA200). Har khayal ki 3 settings (beech wali asal, 2 padosi).

  A HAMMER     : lambi neechi dum wali candle - din ka low pichle close se 5%+ neeche gaya magar candle ki range ka
                 L hissa neechi dum (L = 0.5 / 0.6 / 0.7) - kharidar wapas aa gaye. BTC > EMA50.
  B RS_SELLOFF : BTC 3 din mein X% gira (X = 3 / 5 / 7%) magar coin ka 3 din return musbat (market girne par bhi mazboot).
                 BTC > EMA200 (bull market ki girawat).
  C DIP_CONFIRM: pichle 3 din mein RSI3 < R hua (R = 7 / 10 / 15) aur AAJ close > kal ka high (ulatne ki tasdeeq) - pehli dafa.
                 EMA50 > EMA200, BTC > EMA50.
  D LOWVOL_PULL: EMA20 > EMA50 > EMA200, 3 din lagataar neeche close, 3 din mein 5%+ girawat, aur in 3 din ka ausat
                 volume < V x 20-din ausat (V = 0.6 / 0.8 / 1.0) - bechne wale thak gaye.
  E BTC_DIP_LEAD: BTC ka RSI3 < Q (Q = 7 / 10 / 15) aur BTC > EMA200 -> us din ke top-5 momentum (60 din) uptrend coins.
  REF DIP      : live Dip v2 (RSI3 < 7) - muqable ke liye.
SACHAI test andar (30 din data kaat kar). PASS (beech wali setting, DIP exit):
  jeet >= 60% | PF > random p95 | OOS >= 1.3 | 4/4 folds | boot p5 >= 1.2 | top-10 hata kar >= 1.2 | 2x kharcha >= 1.2 |
  >= 0.3 / hafta | 5-din fark > 0 aur random p95 se ooper | padosi 1 PF > random p95 | DIP+RESID khate mein milane se Sharpe barhe.
Natija: timing_lab8_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema, rsi, atr_w, portfolio, stats
from winrate_lab import sim, tstats, pf_of
import strategy_lab5 as L5

OUT = "timing_lab8_RESULTS.txt"
TOP_FETCH = 200
DAYS = 2400
RND = 10
GRID = {
    "A_HAMMER": [0.5, 0.6, 0.7],
    "B_RS_SELLOFF": [0.03, 0.05, 0.07],
    "C_DIP_CONFIRM": [7, 10, 15],
    "D_LOWVOL_PULL": [0.6, 0.8, 1.0],
    "E_BTC_DIP_LEAD": [7, 10, 15],
}
MID = 1


def names():
    out = ["REF_DIP"]
    for f, g in GRID.items():
        out += [f"{f} {v}" for v in g]
    return out


def build(daily, start):
    btc_ok, al100 = L5.context(daily)
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc200 = bd > ema(bd, 200)
    btc_r3 = bd / bd.shift(3) - 1
    btc_rsi = rsi(bd, 3)
    P = {}
    mom = {}
    for sym, d in daily.items():
        c, o, h, l, v = d["close"], d["open"], d["high"], d["low"], d["volume"]
        ts = d["timestamp"]
        e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
        n_ok = np.arange(len(d)) >= 200
        ok = np.array([sym in al100.get(t, ()) for t in ts]) & (ts >= start).to_numpy()
        up = (c > e200).to_numpy() & n_ok
        bok = btc_ok.reindex(ts).fillna(False).to_numpy(bool)
        b200 = btc200.reindex(ts).fillna(False).to_numpy(bool)
        pool = ok & up                      # random-control pool: wohi coins, uptrend din
        r3 = rsi(c, 3)
        sig = {"REF_DIP": pool & bok & (e50 > e200).to_numpy() & (r3 < 7).to_numpy()}
        # A hammer
        rng_ = (h - l).replace(0, np.nan)
        lw = (np.minimum(o, c) - l) / rng_
        dropped = (l / c.shift(1) - 1) <= -0.05
        for L in GRID["A_HAMMER"]:
            sig[f"A_HAMMER {L}"] = pool & bok & (dropped & (lw >= L)).fillna(False).to_numpy()
        # B RS during BTC selloff
        cr3 = (c / c.shift(3) - 1)
        br3 = btc_r3.reindex(ts).to_numpy()
        for X in GRID["B_RS_SELLOFF"]:
            sig[f"B_RS_SELLOFF {X}"] = pool & b200 & (np.nan_to_num(br3, nan=0) <= -X) & (cr3 > 0).fillna(False).to_numpy()
        # C dip confirm (pehli dafa)
        for R in GRID["C_DIP_CONFIRM"]:
            recent = (r3 < R).rolling(3, min_periods=1).max().astype(bool)
            cond = recent & (c > h.shift(1)) & (e50 > e200)
            cond = cond & ~cond.shift(1, fill_value=False) & ~cond.shift(2, fill_value=False)
            sig[f"C_DIP_CONFIRM {R}"] = pool & bok & cond.fillna(False).to_numpy()
        # D low-volume pullback
        down3 = (c < c.shift(1)) & (c.shift(1) < c.shift(2)) & (c.shift(2) < c.shift(3))
        dec = (c / c.shift(3) - 1) <= -0.05
        vavg = v.rolling(20).mean().shift(3)
        v3 = v.rolling(3).mean()
        stack = (e20 > e50) & (e50 > e200)
        for V in GRID["D_LOWVOL_PULL"]:
            cond = down3 & dec & (v3 < V * vavg) & stack
            sig[f"D_LOWVOL_PULL {V}"] = pool & bok & cond.fillna(False).to_numpy()
        mom[sym] = pd.Series(np.where(pool & b200, (c / c.shift(60) - 1).to_numpy(), np.nan), index=ts)
        P[sym] = dict(sym=sym, ts=ts.to_numpy(), o=o.to_numpy(float), h=h.to_numpy(float), l=l.to_numpy(float),
                      c=c.to_numpy(float), atr=atr_w(d).to_numpy(), sma3=(c > c.rolling(3).mean()).to_numpy(),
                      pool=pool, sig=sig, prio=-r3.to_numpy(), d_ts=ts)
    # E: BTC dip -> top-5 momentum coins
    M = pd.DataFrame(mom)
    rk = M.rank(axis=1, ascending=False)
    brs = btc_rsi
    for sym, x in P.items():
        rks = rk[sym].reindex(x["d_ts"]).to_numpy() if sym in rk else np.full(len(x["c"]), np.nan)
        bq = brs.reindex(x["d_ts"]).to_numpy()
        for Q in GRID["E_BTC_DIP_LEAD"]:
            x["sig"][f"E_BTC_DIP_LEAD {Q}"] = x["pool"] & (np.nan_to_num(rks, nan=1e9) <= 5) & (np.nan_to_num(bq, nan=100) < Q)
    return P


def run(P, name, mode="dip", cost=1.0, rng=None):
    out = []
    for sym, x in P.items():
        idx = np.where(x["sig"][name])[0]
        if rng is not None:
            pool = np.where(x["pool"] & np.isfinite(x["atr"]))[0]
            if len(idx) == 0 or len(pool) == 0:
                continue
            idx = np.sort(rng.choice(pool, size=min(len(idx), len(pool)), replace=False))
        for i in idx:
            if mode == "dip":
                st = x["c"][i] - 3.0 * x["atr"][i]
                if not np.isfinite(st) or st <= 0:
                    continue
                t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, None, x["sma3"], 10, {"tp": 0.05}, cost)
            else:   # saaf 5-din: koi SL/TP nahi
                t = sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, 1e-12, None, None, 5, {}, cost)
            if t:
                pr = x["prio"][i]
                t.update(sym=sym, prio=pr if np.isfinite(pr) else -9)
                out.append(t)
    return out


def causality(daily, start, emit, n_days=30):
    rng = np.random.default_rng(13)
    full = build(daily, start)
    days = [d for d in daily["BTC/USDT"]["timestamp"] if d >= start + pd.Timedelta(days=30)]
    pick = sorted(rng.choice(len(days), size=min(n_days, len(days)), replace=False))
    bad = {n: 0 for n in names()}
    nsig = {n: 0 for n in names()}
    for k, j in enumerate(pick, 1):
        D = days[j]
        cut = {s: d[d["timestamp"] <= D].reset_index(drop=True) for s, d in daily.items()}
        cut = {s: d for s, d in cut.items() if len(d) > 0}
        part = build(cut, start)
        for sym, x in full.items():
            if sym not in part:
                continue
            w = np.where(x["ts"] == np.datetime64(D))[0]
            if len(w) == 0:
                continue
            i = w[0]
            for n in bad:
                a, b = bool(x["sig"][n][i]), bool(part[sym]["sig"][n][-1])
                bad[n] += a != b
                nsig[n] += a or b
        if k % 10 == 0:
            print(f"  sachai {k}/{len(pick)}", flush=True)
    emit(f"\n# SACHAI TEST - {len(pick)} random din, data kaat kar (farq 0 hona chahiye)")
    for n in bad:
        emit(f"{n:>20}: farq {bad[n]} ({nsig[n]} signals) -> {'SAHI' if bad[n] == 0 else 'GHALAT - lookahead'}")
    return {n: bad[n] == 0 for n in bad}


def folds(tr, t0, t1):
    e = pd.date_range(t0, t1, periods=5)
    return [pf_of([t["ret"] for t in tr if a <= pd.Timestamp(t["t_in"]) < b]) for a, b in zip(e[:-1], e[1:])]


def boot_p5(tr, n=2000):
    r = np.array([t["ret"] for t in tr])
    g = np.random.default_rng(3)
    return np.percentile([pf_of(g.choice(r, len(r))) for _ in range(n)], 5)


def main(daily=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    try:
        if daily is None:
            import data_fetcher
            data_fetcher.AUTO_TOP_N_COINS = TOP_FETCH
            ex = data_fetcher.get_exchange()
            coins = [s for s in data_fetcher.get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:TOP_FETCH]
            if "BTC/USDT" not in coins:
                coins.insert(0, "BTC/USDT")
            daily = {}
            for k, sym in enumerate(coins, 1):
                try:
                    df = fetch_full(ex, sym, "1d", DAYS)
                    if df is not None and len(df) >= 250:
                        daily[sym] = norm(df)
                        print(f"[{k}/{len(coins)}] {sym}: {len(df)}", flush=True)
                except Exception as e:
                    print(f"[{k}] {sym}: SKIP ({e})", flush=True)

        closes = pd.DataFrame({s: d.set_index("timestamp")["close"] for s, d in daily.items()}).sort_index().ffill()
        start = closes.index[0] + pd.Timedelta(days=210)
        closes = closes[closes.index >= start]
        t0, t1 = closes.index[0], closes.index[-1]
        weeks = (t1 - t0).days / 7
        emit("=" * 140)
        emit("TIMING LAB 8 - naye ENTRY khayal: (1) saaf 5-din waqt-paimana vs random  (2) Dip exit ke sath poori trade")
        emit("=" * 140)
        emit(f"Coins: {len(daily)} | {t0.date()} -> {t1.date()} ({weeks:.0f} hafte) | top-100, coin uptrend | OOS = 2025+")
        emit("jeet/10 = 10 trades mein se nafa wali | PF = har 1 rupay nuqsan par kitna nafa | 5d fark = ausat 5-din nafa minus random ka")

        ok_c = causality(daily, start, emit)
        P = build(daily, start)

        rows = {}
        emit(f"\n{'entry':>20} | {'n':>4} | {'/hafta':>6} | {'5d avg':>7} | {'5d fark':>7} | {'rnd fark95':>10} | {'5d ooper/10':>11} || "
             f"{'jeet/10':>7} | {'PF':>5} | {'rnd95':>5} | {'OOS':>5} | {'folds':>23} | {'boot5':>5} | {'-top10':>6} | {'2xcost':>6}")
        for nm in names():
            h5 = run(P, nm, mode="h5")
            if len(h5) < 25:
                emit(f"{nm:>20} | sirf {len(h5)} signals")
                continue
            r5 = np.array([t["ret"] for t in h5])
            rnd5 = [np.mean([t["ret"] for t in run(P, nm, mode="h5", rng=np.random.default_rng(700 + q))]) for q in range(RND)]
            base5 = np.mean(rnd5)
            # random ka apna fark (random - random ausat) ka p95 = shor ki had
            fark95 = np.percentile(np.array(rnd5) - base5, 95)
            tr = run(P, nm)
            s = tstats(tr)
            rnd = [pf_of([t["ret"] for t in run(P, nm, rng=np.random.default_rng(800 + q))]) for q in range(RND)]
            fo = folds(tr, t0, t1)
            bp = boot_p5(tr)
            mt = pf_of([t["ret"] for t in sorted(tr, key=lambda t: -t["ret"])[10:]])
            c2 = pf_of([t["ret"] for t in run(P, nm, cost=2.0)])
            rows[nm] = dict(s=s, tr=tr, r95=np.percentile(rnd, 95), fo=fo, bp=bp, mt=mt, c2=c2, wk=s["n"] / weeks,
                            fark=r5.mean() - base5, fark95=fark95)
            emit(f"{nm:>20} | {s['n']:>4} | {s['n']/weeks:>6.2f} | {r5.mean()*100:>+6.2f}% | {(r5.mean()-base5)*100:>+6.2f}% | "
                 f"{fark95*100:>+9.2f}% | {(r5 > 0).mean()*10:>11.1f} || {s['win']/10:>7.1f} | {s['pf']:>5.2f} | "
                 f"{np.percentile(rnd,95):>5.2f} | {s['oos']:>5.2f} | {' '.join(f'{f:>5.2f}' for f in fo):>23} | {bp:>5.2f} | "
                 f"{mt:>6.2f} | {c2:>6.2f}")

        # khata: DIP + RESID (combo lab 6 behtareen) vs + NEW
        L5.GRID["R1_RESID"] = [("z-2.0", -2.0)]
        btc_ok, al100 = L5.context(daily)
        P5 = L5.build(daily, btc_ok, al100, start)
        book = L5.run(P5, "REF_DIP") + L5.run(P5, "R1_RESID z-2.0")
        bstat = stats(portfolio(book, closes, "fixed", 0.20, max_pos=10, cap=1.0)[0])
        emit(f"\n# KHATA: DIP+RESID (20% / trade, max 10): CAGR {bstat['cagr']*100:+.1f}% | DD {bstat['dd']*100:.1f}% | "
             f"Sharpe {bstat['sharpe']:.2f} | +mah {bstat['pos_months']:.0f}% | ausat mah {bstat['monthly'].mean()*100:+.2f}%")
        port = {}
        for nm, r in rows.items():
            if nm == "REF_DIP":
                continue
            seen = {(t["sym"], pd.Timestamp(t["t_in"])) for t in book}
            extra = [t for t in r["tr"] if (t["sym"], pd.Timestamp(t["t_in"])) not in seen]
            p = stats(portfolio(book + extra, closes, "fixed", 0.20, max_pos=10, cap=1.0)[0])
            port[nm] = p["sharpe"] > bstat["sharpe"]
            emit(f"{'+ ' + nm:>22} ({len(extra):>4} naye): CAGR {p['cagr']*100:+.1f}% | DD {p['dd']*100:.1f}% | Sharpe {p['sharpe']:.2f} | "
                 f"+mah {p['pos_months']:.0f}% | ausat mah {p['monthly'].mean()*100:+.2f}%")

        emit("\n# PASS / FAIL (beech wali setting)")
        for fam, g in GRID.items():
            mid = f"{fam} {g[MID]}"
            nb = [f"{fam} {v}" for k, v in enumerate(g) if k != MID]
            if mid not in rows:
                emit(f"{mid:>22}: FAIL (kam signals)")
                continue
            r, s = rows[mid], rows[mid]["s"]
            chk = {"sachai": all(ok_c.get(f"{fam} {v}", False) for v in g), "jeet>=60": s["win"] >= 60,
                   "PF>rnd95": s["pf"] > r["r95"], "OOS>=1.3": s["oos"] >= 1.3,
                   "folds": all(np.isfinite(f) and f > 1 for f in r["fo"]), "boot>=1.2": r["bp"] >= 1.2,
                   "-top10>=1.2": r["mt"] >= 1.2, "2xcost>=1.2": r["c2"] >= 1.2, ">=0.3/hafta": r["wk"] >= 0.3,
                   "5d fark": r["fark"] > max(0, r["fark95"]),
                   "padosi": any(n in rows and rows[n]["s"]["pf"] > rows[n]["r95"] for n in nb),
                   "khata+": port.get(mid, False)}
            fails = [k for k, v in chk.items() if not v]
            emit(f"{mid:>22}: {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    except Exception:
        emit("\n!! GHALTI (code crash):\n" + traceback.format_exc())

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
