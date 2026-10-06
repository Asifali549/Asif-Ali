"""
STREAK ATTACK - "4 din ki girawat" ko MUKHALIF (contrarian / shakki) nazar se torne ki koshish (user, 2026-10-06)
==============================================================================================================
Har sawal ka maqsad: kya ye nafa asal mein kisi aur cheez ka hai?
 1) SAKHT RANDOM: utni hi girawat (4-din nafa signal jaisa) magar LAGATAAR nahi - kya "lagataar" ka koi faida hai?
 2) DIP v2 KA SAYA: kitni trades Dip v2 ke signal ke usi din ya pichle 3 din mein? baqi (alag) trades akeli kaisi?
 3) EK DIN DER: entry ek din baad ho to edge bachta hai ya ghayab (nazuk)?
 4) TAALA ki 46 trades: kitne alag din (sab ek crash ke din to ek hi sharat), top-5 trades hata kar PF.
 5) COINS: 20% coins random hata kar x8 (poora + taala PF); top coins ka hissa.
 6) KHARCHA 3x; EXIT ka hissa: TP 5% hata kar / SMA3 hata kar / sirf 5 din.
 7) SAAL-WAR PF + BTC filter hata kar (kya sirf bull market ka kamal hai?).
Natija: streak_attack_RESULTS.txt
"""
import traceback

import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema
from winrate_lab import sim, tstats, pf_of
import search_lab9 as S9
import search_lab12 as L

OUT = "streak_attack_RESULTS.txt"
NM = "P_DOWN_STREAK 4"


def trade(x, i, cost=1.0, tp=0.05, sma=True, hold=10, stop=True):
    st = x["c"][i] - 3.0 * x["atr"][i] if stop else 1e-12
    if not np.isfinite(st) or st <= 0:
        return None
    return sim(x["o"], x["h"], x["l"], x["c"], x["ts"], i, st, None, x["sma3"] if sma else None, hold,
               {"tp": tp} if tp else {}, cost)


def collect(P, idx_map, period=None, **kw):
    out = []
    for sym, idx in idx_map.items():
        x = P[sym]
        for i in idx:
            tsi = pd.Timestamp(x["ts"][i])
            if (period == "dev" and tsi >= L.HOLDOUT) or (period == "hold" and tsi < L.HOLDOUT):
                continue
            t = trade(x, i, **kw)
            if t:
                t.update(sym=sym, i=i)
                out.append(t)
    return out


def line(name, tr):
    if len(tr) < 5:
        return f"{name:>44}: sirf {len(tr)} trades"
    s = tstats(tr)
    return f"{name:>44}: {s['n']:>4} trades | jeet/10 {s['win']/10:.1f} | PF {s['pf']:.2f} | ausat {s['avg']:+.2f}%"


def main(daily=None):
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(x)

    try:
        if daily is None:
            import data_fetcher
            data_fetcher.AUTO_TOP_N_COINS = S9.TOP_FETCH
            ex = data_fetcher.get_exchange()
            coins = [s for s in data_fetcher.get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:S9.TOP_FETCH]
            if "BTC/USDT" not in coins:
                coins.insert(0, "BTC/USDT")
            daily = {}
            for k, sym in enumerate(coins, 1):
                try:
                    df = fetch_full(ex, sym, "1d", S9.DAYS)
                    if df is not None and len(df) >= 250:
                        daily[sym] = norm(df)
                except Exception as e:
                    print(f"[{k}] {sym}: SKIP ({e})", flush=True)
        idx_all = pd.DatetimeIndex(sorted(set().union(*[set(d["timestamp"]) for d in daily.values()])))
        start = idx_all[0] + pd.Timedelta(days=210)
        P = L.build(daily, start)
        sig = {s: np.where(x["sig"][NM])[0] for s, x in P.items()}
        base_all, base_hold = collect(P, sig), collect(P, sig, period="hold")
        emit("=" * 120)
        emit(f"STREAK ATTACK - 4 din ki girawat ko torne ki koshish | coins {len(daily)} | taala {L.HOLDOUT.date()} se")
        emit("=" * 120)
        emit(line("ASAL (poora)", base_all))
        emit(line("ASAL (taala)", base_hold))

        # extra arrays
        ex_ = {}
        for s, x in P.items():
            c = pd.Series(x["c"])
            r4 = (c / c.shift(4) - 1).to_numpy()
            down = (c < c.shift(1)).astype(float)
            streak = (down.rolling(4, min_periods=4).sum() >= 4).to_numpy()
            gold = (ema(c, 50) > ema(c, 200)).to_numpy()
            ex_[s] = dict(r4=r4, streak=streak, gold=gold)

        # 1) sakht random
        emit("\n# 1) SAKHT RANDOM - utni hi girawat (4-din), golden + pool, magar 4 din LAGATAAR nahi (10 dafa)")
        sig_r4 = np.array([ex_[s]["r4"][i] for s, idx in sig.items() for i in idx], dtype=float)
        lo, hi = np.nanpercentile(sig_r4, 10), np.nanpercentile(sig_r4, 90)
        emit(f"signal ka 4-din nafa: median {np.nanmedian(sig_r4)*100:.1f}% (beech ke 80%: {lo*100:.1f}% se {hi*100:.1f}%)")
        for per in (None, "hold"):
            pfs, wins = [], []
            for q in range(10):
                g = np.random.default_rng(500 + q)
                m = {}
                for s, x in P.items():
                    e = ex_[s]
                    pool = np.where(x["pool"] & e["gold"] & ~e["streak"] & (e["r4"] >= lo) & (e["r4"] <= hi) & np.isfinite(x["atr"]))[0]
                    k = len(sig[s])
                    if k and len(pool):
                        m[s] = np.sort(g.choice(pool, size=min(k, len(pool)), replace=False))
                tr = collect(P, m, period=per)
                pfs.append(pf_of([t["ret"] for t in tr]))
                wins.append(tstats(tr)["win"] / 10 if tr else np.nan)
            asal = tstats(base_all if per is None else base_hold)
            emit(f"{'poora' if per is None else 'taala':>6}: ASAL PF {asal['pf']:.2f} jeet {asal['win']/10:.1f} | sakht random PF p50 "
                 f"{np.nanmedian(pfs):.2f} p95 {np.nanpercentile(pfs,95):.2f} jeet {np.nanmedian(wins):.1f} -> "
                 f"{'lagataar ka FAIDA' if asal['pf'] > np.nanpercentile(pfs, 95) else 'lagataar ka koi khaas faida NAHI'}")

        # 2) dip saya
        emit("\n# 2) DIP v2 KA SAYA - kya ye Dip hi hai?")
        dip = {s: set(np.where(x["sig"]["REF_DIP"])[0]) for s, x in P.items()}
        near = lambda t: any(0 <= t["i"] - j <= 3 for j in dip[t["sym"]])   # Dip signal usi din ya pichle 3 din mein
        for per, tr in (("poora", base_all), ("taala", base_hold)):
            a = [t for t in tr if near(t)]
            b = [t for t in tr if not near(t)]
            emit(f"{per}: Dip ke qareeb {len(a)} / {len(tr)} ({len(a)/max(len(tr),1)*100:.0f}%)")
            emit(line(f"{per} - Dip se ALAG trades", b))
            emit(line(f"{per} - Dip ke qareeb trades", a))

        # 3) ek din der
        emit("\n# 3) EK DIN DER SE ENTRY (nazuk to nahi?)")
        late = {s: np.array([i + 1 for i in idx if i + 1 < len(P[s]["c"])], dtype=int) for s, idx in sig.items()}
        emit(line("1 din der (poora)", collect(P, late)))
        emit(line("1 din der (taala)", collect(P, late, period="hold")))
        late2 = {s: np.array([i + 2 for i in idx if i + 2 < len(P[s]["c"])], dtype=int) for s, idx in sig.items()}
        emit(line("2 din der (poora)", collect(P, late2)))

        # 4) taala ki trades
        emit("\n# 4) TAALA KI TRADES - kitne alag din / coins, top-5 hata kar")
        days = pd.Series([str(pd.Timestamp(t["t_in"]).date()) for t in base_hold])
        emit(f"{len(base_hold)} trades | {days.nunique()} alag din | {len(set(t['sym'] for t in base_hold))} alag coins | "
             f"sab se bhare din: " + ", ".join(f"{d} ({n})" for d, n in days.value_counts().head(5).items()))
        srt = sorted(base_hold, key=lambda t: -t["ret"])
        emit(line("taala - top-5 hata kar", srt[5:]))
        emit(line("taala - top-10 hata kar", srt[10:]))
        bym = pd.Series([t["ret"] for t in base_hold], index=[pd.Timestamp(t["t_in"]).strftime("%Y-%m") for t in base_hold])
        emit("taala mahana: " + " | ".join(f"{m}: {len(g)}tr {g.sum()*100:+.0f}%" for m, g in bym.groupby(level=0)))

        # 5) coins
        emit("\n# 5) COINS - 20% coins random hata kar (8 dafa)")
        syms = [s for s in P if s != "BTC/USDT"]
        for q in range(8):
            g = np.random.default_rng(700 + q)
            drop = set(g.choice(syms, size=int(len(syms) * 0.2), replace=False))
            a = [t for t in base_all if t["sym"] not in drop]
            b = [t for t in base_hold if t["sym"] not in drop]
            emit(f"  {q+1}: poora PF {pf_of([t['ret'] for t in a]):.2f} ({len(a)}) | taala PF {pf_of([t['ret'] for t in b]):.2f} ({len(b)})")
        bys = pd.Series([t["ret"] for t in base_all], index=[t["sym"] for t in base_all]).groupby(level=0).sum().sort_values()
        tot = bys.sum()
        emit(f"coins: {len(bys)} | nafa wale {int((bys > 0).sum())} | nuqsan wale {int((bys < 0).sum())} | top-5 coins = kul nafa ka "
             f"{bys.tail(5).sum()/tot*100:.0f}% ({', '.join(bys.tail(5).index)})")

        # 6) kharcha / exit
        emit("\n# 6) KHARCHA aur EXIT ka hissa")
        emit(line("kharcha 3x (poora)", collect(P, sig, cost=3.0)))
        emit(line("kharcha 3x (taala)", collect(P, sig, period="hold", cost=3.0)))
        emit(line("TP 5% NAHI", collect(P, sig, tp=None)))
        emit(line("SMA3 exit NAHI", collect(P, sig, sma=False)))
        emit(line("sirf 5 din, koi SL/TP nahi", collect(P, sig, tp=None, sma=False, hold=5, stop=False)))
        emit(line("SL NAHI (sirf TP/SMA3/10 din)", collect(P, sig, stop=False)))

        # 7) saal-war + BTC filter
        emit("\n# 7) SAAL-WAR PF (asal) + BTC filter ke BAGHAIR")
        yr = pd.Series([t["ret"] for t in base_all], index=[pd.Timestamp(t["t_in"]).year for t in base_all])
        emit("saal: " + " | ".join(f"{y}: {len(g)}tr PF {pf_of(list(g)):.2f}" for y, g in yr.groupby(level=0)))
        import strategy_lab5 as L5
        btc_ok, _ = L5.context(daily)
        nob = {}
        for s, x in P.items():
            bok = btc_ok.reindex(pd.DatetimeIndex(x["ts"])).fillna(False).to_numpy(bool)
            e = ex_[s]
            # pool bina BTC: pool mein BTC shamil hai, is liye wapas bana nahi sakte -> sirf BTC < EMA50 wale din alag dekhne ke liye
            nob[s] = bok
        emit("(pool mein BTC > EMA50 pehle se shamil; BTC kamzor dinon ki jaanch ke liye niche wala hissa)")
        # BTC kamzor din: same streak signal magar BTC < EMA50 (pool ke bina hisaab)
        weak = {}
        for s, x in P.items():
            e = ex_[s]
            c = pd.Series(x["c"])
            up = (c > ema(c, 200)).to_numpy() & (np.arange(len(c)) >= 200)
            st4 = pd.Series(e["streak"])
            fresh = (st4 & ~st4.shift(1, fill_value=False)).to_numpy()
            weak[s] = np.where(fresh & up & e["gold"] & ~nob[s] & (pd.DatetimeIndex(x["ts"]) >= start) & np.isfinite(x["atr"]))[0]
        emit(line("BTC KAMZOR dinon ke signals (bot nahi leta)", collect(P, weak)))
    except Exception:
        emit("\n!! GHALTI:\n" + traceback.format_exc())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
