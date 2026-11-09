"""
DIP PAPER CHECK (2026-11-09, user: "Dip bot aur test mein farq hai to dhoondo aur theek karo")
==============================================================================================
Paper report: Dip v2 bot 15 trades, jeet 5.3/10 vs backtest 8.1/10 (binomial imkaan ~1.4%).
Ye script ASLI data par wohi muddat (2026-10-03 se) BACKTEST engine (dip_v2_validate.prep_dip logic +
winrate_lab.sim) se dobara chalata hai, do data raston se:
  A) BOT jaisa  : KuCoin 1d candles (400 din), liquidity dvol30 usi din tak (bot ka tareeqa)
  B) TEST jaisa : KuCoin 4h candles -> daily (to_daily), liquidity dvol30.shift(1) (backtest ka tareeqa)
Phir har raste ki trades ko dip_paper_trades.csv se milata hai (coin + entry din): entry/exit din aur nafa ka farq,
jo signal bot ne nahi liya, jo bot ne liya magar test mein nahi. Aakhir mein isi muddat ka test win/PF:
agar test bhi isi muddat mein ~5/10 deta hai to bot theek hai, muddat buri thi.
Natija: dip_paper_check_RESULTS.txt (koi bot / state file nahi badalti).
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, STABLES
from portfolio_lab import ema, rsi, atr_w, to_daily
from winrate_lab import sim

OUT = "dip_paper_check_RESULTS.txt"
START = pd.Timestamp("2026-10-03")
RSI_LO, SMA_N, TP, HOLD, UNIV = 7, 3, 0.05, 10, 100


def signals_and_trades(daily, shift_liq):
    bd = daily["BTC/USDT"].set_index("timestamp")["close"]
    btc_ok = bd > ema(bd, 50)
    dv = pd.DataFrame({s: d.set_index("timestamp")["close"] * d.set_index("timestamp")["volume"] for s, d in daily.items()})
    dv = dv.rolling(30, min_periods=20).mean()
    if shift_liq:
        dv = dv.shift(1)
    lens = {s: len(d) for s, d in daily.items()}
    allowed = {}
    for day, row in dv.iterrows():
        r = row.dropna()
        if not shift_liq:
            r = r[[s for s in r.index if lens[s] >= 60]]
        allowed[day] = set(r.nlargest(UNIV).index)
    trades = []
    for sym, d in daily.items():
        c = d["close"]
        e50, e200 = ema(c, 50), ema(c, 200)
        up = ((c > e200) & (e50 > e200)).to_numpy() & (np.arange(len(d)) >= 200)
        ts = d["timestamp"]
        ok = btc_ok.reindex(ts).fillna(False).to_numpy(bool) & np.array([sym in allowed.get(t, ()) for t in ts]) & (ts >= START).to_numpy()
        r = rsi(c, 3).to_numpy()
        stop = (c - 3.0 * atr_w(d)).to_numpy()
        ex = (c > c.rolling(SMA_N).mean()).to_numpy()
        o, h, l, cc = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        for i in np.where(up & ok & (r < RSI_LO))[0]:
            if not np.isfinite(stop[i]) or stop[i] >= cc[i]:
                continue
            t = sim(o, h, l, cc, ts.to_numpy(), i, stop[i], None, ex, HOLD, {"tp": TP}, 1.0)
            if t is None:
                continue
            still_open = pd.Timestamp(t["t_out"]) >= ts.iloc[-1] and (pd.Timestamp(t["t_out"]) - pd.Timestamp(t["t_in"])).days < HOLD
            trades.append(dict(symbol=sym, sig=str(pd.Timestamp(ts.iloc[i]).date()), entry_day=str(pd.Timestamp(t["t_in"]).date()),
                               exit_day=str(pd.Timestamp(t["t_out"]).date()), ret=t["ret"] * 100, open=still_open, rsi=r[i]))
    return pd.DataFrame(trades).sort_values(["entry_day", "symbol"]).reset_index(drop=True) if trades else pd.DataFrame()


def summ(rets):
    rets = np.asarray(rets, float)
    if len(rets) == 0:
        return "0 trades"
    w = (rets > 0.5).mean() * 10
    g, lo = rets[rets > 0].sum(), -rets[rets < 0].sum()
    pf = g / lo if lo > 0 else float("inf")
    return f"{len(rets)} trades | jeet/10 {w:.1f} | PF {pf:.2f} | ausat {rets.mean():+.2f}%"


def main():
    from data_fetcher import get_exchange, get_coin_list
    lines = []

    def emit(x=""):
        print(x, flush=True)
        lines.append(str(x))

    ex = get_exchange()
    coins = [c for c in get_coin_list(ex) if c.split("/")[0].upper() not in STABLES][:150]
    paper = pd.read_csv("dip_paper_trades.csv")
    for s in set(paper["symbol"]) | {"BTC/USDT"}:
        if s not in coins:
            coins.append(s)
    d1, d4 = {}, {}
    for k, sym in enumerate(coins, 1):
        try:
            a = fetch_full(ex, sym, "1d", 400)
            if a is not None and len(a) >= 30:
                d1[sym] = norm(a).iloc[:-1].reset_index(drop=True)      # aakhri (adhoori) din hatao
            b = fetch_full(ex, sym, "4h", 2400)
            if b is not None and len(b) >= 180:
                d4[sym] = to_daily(norm(b)).iloc[:-1].reset_index(drop=True)
            print(f"[{k}/{len(coins)}] {sym}")
        except Exception as e:
            print(f"[{k}] {sym}: SKIP ({e})")

    emit("DIP v2 PAPER CHECK - bot ki asli trades vs backtest engine, wohi muddat (2026-10-03 se), asli KuCoin data")
    emit("=" * 110)
    emit(f"PAPER (bot): {summ(paper['ret_pct'])}")
    for name, daily, sh in (("A BOT-jaisa data (1d)", d1, False), ("B TEST-jaisa data (4h->daily)", d4, True)):
        T = signals_and_trades(daily, sh)
        emit("")
        emit(f"### {name}")
        if T.empty:
            emit("koi trade nahi")
            continue
        closed = T[~T["open"]]
        emit(f"TEST isi muddat mein (sab signals, slot had nahi): {summ(closed['ret'])}  (+{int(T['open'].sum())} khuli)")
        m = paper.merge(T, on=["symbol", "entry_day"], how="outer", suffixes=("_paper", "_test"), indicator=True)
        both = m[m["_merge"] == "both"]
        emit(f"Dono mein: {len(both)} | sirf paper: {int((m['_merge'] == 'left_only').sum())} | sirf test: {int((m['_merge'] == 'right_only').sum())}")
        if len(both):
            emit(f"Mushtarka trades - paper {summ(both['ret_pct'])}")
            emit(f"Mushtarka trades - test  {summ(both['ret'])}")
        emit(f"{'coin':>14} | {'entry':>10} | {'exit P':>10} | {'exit T':>10} | {'nafa P':>7} | {'nafa T':>7} | haal")
        for _, r in m.sort_values("entry_day").iterrows():
            tag = {"both": "dono", "left_only": "SIRF PAPER", "right_only": "SIRF TEST"}[r["_merge"]]
            same = ""
            if r["_merge"] == "both":
                same = "OK" if (str(r["exit_day_paper"]) == str(r["exit_day_test"]) and abs(r["ret_pct"] - r["ret"]) < 0.3) else "FARQ"
            if r["_merge"] == "right_only" and bool(r.get("open")):
                tag += " (khuli)"
            emit(f"{r['symbol']:>14} | {r['entry_day']:>10} | {str(r.get('exit_day_paper', '')):>10} | {str(r.get('exit_day_test', '')):>10} | "
                 f"{(r['ret_pct'] if pd.notna(r.get('ret_pct')) else float('nan')):>+7.2f} | {(r['ret'] if pd.notna(r.get('ret')) else float('nan')):>+7.2f} | {tag} {same}")
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
