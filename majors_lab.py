"""
MAJORS TREND LAB - BTC aur ETH par trend-following (crypto ki sab se mustanad strategy)
======================================================================================
Kyun: BTC/ETH ka data survivorship bias se paak (hamesha bare coins), 2018 (-80%) aur 2022 (-75%)
dono crash test mein, aur trades saal mein chand baar - kharcha kam.

Rule families (daily close par faisla, AGLE din ke open par amal - lookahead-free):
  SMA  N        : close > SMA(N) -> hold, warna cash                      N = 50/100/150/200
  EMA  f/s      : EMA(f) > EMA(s) -> hold                                  10/50, 20/100, 50/200
  DONCH n/m     : close > pichle n din ka high -> khareedo; close < pichle m din ka low -> becho
                                                                           20/10, 50/20, 100/50
  har rule: (a) full position, (b) volatility-target (saalana 50% vol, 30-din, max 100%)
Portfolio: BTC 50% + ETH 50% (har ek apni sleeve), mahana rebalance nahi - har sleeve alag.
Kharcha: har position badalne par fee 0.1% + slip 0.05% (turnover par).
Tasdeeq:
  - Buy & hold se muqabla (CAGR, MaxDD, Sharpe)
  - Random-timing control: wahi exposure aur utne hi in/out switch, lekin random waqt (200 baar)
    -> rule ka Sharpe random ke 95th percentile se ooper hona chahiye (timing mein asli edge)
  - 4 time-folds, saal-war (2018, 2022 crash khaas)
  - Family mein kitne rules pass (parameter-neighbourhood)
Natija: majors_lab_RESULTS.txt
"""
import numpy as np
import pandas as pd

from bot_core import fetch_full, norm, FEE, SLIP

COINS = ["BTC/USDT", "ETH/USDT"]
DAYS = 3400
COST = FEE + SLIP
N_RANDOM = 200
N_FOLDS = 4
TARGET_VOL = 0.50
OUT = "majors_lab_RESULTS.txt"

RULES = ([("SMA", (n,)) for n in (50, 100, 150, 200)]
         + [("EMA", fs) for fs in ((10, 50), (20, 100), (50, 200))]
         + [("DONCH", nm) for nm in ((20, 10), (50, 20), (100, 50))])


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def signal(d, kind, p):
    c, h, l = d["close"], d["high"], d["low"]
    if kind == "SMA":
        s = (c > c.rolling(p[0]).mean()).astype(float)
        s[c.rolling(p[0]).mean().isna()] = np.nan
        return s
    if kind == "EMA":
        s = (ema(c, p[0]) > ema(c, p[1])).astype(float)
        s.iloc[:p[1]] = np.nan
        return s
    up = c > h.shift(1).rolling(p[0]).max()
    dn = c < l.shift(1).rolling(p[1]).min()
    st = pd.Series(np.nan, index=d.index)
    st[up] = 1.0
    st[dn] = 0.0
    st = st.ffill()
    st.iloc[:p[0]] = np.nan
    return st


def vol_weight(d):
    r = d["close"].pct_change()
    vol = r.rolling(30).std() * np.sqrt(365)
    return (TARGET_VOL / vol).clip(upper=1.0)


def sleeve_returns(d, w_dec):
    """w_dec[t] close t par tay -> open t+1 se open t+2 tak lagu. Wapsi: rozana net return."""
    o = d["open"]
    ret_oo = o.shift(-1) / o - 1                     # open t -> open t+1
    w_held = w_dec.shift(1).fillna(0)                # din t par jo position (kal ke close ka faisla)
    turnover = w_held.diff().abs().fillna(w_held.abs())
    return (w_held * ret_oo - turnover * COST).dropna()


def stats(r):
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    dd = (eq / eq.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0.0
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    return {"cagr": eq.iloc[-1] ** (1 / yrs) - 1, "dd": dd, "sharpe": sh, "yearly": yearly}


def random_timing(w, rng):
    """Wahi in/out run-lengths, random tarteeb -> wahi exposure aur switches, random waqt."""
    x = (w.fillna(0) > 0).astype(int).to_numpy()
    if x.sum() == 0:
        return w * 0
    change = np.flatnonzero(np.diff(x)) + 1
    runs = np.split(x, change)
    ins = [len(r) for r in runs if r[0] == 1]
    outs = [len(r) for r in runs if r[0] == 0]
    rng.shuffle(ins)
    rng.shuffle(outs)
    seq, state = [], rng.random() < 0.5
    while ins or outs:
        if state and ins:
            seq += [1] * ins.pop()
        elif not state and outs:
            seq += [0] * outs.pop()
        state = not state
    out = np.array(seq[:len(x)] + [0] * max(0, len(x) - len(seq)), float)
    return pd.Series(out, index=w.index)


def main(data=None):
    lines = []

    def emit(x=""):
        print(x)
        lines.append(x)

    if data is None:
        from data_fetcher import get_exchange
        ex = get_exchange()
        data = {}
        for s in COINS:
            df = fetch_full(ex, s, "1d", DAYS)
            data[s] = norm(df).set_index("timestamp")
            print(s, len(df), df["timestamp"].iloc[0])
    start = max(d.index[0] for d in data.values()) + pd.Timedelta(days=210)     # SMA200 garam
    idx = None
    for d in data.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    idx = idx[idx >= start]
    fold_edges = pd.date_range(idx[0], idx[-1], periods=N_FOLDS + 1)

    emit("=" * 100)
    emit("MAJORS TREND LAB - BTC 50% + ETH 50%, daily trend-following")
    emit("=" * 100)
    emit(f"Period: {idx[0].date()} -> {idx[-1].date()} ({len(idx)/365:.1f} saal) | kharcha {COST*100:.2f}% har position-badlao par")

    def combo(weights_by_coin):
        rs = [sleeve_returns(data[s], weights_by_coin[s]).reindex(idx).dropna() for s in COINS]
        common = rs[0].index.intersection(rs[1].index)
        return 0.5 * rs[0].loc[common] + 0.5 * rs[1].loc[common]

    def fold_line(r):
        out = []
        for a, b in zip(fold_edges[:-1], fold_edges[1:]):
            x = r[(r.index >= a) & (r.index < b)]
            out.append(f"{((1 + x).prod() - 1) * 100:+.0f}%")
        return " / ".join(out)

    bh = combo({s: pd.Series(1.0, index=data[s].index) for s in COINS})
    sb = stats(bh)
    emit(f"\nBUY & HOLD (BTC+ETH): CAGR={sb['cagr']*100:+.1f}% MaxDD={sb['dd']*100:.1f}% Sharpe={sb['sharpe']:.2f} | Folds: {fold_line(bh)}")
    emit("   Saal-war: " + " ".join(f"{y}:{v*100:+.0f}%" for y, v in sb["yearly"].items()))

    rows = []
    for kind, p in RULES:
        for voladj in (False, True):
            name = f"{kind} {'/'.join(map(str, p))}{' +volTarget' if voladj else ''}"
            w = {}
            for s in COINS:
                sig = signal(data[s], kind, p)
                w[s] = (sig * vol_weight(data[s])) if voladj else sig
                w[s] = w[s].fillna(0)
            r = combo(w)
            st = stats(r)
            rng = np.random.default_rng(11)
            rsh = []
            for _ in range(N_RANDOM):
                rw = {}
                for s in COINS:
                    rt = random_timing(w[s], rng)
                    rw[s] = rt * vol_weight(data[s]).fillna(0) if voladj else rt
                rsh.append(stats(combo(rw))["sharpe"])
            r95 = float(np.percentile(rsh, 95))
            folds_ok = all(((1 + r[(r.index >= a) & (r.index < b)]).prod() - 1) > -0.05
                           for a, b in zip(fold_edges[:-1], fold_edges[1:]))
            expo = float(np.mean([w[s].reindex(idx).fillna(0).mean() for s in COINS]))
            switches = int(sum((w[s].reindex(idx).fillna(0).diff().abs() > 0.5).sum() for s in COINS) / 2)
            ok = st["sharpe"] > r95 and st["sharpe"] > sb["sharpe"] and st["dd"] > -0.45 and folds_ok
            rows.append((name, st, r95, ok, expo, switches))
            emit(f"\n[{'PASS' if ok else 'FAIL'}] {name}")
            emit(f"   CAGR={st['cagr']*100:+.1f}% MaxDD={st['dd']*100:.1f}% Sharpe={st['sharpe']:.2f} "
                 f"(random-timing 95th={r95:.2f}) | market mein {expo*100:.0f}% waqt | ~{switches} trades/coin")
            emit(f"   Folds: {fold_line(r)} | Saal-war: " + " ".join(f"{y}:{v*100:+.0f}%" for y, v in st["yearly"].items()))

    emit("\n" + "=" * 100)
    emit("KHULASA (PASS = Sharpe > random-timing 95th aur > Buy&Hold, MaxDD > -45%, koi fold -5% se bura nahi)")
    emit("=" * 100)
    emit(f"{'Rule':>26} | {'CAGR':>7} | {'MaxDD':>7} | {'Sharpe':>6} | {'Rnd95':>5} | {'Expo':>4} | {'Trades':>6} | Faisla")
    emit(f"{'BUY & HOLD':>26} | {sb['cagr']*100:>+6.1f}% | {sb['dd']*100:>6.1f}% | {sb['sharpe']:>6.2f} |   -   | 100% |      - | -")
    for name, st, r95, ok, expo, sw in rows:
        emit(f"{name:>26} | {st['cagr']*100:>+6.1f}% | {st['dd']*100:>6.1f}% | {st['sharpe']:>6.2f} | {r95:>5.2f} | "
             f"{expo*100:>3.0f}% | {sw:>6} | {'PASS' if ok else 'FAIL'}")
    for fam in ("SMA", "EMA", "DONCH"):
        fr = [r[3] for r in rows if r[0].startswith(fam)]
        emit(f"{fam}: {sum(fr)}/{len(fr)} PASS")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SAVE] {OUT}")


if __name__ == "__main__":
    main()
