"""
SMC MULTI-TIMEFRAME STRATEGY (4H trend -> 1H momentum -> 15m entry)  — spot, buy-only
======================================================================================
Tajurbati (EXPERIMENTAL) strategy. Live bot tab banega jab smc_mtf_test.py ke sakht test PASS kare.

LOGIC (har cheez sirf BAND candles se - koi lookahead nahi):
  1) 4H TREND + STRUCTURE (bara ruk)
       - close > EMA200, EMA50 > EMA200, EMA50 ooper ki taraf (6 candles pehle se ooncha)
       - Market structure: aakhri confirmed swing LOW pichle swing low se ooncha (Higher Low)
  2) 1H MOMENTUM (darmiyana ruk)
       - close > EMA50 aur RSI(14) > 50
  3) 15m ENTRY (chhota ruk) - "discount mein pullback, phir structure break"
       - Pullback: pichle 12 candles (3 ghante) mein RSI(14) 40 se neeche gaya (sasti jagah)
       - BOS: close ne aakhri confirmed 15m swing HIGH ko ABHI ooper toda (fresh break)
       - Volume: candle ka volume > 1.5x pichle 20 ka average (asli khareedar)
       - Candle quality: close candle ki range ke ooper 40% mein (upper wick wala fake break nahi)
       - Overbought nahi: RSI(14) < 72
       - Volatility: ATR% pichle 10 din ke 20-90 percentile ke beech (na murda market, na pagal spike)
  4) RISK
       - Stop: aakhri 15m swing low - 0.2 ATR; kam az kam 1 ATR, ziada se ziada 3 ATR door
         (3 ATR se door ho to trade chhor do)
       - Target: 2R (risk ka do guna)  |  Time stop: 96 candles (24 ghante)
  Entry: signal wali 15m candle band hone ke baad AGLI candle ke open par.

Chalayen (abhi ke signals dekhne ke liye):  python smc_mtf_strategy.py
"""
import numpy as np
import pandas as pd

# ============================ INPUTS (default settings) ============================
P = {
    # 4H trend
    "ema_fast_4h": 50, "ema_slow_4h": 200, "ema_slope_bars_4h": 6, "pivot_4h": 3,
    # 1H momentum
    "ema_1h": 50, "rsi_1h_min": 50,
    # 15m entry
    "pivot_15": 3,              # swing high/low = dono taraf 3 candles se ooncha/neecha
    "pullback_bars": 12,        # pullback kitni candles pehle tak dekhna
    "pullback_rsi": 40,         # pullback mein RSI is se neeche jaye
    "vol_sma": 20, "vol_mult": 1.5,
    "close_pos_min": 0.6,       # close candle range ke ooper 40% mein
    "rsi_max": 72,
    "atr_n": 14, "atr_rank_win": 960, "atr_rank_min": 0.2, "atr_rank_max": 0.9,
    # risk
    "sl_buffer_atr": 0.2, "sl_min_atr": 1.0, "sl_max_atr": 3.0,
    "tp_r": 2.0,                # target = entry + 2 x risk
    "be_at_r": None,            # misal 1.0 = 1R par stop ko entry par le aao (None = band)
    "max_hold": 96,             # 96 x 15m = 24 ghante
}
TF15 = pd.Timedelta(minutes=15)


# ============================ Indicators ============================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def confirmed_pivots(df, k):
    """Swing high/low jo k candles BAAD confirm hota hai. Har bar par: aakhri aur us se pichla
    confirmed swing low, aur aakhri confirmed swing high (sirf maazi ka data)."""
    w = 2 * k + 1
    is_ph = df["high"] == df["high"].rolling(w, center=True).max()
    is_pl = df["low"] == df["low"].rolling(w, center=True).min()
    # bar i ka pivot bar i+k par pata chalta hai -> k shift
    ph_val = df["high"].shift(k).where(is_ph.shift(k, fill_value=False))
    pl_val = df["low"].shift(k).where(is_pl.shift(k, fill_value=False))
    last_ph = ph_val.ffill()
    last_pl = pl_val.ffill()
    pls = pl_val.dropna()
    prev_pl = pls.shift(1).reindex(df.index).ffill()
    return last_ph, last_pl, prev_pl


def resample(df15, rule):
    x = df15.set_index("timestamp")
    return x.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def attach_htf(df15, feat, rule):
    """HTF feature sirf tab use ho jab wo HTF candle BAND ho chuki ho."""
    f = feat.copy()
    f["avail"] = f.index + pd.Timedelta(rule)
    f = f.reset_index(drop=True).sort_values("avail")
    left = df15[["timestamp"]].copy()
    left["close_time"] = left["timestamp"] + TF15
    m = pd.merge_asof(left, f, left_on="close_time", right_on="avail", direction="backward")
    return m.drop(columns=["timestamp", "close_time", "avail"])


# ============================ Signal ============================
def compute(df15, p=P):
    """df15: 15m OHLCV (timestamp = candle open, UTC, sirf band candles).
    Wapsi: df15 + columns: signal (bool), stop (signal par stop level), atr."""
    d = df15.reset_index(drop=True).copy()

    # ---- 4H trend + structure ----
    h4 = resample(d, "4h")
    f4 = pd.DataFrame(index=h4.index)
    ef, es = ema(h4["close"], p["ema_fast_4h"]), ema(h4["close"], p["ema_slow_4h"])
    _, pl4, prev_pl4 = confirmed_pivots(h4, p["pivot_4h"])
    f4["trend4h"] = ((h4["close"] > es) & (ef > es) & (ef > ef.shift(p["ema_slope_bars_4h"]))
                     & (pl4 > prev_pl4))
    f4.loc[f4.index[:p["ema_slow_4h"]], "trend4h"] = False          # EMA200 garam hone tak nahi
    # ---- 1H momentum ----
    h1 = resample(d, "1h")
    f1 = pd.DataFrame(index=h1.index)
    f1["mom1h"] = (h1["close"] > ema(h1["close"], p["ema_1h"])) & (rsi(h1["close"]) > p["rsi_1h_min"])

    d["trend4h"] = attach_htf(d, f4, "4h")["trend4h"].fillna(False).astype(bool).values
    d["mom1h"] = attach_htf(d, f1, "1h")["mom1h"].fillna(False).astype(bool).values

    # ---- 15m entry ----
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
    d["atr"] = atr(d, p["atr_n"])
    r = rsi(c)
    sh15, sl15, _ = confirmed_pivots(d, p["pivot_15"])
    pullback = r.rolling(p["pullback_bars"]).min() < p["pullback_rsi"]
    bos = (c > sh15) & (c.shift(1) <= sh15)
    vol_ok = v > p["vol_mult"] * v.rolling(p["vol_sma"]).mean().shift(1)
    rng = (h - l).replace(0, np.nan)
    candle_ok = (c - l) / rng >= p["close_pos_min"]
    not_ob = r < p["rsi_max"]
    atr_rank = (d["atr"] / c).rolling(p["atr_rank_win"]).rank(pct=True)
    vola_ok = (atr_rank >= p["atr_rank_min"]) & (atr_rank <= p["atr_rank_max"])

    # ---- stop: structure + ATR clamp ----
    raw_stop = sl15 - p["sl_buffer_atr"] * d["atr"]
    risk = c - raw_stop
    stop = raw_stop.where(risk >= p["sl_min_atr"] * d["atr"], c - p["sl_min_atr"] * d["atr"])
    risk_ok = (c - raw_stop) <= p["sl_max_atr"] * d["atr"]

    d["setup"] = d["trend4h"] & d["mom1h"]                        # bara + darmiyana ruk theek
    d["signal"] = (d["setup"] & pullback & bos & vol_ok & candle_ok & not_ob & vola_ok
                   & risk_ok & stop.notna() & (stop < c)).fillna(False).astype(bool)
    d["stop"] = stop
    return d


# ============================ Live scan ============================
def scan(top_n=60):
    from data_fetcher import get_exchange, get_coin_list
    from bot_core import fetch_full, norm, STABLES
    ex = get_exchange()
    coins = [s for s in get_coin_list(ex) if s.split("/")[0].upper() not in STABLES][:top_n]
    rows = []
    for sym in coins:
        try:
            df = fetch_full(ex, sym, "15m", 3600)          # EMA200 (4H) ke liye ~38 din
            if df is None or len(df) < 3400:
                continue
            d = compute(norm(df))
            last = d.iloc[-1]
            if last["signal"]:
                e, s = float(last["close"]), float(last["stop"])
                rows.append((sym, e, s, e + P["tp_r"] * (e - s)))
        except Exception as ex_err:
            print(f"{sym}: SKIP ({ex_err})")
    t = pd.Timestamp.now(tz="Asia/Karachi").strftime("%d %b %I:%M %p PKT")
    print(f"\n=== SMC MTF signals ({t}) ===")
    for sym, e, s, tp in rows:
        print(f"BUY {sym}: ~{e:.6g} | Stop {s:.6g} ({(e - s) / e * 100:.1f}%) | Target {tp:.6g} (2R)")
    if not rows:
        print("Abhi koi signal nahi.")


if __name__ == "__main__":
    scan()
