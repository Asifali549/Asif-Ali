"""
Multi-Timeframe Confirmation Backtest - Mobile App

1h Market Structure signal + isi coin ka apna 4h trend (EMA21>EMA50)
confirmation - BTC macro NAHI, har coin ka apna 4h.

⚠️ 400 coins par 2 timeframes (1h+4h) fetch karne mein waqt lagega -
zyada coins chunne se mobile/cloud par kaafi der (5-10+ minute) lag
sakti hai. Sabar karein ya kam coins se shuru karein.

Chalayen: streamlit run mtf_confirm_app.py
"""

import pandas as pd
import streamlit as st

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import simulate_trades

st.set_page_config(page_title="MTF Confirmation Backtest", layout="wide")
st.title("📊 Multi-Timeframe Confirmation Backtest")
st.caption(
    "Market Structure (1h) + isi coin ka apna 4h trend (EMA21>EMA50) - "
    "BTC macro nahi, har coin apna khud ka higher-timeframe check."
)
st.warning(
    "⚠️ BTC par test mein farak bohot kam mila (PF 1.35 -> 1.37). Yahan "
    "zyada coins (altcoins) par dekh rahe hain ke waqai farak parta hai "
    "ya nahi. Zyada coins = zyada waqt (5-10+ minute lag sakta hai)."
)

CE_PARAMS = {"period": 16, "multiplier": 3.0}

n_coins = st.slider("Kitne coins backtest karein", 20, 400, 100)
st.caption("400 coins mein kaafi waqt (10+ minute) lag sakta hai - sabar karein, page band na karein.")

if st.button("📊 Backtest Chalayen", type="primary"):
    exchange = get_exchange()
    try:
        with st.spinner("Coin list le rahe hain..."):
            coins = get_coin_list(exchange)[:n_coins]
    except Exception as e:
        st.error(f"Exchange se connect nahi ho paya: {e}")
        st.stop()

    ms_params = config.STRATEGY_PARAMS["market_structure"]

    all_trades_baseline = []
    all_trades_mtf = []

    progress = st.progress(0.0)
    status = st.empty()

    for i, symbol in enumerate(coins):
        status.text(f"Scanning {symbol} ({i+1}/{len(coins)})...")
        try:
            df_1h = fetch_ohlcv(exchange, symbol, "1h", limit=2000)
            df_4h = fetch_ohlcv(exchange, symbol, "4h", limit=500)
        except Exception:
            df_1h, df_4h = None, None

        if df_1h is not None and len(df_1h) >= 220 and df_4h is not None and len(df_4h) >= 60:
            try:
                ms_sig = STRATEGY_FUNCTIONS["market_structure"](df_1h, ms_params)
                ms_sig = apply_cooldown(ms_sig, config.SIGNAL_COOLDOWN_BARS)

                ema21_4h = df_4h["close"].ewm(span=21, adjust=False).mean()
                ema50_4h = df_4h["close"].ewm(span=50, adjust=False).mean()
                trend_4h_ok = (ema21_4h > ema50_4h).rename("trend_ok")

                merged = pd.merge_asof(
                    df_1h[["timestamp"]].sort_values("timestamp"),
                    pd.DataFrame({"timestamp": df_4h["timestamp"], "trend_ok": trend_4h_ok}).sort_values("timestamp"),
                    on="timestamp", direction="backward",
                )
                trend_ok = merged["trend_ok"].fillna(False).values
                mtf_sig = ms_sig & trend_ok

                base_trades = simulate_trades(df_1h, ms_sig, config.BACKTEST_PARAMS, CE_PARAMS)
                all_trades_baseline.extend(base_trades)

                if mtf_sig.sum() > 0:
                    mtf_trades = simulate_trades(df_1h, mtf_sig, config.BACKTEST_PARAMS, CE_PARAMS)
                    all_trades_mtf.extend(mtf_trades)
            except Exception:
                pass
        progress.progress((i + 1) / len(coins))

    progress.empty()
    status.empty()

    st.subheader("Overall Result")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Baseline (Market Structure 1h solo)**")
        if all_trades_baseline:
            df_b = pd.DataFrame(all_trades_baseline)
            wins = df_b[df_b["return_pct"] > 0]
            losses = df_b[df_b["return_pct"] <= 0]
            win_rate = len(wins) / len(df_b) * 100
            pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
            st.metric("Trades", len(df_b))
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("Profit Factor", f"{pf:.2f}" if pf else "N/A")
        else:
            st.info("Koi trades nahi mile.")

    with col2:
        st.markdown("**+ Own 4h Trend Confirmation**")
        if all_trades_mtf:
            df_m = pd.DataFrame(all_trades_mtf)
            wins = df_m[df_m["return_pct"] > 0]
            losses = df_m[df_m["return_pct"] <= 0]
            win_rate = len(wins) / len(df_m) * 100
            pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
            st.metric("Trades", len(df_m))
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("Profit Factor", f"{pf:.2f}" if pf else "N/A")
        else:
            st.info("Koi trades nahi mile.")