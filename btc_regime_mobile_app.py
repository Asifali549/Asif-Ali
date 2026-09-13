"""
BTC Regime Filter - Mobile Test App (Streamlit Cloud par khud chalta
hai). Union AB (baseline) vs +BTC 200-day EMA Regime Filter compare
karta hai, ~200 din ke lambe dorania ke sath (taake regime tabdeeliyan
shamil hon), 100 coins tak.

⚠️ Ye kaafi bhaari test hai (lamba dorania + kai coins) - Streamlit
Cloud par 5-15+ minute lag sakte hain. Sabar karein, page band na karein.

Chalayen: streamlit run btc_regime_mobile_app.py
"""

import pandas as pd
import streamlit as st

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import simulate_trades

st.set_page_config(page_title="BTC Regime Filter Test", layout="wide")
st.title("📊 BTC Regime Filter Test (Mobile)")
st.caption(
    "Union AB (Ichimoku+MS + EMA+Breakout) BASELINE vs +BTC 200-day "
    "EMA Regime Filter - ~200 din ke lambe dorania ke sath (taake "
    "asli bull/bear regime tabdeeliyan shamil hon)."
)
st.warning(
    "⚠️ Ye BHAARI test hai (200 din ka data + kai coins). 5-15+ minute "
    "lag sakte hain. Sabar karein, page ko band na karein."
)

CE_A = {"period": 16, "multiplier": 3.0}
CE_B = {"period": 12, "multiplier": 3.0}

col1, col2 = st.columns(2)
timeframe = col1.selectbox("Timeframe", ["15m", "1h", "4h"], index=1)
n_coins = col2.slider("Kitne coins (max 100)", 10, 100, 50)

DURATION_DAYS = 200
CANDLE_LIMITS = {
    "15m": DURATION_DAYS * 96,
    "1h": DURATION_DAYS * 24,
    "4h": DURATION_DAYS * 6,
}
st.caption(f"Is timeframe ke liye {CANDLE_LIMITS[timeframe]:,} candles (~{DURATION_DAYS} din) fetch honge.")

if st.button("📊 BTC Regime Test Chalayen", type="primary"):
    exchange = get_exchange()

    try:
        with st.spinner("BTC daily benchmark data le rahe hain (regime filter ke liye)..."):
            btc_daily_raw = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=max(DURATION_DAYS + 200, 400))
        ema200 = btc_daily_raw["close"].ewm(span=200, adjust=False).mean()
        btc_regime = pd.DataFrame({
            "timestamp": btc_daily_raw["timestamp"],
            "regime_ok": btc_daily_raw["close"] > ema200,
        })
    except Exception as e:
        st.error(f"BTC data nahi mil saka: {e}")
        st.stop()

    try:
        with st.spinner("Coin list le rahe hain..."):
            coins = get_coin_list(exchange)[:n_coins]
    except Exception as e:
        st.error(f"Exchange se connect nahi ho paya: {e}")
        st.stop()

    baseline_trades_all = []
    regime_trades_all = []

    progress = st.progress(0.0, text="Coins scan ho rahe hain...")
    for i, symbol in enumerate(coins):
        try:
            df = fetch_ohlcv(exchange, symbol, timeframe, limit=CANDLE_LIMITS[timeframe])
        except Exception:
            df = None

        if df is not None and len(df) >= 250:
            try:
                merged = pd.merge_asof(
                    df[["timestamp"]].sort_values("timestamp"),
                    btc_regime.sort_values("timestamp"),
                    on="timestamp", direction="backward",
                )
                regime_ok = merged["regime_ok"].fillna(False)

                ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
                ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
                ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
                breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)

                combo_a = ichi_sig & ms_sig
                combo_b = ema_sig & breakout_sig
                combo_a_regime = combo_a & regime_ok.values
                combo_b_regime = combo_b & regime_ok.values

                if combo_a.sum() > 0:
                    baseline_trades_all.extend(simulate_trades(df, combo_a, config.BACKTEST_PARAMS, CE_A))
                if combo_b.sum() > 0:
                    baseline_trades_all.extend(simulate_trades(df, combo_b, config.BACKTEST_PARAMS, CE_B))
                if combo_a_regime.sum() > 0:
                    regime_trades_all.extend(simulate_trades(df, combo_a_regime, config.BACKTEST_PARAMS, CE_A))
                if combo_b_regime.sum() > 0:
                    regime_trades_all.extend(simulate_trades(df, combo_b_regime, config.BACKTEST_PARAMS, CE_B))
            except Exception:
                pass
        progress.progress((i + 1) / len(coins), text=f"Scanning {symbol}... {i+1}/{len(coins)}")
    progress.empty()

    st.subheader("Overall Result")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Baseline (bina filter)**")
        if baseline_trades_all:
            df_b = pd.DataFrame(baseline_trades_all)
            w = df_b[df_b["return_pct"] > 0]
            l = df_b[df_b["return_pct"] <= 0]
            pf = w["return_pct"].sum() / abs(l["return_pct"].sum()) if len(l) and l["return_pct"].sum() != 0 else None
            st.metric("Trades", len(df_b))
            st.metric("Win Rate", f"{len(w)/len(df_b)*100:.1f}%")
            st.metric("Profit Factor", f"{pf:.2f}" if pf else "N/A")
        else:
            st.info("Koi trades nahi mile.")

    with col2:
        st.markdown("**+BTC Regime Filter**")
        if regime_trades_all:
            df_r = pd.DataFrame(regime_trades_all)
            w = df_r[df_r["return_pct"] > 0]
            l = df_r[df_r["return_pct"] <= 0]
            pf = w["return_pct"].sum() / abs(l["return_pct"].sum()) if len(l) and l["return_pct"].sum() != 0 else None
            st.metric("Trades", len(df_r))
            st.metric("Win Rate", f"{len(w)/len(df_r)*100:.1f}%")
            st.metric("Profit Factor", f"{pf:.2f}" if pf else "N/A")
        else:
            st.info("Koi trades nahi mile.")