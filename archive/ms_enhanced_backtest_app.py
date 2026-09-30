"""
MS Enhanced Backtest - Mobile App (Streamlit Cloud par khud chalta hai,
PC ki zaroorat nahi)

⚠️ Mobile/cloud par waqt ki hadd hone ki wajah se coins kam rakhein
(50-150 tak behtar rahega, 400 PC par hi karein).

Chalayen: streamlit run ms_enhanced_backtest_app.py
"""

import pandas as pd
import streamlit as st

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import simulate_trades, summarize_trades

st.set_page_config(page_title="MS Enhanced Backtest", layout="wide")
st.title("📊 MS Enhanced Backtest (Mobile)")
st.caption(
    "Market Structure + Resistance Check + BTC Macro Filter ka HISTORICAL "
    "backtest — 'abhi signal hai ya nahi' nahi, balke 'agar pehle istemal "
    "karte to kaisa result aata' dikhata hai."
)
st.warning(
    "⚠️ Ye NAYA idea hai, BTC par 12 trades mein PF 3.53 mila tha (chhota "
    "sample). Yahan zyada coins par confirm ho raha hai — jab tak yahan "
    "bhi bara, positive sample na mile, trading decisions na karein."
)

ENHANCED_PARAMS = {
    "swing_lookback": 20,
    "resistance_lookback": 300,
    "resistance_exclude_recent": 20,
    "atr_period": 14,
    "min_rr_to_resistance": 1.5,
}
CE_PARAMS = {"period": 16, "multiplier": 3.0}


def market_structure_enhanced(df, btc_daily, ms_params, enh_params):
    base_signal = STRATEGY_FUNCTIONS["market_structure"](df, ms_params)

    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(enh_params["atr_period"]).mean()

    swing_low = low.rolling(enh_params["swing_lookback"]).min()
    swing_high = high.shift(enh_params["resistance_exclude_recent"]).rolling(enh_params["resistance_lookback"]).max()

    risk = close - (swing_low - 0.5 * atr)
    upside_to_resistance = swing_high - close
    resistance_ok = upside_to_resistance >= enh_params["min_rr_to_resistance"] * risk

    btc_ema50 = btc_daily["close"].ewm(span=50, adjust=False).mean()
    btc_bullish = (btc_daily["close"] > btc_ema50).rename("btc_bullish")
    merged = pd.merge_asof(
        df[["timestamp"]].sort_values("timestamp"),
        pd.DataFrame({"timestamp": btc_daily["timestamp"], "btc_bullish": btc_bullish}).sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    btc_ok = merged["btc_bullish"].fillna(False).values

    return base_signal, base_signal & resistance_ok.fillna(False) & btc_ok


col1, col2 = st.columns(2)
timeframe = col1.selectbox("Timeframe", ["15m", "1h", "4h"], index=1)
n_coins = col2.slider("Kitne coins backtest karein", 20, 150, 80)

st.caption("Zyada coins = zyada waqt lagega (mobile/cloud par 1-3 minute tak ho sakta hai).")

if st.button("📊 Backtest Chalayen", type="primary"):
    exchange = get_exchange()
    try:
        with st.spinner("BTC daily benchmark data le rahe hain..."):
            btc_daily = fetch_ohlcv(exchange, "BTC/USDT", "1d", limit=800)
        with st.spinner("Coin list le rahe hain..."):
            coins = get_coin_list(exchange)[:n_coins]
    except Exception as e:
        st.error(f"Exchange se connect nahi ho paya: {e}")
        st.stop()

    ms_params = config.STRATEGY_PARAMS["market_structure"]
    candle_limit = max(config.CANDLE_LIMITS.get(timeframe, 500), 350)

    all_trades_baseline = []
    all_trades_enhanced = []
    per_coin_rows = []

    progress = st.progress(0.0)
    for i, symbol in enumerate(coins):
        try:
            df = fetch_ohlcv(exchange, symbol, timeframe, limit=candle_limit)
        except Exception:
            df = None

        if df is not None and len(df) >= 320:
            try:
                base_sig, enh_sig = market_structure_enhanced(df, btc_daily, ms_params, ENHANCED_PARAMS)
                base_sig = apply_cooldown(base_sig, config.SIGNAL_COOLDOWN_BARS)
                enh_sig = apply_cooldown(enh_sig, config.SIGNAL_COOLDOWN_BARS)

                base_trades = simulate_trades(df, base_sig, config.BACKTEST_PARAMS, CE_PARAMS)
                all_trades_baseline.extend(base_trades)

                if enh_sig.sum() > 0:
                    enh_trades = simulate_trades(df, enh_sig, config.BACKTEST_PARAMS, CE_PARAMS)
                    all_trades_enhanced.extend(enh_trades)
                    if enh_trades:
                        s = summarize_trades(enh_trades, symbol, timeframe, "enhanced")
                        per_coin_rows.append(s)
            except Exception:
                pass
        progress.progress((i + 1) / len(coins))
    progress.empty()

    st.subheader("Overall Result")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Baseline (Market Structure solo)**")
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
        st.markdown("**Enhanced (+Resistance +BTC Macro)**")
        if all_trades_enhanced:
            df_e = pd.DataFrame(all_trades_enhanced)
            wins = df_e[df_e["return_pct"] > 0]
            losses = df_e[df_e["return_pct"] <= 0]
            win_rate = len(wins) / len(df_e) * 100
            pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
            st.metric("Trades", len(df_e))
            st.metric("Win Rate", f"{win_rate:.1f}%")
            st.metric("Profit Factor", f"{pf:.2f}" if pf else "N/A")
        else:
            st.info("Koi trades nahi mile.")

    if per_coin_rows:
        st.subheader("Per-Coin Breakdown (Enhanced)")
        st.dataframe(pd.DataFrame(per_coin_rows), use_container_width=True, hide_index=True)