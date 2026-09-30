"""
Complete Coin Dashboard - Union AB signals ke liye MUKAMMAL, CUSTOMIZABLE
context table. Har metric ALAG checkbox se ON/OFF ki ja sakti hai.

⚠️ IMPORTANT: Signal logic (Union AB) BILKUL nahi chhua gaya - ye sirf
MOJOODA signal coins ke liye EXTRA maloomat hai. Koi filter nahi.

Chalayen: streamlit run complete_dashboard.py
"""

import requests
import numpy as np
import pandas as pd
import streamlit as st

import config
from data_fetcher import get_exchange, get_coin_list, fetch_ohlcv
from strategies import STRATEGY_FUNCTIONS, apply_cooldown
from backtest_engine import compute_atr, simulate_trades

st.set_page_config(page_title="Complete Coin Dashboard", layout="wide")
st.title("📊 Complete Coin Dashboard (Customizable)")
st.warning(
    "⚠️ Ye sirf MALOOMAT hai — asal Union AB signal logic bilkul nahi "
    "badla gaya. Jitni zyada cheezein ON karenge, utna waqt lagega."
)

CE_A = {"period": 16, "multiplier": 3.0}
CE_B = {"period": 12, "multiplier": 3.0}
RR_MULTIPLE = 2.0
CONTEXT_TIMEFRAMES = ["15m", "1h", "4h", "1d"]

WHALE_TRACKED_TOKENS = {
    "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec",
    "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "LINK": "0x514910771af9ca656af840dff83e8264ecf986ca",
    "UNI": "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
}

st.sidebar.header("Kya Kya Dikhana Hai (Toggle ON/OFF)")
show_fear_greed = st.sidebar.checkbox("BTC Fear & Greed", value=True)
show_mtf_volume = st.sidebar.checkbox("Multi-Timeframe Volume", value=True)
show_mtf_change = st.sidebar.checkbox("Multi-Timeframe Price Change %", value=True)
show_funding = st.sidebar.checkbox("Funding Rate", value=True)
show_oi = st.sidebar.checkbox("Open Interest", value=True)
show_liquidity = st.sidebar.checkbox("Liquidity Up/Down (Order Book $)", value=True)
show_orderbook_ratio = st.sidebar.checkbox("Order Book Bid/Ask Ratio", value=True)
show_24h_range = st.sidebar.checkbox("24h High/Low Distance", value=True)
show_history = st.sidebar.checkbox("Is Coin ki Purani Performance (Win%/PF)", value=True)
show_btc_corr = st.sidebar.checkbox("BTC Correlation", value=False)
show_long_short = st.sidebar.checkbox("Long/Short Ratio (limited availability)", value=False)
show_whale = st.sidebar.checkbox("Whale Transfers (sirf USDT/USDC/LINK/UNI)", value=False)

etherscan_key = ""
if show_whale:
    etherscan_key = st.sidebar.text_input("Etherscan API Key (whale ke liye)", type="password")

st.sidebar.markdown("---")
coin_source = st.sidebar.radio("Coin Kaise Chunein", ["Auto Scan (top N by volume)", "Manual Paste (jo signal coins hain)"])

if coin_source.startswith("Manual"):
    manual_coins_text = st.sidebar.text_area(
        "Coins paste karein (comma-separated, jaise LUNC/USDT,PENDLE/USDT,KOMA/USDT)",
        height=100,
    )
else:
    n_coins = st.sidebar.slider("Kitne coins scan karein", 10, 400, 20)

signal_timeframe = st.sidebar.selectbox("Signal Timeframe", ["15m", "1h", "4h"], index=1)
st.sidebar.caption("⚠️ Zyada toggles ON + zyada coins = zyada waqt.")


@st.cache_data(ttl=1800)
def get_fear_greed():
    try:
        resp = requests.get("https://api.alternative.me/fng/", timeout=10)
        data = resp.json()
        item = data["data"][0]
        return int(item["value"]), item["value_classification"]
    except Exception:
        return None, None


@st.cache_resource
def get_futures_exchange():
    import ccxt
    return ccxt.kucoinfutures({"enableRateLimit": True})


def get_funding_and_oi(futures_exchange, symbol, need_funding, need_oi):
    base = symbol.split("/")[0]
    futures_symbol = f"{base}/USDT:USDT"
    funding, oi = None, None
    if need_funding:
        try:
            rate_info = futures_exchange.fetch_funding_rate(futures_symbol)
            funding = rate_info.get("fundingRate")
        except Exception:
            pass
    if need_oi:
        try:
            oi_info = futures_exchange.fetch_open_interest(futures_symbol)
            oi = oi_info.get("openInterestValue") or oi_info.get("openInterestAmount")
        except Exception:
            pass
    return funding, oi


def get_long_short_ratio(futures_exchange, symbol):
    try:
        base = symbol.split("/")[0]
        futures_symbol = f"{base}/USDT:USDT"
        if hasattr(futures_exchange, "fetch_long_short_ratio"):
            data = futures_exchange.fetch_long_short_ratio(futures_symbol)
            return data.get("longShortRatio")
        return None
    except Exception:
        return None


def get_orderbook_info(exchange, symbol, depth=20):
    try:
        ob = exchange.fetch_order_book(symbol, limit=depth)
        bids = ob.get("bids", [])[:depth]
        asks = ob.get("asks", [])[:depth]
        if not bids or not asks:
            return None, None, None
        bid_vol = sum(p * q for p, q in bids)
        ask_vol = sum(p * q for p, q in asks)
        ratio = bid_vol / ask_vol if ask_vol > 0 else None
        return bid_vol, ask_vol, ratio
    except Exception:
        return None, None, None


def get_multi_tf_context(exchange, symbol, need_volume, need_change):
    out = {}
    for tf in CONTEXT_TIMEFRAMES:
        try:
            df_tf = fetch_ohlcv(exchange, symbol, tf, limit=25)
            if df_tf is None or len(df_tf) < 21:
                out[tf] = (None, None)
                continue
            vol_ratio, price_change_pct = None, None
            if need_volume:
                vol_ma = df_tf["volume"].iloc[:-1].tail(20).mean()
                last_vol = df_tf["volume"].iloc[-1]
                vol_ratio = last_vol / vol_ma if vol_ma > 0 else None
            if need_change:
                last_open = df_tf["open"].iloc[-1]
                last_close = df_tf["close"].iloc[-1]
                price_change_pct = (last_close - last_open) / last_open * 100 if last_open > 0 else None
            out[tf] = (vol_ratio, price_change_pct)
        except Exception:
            out[tf] = (None, None)
    return out


def get_24h_range_distance(exchange, symbol):
    try:
        df_1d = fetch_ohlcv(exchange, symbol, "1d", limit=2)
        if df_1d is None or len(df_1d) < 1:
            return None, None
        last = df_1d.iloc[-1]
        current = last["close"]
        dist_from_high = (last["high"] - current) / current * 100
        dist_from_low = (current - last["low"]) / current * 100
        return round(dist_from_high, 2), round(dist_from_low, 2)
    except Exception:
        return None, None


def get_historical_performance(df, combo_sig, ce):
    try:
        trades = simulate_trades(df, combo_sig, config.BACKTEST_PARAMS, ce)
        if not trades:
            return None, None
        trades_df = pd.DataFrame(trades)
        wins = trades_df[trades_df["return_pct"] > 0]
        losses = trades_df[trades_df["return_pct"] <= 0]
        win_rate = len(wins) / len(trades_df) * 100
        pf = wins["return_pct"].sum() / abs(losses["return_pct"].sum()) if len(losses) and losses["return_pct"].sum() != 0 else None
        return round(win_rate, 1), round(pf, 2) if pf else None
    except Exception:
        return None, None


def get_btc_correlation(exchange, symbol, timeframe):
    try:
        df_coin = fetch_ohlcv(exchange, symbol, timeframe, limit=100)
        df_btc = fetch_ohlcv(exchange, "BTC/USDT", timeframe, limit=100)
        if df_coin is None or df_btc is None:
            return None
        merged = pd.merge(df_coin[["timestamp", "close"]], df_btc[["timestamp", "close"]],
                           on="timestamp", suffixes=("_coin", "_btc"))
        if len(merged) < 20:
            return None
        ret_coin = merged["close_coin"].pct_change().dropna()
        ret_btc = merged["close_btc"].pct_change().dropna()
        corr = ret_coin.corr(ret_btc)
        return round(corr, 2) if corr is not None else None
    except Exception:
        return None


def get_whale_activity(symbol, api_key):
    base = symbol.split("/")[0]
    if base not in WHALE_TRACKED_TOKENS or not api_key:
        return "N/A"
    try:
        contract = WHALE_TRACKED_TOKENS[base]
        resp = requests.get("https://api.etherscan.io/api", params={
            "module": "account", "action": "tokentx", "contractaddress": contract,
            "page": 1, "offset": 5, "sort": "desc", "apikey": api_key,
        }, timeout=10)
        data = resp.json()
        if data.get("status") == "1" and data.get("result"):
            return f"{len(data['result'])} recent transfers"
        return "Koi recent data nahi"
    except Exception:
        return "N/A"


if st.button("📊 Dashboard Chalayen", type="primary"):
    exchange = get_exchange()
    futures_exchange = get_futures_exchange() if (show_funding or show_oi or show_long_short) else None

    if show_fear_greed:
        fg_value, fg_label = get_fear_greed()
        if fg_value is not None:
            color = "red" if fg_value < 25 else "orange" if fg_value < 45 else "gray" if fg_value < 55 else "lightgreen" if fg_value < 75 else "green"
            st.markdown(f"### 📊 BTC Fear & Greed Index: **{fg_value}/100** — :{color}[{fg_label}]")
        else:
            st.info("Fear & Greed abhi nahi mil saka.")
        st.markdown("---")

    if coin_source.startswith("Manual"):
        coins = [c.strip() for c in manual_coins_text.split(",") if c.strip()]
        if not coins:
            st.error("Koi coin nahi likha gaya. Comma-separated symbols likhein (jaise LUNC/USDT,PENDLE/USDT).")
            st.stop()
    else:
        try:
            with st.spinner("Coin list le rahe hain..."):
                coins = get_coin_list(exchange)[:n_coins]
        except Exception as e:
            st.error(f"Exchange se connect nahi ho paya: {e}")
            st.stop()

    signal_coins = []
    progress = st.progress(0.0, text="Signals dhoond rahe hain...")
    for i, symbol in enumerate(coins):
        try:
            df = fetch_ohlcv(exchange, symbol, signal_timeframe, limit=max(config.CANDLE_LIMITS.get(signal_timeframe, 500), 300))
        except Exception:
            df = None
        if df is not None and len(df) >= 220:
            try:
                ichi_sig = apply_cooldown(STRATEGY_FUNCTIONS["ichimoku"](df, config.STRATEGY_PARAMS["ichimoku"]), config.SIGNAL_COOLDOWN_BARS)
                ms_sig = apply_cooldown(STRATEGY_FUNCTIONS["market_structure"](df, config.STRATEGY_PARAMS["market_structure"]), config.SIGNAL_COOLDOWN_BARS)
                ema_sig = apply_cooldown(STRATEGY_FUNCTIONS["ema_crossover"](df, config.STRATEGY_PARAMS["ema_crossover"]), config.SIGNAL_COOLDOWN_BARS)
                breakout_sig = apply_cooldown(STRATEGY_FUNCTIONS["breakout"](df, config.STRATEGY_PARAMS["breakout"]), config.SIGNAL_COOLDOWN_BARS)
                combo_a = ichi_sig & ms_sig
                combo_b = ema_sig & breakout_sig

                for combo_sig, combo_name, ce in [(combo_a, "Ichimoku+MS", CE_A), (combo_b, "EMA+Breakout", CE_B)]:
                    if combo_sig.tail(3).any():
                        idx = combo_sig.tail(3)[combo_sig.tail(3)].index[-1]
                        atr = compute_atr(df, ce["period"])
                        highest_high = df["high"].rolling(ce["period"]).max()
                        chandelier = (highest_high - ce["multiplier"] * atr).loc[idx]
                        entry_price = df.loc[idx, "close"]
                        current_price = df["close"].iloc[-1]
                        risk = entry_price - chandelier
                        tp_price = entry_price + risk * RR_MULTIPLE
                        signal_coins.append({
                            "Coin": symbol, "Combo": combo_name, "Bars Ago": len(df) - 1 - idx,
                            "Entry": round(entry_price, 6), "Current": round(current_price, 6),
                            "Trail Stop": round(chandelier, 6), "Take Profit": round(tp_price, 6),
                            "_df": df, "_sig": combo_sig, "_ce": ce,
                        })
            except Exception:
                pass
        progress.progress((i + 1) / len(coins), text=f"Signals dhoond rahe hain... {i+1}/{len(coins)}")
    progress.empty()

    if not signal_coins:
        st.info("Is waqt koi fresh signal nahi mila.")
        st.stop()

    st.success(f"✅ {len(signal_coins)} signals mile — ab context le rahe hain...")

    final_rows = []
    progress2 = st.progress(0.0, text="Context le rahe hain...")
    for i, row in enumerate(signal_coins):
        symbol = row["Coin"]
        final_row = {k: v for k, v in row.items() if not k.startswith("_")}

        if show_funding or show_oi:
            funding, oi = get_funding_and_oi(futures_exchange, symbol, show_funding, show_oi)
            if show_funding:
                final_row["Funding Rate"] = f"{funding*100:.3f}%" if funding is not None else "N/A"
            if show_oi:
                final_row["Open Interest"] = f"${oi:,.0f}" if oi is not None else "N/A"

        if show_long_short:
            ls_ratio = get_long_short_ratio(futures_exchange, symbol)
            final_row["Long/Short Ratio"] = f"{ls_ratio:.2f}" if ls_ratio is not None else "N/A"

        if show_liquidity or show_orderbook_ratio:
            bid_liq, ask_liq, ob_ratio = get_orderbook_info(exchange, symbol)
            if show_liquidity:
                final_row["Liquidity Up ($)"] = f"${ask_liq:,.0f}" if ask_liq is not None else "N/A"
                final_row["Liquidity Down ($)"] = f"${bid_liq:,.0f}" if bid_liq is not None else "N/A"
            if show_orderbook_ratio:
                final_row["OrderBook Bid/Ask"] = f"{ob_ratio:.2f}x" if ob_ratio is not None else "N/A"

        if show_mtf_volume or show_mtf_change:
            mtf = get_multi_tf_context(exchange, symbol, show_mtf_volume, show_mtf_change)
            for tf in CONTEXT_TIMEFRAMES:
                vol_ratio, price_chg = mtf[tf]
                if show_mtf_volume:
                    final_row[f"Vol {tf}"] = f"{vol_ratio:.2f}x" if vol_ratio is not None else "N/A"
                if show_mtf_change:
                    final_row[f"Chg {tf}"] = f"{price_chg:+.2f}%" if price_chg is not None else "N/A"

        if show_24h_range:
            dist_high, dist_low = get_24h_range_distance(exchange, symbol)
            final_row["Dist from 24h High"] = f"{dist_high}%" if dist_high is not None else "N/A"
            final_row["Dist from 24h Low"] = f"{dist_low}%" if dist_low is not None else "N/A"

        if show_history:
            win_rate, pf = get_historical_performance(row["_df"], row["_sig"], row["_ce"])
            final_row["Coin's Own Win%"] = f"{win_rate}%" if win_rate is not None else "N/A"
            final_row["Coin's Own PF"] = f"{pf}" if pf is not None else "N/A"

        if show_btc_corr:
            corr = get_btc_correlation(exchange, symbol, signal_timeframe)
            final_row["BTC Correlation"] = f"{corr}" if corr is not None else "N/A"

        if show_whale:
            final_row["Whale Activity"] = get_whale_activity(symbol, etherscan_key)

        final_rows.append(final_row)
        progress2.progress((i + 1) / len(signal_coins), text=f"Context le rahe hain... {i+1}/{len(signal_coins)}")
    progress2.empty()

    df_final = pd.DataFrame(final_rows)
    st.dataframe(df_final, use_container_width=True, hide_index=True)

    csv = df_final.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV Download Karein", csv, "complete_dashboard.csv", "text/csv")

    st.caption(
        "Funding musbat = zyada log LONG (over-leveraged risk). Liquidity Up/Down = "
        "order book mein upar/neeche paisa (resistance/support). 'Coin's Own Win%/PF' "
        "= isi coin par ye combo pehle kaisa raha (chhota sample ho sakta hai, ehtiyat karein)."
    )