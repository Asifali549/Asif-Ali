"""
Data Fetcher - Exchange (KuCoin) se OHLCV data laata hai (ccxt library ke zariye)

NOTE: Ye file sirf aapke apne computer/server par chalegi jahan
Exchange API tak internet access ho. Isay khud fetch karne ke liye
'pip install ccxt' zaroori hai.
"""

import time
import pandas as pd
import ccxt

from config import EXCHANGE_ID, QUOTE_CURRENCY, AUTO_TOP_N_COINS, MANUAL_COIN_LIST, COIN_LIST_MODE, is_excluded_coin


def get_exchange():
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    return exchange


def get_coin_list(exchange=None):
    """
    COIN_LIST_MODE ke mutabiq coin list return karta hai.
    AUTO mode mein top N USDT spot pairs (24h volume ke hisab se) fetch karta hai.
    """
    if COIN_LIST_MODE == "MANUAL":
        return MANUAL_COIN_LIST

    if exchange is None:
        exchange = get_exchange()

    markets = exchange.load_markets()
    usdt_pairs = [
        s for s, m in markets.items()
        if m.get("spot") and m.get("quote") == QUOTE_CURRENCY and m.get("active")
    ]

    # Meme coins aur leveraged/binary tokens nikal dein
    before_count = len(usdt_pairs)
    usdt_pairs = [s for s in usdt_pairs if not is_excluded_coin(s)]
    excluded_count = before_count - len(usdt_pairs)
    if excluded_count > 0:
        print(f"  [FILTER] {excluded_count} meme/leveraged coins exclude kiye "
              f"({len(usdt_pairs)} bache)")

    # 24h volume ke hisab se sort karne ke liye tickers fetch karain.
    # NOTE: symbols list pass NAHI karte kyunke 400+ symbols wala URL Binance
    # reject kar deta hai (URL too long) - is liye SAB tickers le kar
    # baad mein locally filter karte hain.
    tickers = exchange.fetch_tickers()
    ranked = sorted(
        usdt_pairs,
        key=lambda s: tickers.get(s, {}).get("quoteVolume") or 0,
        reverse=True,
    )
    return ranked[:AUTO_TOP_N_COINS]


def fetch_ohlcv(exchange, symbol, timeframe, limit):
    """
    Ek symbol/timeframe ka OHLCV data laata hai aur pandas DataFrame return karta hai.
    Exchange ek call mein max ~1000-1500 candles deta hai, is liye zyada limit
    ke liye pagination (loop) ki zaroorat parti hai.

    IMPORTANT: Abhi tak "band" (closed) NA hui aakhri candle ko hata dete hain -
    warna signals "band hone se pehle" hi ban jayenge aur candle ke aakhir
    tak badalte rahenge (repainting jaisa masla). Ye fix purane 3-combo
    system aur naye AdvancedConfluence system, dono ke liye ek sath kaam
    karta hai (kyunke dono isi function se data lete hain).
    """
    all_candles = []
    since = None
    remaining = limit

    while remaining > 0:
        batch_limit = min(1000, remaining)
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch_limit)
        except Exception as e:
            print(f"  [SKIP] {symbol} {timeframe}: {e}")
            break

        if not candles:
            break

        all_candles = candles + all_candles if since else all_candles + candles
        remaining -= len(candles)

        if len(candles) < batch_limit:
            break

        # pichlay candles ke liye peeche jao (older data)
        since = candles[0][0] - (batch_limit * _timeframe_ms(timeframe))
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    # ---- CANDLE CLOSE ONLY: agar aakhri candle abhi tak band nahi hui (uska
    # close time abhi nahi aaya), to usay hata dete hain. Warna signal
    # candle ke "beech mein" ban jayega aur band hone tak badalta rahega. ----
    if len(df) > 0:
        last_candle_close_time = df["timestamp"].iloc[-1] + pd.Timedelta(milliseconds=_timeframe_ms(timeframe))
        now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
        if last_candle_close_time > now_utc:
            df = df.iloc[:-1].reset_index(drop=True)

    return df.tail(limit).reset_index(drop=True)


def _timeframe_ms(timeframe):
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    mult = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return value * mult[unit]


def fetch_all_data(coin_list, timeframes, candle_limits):
    """
    Sab coins x sab timeframes ka data fetch karta hai.
    Return: dict[symbol][timeframe] = DataFrame
    """
    exchange = get_exchange()
    data = {}
    total = len(coin_list) * len(timeframes)
    done = 0

    for symbol in coin_list:
        data[symbol] = {}
        for tf in timeframes:
            done += 1
            print(f"[{done}/{total}] Fetching {symbol} {tf} ...")
            df = fetch_ohlcv(exchange, symbol, tf, candle_limits[tf])
            if df is not None and len(df) > 50:
                data[symbol][tf] = df
            else:
                print(f"  [SKIP] {symbol} {tf}: not enough data")

    return data