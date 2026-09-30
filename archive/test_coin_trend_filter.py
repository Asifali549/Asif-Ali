import pandas as pd
from coin_trend_filter import compute_ema_trend_series, is_trend_aligned


def make_4h_df(prices, start="2026-01-01"):
    dates = pd.date_range(start, periods=len(prices), freq="4h")
    return pd.DataFrame({"timestamp": dates, "close": prices})


def test_bullish_when_rising_price():
    prices = [100 + i * 2 for i in range(80)]
    df = make_4h_df(prices)
    trend = compute_ema_trend_series(df, ema_period=50)
    assert trend.iloc[-1] == True


def test_bearish_when_falling_price():
    prices = [500 - i * 2 for i in range(80)]
    df = make_4h_df(prices)
    trend = compute_ema_trend_series(df, ema_period=50)
    assert trend.iloc[-1] == False


def test_signal_lookup_uses_last_available_4h_bar():
    prices = [100 + i * 2 for i in range(80)]
    df = make_4h_df(prices)
    trend = compute_ema_trend_series(df, ema_period=50)

    last_4h_ts = pd.Timestamp(df["timestamp"].iloc[-1])
    signal_ts = last_4h_ts + pd.Timedelta(hours=2)
    result = is_trend_aligned(trend, signal_ts)
    assert result == True


def test_missing_data_defaults_to_true():
    prices = [100 + i * 2 for i in range(80)]
    df = make_4h_df(prices)
    trend = compute_ema_trend_series(df, ema_period=50)

    very_old_ts = pd.Timestamp("2000-01-01")
    result = is_trend_aligned(trend, very_old_ts)
    assert result == True


def test_empty_series_defaults_to_true():
    empty = pd.Series([], dtype=bool)
    result = is_trend_aligned(empty, pd.Timestamp("2026-01-01"))
    assert result == True


if __name__ == "__main__":
    tests = [
        test_bullish_when_rising_price,
        test_bearish_when_falling_price,
        test_signal_lookup_uses_last_available_4h_bar,
        test_missing_data_defaults_to_true,
        test_empty_series_defaults_to_true,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -> {e}")
    print(f"\n{passed}/{len(tests)} tests passed")