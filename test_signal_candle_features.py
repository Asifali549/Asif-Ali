import numpy as np
from signal_candle_features import (
    candle_body_pct, candle_upper_wick_pct, candle_lower_wick_pct,
    is_bearish, candle_move_pct, extract_signal_features,
)


def test_full_body_candle_100pct():
    assert candle_body_pct(100, 110, 100, 110) == 100.0


def test_doji_zero_body():
    assert candle_body_pct(105, 110, 100, 105) == 0.0


def test_upper_wick_calc():
    assert candle_upper_wick_pct(100, 110, 100, 105) == 50.0


def test_lower_wick_calc():
    assert candle_lower_wick_pct(105, 110, 100, 110) == 50.0


def test_is_bearish_true():
    assert is_bearish(110, 111, 99, 100) == True


def test_is_bearish_false():
    assert is_bearish(100, 111, 99, 110) == False


def test_candle_move_pct_positive():
    assert round(candle_move_pct(100, 105), 2) == 5.0


def test_candle_move_pct_negative():
    assert round(candle_move_pct(100, 95), 2) == -5.0


def test_extract_features_bearish_next_candle():
    o = np.array([100, 101, 102, 103, 98])
    h = np.array([101, 102, 103.5, 104, 99])
    l = np.array([99.5, 100.5, 101.5, 97, 96])
    c = np.array([100.8, 101.8, 103, 97.5, 97])
    v = np.array([1000, 1000, 1000, 5000, 4000])

    df_values = {"open": o, "high": h, "low": l, "close": c, "volume": v}
    features = extract_signal_features(df_values, signal_idx=2, entry_idx=3,
                                        atr_at_signal=2.0, lookahead=2)

    assert features["next_candle_bearish"] == True
    assert features["next_candle_move_pct"] < 0
    assert features["signal_volume_ratio"] > 0


def test_extract_features_bullish_next_candle():
    o = np.array([100, 101, 102, 103, 106])
    h = np.array([101, 102, 103.5, 106.5, 108])
    l = np.array([99.5, 100.5, 101.5, 102.8, 105.5])
    c = np.array([100.8, 101.8, 103, 106, 107.5])
    v = np.array([1000, 1000, 1000, 2000, 2200])

    df_values = {"open": o, "high": h, "low": l, "close": c, "volume": v}
    features = extract_signal_features(df_values, signal_idx=2, entry_idx=3,
                                        atr_at_signal=2.0, lookahead=2)

    assert features["next_candle_bearish"] == False
    assert features["next_candle_move_pct"] > 0
    assert features["worst_move_pct_in_lookahead"] >= 0


if __name__ == "__main__":
    tests = [
        test_full_body_candle_100pct,
        test_doji_zero_body,
        test_upper_wick_calc,
        test_lower_wick_calc,
        test_is_bearish_true,
        test_is_bearish_false,
        test_candle_move_pct_positive,
        test_candle_move_pct_negative,
        test_extract_features_bearish_next_candle,
        test_extract_features_bullish_next_candle,
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