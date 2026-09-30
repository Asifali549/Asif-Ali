import numpy as np
from partial_profit_filter import compute_partial_target, simulate_partial_profit_trade


def test_compute_partial_target():
    assert round(compute_partial_target(100, 1.5), 2) == 101.5


def test_direct_stop_no_partial():
    n = 10
    o = np.array([100.0] * n)
    h = np.array([100.5] * n)
    l = np.array([94.0] * n)
    c = np.array([99.0] * n)
    ce_stops = np.array([95.0] * n)

    result = simulate_partial_profit_trade(
        o, h, l, c, entry_bar=0, entry_price=100.0, initial_stop=95.0,
        ce_stops=ce_stops, max_hold=10, partial_target_pct=1.5, fee=0.001, slip=0.001,
    )
    assert result["partial_taken"] == False
    assert result["exit_reason"] == "CE_STOP_FULL"
    assert result["return_pct"] < 0


def test_partial_hit_then_further_gain():
    n = 10
    o = np.array([100.0, 102.0, 104.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0])
    h = np.array([102.0, 103.0, 105.0, 107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0])
    l = np.array([99.5, 101.0, 103.0, 105.0, 103.5, 102.5, 101.5, 100.5, 99.5, 99.0])
    c = np.array([101.5, 102.5, 104.5, 106.5, 104.0, 103.0, 102.0, 101.0, 100.0, 99.5])
    ce_stops = np.array([95.0, 96.0, 98.0, 100.0, 101.0, 102.0, 101.5, 101.0, 100.5, 100.0])

    result = simulate_partial_profit_trade(
        o, h, l, c, entry_bar=0, entry_price=100.0, initial_stop=95.0,
        ce_stops=ce_stops, max_hold=10, partial_target_pct=1.5, fee=0.001, slip=0.001,
    )
    assert result["partial_taken"] == True
    assert result["return_pct"] > 0


def test_partial_hit_then_reverses_to_breakeven():
    n = 10
    o = np.array([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 95.0, 95.0, 95.0, 95.0])
    h = np.array([102.0, 100.0, 99.0, 98.0, 97.0, 96.0, 96.0, 96.0, 96.0, 96.0])
    l = np.array([99.5, 98.5, 97.5, 99.5, 95.5, 94.5, 94.5, 94.5, 94.5, 94.5])
    c = np.array([101.5, 99.0, 98.0, 100.0, 96.0, 95.0, 95.0, 95.0, 95.0, 95.0])
    ce_stops = np.array([90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0])

    result = simulate_partial_profit_trade(
        o, h, l, c, entry_bar=0, entry_price=100.0, initial_stop=90.0,
        ce_stops=ce_stops, max_hold=10, partial_target_pct=1.5, fee=0.001, slip=0.001,
    )
    assert result["partial_taken"] == True
    assert result["return_pct"] > -2.0


def test_time_exit_without_partial():
    n = 5
    o = np.array([100.0, 100.2, 100.4, 100.3, 100.5])
    h = np.array([100.5, 100.6, 100.7, 100.6, 100.8])
    l = np.array([99.8, 99.9, 100.0, 100.0, 100.2])
    c = np.array([100.2, 100.4, 100.3, 100.5, 100.6])
    ce_stops = np.array([95.0, 95.0, 95.0, 95.0, 95.0])

    result = simulate_partial_profit_trade(
        o, h, l, c, entry_bar=0, entry_price=100.0, initial_stop=95.0,
        ce_stops=ce_stops, max_hold=5, partial_target_pct=1.5, fee=0.001, slip=0.001,
    )
    assert result["exit_reason"] == "TIME"
    assert result["partial_taken"] == False


if __name__ == "__main__":
    tests = [
        test_compute_partial_target,
        test_direct_stop_no_partial,
        test_partial_hit_then_further_gain,
        test_partial_hit_then_reverses_to_breakeven,
        test_time_exit_without_partial,
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