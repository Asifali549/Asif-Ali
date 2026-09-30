import pandas as pd
from walk_forward_splitter import make_period_boundaries, assign_trades_to_periods


def test_make_period_boundaries_three_equal_parts():
    start = pd.Timestamp("2026-01-01")
    end = pd.Timestamp("2026-01-31")
    boundaries = make_period_boundaries(start, end, 3)
    assert len(boundaries) == 3
    assert boundaries[0][0] == start
    assert boundaries[2][1] == end
    for p_start, p_end in boundaries:
        span_days = (p_end - p_start).days
        assert 9 <= span_days <= 10


def test_assign_trades_to_correct_periods():
    start = pd.Timestamp("2026-01-01")
    end = pd.Timestamp("2026-01-31")
    boundaries = make_period_boundaries(start, end, 3)

    trades = [
        {"entry_time": pd.Timestamp("2026-01-02"), "id": "early"},
        {"entry_time": pd.Timestamp("2026-01-15"), "id": "middle"},
        {"entry_time": pd.Timestamp("2026-01-29"), "id": "late"},
    ]
    periods = assign_trades_to_periods(trades, boundaries)
    assert len(periods) == 3
    assert periods[0][0]["id"] == "early"
    assert periods[1][0]["id"] == "middle"
    assert periods[2][0]["id"] == "late"


def test_last_trade_at_exact_end_goes_to_last_period():
    start = pd.Timestamp("2026-01-01")
    end = pd.Timestamp("2026-01-31")
    boundaries = make_period_boundaries(start, end, 2)

    trades = [{"entry_time": end, "id": "boundary"}]
    periods = assign_trades_to_periods(trades, boundaries)
    assert len(periods[1]) == 1
    assert periods[1][0]["id"] == "boundary"


def test_empty_trades_list():
    boundaries = make_period_boundaries(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-10"), 2)
    periods = assign_trades_to_periods([], boundaries)
    assert periods == [[], []]


if __name__ == "__main__":
    tests = [
        test_make_period_boundaries_three_equal_parts,
        test_assign_trades_to_correct_periods,
        test_last_trade_at_exact_end_goes_to_last_period,
        test_empty_trades_list,
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