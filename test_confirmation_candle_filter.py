from confirmation_candle_filter import passes_confirmation, get_confirmed_entry_bar


def test_bullish_confirmation_passes():
    assert passes_confirmation(o=100, h=105, l=99, c=103) == True


def test_bearish_confirmation_fails():
    assert passes_confirmation(o=100, h=101, l=95, c=97) == False


def test_flat_close_equals_open_passes():
    assert passes_confirmation(o=100, h=102, l=99, c=100) == True


def test_get_confirmed_entry_bar_normal_case():
    conf_idx, entry_idx = get_confirmed_entry_bar(signal_idx=10, n=100)
    assert conf_idx == 11
    assert entry_idx == 12


def test_get_confirmed_entry_bar_near_end_of_data():
    conf_idx, entry_idx = get_confirmed_entry_bar(signal_idx=98, n=100)
    assert conf_idx is None
    assert entry_idx is None


def test_get_confirmed_entry_bar_exact_boundary():
    conf_idx, entry_idx = get_confirmed_entry_bar(signal_idx=97, n=100)
    assert conf_idx == 98
    assert entry_idx == 99


if __name__ == "__main__":
    tests = [
        test_bullish_confirmation_passes,
        test_bearish_confirmation_fails,
        test_flat_close_equals_open_passes,
        test_get_confirmed_entry_bar_normal_case,
        test_get_confirmed_entry_bar_near_end_of_data,
        test_get_confirmed_entry_bar_exact_boundary,
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