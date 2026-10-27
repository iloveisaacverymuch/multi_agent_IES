import numpy as np

from scripts.evaluate_full_year_algorithm_comparison import month_for_hour
from scripts.prepare_full_year_data import seasonal_test_rows


def test_seasonal_test_rows_match_expected_calendar_days():
    values = np.arange(8760, dtype=np.float64).reshape(-1, 1)
    selected = seasonal_test_rows(values).reshape(-1)
    starts = [20 * 24, (31 + 28 + 31 + 20) * 24, (181 + 20) * 24, (273 + 20) * 24]
    expected = np.concatenate([np.arange(start, start + 24) for start in starts])
    np.testing.assert_array_equal(selected, expected)


def test_month_for_hour_covers_complete_non_leap_year():
    assert month_for_hour(0) == 1
    assert month_for_hour(31 * 24 - 1) == 1
    assert month_for_hour(31 * 24) == 2
    assert month_for_hour(8760 - 1) == 12
