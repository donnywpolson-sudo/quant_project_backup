"""Functional test for quant/continuous_contract.py (Finding #12)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime
import polars as pl
from quant.continuous_contract import (
    compute_roll_dates,
    build_ratio_adjusted_series,
    apply_splice,
    build_continuous_series,
)


def test_compute_roll_dates_es():
    """ES should have 4 quarterly roll dates in 2024 (HMUZ)."""
    rolls = compute_roll_dates("ES", datetime(2024, 1, 1), datetime(2024, 12, 31))
    assert rolls.height == 4, f"Expected 4 ES rolls, got {rolls.height}"
    # Verify they are Thursdays
    for row in rolls.iter_rows(named=True):
        assert row["roll_date"].weekday() == 3, (
            f"ES roll {row['roll_date']} should be Thursday"
        )
    print("  test_compute_roll_dates_es: PASS")


def test_compute_roll_dates_cl():
    """CL should have 12 monthly roll dates in 2024."""
    rolls = compute_roll_dates("CL", datetime(2024, 1, 1), datetime(2024, 12, 31))
    assert rolls.height == 12, f"Expected 12 CL rolls, got {rolls.height}"
    print("  test_compute_roll_dates_cl: PASS")


def test_build_continuous_series():
    """Synthetic data with a single roll should have all required columns."""
    ts = pl.datetime_range(
        datetime(2024, 1, 2), datetime(2024, 3, 30), "5m", eager=True
    ).dt.replace_time_zone("UTC")
    n = len(ts)
    close_vals = [4500.0 + i * 0.1 for i in range(n)]
    df = pl.DataFrame(
        {
            "ts_event": ts,
            "open": close_vals,
            "high": [v + 2.0 for v in close_vals],
            "low": [v - 2.0 for v in close_vals],
            "close": close_vals,
            "volume": [1000] * n,
            "session_id": ["sess1"] * n,
        }
    )

    result = build_continuous_series(df, "ES", contract_multiplier=50.0)

    required_cols = [
        "continuous_price",
        "adjustment_factor",
        "contract_month",
        "contract_multiplier",
    ]
    for col in required_cols:
        assert col in result.columns, f"Missing column: {col}"

    # contract_multiplier should be 50.0 for ES
    assert result["contract_multiplier"][0] == 50.0, (
        f"Expected 50.0, got {result['contract_multiplier'][0]}"
    )

    # continuous_price should equal close when no rolls occur in range
    # (Jan-Mar 2024 has a March roll, so there may be a small adjustment)
    # At minimum, continuous_price should be non-null
    assert result["continuous_price"].null_count() == 0, (
        "continuous_price has nulls"
    )
    assert result["adjustment_factor"].null_count() == 0, (
        "adjustment_factor has nulls"
    )

    print("  test_build_continuous_series: PASS")


def test_apply_splice():
    """apply_splice should join cumulative_factor and compute continuous_price."""
    ts = [datetime(2024, 1, 1, 9, 30), datetime(2024, 1, 1, 9, 35)]
    df = pl.DataFrame(
        {
            "ts_event": ts,
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [500, 600],
        }
    )
    adjustments = pl.DataFrame(
        {
            "ts_event": [datetime(2024, 1, 1, 9, 30)],
            "adjustment_factor": [1.05],
            "cumulative_factor": [1.05],
        }
    )

    result = apply_splice(df, adjustments)

    assert "continuous_price" in result.columns
    # Row 0: 101.0 * 1.05 = 106.05 (close at ts where factor=1.05)
    # Row 1: 102.0 * 1.05 = 107.10 (forward-filled factor)
    expected_0 = 101.0 * 1.05
    expected_1 = 102.0 * 1.05
    assert abs(result["continuous_price"][0] - expected_0) < 0.01, (
        f"Expected ~{expected_0}, got {result['continuous_price'][0]}"
    )
    assert abs(result["continuous_price"][1] - expected_1) < 0.01, (
        f"Expected ~{expected_1}, got {result['continuous_price'][1]}"
    )
    print("  test_apply_splice: PASS")


if __name__ == "__main__":
    print("Running continuous_contract tests...")
    test_compute_roll_dates_es()
    test_compute_roll_dates_cl()
    test_build_continuous_series()
    test_apply_splice()
    print("All continuous_contract tests PASSED.")