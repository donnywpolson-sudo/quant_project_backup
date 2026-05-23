"""
tests/test_walkforward.py
Validates the walk-forward simulation engine with the new date‑based rolling window.
"""
import pytest
import polars as pl
import numpy as np
from src.walkforward import run_walkforward

@pytest.fixture
def sample_data():
    """Generates 5 days of 5‑min data with deterministic features."""
    dates = [f"2023-01-{i:02d}" for i in range(1, 6)]
    rows = []
    for d in dates:
        for hour in range(18, 22):  # 18:00 to 22:00
            for minute in range(0, 60, 5):
                ts = pl.datetime(int(d[:4]), int(d[5:7]), int(d[8:10]), hour, minute, time_zone="UTC")
                rows.append({"ts_event": ts, "feature_a": float(hour), "feature_b": float(minute), "target_5m": 0.01})
    df = pl.DataFrame(rows)
    return df

def test_walkforward_runs(sample_data):
    """Ensures walkforward completes without errors."""
    feature_cols = ["feature_a", "feature_b"]
    result = run_walkforward(sample_data, feature_cols, "target_5m")
    assert "prediction" in result.columns
    assert result["prediction"].dtype == pl.Float32
    assert result.height > 0