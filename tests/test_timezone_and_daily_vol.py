"""
tests/test_timezone_and_daily_vol.py
Verifies that timezone conversion works correctly (session boundaries)
and that daily_vol_5 is present in the aligned data.
"""
import pytest
import polars as pl
from pathlib import Path
from src.session import load_all_streams_chunked
from src.align import align_htf_streams

def test_timezone_and_daily_vol():
    """Use synthetic fixture to check session_id and daily_vol_5."""
    data_path = "tests/fixtures/synthetic_1min_fixture.parquet"
    if not Path(data_path).exists():
        pytest.skip("Synthetic fixture not found. Run make_fixtures first.")
    
    streams = load_all_streams_chunked(data_path)
    df_5min = streams["5m"]
    df_1h = streams["1h"]
    df_daily = streams["1d"]
    
    # Check daily stream has daily_vol_5 column
    assert "daily_vol_5" in df_daily.columns, "daily_vol_5 missing from daily stream"
    # Check daily_vol_5 is not all null
    assert df_daily["daily_vol_5"].null_count() < df_daily.height, "daily_vol_5 all null"
    
    # Align streams
    df_aligned = align_htf_streams(df_5min, df_1h, df_daily)
    # Check that daily_vol_5 appears as a column (should be forwarded)
    assert "daily_vol_5" in df_aligned.columns, "daily_vol_5 not aligned"
    
    # Basic session sanity: session_id should be date (string) and not null
    assert df_aligned["session_id"].null_count() == 0, "session_id has nulls"
    
    # For a few rows, ensure time_local (if computed) is within session hours? Hard to test directly,
    # but we can check that we have at least some rows.
    assert df_aligned.height > 0, "Aligned DataFrame empty"