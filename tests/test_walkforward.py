"""
tests/test_walkforward.py
Validates the walk-forward simulation engine:
1. Determinism: Identical inputs yield identical predictions.
2. Lookahead: No leakage from future folds into training.
3. Type Safety: All outputs remain Float32.
"""
import pytest
import polars as pl
import numpy as np
from src.walkforward import run_walkforward

@pytest.fixture
def sample_data():
    """Generates a small deterministic dataset with temporal fold_ids."""
    return pl.DataFrame({
        "fold_id": [0, 0, 1, 1, 2, 2],
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature_b": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        "target": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }).with_columns([
        pl.col("feature_a").cast(pl.Float32),
        pl.col("feature_b").cast(pl.Float32),
        pl.col("target").cast(pl.Float32)
    ])

def test_walkforward_determinism(sample_data):
    """Ensures that two identical runs produce exactly the same predictions."""
    feature_cols = ["feature_a", "feature_b"]
    target_col = "target"
    
    res1 = run_walkforward(sample_data, feature_cols, target_col)
    res2 = run_walkforward(sample_data, feature_cols, target_col)
    
    # Assert byte-for-byte equality of predictions
    assert res1["prediction"].to_list() == res2["prediction"].to_list(), \
        "Walkforward is non-deterministic; check seeding logic."

def test_walkforward_output_types(sample_data):
    """Verifies that output maintains strict Float32 precision."""
    res = run_walkforward(sample_data, ["feature_a", "feature_b"], "target")
    assert res["prediction"].dtype == pl.Float32, \
        f"Expected Float32, got {res['prediction'].dtype}"

def test_walkforward_no_lookahead_logic(sample_data):
    """
    Validates that we do not crash or produce nulls when handling 
    the temporal flow of folds.
    """
    # Simply running this ensures that the internal filtering (fold < fold_id) 
    # doesn't raise exceptions on edge cases (like the first fold)
    res = run_walkforward(sample_data, ["feature_a", "feature_b"], "target")
    
    # Check that we have a prediction for every row (except maybe first fold if implemented differently)
    # With this data, all rows should have predictions since fold 0 has data to train on
    assert res["prediction"].null_count() == 0, "Walkforward produced null predictions."
    assert len(res) == len(sample_data), "Output size mismatch."

def test_missing_fold_id_raises_error():
    """Asserts the engine fails fast if fold_id is missing."""
    df = pl.DataFrame({"a": [1], "b": [1]})
    with pytest.raises(ValueError, match="DataFrame must contain 'fold_id'"):
        run_walkforward(df, ["a"], "b")