import pytest
import polars as pl
from pathlib import Path

def test_column_types():
    # Update the path to match the filename used in src/cli.py
    path = Path("artifacts/baseline_feature_matrix.parquet")
    
    if not path.exists():
        pytest.skip(f"Artifacts not found at {path}")
        
    df = pl.read_parquet(path)
    # Ensure all features are float32 as per spec
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    for col in feature_cols:
        assert df[col].dtype == pl.Float32, f"{col} is not Float32"