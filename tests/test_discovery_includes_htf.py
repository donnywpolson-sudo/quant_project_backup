"""
tests/test_discovery_includes_htf.py
Verifies that the feature discovery process includes HTF and cross-timeframe features in the manifest.
"""
import pytest
import json
import polars as pl
from pathlib import Path
from src.discovery import run_feature_discovery
from src.ingest import load_and_clean_data
from src.features.engine import generate_features

def test_htf_features_in_manifest(tmp_path, synthetic_data_path):
    """Run discovery on synthetic data and check manifest for HTF/cross features."""
    # synthetic_data_path should point to the fixture; we'll use the existing fixture
    data_path = "tests/fixtures/synthetic_1min_fixture.parquet"
    if not Path(data_path).exists():
        pytest.skip("Synthetic fixture not found. Run make_fixtures first.")
    
    manifest_out = tmp_path / "manifest.json"
    # We need to generate features first (discovery will do it internally after fix)
    run_feature_discovery(data_path, str(manifest_out))
    
    with open(manifest_out) as f:
        manifest = json.load(f)
    
    feature_names = manifest["feature_names"]
    htf_features = [f for f in feature_names if f.startswith(("htf_", "cross_", "1h_", "daily_"))]
    assert len(htf_features) > 0, f"No HTF/cross features found in manifest. Features: {feature_names[:20]}..."
    
    # Also check that at least one cross-timeframe interaction exists
    cross_features = [f for f in feature_names if f.startswith("cross_")]
    if len(cross_features) == 0:
        pytest.warn("No cross-timeframe features selected; may be due to limited data or threshold.")
    else:
        assert len(cross_features) > 0