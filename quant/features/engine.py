"""
src/features/engine.py
Orchestrates feature generation. Excludes raw 1h_ and daily_ columns from feature set.
"""
import polars as pl
import logging
from config import config
from quant.features.baseline import compute_baseline_features, load_baseline_feature_names
from quant.features.expansion import expand_features, add_cross_timeframe_interactions
from quant.features.htf_context import add_htf_context_features
from quant.features.target import add_target_5m, drop_incomplete_target

logger = logging.getLogger(__name__)

def generate_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Full feature engineering pipeline for three-stream HTF data.
    Assumes df already contains aligned 1h and daily columns (prefixed 1h_, daily_).
    Final feature columns are those starting with: feature_, ratio_, pair_, zscore, cross_, htf_
    (raw 1h_ and daily_ are excluded).
    """
    df = compute_baseline_features(df)
    baseline_names = load_baseline_feature_names()
    baseline_cols = [c for c in baseline_names if c in df.columns]

    df = add_htf_context_features(df)   # adds htf_* columns

    df = expand_features(df, baseline_cols)   # adds regime, ratios, pairwise, etc.

    # After expansion, add cross-timeframe interactions (using only derived features)
    # But exclude raw 1h_/daily_ columns from ltf features
    htf_cols = [c for c in df.columns if c.startswith("htf_")]
    ltf_candidate = [c for c in df.columns if c.startswith(("feature_", "ratio_", "pair_", "zscore", "cross_")) 
                     and not c.startswith(("1h_", "daily_"))]
    # Also ensure we don't already have cross_ from previous runs (avoid duplication)
    ltf_cols = [c for c in ltf_candidate if not c.startswith("cross_")]
    
    if htf_cols and ltf_cols:
        df = add_cross_timeframe_interactions(df, ltf_cols, htf_cols)
    
    # Add target (binary for classification)
    df = add_target_5m(df)
    df = drop_incomplete_target(df)

    # Ensure all feature columns are float32
    feature_cols = [c for c in df.columns if c.startswith(("feature_", "ratio_", "pair_", "zscore", "cross_", "htf_"))]
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c in feature_cols])
    logger.info(f"Final feature matrix has {len(feature_cols)} features.")
    return df