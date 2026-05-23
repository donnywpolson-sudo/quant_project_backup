"""
src/features/engine.py
Orchestrates generation of baseline features, HTF context, expansion, and target.
"""
import polars as pl
import logging
from config import config
from src.features.baseline import compute_baseline_features, load_baseline_feature_names
from src.features.expansion import expand_features, add_cross_timeframe_interactions
from src.features.htf_context import add_htf_context_features
from src.features.target import add_target_5m, drop_incomplete_target

logger = logging.getLogger(__name__)

def generate_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Full feature engineering pipeline for three-stream HTF data.
    Assumes df already contains aligned 1h and daily columns (prefixed 1h_, daily_).
    """
    print("DEBUG: generate_features - computing baseline...", flush=True)
    df = compute_baseline_features(df)
    baseline_names = load_baseline_feature_names()
    baseline_cols = [c for c in baseline_names if c in df.columns]
    print("DEBUG: baseline done", flush=True)

    print("DEBUG: adding HTF context features...", flush=True)
    df = add_htf_context_features(df)
    print("DEBUG: HTF context done", flush=True)

    print("DEBUG: expanding features (ratios, z-scores, regime, pairwise)...", flush=True)
    df = expand_features(df, baseline_cols)
    print("DEBUG: expansion done", flush=True)

    # After expand_features, add cross-timeframe interactions explicitly
    htf_cols = [c for c in df.columns if c.startswith(("1h_", "daily_", "htf_"))]
    ltf_cols = [c for c in df.columns if c.startswith(("feature_", "ratio_", "pair_", "zscore")) and c not in htf_cols]
    if htf_cols and ltf_cols:
        print("DEBUG: adding cross-timeframe interactions...", flush=True)
        df = add_cross_timeframe_interactions(df, ltf_cols, htf_cols)
        print("DEBUG: cross-timeframe done", flush=True)

    print("DEBUG: adding target_5m...", flush=True)
    df = add_target_5m(df)
    df = drop_incomplete_target(df)
    print("DEBUG: target done", flush=True)

    # Ensure all feature columns are float32
    feature_cols = [c for c in df.columns if c.startswith(("feature_", "ratio_", "pair_", "zscore", "cross_", "htf_", "1h_", "daily_"))]
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c in feature_cols])
    logger.info(f"Final feature matrix has {len(feature_cols)} features.")
    print(f"DEBUG: generate_features finished with {len(feature_cols)} features", flush=True)
    return df