"""
src/discovery.py
Phase 1: Feature Discovery using ExtraTreesClassifier.
Orchestrates the identification of the most predictive feature subset.
"""
import os
import json
import logging
import numpy as np
import polars as pl
from sklearn.ensemble import ExtraTreesClassifier
from config import config
from src.features.engine import generate_features

# Setup logging
logger = logging.getLogger(__name__)

def check_memory_safety():
    """
    Check RSS usage against config limits to prevent Out-Of-Memory termination.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        # Check against maximum memory cap defined in config
        if mem_bytes > getattr(config, "RAM_CAP_BYTES", 13.5 * 1024 * 1024 * 1024):
            logger.error(f"Memory threshold breached: {mem_bytes / (1024**3):.2f} GB RSS. Aborting execution.")
            raise MemoryError("Pipeline auto-aborted due to strict memory constraints.")
    except ImportError:
        pass

def run_feature_discovery(data_path: str, manifest_out: str):
    """
    Phase 1: Feature Discovery via ExtraTreesClassifier.
    Enforces absolute Look-Ahead Bias protection by strictly running discovery 
    on a historical training subset (e.g., the first 50-60% of data).
    """
    logger.info(f"Phase 1: Initializing Feature Discovery from {data_path}")
    check_memory_safety()

    # 1. Ingest Data and guarantee temporal sorting
    df_all = pl.read_parquet(data_path)
    if "ts_event" in df_all.columns:
        df_all = df_all.sort("ts_event")
    
    total_rows = df_all.height
    
    # 2. Slice historical data strictly according to a designated discovery split percentage
    split_pct = getattr(config, "FEATURE_DISCOVERY_SPLIT_PCT", 0.50)
    discovery_cutoff = int(total_rows * split_pct)
    df_discovery = df_all.slice(0, discovery_cutoff)
    
    logger.info(f"Look-ahead safety: Discovery isolated to first {split_pct*100}% of history ({df_discovery.height} / {total_rows} rows).")

    # 3. Generate candidate features
    df_features = generate_features(df_discovery)
    check_memory_safety()

    # 4. Identify feature versus target layout
    feature_cols = [c for c in df_features.columns if c.startswith("feature_")]
    target_col = "target" if "target" in df_features.columns else None
    
    if not target_col:
        target_col = df_features.columns[-1]
        if target_col in feature_cols:
            feature_cols.remove(target_col)

    logger.info(f"Extracted {len(feature_cols)} candidate features. Fitting ExtraTrees against target: '{target_col}'")

    # 5. FIX: Convert to numpy array and cast type separately to avoid 'dtype' argument error
    X = df_features.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y = df_features.select(target_col).fill_null(0.0).to_numpy().astype(np.float32).ravel()

    # 6. Instantiate and train a deterministic ExtraTreesClassifier model
    model = ExtraTreesClassifier(
        n_estimators=100,
        random_state=config.SEED,
        n_jobs=1,
        bootstrap=False
    )
    model.fit(X, y)
    
    # 7. Map, sort and filter feature importances
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    num_to_select = getattr(config, "NUM_TOP_FEATURES_TO_SELECT", 15)
    top_features = [feature_cols[idx] for idx in sorted_indices[:num_to_select]]
    
    logger.info(f"Top {num_to_select} features identified by ExtraTrees: {top_features}")

    # 8. Construct manifest file
    manifest_data = {
        "version": "1.0.0",
        "feature_names": top_features,
        "dtypes": {feat: "Float32" for feat in top_features},
        "selection_metric": "ExtraTrees_Feature_Importance",
        "importances": {feature_cols[idx]: float(importances[idx]) for idx in sorted_indices[:num_to_select]}
    }

    os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
    with open(manifest_out, "w") as f:
        json.dump(manifest_data, f, indent=4)
        
    logger.info(f"Phase 1 Complete. Frozen feature manifest written to {manifest_out}")

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="Standalone Phase 1 Feature Discovery Wrapper")
    parser.add_argument("--data", required=True, help="Path to raw source parquet")
    parser.add_argument("--out", required=True, help="Target path for manifest.json output")
    
    args = parser.parse_args()
    run_feature_discovery(args.data, args.out)