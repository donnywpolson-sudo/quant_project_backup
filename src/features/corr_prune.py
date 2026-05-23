"""
src/features/corr_prune.py
Deterministic correlation pruning.
Implements Section 15: float64 computation, keeps first occurrence, drops subsequent.
"""
import logging
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

def correlation_prune(df: pl.DataFrame, feature_cols: list, threshold: float = 0.90) -> list:
    """
    Deterministic correlation pruning.
    CRITICAL: Must be executed strictly on the TRAIN split to prevent look-ahead bias.
    Uses float64 for computation precision, but keeps data intact for float32 pipelines.
    """
    if df.height == 0 or len(feature_cols) == 0:
        return feature_cols

    logger.info(f"Running correlation pruning on {len(feature_cols)} features (threshold={threshold})...")
    
    # Section 15: Upcast to float64 exclusively for the matrix computation
    X = df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float64)

    # rowvar=False ensures columns are treated as variables
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.corrcoef(X, rowvar=False)

    keep = []
    dropped = set()

    for i, f in enumerate(feature_cols):
        if f in dropped:
            continue

        keep.append(f)

        # Check all subsequent features against the kept feature
        for j in range(i + 1, len(feature_cols)):
            if feature_cols[j] in dropped:
                continue
            
            val = corr[i, j]
            # Drop if correlation exceeds threshold (ignoring NaNs from zero-variance columns)
            if not np.isnan(val) and abs(val) > threshold:
                dropped.add(feature_cols[j])

    logger.info(f"Correlation pruning dropped {len(dropped)} features. Kept {len(keep)} features.")
    return keep