"""
src/features/variance_filter.py
Remove constant (zero variance) features from training data.
"""
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import logging

logger = logging.getLogger(__name__)

def remove_constant_features(df, feature_cols, threshold=1e-9):
    """
    Returns list of feature columns that have variance > threshold.
    Fits only on the provided DataFrame (train fold).
    """
    X = df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    non_const_mask = selector.get_support()
    kept = [c for c, keep in zip(feature_cols, non_const_mask) if keep]
    removed = len(feature_cols) - len(kept)
    if removed > 0:
        logger.info(f"Removed {removed} constant features. Remaining: {len(kept)}")
    return kept