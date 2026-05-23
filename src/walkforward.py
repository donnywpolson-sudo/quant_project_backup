"""
src/walkforward.py
Deterministic, no-lookahead walk-forward validation for RidgeClassifier models.
Strictly enforced: Float32 precision, CPU-only, deterministic seeds, and NaN handling.
"""
import logging
import hashlib
import polars as pl
import numpy as np
from sklearn.linear_model import RidgeClassifier
from config import config

logger = logging.getLogger(__name__)

def get_fold_seed(fold_index: int) -> int:
    """
    Generates a cryptographically stable seed for a specific fold.
    """
    seed_str = f"{config.SEED}_fold_{fold_index}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)

def assign_temporal_folds(df: pl.DataFrame) -> pl.DataFrame:
    """
    Dynamically assigns fold_id based on ts_event.
    Uses dt.truncate and removes timezone to ensure compatibility with list conversion.
    """
    if "fold_id" in df.columns:
        return df
        
    if "ts_event" not in df.columns:
        raise ValueError("DataFrame must contain 'ts_event' to auto-assign temporal folds.")

    logger.info("Auto-assigning fold_id based on ts_event...")
    
    # Truncate to '1mo' (monthly) and strip timezone to naive to prevent 
    # serialization issues when converting to list for unique sorting.
    df = df.with_columns(
        pl.col("ts_event")
        .dt.truncate("1mo")
        .dt.replace_time_zone(None)
        .alias("fold_id")
    )
    
    return df

def train_and_predict(train_df: pl.DataFrame, test_df: pl.DataFrame, 
                      feature_cols: list, target_col: str, fold_index: int) -> pl.Series:
    """
    Trains a RidgeClassifier and predicts on the fold test set.
    Includes fill_null(0.0) to handle rolling window NaN artifacts.
    """
    # Cast features and target to Float32 for compatibility
    X_train = train_df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y_train = train_df.select(target_col).fill_null(0.0).to_numpy().astype(np.float32).ravel()
    X_test = test_df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    
    # Initialize RidgeClassifier for classification
    model = RidgeClassifier(
        alpha=1.0, 
        random_state=get_fold_seed(fold_index)
    )
    
    model.fit(X_train, y_train)
    
    # Predict classes
    preds = model.predict(X_test)
    
    return pl.Series("prediction", preds, dtype=pl.Float32)

def run_walkforward(df: pl.DataFrame, feature_cols: list, target_col: str) -> pl.DataFrame:
    """
    Orchestrates the walk-forward validation loop over unique fold_ids.
    """
    # 1. Ensure fold_id exists
    df = assign_temporal_folds(df)
        
    logger.info("Starting walk-forward simulation...")
    
    unique_folds = sorted(df["fold_id"].unique().to_list())
    all_results = []
    
    for i, fold_id in enumerate(unique_folds):
        # 2. Strictly split: Train on EVERYTHING before this fold, Test on THIS fold.
        train_df = df.filter(pl.col("fold_id") < fold_id)
        test_df = df.filter(pl.col("fold_id") == fold_id)
        
        # 3. Guard against empty training sets
        if train_df.height > 0:
            logger.info(f"Processing fold {fold_id} (Train: {train_df.height}, Test: {test_df.height})")
            
            # Get predictions
            preds = train_and_predict(train_df, test_df, feature_cols, target_col, i)
            
            # Attach prediction to test data
            test_result = test_df.with_columns(preds.alias("prediction"))
            all_results.append(test_result)
        else:
            logger.warning(f"Skipping fold {fold_id}: No training data available.")
            
    # 4. Final safety check before concat
    if not all_results:
        raise ValueError("Walk-forward failed: No folds could be processed.")
        
    return pl.concat(all_results)