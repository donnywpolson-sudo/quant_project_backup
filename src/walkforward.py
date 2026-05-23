"""
src/walkforward.py
Walkforward with LogisticRegression (classification) predicting sign of next 5-bar return.
Includes constant feature removal, correlation pruning, and parallel folds.
"""
import logging
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from config import config
from src.execution.simulator import simulate_execution_classification
from src.features.corr_prune import correlation_prune
from src.features.variance_filter import remove_constant_features

logger = logging.getLogger(__name__)

def train_and_predict(train_df: pl.DataFrame, test_df: pl.DataFrame,
                      feature_cols: list, target_col: str) -> np.ndarray:
    """
    Train LogisticRegression on train, predict probability of upward move on test.
    First removes constant features.
    """
    # Remove constant features on train fold
    feature_cols = remove_constant_features(train_df, feature_cols, threshold=1e-9)
    if len(feature_cols) == 0:
        logger.warning("No non-constant features left. Returning uniform probabilities.")
        return np.full(len(test_df), 0.5, dtype=np.float32)
    
    X_train = train_df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y_train = train_df.select(target_col).to_numpy().astype(np.int8).ravel()
    X_test = test_df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        penalty='l2',
        C=1.0 / config.RIDGE_PARAMS['alpha'] if 'alpha' in config.RIDGE_PARAMS else 1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=config.SEED,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    probs = model.predict_proba(X_test_scaled)[:, 1].astype(np.float32)
    return probs

def compute_benchmark(df: pl.DataFrame) -> pl.Series:
    """Naive benchmark: 20-period SMA crossover using lagged close to avoid lookahead."""
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    # Shift close by 1 to avoid using current bar's close (which is unknown at open)
    close_lagged = np.roll(close, 1)
    close_lagged[0] = close[0]
    sma20 = np.full(len(close), np.nan)
    for i in range(20, len(close)):
        sma20[i] = np.mean(close_lagged[i-20+1:i+1])   # uses lagged close
    signal = np.where(close_lagged > sma20, 1.0, 0.0)
    position = np.roll(signal, 1)
    position[0] = 0.0
    ret_exec = (close - open_) / np.maximum(open_, config.EPS)
    pnl = position * ret_exec
    pnl = np.nan_to_num(pnl, nan=0.0)
    return pl.Series("benchmark_pnl", pnl).cast(pl.Float32)

def process_fold(train_df: pl.DataFrame, test_df: pl.DataFrame,
                 feature_cols: list, target_col: str) -> pl.DataFrame:
    """Process a single walkforward fold: train, predict, simulate execution."""
    probs = train_and_predict(train_df, test_df, feature_cols, target_col)
    test_df = test_df.with_columns(pl.Series("prediction_prob", probs))
    test_df = test_df.with_columns(compute_benchmark(test_df))
    return simulate_execution_classification(test_df)

def run_walkforward(df: pl.DataFrame, feature_cols: list,
                    target_col: str = "target_sign") -> pl.DataFrame:
    """
    Walkforward with 60-day train, 1-day test, rolling by 1 day.
    """
    if "ts_event" not in df.columns:
        raise ValueError("DataFrame must have ts_event for temporal splits.")
    df = df.with_columns(pl.col("ts_event").dt.date().alias("date"))
    unique_dates = sorted(df["date"].unique().to_list())
    train_days = config.WF_TRAIN_DAYS
    test_days = config.WF_TEST_DAYS
    step_days = config.WF_STEP_DAYS

    # Prune correlated features using first training fold
    first_train_dates = unique_dates[:train_days]
    first_train_df = df.filter(pl.col("date").is_in(first_train_dates))
    if len(first_train_df) > 0:
        pruned_features = correlation_prune(first_train_df, feature_cols, threshold=config.CORR_THRESHOLD)
        logger.info(f"Correlation pruning reduced features from {len(feature_cols)} to {len(pruned_features)}")
    else:
        pruned_features = feature_cols
        logger.warning("Could not determine pruned features; using all features.")

    # Prepare folds
    folds = []
    for i in range(0, len(unique_dates) - train_days - test_days + 1, step_days):
        train_end_idx = i + train_days
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_days
        train_dates = unique_dates[i:train_end_idx]
        test_dates = unique_dates[test_start_idx:test_end_idx]
        train_df = df.filter(pl.col("date").is_in(train_dates))
        test_df = df.filter(pl.col("date").is_in(test_dates))
        if train_df.is_empty() or test_df.is_empty():
            continue
        folds.append((train_df, test_df, pruned_features, target_col))
        logger.info(f"Prepared fold: train {train_dates[0]} to {train_dates[-1]}, test {test_dates[0]}")

    if not folds:
        raise ValueError("No folds processed.")

    n_parallel = getattr(config, 'WF_PARALLEL_FOLDS', 1)
    if n_parallel > 1:
        logger.info(f"Processing {len(folds)} folds in parallel with {n_parallel} workers...")
        results = Parallel(n_jobs=n_parallel, backend='loky')(
            delayed(process_fold)(train_df, test_df, feat_cols, tgt)
            for (train_df, test_df, feat_cols, tgt) in folds
        )
    else:
        results = [process_fold(train_df, test_df, feat_cols, tgt) for (train_df, test_df, feat_cols, tgt) in folds]

    final = pl.concat(results)
    final = final.sort(["session_id", "ts_event"])
    return final