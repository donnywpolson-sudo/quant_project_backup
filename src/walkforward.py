"""
src/walkforward.py
Walkforward with RandomForestClassifier – predicts probability of upward move.
Includes probability smoothing to reduce turnover.
"""
import logging
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from config import config
from src.execution.simulator import simulate_execution_classification
from src.features.corr_prune import correlation_prune
from src.features.variance_filter import remove_constant_features

logger = logging.getLogger(__name__)

def train_and_predict(train_X: pl.DataFrame, train_y: pl.Series,
                      test_X: pl.DataFrame, feature_cols: list) -> np.ndarray:
    """Train RandomForest on features only (target not included)."""
    # Remove constant features
    feature_cols = remove_constant_features(train_X.select(feature_cols), feature_cols, threshold=1e-9)
    if len(feature_cols) == 0:
        logger.warning("No non-constant features left. Returning uniform probabilities.")
        return np.full(len(test_X), 0.5, dtype=np.float32)

    X_train = train_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y_train = train_y.to_numpy().astype(np.int8).ravel()
    X_test = test_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)

    # Scaling is not necessary for trees, but we keep for consistency (harmless)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=config.SEED,
        n_jobs=1,                     # deterministic
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    probs = model.predict_proba(X_test_scaled)[:, 1].astype(np.float32)
    return probs

def smooth_probabilities(probs: np.ndarray, session_ids: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Apply exponential moving average smoothing to probabilities,
    resetting at each session boundary.
    alpha = smoothing factor (0 < alpha <= 1). Lower = smoother.
    """
    smoothed = np.zeros_like(probs)
    current_smooth = 0.5  # initial neutral value
    last_session = None
    for i, (p, sess) in enumerate(zip(probs, session_ids)):
        if sess != last_session:
            current_smooth = 0.5  # reset at new session
            last_session = sess
        current_smooth = alpha * p + (1 - alpha) * current_smooth
        smoothed[i] = current_smooth
    return smoothed

def compute_benchmark(df: pl.DataFrame) -> pl.Series:
    """Naive benchmark: 20‑period SMA crossover using lagged close."""
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    close_lagged = np.roll(close, 1)
    close_lagged[0] = close[0]
    sma20 = np.full(len(close), np.nan)
    for i in range(20, len(close)):
        sma20[i] = np.mean(close_lagged[i-20+1:i+1])
    signal = np.where(close_lagged > sma20, 1.0, 0.0)
    position = np.roll(signal, 1)
    position[0] = 0.0
    ret_exec = (close - open_) / np.maximum(open_, config.EPS)
    pnl = position * ret_exec
    pnl = np.nan_to_num(pnl, nan=0.0)
    return pl.Series("benchmark_pnl", pnl).cast(pl.Float32)

def process_fold(train_X: pl.DataFrame, train_y: pl.Series,
                 test_original: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
    """Train, predict, smooth, then simulate execution."""
    probs = train_and_predict(train_X, train_y, test_original, feature_cols)
    # Apply smoothing if enabled
    smoothing_alpha = getattr(config, 'PROBABILITY_SMOOTHING_ALPHA', 0.1)
    if smoothing_alpha > 0:
        session_ids = test_original["session_id"].to_numpy()
        probs = smooth_probabilities(probs, session_ids, alpha=smoothing_alpha)
    result = test_original.with_columns(pl.Series("prediction_prob", probs))
    result = result.with_columns(compute_benchmark(result))
    return simulate_execution_classification(result)

def run_walkforward(X: pl.DataFrame, y: pl.DataFrame, feature_cols: list,
                    target_col: str = "target_sign") -> pl.DataFrame:
    """
    Walkforward with 60‑day train, 1‑day test, rolling daily.
    X : DataFrame with features and metadata (ts_event, open, high, low, close, volume, session_id, regime)
    y : DataFrame with a single column named `target_col` (the target)
    """
    # Combine X and y for date splitting
    df = X.with_columns(y)   # adds the target column (already named target_col)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found after adding y.")
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

        # Split into X and y for training
        train_X = train_df.drop([target_col, "date"])
        train_y = train_df[target_col]   # Series
        # For testing, keep the full test_df without target and date
        test_original = test_df.drop([target_col, "date"])

        folds.append((train_X, train_y, test_original, pruned_features))

    if not folds:
        raise ValueError("No folds processed.")

    n_parallel = getattr(config, 'WF_PARALLEL_FOLDS', 1)
    if n_parallel > 1:
        logger.info(f"Processing {len(folds)} folds in parallel with {n_parallel} workers...")
        results = Parallel(n_jobs=n_parallel, backend='loky')(
            delayed(process_fold)(train_X, train_y, test_original, feat_cols)
            for (train_X, train_y, test_original, feat_cols) in folds
        )
    else:
        results = [process_fold(train_X, train_y, test_original, feat_cols)
                   for (train_X, train_y, test_original, feat_cols) in folds]

    final = pl.concat(results)
    final = final.sort(["session_id", "ts_event"])
    return final