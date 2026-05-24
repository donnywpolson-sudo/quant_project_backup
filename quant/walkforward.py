"""
src/walkforward.py
Walkforward with configurable model (Ridge or RandomForestClassifier).
Predicts probability of upward move. Includes probability smoothing.
Now with Ridge and logistic link for full compliance.
"""
import logging
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from joblib import Parallel, delayed
from config import config
from quant.execution.simulator import simulate_execution_classification
from quant.features.corr_prune import correlation_prune
from quant.features.variance_filter import remove_constant_features
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train_and_predict(train_X: pl.DataFrame, train_y: pl.Series,
                      test_X: pl.DataFrame, feature_cols: list) -> np.ndarray:
    """Train either Ridge or RandomForest and return probabilities."""
    # Remove constant features
    feature_cols = remove_constant_features(train_X.select(feature_cols), feature_cols, threshold=1e-9)
    if len(feature_cols) == 0:
        logger.warning("No non-constant features left. Returning uniform probabilities.")
        return np.full(len(test_X), 0.5, dtype=np.float32)

    X_train = train_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y_train = train_y.to_numpy().astype(np.int8).ravel()
    X_test = test_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if config.MODEL_TYPE == "Ridge":
        model = Ridge(**config.RIDGE_PARAMS)
        model.fit(X_train_scaled, y_train)
        raw_pred = model.predict(X_test_scaled)
        probs = expit(raw_pred).astype(np.float32)
    elif config.MODEL_TYPE == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt',
            random_state=config.SEED, n_jobs=1, class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1].astype(np.float32)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {config.MODEL_TYPE}")
    return probs

def smooth_probabilities(probs: np.ndarray, session_ids: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Exponential moving average smoothing, reset at session boundaries."""
    if alpha <= 0:
        return probs
    smoothed = np.zeros_like(probs)
    current_smooth = 0.5
    last_session = None
    for i, (p, sess) in enumerate(zip(probs, session_ids)):
        if sess != last_session:
            current_smooth = 0.5
            last_session = sess
        current_smooth = alpha * p + (1 - alpha) * current_smooth
        smoothed[i] = current_smooth
    return smoothed

def compute_benchmark(df: pl.DataFrame) -> pl.Series:
    """Naive benchmark: 20-period SMA crossover using lagged close."""
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
    if config.PROBABILITY_SMOOTHING_ALPHA > 0:
        session_ids = test_original["session_id"].to_numpy()
        probs = smooth_probabilities(probs, session_ids, alpha=config.PROBABILITY_SMOOTHING_ALPHA)
    result = test_original.with_columns(pl.Series("prediction_prob", probs))
    result = result.with_columns(compute_benchmark(result))
    return simulate_execution_classification(result)

def run_walkforward(X: pl.DataFrame, y: pl.DataFrame, feature_cols: list,
                    target_col: str = "target_sign") -> pl.DataFrame:
    """Walkforward with train/test split by date."""
    df = X.with_columns(y)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")
    df = df.with_columns(pl.col("ts_event").dt.date().alias("date"))
    unique_dates = sorted(df["date"].unique().to_list())

    # Prune correlated features using first training fold
    first_train_dates = unique_dates[:config.WF_TRAIN_DAYS]
    first_train_df = df.filter(pl.col("date").is_in(first_train_dates))
    if len(first_train_df) > 0:
        pruned_features = correlation_prune(first_train_df, feature_cols, threshold=config.CORR_THRESHOLD)
        logger.info(f"Correlation pruning reduced features from {len(feature_cols)} to {len(pruned_features)}")
    else:
        pruned_features = feature_cols

    folds = []
    for i in range(0, len(unique_dates) - config.WF_TRAIN_DAYS - config.WF_TEST_DAYS + 1, config.WF_STEP_DAYS):
        train_end = i + config.WF_TRAIN_DAYS
        test_start = train_end
        test_end = test_start + config.WF_TEST_DAYS
        train_dates = unique_dates[i:train_end]
        test_dates = unique_dates[test_start:test_end]

        train_df = df.filter(pl.col("date").is_in(train_dates))
        test_df = df.filter(pl.col("date").is_in(test_dates))
        if train_df.is_empty() or test_df.is_empty():
            continue

        train_X = train_df.drop([target_col, "date"])
        train_y = train_df[target_col]
        test_original = test_df.drop([target_col, "date"])
        folds.append((train_X, train_y, test_original, pruned_features))

    if not folds:
        raise ValueError("No folds processed.")

    if config.WF_PARALLEL_FOLDS == 1:
        results = []
        for (train_X, train_y, test_original, feat_cols) in tqdm(folds, desc="Walkforward folds", unit="fold"):
            results.append(process_fold(train_X, train_y, test_original, feat_cols))
    else:
        logger.info(f"Processing {len(folds)} folds in parallel with {config.WF_PARALLEL_FOLDS} workers...")
        results = Parallel(n_jobs=config.WF_PARALLEL_FOLDS, backend='loky')(
            delayed(process_fold)(train_X, train_y, test_original, feat_cols)
            for (train_X, train_y, test_original, feat_cols) in folds
        )
    final = pl.concat(results)
    final = final.sort(["session_id", "ts_event"])
    return final