"""
src/walkforward.py
Walkforward with Ridge regression (continuous target_5m), StandardScaler fit on train only,
and execution simulation (position sizing, costs, leverage, flattening).
Now includes correlation pruning and a naive benchmark (20-period SMA crossover).
"""
import logging
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from config import config
from src.execution.simulator import simulate_execution
from src.features.corr_prune import correlation_prune

logger = logging.getLogger(__name__)

def train_and_predict(train_df: pl.DataFrame, test_df: pl.DataFrame,
                      feature_cols: list, target_col: str) -> np.ndarray:
    """Train Ridge scaler+model on train, predict on test, return predictions (continuous)."""
    X_train = train_df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y_train = train_df.select(target_col).to_numpy().astype(np.float32).ravel()
    X_test = test_df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(**config.RIDGE_PARAMS)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled).astype(np.float32)
    return preds

def compute_benchmark(df: pl.DataFrame) -> pl.Series:
    """
    Naive benchmark: 20-period SMA crossover.
    Long when close > SMA20, flat otherwise. Executes at open[t+1].
    Returns PnL series aligned with df.
    """
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    sma20 = np.full(len(close), np.nan)
    for i in range(19, len(close)):
        sma20[i] = np.mean(close[i-19:i+1])
    signal = np.where(close > sma20, 1.0, 0.0)
    position = np.roll(signal, 1)
    position[0] = 0.0
    ret_exec = (close - open_) / np.maximum(open_, config.EPS)
    pnl = position * ret_exec
    pnl = np.nan_to_num(pnl, nan=0.0)
    return pl.Series("benchmark_pnl", pnl).cast(pl.Float32)

def run_walkforward(df: pl.DataFrame, feature_cols: list, target_col: str = "target_5m") -> pl.DataFrame:
    """
    Walkforward with 60-day train, 1-day test, rolling by 1 day.
    Returns DataFrame with predictions, executed positions, and benchmark PnL.
    """
    if "ts_event" not in df.columns:
        raise ValueError("DataFrame must have ts_event for temporal splits.")
    df = df.with_columns(pl.col("ts_event").dt.date().alias("date"))
    unique_dates = sorted(df["date"].unique().to_list())
    train_days = config.WF_TRAIN_DAYS
    test_days = config.WF_TEST_DAYS
    step_days = config.WF_STEP_DAYS

    # Determine correlation-pruned feature set using the first training fold
    pruned_features = None
    first_train_dates = unique_dates[:train_days]
    first_train_df = df.filter(pl.col("date").is_in(first_train_dates))
    if len(first_train_df) > 0:
        pruned_features = correlation_prune(first_train_df, feature_cols, threshold=config.CORR_THRESHOLD)
        logger.info(f"Correlation pruning reduced features from {len(feature_cols)} to {len(pruned_features)}")
    else:
        pruned_features = feature_cols
        logger.warning("Could not determine pruned features; using all features.")

    all_results = []
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

        preds = train_and_predict(train_df, test_df, pruned_features, target_col)
        test_df = test_df.with_columns(pl.Series("prediction", preds))
        test_df = test_df.with_columns(compute_benchmark(test_df))
        test_df = simulate_execution(test_df)
        all_results.append(test_df)
        logger.info(f"Fold {i}: train {train_dates[0]} to {train_dates[-1]}, test {test_dates[0]}")

    if not all_results:
        raise ValueError("No folds processed.")
    final = pl.concat(all_results)
    return final