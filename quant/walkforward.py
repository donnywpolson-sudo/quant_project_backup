import logging
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from scipy.special import expit
from joblib import Parallel, delayed
from quant.config_manager import config
from quant.execution.simulator import simulate_execution_classification
from quant.features.corr_prune import correlation_prune
from quant.features.variance_filter import remove_constant_features
from tqdm import tqdm
logger = logging.getLogger(__name__)

def safe_clip(X, min_val=-10.0, max_val=10.0):
    return np.clip(X, min_val, max_val)

def safe_replace(X):
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def robust_scale(X_train, X_test):
    med = np.median(X_train, axis=0)
    q1 = np.percentile(X_train, 25, axis=0)
    q3 = np.percentile(X_train, 75, axis=0)
    iqr = np.clip(q3 - q1, 1e-06, None)
    iqr = np.clip(iqr, 0.01, None)
    scale = np.where(iqr > 0, 1.0 / iqr, 1.0)
    X_train = (X_train - med) * scale
    X_test = (X_test - med) * scale
    X_train = np.clip(X_train, -10.0, 10.0)
    X_test = np.clip(X_test, -10.0, 10.0)
    return (X_train.astype(np.float32), X_test.astype(np.float32))

def stabilize_targets(y):
    y = safe_replace(y)
    return np.clip(y, -1.0, 1.0)

def train_and_predict(train_X: pl.DataFrame, train_y: pl.Series, test_X: pl.DataFrame, feature_cols: list) -> np.ndarray:
    feature_cols = remove_constant_features(train_X.select(feature_cols), feature_cols, threshold=1e-08)
    if len(feature_cols) == 0:
        return np.full(len(test_X), 0.5, dtype=np.float32)
    X_train = train_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y_train = stabilize_targets(train_y.to_numpy().astype(np.float32).ravel())
    X_test = test_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    X_train = safe_replace(safe_clip(X_train, -8.0, 8.0))
    X_test = safe_replace(safe_clip(X_test, -8.0, 8.0))
    X_train, X_test = robust_scale(X_train, X_test)
    X_train = safe_clip(X_train, -4.0, 4.0)
    X_test = safe_clip(X_test, -4.0, 4.0)
    if config.MODEL_TYPE == 'Ridge':
        ridge_params = config.RIDGE_PARAMS.copy()
        ridge_params['alpha'] = ridge_params.get('alpha', 1.0)
        model = Ridge(**ridge_params)
        model.fit(X_train, y_train)
        raw_pred = model.predict(X_test)
        raw_pred = safe_clip(raw_pred, -2.0, 2.0)
        probs = expit(raw_pred).astype(np.float32)
    elif config.MODEL_TYPE == 'RandomForestClassifier':
        model = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_split=200, min_samples_leaf=100, max_features=0.2, random_state=config.SEED, n_jobs=1, class_weight='balanced_subsample')
        model.fit(X_train, (y_train > 0).astype(np.int8))
        probs = model.predict_proba(X_test)[:, 1].astype(np.float32)
    else:
        raise ValueError(f'Unknown MODEL_TYPE: {config.MODEL_TYPE}')
    probs = safe_clip(probs, 0.05, 0.95)
    return probs.astype(np.float32)

def smooth_probabilities(probs: np.ndarray, session_ids: np.ndarray, alpha: float=0.3) -> np.ndarray:
    if alpha <= 0:
        return probs.astype(np.float32)
    alpha = min(max(alpha, 0.2), 0.5)
    smoothed = np.zeros_like(probs, dtype=np.float32)
    current = 0.5
    last_session = None
    for i in range(len(probs)):
        p = float(probs[i])
        sess = session_ids[i]
        if sess != last_session:
            current = 0.5
            last_session = sess
        p = min(max(p, 0.1), 0.9)
        current = alpha * p + (1 - alpha) * current
        smoothed[i] = current
    return smoothed.astype(np.float32)

def compute_benchmark(df: pl.DataFrame) -> pl.Series:
    close = df['close'].to_numpy().astype(np.float32)
    open_ = df['open'].to_numpy().astype(np.float32)
    close_lagged = np.roll(close, 1)
    close_lagged[0] = close[0]
    sma20 = np.full(len(close), np.nan, dtype=np.float32)
    for i in range(20, len(close)):
        sma20[i] = np.mean(close_lagged[i - 19:i + 1])
    signal = np.where(close_lagged > sma20, 1.0, 0.0).astype(np.float32)
    position = np.roll(signal, 1)
    position[0] = 0.0
    ret_exec = (close - open_) / np.maximum(open_, config.EPS)
    pnl = position * safe_replace(ret_exec)
    return pl.Series('benchmark_pnl', safe_replace(pnl).astype(np.float32), dtype=pl.Float32)

def process_fold(train_X: pl.DataFrame, train_y: pl.Series, test_original: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
    probs = train_and_predict(train_X, train_y, test_original, feature_cols)
    if config.PROBABILITY_SMOOTHING_ALPHA > 0:
        session_ids = test_original['session_id'].to_numpy()
        probs = smooth_probabilities(probs, session_ids, alpha=config.PROBABILITY_SMOOTHING_ALPHA)
    result = test_original.with_columns(pl.Series('prediction_prob', probs).cast(pl.Float32))
    result = result.with_columns(compute_benchmark(result))
    return simulate_execution_classification(result)

def run_walkforward(X: pl.DataFrame, y: pl.DataFrame, feature_cols: list, target_col: str='target_sign') -> pl.DataFrame:
    df = X.with_columns(y)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")
    df = df.with_columns(pl.col('ts_event').dt.date().alias('date'))
    unique_dates = sorted(df['date'].unique().to_list())
    first_train_dates = unique_dates[:config.WF_TRAIN_DAYS]
    first_train_df = df.filter(pl.col('date').is_in(first_train_dates))
    if len(first_train_df) > 0:
        pruned_features = correlation_prune(first_train_df, feature_cols, threshold=min(config.CORR_THRESHOLD, 0.9))
    else:
        pruned_features = feature_cols
    folds = []
    for i in range(0, len(unique_dates) - config.WF_TRAIN_DAYS - config.WF_TEST_DAYS + 1, config.WF_STEP_DAYS):
        train_end = i + config.WF_TRAIN_DAYS
        test_start = train_end
        test_end = test_start + config.WF_TEST_DAYS
        train_dates = unique_dates[i:train_end]
        test_dates = unique_dates[test_start:test_end]
        train_df = df.filter(pl.col('date').is_in(train_dates))
        test_df = df.filter(pl.col('date').is_in(test_dates))
        if train_df.is_empty() or test_df.is_empty():
            continue
        train_X = train_df.drop([target_col, 'date'])
        train_y = train_df[target_col]
        test_original = test_df.drop([target_col, 'date'])
        folds.append((train_X, train_y, test_original, pruned_features))
    if not folds:
        raise ValueError('No folds processed.')
    if config.WF_PARALLEL_FOLDS == 1:
        results = []
        for train_X, train_y, test_original, feat_cols in tqdm(folds, desc='Walkforward folds', unit='fold'):
            results.append(process_fold(train_X, train_y, test_original, feat_cols))
    else:
        results = Parallel(n_jobs=config.WF_PARALLEL_FOLDS, backend='loky')((delayed(process_fold)(train_X, train_y, test_original, feat_cols) for train_X, train_y, test_original, feat_cols in folds))
    final = pl.concat(results)
    final = final.sort(['session_id', 'ts_event'])
    return final