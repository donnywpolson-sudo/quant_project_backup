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


def safe_clip(X, min_val=-10.0, max_val=10.0):
    return np.clip(X, min_val, max_val)


def safe_replace(X):
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def robust_scale(X_train, X_test):
    # no leakage: fit only on train
    med = np.median(X_train, axis=0)
    q1 = np.percentile(X_train, 25, axis=0)
    q3 = np.percentile(X_train, 75, axis=0)
    iqr = np.clip(q3 - q1, 1e-6, None)

    X_train = (X_train - med) / iqr
    X_test = (X_test - med) / iqr

    return X_train, X_test


def train_and_predict(train_X: pl.DataFrame,
                      train_y: pl.Series,
                      test_X: pl.DataFrame,
                      feature_cols: list) -> np.ndarray:

    feature_cols = remove_constant_features(
        train_X.select(feature_cols),
        feature_cols,
        threshold=1e-9
    )

    if len(feature_cols) == 0:
        logger.warning('No non-constant features left. Returning uniform probabilities.')
        return np.full(len(test_X), 0.5, dtype=np.float32)

    X_train = train_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y_train = train_y.to_numpy().astype(np.int8).ravel()
    X_test = test_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)

    # safe numerics
    X_train = safe_replace(safe_clip(X_train))
    X_test = safe_replace(safe_clip(X_test))

    # robust scaling (better generalization)
    X_train, X_test = robust_scale(X_train, X_test)

    X_train = safe_clip(X_train, -5.0, 5.0)
    X_test = safe_clip(X_test, -5.0, 5.0)

    if config.MODEL_TYPE == 'Ridge':
        ridge_params = config.RIDGE_PARAMS.copy()
        ridge_params['alpha'] = max(ridge_params.get('alpha', 1.0), 10.0)  # stronger regularization

        model = Ridge(**ridge_params)
        model.fit(X_train, y_train)

        raw_pred = model.predict(X_test)
        raw_pred = safe_clip(raw_pred, -3.0, 3.0)

        probs = expit(raw_pred).astype(np.float32)

    elif config.MODEL_TYPE == 'RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=4,
            min_samples_split=50,
            min_samples_leaf=25,
            max_features=0.4,
            random_state=config.SEED,
            n_jobs=1,
            class_weight='balanced_subsample'
        )

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1].astype(np.float32)

    else:
        raise ValueError(f'Unknown MODEL_TYPE: {config.MODEL_TYPE}')

    probs = safe_clip(probs, 0.1, 0.9)
    return probs.astype(np.float32)


def smooth_probabilities(probs: np.ndarray,
                         session_ids: np.ndarray,
                         alpha: float = 0.1) -> np.ndarray:

    if alpha <= 0:
        return probs

    alpha = min(max(alpha, 0.0), 0.2)  # bound smoothing

    smoothed = np.zeros_like(probs, dtype=np.float32)
    current_smooth = 0.5
    last_session = None

    for i in range(len(probs)):
        p = float(probs[i])
        sess = session_ids[i]

        if sess != last_session:
            current_smooth = 0.5
            last_session = sess

        p = min(max(p, 0.05), 0.95)
        current_smooth = alpha * p + (1 - alpha) * current_smooth

        smoothed[i] = current_smooth

    return smoothed.astype(np.float32)


def compute_benchmark(df: pl.DataFrame) -> pl.Series:
    close = df['close'].to_numpy()
    open_ = df['open'].to_numpy()

    close_lagged = np.roll(close, 1)
    close_lagged[0] = close[0]

    sma20 = np.full(len(close), np.nan, dtype=np.float32)

    for i in range(20, len(close)):
        sma20[i] = np.mean(close_lagged[i - 19:i + 1])

    signal = np.where(close_lagged > sma20, 1.0, 0.0).astype(np.float32)

    position = np.roll(signal, 1)
    position[0] = 0.0

    ret_exec = (close - open_) / np.maximum(open_, config.EPS)
    ret_exec = safe_replace(ret_exec).astype(np.float32)

    pnl = position * ret_exec
    pnl = safe_replace(pnl).astype(np.float32)

    return pl.Series('benchmark_pnl', pnl, dtype=pl.Float32)


def process_fold(train_X: pl.DataFrame,
                 train_y: pl.Series,
                 test_original: pl.DataFrame,
                 feature_cols: list) -> pl.DataFrame:

    probs = train_and_predict(train_X, train_y, test_original, feature_cols)

    if config.PROBABILITY_SMOOTHING_ALPHA > 0:
        session_ids = test_original['session_id'].to_numpy()
        probs = smooth_probabilities(
            probs,
            session_ids,
            alpha=min(config.PROBABILITY_SMOOTHING_ALPHA, 0.15)
        )

    result = test_original.with_columns(
        pl.Series('prediction_prob', probs).cast(pl.Float32)
    )

    result = result.with_columns(compute_benchmark(result))

    return simulate_execution_classification(result)


def run_walkforward(X: pl.DataFrame,
                    y: pl.DataFrame,
                    feature_cols: list,
                    target_col: str = 'target_sign') -> pl.DataFrame:

    df = X.with_columns(y)

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    df = df.with_columns(pl.col('ts_event').dt.date().alias('date'))

    unique_dates = sorted(df['date'].unique().to_list())

    first_train_dates = unique_dates[:config.WF_TRAIN_DAYS]
    first_train_df = df.filter(pl.col('date').is_in(first_train_dates))

    if len(first_train_df) > 0:
        pruned_features = correlation_prune(
            first_train_df,
            feature_cols,
            threshold=min(config.CORR_THRESHOLD, 0.95)
        )
    else:
        pruned_features = feature_cols

    folds = []

    for i in range(0,
                   len(unique_dates) - config.WF_TRAIN_DAYS - config.WF_TEST_DAYS + 1,
                   config.WF_STEP_DAYS):

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
            res = process_fold(train_X, train_y, test_original, feat_cols)
            results.append(res)
    else:
        results = Parallel(n_jobs=config.WF_PARALLEL_FOLDS, backend='loky')(
            delayed(process_fold)(train_X, train_y, test_original, feat_cols)
            for train_X, train_y, test_original, feat_cols in folds
        )

    final = pl.concat(results)
    final = final.sort(['session_id', 'ts_event'])

    return final