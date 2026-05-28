import logging
import json
import os
from typing import Optional, Tuple

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

def exclude_warmup(df: pl.DataFrame, burn_in_bars: int) -> pl.DataFrame:
    """Drop the first *burn_in_bars* rows so they are excluded from metrics
    aggregation (PnL, Sharpe, etc.). Returns the trimmed DataFrame unchanged
    if burn_in_bars <= 0 or the DataFrame is shorter than burn_in_bars."""
    if burn_in_bars <= 0 or df.height <= burn_in_bars:
        return df
    return df.slice(burn_in_bars)


def process_fold(train_X: pl.DataFrame, train_y: pl.Series, test_original: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
    import psutil
    rss_bytes = psutil.Process().memory_info().rss
    rss_stop = getattr(config, 'RSS_STOP_BYTES', int(13.5 * 1024**3))
    if rss_bytes > rss_stop:
        raise MemoryError(f'RSS {rss_bytes/(1024**3):.2f} GB exceeds RSS_STOP_BYTES ({rss_stop/(1024**3):.2f} GB) in process_fold')
    probs = train_and_predict(train_X, train_y, test_original, feature_cols)
    if config.PROBABILITY_SMOOTHING_ALPHA > 0:
        session_ids = test_original['session_id'].to_numpy()
        probs = smooth_probabilities(probs, session_ids, alpha=config.PROBABILITY_SMOOTHING_ALPHA)
    result = test_original.with_columns(pl.Series('prediction_prob', probs).cast(pl.Float32))
    result = result.with_columns(compute_benchmark(result))
    result = simulate_execution_classification(result)
    return exclude_warmup(result, getattr(config, 'BURN_IN_BARS', 500))

# ============================================================================
# HMM Regime-Aware Walkforward
# ============================================================================

def _resample_to_1h(df_5min: pl.DataFrame) -> pl.DataFrame:
    """
    Resample 5-minute data to 1-hour frequency for HMM detection layer.
    Preserves session_id grouping and uses only OHLCV columns.
    """
    from quant.session import add_session_id

    df = df_5min.select(['ts_event', 'open', 'high', 'low', 'close', 'volume'])
    df = df.with_columns(
        pl.col('ts_event').dt.convert_time_zone(config.TIMEZONE).alias('ts_local')
    )
    df = df.with_columns(
        pl.col('ts_local').dt.truncate('1h').alias('ts_hour')
    )
    # Add session_id for proper grouping
    session_id = pl.col('ts_local').dt.offset_by('6h').dt.date().cast(pl.String)
    df = df.with_columns(session_id.alias('session_id'))

    agg = df.group_by(['session_id', 'ts_hour'], maintain_order=True).agg([
        pl.col('open').first().alias('open'),
        pl.col('high').max().alias('high'),
        pl.col('low').min().alias('low'),
        pl.col('close').last().alias('close'),
        pl.col('volume').sum().alias('volume'),
        pl.len().alias('n_ticks'),
    ])
    # Filter incomplete hours (< 10 ticks)
    agg = agg.filter(pl.col('n_ticks') >= 10)
    agg = agg.rename({'ts_hour': 'ts_event'})
    agg = agg.drop('n_ticks')
    agg = agg.with_columns(
        pl.col('ts_event').dt.convert_time_zone('UTC').alias('ts_event')
    )
    agg = agg.with_columns([
        pl.col('open').cast(pl.Float32),
        pl.col('high').cast(pl.Float32),
        pl.col('low').cast(pl.Float32),
        pl.col('close').cast(pl.Float32),
    ])
    agg = agg.sort(['session_id', 'ts_event'])
    return agg


def process_fold_with_hmm(
    train_X: pl.DataFrame,
    train_y: pl.Series,
    test_original: pl.DataFrame,
    feature_cols: list,
    hmm_filter: Optional["HMMRegimeFilter"] = None,
    df_1h_test: Optional[pl.DataFrame] = None,
    fold_idx: int = 0,
    hmm_retrain_interval: int = 5,
) -> Tuple[pl.DataFrame, Optional["HMMRegimeFilter"]]:
    """
    Process a single walkforward fold with HMM regime gating.

    Steps:
      1. Train ML model and generate base predictions.
      2. Execute base strategy (computation is side-effect-free PnL).
      3. Apply HMM regime filter if active (gate trades by regime).
      4. Recompute PnL after gating.

    Args:
        train_X, train_y, test_original, feature_cols: Standard fold data.
        hmm_filter: Existing HMMRegimeFilter instance (None on first call).
        df_1h_test: 1H resampled test data for this fold.
        fold_idx: Zero-based fold index (for retrain scheduling).
        hmm_retrain_interval: Retrain HMM every N folds.

    Returns:
        (result_df, updated_hmm_filter): The executed DataFrame and filter.
    """
    from quant.regime.hmm_filter import HMMRegimeFilter, apply_hmm_filter

    # --- Base execution (no HMM) ---
    probs = train_and_predict(train_X, train_y, test_original, feature_cols)
    if config.PROBABILITY_SMOOTHING_ALPHA > 0:
        session_ids = test_original['session_id'].to_numpy()
        probs = smooth_probabilities(probs, session_ids, alpha=config.PROBABILITY_SMOOTHING_ALPHA)

    result = test_original.with_columns(
        pl.Series('prediction_prob', probs).cast(pl.Float32)
    )
    result = result.with_columns(compute_benchmark(result))
    result = simulate_execution_classification(result)

    # --- HMM Regime Gating ---
    if hmm_filter is None:
        hmm_filter = HMMRegimeFilter()

    if df_1h_test is not None and df_1h_test.height > 0:
        should_retrain = (fold_idx > 0 and fold_idx % hmm_retrain_interval == 0)

        # Build 1H train data from train_X if retraining
        df_1h_train = None
        if should_retrain:
            df_1h_train = _resample_to_1h(train_X)

        result, hmm_filter = apply_hmm_filter(
            df_5min_base=result,
            df_1h_train=df_1h_train,
            df_1h_test=df_1h_test,
            hmm_filter=hmm_filter,
            retrain=should_retrain,
        )

        # Recompute PnL after regime gating (position may have changed)
        # The simulator's position/PnL columns are already present; we re-run
        # the full simulation on the gated target_exec to get correct PnL.
        # But to avoid double-counting friction, we do a minimal recompute.
        result = _recompute_pnl_after_gate(result)

    result = exclude_warmup(result, getattr(config, 'BURN_IN_BARS', 500))
    return result, hmm_filter


def _recompute_pnl_after_gate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Recompute position and PnL after target_exec has been gated by HMM.

    This is a lightweight recompute that only updates position, pos_change,
    and pnl — preserving all other columns (including hmm_regime_* and
    hmm_trade_gate).
    """
    eps = config.EPS

    # Position: signal from t-1 executed at t
    position = pl.col('target_exec').shift(1).fill_null(0.0)
    pos_change = (position - position.shift(1)).abs().fill_null(0.0)

    # Need ret_exec column — should already exist from first simulation pass
    if 'ret_exec' not in df.columns:
        open_next = pl.col('open').shift(-1)
        close_next = pl.col('close').shift(-1)
        ret_exec = ((close_next - open_next) / open_next.clip(eps, None)).fill_null(0.0)
        ret_exec = ret_exec.clip(-0.02, 0.02)
        df = df.with_columns(ret_exec.alias('ret_exec'))

    if 'unit_cost' not in df.columns:
        spread = (pl.col('high') - pl.col('low')) / pl.col('close').clip(eps, None)
        spread = spread.clip(0.0, 0.05)
        ret = (pl.col('close') / pl.col('close').shift(1)).log()
        ret_lagged = ret.shift(1)
        vol = ret_lagged.rolling_std(window_size=20).clip(eps, None).fill_null(1e-06)
        unit_cost = (
            config.COMMISSION_PER_TRADE
            + config.SLIPPAGE_K * spread
            + config.VOL_PENALTY * vol
            + config.TX_COST_PER_ROUNDTURN / 2.0
        ).clip(0.0, 0.01)
        df = df.with_columns(unit_cost.alias('unit_cost'))

    # Recompute PnL
    pnl = position * pl.col('ret_exec') - pl.col('unit_cost') * pos_change
    pnl = pnl.fill_nan(0.0).clip(-0.05, 0.05)

    df = df.with_columns([
        position.alias('position'),
        pos_change.alias('pos_change'),
        pnl.alias('pnl'),
    ])

    return df


def run_walkforward_with_hmm(
    X: pl.DataFrame,
    y: pl.DataFrame,
    feature_cols: list,
    target_col: str = 'target_sign',
    hmm_retrain_interval: int = 5,
) -> Tuple[pl.DataFrame, dict]:
    """
    Walkforward with HMM regime-aware risk management.

    Same interface as run_walkforward(), but additionally:
      - Resamples 5-min data to 1H internally for the detection layer.
      - Trains HMM periodically (every N folds).
      - Bridges regime probabilities to 5-min execution bars.
      - Gates trades based on regime (allowed/prohibited).
      - Returns validation report comparing base vs HMM-filtered PnL.

    Args:
        X: Feature DataFrame (5-min).
        y: Target DataFrame (5-min).
        feature_cols: Feature column names.
        target_col: Target column name.
        hmm_retrain_interval: Retrain HMM every N folds.

    Returns:
        (df_hmm_result, validation_dict):
            df_hmm_result: Full executed DataFrame with regime columns.
            validation_dict: Dictionary with base_metrics, hmm_metrics, psr, etc.
    """
    from quant.regime.hmm_filter import HMMRegimeFilter
    from quant.regime.validation import compare_strategies

    df = X.with_columns(y)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    # Use session_id for fold boundaries instead of calendar date (Finding #15).
    # Sessions cross midnight: a session starting 18:00 Jan 6 ends 16:00 Jan 7
    # and shares one session_id. Calendar-date splitting would leak session features.
    df = df.sort(['session_id', 'ts_event'])
    unique_sessions = df['session_id'].unique(maintain_order=True).to_list()

    # Correlation pruning on initial window
    first_train_sessions = unique_sessions[:config.WF_TRAIN_DAYS]
    first_train_df = df.filter(pl.col('session_id').is_in(first_train_sessions))
    if len(first_train_df) > 0:
        pruned_features = correlation_prune(
            first_train_df, feature_cols,
            threshold=min(config.CORR_THRESHOLD, 0.9)
        )
    else:
        pruned_features = feature_cols

    # Build folds
    folds = []
    for i in range(
        0,
        len(unique_sessions) - config.WF_TRAIN_DAYS - config.WF_TEST_DAYS + 1,
        config.WF_STEP_DAYS,
    ):
        train_end = i + config.WF_TRAIN_DAYS
        test_start = train_end
        test_end = test_start + config.WF_TEST_DAYS
        train_sessions = unique_sessions[i:train_end]
        test_sessions = unique_sessions[test_start:test_end]
        train_df = df.filter(pl.col('session_id').is_in(train_sessions))
        test_df = df.filter(pl.col('session_id').is_in(test_sessions))
        if train_df.is_empty() or test_df.is_empty():
            continue
        train_X = train_df.drop([target_col])
        train_y = train_df[target_col]
        test_original = test_df.drop([target_col])
        folds.append((train_X, train_y, test_original, pruned_features))

    if not folds:
        raise ValueError('No folds processed.')

    # Initialize HMM filter
    hmm_filter = HMMRegimeFilter()

    # Try to initialize HMM on the first training window
    first_train = folds[0][0]
    df_1h_init = _resample_to_1h(first_train)
    if df_1h_init.height >= 60:  # at least 60 hours (~2.5 days)
        success = hmm_filter.initialize(df_1h_init)
        logger.info(
            f"HMM initialization: {'success' if success else 'fallback'} "
            f"({df_1h_init.height} 1H bars)"
        )

    results_base = []
    results_hmm = []

    for fold_idx, (train_X, train_y, test_original, feat_cols) in enumerate(
        tqdm(folds, desc='Walkforward + HMM', unit='fold')
    ):
        # Standard base execution
        base_result = process_fold(train_X, train_y, test_original, feat_cols)
        results_base.append(base_result)

        # HMM-aware execution
        df_1h_test = _resample_to_1h(test_original)
        hmm_result, hmm_filter = process_fold_with_hmm(
            train_X=train_X,
            train_y=train_y,
            test_original=test_original,
            feature_cols=feat_cols,
            hmm_filter=hmm_filter,
            df_1h_test=df_1h_test,
            fold_idx=fold_idx,
            hmm_retrain_interval=hmm_retrain_interval,
        )
        results_hmm.append(hmm_result)

    # Concatenate results
    df_base = pl.concat(results_base).sort(['session_id', 'ts_event'])
    df_hmm = pl.concat(results_hmm).sort(['session_id', 'ts_event'])

    # Validation
    report = compare_strategies(
        df_base=df_base,
        df_hmm=df_hmm,
        pnl_column='pnl',
        position_column='position',
        confidence=0.95,
        hmm_active=hmm_filter.is_active,
        fallback_reason=hmm_filter.fallback_reason,
    )

    logger.info(f"\n{report.summary()}")

    # Build validation dict for serialization
    validation_dict = report.to_dict()
    validation_dict['hmm_training_log'] = hmm_filter.training_log
    validation_dict['n_folds'] = len(folds)

    df_hmm = exclude_warmup(df_hmm, getattr(config, 'BURN_IN_BARS', 500))
    return df_hmm, validation_dict


def run_walkforward(X: pl.DataFrame, y: pl.DataFrame, feature_cols: list, target_col: str='target_sign') -> pl.DataFrame:
    df = X.with_columns(y)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    # Use session_id for fold boundaries instead of calendar date (Finding #15).
    # Sessions cross midnight: a session starting 18:00 Jan 6 ends 16:00 Jan 7
    # and shares one session_id. Calendar-date splitting would leak session features.
    df = df.sort(['session_id', 'ts_event'])
    unique_sessions = df['session_id'].unique(maintain_order=True).to_list()

    first_train_sessions = unique_sessions[:config.WF_TRAIN_DAYS]
    first_train_df = df.filter(pl.col('session_id').is_in(first_train_sessions))
    if len(first_train_df) > 0:
        pruned_features = correlation_prune(first_train_df, feature_cols, threshold=min(config.CORR_THRESHOLD, 0.9))
    else:
        pruned_features = feature_cols

    folds = []
    for i in range(0, len(unique_sessions) - config.WF_TRAIN_DAYS - config.WF_TEST_DAYS + 1, config.WF_STEP_DAYS):
        train_end = i + config.WF_TRAIN_DAYS
        test_start = train_end
        test_end = test_start + config.WF_TEST_DAYS
        train_sessions = unique_sessions[i:train_end]
        test_sessions = unique_sessions[test_start:test_end]
        train_df = df.filter(pl.col('session_id').is_in(train_sessions))
        test_df = df.filter(pl.col('session_id').is_in(test_sessions))
        if train_df.is_empty() or test_df.is_empty():
            continue
        train_X = train_df.drop([target_col])
        train_y = train_df[target_col]
        test_original = test_df.drop([target_col])
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
    final = exclude_warmup(final, getattr(config, 'BURN_IN_BARS', 500))
    return final
