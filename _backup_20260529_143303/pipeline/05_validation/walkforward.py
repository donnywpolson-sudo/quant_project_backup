import logging
import importlib
import json
import os
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from scipy.special import expit
from joblib import Parallel, delayed
from quant.config_manager import config

_sim = importlib.import_module("pipeline.04_modeling.simulator")
simulate_execution_classification = _sim.simulate_execution_classification
_cp = importlib.import_module("pipeline.03_engineering.corr_prune")
correlation_prune = _cp.correlation_prune
_vf = importlib.import_module("pipeline.03_engineering.variance_filter")
remove_constant_features = _vf.remove_constant_features
_ml = importlib.import_module("pipeline.03_engineering.meta_label")
add_meta_label_target = _ml.add_meta_label_target
_mg = importlib.import_module("pipeline.03_engineering.meta_gate")
train_meta_model = _mg.train_meta_model
apply_meta_gate = _mg.apply_meta_gate
from tqdm import tqdm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    _hmmf = importlib.import_module("pipeline.04_modeling.hmm_filter")
    HMMRegimeFilter = _hmmf.HMMRegimeFilter

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
    # ---- Pre-ML contract validation ----
    if train_X.height == 0:
        raise RuntimeError('CONTRACT FAIL: train_X has 0 rows')
    if len(train_y) == 0:
        raise RuntimeError('CONTRACT FAIL: train_y has 0 rows')
    if train_X.height != len(train_y):
        raise RuntimeError('CONTRACT FAIL: X/y mismatch (X=%d, y=%d)' % (train_X.height, len(train_y)))
    if test_X.height == 0:
        raise RuntimeError('CONTRACT FAIL: test_X has 0 rows')
    if 'ts_event' not in train_X.columns or 'ts_event' not in test_X.columns:
        raise RuntimeError('CONTRACT FAIL: ts_event missing from X frame')

    feature_cols = remove_constant_features(train_X.select(feature_cols), feature_cols, threshold=1e-08)
    if len(feature_cols) == 0:
        return np.full(len(test_X), 0.5, dtype=np.float32)
    X_train = train_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y_train = stabilize_targets(train_y.to_numpy().astype(np.float32).ravel())
    X_test = test_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)

    # Validate y after stabilization
    if np.any(np.isnan(y_train)):
        raise RuntimeError('SKLEARN FAIL: NaN in y_train after stabilize_targets')
    if np.any(np.isinf(y_train)):
        raise RuntimeError('SKLEARN FAIL: inf in y_train after stabilize_targets')
    if X_train.shape[0] != len(y_train):
        raise RuntimeError('CONTRACT FAIL: X/y shape mismatch (X=%s, y=%s)' % (X_train.shape, y_train.shape))

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
    close = pl.col('close').cast(pl.Float32)
    open_ = pl.col('open').cast(pl.Float32)
    close_lagged = close.shift(1)
    sma20 = close_lagged.rolling_mean(window_size=20, min_periods=20)
    signal = pl.when(close_lagged > sma20).then(pl.lit(1.0, dtype=pl.Float32)).otherwise(pl.lit(0.0, dtype=pl.Float32))
    position = signal.shift(1).fill_null(0.0)
    ret_exec = (close - open_) / open_.clip(config.EPS, None)
    pnl = position * ret_exec.fill_nan(0.0).fill_null(0.0)
    return df.select(pnl.cast(pl.Float32).alias('benchmark_pnl')).to_series()

def exclude_warmup(df: pl.DataFrame, burn_in_bars: int) -> pl.DataFrame:
    """Drop the first *burn_in_bars* rows plus one extra bar (position carry-over
    from the warmup period). Returns the trimmed DataFrame unchanged
    if burn_in_bars <= 0 or the DataFrame is shorter than burn_in_bars + 1."""
    if burn_in_bars <= 0 or df.height <= burn_in_bars:
        return df
    trim = min(burn_in_bars + 1, df.height)
    return df.slice(trim)


def process_fold(train_X: pl.DataFrame, train_y: pl.Series, test_original: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
    import psutil
    if train_X.height == 0 or len(train_y) == 0:
        raise RuntimeError(
            'CONTRACT FAIL: empty fold (train_X=%d, train_y=%d)' %
            (train_X.height, len(train_y))
        )
    if train_X.height != len(train_y):
        raise RuntimeError(
            'CONTRACT FAIL: fold X/y mismatch (X=%d, y=%d)' %
            (train_X.height, len(train_y))
        )
    if test_original.height == 0:
        raise RuntimeError('CONTRACT FAIL: empty test fold')

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

    # Meta-labeling: compute primary prediction direction, train meta-model
    # on held-out portion of training data, gate target_exec by meta-prob.
    enable_meta = getattr(config, 'ENABLE_META_LABELING', False)
    meta_model = None
    if enable_meta:
        pred_dir = np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0)).astype(np.int8)
        unique_preds = np.unique(pred_dir).size
        if unique_preds < 2:
            # Meta-target collapses to primary target when primary has
            # zero directional variation. Meta-model adds only noise.
            logger.info('Meta skipped: primary model predicts single direction')
            enable_meta = False
    if enable_meta:
        result = result.with_columns(pl.Series('primary_prediction', pred_dir))
        result = add_meta_label_target(result, 'primary_prediction')
        meta_threshold = getattr(config, 'META_THRESHOLD', 0.5)

        if train_X.height >= 40:
            split = max(train_X.height // 3, 20)
            train_primary_X = train_X.slice(0, split)
            train_primary_y = train_y.slice(0, split)
            train_val_X = train_X.slice(split)
            train_val_y = train_y.slice(split)
            probs_val = train_and_predict(train_primary_X, train_primary_y, train_val_X, feature_cols)
            meta_val_pred = np.where(probs_val > 0.55, 1, np.where(probs_val < 0.45, -1, 0)).astype(np.int8)
            meta_val_df = pl.DataFrame({'primary_prediction': meta_val_pred})
            meta_val_actual = np.where(
                train_val_y.to_numpy().astype(np.float32).ravel() > 0, 1, -1
            ).astype(np.int8)
            meta_val_df = meta_val_df.with_columns(pl.Series('target_tb', meta_val_actual))
            meta_val_df = add_meta_label_target(meta_val_df, 'primary_prediction')
            meta_train_y = meta_val_df['target_meta'].to_numpy().astype(np.float32)
            meta_train_X = train_val_X.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
            meta_model = train_meta_model(meta_train_X, meta_val_pred.astype(np.float32), meta_train_y)

    result = simulate_execution_classification(result)

    if meta_model is not None:
        test_X_np = test_original.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
        result = apply_meta_gate(result, meta_model, test_X_np, meta_threshold=meta_threshold)
        result = _recompute_pnl_after_gate(result)

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
    _hmmf = importlib.import_module("pipeline.04_modeling.hmm_filter")
    HMMRegimeFilter = _hmmf.HMMRegimeFilter
    apply_hmm_filter = _hmmf.apply_hmm_filter

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

        # Recompute PnL after regime gating (target_exec may be zeroed).
        # Uses the full execution pipeline identical to simulate_execution_classification:
        # intrabar stops, contract multiplier, position clipping, round-turn settlement,
        # and proportional PnL clip — so the HMM PnL is directly comparable to the base PnL.
        result = _recompute_pnl_after_gate(result)

    result = exclude_warmup(result, getattr(config, 'BURN_IN_BARS', 500))
    return result, hmm_filter


def _recompute_pnl_after_gate(df: pl.DataFrame) -> pl.DataFrame:
    """
    Recompute position, intrabar stops, and PnL after target_exec has been
    gated by HMM.

    Uses the full execution pipeline (_compute_pnl_from_target_exec) so the
    recomputed PnL is identical to the main simulation path, including:
      - intrabar stop-loss / take-profit with gap-slippage logic
      - contract multiplier in PnL
      - position clipping (max_position_size + notional cap)
      - round-turn settlement on flatting
      - proportional PnL clip (5 % of notional)

    Preserves all HMM columns (hmm_regime_*, hmm_trade_gate).
    """
    import yaml
    from pathlib import Path
    _compute_pnl_from_target_exec = importlib.import_module(
        "pipeline.04_modeling.simulator"
    )._compute_pnl_from_target_exec

    symbol = getattr(config, 'CURRENT_SYMBOL', None) or 'ES'
    market_cfg_path = config.MARKET_CONFIGS.get(symbol)
    if market_cfg_path and Path(market_cfg_path).exists():
        with open(market_cfg_path, 'r') as f:
            market_cfg = yaml.safe_load(f)
        contract_multiplier = float(market_cfg.get('metadata', {}).get('contract_multiplier', 1.0))
    else:
        contract_multiplier = 1.0

    # Preserve HMM columns so they survive the recompute
    hmm_cols = [c for c in df.columns if c.startswith('hmm_')]
    hmm_data = {c: df[c].clone() for c in hmm_cols}

    # Drop columns that _compute_pnl_from_target_exec will recompute so we
    # get a clean replacement without column-name conflicts.
    recompute_cols = ['ret_exec', 'position', 'pos_change', 'intrabar_pnl', 'pnl']
    df_clean = df.drop([c for c in recompute_cols if c in df.columns])

    # Re-run full PnL pipeline against the HMM-gated target_exec
    df_result = _compute_pnl_from_target_exec(df_clean, contract_multiplier)

    # Restore HMM columns
    for col, series in hmm_data.items():
        df_result = df_result.with_columns(series.alias(col))

    return df_result


def _build_bar_folds(df: pl.DataFrame) -> list:
    """
    Build walkforward fold index ranges from bar count (not calendar days).

    Bar-based folding avoids calendar-day assumptions that break on intraday
    futures data with session gaps, weekends, and variable trading hours.

    Fold sizing:
        train = max(MIN_TRAIN_BARS, int(bars * TRAIN_FRACTION))
        test  = max(MIN_TEST_BARS,  int(bars * TEST_FRACTION))
        step  = test

    Returns:
        list of (i0, i1, i2, i3) index tuples:
            train = df[i0:i1], test = df[i2:i3]
            train < test strictly in time (i1 <= i2).
    """
    MIN_TRAIN_BARS = 1000
    MIN_TEST_BARS = 200
    TRAIN_FRACTION = 0.6
    TEST_FRACTION = 0.2

    bars = df.height
    if bars < MIN_TRAIN_BARS + MIN_TEST_BARS:
        raise RuntimeError(
            'BACKTEST FAILURE: %d bars insufficient for walkforward '
            '(need at least %d train + %d test)' %
            (bars, MIN_TRAIN_BARS, MIN_TEST_BARS)
        )

    train_size = max(MIN_TRAIN_BARS, int(bars * TRAIN_FRACTION))
    test_size = max(MIN_TEST_BARS, int(bars * TEST_FRACTION))
    step_size = test_size
    window_size = train_size + test_size

    folds = []
    start = 0
    while start + window_size <= bars:
        folds.append((start, start + train_size, start + train_size, start + window_size))
        start += step_size

    if len(folds) < 2:
        raise RuntimeError(
            'BACKTEST FAILURE: only %d folds from %d bars '
            '(train=%d, test=%d, step=%d) -- need at least 2' %
            (len(folds), bars, train_size, test_size, step_size)
        )

    logger.info(
        'Bar-based folds: %d folds from %d bars (train=%d, test=%d, step=%d)',
        len(folds), bars, train_size, test_size, step_size,
    )
    return folds


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
    HMMRegimeFilter = importlib.import_module(
        "pipeline.04_modeling.hmm_filter"
    ).HMMRegimeFilter
    compare_strategies = importlib.import_module(
        "pipeline.04_modeling.validation"
    ).compare_strategies

    df = X.with_columns(y)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    # Empty-input guard -- no data to walk forward over
    if df.height == 0:
        logger.warning('Empty input DataFrame for HMM walkforward -- returning empty result.')
        return pl.DataFrame(), {}

    df = df.sort('ts_event')
    fold_indices = _build_bar_folds(df)
    if not fold_indices:
        logger.warning('No folds produced for HMM -- returning empty result.')
        return pl.DataFrame(), {}

    # Correlation pruning on first training window
    first_train_df = df.slice(fold_indices[0][0], fold_indices[0][1] - fold_indices[0][0])
    pruned_features = (
        correlation_prune(first_train_df, feature_cols, threshold=min(config.CORR_THRESHOLD, 0.9))
        if first_train_df.height > 0
        else feature_cols
    )

    # Build folds from index ranges (zero-copy slicing)
    folds = []
    for i0, i1, i2, i3 in fold_indices:
        train_X = df.slice(i0, i1 - i0).drop([target_col])
        train_y = df.slice(i0, i1 - i0)[target_col]
        test_original = df.slice(i2, i3 - i2).drop([target_col])
        folds.append((train_X, train_y, test_original, pruned_features))

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

    # Empty-input guard -- no data to walk forward over
    if df.height == 0:
        raise RuntimeError(
            'BACKTEST FAILURE: 0 rows after feature generation (target=%s)' % target_col
        )

    df = df.sort('ts_event')
    logger.info('[ML-CONTRACT] X=%d rows, y=%d rows, features=%d', df.height, df.height, len(feature_cols))
    logger.info('[ML-CONTRACT] ts_event range: %s -> %s',
                df['ts_event'].min(), df['ts_event'].max())
    fold_indices = _build_bar_folds(df)
    if not fold_indices:
        logger.warning('No folds produced -- returning empty result.')
        return pl.DataFrame()

    # Correlation pruning on first training window
    first_train_df = df.slice(fold_indices[0][0], fold_indices[0][1] - fold_indices[0][0])
    pruned_features = (
        correlation_prune(first_train_df, feature_cols, threshold=min(config.CORR_THRESHOLD, 0.9))
        if first_train_df.height > 0
        else feature_cols
    )

    # Build folds from index ranges (zero-copy slicing)
    folds = []
    for i0, i1, i2, i3 in fold_indices:
        train_X = df.slice(i0, i1 - i0).drop([target_col])
        train_y = df.slice(i0, i1 - i0)[target_col]
        test_original = df.slice(i2, i3 - i2).drop([target_col])
        folds.append((train_X, train_y, test_original, pruned_features))
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
