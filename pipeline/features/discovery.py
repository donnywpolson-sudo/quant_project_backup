import sys
print('Discovery started. Waiting for folds...', flush=True)
import os
import json
import logging
import numpy as np
import polars as pl
import psutil
import hashlib
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import ExtraTreesRegressor
from core.config import config, clamp_to_single_threaded
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def check_rss(limit_bytes):
    return psutil.Process().memory_info().rss > limit_bytes


def _fit_discovery_fold(
    fold_idx: int,
    feature_cols: list,
    et_params: dict,
    seed: int,
    rss_stop: int,
    X: np.ndarray,
    y: np.ndarray,
):
    """Fit a single ExtraTreesRegressor on a bootstrapped sample.

    Standalone function for joblib parallelization.  Clamps threading
    to single-threaded before fitting for reproducibility.

    X and y are pre-materialized numpy arrays shared across workers
    via fork/Copy-on-Write; each worker only materializes its own
    bootstrapped subset inside this function.
    """
    clamp_to_single_threaded()

    n_samples = X.shape[0]
    if n_samples == 0:
        return dict.fromkeys(feature_cols, 0.0), dict.fromkeys(feature_cols, 0.0)

    if psutil.Process().memory_info().rss > rss_stop:
        raise MemoryError(f'RSS stop limit exceeded in fold {fold_idx}')

    rng = np.random.RandomState(
        int(hashlib.sha256(f'{seed}_fold_{fold_idx}'.encode()).hexdigest(), 16) % 2 ** 32
    )
    indices = rng.choice(n_samples, size=n_samples, replace=True)
    X_boot = X[indices]
    y_boot = y[indices]

    # Defense-in-depth: strip any NaN y that survived upstream filtering
    if np.any(np.isnan(y_boot)):
        valid = ~np.isnan(y_boot)
        X_boot = X_boot[valid]
        y_boot = y_boot[valid]
        if len(y_boot) < 2:
            return dict.fromkeys(feature_cols, 0.0), dict.fromkeys(feature_cols, 0.0)

    fold_et_params = et_params.copy()
    fold_et_params['random_state'] = int(
        hashlib.sha256(f'{seed}_fold_{fold_idx}'.encode()).hexdigest(), 16
    ) % 2 ** 32
    et = ExtraTreesRegressor(**fold_et_params)
    et.fit(X_boot, y_boot)

    importances = dict(zip(feature_cols, et.feature_importances_))
    signs = {}
    for i, f in enumerate(feature_cols):
        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(X_boot[:, i], y_boot)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        signs[f] = np.sign(corr)
    return importances, signs


def run_feature_discovery(data_path: str, manifest_out: str):
    logger.info('Phase 1: Feature Discovery')

    # --- SARGable date filter: filter on raw ts_event BEFORE any function ---
    # Previously: ts_event → dt.convert_time_zone() → dt.date() → filter
    # that pattern prevents row-group pruning in parquet scans because the
    # predicate can't be evaluated against column statistics.
    #
    # Fix: compute the UTC timestamp boundary for the discovery window,
    # filter on raw ts_event >= cutoff_utc as a predicate that can be
    # pushed down to the parquet scan, THEN apply timezone conversion.
    #
    # Strategy: use a cheap first-pass scan (ts_event only) to find the
    # latest timestamp, compute the window boundary, then re-scan with the
    # predicate baked into the scan.

    # Pass 1 — cheap: read only ts_event to find the window boundary
    lf_ts = pl.scan_parquet(data_path).select('ts_event')
    try:
        df_ts = lf_ts.collect(engine='streaming')
    except TypeError:
        df_ts = lf_ts.collect(streaming=True)

    df_ts = df_ts.sort('ts_event')
    logger.info('[DIAG] Pass 1 (ts_event scan) rows=%d', df_ts.height)
    if df_ts.height == 0:
        logger.warning('Empty ts_event — skipping window trim, using all rows')
        cutoff_date = None
        cutoff_utc = None
    else:
        latest_ts = df_ts['ts_event'].to_list()[-1]
        if latest_ts.tzinfo is None:
            latest_ts = pytz.utc.localize(latest_ts)
        local_tz = pytz.timezone(config.TIMEZONE)
        latest_local = latest_ts.astimezone(local_tz)
        cutoff_local_date = latest_local.date()
        cutoff_date = cutoff_local_date - timedelta(days=config.DISCOVERY_WINDOW_DAYS)
        cutoff_local_dt = local_tz.localize(
            datetime(cutoff_date.year, cutoff_date.month, cutoff_date.day)
        )
        cutoff_utc = cutoff_local_dt.astimezone(pytz.utc)

    # Free the first-pass frame before loading full data
    del df_ts, lf_ts

    # Pass 2 — full scan with SARGable predicate pushed down
    # pl.scan_parquet supports predicate pushdown on raw columns.
    # Filtering on ts_event >= cutoff_utc allows parquet row-group pruning.
    lf = pl.scan_parquet(data_path)
    if cutoff_utc is not None:
        lf = lf.filter(pl.col('ts_event') >= cutoff_utc)

    try:
        df_features = lf.collect(engine='streaming')
    except TypeError:
        df_features = lf.collect(streaming=True)

    logger.info('[DIAG] Pass 2 (full scan) rows=%d', df_features.height)

    # Normalize ts_event dtype after parquet reload (defense in depth)
    if 'ts_event' in df_features.columns and df_features.height > 0:
        ts_dtype = df_features['ts_event'].dtype
        target_dtype = pl.Datetime(time_unit='us', time_zone='UTC')
        if ts_dtype != target_dtype:
            df_features = df_features.with_columns(
                pl.col('ts_event').cast(target_dtype)
            )

    # Guard: if no rows loaded, skip discovery instead of crashing
    if df_features.height == 0:
        logger.warning('Discovery skipped: 0 rows after Pass 2 scan.')
        os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
        placeholder = {
            'version': '1.0', 'feature_names': [],
            'selected_K': 0, 'selection_seed': config.SEED,
            'selection_date': datetime.utcnow().isoformat() + 'Z',
            'discovery_status': 'skipped',
            'reason': 'no rows available after scan',
        }
        with open(manifest_out, 'w') as f:
            json.dump(placeholder, f, indent=4)
        return

    # Now that rows are pruned, apply timezone conversion and date extraction
    # on the much smaller dataframe
    df_features = df_features.with_columns(
        pl.col('ts_event')
        .dt.convert_time_zone(config.TIMEZONE)
        .dt.date()
        .alias('date')
    )

    # Secondary precise filter on local date (handles edge cases where the
    # UTC boundary doesn't perfectly align with local-date boundaries)
    if cutoff_date is not None:
        df_features = df_features.filter(pl.col('date') >= cutoff_date)
        logger.info(
            f'Discovery limited to {config.DISCOVERY_WINDOW_DAYS} days '
            f'(from {cutoff_date.isoformat()} onwards)'
        )

    df_features = df_features.drop('date')

    # --- Memory cap: time-stratified sampling to avoid regime bias ---
    # Previously used df.tail(200000) which dropped the oldest rows
    # deterministically, discarding historical regimes.
    max_rows = 200000
    if df_features.height > max_rows:
        logger.info(
            f'Capping discovery rows from {df_features.height} to {max_rows} '
            f'via time-stratified sampling'
        )
        n = df_features.height
        # Stratify into 3 equal time segments (old/mid/recent), then
        # sample evenly from each to preserve diverse regimes.
        seg_size = n // 3
        indices = []
        for seg_start in (0, seg_size, 2 * seg_size):
            seg_end = min(seg_start + seg_size, n)
            seg_count = seg_end - seg_start
            sample_count = min(seg_count, max_rows // 3)
            if seg_count > 1:
                step = max(1, seg_count // sample_count)
                indices.extend(range(seg_start, seg_end, step))
        if len(indices) > max_rows:
            step = max(1, len(indices) // max_rows)
            indices = indices[::step][:max_rows]
        df_features = df_features.select(
            pl.int_range(pl.len()).alias('_row')
        ).with_columns(
            df_features.select(pl.all())
        ).filter(
            pl.col('_row').is_in(indices)
        ).drop('_row')

    # --- Feature column selection ---
    target_col = 'target_tb'
    if target_col not in df_features.columns:
        raise ValueError(f'Target column {target_col} not found.')

    exclude_cols = {
        'ts_event', 'open', 'high', 'low', 'close', 'volume',
        'session_id', target_col, 'regime', 'benchmark_pnl',
    }
    exclude_cols |= {c for c in df_features.columns if c.startswith('target_')}
    feature_cols = [
        c for c in df_features.columns
        if c not in exclude_cols and (not c.startswith('_'))
    ]
    feature_cols = [
        c for c in feature_cols
        if df_features[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]

    logger.info(f'Discovery using {df_features.height} rows, {len(feature_cols)} features.')
    logger.info('[DIAG] post-date-filter rows=%d feature_cols=%d', df_features.height, len(feature_cols))

    n_folds = config.BOOTSTRAP_FOLDS
    et_params = dict(config.EXTRA_TREES_PARAMS)

    # Materialize feature matrix and target once — sklearn models need numpy
    X = df_features.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y = df_features.select(target_col).to_numpy().astype(np.float32).ravel()

    # Drop rows where target is NaN (trailing shift-NaN from horizon windows).
    # Must be aligned on both X and y to preserve row indexing.
    nan_mask = ~np.isnan(y)
    logger.info('[DIAG] target NaN rows: %d / %d', int((~nan_mask).sum()), len(y))
    X = X[nan_mask]
    y = y[nan_mask]

    if X.shape[0] < 2:
        logger.warning(
            'Discovery skipped: only %d samples available '
            '(need at least 2 for bootstrap).', X.shape[0]
        )
        # Produce a minimal manifest with empty feature list
        os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
        placeholder = {
            'version': '1.0', 'feature_names': [],
            'selected_K': 0, 'selection_seed': config.SEED,
            'selection_date': datetime.utcnow().isoformat() + 'Z',
            'discovery_status': 'skipped',
            'reason': f'only {X.shape[0]} samples available (need >= 2)',
        }
        with open(manifest_out, 'w') as f:
            json.dump(placeholder, f, indent=4)
        return

    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(_fit_discovery_fold)(
            fold_idx, feature_cols, et_params, config.SEED,
            config.RSS_STOP_BYTES, X, y,
        )
        for fold_idx in range(n_folds)
    )

    importances_list = [r[0] for r in results]
    signs_list = [r[1] for r in results]

    importances_sum = {f: 0.0 for f in feature_cols}
    selection_count = {f: 0 for f in feature_cols}
    n_folds = len(importances_list)
    for imp_dict, sign_dict in zip(importances_list, signs_list):
        for f, imp in imp_dict.items():
            importances_sum[f] += imp
            if imp > 0:
                selection_count[f] += 1

    majority_sign = {}
    for f in feature_cols:
        pos = sum((1 for sd in signs_list if sd.get(f, 0) > 0))
        neg = n_folds - pos
        majority_sign[f] = 1 if pos > neg else -1

    sign_consistency_frac = {}
    for f in feature_cols:
        consistent = sum(
            (1 for sd in signs_list if sd.get(f, 0) == majority_sign[f])
        )
        sign_consistency_frac[f] = consistent / n_folds

    freq = {f: selection_count[f] / n_folds for f in feature_cols}
    mean_imp = {f: importances_sum[f] / n_folds for f in feature_cols}

    selected = [
        f for f in feature_cols
        if freq[f] >= config.SELECTION_FREQ_THRESHOLD
        and sign_consistency_frac[f] >= config.SIGN_CONSISTENCY_THRESHOLD
    ]
    selected_sorted = sorted(selected, key=lambda x: mean_imp[x], reverse=True)

    cumsum = 0.0
    final_selected = []
    total_imp = (
        sum((mean_imp[f] for f in selected_sorted))
        if selected_sorted else 1.0
    )
    for f in selected_sorted:
        cumsum += mean_imp[f] / total_imp
        final_selected.append(f)
        if cumsum >= config.CUMULATIVE_IMPORTANCE_THRESHOLD:
            break

    if len(final_selected) < config.MIN_SELECTED_FEATURES:
        if len(selected_sorted) == 0:
            all_sorted = sorted(mean_imp.items(), key=lambda x: x[1], reverse=True)
            fallback_features = [
                f for f, _ in all_sorted[:config.MIN_SELECTED_FEATURES]
            ]
            final_selected = fallback_features
        else:
            needed = config.MIN_SELECTED_FEATURES - len(final_selected)
            for f in selected_sorted:
                if f not in final_selected:
                    final_selected.append(f)
                    needed -= 1
                    if needed == 0:
                        break
            if needed > 0:
                for f, _ in sorted(
                    mean_imp.items(), key=lambda x: x[1], reverse=True
                ):
                    if f not in final_selected:
                        final_selected.append(f)
                        needed -= 1
                        if needed == 0:
                            break

    logger.info(
        f'Selected {len(final_selected)} features '
        f'(min required: {config.MIN_SELECTED_FEATURES}).'
    )

    feature_list_str = json.dumps(sorted(final_selected), sort_keys=True).encode()
    features_hash = hashlib.sha256(feature_list_str).hexdigest()

    manifest = {
        'version': '1.0',
        'feature_names': final_selected,
        'dtypes': {f: 'float32' for f in final_selected},
        'selection_seed': config.SEED,
        'selection_date': datetime.utcnow().isoformat() + 'Z',
        'selection_model': 'ExtraTreesRegressor',
        'selection_params': config.EXTRA_TREES_PARAMS,
        'selected_K': len(final_selected),
        'cumulative_importance': config.CUMULATIVE_IMPORTANCE_THRESHOLD,
        'stability_stats': {
            'min_selection_freq': config.SELECTION_FREQ_THRESHOLD,
            'sign_consistency': config.SIGN_CONSISTENCY_THRESHOLD,
            'sign_consistency_observed': {
                f: round(sign_consistency_frac.get(f, 0), 3)
                for f in final_selected[:10]
            },
        },
        'baseline_feature_list': [
            c for c in feature_cols if c.startswith('feature_')
        ][:40],
        'baseline_features_hash': f'sha256:{features_hash}',
        'baseline_feature_matrix_path': config.BASELINE_FEATURES_PERSIST_PATH,
        'serialization_params': {
            'parquet_version': '2.0',
            'compression': 'snappy',
            'row_group_size': config.ROW_GROUP_SIZE,
            'column_ordering': 'lexicographic',
        },
        'discovery_status': 'completed',
        'folds': [],
        'htf_features_included': any(
            (c.startswith(('htf_', 'cross_', '1h_', 'daily_')) for c in feature_cols)
        ),
    }

    os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
    with open(manifest_out, 'w') as f:
        json.dump(manifest, f, indent=4)
    logger.info(f'Manifest saved to {manifest_out}')