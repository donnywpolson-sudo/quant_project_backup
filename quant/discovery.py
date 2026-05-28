pass
import sys
print('Discovery started. Waiting for folds...', flush=True)
import os
import json
import logging
import numpy as np
import polars as pl
import psutil
import hashlib
from datetime import datetime
from sklearn.ensemble import ExtraTreesRegressor
from quant.config_manager import config
from tqdm import tqdm
logger = logging.getLogger(__name__)

def get_fold_seed(fold_idx: int) -> int:
    seed_str = f'{config.SEED}_fold_{fold_idx}'
    return int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % 2 ** 32

def check_rss(limit_bytes):
    return psutil.Process().memory_info().rss > limit_bytes

def run_feature_discovery(data_path: str, manifest_out: str):
    logger.info('Phase 1: Feature Discovery')
    try:
        df_features = pl.scan_parquet(data_path).collect(engine='streaming')
    except TypeError:
        df_features = pl.scan_parquet(data_path).collect(streaming=True)
    target_col = 'target_5m'
    if target_col not in df_features.columns:
        raise ValueError(f'Target column {target_col} not found.')
    df_features = df_features.with_columns(pl.col('ts_event').dt.convert_time_zone(config.TIMEZONE).dt.date().alias('date'))
    unique_dates = sorted(df_features['date'].unique().to_list())
    if len(unique_dates) > config.DISCOVERY_WINDOW_DAYS:
        cutoff_date = unique_dates[-config.DISCOVERY_WINDOW_DAYS]
        df_features = df_features.filter(pl.col('date') >= cutoff_date)
        logger.info(f'Discovery limited to {config.DISCOVERY_WINDOW_DAYS} days ({cutoff_date} onwards)')
    df_features = df_features.drop('date')
    if df_features.height > 200000:
        logger.info(f'Capping discovery rows from {df_features.height} to 200000 for memory safety')
        df_features = df_features.tail(200000)
    exclude_cols = {'ts_event', 'open', 'high', 'low', 'close', 'volume', 'session_id', target_col, 'regime', 'benchmark_pnl'}
    feature_cols = [c for c in df_features.columns if c not in exclude_cols and (not c.startswith('_'))]
    feature_cols = [c for c in feature_cols if df_features[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
    X = df_features.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y = df_features.select(target_col).to_numpy().astype(np.float32).ravel()
    logger.info(f'Discovery using {X.shape[0]} rows, {X.shape[1]} features.')
    rss_stop = config.RSS_STOP_BYTES
    n_folds = config.BOOTSTRAP_FOLDS
    importances_list = []
    signs_list = []
    for fold_idx in tqdm(range(n_folds), desc='Bootstrap folds', unit='fold'):
        print("Fold " + str(fold_idx + 1) + " started at " + datetime.now().strftime('%H:%M:%S'), flush=True)
        if check_rss(rss_stop):
            raise MemoryError(f'RSS stop limit exceeded in fold {fold_idx}')
        n_samples = X.shape[0]
        rng = np.random.RandomState(get_fold_seed(fold_idx))
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        et_params = config.EXTRA_TREES_PARAMS.copy()
        et_params['random_state'] = get_fold_seed(fold_idx)
        et = ExtraTreesRegressor(**et_params)
        et.fit(X_boot, y_boot)
        importances = dict(zip(feature_cols, et.feature_importances_))
        signs = {}
        for i, f in enumerate(feature_cols):
            with np.errstate(invalid='ignore'):
                corr = np.corrcoef(X_boot[:, i], y_boot)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            signs[f] = np.sign(corr)
        importances_list.append(importances)
        signs_list.append(signs)
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
        consistent = sum((1 for sd in signs_list if sd.get(f, 0) == majority_sign[f]))
        sign_consistency_frac[f] = consistent / n_folds
    freq = {f: selection_count[f] / n_folds for f in feature_cols}
    mean_imp = {f: importances_sum[f] / n_folds for f in feature_cols}
    selected = [f for f in feature_cols if freq[f] >= config.SELECTION_FREQ_THRESHOLD and sign_consistency_frac[f] >= config.SIGN_CONSISTENCY_THRESHOLD]
    selected_sorted = sorted(selected, key=lambda x: mean_imp[x], reverse=True)
    cumsum = 0.0
    final_selected = []
    total_imp = sum((mean_imp[f] for f in selected_sorted)) if selected_sorted else 1.0
    for f in selected_sorted:
        cumsum += mean_imp[f] / total_imp
        final_selected.append(f)
        if cumsum >= config.CUMULATIVE_IMPORTANCE_THRESHOLD:
            break
    if len(final_selected) < config.MIN_SELECTED_FEATURES:
        if len(selected_sorted) == 0:
            all_sorted = sorted(mean_imp.items(), key=lambda x: x[1], reverse=True)
            fallback_features = [f for f, _ in all_sorted[:config.MIN_SELECTED_FEATURES]]
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
                for f, _ in sorted(mean_imp.items(), key=lambda x: x[1], reverse=True):
                    if f not in final_selected:
                        final_selected.append(f)
                        needed -= 1
                        if needed == 0:
                            break
    logger.info(f'Selected {len(final_selected)} features (min required: {config.MIN_SELECTED_FEATURES}).')
    feature_list_str = json.dumps(sorted(final_selected), sort_keys=True).encode()
    features_hash = hashlib.sha256(feature_list_str).hexdigest()
    manifest = {'version': '1.0', 'feature_names': final_selected, 'dtypes': {f: 'float32' for f in final_selected}, 'selection_seed': config.SEED, 'selection_date': datetime.utcnow().isoformat() + 'Z', 'selection_model': 'ExtraTreesRegressor', 'selection_params': config.EXTRA_TREES_PARAMS, 'selected_K': len(final_selected), 'cumulative_importance': config.CUMULATIVE_IMPORTANCE_THRESHOLD, 'stability_stats': {'min_selection_freq': config.SELECTION_FREQ_THRESHOLD, 'sign_consistency': config.SIGN_CONSISTENCY_THRESHOLD, 'sign_consistency_observed': {f: round(sign_consistency_frac.get(f, 0), 3) for f in final_selected[:10]}}, 'baseline_feature_list': [c for c in feature_cols if c.startswith('feature_')][:40], 'baseline_features_hash': f'sha256:{features_hash}', 'baseline_feature_matrix_path': config.BASELINE_FEATURES_PERSIST_PATH, 'serialization_params': {'parquet_version': '2.0', 'compression': 'snappy', 'row_group_size': config.ROW_GROUP_SIZE, 'column_ordering': 'lexicographic'}, 'discovery_status': 'completed', 'folds': [], 'htf_features_included': any((c.startswith(('htf_', 'cross_', '1h_', 'daily_')) for c in feature_cols))}
    os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
    with open(manifest_out, 'w') as f:
        json.dump(manifest, f, indent=4)
    logger.info(f'Manifest saved to {manifest_out}')