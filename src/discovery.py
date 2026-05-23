"""
src/discovery.py
Phase 1: Feature discovery using ExtraTrees with bootstrap folds, stability selection,
memory isolation via joblib loky, RSS monitoring, and sign consistency filtering.
Now includes all feature types: baseline, ratios, pairwise, cross-timeframe, and HTF context.

Folds can be run in parallel (config.DISCOVERY_PARALLEL_FOLDS) without affecting determinism
because each fold has its own independent random seed derived from global seed and fold index.
"""
import sys
print("Discovery started. Waiting for folds...", flush=True)
import os
import json
import logging
import numpy as np
import polars as pl
import psutil
import hashlib
from datetime import datetime
from sklearn.ensemble import ExtraTreesRegressor
from joblib import Parallel, delayed
from config import config

logger = logging.getLogger(__name__)

def get_fold_seed(fold_idx: int) -> int:
    seed_str = f"{config.SEED}_fold_{fold_idx}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)

def check_rss(limit_bytes):
    return psutil.Process().memory_info().rss > limit_bytes

def fit_etree_fold(X, y, fold_idx, feature_names, rss_stop_bytes):
    """Fit ExtraTrees on one bootstrap sample. Returns importances dict and sign of correlation."""
    print(f"Fold {fold_idx+1} started at {datetime.now().strftime('%H:%M:%S')}")
    if check_rss(rss_stop_bytes):
        raise MemoryError(f"RSS stop limit exceeded in fold {fold_idx}")
    n_samples = X.shape[0]
    rng = np.random.RandomState(get_fold_seed(fold_idx))
    indices = rng.choice(n_samples, size=n_samples, replace=True)
    X_boot = X[indices]
    y_boot = y[indices]
    et_params = config.EXTRA_TREES_PARAMS.copy()
    et_params['random_state'] = get_fold_seed(fold_idx)
    et = ExtraTreesRegressor(**et_params)
    et.fit(X_boot, y_boot)
    importances = dict(zip(feature_names, et.feature_importances_))

    # Compute sign consistency: correlation between feature and target (simple proxy)
    signs = {}
    for i, f in enumerate(feature_names):
        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(X_boot[:, i], y_boot)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        signs[f] = np.sign(corr)
    return importances, signs

def run_feature_discovery(data_path: str, manifest_out: str):
    logger.info("Phase 1: Feature Discovery")
    df_raw = pl.read_parquet(data_path)
    from src.features.engine import generate_features
    df_features = generate_features(df_raw)

    target_col = "target_5m"
    if target_col not in df_features.columns:
        raise ValueError(f"Target column {target_col} not found.")
    
    # --- Include ALL feature columns (HTF, cross, etc.) ---
    exclude_cols = {
        "ts_event", "open", "high", "low", "close", "volume", 
        "session_id", "date", target_col, "regime", "benchmark_pnl"
    }
    feature_cols = [c for c in df_features.columns if c not in exclude_cols and not c.startswith("_")]
    feature_cols = [c for c in feature_cols if df_features[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
    
    htf_features = [c for c in feature_cols if c.startswith(("htf_", "cross_", "1h_", "daily_"))]
    if not htf_features:
        logger.warning("No HTF or cross-timeframe features found in feature set. Check generate_features.")
    else:
        logger.info(f"Discovery includes {len(htf_features)} HTF/cross features.")

    X = df_features.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y = df_features.select(target_col).to_numpy().astype(np.float32).ravel()

    n_bars = min(15840, X.shape[0])
    X = X[:n_bars]
    y = y[:n_bars]
    logger.info(f"Discovery using {X.shape[0]} rows, {X.shape[1]} features.")

    rss_stop = config.RSS_STOP_BYTES
    n_folds = config.BOOTSTRAP_FOLDS

    # --- Parallel folds (deterministic, zero loss of accuracy) ---
    n_parallel = min(getattr(config, 'DISCOVERY_PARALLEL_FOLDS', 1), n_folds)
    logger.info(f"Running {n_folds} bootstrap folds in parallel with {n_parallel} workers...")

    results = Parallel(n_jobs=n_parallel, backend='loky', verbose=10)(
        delayed(fit_etree_fold)(X, y, i, feature_cols, rss_stop)
        for i in range(n_folds)
    )

    importances_list = [r[0] for r in results]
    signs_list = [r[1] for r in results]

    # Compute selection frequencies and mean importance
    importances_sum = {f: 0.0 for f in feature_cols}
    selection_count = {f: 0 for f in feature_cols}
    n_folds = len(importances_list)

    for imp_dict, sign_dict in zip(importances_list, signs_list):
        for f, imp in imp_dict.items():
            importances_sum[f] += imp
            if imp > 0:
                selection_count[f] += 1

    # Determine majority sign per feature across folds
    majority_sign = {}
    for f in feature_cols:
        pos = sum(1 for sd in signs_list if sd.get(f, 0) > 0)
        neg = n_folds - pos
        majority_sign[f] = 1 if pos > neg else -1
    sign_consistency_frac = {}
    for f in feature_cols:
        consistent = sum(1 for sd in signs_list if sd.get(f, 0) == majority_sign[f])
        sign_consistency_frac[f] = consistent / n_folds

    freq = {f: selection_count[f] / n_folds for f in feature_cols}
    mean_imp = {f: importances_sum[f] / n_folds for f in feature_cols}

    # Apply frequency threshold AND sign consistency threshold
    selected = [f for f in feature_cols
                if freq[f] >= config.SELECTION_FREQ_THRESHOLD
                and sign_consistency_frac[f] >= config.SIGN_CONSISTENCY_THRESHOLD]
    selected_sorted = sorted(selected, key=lambda x: mean_imp[x], reverse=True)

    # Cumulative importance selection
    cumsum = 0.0
    final_selected = []
    total_imp = sum(mean_imp[f] for f in selected_sorted) if selected_sorted else 1.0
    for f in selected_sorted:
        cumsum += mean_imp[f] / total_imp
        final_selected.append(f)
        if cumsum >= config.CUMULATIVE_IMPORTANCE_THRESHOLD:
            break
    if len(final_selected) < config.MIN_SELECTED_FEATURES:
        final_selected = selected_sorted[:config.MIN_SELECTED_FEATURES]

    logger.info(f"Selected {len(final_selected)} features (sign consistency threshold={config.SIGN_CONSISTENCY_THRESHOLD}).")
    if htf_features:
        selected_htf = [f for f in final_selected if f.startswith(("htf_", "cross_", "1h_", "daily_"))]
        logger.info(f"Selected HTF/cross features: {len(selected_htf)} / {len(htf_features)}")

    # Compute hash of frozen feature list
    feature_list_str = json.dumps(sorted(final_selected), sort_keys=True).encode()
    features_hash = hashlib.sha256(feature_list_str).hexdigest()

    manifest = {
        "version": "1.0",
        "feature_names": final_selected,
        "dtypes": {f: "float32" for f in final_selected},
        "selection_seed": config.SEED,
        "selection_date": datetime.utcnow().isoformat() + "Z",
        "selection_model": "ExtraTreesRegressor",
        "selection_params": config.EXTRA_TREES_PARAMS,
        "selected_K": len(final_selected),
        "cumulative_importance": config.CUMULATIVE_IMPORTANCE_THRESHOLD,
        "stability_stats": {
            "min_selection_freq": config.SELECTION_FREQ_THRESHOLD,
            "sign_consistency": config.SIGN_CONSISTENCY_THRESHOLD,
            "sign_consistency_observed": {f: round(sign_consistency_frac[f], 3) for f in final_selected[:10]}
        },
        "baseline_feature_list": [c for c in feature_cols if c.startswith("feature_")][:40],
        "baseline_features_hash": f"sha256:{features_hash}",
        "baseline_feature_matrix_path": config.BASELINE_FEATURES_PERSIST_PATH,
        "serialization_params": {
            "parquet_version": "2.0",
            "compression": "snappy",
            "row_group_size": config.ROW_GROUP_SIZE,
            "column_ordering": "lexicographic"
        },
        "discovery_status": "completed",
        "folds": [],
        "htf_features_included": len(htf_features) > 0
    }
    os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=4)
    logger.info(f"Manifest saved to {manifest_out}")