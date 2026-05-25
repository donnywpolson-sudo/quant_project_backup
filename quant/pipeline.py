"""
quant/pipeline.py
MVP pipeline entrypoint: basic I/O schema, baseline feature persistence, and a fast discovery run.
"""
import os
import json
import logging
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.ensemble import ExtraTreesRegressor

from config import config
from quant.ingest import load_and_clean_data
from quant.features.baseline import compute_baseline_features, load_baseline_feature_names
from quant.features.target import add_target_4h

logger = logging.getLogger(__name__)


def _ensure_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def run_pipeline(data_path: str, out_manifest: str = "artifacts/manifest_mvp.json", prod_mode: bool = False,
                 top_k: int = 20, discovery_run: bool = True, bootstrap_folds: int = 1):
    """Run a minimal pipeline: load data, compute 20 baseline features, persist baseline matrix,
    and optionally run a single ExtraTrees discovery to freeze top_k features.

    Args:
        data_path: glob or path to input parquet files (1-min schema)
        out_manifest: path to write manifest JSON
        prod_mode: if True, enable additional serialization (not implemented fully here)
        top_k: number of features to freeze from discovery
        discovery_run: whether to run discovery; set False to only persist baseline
    Returns:
        manifest dict
    """
    print("[PIPELINE] Loading and cleaning data...", flush=True)
    df = load_and_clean_data(data_path)
    print(f"[PIPELINE] rows: {df.height}", flush=True)

    # Compute full baseline features
    df = compute_baseline_features(df)
    # Add 4-hour target (MVP objective)
    df = add_target_4h(df)

    # Select up to top_k core baseline features from YAML order
    baseline_names = load_baseline_feature_names()
    available = [c for c in baseline_names if c in df.columns]
    selected = available[:top_k]
    # If not enough, supplement with any other feature_ columns
    if len(selected) < top_k:
        extras = [c for c in df.columns if c.startswith("feature_") and c not in selected]
        selected += extras[:(top_k - len(selected))]

    # Persist baseline matrix (session-scoped): include metadata and selected features
    baseline_out = config.BASELINE_FEATURES_PERSIST_PATH
    _ensure_dir(baseline_out)
    cols_to_write = ["ts_event", "session_id", "open", "high", "low", "close", "volume"] + selected
    baseline_df = df.select([c for c in cols_to_write if c in df.columns])
    baseline_df = baseline_df.with_columns([pl.col(c).cast(pl.Float32) for c in baseline_df.columns if baseline_df[c].dtype in (pl.Float64, )])
    baseline_df.write_parquet(baseline_out)
    print(f"[PIPELINE] Baseline matrix persisted to {baseline_out}", flush=True)

    manifest = {
        "version": "mvp-1",
        "feature_names": selected,
        "selection_method": "baseline_yaml_priority",
        "baseline_matrix_path": baseline_out,
        "prod_mode": bool(prod_mode),
    }

    # Fast discovery: ExtraTrees with optional bootstrap folds (optional)
    if discovery_run:
        print(f"[PIPELINE] Running ExtraTrees discovery (folds={bootstrap_folds})...", flush=True)
        target_col = "target_sign_4h"
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in feature matrix. Run target generation first.")
        X_all = df.select(selected).fill_null(0.0).to_numpy().astype(np.float32)
        y_all = df.select(target_col).fill_null(0).to_numpy().astype(np.float32).ravel()

        n_folds = max(1, int(bootstrap_folds))
        acc_importances = np.zeros(len(selected), dtype=np.float64)
        for fold in range(n_folds):
            rng = np.random.RandomState(config.SEED + fold)
            if n_folds == 1:
                Xb = X_all
                yb = y_all
                rs = config.SEED
            else:
                idx = rng.choice(X_all.shape[0], size=X_all.shape[0], replace=True)
                Xb = X_all[idx]
                yb = y_all[idx]
                rs = int(rng.randint(0, 2**31 - 1))
            et = ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=rs, n_jobs=1)
            et.fit(Xb, yb)
            acc_importances += np.array(et.feature_importances_, dtype=np.float64)

        mean_imp = acc_importances / n_folds
        importances = dict(zip(selected, mean_imp.tolist()))
        sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        frozen = [f for f, _ in sorted_feats[:top_k]]
        manifest.update({
            "selection_model": "ExtraTreesRegressor",
            "selection_params": {"n_estimators": 100, "max_depth": 8, "bootstrap_folds": n_folds},
            "frozen_features": frozen,
        })
        print(f"[PIPELINE] Frozen {len(frozen)} features.", flush=True)

    _ensure_dir(out_manifest)
    with open(out_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"[PIPELINE] Manifest written to {out_manifest}", flush=True)
    return manifest
