"""
src/cli.py
Entrypoint for the Deterministic Quant Pipeline.
Now with global seeding for reproducibility.
"""
import argparse
import logging
import os
import random
import numpy as np
import psutil
from pathlib import Path
import polars as pl
import json

from config import config
from src.ingest import load_and_clean_data
from src.features.engine import generate_features
from src.discovery import run_feature_discovery
from src.walkforward import run_walkforward
from src.io.canonical_parquet import write_canonical_parquet
from src.analytics import calculate_metrics

# Set deterministic seeds
random.seed(config.SEED)
np.random.seed(config.SEED)

logger = logging.getLogger(__name__)

def check_memory_safety():
    try:
        mem = psutil.Process().memory_info().rss
        if mem > config.RAM_CAP_BYTES:
            raise MemoryError(f"RSS {mem/(1024**3):.2f}GB > cap {config.RAM_CAP_BYTES/(1024**3):.2f}GB")
    except ImportError:
        pass

def prune_features_by_manifest(df: pl.DataFrame, manifest_path: str, target_col: str) -> pl.DataFrame:
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    selected = manifest['feature_names']
    essential = {"ts_event", "open", "high", "low", "close", "volume", "session_id", "regime"}
    non_feature = [c for c in df.columns 
                   if not c.startswith(("feature_", "ratio_", "pair_", "zscore", "cross_", "htf_", "1h_", "daily_"))
                   and c not in essential]
    keep = list(essential) + non_feature + [c for c in selected if c in df.columns]
    keep = list(dict.fromkeys(keep))
    return df.select(keep)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--data", required=True)
    discover_parser.add_argument("--out", default="artifacts/manifest.json")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--data", required=True)
    run_parser.add_argument("--manifest", default="artifacts/manifest.json")
    run_parser.add_argument("--out", required=True)

    args = parser.parse_args()
    check_memory_safety()

    if args.command in ("discover", "run"):
        from src.market_config import detect_symbol_from_path, load_market_config
        symbol = detect_symbol_from_path(args.data)
        load_market_config(symbol)

    if args.command == "discover":
        print("\n[CLI] === PHASE 1: FEATURE DISCOVERY ===", flush=True)
        cache_dir = Path(args.out).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        aligned_cache = cache_dir / "aligned_data.parquet"
        
        print("[CLI] Loading and cleaning data...", flush=True)
        df_aligned = load_and_clean_data(args.data, cache_path=str(aligned_cache))
        print(f"[CLI] Data loaded. Rows: {df_aligned.height}", flush=True)
        
        print("[CLI] Generating feature matrix...", flush=True)
        df_features = generate_features(df_aligned)
        if df_features.estimated_size() > config.RAM_CAP_BYTES:
            raise MemoryError("Feature matrix exceeds RAM cap.")
        
        feature_cache = cache_dir / "full_feature_matrix.parquet"
        write_canonical_parquet(df_features, str(feature_cache))
        logger.info(f"Full feature matrix cached to {feature_cache}")
        print(f"[CLI] Feature matrix saved to {feature_cache}", flush=True)
        
        print("[CLI] Running feature discovery...", flush=True)
        run_feature_discovery(str(feature_cache), args.out)

    elif args.command == "run":
        print("\n[CLI] === PHASE 2: WALKFORWARD & EXECUTION ===", flush=True)
        target_col = "target_sign"
        cache_dir = Path(args.manifest).parent
        aligned_cache = cache_dir / "aligned_data.parquet"
        cross_assets = getattr(config, 'CROSS_ASSET_SYMBOLS', [])
        print("[CLI] Loading aligned data...", flush=True)
        df_aligned = load_and_clean_data(args.data, 
                                         cache_path=str(aligned_cache) if aligned_cache.exists() else None,
                                         cross_asset_symbols=cross_assets)
        check_memory_safety()

        feature_cache = cache_dir / "full_feature_matrix.parquet"
        if feature_cache.exists():
            print(f"[CLI] Loading pre-computed feature matrix from {feature_cache}", flush=True)
            df_features = pl.read_parquet(feature_cache)
        else:
            print("[CLI] No cached feature matrix found; generating features (slower).", flush=True)
            df_features = generate_features(df_aligned)

        print("[CLI] Pruning features by manifest...", flush=True)
        df_pruned = prune_features_by_manifest(df_features, args.manifest, target_col)
        
        if target_col not in df_pruned.columns:
            raise KeyError(f"Target column '{target_col}' missing after pruning.")
        
        y = df_pruned.select(target_col)
        X = df_pruned.drop(target_col)
        
        excluded_meta = {"ts_event", "open", "high", "low", "close", "volume", "session_id", "regime"}
        feature_cols = [c for c in X.columns if c not in excluded_meta]
        
        if target_col in feature_cols:
            raise RuntimeError(f"Target column '{target_col}' still in feature columns! Aborting.")
        
        logger.info(f"Walkforward with {len(feature_cols)} features.")
        print(f"[CLI] Running walkforward with {len(feature_cols)} features...", flush=True)
        result_df = run_walkforward(X, y, feature_cols, target_col)

        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, "backtest_results.parquet")
        result_df.write_parquet(out_path)
        logger.info(f"Results saved to {out_path}")
        print(f"[CLI] Results saved to {out_path}", flush=True)

        print("\n" + "="*60)
        print("FINAL PERFORMANCE METRICS")
        print("="*60)
        calculate_metrics(out_path)
        print("="*60 + "\n")

if __name__ == "__main__":
    main()