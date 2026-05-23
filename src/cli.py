"""
src/cli.py
Entrypoint for the Deterministic Quant Pipeline.
Integrates resampling, discovery, walkforward, and execution.
Now with market‑specific config loading, single feature generation pass,
aligned data caching, and automatic performance metrics output.
"""
import argparse
import logging
import os
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
from src.analytics import calculate_metrics   # <-- import analytics function

logger = logging.getLogger(__name__)

def check_memory_safety():
    try:
        mem = psutil.Process().memory_info().rss
        if mem > config.RAM_CAP_BYTES:
            raise MemoryError(f"RSS {mem/(1024**3):.2f}GB > cap {config.RAM_CAP_BYTES/(1024**3):.2f}GB")
    except ImportError:
        pass

def prune_features_by_manifest(df: pl.DataFrame, manifest_path: str) -> pl.DataFrame:
    """Keep only features listed in manifest['feature_names']."""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    selected = manifest['feature_names']
    non_feature = [c for c in df.columns if not c.startswith(("feature_", "ratio_", "pair_", "zscore", "cross_", "htf_", "1h_", "daily_"))]
    keep = non_feature + [c for c in selected if c in df.columns]
    missing = set(selected) - set(df.columns)
    if missing:
        logger.warning(f"Missing features in manifest: {missing}")
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

    # Load market‑specific configuration if the command uses data
    if args.command in ("discover", "run"):
        from src.market_config import detect_symbol_from_path, load_market_config
        symbol = detect_symbol_from_path(args.data)
        load_market_config(symbol)

    if args.command == "discover":
        # Determine cache path for aligned data (store alongside manifest)
        cache_dir = Path(args.out).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        aligned_cache = cache_dir / "aligned_data.parquet"
        
        # Load aligned data (will use cache if exists)
        df_aligned = load_and_clean_data(args.data, cache_path=str(aligned_cache))
        
        # Generate full feature matrix (includes target, HTF, cross)
        df_features = generate_features(df_aligned)
        
        # Cache the full feature matrix for later reuse by "run"
        feature_cache = cache_dir / "full_feature_matrix.parquet"
        write_canonical_parquet(df_features, str(feature_cache))
        logger.info(f"Full feature matrix cached to {feature_cache}")
        
        # Run discovery on the cached matrix
        run_feature_discovery(str(feature_cache), args.out)

    elif args.command == "run":
        # Load aligned data (try cache first)
        cache_dir = Path(args.manifest).parent
        aligned_cache = cache_dir / "aligned_data.parquet"
        df_aligned = load_and_clean_data(args.data, cache_path=str(aligned_cache) if aligned_cache.exists() else None)
        check_memory_safety()

        # Try to load pre‑computed feature matrix from discovery phase
        feature_cache = cache_dir / "full_feature_matrix.parquet"
        if feature_cache.exists():
            logger.info(f"Loading pre‑computed feature matrix from {feature_cache}")
            df_features = pl.read_parquet(feature_cache)
        else:
            logger.info("No cached feature matrix found; generating features (this may be slower).")
            df_features = generate_features(df_aligned)

        # Prune to only features selected in manifest
        df_pruned = prune_features_by_manifest(df_features, args.manifest)
        target_col = "target_sign"
        if target_col not in df_pruned.columns:
            raise KeyError(f"Target {target_col} missing.")

        # All feature columns are those kept after pruning (excluding metadata)
        feature_cols = [c for c in df_pruned.columns
                        if c not in ("ts_event", "open", "high", "low", "close", "volume",
                                     "session_id", "date", target_col, "regime", "benchmark_pnl")]

        logger.info(f"Walkforward with {len(feature_cols)} features.")
        result_df = run_walkforward(df_pruned, feature_cols, target_col)

        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, "backtest_results.parquet")
        result_df.write_parquet(out_path)
        logger.info(f"Results saved to {out_path}")

        # --- AUTOMATIC METRICS OUTPUT ---
        print("\n" + "="*60)
        print("FINAL PERFORMANCE METRICS")
        print("="*60)
        # Call the analytics function on the saved file
        calculate_metrics(out_path)
        print("="*60 + "\n")

if __name__ == "__main__":
    main()