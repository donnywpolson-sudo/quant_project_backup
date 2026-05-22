"""
src/cli.py
Entrypoint for the Deterministic Quant Pipeline.
Orchestrates Phase 1 (Feature Discovery via ExtraTrees) and 
Phase 2 (Enforced Manifest Pruning and Walk-Forward Ridge Simulation).
"""
import argparse
import logging
import os
import psutil
import polars as pl
from pathlib import Path

# Import project modules
from config import config
from src.discovery import run_feature_discovery
from src.features.engine import generate_features, prune_features_by_manifest
from src.walkforward import run_walkforward
from src.io.canonical_parquet import write_canonical_parquet

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def check_memory_safety():
    """
    Implements mandatory memory verification before loading matrices.
    Aborts immediately if current RSS usage violates RAM boundaries.
    """
    try:
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        ram_cap = getattr(config, "RAM_CAP_BYTES", 13.5 * 1024 * 1024 * 1024)
        if mem_bytes > ram_cap:
            logger.error(f"Memory ceiling breached: {mem_bytes / (1024**3):.2f} GB RSS. Aborting.")
            raise MemoryError("Pipeline execution terminated due to strict memory constraints.")
    except ImportError:
        pass

def main():
    parser = argparse.ArgumentParser(description="Deterministic Two-Phase Quant Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pipeline phase execution command")
    
    # -------------------------------------------------------------------------
    # Subcommand: discover (Phase 1)
    # -------------------------------------------------------------------------
    discover_parser = subparsers.add_parser("discover", help="Phase 1: Isolated Feature Discovery")
    discover_parser.add_argument("--data", required=True, help="Path to input parquet")
    discover_parser.add_argument("--out", default="artifacts/manifest.json", help="Output manifest path")

    # -------------------------------------------------------------------------
    # Subcommand: run (Phase 2)
    # -------------------------------------------------------------------------
    run_parser = subparsers.add_parser("run", help="Phase 2: Enforced Simulation")
    run_parser.add_argument("--data", required=True, help="Path to input parquet")
    run_parser.add_argument("--manifest", default="artifacts/manifest.json", help="Path to manifest")
    run_parser.add_argument("--out", required=True, help="Output directory")

    args = parser.parse_args()
    check_memory_safety()

    if args.command == "discover":
        logger.info("Initializing Stage 1: Running Feature Importance Ranker...")
        run_feature_discovery(args.data, args.out)
        logger.info("Stage 1 execution completed successfully.")

    elif args.command == "run":
        logger.info("Initializing Stage 2: Feature Generation and Backtesting...")
        
        if not os.path.exists(args.data):
            raise FileNotFoundError(f"Source data file not found at: {args.data}")
            
        df_raw = pl.read_parquet(args.data)
        check_memory_safety()

        # 1. Complete candidate pool expansion
        df_all_features = generate_features(df_raw)
        check_memory_safety()

        # 2. Apply the frozen structural layout contract
        df_pruned = prune_features_by_manifest(df_all_features, args.manifest)
        check_memory_safety()

        # 3. Extract explicit feature/target column signatures
        # Strictly enforce target column existence
        target_col = getattr(config, "TARGET_COL", "target")
        if target_col not in df_pruned.columns:
            raise KeyError(f"Target column '{target_col}' not found in engineered features. Pipeline stopped.")
        
        feature_cols = [c for c in df_pruned.columns if c.startswith("feature_")]
        if target_col in feature_cols:
            feature_cols.remove(target_col)

        logger.info(f"Using {len(feature_cols)} frozen features to predict target: {target_col}")

        # 4. Deterministic Parquet Serialization
        os.makedirs(args.out, exist_ok=True)
        out_parquet_path = os.path.join(args.out, "final_features.parquet")
        write_canonical_parquet(df_pruned.to_arrow(), out_parquet_path)
        check_memory_safety()

        # 5. Run Look-ahead safe Walk-Forward Validation
        df_backtest_results = run_walkforward(df_pruned, feature_cols, target_col)
        
        logger.info(f"Simulation completed. Evaluated backtest series size: {df_backtest_results.height} rows.")

if __name__ == "__main__":
    main()