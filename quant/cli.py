import argparse
import logging
import os
import random
import numpy as np
import psutil
from pathlib import Path
import polars as pl
import json
import hashlib
from quant.config_manager import config, load_config
from quant.ingest import load_and_clean_data
from quant.features.engine import generate_features
from quant.discovery import run_feature_discovery
from quant.walkforward import run_walkforward
from quant.io.canonical_parquet import write_canonical_parquet
from quant.analytics import calculate_metrics, run_aggregation
logger = logging.getLogger(__name__)

def check_memory_safety():
    try:
        mem = psutil.Process().memory_info().rss
        if mem > config.RAM_CAP_BYTES:
            raise MemoryError(f'RSS {mem / 1024 ** 3:.2f}GB > cap {config.RAM_CAP_BYTES / 1024 ** 3:.2f}GB')
    except ImportError:
        pass

def prune_features_by_manifest(df, manifest_path, target_col):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    selected = manifest['feature_names']
    essential = {'ts_event', 'open', 'high', 'low', 'close', 'volume', 'session_id', 'regime'}
    non_feature = [c for c in df.columns if not c.startswith(('feature_', 'ratio_', 'pair_', 'zscore', 'cross_', 'htf_', '1h_', 'daily_')) and c not in essential]
    keep = list(essential) + non_feature + [c for c in selected if c in df.columns]
    keep = list(dict.fromkeys(keep))
    return df.select(keep)

def _stable_data_tag(data_arg: str) -> str:
    h = hashlib.sha256(data_arg.encode()).hexdigest()[:12]
    return h

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    discover_parser = subparsers.add_parser('discover')
    discover_parser.add_argument('--data', required=True)
    discover_parser.add_argument('--out', default='output/manifests/manifest.json')
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--data', required=True)
    run_parser.add_argument('--manifest', default='output/manifests/manifest.json')
    run_parser.add_argument('--out', required=True)
    aggregate_parser = subparsers.add_parser('aggregate')
    aggregate_parser.add_argument('--artifacts', default='output')
    args = parser.parse_args()
    load_config()  # populate config namespace from config.yaml
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    check_memory_safety()
    if args.command in ('discover', 'run'):
        from quant.market_config import detect_symbol_from_path, load_market_config
        symbol = detect_symbol_from_path(args.data)
        load_market_config(symbol)
    if args.command == 'discover':
        print('\n[CLI] === PHASE 1: FEATURE DISCOVERY ===', flush=True)
        cache_dir = Path('output/cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_tag = _stable_data_tag(args.data)
        aligned_cache = cache_dir / f'aligned_data_{data_tag}.parquet'
        print('[CLI] Loading and cleaning data...', flush=True)
        df_aligned = load_and_clean_data(args.data, cache_path=str(aligned_cache))
        print(f'[CLI] Data loaded. Rows: {df_aligned.height}', flush=True)
        print('[CLI] Generating feature matrix...', flush=True)
        df_features = generate_features(df_aligned)
        feature_cache = cache_dir / f'full_feature_matrix_{data_tag}.parquet'
        write_canonical_parquet(df_features, str(feature_cache))
        print(f'[CLI] Feature matrix saved to {feature_cache}', flush=True)
        print('[CLI] Running feature discovery...', flush=True)
        run_feature_discovery(str(feature_cache), args.out)
    elif args.command == 'run':
        print('\n[CLI] === PHASE 2: WALKFORWARD & EXECUTION ===', flush=True)
        target_col = 'target_sign_4h'
        cache_dir = Path('output/cache')
        data_tag = _stable_data_tag(args.data)
        aligned_cache = cache_dir / f'aligned_data_{data_tag}.parquet'
        feature_cache = cache_dir / f'full_feature_matrix_{data_tag}.parquet'
        print('[CLI] Loading aligned data...', flush=True)
        df_aligned = load_and_clean_data(args.data, cache_path=str(aligned_cache) if aligned_cache.exists() else None)
        if feature_cache.exists():
            print(f'[CLI] Loading cached feature matrix: {feature_cache}', flush=True)
            df_features = pl.read_parquet(feature_cache)
        else:
            print('[CLI] Generating feature matrix (no cache)...', flush=True)
            df_features = generate_features(df_aligned)
        print('[CLI] Applying manifest...', flush=True)
        if config.ENABLE_DISCOVERY:
            df_pruned = prune_features_by_manifest(df_features, args.manifest, target_col)
        else:
            print('[CLI] Discovery disabled — skipping manifest pruning, using baseline features only.', flush=True)
            df_pruned = df_features
        if target_col not in df_pruned.columns:
            raise KeyError(f"Target column '{target_col}' missing!")
        y = df_pruned[target_col]
        X = df_pruned.drop(target_col)
        _exclude = {'ts_event', 'open', 'high', 'low', 'close', 'volume', 'session_id', 'regime', 'date', 'benchmark_pnl'}
        _numeric_types = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        feature_cols = [c for c in X.columns if c not in _exclude and X[c].dtype in _numeric_types]
        print(f'[CLI] Running walkforward with {len(feature_cols)} features...', flush=True)
        result_df = run_walkforward(X, y, feature_cols, target_col)
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, 'backtest_results.parquet')
        result_df.write_parquet(out_path)
        print(f'[CLI] Results saved to {out_path}', flush=True)
        print('\n================ METRICS ================')
        calculate_metrics(out_path)
        print('========================================\n')
        try:
            print('[CLI] Running aggregation...', flush=True)
            run_aggregation()
        except Exception as e:
            print(f'[CLI] Aggregation skipped: {e}', flush=True)
    elif args.command == 'aggregate':
        print('\n[CLI] === AGGREGATING RESULTS ===', flush=True)
        run_aggregation(args.artifacts)
if __name__ == '__main__':
    main()