import os
import sys

# ── UTF-8 everywhere (Windows hardening) ──
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

print('[HEARTBEAT] pipeline.cli imported', flush=True)

import argparse
import logging
import random
import numpy as np
import psutil
from pathlib import Path
import polars as pl
import json
import hashlib
import time
from core.io.atomic import atomic_write_parquet, atomic_write_json


def _check_target_contract(y, X, target_col):
    """Hard target validation — must pass before walkforward."""
    if len(y) == 0:
        raise RuntimeError('TARGET FAIL: y is empty after pruning')
    y_nan = y.null_count()
    if y_nan > 0:
        raise RuntimeError(
            'TARGET FAIL: %d NaN values in y (%s) after drop_incomplete_target' %
            (y_nan, target_col)
        )
    if X.height != len(y):
        raise RuntimeError(
            'TARGET FAIL: X/y misalignment (X=%d, y=%d)' % (X.height, len(y))
        )
    print('[TARGET] %s: rows=%d NaN=0 aligned=X(%d)' %
          (target_col, len(y), X.height), flush=True)


def _diag(df, stage):
    """Pipeline diagnostic — row count and ts_event span at each stage."""
    print(f'[DIAG] stage={stage} rows={df.height} cols={len(df.columns)}', flush=True)
    if 'ts_event' in df.columns and df.height > 0:
        t = df.select([pl.col('ts_event').min().alias('lo'), pl.col('ts_event').max().alias('hi')])
        print(f'[DIAG]   ts_event min={t["lo"][0]} max={t["hi"][0]}', flush=True)
from core.config import config, load_config
from _legacy.ingest import load_and_clean_data
from pipeline.features.engine import generate_features
from pipeline.features.discovery import run_feature_discovery
from pipeline.walkforward.walkforward import run_walkforward, run_walkforward_with_hmm, run_outer_train_test_eval, run_outer_train_test_eval_with_hmm
from core.io.canonical import write_canonical_parquet
from pipeline.analytics.aggregate import calculate_metrics, run_aggregation
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
    keep = list({c for c in essential if c in df.columns}) + non_feature + [c for c in selected if c in df.columns]
    keep = list(dict.fromkeys(keep))
    return df.select(keep)

def _stable_data_tag(data_arg: str, start: str = None, end: str = None) -> str:
    key = f"{config.ACTIVE_PROFILE}|{data_arg}"
    if start:
        key += '|' + start
    if end:
        key += '|' + end
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    return h


def _slice_window(df: pl.DataFrame, start_str: str, end_str: str, label: str) -> pl.DataFrame:
    from datetime import datetime as _dt, timezone
    if not start_str or not end_str:
        raise RuntimeError(f'MISSING BOUNDARY: {label} start/end required for outer_split mode')
    start_dt = _dt.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    end_dt = _dt.fromisoformat(end_str).replace(tzinfo=timezone.utc)
    before = df.height
    result = df.filter((pl.col('ts_event') >= start_dt) & (pl.col('ts_event') < end_dt))
    ts_min = result['ts_event'].min() if result.height > 0 else None
    ts_max = result['ts_event'].max() if result.height > 0 else None
    print(f'[OUTER-TRUE] {label} filter ({start_str} -> {end_str}): {before} -> {result.height} rows, ts=[{ts_min}, {ts_max})', flush=True)
    if result.height == 0:
        raise RuntimeError(f'EMPTY WINDOW: {label} [{start_str}, {end_str}) returned 0 rows')
    return result

def main():
    print('[HEARTBEAT] pipeline.cli main entered', flush=True)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    discover_parser = subparsers.add_parser('discover')
    discover_parser.add_argument('--data', required=True)
    discover_parser.add_argument('--out', default='output/manifests/manifest.json')
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--data', required=True)
    run_parser.add_argument('--manifest', default='output/manifests/manifest.json')
    run_parser.add_argument('--out', required=True)
    run_parser.add_argument('--start', default=None, help='Start date (ISO format)')
    run_parser.add_argument('--end', default=None, help='End date (ISO format)')
    run_parser.add_argument('--train-start', default=None, help='Train window start date (ISO)')
    run_parser.add_argument('--train-end', default=None, help='Train window end date (ISO)')
    run_hmm_parser = subparsers.add_parser('run-hmm')
    run_hmm_parser.add_argument('--data', required=True)
    run_hmm_parser.add_argument('--manifest', default='output/manifests/manifest.json')
    run_hmm_parser.add_argument('--out', required=True)
    run_hmm_parser.add_argument('--start', default=None, help='Start date (ISO format)')
    run_hmm_parser.add_argument('--end', default=None, help='End date (ISO format)')
    run_hmm_parser.add_argument('--train-start', default=None, help='Train window start date (ISO)')
    run_hmm_parser.add_argument('--train-end', default=None, help='Train window end date (ISO)')
    run_hmm_parser.add_argument('--retrain-interval', type=int, default=5,
                                help='Retrain HMM every N folds (default: 5).')
    aggregate_parser = subparsers.add_parser('aggregate')
    aggregate_parser.add_argument('--artifacts', default='output')
    args = parser.parse_args()
    load_config()  # populate config namespace from config.yaml
    wf_mode = getattr(config, 'WF_MODE', '')
    env_name = os.environ.get('CONFIG_ENV') or os.environ.get('QUANT_ENV', 'default')
    print(f'[CONFIG] env={env_name} WF_MODE={wf_mode!r} train_start={getattr(args, "train_start", None)} train_end={getattr(args, "train_end", None)} start={getattr(args, "start", None)} end={getattr(args, "end", None)}', flush=True)
    print(f'[BRANCH] command={args.command} using_outer_split={wf_mode == "outer_split"}', flush=True)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    check_memory_safety()
    if args.command in ('discover', 'run', 'run-hmm'):
        from core.market import detect_symbol_from_path, load_market_config
        symbol = detect_symbol_from_path(args.data)
        load_market_config(symbol)
        config.CURRENT_SYMBOL = symbol
    if args.command == 'discover':
        print('\n[CLI] === PHASE 1: FEATURE DISCOVERY ===', flush=True)
        cache_dir = Path('output/cache')
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_tag = _stable_data_tag(args.data)
        aligned_cache = cache_dir / f'aligned_data_{data_tag}.parquet'
        print('[CLI] Loading and cleaning data...', flush=True)
        df_aligned = load_and_clean_data(args.data, cache_path=str(aligned_cache))
        print(f'[CLI] Data loaded. Rows: {df_aligned.height}', flush=True)
        _diag(df_aligned, 'post-ingest')
        print('[CLI] Generating feature matrix...', flush=True)
        df_features = generate_features(df_aligned)
        _diag(df_features, 'post-generate-features')
        feature_cache = cache_dir / f'full_feature_matrix_{data_tag}.parquet'
        write_canonical_parquet(df_features, str(feature_cache))
        print(f'[CLI] Feature matrix saved to {feature_cache}', flush=True)
        print('[CLI] Running feature discovery...', flush=True)
        run_feature_discovery(str(feature_cache), args.out)
    elif args.command == 'run':
        print('\n[CLI] === PHASE 2: WALKFORWARD & EXECUTION ===', flush=True)
        target_col = 'target_sign_4h'  # 4h direction — features frozen from triple-barrier discovery
        cache_dir = Path('output/cache')
        data_tag = _stable_data_tag(args.data, getattr(args, 'start', None), getattr(args, 'end', None))
        aligned_cache = cache_dir / f'aligned_data_{data_tag}.parquet'
        feature_cache = cache_dir / f'full_feature_matrix_{data_tag}.parquet'
        print(f'[CLI] Cache key={data_tag} start={getattr(args, "start", None)} end={getattr(args, "end", None)}', flush=True)
        print('[CLI] Loading aligned data...', flush=True)
        df_aligned = load_and_clean_data(args.data, cache_path=str(aligned_cache) if aligned_cache.exists() else None)
        print(f'[CLI] Aligned data: {df_aligned.height} rows (from {"cache" if aligned_cache.exists() else "fresh ingest"})', flush=True)
        if feature_cache.exists():
            print(f'[CLI] Loading cached feature matrix: {feature_cache}', flush=True)
            df_features = pl.read_parquet(feature_cache)
            ts_dtype = df_features['ts_event'].dtype
            if ts_dtype != pl.Datetime(time_unit='us', time_zone='UTC'):
                df_features = df_features.with_columns(
                    pl.col('ts_event').cast(pl.Datetime(time_unit='us', time_zone='UTC'))
                )
        else:
            print('[CLI] Generating feature matrix (no cache)...', flush=True)
            df_features = generate_features(df_aligned)
        print(f'[CLI] Feature matrix: {df_features.height} rows, {len(df_features.columns)} cols', flush=True)
        print('[CLI] Applying manifest...', flush=True)
        if config.ENABLE_DISCOVERY:
            df_pruned = prune_features_by_manifest(df_features, args.manifest, target_col)
        else:
            print('[CLI] Discovery disabled - skipping manifest pruning.', flush=True)
            df_pruned = df_features
        print(f'[CLI] After manifest: {df_pruned.height} rows, {len(df_pruned.columns)} cols (target={"included" if target_col in df_pruned.columns else "MISSING"})', flush=True)
        # Per-split date window filtering (skipped in outer_split — _slice_window handles dual slicing)
        if getattr(config, 'WF_MODE', '') != 'outer_split':
            if getattr(args, 'start', None) and getattr(args, 'end', None):
                from datetime import datetime as _dt, timezone
                start_dt = _dt.fromisoformat(args.start).replace(tzinfo=timezone.utc)
                end_dt = _dt.fromisoformat(args.end).replace(tzinfo=timezone.utc)
                before = df_pruned.height
                df_pruned = df_pruned.filter(
                    (pl.col('ts_event') >= start_dt) & (pl.col('ts_event') < end_dt)
                )
                print(f'[CLI] Date filter ({args.start} -> {args.end}): {before} -> {df_pruned.height} rows', flush=True)
                if df_pruned.height == 0:
                    print('[CLI] Empty date window -- writing placeholder output and exiting', flush=True)
                    os.makedirs(args.out, exist_ok=True)
                    placeholder = pl.DataFrame(schema={'pnl': pl.Float32})
                    placeholder.write_parquet(os.path.join(args.out, 'backtest_results.parquet'))
                    print(f'[CLI] Empty backtest written to {args.out}', flush=True)
                    return
            # Assertion: all rows must be within the requested date window
            t_min = df_pruned['ts_event'].min()
            t_max = df_pruned['ts_event'].max()
            assert t_min >= start_dt, (
                f'SPLIT BOUNDARY VIOLATION: ts_event min={t_min} < start={start_dt}. '
                f'Upstream data or cache may have stale full-year data.'
            )
            assert t_max < end_dt, (
                f'SPLIT BOUNDARY VIOLATION: ts_event max={t_max} >= end={end_dt}. '
                f'Upstream data or cache may have stale full-year data.'
            )
            logger.info('[SPLIT] run: %d rows ts_event range [%s, %s) within [%s, %s)',
                df_pruned.height, t_min, t_max, start_dt, end_dt)
        if target_col not in df_pruned.columns:
            raise KeyError(f"Target column '{target_col}' missing!")
        y = df_pruned[target_col]
        X = df_pruned.drop(target_col)

        # ---- Target contract enforcement (run path) ----
        _check_target_contract(y, X, target_col)

        _exclude = {'ts_event', 'open', 'high', 'low', 'close', 'volume', 'session_id', 'regime', 'date', 'benchmark_pnl'}
        _exclude |= {c for c in X.columns if c.startswith('target_')}
        _numeric_types = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        feature_cols = [c for c in X.columns if c not in _exclude and X[c].dtype in _numeric_types]
        print(f'[CLI] Running walkforward with {len(feature_cols)} features...', flush=True)
        if getattr(config, 'WF_MODE', '') == 'outer_split':
            assert getattr(args, 'train_start', None), 'OUTER_SPLIT active but --train-start missing from CLI args'
            assert getattr(args, 'train_end', None), 'OUTER_SPLIT active but --train-end missing from CLI args'
            train_df = _slice_window(df_pruned, getattr(args, 'train_start', None), getattr(args, 'train_end', None), 'train')
            test_df = _slice_window(df_pruned, getattr(args, 'start', None), getattr(args, 'end', None), 'test')
            result_df = run_outer_train_test_eval(train_df, test_df, feature_cols, target_col)
        else:
            result_df = run_walkforward(X, y, feature_cols, target_col)
        print(f'[CLI] Walkforward result: {result_df.height} rows (input was {X.height} rows, features={len(feature_cols)})', flush=True)
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, 'backtest_results.parquet')
        atomic_write_parquet(result_df, out_path)
        print(f'[CLI] Results saved to {out_path}', flush=True)
        print('\n================ METRICS ================')
        calculate_metrics(out_path)
        print('========================================\n')
        try:
            print('[CLI] Running aggregation...', flush=True)
            run_aggregation('output')
        except Exception as e:
            print(f'[CLI] Aggregation skipped: {e}', flush=True)
    elif args.command == 'run-hmm':
        t0 = time.perf_counter()
        print('\n[CLI] === PHASE 2H: WALKFORWARD + HMM REGIME FILTER ===', flush=True)
        print('[HEARTBEAT] cli entered run-hmm', flush=True)
        print(f'[HEARTBEAT] config loaded WF_MODE={getattr(config, "WF_MODE", "")!r}', flush=True)
        target_col = 'target_sign_4h'  # 4h direction — features frozen from triple-barrier discovery
        cache_dir = Path('output/cache')
        data_tag = _stable_data_tag(args.data, getattr(args, 'start', None), getattr(args, 'end', None))
        aligned_cache = cache_dir / f'aligned_data_{data_tag}.parquet'
        feature_cache = cache_dir / f'full_feature_matrix_{data_tag}.parquet'
        print(f'[CLI] Cache key={data_tag} start={getattr(args, "start", None)} end={getattr(args, "end", None)}', flush=True)
        print('[HEARTBEAT] loading aligned data start', flush=True)
        t_load = time.perf_counter()
        df_aligned = load_and_clean_data(
            args.data,
            cache_path=str(aligned_cache) if aligned_cache.exists() else None,
        )
        dt_load = time.perf_counter() - t_load
        print(f'[HEARTBEAT] loading aligned data done rows={df_aligned.height} seconds={dt_load:.1f}', flush=True)
        if dt_load > 60:
            print(f'[SLOW] stage=load_aligned_data seconds={dt_load:.1f}', flush=True)
        print('[HEARTBEAT] generating/loading feature matrix start', flush=True)
        t_feat = time.perf_counter()
        if feature_cache.exists():
            print(f'[CLI] Loading cached feature matrix: {feature_cache}', flush=True)
            df_features = pl.read_parquet(feature_cache)
            ts_dtype = df_features['ts_event'].dtype
            if ts_dtype != pl.Datetime(time_unit='us', time_zone='UTC'):
                df_features = df_features.with_columns(
                    pl.col('ts_event').cast(pl.Datetime(time_unit='us', time_zone='UTC'))
                )
        else:
            print('[CLI] Generating feature matrix (no cache)...', flush=True)
            df_features = generate_features(df_aligned)
        dt_feat = time.perf_counter() - t_feat
        print(f'[HEARTBEAT] feature matrix done rows={df_features.height} cols={len(df_features.columns)} seconds={dt_feat:.1f}', flush=True)
        if dt_feat > 60:
            print(f'[SLOW] stage=feature_matrix seconds={dt_feat:.1f}', flush=True)
        print(f'[CLI] Feature matrix: {df_features.height} rows, {len(df_features.columns)} cols', flush=True)
        print('[HEARTBEAT] applying manifest start', flush=True)
        t_man = time.perf_counter()
        print('[CLI] Applying manifest...', flush=True)
        if config.ENABLE_DISCOVERY:
            df_pruned = prune_features_by_manifest(df_features, args.manifest, target_col)
        else:
            print('[CLI] Discovery disabled - skipping manifest pruning.', flush=True)
            df_pruned = df_features
        print(f'[CLI] After manifest: {df_pruned.height} rows, {len(df_pruned.columns)} cols', flush=True)
        dt_man = time.perf_counter() - t_man
        print(f'[HEARTBEAT] manifest done rows={df_pruned.height} cols={len(df_pruned.columns)} seconds={dt_man:.1f}', flush=True)
        if dt_man > 60:
            print(f'[SLOW] stage=manifest seconds={dt_man:.1f}', flush=True)
        # Per-split date window filtering (skipped in outer_split — _slice_window handles dual slicing)
        if getattr(config, 'WF_MODE', '') != 'outer_split':
            if getattr(args, 'start', None) and getattr(args, 'end', None):
                from datetime import datetime as _dt, timezone
                start_dt = _dt.fromisoformat(args.start).replace(tzinfo=timezone.utc)
                end_dt = _dt.fromisoformat(args.end).replace(tzinfo=timezone.utc)
                before = df_pruned.height
                df_pruned = df_pruned.filter(
                    (pl.col('ts_event') >= start_dt) & (pl.col('ts_event') < end_dt)
                )
                print(f'[CLI] Date filter ({args.start} -> {args.end}): {before} -> {df_pruned.height} rows', flush=True)
                if df_pruned.height == 0:
                    print('[CLI] Empty date window — writing placeholder output and exiting', flush=True)
                    os.makedirs(args.out, exist_ok=True)
                    placeholder = pl.DataFrame(schema={'pnl': pl.Float32})
                    placeholder.write_parquet(os.path.join(args.out, 'backtest_results_hmm.parquet'))
                    print(f'[CLI] Empty backtest written to {args.out}', flush=True)
                    return
            t_min = df_pruned['ts_event'].min()
            t_max = df_pruned['ts_event'].max()
            assert t_min >= start_dt, f'SPLIT BOUNDARY VIOLATION: min={t_min} < start={start_dt}'
            assert t_max < end_dt, f'SPLIT BOUNDARY VIOLATION: max={t_max} >= end={end_dt}'
            logger.info('[SPLIT] run-hmm: %d rows ts_event [%s, %s) within [%s, %s)',
                df_pruned.height, t_min, t_max, start_dt, end_dt)
        if target_col not in df_pruned.columns:
            raise KeyError(f"Target column '{target_col}' missing!")
        y = df_pruned[target_col]
        X = df_pruned.drop(target_col)

        # ---- Target contract enforcement (hmm path) ----
        _check_target_contract(y, X, target_col)

        _exclude = {
            'ts_event', 'open', 'high', 'low', 'close', 'volume',
            'session_id', 'regime', 'date', 'benchmark_pnl',
        }
        _exclude |= {c for c in X.columns if c.startswith('target_')}
        _numeric_types = (
            pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        )
        feature_cols = [
            c for c in X.columns
            if c not in _exclude and X[c].dtype in _numeric_types
        ]
        print(
            f'[CLI] Running HMM-aware walkforward with '
            f'{len(feature_cols)} features '
            f'(HMM retrain every {args.retrain_interval} folds)...',
            flush=True,
        )
        if getattr(config, 'WF_MODE', '') == 'outer_split':
            assert getattr(args, 'train_start', None), 'OUTER_SPLIT HMM active but --train-start missing from CLI args'
            assert getattr(args, 'train_end', None), 'OUTER_SPLIT HMM active but --train-end missing from CLI args'
            print('[HEARTBEAT] slicing train/test start', flush=True)
            t_slice = time.perf_counter()
            train_df = _slice_window(df_pruned, getattr(args, 'train_start', None), getattr(args, 'train_end', None), 'train')
            test_df = _slice_window(df_pruned, getattr(args, 'start', None), getattr(args, 'end', None), 'test')
            dt_slice = time.perf_counter() - t_slice
            print(f'[HEARTBEAT] train rows={train_df.height} test rows={test_df.height} seconds={dt_slice:.1f}', flush=True)
            if dt_slice > 60:
                print(f'[SLOW] stage=slicing seconds={dt_slice:.1f}', flush=True)
            print('[HEARTBEAT] walkforward+hmm start', flush=True)
            t_wf = time.perf_counter()
            result_df, validation = run_outer_train_test_eval_with_hmm(
                train_df, test_df, feature_cols, target_col,
            )
            dt_wf = time.perf_counter() - t_wf
            print(f'[HEARTBEAT] walkforward+hmm done rows={result_df.height} seconds={dt_wf:.1f}', flush=True)
            if dt_wf > 60:
                print(f'[SLOW] stage=walkforward+hmm seconds={dt_wf:.1f}', flush=True)
        else:
            result_df, validation = run_walkforward_with_hmm(
                X, y, feature_cols, target_col,
                hmm_retrain_interval=args.retrain_interval,
            )
        print(f'[CLI] HMM walkforward result: {result_df.height} rows (input was {X.height} rows, features={len(feature_cols)})', flush=True)
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, 'backtest_results_hmm.parquet')
        print('[HEARTBEAT] writing parquet start', flush=True)
        t_write = time.perf_counter()
        atomic_write_parquet(result_df, out_path)
        dt_write = time.perf_counter() - t_write
        print(f'[HEARTBEAT] writing parquet done path={out_path} seconds={dt_write:.1f}', flush=True)
        if dt_write > 60:
            print(f'[SLOW] stage=parquet_write seconds={dt_write:.1f}', flush=True)
        dt_total = time.perf_counter() - t0
        print(f'[HEARTBEAT] total run-hmm seconds={dt_total:.1f}', flush=True)
        print(f'[CLI] HMM-filtered results saved to {out_path}', flush=True)
        # Save validation report
        val_path = os.path.join(args.out, 'hmm_validation_report.json')
        atomic_write_json(validation, val_path)
        print(f'[CLI] Validation report saved to {val_path}', flush=True)
        print('\n================ HMM METRICS ================')
        calculate_metrics(out_path)
        print('==============================================\n')
        if not validation.get('fallback_triggered', True):
            print(
                f"[CLI] PSR: {validation['psr_result']['psr']:.4f} | "
                f"Significant: {validation['psr_result']['significant']} | "
                f"ΔSR: {validation['psr_result']['sharpe_difference']:+.4f}",
                flush=True,
            )
        else:
            print(
                f"[CLI] HMM FALLBACK ACTIVE: {validation.get('fallback_reason', 'unknown')}",
                flush=True,
            )
        print(f"[CLI] Recommendation: {validation.get('recommendation', 'N/A')}", flush=True)
        try:
            print('[CLI] Running aggregation...', flush=True)
            run_aggregation('output')
        except Exception as e:
            print(f'[CLI] Aggregation skipped: {e}', flush=True)
    elif args.command == 'aggregate':
        print('\n[CLI] === AGGREGATING RESULTS ===', flush=True)
        run_aggregation(args.artifacts)
if __name__ == '__main__':
    main()