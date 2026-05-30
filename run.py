import os
import sys
import subprocess
import logging
import time
import json
from pathlib import Path
import shutil
import polars as pl
import numpy as np
from core.config import load_config, RootConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('QuantRunner')
_VERBOSE = os.environ.get('QUANT_VERBOSE', '0') == '1'
_MIN_TRAIN_DAYS = 90

def get_files(data_dir: str, config: RootConfig) -> list[Path]:
    files = list(Path(data_dir).rglob('*.parquet'))
    allowed_markets = set(config.symbols)
    valid = []
    for f in files:
        try:
            year = int(f.stem)
        except Exception:
            continue
        if year < config.start_year:
            continue
        if year > config.end_year:
            continue
        if allowed_markets and f.parent.name not in allowed_markets:
            continue
        valid.append(f)
    valid = sorted(valid, key=lambda x: (x.parent.name, x.stem))
    if config.io.max_files:
        valid = valid[:config.io.max_files]
    return valid

def _load_file_date_bounds(files: list[Path]) -> dict[int, tuple]:
    """
    Scan each yearly parquet file to extract its (min_date, max_date) range.
    Uses lazy scanning to avoid loading full datasets into memory.
    Returns dict mapping file_stem_year -> (date_min, date_max).
    """
    import datetime as _dt
    bounds: dict[int, tuple] = {}
    for f in files:
        try:
            year = int(f.stem)
        except ValueError:
            continue
        try:
            lf = pl.scan_parquet(f).select(pl.col('ts_event'))
            min_ts = lf.select(pl.col('ts_event').min()).collect().item()
            max_ts = lf.select(pl.col('ts_event').max()).collect().item()
            if min_ts is not None and max_ts is not None:
                bounds[year] = (min_ts.date(), max_ts.date())
        except Exception:
            logger.debug('Could not read date bounds from %s — skipping', f)
    return bounds


def generate_walkforward_splits(files: list[Path], config: RootConfig) -> list[tuple[list[int], list[int]]]:
    """
    Generate walk-forward split index pairs.

    When the walkforward config defines positive *day-based* parameters
    (wf_train_days / wf_test_days / wf_step_days), each split is a
    rolling calendar-day window anchored to the full date range of the
    available data.  Files are assigned to each window based on whether
    their date span overlaps the train or test interval.

    Otherwise falls back to legacy year-count splitting.
    """
    import datetime as _dt

    wf_cfg = config.walkforward

    # ---- Day-based rolling-window mode ------------------------------------
    day_based = (
        wf_cfg.wf_train_days > 0
        and wf_cfg.wf_test_days > 0
        and wf_cfg.wf_step_days > 0
    )

    if day_based:
        file_bounds = _load_file_date_bounds(files)
        if not file_bounds:
            logger.warning('No file date bounds available — returning empty splits')
            return []

        all_dates = sorted({
            d for b in file_bounds.values()
            for d in (b[0], b[1])
        })
        if not all_dates:
            logger.warning('No dates extracted from files')
            return []

        data_start = all_dates[0]
        data_end = all_dates[-1]
        total_days = (data_end - data_start).days + 1
        window = wf_cfg.wf_train_days + wf_cfg.wf_test_days

        if total_days < window:
            logger.warning(
                'Insufficient date range for day-based walkforward: '
                'need at least %d days, have %d days (%s → %s)',
                window, total_days, data_start, data_end,
            )
            # Single split covering the full range — temporal separation
            # is enforced inside cli.py via _build_ts_folds which slices
            # by actual timestamps, not file boundaries.
            all_years = sorted(file_bounds.keys())
            return [(all_years, all_years)]

        splits: list[tuple[list[int], list[int]]] = []
        cursor = 0  # day offset from data_start

        while cursor + window <= total_days:
            train_start = data_start + _dt.timedelta(days=cursor)
            train_end   = data_start + _dt.timedelta(days=cursor + wf_cfg.wf_train_days)
            test_start  = train_end
            test_end    = data_start + _dt.timedelta(days=cursor + window)

            # File assignment by date overlap — yearly parquet files span
            # full calendar years. Overlap-based assignment is safe because
            # cli.py's _build_ts_folds enforces strict temporal separation
            # by slicing within each file at the timestamp level.
            train_files = [
                yr for yr, (fmin, fmax) in file_bounds.items()
                if fmax >= train_start and fmin < train_end
            ]
            test_files = [
                yr for yr, (fmin, fmax) in file_bounds.items()
                if fmax >= test_start and fmin < test_end
            ]

            if train_files and test_files:
                splits.append((sorted(train_files), sorted(test_files),
                               test_start, test_end))
                logger.debug(
                    'Split %d: train=%s→%s (%d files) | test=%s→%s (%d files)',
                    len(splits),
                    train_start.isoformat(), train_end.isoformat(), len(train_files),
                    test_start.isoformat(), test_end.isoformat(), len(test_files),
                )

            cursor += wf_cfg.wf_step_days
            # Safety guard for very large step counts (prevent accidental infinite loop)
            if len(splits) >= 10_000:
                logger.warning('Split limit 10,000 reached — truncating')
                break

        if splits:
            logger.info(
                'Generated %d walkforward splits (day-based: train=%dd, test=%dd, step=%dd, '
                'range=%s→%s)',
                len(splits),
                wf_cfg.wf_train_days, wf_cfg.wf_test_days, wf_cfg.wf_step_days,
                data_start.isoformat(), data_end.isoformat(),
            )
            return splits

        # No valid rolling windows produced — fall back to single split.
        all_years = sorted(file_bounds.keys())
        logger.warning(
            'No day-based splits generated (step %d may be too large for '
            'the %d-day range). Falling back to single merged split.',
            wf_cfg.wf_step_days, total_days,
        )
        return [(all_years, all_years)]

    # ---- Legacy year-count mode ------------------------------------------
    train_years = config.data_years
    wf_years = config.folds
    years = sorted({int(f.stem) for f in files})

    if not years:
        logger.warning('No years available for walkforward splitting')
        return []

    min_years_needed = train_years + wf_years

    splits = []
    for i in range(len(years)):
        train_start = i
        train_end = i + train_years
        test_end = train_end + wf_years
        if test_end > len(years):
            break
        train_range = years[train_start:train_end]
        test_range = years[train_end:test_end]
        splits.append((train_range, test_range))

    if not splits:
        train_range = years[:train_years] if len(years) >= train_years else years[:-1] if len(years) > 1 else []
        test_range = years[train_years:] if len(years) > train_years else []
        if not train_range or not test_range:
            logger.error(
                'Insufficient years for walkforward: need at least %d years '
                '(train=%d + test=%d), have %d. No splits generated.',
                min_years_needed, train_years, wf_years, len(years),
            )
            return []
        logger.warning(
            'Insufficient years for walkforward: need %d (train=%d + test=%d), '
            'have %d. Falling back to reduced split: train=%s, test=%s',
            min_years_needed, train_years, wf_years, len(years),
            train_range, test_range,
        )
        splits.append((train_range, test_range))

    logger.info('Generated %d walkforward splits', len(splits))
    return splits


_CONTRACT_MULTIPLIERS = {
    'CL': 1000, 'ES': 50, 'NQ': 20, 'RTY': 50,
    'GC': 100, 'SI': 5000, 'HG': 25000,
    'ZB': 1000, 'ZN': 1000, 'ZF': 1000, 'ZT': 2000, 'ZC': 50, 'NG': 10000,
    '6E': 125000, '6J': 12500000, '6B': 62500, 'YM': 5,
}
_MIN_ROWS = 1000
_MIN_PRICE = {'CL': 0.1, 'ES': 500, 'NQ': 1000, 'GC': 500, 'SI': 5, 'HG': 1,
              'ZB': 50, 'ZN': 50, 'ZC': 50, 'NG': 0.5}
_MAX_PRICE = {'CL': 200, 'ES': 10000, 'NQ': 30000, 'GC': 5000, 'SI': 60, 'HG': 10,
              'ZB': 250, 'ZN': 250, 'ZC': 20, 'NG': 30}


def _validate_symbol_data(f: Path, config) -> bool:
    symbol = f.parent.name
    try:
        df = pl.read_parquet(f)
    except Exception as e:
        logger.error('[SAFETY] Cannot read %s: %s -- SKIPPING', f, e)
        return False
    if df.height < _MIN_ROWS:
        logger.error('[SAFETY] %s: %d rows < %d minimum -- SKIPPING', symbol, df.height, _MIN_ROWS)
        return False
    if 'ts_event' not in df.columns:
        logger.error('[SAFETY] %s: missing ts_event column -- SKIPPING', symbol)
        return False
    if 'close' not in df.columns:
        logger.error('[SAFETY] %s: missing close column -- SKIPPING', symbol)
        return False
    null_ts = df['ts_event'].null_count()
    if null_ts > 0:
        logger.error('[SAFETY] %s: %d null ts_event values -- SKIPPING', symbol, null_ts)
        return False
    close_mean = df['close'].mean()
    lo, hi = _MIN_PRICE.get(symbol, 0), _MAX_PRICE.get(symbol, 1e9)
    if close_mean < lo or close_mean > hi:
        logger.error('[SAFETY] %s: close mean %.2f outside [%.2f, %.2f] -- SKIPPING',
                     symbol, close_mean, lo, hi)
        return False
    ts_min = df['ts_event'].min()
    ts_max = df['ts_event'].max()
    if ts_min is None or ts_max is None:
        logger.error('[SAFETY] %s: null ts_event range -- SKIPPING', symbol)
        return False
    available_days = (ts_max - ts_min).days
    wf_window = getattr(config, 'WF_TRAIN_DAYS', 30) + getattr(config, 'WF_TEST_DAYS', 1)
    if available_days < wf_window * 2:
        logger.warning('[SAFETY] %s: %d days available < %d required (2x window) -- may have few folds',
                       symbol, available_days, wf_window * 2)
    logger.info('[SAFETY] %s: OK rows=%d close=%.2f range=%s->%s days=%d',
                symbol, df.height, close_mean, ts_min.date(), ts_max.date(), available_days)
    return True


def _print_split_dashboard(split_idx: int, total: int, per_symbol: dict):
    if not per_symbol:
        return
    h = '\u2500'
    tl, tr, bl, br, v, c = '\u250c', '\u2510', '\u2514', '\u2518', '\u2502', '\u251c'
    r = '\u2524'
    check, cross, warn = '\u2705', '\u274c', '\u26a0\ufe0f'
    print(f'\n{tl}{h * 52}{tr}')
    print(f'{v} [SPLIT {split_idx}/{total}] Processing Assets...{" " * 3}{v}')
    print(f'{c}{h * 52}{r}')
    combined_sharpe = 0.0
    for symbol, entry in sorted(per_symbol.items()):
        sharpe = entry[0]; pnl = entry[1]; hit = entry[2]
        hmm_delta = entry[3] if len(entry) > 3 else 0.0
        icon = check if sharpe > 0 else cross
        delta_str = f' HMM_delta={hmm_delta:+.2f}' if abs(hmm_delta) > 0.001 else ''
        print(f'{v}  {icon} {symbol:<4s}  Sharpe={sharpe:+.2f}  PnL={pnl:+,.2f}  Hit={hit:.1%}{delta_str} {"":>6s}{v}')
        combined_sharpe += sharpe
    status = f'{check} Success' if combined_sharpe > 0 else f'{warn} Mixed'
    print(f'{v} {"":>24s}Combined Sharpe: {combined_sharpe:+.3f}  Status: {status}{v}')
    print(f'{bl}{h * 52}{br}\n', flush=True)


def _validate_backtest_output(out_dir: Path, symbol: str) -> None:
    bt_path = out_dir / 'backtest_results.parquet'
    if not bt_path.exists():
        raise RuntimeError('BACKTEST FAILURE: %s no output at %s' % (symbol, bt_path))
    try:
        df = pl.read_parquet(bt_path)
    except Exception as e:
        raise RuntimeError('BACKTEST FAILURE: %s cannot read %s: %s' % (symbol, bt_path, e))
    if df.height == 0:
        raise RuntimeError('BACKTEST FAILURE: %s empty output at %s' % (symbol, bt_path))
    if 'pnl' not in df.columns:
        raise RuntimeError('BACKTEST FAILURE: %s missing pnl column at %s' % (symbol, bt_path))
    pnl_sum = df['pnl'].sum()
    pnl_mean = df['pnl'].mean()
    pnl_std = df['pnl'].std()
    multiplier = _CONTRACT_MULTIPLIERS.get(symbol, 1)
    notional = df['close'].mean() * multiplier if 'close' in df.columns else 0
    logger.info('[VALIDATE] %s: rows=%d pnl_sum=%.2f pnl_mean=%.6f pnl_std=%.4f notional=%.0f mult=%d',
                symbol, df.height, pnl_sum, pnl_mean, pnl_std, notional, multiplier)
    if abs(pnl_std) < 0.0001 and multiplier > 1:
        raise RuntimeError(
            'BACKTEST FAILURE: %s pnl std %.8f is zero-scale '
            '(multiplier=%d, notional=%.0f) -- PnL not in USD futures' %
            (symbol, pnl_std, multiplier, notional)
        )


def load_all_splits_for_year(year: int) -> list:
    """Load per-split manifests and backtest results for a given year."""
    import glob as _glob
    manifests = sorted(_glob.glob(f'output/manifest_{year}_split_*.json'))
    if not manifests:
        fallback = Path(f'output/manifest_{year}.json')
        if fallback.exists():
            manifests = [str(fallback)]
    summaries = []
    for mpath in manifests:
        with open(mpath, 'r', encoding='utf-8') as f:
            mf = json.load(f)
        out_path = mf.get('output_path')
        if out_path and Path(out_path).exists():
            try:
                df = pl.read_parquet(out_path)
                summaries.append({
                    'symbol': mf.get('symbol'),
                    'year': mf.get('year'),
                    'split_idx': mf.get('split_idx'),
                    'rows': df.height,
                })
            except Exception as e:
                summaries.append({'symbol': mf.get('symbol'), 'year': mf.get('year'),
                                  'split_idx': mf.get('split_idx'), 'error': str(e)})
        else:
            summaries.append({'symbol': mf.get('symbol'), 'year': mf.get('year'),
                              'split_idx': mf.get('split_idx'), 'error': 'backtest file missing'})
    return summaries


def process_split(train_years: list[int], test_years: list[int], files: list[Path],
                  config: RootConfig, split_idx: int, total_splits: int,
                  test_start=None, test_end=None) -> None:
    train_files = [f for f in files if int(f.stem) in train_years]
    test_files = [f for f in files if int(f.stem) in test_years]
    if not train_files or not test_files:
        logger.warning('Empty train/test split — skipping')
        return
    # ---- Config safeguard: bootstrap discovery needs enough samples ----
    wf_train = getattr(config.walkforward, 'wf_train_days', 30)
    if wf_train < _MIN_TRAIN_DAYS:
        logger.warning(
            'wf_train_days=%d < %d — discovery may have insufficient samples. '
            'Consider wf_train_days >= %d for stable bootstrap.',
            wf_train, _MIN_TRAIN_DAYS, _MIN_TRAIN_DAYS
        )
    train_dir = Path('output') / f"train_{'_'.join(map(str, train_years))}_split_{split_idx}"
    train_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Preparing TRAIN dataset for years %s', train_years)
    for f in train_files:
        dst = train_dir / f'{f.parent.name}_{f.name}'
        if dst.exists():
            continue
        if os.name != 'nt':
            try:
                os.symlink(f.resolve(), dst)
                continue
            except Exception:
                pass
        shutil.copy2(f, dst)
    train_glob = str(train_dir / '*.parquet')
    manifest_path = Path('output') / f"manifest_{'_'.join(map(str, train_years))}_split_{split_idx}.json"
    # Silence subprocess output unless verbose mode
    env = os.environ.copy()
    env['TQDM_DISABLE'] = '1'
    kw = {'check': True, 'env': env, 'stderr': subprocess.PIPE}
    if not _VERBOSE:
        kw['stdout'] = subprocess.DEVNULL
    if config.pipeline.enable_discovery:
        subprocess.run([sys.executable, '-m', 'pipeline.cli', 'discover',
                        '--data', train_glob, '--out', str(manifest_path)], **kw)
        time.sleep(0.2)
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(manifest_path) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'version': '1.0', 'feature_names': [], 'selection_seed': config.preprocessing.seed,
                       'selection_date': 'placeholder', 'discovery_status': 'disabled'}, f)
        os.replace(tmp, str(manifest_path))
    per_symbol = {}
    for f in test_files:
        symbol = f.parent.name
        if not _validate_symbol_data(f, config):
            continue
        out_dir = Path('output') / symbol / f'{f.stem}_split_{split_idx}'
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-m', 'pipeline.cli', 'run-hmm', '--data', str(f),
               '--manifest', str(manifest_path), '--out', str(out_dir)]
        if test_start and test_end:
            cmd.extend(['--start', test_start.isoformat(), '--end', test_end.isoformat()])
        try:
            subprocess.run(cmd, **kw)
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr.decode(errors='replace') if isinstance(e.stderr, bytes) else str(e.stderr or '')
            err_lines = stderr_text.strip().split('\n')
            print(f'[WARNING] HMM failed on split {split_idx} ({symbol}). Last 3 lines of stderr:')
            for line in err_lines[-3:]:
                print(f'  > {line}')
            logger.warning(
                'run-hmm failed for %s split=%d (rc=%d)',
                symbol, split_idx, e.returncode,
            )
            # Safe fallback: retry with plain run (no HMM)
            logger.info('Falling back to non-HMM run for %s split=%d', symbol, split_idx)
            cmd_fb = [sys.executable, '-m', 'pipeline.cli', 'run', '--data', str(f),
                       '--manifest', str(manifest_path), '--out', str(out_dir)]
            if test_start and test_end:
                cmd_fb.extend(['--start', test_start.isoformat(), '--end', test_end.isoformat()])
            try:
                subprocess.run(cmd_fb, **kw)
            except subprocess.CalledProcessError as e2:
                stderr2 = e2.stderr.decode(errors='replace') if isinstance(e2.stderr, bytes) else str(e2.stderr or '')
                err2_lines = stderr2.strip().split('\n')
                print(f'[ERROR] Fallback also failed on split {split_idx} ({symbol}). Last 3 lines:')
                for line in err2_lines[-3:]:
                    print(f'  > {line}')
                logger.error('Fallback run also failed for %s split=%d', symbol, split_idx)
        time.sleep(0.2)
        bt_path = out_dir / 'backtest_results_hmm.parquet'
        if not bt_path.exists():
            bt_path = out_dir / 'backtest_results.parquet'
        if bt_path.exists():
            try:
                bt = pl.read_parquet(bt_path)
                pnl = bt['pnl'].sum()
                sharpe = float(bt['pnl'].mean() / max(bt['pnl'].std(), 1e-9) * np.sqrt(252))
                hit = float((bt['pnl'] > 0).mean()) if 'pnl' in bt.columns else 0
                # HMM delta: compare with raw (non-HMM) baseline if available
                hmm_delta = 0.0
                raw_path = out_dir / 'backtest_results.parquet'
                if raw_path.exists() and raw_path != bt_path:
                    try:
                        raw = pl.read_parquet(raw_path)
                        raw_sharpe = float(raw['pnl'].mean() / max(raw['pnl'].std(), 1e-9) * np.sqrt(252))
                        hmm_delta = sharpe - raw_sharpe
                    except Exception:
                        hmm_delta = 0.0
                per_symbol[symbol] = (sharpe, pnl, hit, hmm_delta)
            except Exception:
                per_symbol[symbol] = (0.0, 0.0, 0.0, 0.0)
    # ---- Master dashboard ----
    _print_split_dashboard(split_idx, total_splits, per_symbol)
if __name__ == '__main__':
    config = load_config(os.environ.get('QUANT_ENV', 'alpha_1'))
    data_dir = 'data'
    files = get_files(data_dir, config)
    if not files:
        logger.warning('No files found after filtering')
        sys.exit(0)
    splits = generate_walkforward_splits(files, config)
    total = len(splits)
    if not _VERBOSE:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger('quant').setLevel(logging.WARNING)
    for i, split_data in enumerate(splits, 1):
        train, test = split_data[0], split_data[1]
        test_start = split_data[2] if len(split_data) > 2 else None
        test_end = split_data[3] if len(split_data) > 3 else None
        process_split(train, test, files, config, i, total, test_start, test_end)
    logging.getLogger().setLevel(logging.INFO)
    logger.info('Done')