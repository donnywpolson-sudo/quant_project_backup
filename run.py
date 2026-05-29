import os
import sys
import subprocess
import logging
import time
from pathlib import Path
import shutil
import polars as pl
from quant.config_manager import load_config, RootConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('QuantRunner')

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
                splits.append((sorted(train_files), sorted(test_files)))
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

def process_split(train_years: list[int], test_years: list[int], files: list[Path], config: RootConfig) -> None:
    train_files = [f for f in files if int(f.stem) in train_years]
    test_files = [f for f in files if int(f.stem) in test_years]
    if not train_files or not test_files:
        logger.warning('Empty train/test split — skipping')
        return
    train_dir = Path('output') / f"train_{'_'.join(map(str, train_years))}"
    train_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Preparing TRAIN dataset for years {train_years}')
    for f in train_files:
        dst = train_dir / f'{f.parent.name}_{f.name}'
        if dst.exists():
            continue
        logger.info(f'Linking train file: {f}')
        if os.name != 'nt':
            try:
                os.symlink(f.resolve(), dst)
                continue
            except Exception:
                pass
        shutil.copy2(f, dst)
    train_glob = str(train_dir / '*.parquet')
    manifest_path = Path('output') / f"manifest_{'_'.join(map(str, train_years))}.json"
    if config.pipeline.enable_discovery:
        logger.info('Running feature discovery on TRAIN data...')
        subprocess.run([sys.executable, '-m', 'quant.cli', 'discover', '--data', train_glob, '--out', str(manifest_path)], check=True)
        time.sleep(0.2)  # NTFS lock release
    else:
        logger.info('Discovery disabled — generating placeholder manifest from baseline features.')
        import json
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        placeholder = {
            'version': '1.0',
            'feature_names': [],
            'selection_seed': config.preprocessing.seed,
            'selection_date': 'placeholder',
            'discovery_status': 'disabled',
        }
        with open(manifest_path, 'w') as f:
            json.dump(placeholder, f)
    for f in test_files:
        logger.info(f'Evaluating TEST file: {f}')
        out_dir = Path('output') / f.parent.name / f.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'quant.cli', 'run', '--data', str(f), '--manifest', str(manifest_path), '--out', str(out_dir)], check=True)
        time.sleep(0.2)  # NTFS lock release
if __name__ == '__main__':
    config = load_config(os.environ.get('QUANT_ENV', 'alpha_1'))
    data_dir = 'data'
    files = get_files(data_dir, config)
    if not files:
        logger.warning('No files found after filtering')
        sys.exit(0)
    splits = generate_walkforward_splits(files, config)
    for i, (train, test) in enumerate(splits, 1):
        logger.info(f'[Split {i}] Train({len(train)}): {train} | Test({len(test)}): {test}')
        process_split(train, test, files, config)
    logger.info('Done')