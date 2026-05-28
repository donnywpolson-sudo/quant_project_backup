import os
import sys
import subprocess
import logging
from pathlib import Path
import shutil
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

def generate_walkforward_splits(files: list[Path], config: RootConfig) -> list[tuple[list[int], list[int]]]:
    train_years = config.data_years
    wf_years = config.folds
    years = sorted({int(f.stem) for f in files})

    if not years:
        logger.warning('No years available for walkforward splitting')
        return []

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

    # Fallback: insufficient years for a proper walkforward — use all
    # available years for training and the last year for testing.
    if not splits:
        train_range = years[:train_years] if len(years) >= train_years else years[:]
        test_range = [years[-1]] if years else []
        if not train_range:
            logger.warning('No train range could be formed — no splits generated')
            return []
        logger.warning(
            'Insufficient years for walkforward: need %d (train=%d + test=%d), '
            'have %d. Falling back to single split: train=%s, test=%s',
            train_years + wf_years, train_years, wf_years, len(years),
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
    train_dir = Path('artifacts') / f"train_{'_'.join(map(str, train_years))}"
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
    manifest_path = Path('artifacts') / f"manifest_{'_'.join(map(str, train_years))}.json"
    if config.pipeline.enable_discovery:
        logger.info('Running feature discovery on TRAIN data...')
        subprocess.run([sys.executable, '-m', 'quant.cli', 'discover', '--data', train_glob, '--out', str(manifest_path)], check=True)
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
        out_dir = Path('artifacts') / f.parent.name / f.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'quant.cli', 'run', '--data', str(f), '--manifest', str(manifest_path), '--out', str(out_dir)], check=True)
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