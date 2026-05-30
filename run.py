import os
import sys

# ── UTF-8 everywhere (Windows hardening) ──
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import subprocess
import logging
import time
import json
import hashlib
import threading
from pathlib import Path
import shutil
import polars as pl
import numpy as np
from core.config import load_config, RootConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    encoding='utf-8',
    force=True,
)
logger = logging.getLogger('QuantRunner')
_VERBOSE = os.environ.get('QUANT_VERBOSE', '0') == '1'
_MIN_TRAIN_DAYS = 90
_RUN_START = time.time()
_RUN_ID = hashlib.sha256(str(_RUN_START).encode()).hexdigest()[:8]
_PER_SYMBOL_PNL_CS = {}  # {symbol: {split_id: pnl_cs}}
_VERIFICATION_TABLE = []  # rows for the verification table printed per split

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
            # is enforced inside cli.py via --start/--end date boundaries
            # which slice by actual timestamps, not file boundaries.
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
            # cli.py enforces strict temporal separation by passing
            # --start/--end date boundaries, slicing within each file at
            # the timestamp level.
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
                               train_start, train_end, test_start, test_end))
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


def _log_subprocess_failure(cmd: list, returncode: int, stderr_text: str, stdout_text: str,
                            log_dir: Path, symbol: str, split_idx: int, stage: str) -> None:
    """Persist full subprocess failure output to a deterministic log file."""
    import datetime as dt
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
    log_path = log_dir / f'fail_{symbol}_split{split_idx}_{stage}_{timestamp}.log'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f'command: {" ".join(cmd)}\n')
        f.write(f'returncode: {returncode}\n')
        f.write(f'symbol: {symbol}\n')
        f.write(f'split_idx: {split_idx}\n')
        f.write(f'stage: {stage}\n')
        f.write(f'timestamp: {timestamp}\n')
        f.write(f'--- STDOUT ---\n{stdout_text}\n--- STDERR ---\n{stderr_text}\n')


def _run_subprocess_streaming(cmd: list, env: dict, timeout_idle: int = 120) -> tuple:
    """Run subprocess with real-time stdout/stderr streaming and idle timeout.

    Returns (returncode, stdout_text, stderr_text).
    On idle timeout (no output for *timeout_idle* seconds): terminates child,
    prints diagnostics, and returns (-1, stdout, stderr).
    """
    env_out = env.copy()
    env_out['PYTHONUNBUFFERED'] = '1'
    full_cmd = [sys.executable, '-u'] + cmd[1:] if cmd[0] == sys.executable else cmd
    proc = subprocess.Popen(
        full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=env_out,
    )
    stdout_lines = []
    stderr_lines = []
    last_output = time.time()
    lock = threading.Lock()
    done = threading.Event()

    def _read_stream(stream, lines_list, prefix):
        nonlocal last_output
        try:
            for line in iter(stream.readline, ''):
                line = line.rstrip('\n\r')
                with lock:
                    lines_list.append(line)
                    last_output = time.time()
                print(f'[{prefix}] {line}', flush=True)
        except Exception:
            pass

    t_stdout = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_lines, 'CHILD-OUT'), daemon=True)
    t_stderr = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_lines, 'CHILD-ERR'), daemon=True)
    t_stdout.start()
    t_stderr.start()

    # Idle timeout watchdog
    def _watchdog():
        while not done.is_set():
            time.sleep(1)
            with lock:
                idle = time.time() - last_output
            if idle > timeout_idle and not done.is_set():
                print(f'\n[TIMEOUT] No output for {timeout_idle}s — terminating child', flush=True)
                print(f'[TIMEOUT] cmd={" ".join(full_cmd)}', flush=True)
                print(f'[TIMEOUT] cwd={os.getcwd()}', flush=True)
                print(f'[TIMEOUT] env={os.environ.get("QUANT_ENV", "default")}', flush=True)
                with lock:
                    if stdout_lines:
                        print(f'[TIMEOUT] last stdout lines: {stdout_lines[-5:]}', flush=True)
                    if stderr_lines:
                        print(f'[TIMEOUT] last stderr lines: {stderr_lines[-5:]}', flush=True)
                proc.kill()
                done.set()
                break

    t_watch = threading.Thread(target=_watchdog, daemon=True)
    t_watch.start()

    try:
        ret = proc.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        proc.kill()
        ret = -1
    done.set()
    t_stdout.join(timeout=5)
    t_stderr.join(timeout=5)
    t_watch.join(timeout=5)
    stdout_text = '\n'.join(stdout_lines)
    stderr_text = '\n'.join(stderr_lines)
    if ret == -1:
        raise subprocess.CalledProcessError(-1, full_cmd, stdout_text, stderr_text)
    if ret != 0:
        raise subprocess.CalledProcessError(ret, full_cmd, stdout_text, stderr_text)
    return ret, stdout_text, stderr_text


_log_failures = os.environ.get('QUANT_LOG_FAILURES', '1') == '1'

def _print_verification_table():
    if not _VERIFICATION_TABLE:
        return
    print('\n' + '=' * 140)
    print(' VERIFICATION TABLE — each row must have unique path, ts_min, pnl_cs')
    print('=' * 140)
    print(f'{"split":>5} {"sym":>4} {"path":>55} {"mtime":>12} {"run_id":>8} {"rows":>6} {"ts_min":>26} {"ts_max":>26} {"pnl_cs":>8} {"sharpe":>7} {"pnl":>12} {"trades":>6}')
    print('-' * 140)
    for row in _VERIFICATION_TABLE:
        path_short = row['path'][-55:] if len(row['path']) > 55 else row['path']
        print(f'{row["split"]:>5} {row["symbol"]:>4} {path_short:>55} {row["mtime"]:>12.0f} {row["run_id"]:>8} {row["rows"]:>6} {str(row["ts_min"]):>26} {str(row["ts_max"]):>26} {row["pnl_cs"]:>8} {row["sharpe"]:>+7.3f} {row["pnl"]:>+12.2f} {row["trades"]:>6}')
    print('=' * 140)
    # Verify: no duplicate pnl_cs per symbol
    seen = {}
    for row in _VERIFICATION_TABLE:
        key = (row['symbol'], row['pnl_cs'])
        if row['pnl_cs'] not in ('missing', 'all_nan'):
            if key in seen:
                print(f'\n*** IDENTICAL PNL OUTPUT DETECTED: {row["symbol"]} split={row["split"]} matches split={seen[key]["split"]} pnl_cs={row["pnl_cs"]} ***\n', flush=True)
            seen[key] = row
    # Verify: no duplicate paths
    paths_seen = set()
    for row in _VERIFICATION_TABLE:
        if row['path'] in paths_seen:
            print(f'\n*** DUPLICATE RESULT PATH: {row["path"]} split={row["split"]} ***\n', flush=True)
        paths_seen.add(row['path'])
    print(flush=True)


def _print_split_dashboard(split_idx: int, total: int, per_symbol: dict):
    _print_verification_table()
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


def _hash_column(df: pl.DataFrame, col: str) -> str:
    if col not in df.columns:
        return 'missing'
    vals = df[col].to_numpy()
    mask = np.isfinite(vals)
    if mask.sum() == 0:
        return 'all_nan'
    h = hashlib.sha256(vals[mask].tobytes()).hexdigest()[:8]
    return h


def _print_symbol_diagnostics(bt: pl.DataFrame, symbol: str, split_idx: int) -> None:
    probs = bt['prediction_prob'].to_numpy().astype(np.float64) if 'prediction_prob' in bt.columns else None
    pmean = float(probs.mean()) if probs is not None else float('nan')
    pstd = float(probs.std()) if probs is not None else float('nan')
    gt055 = float((probs > 0.55).mean()) if probs is not None else float('nan')
    lt045 = float((probs < 0.45).mean()) if probs is not None else float('nan')
    bar_sqrt = np.sqrt(252)
    gross_sharpe = 'missing'
    net_sharpe = 'missing'
    cost_drag = 'missing'
    if 'gross_pnl' in bt.columns:
        gp = bt['gross_pnl'].to_numpy().astype(np.float64)
        if gp.std() > 1e-12:
            gross_sharpe = f'{float(gp.mean() / gp.std() * bar_sqrt):.3f}'
    if 'pnl' in bt.columns:
        np_ = bt['pnl'].to_numpy().astype(np.float64)
        if np_.std() > 1e-12:
            net_sharpe = f'{float(np_.mean() / np_.std() * bar_sqrt):.3f}'
        if 'gross_pnl' in bt.columns:
            cost_drag = f'{float(np_.sum() - gp.sum()):+.2f}'
    turnover = 'missing'
    if 'pos_change' in bt.columns:
        turnover = f'{float(bt["pos_change"].sum()):.1f}'
    trades = 'missing'
    if 'position' in bt.columns:
        pos = bt['position'].to_numpy().astype(np.float64)
        shifts = np.abs(np.diff(pos, prepend=pos[0]))
        trades = f'{int(np.sum(shifts > 1e-9))}'
    pred_cs = _hash_column(bt, 'prediction_prob')
    pnl_cs = _hash_column(bt, 'pnl')
    sig_cs = _hash_column(bt, 'raw_signal') if 'raw_signal' in bt.columns else _hash_column(bt, 'target_exec')
    logger.info(
        '[DIAG] symbol=%s split=%d prob_mean=%.4f prob_std=%.4f gt055=%.3f lt045=%.3f '
        'gross_sharpe=%s net_sharpe=%s cost_drag=%s turnover=%s trades=%s '
        'pred_cs=%s pnl_cs=%s sig_cs=%s',
        symbol, split_idx, pmean, pstd, gt055, lt045,
        gross_sharpe, net_sharpe, cost_drag, turnover, trades,
        pred_cs, pnl_cs, sig_cs,
    )


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
                   train_start=None, train_end=None, test_start=None, test_end=None) -> None:
    train_files = [f for f in files if int(f.stem) in train_years]
    test_files = [f for f in files if int(f.stem) in test_years]
    print(f'[SPLIT {split_idx}/{total_splits}] test_window=[{test_start}, {test_end}) run_id={_RUN_ID}',
          f'cwd={os.getcwd()} run_py_mtime={Path("run.py").stat().st_mtime:.0f}',
          f'cli_py_mtime={Path("pipeline/cli.py").stat().st_mtime:.0f}',
          flush=True)
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
    env = os.environ.copy()
    env['TQDM_DISABLE'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    if config.pipeline.enable_discovery:
        _run_subprocess_streaming([sys.executable, '-m', 'pipeline.cli', 'discover',
                        '--data', train_glob, '--out', str(manifest_path)], env)
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
        # Purge stale backtest files from previous runs
        for _stale_name in ('backtest_results_hmm.parquet', 'backtest_results.parquet'):
            _stale_path = out_dir / _stale_name
            if _stale_path.exists() and _stale_path.stat().st_mtime < _RUN_START:
                logger.warning('[STALE] Removing pre-existing file: %s (mtime=%.0f < run_start=%.0f)',
                               _stale_path, _stale_path.stat().st_mtime, _RUN_START)
                _stale_path.unlink()
        cmd = [sys.executable, '-m', 'pipeline.cli', 'run-hmm', '--data', str(f),
               '--manifest', str(manifest_path), '--out', str(out_dir)]
        if train_start and train_end:
            cmd.extend(['--train-start', train_start.isoformat(), '--train-end', train_end.isoformat()])
        if test_start and test_end:
            cmd.extend(['--start', test_start.isoformat(), '--end', test_end.isoformat()])
        wf_mode = getattr(config, 'WF_MODE', '')
        if wf_mode == 'outer_split':
            assert train_start and train_end, 'outer_split mode but train_start/train_end not set'
            assert '--train-start' in cmd and '--train-end' in cmd, '--train-start/--train-end missing from subprocess command'
        print(f'[SUBPROCESS-CMD] {" ".join(cmd)}', flush=True)
        logger.info('[SUBPROCESS] split=%d symbol=%s start=%s end=%s cmd=%s',
                    split_idx, symbol,
                    test_start.isoformat() if test_start else 'None',
                    test_end.isoformat() if test_end else 'None',
                    ' '.join(cmd))
        try:
            _run_subprocess_streaming(cmd, env)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            stderr_text = e.stderr if hasattr(e, 'stderr') else ''
            stdout_text = e.stdout if hasattr(e, 'stdout') else ''
            if isinstance(stderr_text, bytes):
                stderr_text = stderr_text.decode(errors='replace')
            if isinstance(stdout_text, bytes):
                stdout_text = stdout_text.decode(errors='replace')
            err_lines = stderr_text.strip().split('\n') if stderr_text else ['(no stderr)']
            print(f'[WARNING] HMM failed on split {split_idx} ({symbol}). Last 3 lines of stderr:')
            for line in err_lines[-3:]:
                print(f'  > {line}')
            if _log_failures:
                _log_subprocess_failure(cmd, getattr(e, 'returncode', -1), stderr_text, stdout_text,
                                        Path('output') / 'logs', symbol, split_idx, 'hmm')
            logger.warning(
                'run-hmm failed for %s split=%d (rc=%d)',
                symbol, split_idx, getattr(e, 'returncode', -1),
            )
            # Safe fallback: retry with plain run (no HMM)
            logger.info('Falling back to non-HMM run for %s split=%d', symbol, split_idx)
            cmd_fb = [sys.executable, '-m', 'pipeline.cli', 'run', '--data', str(f),
                       '--manifest', str(manifest_path), '--out', str(out_dir)]
            if train_start and train_end:
                cmd_fb.extend(['--train-start', train_start.isoformat(), '--train-end', train_end.isoformat()])
            if test_start and test_end:
                cmd_fb.extend(['--start', test_start.isoformat(), '--end', test_end.isoformat()])
            try:
                _run_subprocess_streaming(cmd_fb, env)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e2:
                stderr2 = e2.stderr if hasattr(e2, 'stderr') else ''
                stdout2 = e2.stdout if hasattr(e2, 'stdout') else ''
                if isinstance(stderr2, bytes):
                    stderr2 = stderr2.decode(errors='replace')
                if isinstance(stdout2, bytes):
                    stdout2 = stdout2.decode(errors='replace')
                err2_lines = stderr2.strip().split('\n') if stderr2 else ['(no stderr)']
                print(f'[ERROR] Fallback also failed on split {split_idx} ({symbol}). Last 3 lines:')
                for line in err2_lines[-3:]:
                    print(f'  > {line}')
                if _log_failures:
                    _log_subprocess_failure(cmd_fb, getattr(e2, 'returncode', -1), stderr2, stdout2,
                                            Path('output') / 'logs', symbol, split_idx, 'fallback')
                logger.error('Fallback run also failed for %s split=%d', symbol, split_idx)
        time.sleep(0.2)
        bt_path = out_dir / 'backtest_results_hmm.parquet'
        if not bt_path.exists():
            bt_path = out_dir / 'backtest_results.parquet'
        if bt_path.exists():
            _bt_mtime = bt_path.stat().st_mtime
            assert _bt_mtime >= _RUN_START, (
                f'STALE RESULT: {bt_path} mtime={_bt_mtime:.0f} < run_start={_RUN_START:.0f}. '
                f'Subprocess may have failed silently and an old file was read.'
            )
            print(f'[BACKTEST] {symbol} path={bt_path} mtime={_bt_mtime:.0f} run_id={_RUN_ID}', flush=True)
            try:
                bt = pl.read_parquet(bt_path)
                t_min = str(bt['ts_event'].min()) if 'ts_event' in bt.columns else 'missing'
                t_max = str(bt['ts_event'].max()) if 'ts_event' in bt.columns else 'missing'
                print(f'[BACKTEST] {symbol} rows={bt.height} ts_min={t_min} ts_max={t_max}', flush=True)
                # Assert timestamps are within the test window
                if test_start and test_end and 'ts_event' in bt.columns:
                    from datetime import datetime as _dt, timezone
                    ts_start = _dt.fromisoformat(test_start.isoformat()).replace(tzinfo=timezone.utc)
                    ts_end = _dt.fromisoformat(test_end.isoformat()).replace(tzinfo=timezone.utc)
                    bt_t_min = bt['ts_event'].min()
                    bt_t_max = bt['ts_event'].max()
                    assert bt_t_min >= ts_start, (
                        f'BOUNDARY VIOLATION: {symbol} split={split_idx} bt_t_min={bt_t_min} < test_start={ts_start}'
                    )
                    assert bt_t_max < ts_end, (
                        f'BOUNDARY VIOLATION: {symbol} split={split_idx} bt_t_max={bt_t_max} >= test_end={ts_end}'
                    )
                pnl_cs = _hash_column(bt, 'pnl')
                pred_cs = _hash_column(bt, 'prediction_prob')
                pnl = bt['pnl'].sum()
                sharpe = float(bt['pnl'].mean() / max(bt['pnl'].std(), 1e-9) * np.sqrt(252))
                hit = float((bt['pnl'] > 0).mean()) if 'pnl' in bt.columns else 0
                print(f'[VERIFY] {symbol} split={split_idx} pnl_cs={pnl_cs} pred_cs={pred_cs} sharpe={sharpe:.3f} pnl={pnl:.2f}', flush=True)
                if symbol not in _PER_SYMBOL_PNL_CS:
                    _PER_SYMBOL_PNL_CS[symbol] = {}
                if pnl_cs != 'missing' and pnl_cs != 'all_nan':
                    for prior_split, prior_cs in _PER_SYMBOL_PNL_CS[symbol].items():
                        assert prior_cs != pnl_cs, (
                            f'IDENTICAL PNL OUTPUT: {symbol} split={split_idx} pnl_cs={pnl_cs} '
                            f'matches split={prior_split}. Stale cache or file reuse detected.'
                        )
                    _PER_SYMBOL_PNL_CS[symbol][split_idx] = pnl_cs
                # Collect for verification table
                trades = 0
                if 'position' in bt.columns:
                    pos = bt['position'].to_numpy().astype(np.float64)
                    trades = int(np.sum(np.abs(np.diff(pos, prepend=pos[0])) > 1e-9))
                _VERIFICATION_TABLE.append({
                    'split': split_idx, 'symbol': symbol, 'path': str(bt_path),
                    'mtime': _bt_mtime, 'run_id': _RUN_ID, 'rows': bt.height,
                    'ts_min': str(t_min), 'ts_max': str(t_max),
                    'pnl_cs': pnl_cs, 'pnl': pnl, 'sharpe': sharpe, 'trades': trades,
                })
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
                _print_symbol_diagnostics(bt, symbol, split_idx)
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
        train_start = split_data[2] if len(split_data) > 2 else None
        train_end   = split_data[3] if len(split_data) > 3 else None
        test_start  = split_data[4] if len(split_data) > 4 else None
        test_end    = split_data[5] if len(split_data) > 5 else None
        process_split(train, test, files, config, i, total, train_start, train_end, test_start, test_end)
    logging.getLogger().setLevel(logging.INFO)
    logger.info('Done')