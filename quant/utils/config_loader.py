"""
config_loader.py
Single-source-of-truth configuration loader.

Reads config.yaml (project root) and populates quant.config.config
SimpleNamespace with every runtime parameter.  Idempotent — subsequent
calls are no-ops.

Market-specific overrides from config/markets/<SYMBOL>.yaml are applied
by quant.market_config.

Usage:
    from quant.utils.config_loader import load_config
    load_config()  # reads config.yaml, updates quant.config.config
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from datetime import time

import yaml

from quant.config import config

logger = logging.getLogger(__name__)

_LOADED = False

# ---------------------------------------------------------------------------
# FALLBACK DEFAULTS — exact copies of the old quant/config.py hardcoded values.
# If a key is missing from config.yaml we use these so behaviour is unchanged.
# ---------------------------------------------------------------------------
_DEFAULTS = {
    # -- data paths ----------------------------------------------------------
    'DATA_GLOB': 'data/futures/*.parquet',
    'MANIFEST_PATH': 'artifacts/manifest.json',
    'BASELINE_FEATURES_FILE': 'config/baseline_features.yaml',
    'BASELINE_FEATURES_PERSIST_PATH': 'artifacts/baseline_feature_matrix.parquet',
    'TRADES_OUT': 'artifacts/trades.csv',
    'LOG_DIR': 'logs/',
    # -- memory --------------------------------------------------------------
    'RAM_CAP_BYTES': 14 * 1024 ** 3,
    'RSS_STOP_BYTES': int(13.5 * 1024 ** 3),
    'ROWS_PER_CHUNK_MAX': 5000000,
    'MEMORY_SAFETY_MARGIN': 0.95,
    'MEMORY_LOG_ENABLED': True,
    # -- session -------------------------------------------------------------
    'TIMEZONE': 'America/New_York',
    'SESSION_START_LOCAL': time(18, 0),
    'SESSION_END_LOCAL': time(16, 0),
    'SESSION_BREAK_START_LOCAL': time(17, 0),
    'SESSION_BREAK_END_LOCAL': time(18, 0),
    # -- features ------------------------------------------------------------
    'RESAMPLE_FREQUENCIES': ['5m', '1h', '1d'],
    'DROP_INCOMPLETE_ROWS': True,
    'ROLL_WINDOWS': [5, 10, 20, 50],
    'ROLL_WINDOWS_1H': [2, 4, 6, 12],
    'ROLL_WINDOWS_DAILY': [5, 10, 20],
    'ROLL_WINDOW_MIN_ROWS': 20,
    'FEATURE_TRANSFORMS': ['lags', 'ratios', 'z_scores',
                           'pairwise_products_limited', 'cross_timeframe_ratios'],
    'MAX_PAIRWISE_INTERACTIONS': 500,
    'MAX_CROSS_TIMEFRAME_INTERACTIONS': 200,
    'HTF_TREND_WINDOWS': [5, 10, 20],
    'HTF_VOLATILITY_WINDOWS': [5, 10, 20],
    'HTF_ALIGNMENT_FILTER': True,
    'HTF_TREND_THRESHOLD': 0.1,
    'VOL_MEDIAN_WINDOW': 20,
    'VOL_SMOOTH_WINDOW': 5,
    'REGIME_HIGH_THRESH': 0.6,
    'REGIME_LOW_THRESH': 0.4,
    'REGIME_MISSING_DEFAULT': 0.0,
    # -- target --------------------------------------------------------------
    'TARGET_5M_HORIZON': 1,
    'TARGET_SCALE_FACTOR': 100.0,
    # -- discovery -----------------------------------------------------------
    'DISCOVERY_WINDOW_DAYS': 60,
    'BOOTSTRAP_FOLDS': 30,
    'EXTRA_TREES_PARAMS': {
        'random_state': 42, 'n_jobs': 1, 'n_estimators': 100,
        'max_depth': 8, 'max_features': 0.3, 'bootstrap': False,
    },
    'SELECTION_FREQ_THRESHOLD': 0.75,
    'SIGN_CONSISTENCY_THRESHOLD': 0.8,
    'CUMULATIVE_IMPORTANCE_THRESHOLD': 0.95,
    'MIN_SELECTED_FEATURES': 10,
    'MAX_SELECTED_FEATURES': 1000,
    # -- walkforward ---------------------------------------------------------
    'WF_TRAIN_DAYS': 60,
    'WF_TEST_DAYS': 1,
    'WF_STEP_DAYS': 1,
    'RIDGE_PARAMS': {
        'alpha': 1.0, 'solver': 'cholesky',
        'fit_intercept': True, 'random_state': 42,
    },
    'MODEL_TYPE': 'Ridge',
    'PROBABILITY_SMOOTHING_ALPHA': 0.1,
    'CORR_THRESHOLD': 0.95,
    'WF_PARALLEL_FOLDS': 1,
    # -- execution -----------------------------------------------------------
    'EXECUTE_AT': 'open[t+1]',
    'SLIPPAGE_K': 0.001,
    'VOL_PENALTY': 0.005,
    'COMMISSION_PER_TRADE': 2e-05,
    'TARGET_VOL': 0.01,
    'MAX_LEVERAGE': 3.0,
    'MAX_POS_CHANGE_PER_MIN': 0.1,
    'FLAT_BEFORE_CLOSE_MINUTES': 5,
    'HTF_TREND_ALIGNMENT': True,
    'HTF_VOL_SCALING': True,
    'HTF_VOL_WINDOW': 10,
    # -- preprocessing -------------------------------------------------------
    'CLIP_MIN': -10.0,
    'CLIP_MAX': 10.0,
    'EPS': 1e-09,
    'REPLACE_INF_NAN_WITH': 0.0,
    'REMOVE_PREDICTION_BIAS': False,
    'SEED': 42,
    # -- io ------------------------------------------------------------------
    'ROW_GROUP_SIZE': 65536,
    # -- legacy top-level keys (kept for backwards compat) -------------------
    'MARKETS': ['ES'],
    'MARKET_CONFIGS': {
        'ES': 'config/markets/ES.yaml',
        'CL': 'config/markets/CL.yaml',
        'ZB': 'config/markets/ZB.yaml',
    },
    'USE_CORRELATION_FILTER': False,
    'CORRELATION_THRESHOLD': 0.75,
    'MAX_FILES': 20,
    'SKIP_COMPLETED': True,
    'ROLLING_WF': True,
    'DATA_START_YEAR': 2010,
    'DATA_END_YEAR': 2026,
    'START_YEAR': 2022,
    'END_YEAR': 2023,
    'WF_TRAIN_DAYS_YEARLY': 1,
    'WF_TEST_DAYS_YEARLY': 1,
}


# ---- helpers ---------------------------------------------------------------

def _parse_time(raw: str) -> time | None:
    """Parse 'HH:MM' → datetime.time, or None on failure."""
    try:
        h, m = map(int, str(raw).split(':'))
        return time(h, m)
    except (ValueError, TypeError):
        logger.warning('Could not parse time string %r', raw)
        return None


def _apply_flat(cfg: dict, key: str, attr: str):
    """If *key* exists in *cfg*, set config.*attr*."""
    if key in cfg:
        setattr(config, attr, cfg[key])


def _apply_nested(cfg: dict, section: str, mapping: dict[str, str]):
    """For each (yaml_key → config_attr) in *mapping*, read from
    cfg[section] and set config.*config_attr*."""
    sec = cfg.get(section)
    if not isinstance(sec, dict):
        return
    for yaml_key, attr_name in mapping.items():
        if yaml_key in sec:
            setattr(config, attr_name, sec[yaml_key])


# ---- main loader -----------------------------------------------------------

def load_config(config_path: str | Path = None) -> SimpleNamespace:
    """
    Read the project-level config.yaml and set every constant on
    ``quant.config.config``.  Idempotent: subsequent calls are no-ops.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to config.yaml.  Defaults to <project-root>/config.yaml.

    Returns
    -------
    quant.config.config (SimpleNamespace)
    """
    global _LOADED
    if _LOADED:
        return config

    # ---- 1. load raw YAML --------------------------------------------------
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / 'config.yaml'

    cfg: dict = {}
    path = Path(config_path)
    if path.exists():
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        logger.info('Loaded project config from %s', path)
    else:
        logger.warning('config.yaml not found at %s — using defaults.', path)

    # ---- 2. apply fallback defaults first ----------------------------------
    for attr, value in _DEFAULTS.items():
        setattr(config, attr, value)

    # ---- 3. overlay YAML values section by section -------------------------

    # -- data ----------------------------------------------------------------
    _apply_nested(cfg, 'data', {
        'data_glob': 'DATA_GLOB',
        'manifest_path': 'MANIFEST_PATH',
        'baseline_features_file': 'BASELINE_FEATURES_FILE',
        'baseline_features_persist_path': 'BASELINE_FEATURES_PERSIST_PATH',
        'trades_out': 'TRADES_OUT',
        'log_dir': 'LOG_DIR',
    })

    # -- memory --------------------------------------------------------------
    _apply_nested(cfg, 'memory', {
        'ram_cap_bytes': 'RAM_CAP_BYTES',
        'rss_stop_bytes': 'RSS_STOP_BYTES',
        'rows_per_chunk_max': 'ROWS_PER_CHUNK_MAX',
        'memory_safety_margin': 'MEMORY_SAFETY_MARGIN',
        'memory_log_enabled': 'MEMORY_LOG_ENABLED',
    })

    # -- session (times are strings → parse) ---------------------------------
    sec_sess = cfg.get('session')
    if isinstance(sec_sess, dict):
        if 'timezone' in sec_sess:
            config.TIMEZONE = sec_sess['timezone']
        for yk, ak in [('session_start_local', 'SESSION_START_LOCAL'),
                       ('session_end_local', 'SESSION_END_LOCAL'),
                       ('session_break_start_local', 'SESSION_BREAK_START_LOCAL'),
                       ('session_break_end_local', 'SESSION_BREAK_END_LOCAL')]:
            val = sec_sess.get(yk)
            if val is not None:
                t = _parse_time(str(val))
                if t is not None:
                    setattr(config, ak, t)

    # -- features ------------------------------------------------------------
    _apply_nested(cfg, 'features', {
        'resample_frequencies': 'RESAMPLE_FREQUENCIES',
        'drop_incomplete_rows': 'DROP_INCOMPLETE_ROWS',
        'roll_windows': 'ROLL_WINDOWS',
        'roll_windows_1h': 'ROLL_WINDOWS_1H',
        'roll_windows_daily': 'ROLL_WINDOWS_DAILY',
        'roll_window_min_rows': 'ROLL_WINDOW_MIN_ROWS',
        'feature_transforms': 'FEATURE_TRANSFORMS',
        'max_pairwise_interactions': 'MAX_PAIRWISE_INTERACTIONS',
        'max_cross_timeframe_interactions': 'MAX_CROSS_TIMEFRAME_INTERACTIONS',
        'htf_trend_windows': 'HTF_TREND_WINDOWS',
        'htf_volatility_windows': 'HTF_VOLATILITY_WINDOWS',
        'htf_alignment_filter': 'HTF_ALIGNMENT_FILTER',
        'htf_trend_threshold': 'HTF_TREND_THRESHOLD',
        'vol_median_window': 'VOL_MEDIAN_WINDOW',
        'vol_smooth_window': 'VOL_SMOOTH_WINDOW',
        'regime_high_thresh': 'REGIME_HIGH_THRESH',
        'regime_low_thresh': 'REGIME_LOW_THRESH',
        'regime_missing_default': 'REGIME_MISSING_DEFAULT',
    })

    # -- target --------------------------------------------------------------
    _apply_nested(cfg, 'target', {
        'target_5m_horizon': 'TARGET_5M_HORIZON',
        'target_scale_factor': 'TARGET_SCALE_FACTOR',
    })

    # -- discovery -----------------------------------------------------------
    _apply_nested(cfg, 'discovery', {
        'discovery_window_days': 'DISCOVERY_WINDOW_DAYS',
        'bootstrap_folds': 'BOOTSTRAP_FOLDS',
    })
    if isinstance(cfg.get('discovery'), dict):
        d = cfg['discovery']
        if 'extra_trees_params' in d:
            config.EXTRA_TREES_PARAMS = dict(d['extra_trees_params'])
        if 'selection_freq_threshold' in d:
            config.SELECTION_FREQ_THRESHOLD = d['selection_freq_threshold']
        if 'sign_consistency_threshold' in d:
            config.SIGN_CONSISTENCY_THRESHOLD = d['sign_consistency_threshold']
        if 'cumulative_importance_threshold' in d:
            config.CUMULATIVE_IMPORTANCE_THRESHOLD = d['cumulative_importance_threshold']
        if 'min_selected_features' in d:
            config.MIN_SELECTED_FEATURES = d['min_selected_features']
        if 'max_selected_features' in d:
            config.MAX_SELECTED_FEATURES = d['max_selected_features']

    # -- walkforward ---------------------------------------------------------
    _apply_nested(cfg, 'walkforward', {
        'wf_train_days': 'WF_TRAIN_DAYS',
        'wf_test_days': 'WF_TEST_DAYS',
        'wf_step_days': 'WF_STEP_DAYS',
    })
    if isinstance(cfg.get('walkforward'), dict):
        w = cfg['walkforward']
        if 'ridge_params' in w:
            config.RIDGE_PARAMS = dict(w['ridge_params'])
        if 'model_type' in w:
            config.MODEL_TYPE = w['model_type']
        if 'probability_smoothing_alpha' in w:
            config.PROBABILITY_SMOOTHING_ALPHA = w['probability_smoothing_alpha']
        if 'corr_threshold' in w:
            config.CORR_THRESHOLD = w['corr_threshold']
        if 'wf_parallel_folds' in w:
            config.WF_PARALLEL_FOLDS = w['wf_parallel_folds']
        # legacy year-level keys inside walkforward section
        if 'training_years' in w:
            config.WF_TRAIN_DAYS_YEARLY = w['training_years']
        if 'walkforward_years' in w:
            config.WF_TEST_DAYS_YEARLY = w['walkforward_years']
        if 'rolling' in w:
            config.ROLLING_WF = w['rolling']

    # -- execution -----------------------------------------------------------
    _apply_nested(cfg, 'execution', {
        'execute_at': 'EXECUTE_AT',
        'slippage_k': 'SLIPPAGE_K',
        'vol_penalty': 'VOL_PENALTY',
        'commission_per_trade': 'COMMISSION_PER_TRADE',
        'target_vol': 'TARGET_VOL',
        'max_leverage': 'MAX_LEVERAGE',
        'max_pos_change_per_min': 'MAX_POS_CHANGE_PER_MIN',
        'flat_before_close_minutes': 'FLAT_BEFORE_CLOSE_MINUTES',
        'htf_trend_alignment': 'HTF_TREND_ALIGNMENT',
        'htf_vol_scaling': 'HTF_VOL_SCALING',
        'htf_vol_window': 'HTF_VOL_WINDOW',
    })

    # -- preprocessing -------------------------------------------------------
    _apply_nested(cfg, 'preprocessing', {
        'clip_min': 'CLIP_MIN',
        'clip_max': 'CLIP_MAX',
        'eps': 'EPS',
        'replace_inf_nan_with': 'REPLACE_INF_NAN_WITH',
        'remove_prediction_bias': 'REMOVE_PREDICTION_BIAS',
        'seed': 'SEED',
    })

    # -- io ------------------------------------------------------------------
    _apply_nested(cfg, 'io', {
        'row_group_size': 'ROW_GROUP_SIZE',
    })
    # legacy top-level keys inside io section
    if isinstance(cfg.get('io'), dict):
        io = cfg['io']
        if 'max_files' in io:
            config.MAX_FILES = io['max_files']
        if 'skip_completed' in io:
            config.SKIP_COMPLETED = io['skip_completed']

    # ---- 4. legacy top-level keys (backwards compat) -----------------------
    _apply_flat(cfg, 'markets', 'MARKETS')
    _apply_flat(cfg, 'use_correlation_filter', 'USE_CORRELATION_FILTER')
    _apply_flat(cfg, 'correlation_threshold', 'CORRELATION_THRESHOLD')
    _apply_flat(cfg, 'max_markets', 'MAX_MARKETS' if hasattr(config, 'MAX_MARKETS') or True else 'MAX_MARKETS')
    _apply_flat(cfg, 'max_files', 'MAX_FILES')
    _apply_flat(cfg, 'skip_completed', 'SKIP_COMPLETED')
    _apply_flat(cfg, 'data_start_year', 'DATA_START_YEAR')
    _apply_flat(cfg, 'data_end_year', 'DATA_END_YEAR')
    _apply_flat(cfg, 'start_year', 'START_YEAR')
    _apply_flat(cfg, 'end_year', 'END_YEAR')

    # training_years / walkforward_years / rolling at top-level (legacy)
    if 'training_years' in cfg and 'walkforward' not in cfg:
        config.WF_TRAIN_DAYS_YEARLY = cfg['training_years']
    if 'walkforward_years' in cfg and 'walkforward' not in cfg:
        config.WF_TEST_DAYS_YEARLY = cfg['walkforward_years']
    if 'rolling' in cfg and 'walkforward' not in cfg:
        config.ROLLING_WF = cfg['rolling']

    # Auto-build MARKET_CONFIGS from markets list if not explicitly set
    if 'markets' in cfg and isinstance(cfg['markets'], list):
        config.MARKET_CONFIGS = {
            m: f'config/markets/{m}.yaml' for m in cfg['markets']
        }

    _LOADED = True
    return config