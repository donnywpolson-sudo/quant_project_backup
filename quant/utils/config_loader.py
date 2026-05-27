"""
config_loader.py
Single-source-of-truth configuration loader.

Reads config.yaml (project root) and applies relevant settings to the
quant.config.config SimpleNamespace, ensuring there is only one place
where runtime constants are defined.  Market-specific overrides from
config/markets/<SYMBOL>.yaml are applied by quant.market_config.

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


def load_config(config_path: str | Path = None) -> SimpleNamespace:
    """
    Read the project-level config.yaml and set constants on
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

    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / 'config.yaml'

    cfg = {}
    path = Path(config_path)
    if path.exists():
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        logger.info('Loaded project config from %s', path)
    else:
        logger.warning('config.yaml not found at %s — using defaults.', path)

    # ------------------------------------------------------------------
    # Market / universe
    # ------------------------------------------------------------------
    if 'markets' in cfg:
        config.MARKETS = cfg['markets']
        if not hasattr(config, 'MARKET_CONFIGS') or not config.MARKET_CONFIGS:
            config.MARKET_CONFIGS = {
                m: f'config/markets/{m}.yaml' for m in cfg['markets']
            }

    # ------------------------------------------------------------------
    # Walk-forward windows (year-level)
    # ------------------------------------------------------------------
    if 'training_years' in cfg:
        config.WF_TRAIN_DAYS_YEARLY = cfg['training_years']  # kept distinct
    if 'walkforward_years' in cfg:
        config.WF_TEST_DAYS_YEARLY = cfg['walkforward_years']

    # ------------------------------------------------------------------
    # Correlation filter for multi-market
    # ------------------------------------------------------------------
    if 'use_correlation_filter' in cfg:
        config.USE_CORRELATION_FILTER = cfg['use_correlation_filter']
    if 'correlation_threshold' in cfg:
        config.CORRELATION_THRESHOLD = cfg['correlation_threshold']

    # ------------------------------------------------------------------
    # Data range
    # ------------------------------------------------------------------
    if 'data_start_year' in cfg:
        config.DATA_START_YEAR = cfg['data_start_year']
    if 'data_end_year' in cfg:
        config.DATA_END_YEAR = cfg['data_end_year']
    if 'start_year' in cfg:
        config.START_YEAR = cfg['start_year']
    if 'end_year' in cfg:
        config.END_YEAR = cfg['end_year']

    # ------------------------------------------------------------------
    # Performance / I/O
    # ------------------------------------------------------------------
    if 'max_files' in cfg:
        config.MAX_FILES = cfg['max_files']
    if 'skip_completed' in cfg:
        config.SKIP_COMPLETED = cfg['skip_completed']
    if 'rolling' in cfg:
        config.ROLLING_WF = cfg['rolling']

    # ------------------------------------------------------------------
    # Session times (may be given as 'HH:MM' strings)
    # ------------------------------------------------------------------
    for key, attr in [
        ('session_start_local', 'SESSION_START_LOCAL'),
        ('session_end_local', 'SESSION_END_LOCAL'),
        ('session_break_start_local', 'SESSION_BREAK_START_LOCAL'),
        ('session_break_end_local', 'SESSION_BREAK_END_LOCAL'),
    ]:
        val = cfg.get(key)
        if val is not None:
            try:
                h, m = map(int, str(val).split(':'))
                setattr(config, attr, time(h, m))
            except (ValueError, TypeError):
                logger.warning('Could not parse %s = %r', key, val)

    _LOADED = True
    return config