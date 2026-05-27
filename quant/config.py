"""
config.py — Global configuration namespace.

Creates a shared SimpleNamespace that is populated at runtime by
``quant.utils.config_loader.load_config()`` which reads ``config.yaml``.

All other modules import ``config`` from here, ensuring a single
source of truth for every parameter.

Usage:
    from quant.config import config
    print(config.SEED)
    print(config.BOOTSTRAP_FOLDS)
"""

from types import SimpleNamespace

# The *only* line — all values are set by config_loader.load_config().
config = SimpleNamespace()