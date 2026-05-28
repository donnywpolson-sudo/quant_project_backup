"""Hierarchical config pipeline: base + tier deep-merge, pydantic-validated."""
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# pydantic schema — all fields optional in base; tier-specific enforcement
# ---------------------------------------------------------------------------
class DiscoveryConfig(BaseModel):
    discovery_window_days: int = 30
    bootstrap_folds: int = 30
    selection_freq_threshold: float = 0.75
    sign_consistency_threshold: float = 0.8
    cumulative_importance_threshold: float = 0.95
    min_selected_features: int = 5
    max_selected_features: int = 50

class ExecutionConfig(BaseModel):
    slippage_k: float = 0.0005
    vol_penalty: float = 0.005
    tx_cost_per_roundturn: float = 0.00015
    commission_per_trade: float = 2e-05
    max_leverage: float = 3.0
    htf_trend_alignment: bool = True
    htf_vol_scaling: bool = True
    max_position_size: str | None = None
    daily_loss_limit: str | None = None

class SessionConfig(BaseModel):
    timezone: str = "America/New_York"
    session_start_local: str = "18:00"
    session_end_local: str = "16:00"
    session_break_start_local: str = "17:00"
    session_break_end_local: str = "18:00"

class FeaturesConfig(BaseModel):
    resample_frequencies: list[str] = ["5m"]
    drop_incomplete_rows: bool = True
    roll_windows: list[int] = [5, 10, 20]
    roll_window_min_rows: int = 20
    feature_transforms: list[str] = ["lags", "ratios", "z_scores"]
    max_pairwise_interactions: int = 50
    max_cross_timeframe_interactions: int = 0
    htf_alignment_filter: bool = False
    vol_median_window: int = 20
    vol_smooth_window: int = 5
    regime_high_thresh: float = 0.6
    regime_low_thresh: float = 0.4
    regime_missing_default: float = 0.0

class TargetConfig(BaseModel):
    target_5m_horizon: int = 1
    target_scale_factor: float = 100.0

class WalkforwardConfig(BaseModel):
    wf_train_days: int = 60
    wf_test_days: int = 1
    wf_step_days: int = 1

class PreprocessingConfig(BaseModel):
    clip_min: float = -10.0
    clip_max: float = 10.0
    eps: float = 1e-09
    replace_inf_nan_with: float = 0.0
    remove_prediction_bias: bool = False
    seed: int = 42

class IOConfig(BaseModel):
    row_group_size: int = 65536
    max_files: int = 20
    skip_completed: bool = True

class PipelineConfig(BaseModel):
    enable_discovery: bool = True
    enable_expansion: bool = True

class RootConfig(BaseModel):
    symbols: list[str] = ["ES"]
    time_zone: str = "America/New_York"
    log_level: str = "INFO"
    data_years: int = 1
    folds: int = 1
    start_year: int = 2024
    end_year: int = 2024
    session: SessionConfig = Field(default_factory=SessionConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    walkforward: WalkforwardConfig = Field(default_factory=WalkforwardConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    io: IOConfig = Field(default_factory=IOConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


# ---------------------------------------------------------------------------
# deep merge — nested dicts merged recursively, lists/scalars overwritten
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ---------------------------------------------------------------------------
# env-var interpolation — replaces ${VAR} with os.environ[VAR]
# ---------------------------------------------------------------------------
_ENV_RE = re.compile(r"\$\{([^}]+)\}")

def _resolve_env_vars(obj: Any) -> Any:
    if isinstance(obj, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
_CONFIGS_DIR = Path(__file__).resolve().parent

def load_config(env: str, configs_dir: Path | None = None) -> RootConfig:
    """
    Load configs/base.yaml, deep-merge configs/{env}.yaml,
    resolve ${ENV_VAR} placeholders, and validate with pydantic.

    Args:
        env: tier name (e.g. "alpha_1", "alpha_2", "production")
        configs_dir: override configs directory (default: project-root/configs/)
    """
    base_path = (configs_dir or _CONFIGS_DIR) / "alpha_base.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with open(base_path) as f:
        merged = yaml.safe_load(f) or {}

    tier_name = "alpha_production" if env == "production" else env
    tier_path = (configs_dir or _CONFIGS_DIR) / f"{tier_name}.yaml"
    if tier_path.exists():
        with open(tier_path) as f:
            tier_cfg = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, tier_cfg)

    merged = _resolve_env_vars(merged)

    try:
        return RootConfig(**merged)
    except ValidationError as e:
        raise ValueError(f"Config validation failed for env '{env}': {e}") from e


def load_env_config() -> RootConfig:
    """Convenience: reads ENV tier from QUANT_ENV environment variable."""
    env = os.environ.get("QUANT_ENV", "alpha_1")
    return load_config(env)