"""
src/features/expansion.py
Expand feature space with ratios, z-scores, regime-conditioned transforms,
pairwise interactions (capped at MAX_PAIRWLE_INTERACTIONS), cross-timeframe interactions,
and additional advanced features (quantiles, Fourier, moments, acceleration, VWAP, etc.).

All features are past-only, float32, clipped.

Now with memory safety estimation to avoid OOM from combinatorial explosion.
"""
import polars as pl
import numpy as np
import logging
from itertools import combinations
from config import config

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Existing functions (unchanged)
# ----------------------------------------------------------------------

def add_regime(df: pl.DataFrame) -> pl.DataFrame:
    """Add regime column: 1=high vol, 0=low vol, using rolling median volatility."""
    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    vol20 = ret.rolling_std(window_size=20)
    med_vol = vol20.rolling_median(window_size=config.VOL_MEDIAN_WINDOW)
    smooth_vol = med_vol.rolling_mean(window_size=config.VOL_SMOOTH_WINDOW)
    regime = pl.when(smooth_vol >= config.REGIME_HIGH_THRESH).then(1.0) \
              .when(smooth_vol <= config.REGIME_LOW_THRESH).then(0.0) \
              .otherwise(None)
    regime = regime.fill_null(strategy="forward").fill_null(config.REGIME_MISSING_DEFAULT)
    df = df.with_columns(regime.cast(pl.Float32).alias("regime"))
    return df

def add_ratios_and_z_scores(df: pl.DataFrame, base_features: list) -> pl.DataFrame:
    """Add ratio features (feature_i / feature_j) and z-scores (rolling z-score)."""
    core = ["close", "volume", "feature_spread_proxy", "feature_high_low_range_norm"]
    existing = [c for c in core if c in df.columns]
    exprs = []
    for i, a in enumerate(existing):
        for b in existing[i+1:]:
            name = f"ratio_{a}_over_{b}"
            expr = (pl.col(a) / pl.col(b).clip(config.EPS, None)).cast(pl.Float32)
            exprs.append(expr.clip(config.CLIP_MIN, config.CLIP_MAX).alias(name))
    for col in base_features[:20]:
        if col in df.columns:
            mean = pl.col(col).rolling_mean(window_size=20)
            std = pl.col(col).rolling_std(window_size=20)
            z = (pl.col(col) - mean) / std.clip(config.EPS, None)
            exprs.append(z.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f"{col}_zscore"))
    df = df.with_columns(exprs)
    return df

def add_regime_conditioned_transforms(df: pl.DataFrame) -> pl.DataFrame:
    """Multiply selected features by regime indicator."""
    regime = pl.col("regime")
    interact_cols = ["feature_ret_1", "feature_ret_5", "feature_ewma_vol_20", "feature_volume_z_20"]
    exprs = []
    for col in interact_cols:
        if col in df.columns:
            expr = (pl.col(col) * regime).alias(f"{col}_regime")
            exprs.append(expr.clip(config.CLIP_MIN, config.CLIP_MAX))
    df = df.with_columns(exprs)
    return df

def add_pairwise_interactions(df: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
    """Generate pairwise products up to MAX_PAIRWISE_INTERACTIONS."""
    sorted_features = sorted(feature_cols)
    exprs = []
    count = 0
    for a, b in combinations(sorted_features, 2):
        if count >= config.MAX_PAIRWISE_INTERACTIONS:
            break
        name = f"pair_{a}_x_{b}"
        expr = (pl.col(a) * pl.col(b)).cast(pl.Float32)
        exprs.append(expr.clip(config.CLIP_MIN, config.CLIP_MAX).alias(name))
        count += 1
    if exprs:
        df = df.with_columns(exprs)
    return df

def safe_add_pairwise_interactions(df: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
    n_features = len(feature_cols)
    max_combinations = config.MAX_PAIRWISE_INTERACTIONS
    total_possible = n_features * (n_features - 1) // 2
    if total_possible > max_combinations:
        logger.info(f"Pairwise combinations would exceed {max_combinations}, capping.")
    return add_pairwise_interactions(df, feature_cols)

def add_cross_timeframe_interactions(df: pl.DataFrame, ltf_features: list, htf_features: list) -> pl.DataFrame:
    """Multiply low‑frequency (5min) features with high‑frequency (1h/daily/htf) features."""
    ltf_sorted = sorted(ltf_features)
    htf_sorted = sorted(htf_features)
    exprs = []
    count = 0
    for a in ltf_sorted:
        for b in htf_sorted:
            if count >= config.MAX_CROSS_TIMEFRAME_INTERACTIONS:
                break
            name = f"cross_{a}_x_{b}"
            expr = (pl.col(a) * pl.col(b)).cast(pl.Float32)
            exprs.append(expr.clip(config.CLIP_MIN, config.CLIP_MAX).alias(name))
            count += 1
        if count >= config.MAX_CROSS_TIMEFRAME_INTERACTIONS:
            break
    if exprs:
        df = df.with_columns(exprs)
    return df

# ----------------------------------------------------------------------
# NEW FEATURE FAMILIES
# ----------------------------------------------------------------------

def add_rolling_quantiles(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Add rolling quantiles (20th, 50th, 80th) of log returns."""
    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    for q in [0.2, 0.5, 0.8]:
        expr = ret.rolling_quantile(probability=q, window_size=window)
        df = df.with_columns(expr.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias(f"feature_ret_quantile_{q}_{window}"))
    return df

def add_fourier_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add sine/cosine of time of day (period 24h) and day of week."""
    ts_local = pl.col("ts_event").dt.convert_time_zone(config.TIMEZONE)
    minute_of_day = ts_local.dt.hour() * 60 + ts_local.dt.minute()
    period = 24 * 60
    sin_time = (2 * np.pi * minute_of_day / period).sin()
    cos_time = (2 * np.pi * minute_of_day / period).cos()
    dow = ts_local.dt.weekday()
    df = df.with_columns([
        sin_time.cast(pl.Float32).clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_sin_time"),
        cos_time.cast(pl.Float32).clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_cos_time"),
        dow.cast(pl.Float32).clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_dow"),
    ])
    return df

def add_rolling_moments(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Add rolling skewness and kurtosis of log returns."""
    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    skew = ret.rolling_skew(window_size=window)
    kurt = ret.rolling_kurt(window_size=window)
    df = df.with_columns([
        skew.fill_nan(0.0).fill_null(0.0).clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias(f"feature_ret_skew_{window}"),
        kurt.fill_nan(0.0).fill_null(0.0).clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias(f"feature_ret_kurt_{window}"),
    ])
    return df

def add_acceleration(df: pl.DataFrame) -> pl.DataFrame:
    """Price acceleration: second difference of log returns."""
    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    acc = ret - ret.shift(1)
    df = df.with_columns(acc.fill_nan(0.0).fill_null(0.0).clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias("feature_ret_acceleration"))
    return df

def add_vwap_deviation(df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
    """Deviation of close from rolling Volume-Weighted Average Price (VWAP)."""
    typical_price = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    cum_pv = (typical_price * pl.col("volume")).rolling_sum(window_size=window)
    cum_vol = pl.col("volume").rolling_sum(window_size=window)
    vwap = cum_pv / cum_vol.clip(config.EPS, None)
    deviation = (pl.col("close") - vwap) / vwap.clip(config.EPS, None)
    df = df.with_columns(deviation.fill_nan(0.0).fill_null(0.0).clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias("feature_vwap_deviation"))
    return df

def add_regime_all_interactions(df: pl.DataFrame, baseline_cols: list) -> pl.DataFrame:
    """Multiply all baseline features by regime indicator."""
    regime = pl.col("regime")
    exprs = []
    for col in baseline_cols:
        if col in df.columns:
            expr = (pl.col(col) * regime).alias(f"{col}_regime")
            exprs.append(expr.clip(config.CLIP_MIN, config.CLIP_MAX))
    if exprs:
        df = df.with_columns(exprs)
    return df

# ----------------------------------------------------------------------
# Main expand_features (updated to call new functions)
# ----------------------------------------------------------------------

def expand_features(df: pl.DataFrame, baseline_feature_cols: list) -> pl.DataFrame:
    """
    Full expansion pipeline:
      - regime
      - ratios & z-scores
      - regime-conditioned transforms (limited)
      - rolling quantiles
      - Fourier (time of day, day of week)
      - rolling skew/kurtosis
      - acceleration
      - VWAP deviation
      - regime interactions for all baseline features
      - pairwise interactions (capped)
    (Cross‑timeframe interactions are added later in engine.py)
    """
    df = add_regime(df)
    df = add_ratios_and_z_scores(df, baseline_feature_cols)
    df = add_regime_conditioned_transforms(df)

    # --- New features ---
    df = add_rolling_quantiles(df)
    df = add_fourier_features(df)
    df = add_rolling_moments(df)
    df = add_acceleration(df)
    df = add_vwap_deviation(df)
    df = add_regime_all_interactions(df, baseline_feature_cols)

    # Collect all existing feature-like columns for further expansion
    current_features = [c for c in df.columns if c.startswith(("feature_", "ratio_", "pair_", "zscore", "cross_", "htf_", "1h_", "daily_"))]
    htf_cols = [c for c in df.columns if c.startswith(("1h_", "daily_", "htf_"))]

    # Memory safety: estimate total column count after adding pairwise and cross interactions
    est_pairwise = min(config.MAX_PAIRWISE_INTERACTIONS, len(current_features) * (len(current_features) - 1) // 2)
    est_cross = 0
    if htf_cols:
        est_cross = min(config.MAX_CROSS_TIMEFRAME_INTERACTIONS, len(current_features) * len(htf_cols))
    total_est = len(df.columns) + est_pairwise + est_cross
    if total_est > 5000:  # conservative limit to avoid OOM
        raise MemoryError(f"Estimated feature count {total_est} exceeds safety limit of 5000. "
                          f"Reduce MAX_PAIRWISE_INTERACTIONS or MAX_CROSS_TIMEFRAME_INTERACTIONS.")

    # Add pairwise interactions (capped)
    df = safe_add_pairwise_interactions(df, current_features)

    # Final clipping and nan fill for all non‑metadata columns
    exclude_cols = {"ts_event", "open", "high", "low", "close", "volume", "session_id", "regime"}
    all_feature_cols = [c for c in df.columns if c not in exclude_cols]
    for col in all_feature_cols:
        df = df.with_columns(
            pl.col(col).fill_nan(config.REPLACE_INF_NAN_WITH).fill_null(config.REPLACE_INF_NAN_WITH).clip(config.CLIP_MIN, config.CLIP_MAX)
        )
    return df