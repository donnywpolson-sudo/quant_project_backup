"""
src/features/target.py
Construct target for horizon = TARGET_5M_HORIZON (default 5 bars).
Binary target: 1 if forward log return > 0 else 0.
No lookahead.
"""
import polars as pl
from config import config

def add_target_5m(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add target_sign = 1 if log(close[t+horizon]/close[t]) > 0 else 0.
    Also keep target_5m (scaled continuous) for compatibility.
    """
    horizon = config.TARGET_5M_HORIZON
    log_close = pl.col("close").log()
    forward_ret_raw = (log_close.shift(-horizon) - log_close).alias("forward_ret_raw")
    # Scaled continuous target (not used in classification)
    forward_ret_scaled = forward_ret_raw * config.TARGET_SCALE_FACTOR
    df = df.with_columns(forward_ret_scaled.alias("target_5m"))
    df = df.with_columns(pl.col("target_5m").clip(config.CLIP_MIN, config.CLIP_MAX))
    # Binary classification target
    df = df.with_columns((forward_ret_raw > 0).cast(pl.Int8).alias("target_sign"))
    return df

def drop_incomplete_target(df: pl.DataFrame) -> pl.DataFrame:
    """Remove rows where target is null (end of dataset, horizon bars lost)."""
    return df.filter(pl.col("target_sign").is_not_null())