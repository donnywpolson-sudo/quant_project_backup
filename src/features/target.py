"""
src/features/target.py
Construct target_5m = forward 1-bar log return (5-min bars -> 5-min).
Continuous target, no lookahead.
"""
import polars as pl
from config import config

def add_target_5m(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add target_5m = log(close[t+1] / close[t]).
    The last row will have null target (no forward data) – dropped later.
    """
    log_close = pl.col("close").log()
    forward_ret = (log_close.shift(-1) - log_close).alias("target_5m")
    df = df.with_columns(forward_ret)
    df = df.with_columns(pl.col("target_5m").clip(config.CLIP_MIN, config.CLIP_MAX))
    return df

def drop_incomplete_target(df: pl.DataFrame) -> pl.DataFrame:
    """Remove rows where target is null (end of dataset)."""
    return df.filter(pl.col("target_5m").is_not_null())