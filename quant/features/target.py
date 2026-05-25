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
    # Keep rows even if 1h target is not available yet; by default drop rows
    # where the 5m target is missing to preserve compatibility.
    if "target_sign" in df.columns:
        return df.filter(pl.col("target_sign").is_not_null())
    return df


def add_target_1h(df: pl.DataFrame) -> pl.DataFrame:
    """
    Construct a 1-hour horizon target aligned to 5-minute rows.
    Method: build a small table of unique `1h_ts_event` and its `1h_close`,
    compute forward 1h return between consecutive 1h bars, then join back
    to the main dataframe on `1h_ts_event` so every 5-min row inherits the
    1h target for its current hourly bar.
    Produces: `target_1h` (scaled continuous) and `target_sign_1h` (Int8).
    """
    if "1h_ts_event" not in df.columns or "1h_close" not in df.columns:
        return df

    # Extract distinct 1h bars (keep first occurrence per 1h_ts_event)
    one_h = (
        df.select(["1h_ts_event", "1h_close"]) 
        .drop_nulls("1h_ts_event")
        .unique(subset=["1h_ts_event"]) 
        .sort("1h_ts_event")
    )
    if one_h.height < 2:
        # Not enough 1h bars to form a forward target
        df = df.with_columns(
            pl.lit(None).alias("target_1h"),
        )
        df = df.with_columns(pl.lit(None).cast(pl.Int8).alias("target_sign_1h"))
        return df

    # Compute next 1h close and forward return
    one_h = one_h.with_columns(
        pl.col("1h_close").shift(-1).alias("1h_close_next")
    )
    one_h = one_h.with_columns(
        (pl.col("1h_close_next").log() - pl.col("1h_close").log()).alias("forward_ret_1h_raw")
    )
    one_h = one_h.with_columns((pl.col("forward_ret_1h_raw") * config.TARGET_SCALE_FACTOR).alias("target_1h"))
    one_h = one_h.with_columns((pl.col("forward_ret_1h_raw") > 0).cast(pl.Int8).alias("target_sign_1h"))

    # Drop the last row which has no forward target
    one_h = one_h.filter(pl.col("target_sign_1h").is_not_null())

    # Join back to main df on 1h_ts_event
    df = df.join(one_h.select(["1h_ts_event", "target_1h", "target_sign_1h"]), on="1h_ts_event", how="left")
    return df


def add_target_4h(df: pl.DataFrame) -> pl.DataFrame:
    """
    Construct a 4-hour horizon target aligned to 5-minute rows using a fixed-bar shift.
    Assumes 5-minute bars are contiguous; 4 hours == 48 * 5min bars.
    Produces: `target_4h` (scaled continuous) and `target_sign_4h` (Int8).
    """
    H_BARS = int((4 * 60) / 5)  # 48
    if "close" not in df.columns:
        return df

    log_close = pl.col("close").log()
    forward_ret_raw = (log_close.shift(-H_BARS) - log_close)
    df = df.with_columns((forward_ret_raw * config.TARGET_SCALE_FACTOR).alias("target_4h"))
    df = df.with_columns((forward_ret_raw > 0).cast(pl.Int8).alias("target_sign_4h"))
    # Drop rows at the end where target is null is handled by downstream drop_incomplete_target
    return df