"""
src/align.py
Align 5‑min, 1‑hour and Daily streams without lookahead.
Uses asof join for 1h and proper daily lag.
Now includes daily_vol_5 from the daily stream.
"""
import polars as pl
import logging
from config import config

logger = logging.getLogger(__name__)


def align_htf_streams(df_5min: pl.DataFrame, df_1h: pl.DataFrame, df_daily: pl.DataFrame) -> pl.DataFrame:
    """
    For each 5‑min bar, add columns from the most recent 1h bar (closed <= 5min timestamp)
    and the most recent daily bar (closed before the session).
    Returns a single DataFrame with all 5min columns plus prefixed HTF columns.
    """
    # Ensure sorted
    df_5min = df_5min.sort("ts_event")
    df_1h = df_1h.sort("ts_event")
    df_daily = df_daily.sort("ts_event")

    # ---- 1. Join 1h using asof (backward) ----
    df_1h_renamed = df_1h.select([
        "ts_event",
        pl.col("open").alias("1h_open"),
        pl.col("high").alias("1h_high"),
        pl.col("low").alias("1h_low"),
        pl.col("close").alias("1h_close"),
        pl.col("volume").alias("1h_volume"),
    ])
    df_aligned = df_5min.join_asof(
        df_1h_renamed,
        on="ts_event",
        strategy="backward"
    )

    # ---- 2. Join daily using previous day's close ----
    df_aligned = df_aligned.with_columns(
        pl.col("ts_event").dt.date().alias("date_5min")
    )
    df_daily = df_daily.with_columns(
        pl.col("ts_event").dt.date().alias("date_daily")
    )
    # For each 5min date, take the daily bar from the previous trading day
    df_daily_prev = df_daily.with_columns(
        (pl.col("date_daily") + pl.duration(days=1)).alias("next_day")
    ).select([
        pl.col("date_daily").alias("prev_date"),
        pl.col("next_day"),
        pl.col("open").alias("daily_open"),
        pl.col("high").alias("daily_high"),
        pl.col("low").alias("daily_low"),
        pl.col("close").alias("daily_close"),
        pl.col("volume").alias("daily_volume"),
        pl.col("daily_vol_5").alias("daily_vol_5"),   # <-- added
    ])
    df_aligned = df_aligned.join(
        df_daily_prev,
        left_on="date_5min",
        right_on="next_day",
        how="left"
    )
    # Forward fill daily columns for the first days where no previous day exists
    daily_cols = ["daily_open", "daily_high", "daily_low", "daily_close", "daily_volume", "daily_vol_5"]
    for col in daily_cols:
        df_aligned = df_aligned.with_columns(pl.col(col).fill_null(strategy="forward"))

    # Drop helper columns (ignore if missing)
    df_aligned = df_aligned.drop(["date_5min", "prev_date", "next_day"], strict=False)
    return df_aligned