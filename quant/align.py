"""
src/align.py
Align 5‑min, 1‑hour and Daily streams without lookahead.
Uses asof join for 1h and asof join for daily (backward).
Now includes daily_vol_5 from the daily stream.
"""
import polars as pl
import logging
from config import config

logger = logging.getLogger(__name__)


def align_htf_streams(df_5min: pl.DataFrame, df_1h: pl.DataFrame, df_daily: pl.DataFrame) -> pl.DataFrame:
    """
    For each 5‑min bar, add columns from the most recent 1h bar (closed <= 5min timestamp)
    and the most recent daily bar (closed before the 5min timestamp).
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

    # ---- 2. Join daily using asof (backward) – no future leakage ----
    df_daily_renamed = df_daily.select([
        "ts_event",
        pl.col("open").alias("daily_open"),
        pl.col("high").alias("daily_high"),
        pl.col("low").alias("daily_low"),
        pl.col("close").alias("daily_close"),
        pl.col("volume").alias("daily_volume"),
        pl.col("daily_vol_5").alias("daily_vol_5"),
    ])
    df_aligned = df_aligned.join_asof(
        df_daily_renamed,
        on="ts_event",
        strategy="backward"
    )

    # No forward fill – asof join already gives last known daily bar
    return df_aligned