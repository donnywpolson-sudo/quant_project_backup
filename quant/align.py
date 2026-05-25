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
    # Ensure sorted; guard against empty HTF inputs
    df_5min = df_5min.sort("ts_event")
    if df_1h is None or df_1h.is_empty() or "ts_event" not in df_1h.columns:
        df_1h = None
    else:
        df_1h = df_1h.sort("ts_event")
    if df_daily is None or df_daily.is_empty() or "ts_event" not in df_daily.columns:
        df_daily = None
    else:
        df_daily = df_daily.sort("ts_event")

    # ---- 1. Join 1h using asof (backward) ----
    # Keep the original 1h timestamp as an explicit column so downstream
    # logic can reference the 1h bar boundary (no ambiguity with the 5m ts_event).
    if df_1h is not None:
        # keep the original ts_event (for join) and also expose the 1h timestamp
        df_1h_renamed = df_1h.select([
        pl.col("ts_event"),
        pl.col("ts_event").alias("1h_ts_event"),
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
    else:
        df_aligned = df_5min

    # ---- 2. Join daily using asof (backward) – no future leakage ----
    if df_daily is not None:
        # keep original ts_event (for asof join) and expose daily_ts_event
        df_daily_renamed = df_daily.select([
        pl.col("ts_event"),
        pl.col("ts_event").alias("daily_ts_event"),
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
        # Ensure daily volatility is forward-filled so early bars inherit previous daily stats
        if "daily_vol_5" in df_aligned.columns:
            df_aligned = df_aligned.with_columns(
                pl.col("daily_vol_5").fill_null(strategy="forward").fill_nan(0.0).fill_null(0.0)
            )

    # No forward fill – asof join already gives last known daily bar
    return df_aligned