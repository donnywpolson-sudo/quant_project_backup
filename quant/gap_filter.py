"""
quant/gap_filter.py — Explicit gap detection and removal.

Catches session gaps, exchange outages, and data feed interruptions
that resampling n_ticks thresholds alone may miss.
"""

import polars as pl


def filter_gaps(df: pl.DataFrame, max_gap_minutes: float = 30) -> pl.DataFrame:
    """
    Remove bars where the time gap between consecutive ts_event values
    exceeds *max_gap_minutes*.

    Keeps existing resampling tick thresholds unchanged; this is an
    additional safety net run after alignment.
    """
    df = df.sort('ts_event')
    # ts_event is datetime[ns, UTC] — diff yields nanoseconds.
    # 60e9 ns = 1 minute.
    gap = df['ts_event'].diff().cast(pl.Int64) / 60_000_000_000.0  # minutes
    df = df.with_columns(gap.alias('_gap_minutes'))
    # The first row has diff=None; keep it.
    df = df.filter(
        pl.col('_gap_minutes').is_null() | (pl.col('_gap_minutes') <= max_gap_minutes)
    )
    return df.drop('_gap_minutes')
