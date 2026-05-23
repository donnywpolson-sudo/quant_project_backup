"""
src/features/htf_context.py
Compute higher‑timeframe context features from aligned 1h and daily data.
All features are past‑only, float32, clipped.
Now uses precomputed daily_vol_5 from the daily stream.
"""
import polars as pl
from config import config


def add_htf_context_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add HTF state features. Expects columns:
       1h_close, 1h_high, 1h_low, 1h_volume
       daily_close, daily_high, daily_low, daily_volume, daily_vol_5
    Returns df with additional columns prefixed 'htf_'.
    """
    # 1. Daily return (log, 1 day lag – already aligned to previous day)
    df = df.with_columns(
        (pl.col("daily_close") / pl.col("daily_close").shift(1)).log().alias("htf_daily_return_1")
    )
    # 2. Daily volatility – use precomputed daily_vol_5 from daily stream
    #    (already aligned as previous day's volatility)
    df = df.with_columns(
        pl.col("daily_vol_5").alias("htf_daily_vol_5")
    )
    # 3. Daily trend slope (10‑day linear approximation)
    df = df.with_columns(
        ((pl.col("daily_close") - pl.col("daily_close").shift(10)) / 10.0 / pl.col("daily_close").shift(10).clip(config.EPS, None))
        .alias("htf_daily_trend_slope_10")
    )
    # 4. Distance to daily high/low (normalized by daily high/low)
    df = df.with_columns(
        ((pl.col("daily_high") - pl.col("close")) / pl.col("daily_high").clip(config.EPS, None)).alias("htf_distance_to_daily_high"),
        ((pl.col("close") - pl.col("daily_low")) / pl.col("daily_low").clip(config.EPS, None)).alias("htf_distance_to_daily_low")
    )
    # 5. Hourly trend alignment (sign of 1h return vs daily trend)
    df = df.with_columns(
        (pl.col("1h_close") / pl.col("1h_close").shift(1)).log().alias("1h_return")
    )
    df = df.with_columns(
        (pl.col("1h_return") * pl.col("htf_daily_trend_slope_10").sign()).alias("htf_hourly_trend_alignment")
    )
    # 6. Volatility ratio (1h volatility / daily volatility)
    df = df.with_columns(
        pl.col("1h_return").rolling_std(window_size=4).alias("1h_vol_4")
    )
    df = df.with_columns(
        (pl.col("1h_vol_4") / pl.col("htf_daily_vol_5").clip(config.EPS, None)).alias("htf_volatility_ratio")
    )
    # 7. Daily session phase – reuse existing feature_session_pos (already in df)

    # Clean and cast all HTF columns
    htf_cols = [c for c in df.columns if c.startswith("htf_")]
    for col in htf_cols:
        df = df.with_columns(
            pl.col(col).fill_nan(config.REPLACE_INF_NAN_WITH).fill_null(config.REPLACE_INF_NAN_WITH).clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32)
        )
    # Drop intermediate columns
    df = df.drop(["1h_return", "1h_vol_4"])
    return df