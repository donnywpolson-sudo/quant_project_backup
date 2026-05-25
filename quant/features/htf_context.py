"""
src/features/htf_context.py
Compute high-timeframe (1h, daily) context features and join back to 5-min rows.
Produces columns such as `htf_daily_vol_5`, `htf_daily_trend_slope_10`,
`htf_hourly_trend_slope_10`, and `htf_hourly_trend_alignment`.
Handles missing HTF streams gracefully.
"""
import polars as pl
from config import config


def _compute_trend_and_vol(df_agg: pl.DataFrame, ts_col: str, close_col: str, trend_window: int = 10, vol_window: int = 5):
    """Given an aggregated HTF dataframe with `ts_col` and `close_col` compute
    trend slope (simple finite difference) and rolling vol (std of log returns).
    Returns a DataFrame with ts_col, trend_slope_<w>, vol_<w>.
    """
    if df_agg is None or df_agg.height == 0:
        return pl.DataFrame()

    s = df_agg.select([ts_col, close_col]).unique(subset=[ts_col]).sort(ts_col)
    # compute log close
    s = s.with_columns((pl.col(close_col).log()).alias("log_close"))
    # forward/backward safe: compute slope = (log_close - log_close.shift(window)) / window
    s = s.with_columns(
        ((pl.col("log_close") - pl.col("log_close").shift(trend_window)) / trend_window).alias(f"trend_slope_{trend_window}")
    )
    # compute rolling std of log returns (vol)
    ret = (pl.col(close_col).log() - pl.col(close_col).log().shift(1)).alias("log_ret")
    s = s.with_columns(ret)
    s = s.with_columns((pl.col("log_ret").rolling_std(window_size=vol_window)).alias(f"vol_{vol_window}"))
    # keep only relevant columns
    keep = [ts_col, f"trend_slope_{trend_window}", f"vol_{vol_window}"]
    return s.select(keep)


def add_htf_context_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute HTF features and join back to 5-min DataFrame.

    Expects `1h_ts_event`/`1h_close` and `daily_ts_event`/`daily_close` to be
    present in `df` (as produced by `quant.align.align_htf_streams`).
    """
    if df is None or df.height == 0:
        return df

    out = df

    # DAILY HTF
    if "daily_ts_event" in df.columns and "daily_close" in df.columns:
        daily_agg = df.select(["daily_ts_event", "daily_close"]).unique(subset=["daily_ts_event"]).sort("daily_ts_event")
        daily_feats = _compute_trend_and_vol(daily_agg, "daily_ts_event", "daily_close",
                                            trend_window=10, vol_window=5)
        if daily_feats.height > 0:
            daily_feats = daily_feats.rename({"daily_ts_event": "daily_ts_event", "trend_slope_10": "htf_daily_trend_slope_10", "vol_5": "htf_daily_vol_5"})
            out = out.join(daily_feats, on="daily_ts_event", how="left")
    else:
        out = out.with_columns([
            pl.lit(None).alias("htf_daily_trend_slope_10"),
            pl.lit(None).alias("htf_daily_vol_5"),
        ])

    # HOURLY HTF
    if "1h_ts_event" in df.columns and "1h_close" in df.columns:
        hourly_agg = df.select(["1h_ts_event", "1h_close"]).unique(subset=["1h_ts_event"]).sort("1h_ts_event")
        hourly_feats = _compute_trend_and_vol(hourly_agg, "1h_ts_event", "1h_close",
                                             trend_window=10, vol_window=5)
        if hourly_feats.height > 0:
            hourly_feats = hourly_feats.rename({"1h_ts_event": "1h_ts_event", "trend_slope_10": "htf_hourly_trend_slope_10", "vol_5": "htf_hourly_vol_5"})
            out = out.join(hourly_feats, on="1h_ts_event", how="left")
    else:
        out = out.with_columns([
            pl.lit(None).alias("htf_hourly_trend_slope_10"),
            pl.lit(None).alias("htf_hourly_vol_5"),
        ])

    # Alignment: whether hourly trend aligns with daily trend
    if "htf_daily_trend_slope_10" in out.columns and "htf_hourly_trend_slope_10" in out.columns:
        out = out.with_columns(
            (pl.when((pl.col("htf_daily_trend_slope_10").is_not_null()) & (pl.col("htf_hourly_trend_slope_10").is_not_null()) &
                     (pl.col("htf_daily_trend_slope_10").sign() == pl.col("htf_hourly_trend_slope_10").sign())
             ).then(1.0).otherwise(0.0)).alias("htf_hourly_trend_alignment")
        )
    else:
        out = out.with_columns(pl.lit(0.0).alias("htf_hourly_trend_alignment"))

    # Cast numeric HTF features to Float32 for consistency
    cast_cols = [c for c in out.columns if c.startswith(("htf_"))]
    for c in cast_cols:
        out = out.with_columns(pl.col(c).cast(pl.Float32))

    return out
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
        # Preserve NaNs for HTF columns (don't replace with 0); just cast and clip
        df = df.with_columns(
            pl.col(col).clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32)
        )
    # Drop intermediate columns
    df = df.drop(["1h_return", "1h_vol_4"])
    return df