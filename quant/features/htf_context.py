pass
import polars as pl
from config import config

def _compute_trend_and_vol(df_agg: pl.DataFrame, ts_col: str, close_col: str, trend_window: int=10, vol_window: int=5):
    pass
    if df_agg is None or df_agg.height == 0:
        return pl.DataFrame()
    s = df_agg.select([ts_col, close_col]).unique(subset=[ts_col]).sort(ts_col)
    s = s.with_columns(pl.col(close_col).log().alias('log_close'))
    s = s.with_columns(((pl.col('log_close') - pl.col('log_close').shift(trend_window)) / trend_window).alias(f'trend_slope_{trend_window}'))
    ret = (pl.col(close_col).log() - pl.col(close_col).log().shift(1)).alias('log_ret')
    s = s.with_columns(ret)
    s = s.with_columns(pl.col('log_ret').rolling_std(window_size=vol_window).alias(f'vol_{vol_window}'))
    keep = [ts_col, f'trend_slope_{trend_window}', f'vol_{vol_window}']
    return s.select(keep)

def add_htf_context_features(df: pl.DataFrame) -> pl.DataFrame:
    pass
    if df is None or df.height == 0:
        return df
    out = df
    if 'daily_ts_event' in df.columns and 'daily_close' in df.columns:
        daily_agg = df.select(['daily_ts_event', 'daily_close']).unique(subset=['daily_ts_event']).sort('daily_ts_event')
        daily_feats = _compute_trend_and_vol(daily_agg, 'daily_ts_event', 'daily_close', trend_window=10, vol_window=5)
        if daily_feats.height > 0:
            daily_feats = daily_feats.rename({'daily_ts_event': 'daily_ts_event', 'trend_slope_10': 'htf_daily_trend_slope_10', 'vol_5': 'htf_daily_vol_5'})
            out = out.join(daily_feats, on='daily_ts_event', how='left')
    else:
        out = out.with_columns([pl.lit(None).alias('htf_daily_trend_slope_10'), pl.lit(None).alias('htf_daily_vol_5')])
    if '1h_ts_event' in df.columns and '1h_close' in df.columns:
        hourly_agg = df.select(['1h_ts_event', '1h_close']).unique(subset=['1h_ts_event']).sort('1h_ts_event')
        hourly_feats = _compute_trend_and_vol(hourly_agg, '1h_ts_event', '1h_close', trend_window=10, vol_window=5)
        if hourly_feats.height > 0:
            hourly_feats = hourly_feats.rename({'1h_ts_event': '1h_ts_event', 'trend_slope_10': 'htf_hourly_trend_slope_10', 'vol_5': 'htf_hourly_vol_5'})
            out = out.join(hourly_feats, on='1h_ts_event', how='left')
    else:
        out = out.with_columns([pl.lit(None).alias('htf_hourly_trend_slope_10'), pl.lit(None).alias('htf_hourly_vol_5')])
    if 'htf_daily_trend_slope_10' in out.columns and 'htf_hourly_trend_slope_10' in out.columns:
        out = out.with_columns(pl.when(pl.col('htf_daily_trend_slope_10').is_not_null() & pl.col('htf_hourly_trend_slope_10').is_not_null() & (pl.col('htf_daily_trend_slope_10').sign() == pl.col('htf_hourly_trend_slope_10').sign())).then(1.0).otherwise(0.0).alias('htf_hourly_trend_alignment'))
    else:
        out = out.with_columns(pl.lit(0.0).alias('htf_hourly_trend_alignment'))
    cast_cols = [c for c in out.columns if c.startswith('htf_')]
    for c in cast_cols:
        out = out.with_columns(pl.col(c).cast(pl.Float32))
    return out
'\nsrc/features/htf_context.py\nCompute higher‑timeframe context features from aligned 1h and daily data.\nAll features are past‑only, float32, clipped.\nNow uses precomputed daily_vol_5 from the daily stream.\n'
import polars as pl
from config import config

def add_htf_context_features(df: pl.DataFrame) -> pl.DataFrame:
    pass
    df = df.with_columns((pl.col('daily_close') / pl.col('daily_close').shift(1)).log().alias('htf_daily_return_1'))
    df = df.with_columns(pl.col('daily_vol_5').alias('htf_daily_vol_5'))
    df = df.with_columns(((pl.col('daily_close') - pl.col('daily_close').shift(10)) / 10.0 / pl.col('daily_close').shift(10).clip(config.EPS, None)).alias('htf_daily_trend_slope_10'))
    df = df.with_columns(((pl.col('daily_high') - pl.col('close')) / pl.col('daily_high').clip(config.EPS, None)).alias('htf_distance_to_daily_high'), ((pl.col('close') - pl.col('daily_low')) / pl.col('daily_low').clip(config.EPS, None)).alias('htf_distance_to_daily_low'))
    df = df.with_columns((pl.col('1h_close') / pl.col('1h_close').shift(1)).log().alias('1h_return'))
    df = df.with_columns((pl.col('1h_return') * pl.col('htf_daily_trend_slope_10').sign()).alias('htf_hourly_trend_alignment'))
    df = df.with_columns(pl.col('1h_return').rolling_std(window_size=4).alias('1h_vol_4'))
    df = df.with_columns((pl.col('1h_vol_4') / pl.col('htf_daily_vol_5').clip(config.EPS, None)).alias('htf_volatility_ratio'))
    htf_cols = [c for c in df.columns if c.startswith('htf_')]
    for col in htf_cols:
        df = df.with_columns(pl.col(col).clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32))
    df = df.drop(['1h_return', '1h_vol_4'])
    return df