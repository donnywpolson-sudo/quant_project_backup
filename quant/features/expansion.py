import polars as pl
import numpy as np
import logging
from itertools import combinations
from config import config
logger = logging.getLogger(__name__)

def add_regime(df: pl.DataFrame) -> pl.DataFrame:
    ret = (pl.col('close') / pl.col('close').shift(1)).log().cast(pl.Float32)
    vol20 = ret.rolling_std(window_size=20)
    med_vol = vol20.rolling_median(window_size=config.VOL_MEDIAN_WINDOW)
    smooth_vol = med_vol.rolling_mean(window_size=config.VOL_SMOOTH_WINDOW)
    regime = pl.when(smooth_vol >= config.REGIME_HIGH_THRESH).then(1.0).when(smooth_vol <= config.REGIME_LOW_THRESH).then(0.0).otherwise(None)
    regime = regime.fill_null(strategy='forward').fill_null(config.REGIME_MISSING_DEFAULT)
    df = df.with_columns(regime.cast(pl.Float32).alias('regime'))
    return df

def add_ratios_and_z_scores(df: pl.DataFrame, base_features: list) -> pl.DataFrame:
    core = ['close', 'volume', 'feature_spread_proxy', 'feature_high_low_range_norm']
    existing = [c for c in core if c in df.columns]
    exprs = []
    for i, a in enumerate(existing):
        for b in existing[i + 1:]:
            name = f'ratio_{a}_over_{b}'
            expr = (pl.col(a) / pl.col(b).clip(config.EPS, None)).cast(pl.Float32)
            exprs.append(expr.clip(config.CLIP_MIN, config.CLIP_MAX).alias(name))
    for col in base_features[:20]:
        if col in df.columns:
            mean = pl.col(col).rolling_mean(window_size=20)
            std = pl.col(col).rolling_std(window_size=20)
            z = (pl.col(col) - mean) / std.clip(config.EPS, None)
            exprs.append(z.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f'{col}_zscore'))
    df = df.with_columns(exprs)
    return df

def add_regime_conditioned_transforms(df: pl.DataFrame) -> pl.DataFrame:
    regime = pl.col('regime')
    interact_cols = ['feature_ret_1', 'feature_ret_5', 'feature_ewma_vol_20', 'feature_volume_z_20']
    exprs = []
    for col in interact_cols:
        if col in df.columns:
            expr = (pl.col(col) * regime).alias(f'{col}_regime')
            exprs.append(expr.clip(config.CLIP_MIN, config.CLIP_MAX))
    df = df.with_columns(exprs)
    return df

def add_regime_all_interactions(df: pl.DataFrame, baseline_cols: list) -> pl.DataFrame:
    regime = pl.col('regime')
    subset = [c for c in baseline_cols if any((x in c for x in ('vol', 'ret_1', 'momentum', 'spread')))]
    exprs = []
    for col in subset:
        if col in df.columns:
            exprs.append((pl.col(col) * regime).alias(f'{col}_regime'))
    if exprs:
        df = df.with_columns(exprs)
    return df

def add_pairwise_interactions(df: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
    sorted_features = sorted(feature_cols)
    exprs = []
    count = 0
    for a, b in combinations(sorted_features, 2):
        if count >= config.MAX_PAIRWISE_INTERACTIONS:
            break
        name = f'pair_{a}_x_{b}'
        expr = (pl.col(a) * pl.col(b)).cast(pl.Float32)
        exprs.append(expr.clip(config.CLIP_MIN, config.CLIP_MAX).alias(name))
        count += 1
    if exprs:
        batch_size = 50
        for i in range(0, len(exprs), batch_size):
            batch = exprs[i:i + batch_size]
            df = df.with_columns(batch)
    return df

def safe_add_pairwise_interactions(df: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
    n_features = len(feature_cols)
    total_possible = n_features * (n_features - 1) // 2
    if total_possible > config.MAX_PAIRWISE_INTERACTIONS:
        logger.info(f'Pairwise combinations would exceed {config.MAX_PAIRWISE_INTERACTIONS}, capping.')
    return add_pairwise_interactions(df, feature_cols)

def add_cross_timeframe_interactions(df: pl.DataFrame, ltf_features: list, htf_features: list) -> pl.DataFrame:
    ltf_sorted = [c for c in sorted(ltf_features) if c in df.columns]
    htf_sorted = [c for c in sorted(htf_features) if c in df.columns]
    numeric_types = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    exprs = []
    count = 0
    batch_size = 50
    for a in ltf_sorted:
        if count >= config.MAX_CROSS_TIMEFRAME_INTERACTIONS:
            break
        if df[a].dtype not in numeric_types:
            continue
        for b in htf_sorted:
            if count >= config.MAX_CROSS_TIMEFRAME_INTERACTIONS:
                break
            if df[b].dtype not in numeric_types:
                continue
            name = f'cross_{a}_x_{b}'
            expr = (pl.col(a) * pl.col(b)).cast(pl.Float32)
            exprs.append(expr.clip(config.CLIP_MIN, config.CLIP_MAX).alias(name))
            count += 1
            if len(exprs) >= batch_size:
                df = df.with_columns(exprs)
                exprs = []
    if exprs:
        df = df.with_columns(exprs)
    return df

def add_rolling_quantiles(df: pl.DataFrame, window: int=20) -> pl.DataFrame:
    ret = (pl.col('close') / pl.col('close').shift(1)).log().cast(pl.Float32)
    for q in [0.2, 0.5, 0.8]:
        expr = ret.rolling_quantile(quantile=q, window_size=window)
        df = df.with_columns(expr.clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias(f'feature_ret_quantile_{q}_{window}'))
    return df

def add_fourier_features(df: pl.DataFrame) -> pl.DataFrame:
    ts_local = pl.col('ts_event').dt.convert_time_zone(config.TIMEZONE)
    minute_of_day = ts_local.dt.hour() * 60 + ts_local.dt.minute()
    period = 24 * 60
    sin_time = (2 * np.pi * minute_of_day / period).sin()
    cos_time = (2 * np.pi * minute_of_day / period).cos()
    dow = ts_local.dt.weekday()
    df = df.with_columns([sin_time.cast(pl.Float32).clip(config.CLIP_MIN, config.CLIP_MAX).alias('feature_sin_time'), cos_time.cast(pl.Float32).clip(config.CLIP_MIN, config.CLIP_MAX).alias('feature_cos_time'), dow.cast(pl.Float32).clip(config.CLIP_MIN, config.CLIP_MAX).alias('feature_dow')])
    return df

def add_rolling_moments(df: pl.DataFrame, window: int=20) -> pl.DataFrame:
    ret = (pl.col('close') / pl.col('close').shift(1)).log().cast(pl.Float32)
    w = window
    sum_x = ret.rolling_sum(window_size=w)
    sum_x2 = (ret * ret).rolling_sum(window_size=w)
    sum_x3 = (ret * ret * ret).rolling_sum(window_size=w)
    sum_x4 = (ret * ret * ret * ret).rolling_sum(window_size=w)
    mean = sum_x / w
    var = (sum_x2 - w * mean * mean) / (w - 1)
    std = var.sqrt()
    m3 = sum_x3 - 3 * mean * sum_x2 + 2 * w * mean * mean * mean
    skew = pl.when(w > 2).then(m3 / (w - 1) / (std.pow(3) + config.EPS) * (pl.lit(w) / (pl.lit(w) - 2))).otherwise(pl.lit(0.0))
    m4 = sum_x4 - 4 * mean * sum_x3 + 6 * mean * mean * sum_x2 - 3 * w * mean * mean * mean * mean
    kurt = m4 / (w * (var * var + config.EPS)) - 3.0
    skew = skew.fill_nan(0.0).fill_null(0.0).clip(config.CLIP_MIN, config.CLIP_MAX)
    kurt = kurt.fill_nan(0.0).fill_null(0.0).clip(config.CLIP_MIN, config.CLIP_MAX)
    df = df.with_columns([skew.cast(pl.Float32).alias(f'feature_ret_skew_{window}'), kurt.cast(pl.Float32).alias(f'feature_ret_kurt_{window}')])
    return df

def add_acceleration(df: pl.DataFrame) -> pl.DataFrame:
    ret = (pl.col('close') / pl.col('close').shift(1)).log().cast(pl.Float32)
    acc = ret - ret.shift(1)
    df = df.with_columns(acc.fill_nan(0.0).fill_null(0.0).clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_ret_acceleration'))
    return df

def add_vwap_deviation(df: pl.DataFrame, window: int=20) -> pl.DataFrame:
    typical_price = (pl.col('high') + pl.col('low') + pl.col('close')) / 3.0
    cum_pv = (typical_price * pl.col('volume')).rolling_sum(window_size=window)
    cum_vol = pl.col('volume').rolling_sum(window_size=window)
    vwap = cum_pv / cum_vol.clip(config.EPS, None)
    deviation = (pl.col('close') - vwap) / vwap.clip(config.EPS, None)
    df = df.with_columns(deviation.fill_nan(0.0).fill_null(0.0).clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32).alias('feature_vwap_deviation'))
    return df

def expand_features(df: pl.DataFrame, baseline_feature_cols: list) -> pl.DataFrame:
    df = add_regime(df)
    df = add_ratios_and_z_scores(df, baseline_feature_cols)
    df = add_regime_conditioned_transforms(df)
    df = add_rolling_quantiles(df)
    df = add_fourier_features(df)
    df = add_rolling_moments(df)
    df = add_acceleration(df)
    df = add_vwap_deviation(df)
    df = add_regime_all_interactions(df, baseline_feature_cols)
    if df.height <= 300000:
        current_features = [c for c in df.columns if c.startswith(('feature_', 'ratio_', 'pair_', 'zscore', 'cross_', 'htf_'))]
        df = safe_add_pairwise_interactions(df, current_features)
    exclude_cols = {'ts_event', 'open', 'high', 'low', 'close', 'volume', 'session_id', 'regime'}
    numeric_types = (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    exprs = []
    for c, t in zip(df.columns, df.dtypes):
        if c in exclude_cols:
            continue
        if c.startswith('htf_') or c.startswith('daily_') or c.startswith('1h_'):
            continue
        if isinstance(t, tuple(numeric_types)) or t in numeric_types:
            exprs.append(pl.col(c).fill_nan(config.REPLACE_INF_NAN_WITH).fill_null(config.REPLACE_INF_NAN_WITH).clip(config.CLIP_MIN, config.CLIP_MAX).alias(c))
    if exprs:
        df = df.with_columns(exprs)
    return df