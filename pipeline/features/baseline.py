import polars as pl
import yaml
from core.config import config


def load_baseline_feature_names() -> list:
    with open(config.BASELINE_FEATURES_FILE, 'r') as f:
        data = yaml.safe_load(f)
    return data['baseline_features']


def compute_baseline_features(df: pl.DataFrame) -> pl.DataFrame:
    close = pl.col('close').cast(pl.Float32)
    high = pl.col('high').cast(pl.Float32)
    low = pl.col('low').cast(pl.Float32)
    open_ = pl.col('open').cast(pl.Float32)
    volume = pl.col('volume').cast(pl.Float32)
    eps = config.EPS
    exprs = []

    # Lagged returns: ret_t = log(close_t / close_{t-lag})
    for lag in [1, 5, 10, 20]:
        ret = (close / close.shift(lag).clip(eps, None)).log()
        exprs.append(ret.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f'feature_ret_{lag}'))

    # High-low range normalized by close
    range_norm = (high - low) / close.clip(eps, None)
    exprs.append(range_norm.clip(config.CLIP_MIN, config.CLIP_MAX).alias('feature_high_low_range_norm'))

    # True range
    prev_close = close.shift(1)
    tr = pl.max_horizontal(high - low, (high - prev_close).abs(), (low - prev_close).abs())
    exprs.append(tr.clip(config.CLIP_MIN, config.CLIP_MAX).alias('feature_true_range'))

    # EWMA-style volatility: rolling std of 1-bar returns, strictly lagged
    # Use returns [t-20, t-1] to avoid the current bar's return contaminating vol
    ret_1 = (close / close.shift(1).clip(eps, None)).log()
    ret_1_lagged = ret_1.shift(1)
    vol = ret_1_lagged.rolling_std(window_size=20, min_periods=5).clip(eps, None)
    exprs.append(vol.alias('feature_ewma_vol_20'))

    # Spread proxy
    spread_proxy = (high - low) / close.clip(eps, None)
    exprs.append(spread_proxy.alias('feature_spread_proxy'))

    df = df.with_columns(exprs)

    # Clean up: fill NaN/Null and clip all feature columns
    for col in df.columns:
        if col.startswith('feature_'):
            df = df.with_columns(
                pl.col(col).fill_nan(0.0).fill_null(0.0).clip(config.CLIP_MIN, config.CLIP_MAX)
            )
    return df