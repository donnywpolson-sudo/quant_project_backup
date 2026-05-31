import polars as pl
import yaml
from pipeline.common.config import config


BASELINE_FEATURES = [
    # Returns / momentum
    'feature_ret_1', 'feature_ret_3', 'feature_ret_5',
    'feature_ret_15', 'feature_ret_30', 'feature_ret_60',
    'feature_mom_accel_5_15', 'feature_mom_accel_15_30',

    # Rolling range / location
    'feature_close_pos_15', 'feature_close_pos_30', 'feature_close_pos_60',
    'feature_dist_to_high_15', 'feature_dist_to_low_15',
    'feature_dist_to_high_60', 'feature_dist_to_low_60',

    # Bar shape
    'feature_body_pct', 'feature_close_location_in_bar',
    'feature_upper_wick_pct', 'feature_lower_wick_pct',

    # Volume / volatility
    'feature_volume_z_20', 'feature_volume_z_60', 'feature_volume_change_5',
    'feature_vol_15', 'feature_vol_60', 'feature_vol_ratio_15_60',
    'feature_range_z_20',

    # VWAP / session state
    'feature_session_vwap_dist', 'feature_dist_rolling_vwap_15',
    'feature_dist_rolling_vwap_60', 'feature_session_open_ret',
    'feature_session_high_dist', 'feature_session_low_dist',
    'feature_minutes_since_session_open', 'feature_minutes_to_session_close',
    'feature_is_rth_open_window', 'feature_is_lunch_window',
    'feature_is_last_hour',

    # EMA / RSI / MACD
    'feature_dist_ema_5', 'feature_dist_ema_15', 'feature_dist_ema_30',
    'feature_dist_ema_60', 'feature_slope_ema_15', 'feature_slope_ema_30',
    'feature_slope_ema_60', 'feature_rsi_14', 'feature_rsi_30',
    'feature_macd', 'feature_macd_signal', 'feature_macd_hist',

    # Time and opening range
    'feature_minute_of_day_sin', 'feature_minute_of_day_cos',
    'feature_day_of_week', 'feature_or15_high_dist',
    'feature_or15_low_dist', 'feature_or15_range_pct',

    # Daily bias / prior session context
    'feature_daily_trend_slope_10', 'feature_daily_ret_1', 'feature_daily_ret_5',
    'feature_dist_to_prior_day_high', 'feature_dist_to_prior_day_low',
    'feature_dist_to_prior_day_close',
]


def load_baseline_feature_names() -> list:
    with open(config.BASELINE_FEATURES_FILE, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data.get('baseline_features', BASELINE_FEATURES)


def _daily_context(df: pl.DataFrame) -> pl.DataFrame:
    if 'session_id' not in df.columns:
        return df
    daily = (
        df.select(['session_id', 'ts_event', 'high', 'low', 'close'])
        .sort('ts_event')
        .group_by('session_id', maintain_order=True)
        .agg([
            pl.col('ts_event').first().alias('_session_start'),
            pl.col('high').max().alias('_session_high'),
            pl.col('low').min().alias('_session_low'),
            pl.col('close').last().alias('_session_close'),
        ])
        .sort('_session_start')
        .with_columns([
            pl.col('_session_high').shift(1).alias('_prior_day_high'),
            pl.col('_session_low').shift(1).alias('_prior_day_low'),
            pl.col('_session_close').shift(1).alias('_prior_day_close'),
            pl.col('_session_close').shift(2).alias('_two_day_close'),
            pl.col('_session_close').shift(6).alias('_six_day_close'),
            pl.col('_session_close').shift(11).alias('_eleven_day_close'),
        ])
        .select([
            'session_id', '_prior_day_high', '_prior_day_low', '_prior_day_close',
            '_two_day_close', '_six_day_close', '_eleven_day_close',
        ])
    )
    return df.join(daily, on='session_id', how='left')


def compute_baseline_features(df: pl.DataFrame) -> pl.DataFrame:
    eps = float(getattr(config, 'EPS', 1e-9))
    clip_min = float(getattr(config, 'CLIP_MIN', -10.0))
    clip_max = float(getattr(config, 'CLIP_MAX', 10.0))

    df = _daily_context(df)

    close = pl.col('close').cast(pl.Float64)
    high = pl.col('high').cast(pl.Float64)
    low = pl.col('low').cast(pl.Float64)
    open_ = pl.col('open').cast(pl.Float64)
    volume = pl.col('volume').cast(pl.Float64)
    range_ = (high - low).clip(eps, None)
    body = (close - open_).abs()
    ret_1 = (close / close.shift(1).clip(eps, None)).log()

    exprs = []

    for lag in (1, 3, 5, 15, 30, 60):
        ret = (close / close.shift(lag).clip(eps, None)).log()
        exprs.append(ret.alias(f'feature_ret_{lag}'))

    ret_5 = (close / close.shift(5).clip(eps, None)).log()
    ret_15 = (close / close.shift(15).clip(eps, None)).log()
    ret_30 = (close / close.shift(30).clip(eps, None)).log()
    exprs.extend([
        (ret_5 - ret_15).alias('feature_mom_accel_5_15'),
        (ret_15 - ret_30).alias('feature_mom_accel_15_30'),
    ])

    for w in (15, 30, 60):
        roll_high = high.shift(1).rolling_max(window_size=w, min_periods=max(3, w // 5))
        roll_low = low.shift(1).rolling_min(window_size=w, min_periods=max(3, w // 5))
        roll_range = (roll_high - roll_low).clip(eps, None)
        exprs.append(((close - roll_low) / roll_range).alias(f'feature_close_pos_{w}'))
        if w in (15, 60):
            exprs.append(((close - roll_high) / close.clip(eps, None)).alias(f'feature_dist_to_high_{w}'))
            exprs.append(((close - roll_low) / close.clip(eps, None)).alias(f'feature_dist_to_low_{w}'))

    upper_wick = high - pl.max_horizontal(open_, close)
    lower_wick = pl.min_horizontal(open_, close) - low
    exprs.extend([
        (body / range_).alias('feature_body_pct'),
        ((close - low) / range_).alias('feature_close_location_in_bar'),
        (upper_wick / range_).alias('feature_upper_wick_pct'),
        (lower_wick / range_).alias('feature_lower_wick_pct'),
    ])

    for w in (20, 60):
        vol_mean = volume.shift(1).rolling_mean(window_size=w, min_periods=max(5, w // 4))
        vol_std = volume.shift(1).rolling_std(window_size=w, min_periods=max(5, w // 4)).clip(eps, None)
        exprs.append(((volume - vol_mean) / vol_std).alias(f'feature_volume_z_{w}'))
    exprs.append((volume / volume.shift(5).clip(eps, None) - 1.0).alias('feature_volume_change_5'))

    vol_15 = ret_1.shift(1).rolling_std(window_size=15, min_periods=5)
    vol_60 = ret_1.shift(1).rolling_std(window_size=60, min_periods=15)
    exprs.extend([
        vol_15.alias('feature_vol_15'),
        vol_60.alias('feature_vol_60'),
        (vol_15 / vol_60.clip(eps, None)).alias('feature_vol_ratio_15_60'),
    ])
    bar_range_pct = (high - low) / close.clip(eps, None)
    range_mean = bar_range_pct.shift(1).rolling_mean(window_size=20, min_periods=5)
    range_std = bar_range_pct.shift(1).rolling_std(window_size=20, min_periods=5).clip(eps, None)
    exprs.append(((bar_range_pct - range_mean) / range_std).alias('feature_range_z_20'))

    if 'session_id' in df.columns:
        df = df.with_columns([
            ((close * volume).cum_sum().over('session_id') / volume.cum_sum().over('session_id').clip(eps, None)).alias('_session_vwap'),
            open_.first().over('session_id').alias('_session_open'),
            high.shift(1).cum_max().over('session_id').alias('_session_high_so_far'),
            low.shift(1).cum_min().over('session_id').alias('_session_low_so_far'),
        ])
    else:
        df = df.with_columns([
            close.alias('_session_vwap'), open_.alias('_session_open'),
            high.shift(1).alias('_session_high_so_far'), low.shift(1).alias('_session_low_so_far'),
        ])

    rolling_vwap_15 = (close * volume).rolling_sum(window_size=15, min_periods=5) / volume.rolling_sum(window_size=15, min_periods=5).clip(eps, None)
    rolling_vwap_60 = (close * volume).rolling_sum(window_size=60, min_periods=15) / volume.rolling_sum(window_size=60, min_periods=15).clip(eps, None)
    exprs.extend([
        ((close - pl.col('_session_vwap')) / close.clip(eps, None)).alias('feature_session_vwap_dist'),
        ((close - rolling_vwap_15) / close.clip(eps, None)).alias('feature_dist_rolling_vwap_15'),
        ((close - rolling_vwap_60) / close.clip(eps, None)).alias('feature_dist_rolling_vwap_60'),
        ((close - pl.col('_session_open')) / pl.col('_session_open').clip(eps, None)).alias('feature_session_open_ret'),
        ((close - pl.col('_session_high_so_far')) / close.clip(eps, None)).alias('feature_session_high_dist'),
        ((close - pl.col('_session_low_so_far')) / close.clip(eps, None)).alias('feature_session_low_dist'),
    ])

    for span in (5, 15, 30, 60):
        ema = close.ewm_mean(span=span, adjust=False)
        exprs.append(((close - ema) / close.clip(eps, None)).alias(f'feature_dist_ema_{span}'))
        if span in (15, 30, 60):
            exprs.append(((ema - ema.shift(5)) / (5.0 * ema.shift(5).clip(eps, None))).alias(f'feature_slope_ema_{span}'))

    for w in (14, 30):
        delta = close.diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0.0).rolling_mean(window_size=w, min_periods=max(5, w // 3))
        loss = pl.when(delta < 0).then(-delta).otherwise(0.0).rolling_mean(window_size=w, min_periods=max(5, w // 3))
        rs = gain / loss.clip(eps, None)
        exprs.append((100.0 - (100.0 / (1.0 + rs))).alias(f'feature_rsi_{w}'))

    ema_12 = close.ewm_mean(span=12, adjust=False)
    ema_26 = close.ewm_mean(span=26, adjust=False)
    macd = ema_12 - ema_26
    macd_signal = macd.ewm_mean(span=9, adjust=False)
    exprs.extend([
        (macd / close.clip(eps, None)).alias('feature_macd'),
        (macd_signal / close.clip(eps, None)).alias('feature_macd_signal'),
        ((macd - macd_signal) / close.clip(eps, None)).alias('feature_macd_hist'),
    ])

    local_ts = pl.col('ts_event').dt.convert_time_zone(getattr(config, 'TIMEZONE', 'America/New_York'))
    minute_of_day = local_ts.dt.hour() * 60 + local_ts.dt.minute()
    session_open_min = 18 * 60
    session_close_min = 16 * 60
    since_open = pl.when(minute_of_day >= session_open_min).then(minute_of_day - session_open_min).otherwise(minute_of_day + 24 * 60 - session_open_min)
    to_close = (22 * 60 - since_open).clip(0, 22 * 60)
    exprs.extend([
        since_open.cast(pl.Float64).alias('feature_minutes_since_session_open'),
        to_close.cast(pl.Float64).alias('feature_minutes_to_session_close'),
        ((minute_of_day >= 9 * 60 + 30) & (minute_of_day < 10 * 60)).cast(pl.Float64).alias('feature_is_rth_open_window'),
        ((minute_of_day >= 12 * 60) & (minute_of_day < 13 * 60)).cast(pl.Float64).alias('feature_is_lunch_window'),
        ((minute_of_day >= 15 * 60) & (minute_of_day < session_close_min)).cast(pl.Float64).alias('feature_is_last_hour'),
        (2.0 * np_pi() * minute_of_day.cast(pl.Float64) / 1440.0).sin().alias('feature_minute_of_day_sin'),
        (2.0 * np_pi() * minute_of_day.cast(pl.Float64) / 1440.0).cos().alias('feature_minute_of_day_cos'),
        local_ts.dt.weekday().cast(pl.Float64).alias('feature_day_of_week'),
    ])

    df = df.with_columns(since_open.cast(pl.Int32).alias('_minutes_since_open'))
    if 'session_id' in df.columns:
        df = df.with_columns([
            pl.when(pl.col('_minutes_since_open') < 15).then(high).otherwise(None).max().over('session_id').alias('_or15_high'),
            pl.when(pl.col('_minutes_since_open') < 15).then(low).otherwise(None).min().over('session_id').alias('_or15_low'),
        ])
    else:
        df = df.with_columns([high.alias('_or15_high'), low.alias('_or15_low')])
    or_ready = pl.col('_minutes_since_open') >= 15
    or_range = (pl.col('_or15_high') - pl.col('_or15_low')).clip(eps, None)
    exprs.extend([
        pl.when(or_ready).then((close - pl.col('_or15_high')) / close.clip(eps, None)).otherwise(0.0).alias('feature_or15_high_dist'),
        pl.when(or_ready).then((close - pl.col('_or15_low')) / close.clip(eps, None)).otherwise(0.0).alias('feature_or15_low_dist'),
        pl.when(or_ready).then(or_range / close.clip(eps, None)).otherwise(0.0).alias('feature_or15_range_pct'),
    ])

    if '_prior_day_close' in df.columns:
        exprs.extend([
            ((pl.col('_prior_day_close') / pl.col('_two_day_close').clip(eps, None)).log()).alias('feature_daily_ret_1'),
            ((pl.col('_prior_day_close') / pl.col('_six_day_close').clip(eps, None)).log()).alias('feature_daily_ret_5'),
            ((pl.col('_prior_day_close') - pl.col('_eleven_day_close')) / (10.0 * pl.col('_eleven_day_close').clip(eps, None))).alias('feature_daily_trend_slope_10'),
            ((close - pl.col('_prior_day_high')) / close.clip(eps, None)).alias('feature_dist_to_prior_day_high'),
            ((close - pl.col('_prior_day_low')) / close.clip(eps, None)).alias('feature_dist_to_prior_day_low'),
            ((close - pl.col('_prior_day_close')) / close.clip(eps, None)).alias('feature_dist_to_prior_day_close'),
        ])
    else:
        exprs.extend([pl.lit(0.0).alias(c) for c in (
            'feature_daily_ret_1', 'feature_daily_ret_5', 'feature_daily_trend_slope_10',
            'feature_dist_to_prior_day_high', 'feature_dist_to_prior_day_low', 'feature_dist_to_prior_day_close')])

    df = df.with_columns(exprs)

    feature_cols = [c for c in BASELINE_FEATURES if c in df.columns]
    df = df.with_columns([
        pl.col(c).fill_nan(0.0).fill_null(0.0).clip(clip_min, clip_max).cast(pl.Float32)
        for c in feature_cols
    ])
    temp_cols = [c for c in df.columns if c.startswith('_')]
    if temp_cols:
        df = df.drop(temp_cols)
    return df


def np_pi() -> float:
    return 3.141592653589793
