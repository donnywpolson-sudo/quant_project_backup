"""
src/features/baseline.py
Generate baseline features from YAML spec, plus Keltner Channels and Elder's Ray.
All features are past-only and return float32.
"""
import polars as pl
import yaml
from config import config

def load_baseline_feature_names() -> list:
    """Load feature names from baseline_features.yaml"""
    with open(config.BASELINE_FEATURES_FILE, 'r') as f:
        data = yaml.safe_load(f)
    return data['baseline_features']

def compute_baseline_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add all baseline features to the DataFrame.
    """
    close = pl.col("close").cast(pl.Float32)
    high = pl.col("high").cast(pl.Float32)
    low = pl.col("low").cast(pl.Float32)
    open_ = pl.col("open").cast(pl.Float32)
    volume = pl.col("volume").cast(pl.Float32)

    exprs = []

    # 1-4: Log returns at lags 1,5,10,20
    for lag in [1, 5, 10, 20]:
        ret = (close / close.shift(lag)).log()
        exprs.append(ret.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f"feature_ret_{lag}"))

    # 5-7: Simple moving averages of close
    for window in [5, 20, 50]:
        ma = close.rolling_mean(window_size=window)
        exprs.append(ma.alias(f"feature_ma_{window}"))

    # 8: dist_ma_20 = (close - MA20)/MA20
    ma20 = close.rolling_mean(window_size=20)
    dist_ma20 = (close - ma20) / ma20.clip(config.EPS, None)
    exprs.append(dist_ma20.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_dist_ma_20"))

    # 9: dist_ma_50
    ma50 = close.rolling_mean(window_size=50)
    dist_ma50 = (close - ma50) / ma50.clip(config.EPS, None)
    exprs.append(dist_ma50.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_dist_ma_50"))

    # 10: ma_slope_20
    slope20 = (close - close.shift(20)) / 20.0 / ma20.clip(config.EPS, None)
    exprs.append(slope20.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_ma_slope_20"))

    # 11,12: price_z_20, price_z_50
    for window in [20, 50]:
        mean = close.rolling_mean(window_size=window)
        std = close.rolling_std(window_size=window)
        z = (close - mean) / std.clip(config.EPS, None)
        exprs.append(z.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f"feature_price_z_{window}"))

    # 13: high_low_range_norm
    range_norm = (high - low) / pl.max_horizontal(close, config.EPS)
    exprs.append(range_norm.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_high_low_range_norm"))

    # 14: true_range
    prev_close = close.shift(1)
    tr = pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    )
    exprs.append(tr.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_true_range"))

    # 15: atr_14
    atr14 = tr.rolling_mean(window_size=14)
    exprs.append(atr14.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_atr_14"))

    # 16,17: realized_vol_5, realized_vol_20
    ret_1 = (close / close.shift(1)).log()
    for window in [5, 20]:
        rvol = ret_1.rolling_std(window_size=window)
        exprs.append(rvol.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f"feature_realized_vol_{window}"))

    # 18: ewma_vol_20
    alpha = 2.0 / (20 + 1)
    ewma_vol = ret_1.pow(2).ewm_mean(alpha=alpha, adjust=False).sqrt()
    exprs.append(ewma_vol.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_ewma_vol_20"))

    # 19,20: price_momentum_5, price_momentum_10
    for window in [5, 10]:
        mom = (close - close.shift(window)) / close.shift(window).clip(config.EPS, None)
        exprs.append(mom.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f"feature_price_momentum_{window}"))

    # 21,22: mom_z_5, mom_z_10
    for window in [5, 10]:
        mom = (close - close.shift(window)) / close.shift(window).clip(config.EPS, None)
        mean_mom = mom.rolling_mean(window_size=window)
        std_mom = mom.rolling_std(window_size=window)
        mom_z = (mom - mean_mom) / std_mom.clip(config.EPS, None)
        exprs.append(mom_z.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f"feature_mom_z_{window}"))

    # 23: rsi_14
    delta = close.diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)
    avg_gain = gain.rolling_mean(window_size=14)
    avg_loss = loss.rolling_mean(window_size=14)
    rs = avg_gain / avg_loss.clip(config.EPS, None)
    rsi = 100 - 100 / (1 + rs)
    exprs.append(rsi.clip(0, 100).alias("feature_rsi_14"))

    # 24: macd
    ema12 = close.ewm_mean(alpha=2/13, adjust=False)
    ema26 = close.ewm_mean(alpha=2/27, adjust=False)
    macd = ema12 - ema26
    exprs.append(macd.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_macd"))

    # 25: macd_signal
    signal = macd.ewm_mean(alpha=2/10, adjust=False)
    exprs.append(signal.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_macd_signal"))

    # 26: stoch_k
    low14 = low.rolling_min(window_size=14)
    high14 = high.rolling_max(window_size=14)
    stoch_k = (close - low14) / (high14 - low14).clip(config.EPS, None) * 100
    exprs.append(stoch_k.clip(0, 100).alias("feature_stoch_k"))

    # 27: log_volume
    log_vol = volume.log().fill_null(0.0)
    exprs.append(log_vol.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_log_volume"))

    # 28: volume_z_20
    vol_mean = volume.rolling_mean(window_size=20)
    vol_std = volume.rolling_std(window_size=20)
    vol_z = (volume - vol_mean) / vol_std.clip(config.EPS, None)
    exprs.append(vol_z.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_volume_z_20"))

    # 29: obv
    sign = pl.when(close > close.shift(1)).then(1).when(close < close.shift(1)).then(-1).otherwise(0)
    obv = (sign * volume).cum_sum()
    exprs.append(obv.cast(pl.Float32).clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_obv"))

    # 30: signed_bar_strength
    bar_sign = (close - open_).sign()
    bar_sign = bar_sign.fill_null(strategy="forward")
    signed_volume = bar_sign * volume
    signed_strength = signed_volume / volume.clip(config.EPS, None)
    exprs.append(signed_strength.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_signed_bar_strength"))

    # 31: volume_price_divergence
    vol_price_div = (volume * ret_1).cast(pl.Float32)
    exprs.append(vol_price_div.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_volume_price_divergence"))

    # 32: spread_proxy
    spread_proxy = (high - low) / close.clip(config.EPS, None)
    exprs.append(spread_proxy.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_spread_proxy"))

    # 33: session_pos
    session_pos = (pl.col("ts_event").rank("ordinal").over("session_id") - 1) / (pl.col("ts_event").count().over("session_id") - 1)
    exprs.append(session_pos.fill_nan(0.5).cast(pl.Float32).alias("feature_session_pos"))

    # 34: time_of_day_bucket
    bucket = pl.when(session_pos < 0.33).then(0.0).when(session_pos < 0.66).then(1.0).otherwise(2.0)
    exprs.append(bucket.cast(pl.Float32).alias("feature_time_of_day_bucket"))

    # 35: 1h_bias placeholder
    exprs.append(pl.lit(0.0).alias("feature_1h_bias"))

    # 36: session_volatility
    session_vol = ret_1.rolling_std(window_size=config.ROLL_WINDOW_MIN_ROWS).over("session_id")
    exprs.append(session_vol.fill_null(0.0).clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_session_volatility"))

    # 37-40: placeholders
    exprs.append(pl.lit(0.0).alias("feature_pair_prod_template"))
    exprs.append(pl.lit(0.0).alias("feature_ratio_template"))
    exprs.append(pl.lit(0.0).alias("feature_pca_comp_1"))
    exprs.append(pl.lit(0.0).alias("feature_pca_comp_2"))

    # ---------- ADDITIONAL INDICATORS ----------
    # Keltner Channels (20-period EMA, 2*ATR bands)
    close_ema = close.ewm_mean(span=20, adjust=False)
    atr = atr14  # already computed
    keltner_upper = close_ema + 2 * atr
    keltner_lower = close_ema - 2 * atr
    keltner_width = (keltner_upper - keltner_lower) / close_ema.clip(config.EPS, None)
    exprs.append(keltner_width.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_keltner_width"))
    dist_to_upper = (keltner_upper - close) / close.clip(config.EPS, None)
    dist_to_lower = (close - keltner_lower) / close.clip(config.EPS, None)
    exprs.append(dist_to_upper.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_dist_to_keltner_upper"))
    exprs.append(dist_to_lower.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_dist_to_keltner_lower"))

    # Elder's Ray: Bull Power = high - EMA(close), Bear Power = low - EMA(close)
    bull_power = high - close_ema
    bear_power = low - close_ema
    exprs.append(bull_power.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_bull_power"))
    exprs.append(bear_power.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_bear_power"))

    # Apply all expressions
    df = df.with_columns(exprs)

    # Clean NaN/Inf and clip
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    for col in feature_cols:
        df = df.with_columns(
            pl.col(col).fill_nan(config.REPLACE_INF_NAN_WITH).fill_null(config.REPLACE_INF_NAN_WITH).clip(config.CLIP_MIN, config.CLIP_MAX)
        )
    return df