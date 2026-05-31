import logging

import polars as pl

from pipeline.common.config import config

logger = logging.getLogger(__name__)


def add_target_15m(df: pl.DataFrame) -> pl.DataFrame:
    """
    Primary supervised label.

    On 1-minute bars, signal at bar t enters at open[t+1] and exits at
    close[t+15].
    That makes the label executable under execute_at=open[t+1] and avoids
    close[t] -> open[t+1] gap leakage.
    """
    horizon = int(getattr(config, "TARGET_15M_HORIZON", 15))
    eps = getattr(config, "EPS", 1e-9)
    scale = getattr(config, "TARGET_SCALE_FACTOR", 100.0)
    clip_min = getattr(config, "CLIP_MIN", -10.0)
    clip_max = getattr(config, "CLIP_MAX", 10.0)
    exit_close = pl.col("close").shift(-horizon)
    entry_open = pl.col("open").shift(-1)
    forward_ret_raw = (exit_close - entry_open) / entry_open.clip(eps, None)
    return df.with_columns(
        [
            (forward_ret_raw * scale)
            .clip(clip_min, clip_max)
            .alias("target_15m_return"),
            (forward_ret_raw > 0).cast(pl.Int8).alias("target_sign_15m"),
        ]
    )


def add_target_daily_regime(df: pl.DataFrame) -> pl.DataFrame:
    """
    Daily context/filter column, not the model label.

    Uses existing causal HTF daily trend if present. If HTF expansion is off,
    emit neutral regime so downstream schemas stay stable.
    """
    if "htf_daily_trend_slope_10" in df.columns:
        regime = (
            pl.when(pl.col("htf_daily_trend_slope_10") > 0)
            .then(1)
            .when(pl.col("htf_daily_trend_slope_10") < 0)
            .then(-1)
            .otherwise(0)
        )
    else:
        regime = pl.lit(0)
    return df.with_columns(regime.cast(pl.Int8).alias("target_daily_regime"))


def drop_incomplete_target(df: pl.DataFrame) -> pl.DataFrame:
    filter_expr = None
    for target_col in ("target_15m_return", "target_sign_15m"):
        if target_col in df.columns:
            null_count = df[target_col].null_count()
            if null_count == df.height:
                logger.warning(
                    "[TARGET] %s is entirely null (%d/%d rows) - skipping",
                    target_col,
                    null_count,
                    df.height,
                )
                continue
            col_filter = pl.col(target_col).is_not_null()
            filter_expr = col_filter if filter_expr is None else filter_expr & col_filter
    if filter_expr is None:
        return df
    before = df.height
    df = df.filter(filter_expr)
    if df.height == 0 and before > 0:
        raise RuntimeError(
            "DROPNULL KILL: drop_incomplete_target collapsed %d rows to 0. "
            "Check target columns for full-null cascade." % before
        )
    return df
