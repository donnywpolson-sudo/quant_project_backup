import importlib
import polars as pl
import logging
from quant.config_manager import config

logger = logging.getLogger(__name__)

_tb = importlib.import_module("pipeline.03_engineering.triple_barrier")
add_triple_barrier_target = _tb.add_triple_barrier_target
_ml = importlib.import_module("pipeline.03_engineering.meta_label")
add_meta_label_target = _ml.add_meta_label_target

def add_target_5m(df: pl.DataFrame) -> pl.DataFrame:
    horizon = config.TARGET_5M_HORIZON
    # Execution-aligned target: enter at open[t+1], exit at close[t+1]
    # This is the exact return the simulator realizes (before costs).
    # Previously used close-to-close log return, which included the
    # un-capturable close[t] -> open[t+1] gap — a structural mismatch
    # between what the model predicted and what the execution engine traded.
    close_next = pl.col('close').shift(-horizon)
    open_next = pl.col('open').shift(-horizon)
    forward_ret_raw = (close_next - open_next) / open_next.clip(config.EPS, None)
    df = df.with_columns([
        (forward_ret_raw * config.TARGET_SCALE_FACTOR).clip(config.CLIP_MIN, config.CLIP_MAX).alias('target_5m'),
        (forward_ret_raw > 0).cast(pl.Int8).alias('target_sign')
    ])
    return df

def drop_incomplete_target(df: pl.DataFrame) -> pl.DataFrame:
    filter_expr = None
    for target_col in ('target_5m', 'target_4h', 'target_1h', 'target_tb'):
        if target_col in df.columns:
            null_count = df[target_col].null_count()
            if null_count == df.height:
                logger.warning('[TARGET] %s is entirely null (%d/%d rows) — skipping',
                               target_col, null_count, df.height)
                continue
            col_filter = pl.col(target_col).is_not_null()
            filter_expr = col_filter if filter_expr is None else filter_expr & col_filter
    if filter_expr is None:
        return df
    before = df.height
    df = df.filter(filter_expr)
    after = df.height
    if after == 0 and before > 0:
        raise RuntimeError(
            'DROPNULL KILL: drop_incomplete_target collapsed %d rows to 0. '
            'Check target columns for full-null cascade.' % before
        )
    return df

def add_target_1h(df: pl.DataFrame) -> pl.DataFrame:
    if '1h_ts_event' not in df.columns or '1h_close' not in df.columns:
        return df
    null_count = df['1h_ts_event'].null_count()
    if null_count == df.height:
        logger.warning('[TARGET] 1h_ts_event fully null (%d rows) — skipping target_1h', df.height)
        return df
    one_h = df.select(['1h_ts_event', '1h_close']).drop_nulls('1h_ts_event').unique(subset=['1h_ts_event']).sort('1h_ts_event')
    if one_h.height == 0:
        logger.warning('[TARGET] no valid 1h rows after drop_nulls — skipping target_1h')
        return df
    one_h = one_h.with_columns((pl.col('1h_close').shift(-1).log() - pl.col('1h_close').log()).alias('forward_ret_1h_raw'))
    df = df.join(one_h.select(['1h_ts_event', (pl.col('forward_ret_1h_raw') * config.TARGET_SCALE_FACTOR).clip(config.CLIP_MIN, config.CLIP_MAX).alias('target_1h'), (pl.col('forward_ret_1h_raw') > 0).cast(pl.Int8).alias('target_sign_1h')]), on='1h_ts_event', how='left')
    return df

def add_target_4h(df: pl.DataFrame) -> pl.DataFrame:
    H_BARS = int(4 * 60 / 5)
    log_close = pl.col('close').log()
    forward_ret_raw = log_close.shift(-H_BARS) - log_close
    df = df.with_columns([(forward_ret_raw * config.TARGET_SCALE_FACTOR).clip(config.CLIP_MIN, config.CLIP_MAX).alias('target_4h'), (forward_ret_raw > 0).cast(pl.Int8).alias('target_sign_4h')])
    return df