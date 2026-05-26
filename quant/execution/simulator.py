import polars as pl
import numpy as np
from config import config

def simulate_execution_classification(df: pl.DataFrame) -> pl.DataFrame:
    signal_expr = pl.when(pl.col('prediction_prob').fill_null(0.5) > 0.55).then(1.0).when(pl.col('prediction_prob').fill_null(0.5) < 0.45).then(-1.0).otherwise(0.0)
    df = df.with_columns(signal_expr.alias('raw_signal'))
    df = df.with_columns(pl.col('ts_event').dt.time().alias('t_local'))
    df = df.with_columns(pl.when(pl.col('t_local') >= pl.lit(config.SESSION_END_LOCAL)).then(0.0).otherwise(pl.col('raw_signal')).alias('target_exec'))
    df = df.drop('t_local')
    ret = (pl.col('close') / pl.col('close').shift(1)).log()
    vol = ret.rolling_std(window_size=20).clip(config.EPS, None)
    df = df.with_columns(vol.fill_null(1e-06).alias('vol'))
    spread = (pl.col('high') - pl.col('low')) / pl.col('close').clip(config.EPS, None)
    df = df.with_columns(spread.alias('spread'))
    unit_cost = config.COMMISSION_PER_TRADE + config.SLIPPAGE_K * pl.col('spread') + config.VOL_PENALTY * pl.col('vol')
    open_next = pl.col('open').shift(-1)
    close_next = pl.col('close').shift(-1)
    ret_exec = ((close_next - open_next) / open_next.clip(config.EPS, None)).fill_null(0)
    df = df.with_columns([unit_cost.alias('cost'), ret_exec.alias('ret_exec')])
    df = df.with_columns((pl.col('target_exec') * pl.col('ret_exec') - pl.col('cost')).fill_nan(0).alias('pnl'))
    df = df.with_columns(pl.col('target_exec').alias('position'))
    return df

def simulate_execution(df: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError('Use simulate_execution_classification for new pipeline.')