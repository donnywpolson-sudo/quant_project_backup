import polars as pl

def align_htf_streams(df_5min: pl.DataFrame, df_1h: pl.DataFrame, df_daily: pl.DataFrame) -> pl.DataFrame:
    df_5min = df_5min.sort('ts_event')
    if df_1h is not None and (not df_1h.is_empty()):
        df_1h = df_1h.sort('ts_event')
        df_5min = df_5min.join_asof(df_1h.rename({'ts_event': 'ts_event'}), on='ts_event', strategy='backward')
    if df_daily is not None and (not df_daily.is_empty()):
        df_daily = df_daily.sort('ts_event')
        df_5min = df_5min.join_asof(df_daily.rename({'ts_event': 'ts_event'}), on='ts_event', strategy='backward')
    return df_5min