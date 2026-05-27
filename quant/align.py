import polars as pl

_OHLCV_COLS = {'open', 'high', 'low', 'close', 'volume', 'session_id'}

def align_htf_streams(df_5min: pl.DataFrame, df_1h: pl.DataFrame, df_daily: pl.DataFrame) -> pl.DataFrame:
    df_5min = df_5min.sort('ts_event')
    if df_1h is not None and (not df_1h.is_empty()):
        df_1h = df_1h.sort('ts_event')
        renames_1h = {c: '1h_' + c for c in df_1h.columns if c in _OHLCV_COLS}
        if renames_1h:
            df_1h = df_1h.rename(renames_1h)
        df_5min = df_5min.join_asof(df_1h, on='ts_event', strategy='backward')
    if df_daily is not None and (not df_daily.is_empty()):
        df_daily = df_daily.sort('ts_event')
        renames_daily = {c: 'daily_' + c for c in df_daily.columns if c in _OHLCV_COLS}
        if renames_daily:
            df_daily = df_daily.rename(renames_daily)
        df_5min = df_5min.join_asof(df_daily, on='ts_event', strategy='backward')
    return df_5min