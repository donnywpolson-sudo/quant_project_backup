import polars as pl
import pytest
import tempfile
from pathlib import Path
import pytz
from datetime import datetime, timedelta
from quant.session import add_session_id, filter_session_hours, resample_to_frequency
from quant.align import align_htf_streams
TZ = pytz.timezone('America/New_York')

def make_synthetic_1min(session_start_et, session_end_et, daily_close_prev, daily_close_curr):
    pass
    start_dt = datetime.combine(session_start_et.date(), session_start_et.time())
    end_dt = datetime.combine(session_end_et.date(), session_end_et.time())
    ts = []
    current = start_dt
    while current < end_dt:
        ts.append(current)
        current += timedelta(minutes=1)
    df = pl.DataFrame({'ts_event': ts, 'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000})
    df = df.with_columns(pl.col('ts_event').dt.replace_time_zone('America/New_York').dt.convert_time_zone('UTC'))
    df = add_session_id(df)
    df = filter_session_hours(df)
    return df

def test_daily_alignment_first_bar():
    pass
    start_day0 = datetime(2026, 1, 6, 18, 0)
    end_day0 = datetime(2026, 1, 7, 16, 0)
    start_day1 = datetime(2026, 1, 7, 18, 0)
    end_day1 = datetime(2026, 1, 8, 16, 0)
    df0 = make_synthetic_1min(start_day0, end_day0, None, None)
    df1 = make_synthetic_1min(start_day1, end_day1, None, None)
    df_all = pl.concat([df0, df1])
    df_daily = resample_to_frequency(df_all, '1d')
    df_5min = resample_to_frequency(df_all, '5m')
    df_aligned = align_htf_streams(df_5min, pl.DataFrame(), df_daily)
    first_bar_second_session = df_aligned.filter(pl.col('ts_event').dt.hour() == 18, pl.col('ts_event').dt.minute() == 0, pl.col('ts_event').dt.date() == datetime(2026, 1, 7).date()).head(1)
    assert not first_bar_second_session.is_empty()
    daily_close = first_bar_second_session['daily_close'][0]
    assert daily_close is not None
    daily_vol = first_bar_second_session['daily_vol_5'][0]
    assert daily_vol is not None

def test_filter_session_hours_excludes_cme_settlement_gap():
    ts = [datetime(2026, 1, 7, 16, 30), datetime(2026, 1, 7, 17, 30), datetime(2026, 1, 7, 18, 30)]
    df = pl.DataFrame({'ts_event': ts, 'open': [100.0, 100.0, 100.0], 'high': [101.0, 101.0, 101.0], 'low': [99.0, 99.0, 99.0], 'close': [100.0, 100.0, 100.0], 'volume': [1000, 1000, 1000]})
    df = df.with_columns(pl.col('ts_event').dt.replace_time_zone('America/New_York').dt.convert_time_zone('UTC'))
    df = filter_session_hours(df)
    local_hours = df['ts_event'].dt.convert_time_zone('America/New_York').dt.hour().to_list()
    assert 17 not in local_hours
    assert 18 in local_hours
if __name__ == '__main__':
    pytest.main([__file__])