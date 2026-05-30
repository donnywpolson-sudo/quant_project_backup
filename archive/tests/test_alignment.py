import polars as pl
import pytest
from datetime import datetime, timedelta, time
from pipeline.session.session import add_session_id, filter_session_hours, resample_to_frequency
from pipeline.align.align import align_htf_streams

def make_synthetic_1min(session_start_et, session_end_et):
    ts = []
    current = session_start_et
    while current < session_end_et:
        ts.append(current)
        current += timedelta(minutes=1)
    df = pl.DataFrame({'ts_event': ts, 'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000})
    df = df.with_columns(pl.col('ts_event').dt.replace_time_zone('America/New_York').dt.convert_time_zone('UTC'))
    df = add_session_id(df)
    df = filter_session_hours(df)
    return df

def test_daily_alignment_first_bar():
    start_day0 = datetime(2026, 1, 6, 18, 0)
    end_day0 = datetime(2026, 1, 7, 16, 0)
    start_day1 = datetime(2026, 1, 7, 18, 0)
    end_day1 = datetime(2026, 1, 8, 16, 0)
    df_all = pl.concat([make_synthetic_1min(start_day0, end_day0), make_synthetic_1min(start_day1, end_day1)])
    df_daily = resample_to_frequency(df_all, '1d')
    df_5min = resample_to_frequency(df_all, '5m')
    df_aligned = align_htf_streams(df_5min, pl.DataFrame(), df_daily)
    target = df_aligned.filter((pl.col('ts_event').dt.hour() == 18) & (pl.col('ts_event').dt.minute() == 0) & (pl.col('ts_event').dt.date() == datetime(2026, 1, 7).date()))
    assert not target.is_empty()
    daily_close = target['daily_close'][0]
    daily_vol = target['daily_vol_5'][0]
    assert daily_close is not None and daily_close > 0
    assert daily_vol is not None and daily_vol >= 0
    session_vals = df_aligned.filter(pl.col('ts_event').dt.date() == datetime(2026, 1, 7).date())['daily_close']
    assert session_vals.n_unique() == 1

def test_filter_session_hours_excludes_gap():
    ts = [datetime(2026, 1, 7, 16, 30), datetime(2026, 1, 7, 17, 30), datetime(2026, 1, 7, 18, 30)]
    df = pl.DataFrame({'ts_event': ts, 'open': [100.0] * 3, 'high': [101.0] * 3, 'low': [99.0] * 3, 'close': [100.0] * 3, 'volume': [1000] * 3})
    df = df.with_columns(pl.col('ts_event').dt.replace_time_zone('America/New_York').dt.convert_time_zone('UTC'))
    df = filter_session_hours(df)
    local_hours = df['ts_event'].dt.convert_time_zone('America/New_York').dt.hour().to_list()
    assert 17 not in local_hours
    assert 18 in local_hours


def test_session_offset_consistency():
    """
    Verify that session.py's add_session_id and walkforward.py's
    _resample_to_1h produce identical session_id assignments.
    Both must derive the offset from config.SESSION_START_LOCAL,
    not a hardcoded value.
    """
    from datetime import timezone as tz
    from pipeline.walkforward.walkforward import _resample_to_1h
    from archive.core.config import config

    # Two bars straddling midnight UTC with 18:00 ET session start.
    # Session offset = 24 - 18 = 6h, so bars from 18:00 ET Jan 7
    # through 17:55 ET Jan 8 all map to date 2026-01-08.
    # Generate enough 5-min bars to fill one complete hour (12 bars
    # needed, >10 for the 1H resample filter).
    ts = []
    base = datetime(2026, 1, 7, 23, 0, tzinfo=tz.utc)
    for i in range(14):
        ts.append(base + timedelta(minutes=i * 5))
    df = pl.DataFrame({
        'ts_event': ts,
        'open':  [100.0 + i * 0.1 for i in range(14)],
        'high':  [101.0 + i * 0.1 for i in range(14)],
        'low':   [99.0 + i * 0.1 for i in range(14)],
        'close': [100.5 + i * 0.1 for i in range(14)],
        'volume': [1000] * 14,
    }).with_columns(pl.col('ts_event').cast(pl.Datetime(time_unit='us', time_zone='UTC')))

    df_session = add_session_id(df)
    sid_session = df_session['session_id'].to_list()

    df_hh = _resample_to_1h(df)
    sid_hh = df_hh['session_id'].unique().sort().to_list()

    assert sid_session[0] == sid_session[-1], \
        f'bars in same session should share session_id: first={sid_session[0]}, last={sid_session[-1]}'
    assert sid_session[0] == '2026-01-08', \
        f'session_id should be 2026-01-08 (18:00 ET start + 6h offset), got {sid_session[0]}'
    assert len(sid_hh) > 0 and sid_hh[0] == '2026-01-08', \
        f'1H session_id should be 2026-01-08, got: {sid_hh}'


def test_session_offset_non_hour_start():
    """Canonical session offset must preserve minute component."""
    from archive.core.config import config
    from pipeline.session.session import session_start_offset_by

    saved = config.SESSION_START_LOCAL
    try:
        config.SESSION_START_LOCAL = time(18, 30)
        assert session_start_offset_by() == '5h30m'
    finally:
        config.SESSION_START_LOCAL = saved
