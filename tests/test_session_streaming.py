import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
import numpy as np
from quant.session import load_all_streams_chunked

def make_small_synthetic(start_dt: datetime, end_dt: datetime) -> pl.DataFrame:
    ts = []
    cur = start_dt
    while cur < end_dt:
        ts.append(cur)
        cur += timedelta(minutes=1)
    df = pl.DataFrame({'ts_event': ts})
    df = df.with_columns(pl.col('ts_event').dt.replace_time_zone('America/New_York').dt.convert_time_zone('UTC'))
    n = df.height
    prices = 100.0 + np.arange(n) * 0.01
    df = df.with_columns([pl.Series('open', prices).cast(pl.Float32), pl.Series('high', [p + 0.2 for p in prices]).cast(pl.Float32), pl.Series('low', [p - 0.2 for p in prices]).cast(pl.Float32), pl.Series('close', [p + 0.1 for p in prices]).cast(pl.Float32), pl.Series('volume', [1000] * n).cast(pl.Int64)])
    return df

def test_streaming_resample_produces_htf():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'synth.parquet'
        start = datetime(2026, 1, 1, 18, 0)
        end = datetime(2026, 1, 4, 16, 0)
        df = make_small_synthetic(start, end)
        df.write_parquet(path)
        streams = load_all_streams_chunked(str(path.parent / '*.parquet'))
        assert '5m' in streams and '1h' in streams and ('1d' in streams)
        df_1h = streams['1h']
        df_1d = streams['1d']
        assert df_1h.height > 0
        assert df_1d.height > 0
        assert 'daily_vol_5' in df_1d.columns