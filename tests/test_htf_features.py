import polars as pl
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from _legacy.ingest import load_and_clean_data
from pipeline.features.engine import generate_features
from core.config import config

def make_synthetic_with_trend():
    start_dt = datetime(2026, 1, 1, 18, 0)
    end_dt = datetime(2026, 2, 1, 16, 0)
    ts = []
    cur = start_dt
    while cur < end_dt:
        ts.append(cur)
        cur += timedelta(minutes=1)
    df = pl.DataFrame({'ts_event': ts})
    df = df.with_columns(pl.col('ts_event').dt.replace_time_zone('America/New_York').dt.convert_time_zone('UTC'))
    n = df.height
    base = 100 + np.arange(n) * 0.01
    noise = np.random.normal(0, 0.05, n)
    price = base + noise
    df = df.with_columns([pl.Series('open', price), pl.Series('high', price + 0.2), pl.Series('low', price - 0.2), pl.Series('close', price + 0.1), pl.Series('volume', np.full(n, 1000))])
    return df

def test_htf_features_synthetic():
    config.ENABLE_EXPANSION = True
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / 'ES' / 'synthetic.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        df_raw = make_synthetic_with_trend()
        df_raw.write_parquet(path)
        df = load_and_clean_data(str(path))
        df_feat = generate_features(df)
        daily_vol = df_feat['htf_daily_vol_5'].drop_nulls()
        daily_trend = df_feat['htf_daily_trend_slope_10'].drop_nulls()
        assert daily_vol.len() > 0
        assert (daily_vol > 0).any()
        assert daily_trend.len() > 0
        assert daily_trend.mean() > 0
