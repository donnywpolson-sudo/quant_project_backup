import polars as pl
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime
from config import config
from quant.ingest import load_and_clean_data
from quant.features.engine import generate_features
from quant.features.htf_context import add_htf_context_features

def make_synthetic_with_trend():
    pass
    start_dt = datetime(2026, 1, 1, 18, 0)
    end_dt = datetime(2026, 1, 11, 16, 0)
    ts = []
    cur = start_dt
    from datetime import timedelta
    while cur < end_dt:
        ts.append(cur)
        cur += timedelta(minutes=1)
    df = pl.DataFrame({'ts_event': ts})
    df = df.with_columns(pl.col('ts_event').dt.replace_time_zone('America/New_York').dt.convert_time_zone('UTC'))
    df = df.with_columns(pl.col('ts_event').dt.convert_time_zone('America/New_York').dt.hour().alias('hour'))
    df = df.filter((pl.col('hour') >= 18) | (pl.col('hour') < 16))
    df = df.drop('hour').sort('ts_event')
    n = df.height
    base_price = 100.0 + np.arange(n) * 0.01
    np.random.seed(42)
    noise = np.random.normal(loc=0.0, scale=0.05, size=n)
    base_price = base_price + noise
    df = df.with_columns([pl.Series('open', base_price), pl.Series('high', base_price + 0.2), pl.Series('low', base_price - 0.2), pl.Series('close', base_price + 0.1), pl.Series('volume', np.full(n, 1000))])
    return df

def test_htf_features_synthetic():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / 'synthetic.parquet'
        df_raw = make_synthetic_with_trend()
        df_raw.write_parquet(data_path)
        df_aligned = load_and_clean_data(str(data_path))
        df_features = generate_features(df_aligned)
        daily_vol = df_features['htf_daily_vol_5']
        daily_trend = df_features['htf_daily_trend_slope_10']
        assert (daily_vol.drop_nulls() > 0).any()
        assert (daily_trend.drop_nulls() > 0).any()
        print('Daily vol (first non-null):', daily_vol.drop_nulls().head(1)[0])
        print('Daily trend slope (first non-null):', daily_trend.drop_nulls().head(1)[0])
if __name__ == '__main__':
    test_htf_features_synthetic()