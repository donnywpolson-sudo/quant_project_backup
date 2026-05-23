"""
tests/fixtures/make_fixtures.py
Generates a valid, reproducible synthetic 1‑min OHLCV parquet fixture.
Uses real UTC timestamps (via pytz), includes open/high/low/close/volume.
"""
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz

def create_synthetic_data():
    """
    Creates 60 days of 1‑min data starting from 2020-01-01 00:00 UTC.
    Includes realistic price action and volume.
    """
    # Use pytz for reliable UTC timezone
    utc = pytz.UTC
    start = datetime(2020, 1, 1, 0, 0, tzinfo=utc)
    # 60 days * 24h * 60min = 86400 rows
    n_rows = 86400
    timestamps = [start + timedelta(minutes=i) for i in range(n_rows)]

    base_price = 100.0
    # Random walk
    returns = np.random.normal(0, 0.0001, n_rows)
    close = base_price * np.exp(np.cumsum(returns))
    # Add spread
    spread = np.random.uniform(0.001, 0.005, n_rows) * close
    high = close + spread
    low = close - spread
    # Open is previous close (except first bar)
    open_price = np.roll(close, 1)
    open_price[0] = close[0] - spread[0]
    # Volume
    volume = np.random.randint(100, 5000, n_rows)

    df = pl.DataFrame({
        "ts_event": timestamps,
        "open": open_price.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": volume.astype(np.int64),
    })

    output_path = Path("tests/fixtures/synthetic_1min_fixture.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    print(f"Fixture created at {output_path}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    create_synthetic_data()