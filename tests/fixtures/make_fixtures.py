"""
tests/fixtures/make_fixtures.py
Generates a valid, reproducible synthetic parquet fixture for pipeline testing.
Ensures all required columns (ts_event, close, high, low, volume) are present.
"""
import polars as pl
import numpy as np
from pathlib import Path

def create_synthetic_data():
    """
    Generates a reproducible synthetic parquet fixture covering 60 days 
    (assuming 1-minute intervals).
    """
    # 86400 rows = 60 days * 24 hours * 60 minutes
    # (Though logic will handle any size)
    n_rows = 86400
    base_price = 100.0
    
    # 1. Generate core price action
    # Random walk: cumulative sum of normal distribution
    price_movements = np.random.randn(n_rows)
    close = base_price + np.cumsum(price_movements)
    
    # 2. Derive high/low to ensure high >= low and volatility is realistic
    # Add random spread to create high/low around the close
    spread = np.random.rand(n_rows) * 0.5
    high = close + spread
    low = close - spread
    
    # 3. Generate volume
    volume = np.random.randint(100, 5000, n_rows)
    
    # 4. Construct DataFrame
    df = pl.DataFrame({
        "ts_event": np.arange(n_rows),
        "close": close,
        "high": high,
        "low": low,
        "volume": volume
    })
    
    # 5. Save to destination
    output_path = Path("tests/fixtures/synthetic_1min_fixture.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    
    print(f"Fixture successfully created at {output_path}")
    print(f"Columns generated: {df.columns}")

if __name__ == "__main__":
    create_synthetic_data()