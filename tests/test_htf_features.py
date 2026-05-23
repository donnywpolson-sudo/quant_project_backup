import polars as pl
import numpy as np
from pathlib import Path
from config import config
from src.ingest import load_and_clean_data
from src.features.engine import generate_features
from src.features.htf_context import add_htf_context_features

def make_synthetic_with_trend():
    """Create 10 days of 1-min data with a clear upward trend."""
    dates = pl.date_range(
        start=datetime(2026, 1, 1, 18, 0),
        end=datetime(2026, 1, 11, 16, 0),
        interval="1m",
        time_zone="America/New_York"
    )
    # Keep only session hours (18:00 to next day 16:00)
    df = pl.DataFrame({"ts_event": dates})
    df = df.with_columns(pl.col("ts_event").dt.hour().alias("hour"))
    df = df.filter((pl.col("hour") >= 18) | (pl.col("hour") < 16))
    df = df.drop("hour").sort("ts_event")
    # Add synthetic price: start at 100, increase by 0.01 per minute (upward trend)
    n = df.height
    base_price = 100.0 + np.arange(n) * 0.01
    df = df.with_columns([
        pl.Series("open", base_price),
        pl.Series("high", base_price + 0.2),
        pl.Series("low", base_price - 0.2),
        pl.Series("close", base_price + 0.1),
        pl.Series("volume", 1000),
    ])
    return df

def test_htf_features_synthetic():
    # Write synthetic to temp parquet
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "synthetic.parquet"
        df_raw = make_synthetic_with_trend()
        df_raw.write_parquet(data_path)
        # Load and clean (resample, align)
        df_aligned = load_and_clean_data(str(data_path))
        # Generate features (includes HTF context)
        df_features = generate_features(df_aligned)
        # Check daily_vol_5 and daily_trend_slope_10
        daily_vol = df_features["htf_daily_vol_5"]
        daily_trend = df_features["htf_daily_trend_slope_10"]
        # Volatility should be positive and trend slope positive (since price increases)
        assert (daily_vol.drop_nulls() > 0).all()
        assert (daily_trend.drop_nulls() > 0).all()
        # The first few bars may have nulls; after enough daily bars, values become stable
        print("Daily vol (first non-null):", daily_vol.drop_nulls().head(1)[0])
        print("Daily trend slope (first non-null):", daily_trend.drop_nulls().head(1)[0])

if __name__ == "__main__":
    test_htf_features_synthetic()