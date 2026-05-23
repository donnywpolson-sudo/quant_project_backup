"""
src/session.py
Implements Globex session definition, session_id, and resampling to multiple frequencies (5m, 1h, 1d).
Now supports streaming for large aggregations and returns all three streams with early float32.
Uses ThreadPoolExecutor to process frequencies in parallel for speed.
Temporary directories are automatically cleaned up after each frequency.
"""
import polars as pl
import logging
from datetime import time
import pytz
from pathlib import Path
import tempfile
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import config

logger = logging.getLogger(__name__)

TZ = pytz.timezone(config.TIMEZONE)
SESSION_START = config.SESSION_START_LOCAL
SESSION_END = config.SESSION_END_LOCAL


def add_session_id(df: pl.DataFrame) -> pl.DataFrame:
    """Add session_id using Globex rollover rule: shift by 6h for dates."""
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone(config.TIMEZONE).alias("ts_local")
    )
    session_id = pl.col("ts_local").dt.offset_by("6h").dt.date().cast(pl.String)
    df = df.with_columns(session_id.alias("session_id"))
    return df.drop("ts_local")


def filter_session_hours(df: pl.DataFrame) -> pl.DataFrame:
    """Keep rows within [18:00, 16:00) ET."""
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone(config.TIMEZONE).dt.time().alias("time_local")
    )
    df = df.filter(
        (pl.col("time_local") >= SESSION_START) | (pl.col("time_local") < SESSION_END)
    )
    return df.drop("time_local")


def resample_to_frequency(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    """
    Resample 1‑min df to given frequency (e.g., '5m', '1h', '1d') within each session.
    For 1h, require at least 45 minutes of ticks; for 1d, require at least 360 minutes (6 hours).
    For daily, also compute rolling 5-day volatility of log returns.
    Returns a DataFrame (already collected, no extra collect needed).
    """
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone(config.TIMEZONE).alias("ts_local")
    )
    df = df.with_columns(
        pl.col("ts_local").dt.truncate(every=freq).alias(f"ts_{freq}")
    )
    agg = df.group_by(["session_id", f"ts_{freq}"], maintain_order=True).agg([
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.len().alias("n_ticks"),
    ])
    
    # Drop incomplete bars based on frequency
    if freq == "5m" and config.DROP_INCOMPLETE_ROWS:
        agg = agg.filter(pl.col("n_ticks") == 5)
    elif freq == "1h":
        agg = agg.filter(pl.col("n_ticks") >= 45)
    elif freq == "1d":
        agg = agg.filter(pl.col("n_ticks") >= 360)
    
    agg = agg.rename({f"ts_{freq}": "ts_event"})
    agg = agg.drop("n_ticks")
    agg = agg.sort(["session_id", "ts_event"])

    # For daily, add rolling 5-day volatility (using log returns of daily closes)
    if freq == "1d":
        agg = agg.with_columns(
            pl.col("close").log().alias("log_close")
        )
        agg = agg.with_columns(
            (pl.col("log_close") - pl.col("log_close").shift(1)).alias("daily_log_return")
        )
        agg = agg.with_columns(
            pl.col("daily_log_return").rolling_std(window_size=5).alias("daily_vol_5")
        )
        agg = agg.with_columns(pl.col("daily_vol_5").fill_null(strategy="forward"))
        agg = agg.drop(["log_close", "daily_log_return"])

    # Convert back to UTC for storage
    agg = agg.with_columns(
        pl.col("ts_event").dt.convert_time_zone("UTC").alias("ts_event")
    )
    
    # Cast all price/volume columns to float32 early
    agg = agg.with_columns([
        pl.col("open").cast(pl.Float32),
        pl.col("high").cast(pl.Float32),
        pl.col("low").cast(pl.Float32),
        pl.col("close").cast(pl.Float32),
    ])
    
    return agg


def process_one_file_multi(file_path: str, out_temp_dir: str, freq: str) -> str:
    """
    Read a single 1‑min Parquet file, filter sessions, add session_id,
    resample to given frequency, and write to a temporary file.
    Returns path to the written file, or None if empty.
    """
    logger.info(f"Processing file {file_path} for freq {freq}")
    df = pl.read_parquet(file_path)
    if df["ts_event"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("ts_event").cast(pl.Datetime(time_unit="us", time_zone="UTC")))
    df = filter_session_hours(df)
    if df.is_empty():
        return None
    df = add_session_id(df)
    df_resampled = resample_to_frequency(df, freq)
    if df_resampled.is_empty():
        return None
    out_file = Path(out_temp_dir) / f"{Path(file_path).stem}_{freq}.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_resampled.write_parquet(out_file)
    return str(out_file)


def process_frequency(freq: str, all_files: list) -> tuple:
    """
    Process a single frequency across all 1‑min files.
    Returns (freq, combined DataFrame). The temporary directory is cleaned up afterwards.
    """
    temp_dir = tempfile.mkdtemp(prefix=f"resampled_{freq}_")
    temp_paths = []
    try:
        for f in all_files:
            out = process_one_file_multi(f, temp_dir, freq)
            if out:
                temp_paths.append(out)
        if not temp_paths:
            raise ValueError(f"No data after resampling to {freq}")
        # Use lazy scanning for concatenation to reduce memory
        lf = pl.scan_parquet(temp_paths[0])
        for p in temp_paths[1:]:
            lf = pl.concat([lf, pl.scan_parquet(p)], how="vertical")
        lf = lf.sort(["session_id", "ts_event"])
        df = lf.collect(streaming=True)
        return freq, df
    finally:
        # Clean up temporary directory even if an exception occurred
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_all_streams_chunked(data_glob: str) -> dict:
    """
    Process all 1‑min files and generate three streams: 5m, 1h, 1d.
    Uses ThreadPoolExecutor to process frequencies in parallel.
    Returns dictionary with keys '5m', '1h', '1d' containing Polars DataFrames.
    """
    all_files = glob.glob(data_glob)
    if not all_files:
        raise FileNotFoundError(f"No parquet files found matching {data_glob}")
    print(f"DEBUG: Found {len(all_files)} files for {data_glob}", flush=True)

    streams = {}
    # Process all frequencies in parallel (3 threads max, one per frequency)
    with ThreadPoolExecutor(max_workers=len(config.RESAMPLE_FREQUENCIES)) as executor:
        futures = {executor.submit(process_frequency, freq, all_files): freq for freq in config.RESAMPLE_FREQUENCIES}
        for future in as_completed(futures):
            freq, df = future.result()
            streams[freq] = df
            print(f"DEBUG:   {freq} stream has {df.height} rows", flush=True)
    return streams