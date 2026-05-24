"""
src/session.py
Implements Globex session definition, session_id, and resampling to 5m, 1h, 1d.
Now sequential (no threads) and deterministic.
"""
import polars as pl
import logging
from datetime import time
import pytz
from pathlib import Path
import tempfile
import glob
import shutil
from config import config
from tqdm import tqdm

logger = logging.getLogger(__name__)

TZ = pytz.timezone(config.TIMEZONE)
SESSION_START = config.SESSION_START_LOCAL
SESSION_END = config.SESSION_END_LOCAL
SESSION_BREAK_START = config.SESSION_BREAK_START_LOCAL
SESSION_BREAK_END = config.SESSION_BREAK_END_LOCAL

def add_session_id(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone(config.TIMEZONE).alias("ts_local")
    )
    session_id = pl.col("ts_local").dt.offset_by("6h").dt.date().cast(pl.String)
    df = df.with_columns(session_id.alias("session_id"))
    return df.drop("ts_local")

def filter_session_hours(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone(config.TIMEZONE).dt.time().alias("time_local")
    )
    time_local = pl.col("time_local")
    in_session = (time_local >= SESSION_START) | (time_local < SESSION_END)
    if SESSION_BREAK_START is not None and SESSION_BREAK_END is not None:
        in_break = (time_local >= SESSION_BREAK_START) & (time_local < SESSION_BREAK_END)
        in_session = in_session & ~in_break
    df = df.filter(in_session)
    return df.drop("time_local")

def resample_to_frequency(df: pl.DataFrame, freq: str) -> pl.DataFrame:
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

    if freq == "5m" and config.DROP_INCOMPLETE_ROWS:
        agg = agg.filter(pl.col("n_ticks") == 5)
    elif freq == "1h":
        agg = agg.filter(pl.col("n_ticks") >= 45)
    elif freq == "1d":
        agg = agg.filter(pl.col("n_ticks") >= 360)

    agg = agg.rename({f"ts_{freq}": "ts_event"})
    agg = agg.drop("n_ticks")
    agg = agg.sort(["session_id", "ts_event"])

    if freq == "1d":
        agg = agg.with_columns(pl.col("close").log().alias("log_close"))
        agg = agg.with_columns(
            (pl.col("log_close") - pl.col("log_close").shift(1)).alias("daily_log_return")
        )
        agg = agg.with_columns(
            pl.col("daily_log_return").rolling_std(window_size=5).alias("daily_vol_5")
        )
        agg = agg.with_columns(pl.col("daily_vol_5").fill_null(strategy="forward"))
        agg = agg.drop(["log_close", "daily_log_return"])

    agg = agg.with_columns(
        pl.col("ts_event").dt.convert_time_zone("UTC").alias("ts_event")
    )
    agg = agg.with_columns([
        pl.col("open").cast(pl.Float32),
        pl.col("high").cast(pl.Float32),
        pl.col("low").cast(pl.Float32),
        pl.col("close").cast(pl.Float32),
    ])
    return agg

def process_one_file(file_path: str, out_temp_dir: str, freq: str) -> str:
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
    print(f"\n[SESSION] Resampling {freq} (found {len(all_files)} files)", flush=True)
    temp_dir = tempfile.mkdtemp(prefix=f"resampled_{freq}_")
    temp_paths = []
    try:
        for f in tqdm(all_files, desc=f"Resampling {freq}", unit="file"):
            out = process_one_file(f, temp_dir, freq)
            if out:
                temp_paths.append(out)
        if not temp_paths:
            raise ValueError(f"No data after resampling to {freq}")
        lf = pl.scan_parquet(temp_paths[0])
        for p in temp_paths[1:]:
            lf = pl.concat([lf, pl.scan_parquet(p)], how="vertical")
        lf = lf.sort(["session_id", "ts_event"])
        df = lf.collect(streaming=True)
        print(f"[SESSION] {freq} stream has {df.height} rows.", flush=True)
        return freq, df
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def load_all_streams_chunked(data_glob: str) -> dict:
    all_files = sorted(glob.glob(data_glob))   # deterministic order
    if not all_files:
        raise FileNotFoundError(f"No parquet files found matching {data_glob}")
    print(f"[SESSION] Found {len(all_files)} files.", flush=True)

    streams = {}
    for freq in config.RESAMPLE_FREQUENCIES:
        _, df = process_frequency(freq, all_files)
        streams[freq] = df
    return streams