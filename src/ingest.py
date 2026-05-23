"""
src/ingest.py
Handles ingestion of all three streams (5m, 1h, 1d) and alignment.
"""
import polars as pl
import logging
import psutil
from config import config
from src.session import load_all_streams_chunked
from src.align import align_htf_streams

logger = logging.getLogger(__name__)


def validate_memory_and_integrity(df: pl.DataFrame):
    """Same as before, but now df includes HTF columns; we still check OHLC."""
    logger.info("Running memory and integrity validation...")
    if not df["ts_event"].is_sorted():
        raise ValueError("ts_event not strictly increasing.")
    critical_cols = ["open", "high", "low", "close", "volume", "session_id"]
    for col in critical_cols:
        if df[col].null_count() > 0:
            raise ValueError(f"Nulls in column {col}.")
    if (df["high"] < df["low"]).any():
        raise ValueError("High < Low detected.")
    if ((df["open"] < df["low"]) | (df["open"] > df["high"])).any():
        raise ValueError("Open outside [Low, High].")
    if ((df["close"] < df["low"]) | (df["close"] > df["high"])).any():
        raise ValueError("Close outside [Low, High].")
    est_bytes = df.estimated_size()
    rows = df.height
    logger.info(f"Memory usage: {est_bytes / 1024**3:.2f} GB")
    if est_bytes > config.RAM_CAP_BYTES:
        raise MemoryError(f"Data size {est_bytes} exceeds RAM_CAP_BYTES.")
    avg_row_bytes = est_bytes / rows if rows > 0 else 0
    rows_per_chunk = min(
        config.ROWS_PER_CHUNK_MAX,
        int((config.RAM_CAP_BYTES * config.MEMORY_SAFETY_MARGIN) / (avg_row_bytes + 1))
    )
    logger.info(f"Safe rows_per_chunk: {rows_per_chunk}")
    return rows_per_chunk


def load_and_clean_data(data_glob: str) -> pl.DataFrame:
    """
    Load all three streams (5m, 1h, 1d) from the given glob pattern,
    align them without lookahead, and validate.
    """
    logger.info(f"Loading three streams from: {data_glob}")
    print("DEBUG: Starting load_all_streams_chunked...", flush=True)
    streams = load_all_streams_chunked(data_glob)
    print(f"DEBUG: Streams loaded. 5min rows: {streams['5m'].height}", flush=True)
    df_5min = streams["5m"]
    df_1h = streams["1h"]
    df_daily = streams["1d"]
    print("DEBUG: Aligning streams...", flush=True)
    df_aligned = align_htf_streams(df_5min, df_1h, df_daily)
    print(f"DEBUG: Alignment done. Aligned rows: {df_aligned.height}", flush=True)
    validate_memory_and_integrity(df_aligned)
    if config.MEMORY_LOG_ENABLED:
        logger.info(f"RSS after load: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
    return df_aligned