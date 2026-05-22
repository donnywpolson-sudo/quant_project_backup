"""
src/ingest.py
Handles lazy ingestion, session filtering, and mandatory memory/integrity validation.
"""
import polars as pl
import logging
import psutil
from config import config

logger = logging.getLogger(__name__)

def validate_memory_and_integrity(df: pl.DataFrame):
    """
    Implements Section 6: Memory Pre-check and Data Integrity Hook.
    Validates data quality, aborts if memory exceeds RAM_CAP_BYTES,
    and calculates downstream chunking requirements.
    """
    logger.info("Running memory and integrity validation...")

    # 1. Integrity Assertions
    # Check for sorting (Strictly increasing ts_event is required for no lookahead)
    if not df["ts_event"].is_sorted():
        raise ValueError("Integrity Error: ts_event not strictly increasing.")
    
    # Check for nulls in critical columns
    if df.select(pl.all().is_null()).sum().sum().item() > 0:
        raise ValueError("Integrity Error: Unexpected nulls found in data.")
        
    # Logic Checks: High >= Low
    if (df["high"] < df["low"]).any():
        raise ValueError("Integrity Error: High < Low detected.")

    # 2. Memory Estimation
    est_bytes = df.estimated_size()
    rows = df.height
    
    logger.info(f"Memory check: {est_bytes / 1024**3:.2f} GB used by data.")
    
    # Safety Abort
    if est_bytes > config.RAM_CAP_BYTES:
        logger.critical(f"Memory limit exceeded! Data size {est_bytes} > {config.RAM_CAP_BYTES}")
        raise MemoryError("Process RSS/DataFrame size exceeds safety threshold (RAM_CAP_BYTES).")
    
    # 3. Calculate rows_per_chunk for downstream operations
    # Ensures chunking aligns with memory capacity
    avg_row_bytes = est_bytes / rows if rows > 0 else 0
    rows_per_chunk = min(
        config.ROWS_PER_CHUNK_MAX, 
        int((config.RAM_CAP_BYTES * config.MEMORY_SAFETY_MARGIN) / (avg_row_bytes + 1))
    )
    
    logger.info(f"Validation passed. Safe rows_per_chunk: {rows_per_chunk}")
    return rows_per_chunk

def load_and_clean_data(data_glob: str) -> pl.DataFrame:
    """
    Ingests data using a lazy scan, applies cleaning rules, 
    and triggers the memory pre-check hook.
    """
    logger.info(f"Scanning data from: {data_glob}")
    
    # Lazy scan and apply initial filters
    lf = pl.scan_parquet(data_glob)
    
    # Apply cleaning filters defined in config
    if config.DROP_VOLUME_ZERO:
        lf = lf.filter(pl.col("volume") > 0)
        
    # Sort by the stable sort keys
    lf = lf.sort(config.STABLE_SORT_KEYS)
    
    # Collect data into memory
    logger.info("Collecting data into memory...")
    df = lf.collect()
    
    # 4. Mandatory Memory Pre-check Hook (Section 6)
    rows_per_chunk = validate_memory_and_integrity(df)
    
    # Optional: Log the memory trace
    if config.MEMORY_LOG_ENABLED:
        logger.info(f"RSS Memory usage after collection: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
        
    return df