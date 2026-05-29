"""
pipeline/02_normalization/__init__.py
Normalization stage — session filtering, resampling, HTF alignment,
gap filtering, and cross-asset feature joining.

Patch 1 (CRITICAL): gap-filter divisor corrected from 60_000_000_000.0
(nanosecond conversion) to 60_000_000.0 (microsecond conversion).
The ts_event column uses datetime[us]; the old divisor under-reported
gaps by a factor of 1000.
"""

import logging
from pathlib import Path

import polars as pl
import yaml

from quant.config_manager import config
from quant.align import align_htf_streams
from quant.session import (
    add_session_id,
    filter_session_hours,
    resample_to_frequency,
)
from quant.io.canonical_parquet import write_canonical_parquet
from quant.market_config import detect_symbol_from_path, load_market_config

from pipeline.state import PipelineState
from .gap_filter import filter_gaps

logger = logging.getLogger(__name__)

_OHLCV_COLS = {"open", "high", "low", "close", "volume", "session_id"}


# ===========================================================================
# Cross-asset feature loading (unchanged from quant/ingest.py)
# ===========================================================================


def _load_cross_asset_feature(
    secondary_symbol: str, primary_path: Path
) -> pl.DataFrame:
    from quant.session import load_all_streams_chunked

    secondary_glob = str(
        primary_path.parent.parent / secondary_symbol / primary_path.name
    )
    logger.info(
        "[02_NORM] Loading cross-asset features for %s from %s",
        secondary_symbol,
        secondary_glob,
    )
    try:
        streams = load_all_streams_chunked(secondary_glob)
        df_5m = streams["5m"]
        df_5m = df_5m.with_columns(
            (pl.col("close") / pl.col("close").shift(1))
            .log()
            .alias(f"{secondary_symbol}_ret_1")
        )
        df_5m = df_5m.select(["ts_event", f"{secondary_symbol}_ret_1"])
        return df_5m
    except Exception:
        logger.warning(
            "Could not load cross-asset features for %s",
            secondary_symbol,
            exc_info=True,
        )
        return pl.DataFrame()


def _join_cross_asset_features(
    df_aligned: pl.DataFrame,
    cross_asset_symbols: list,
    data_glob: str,
) -> pl.DataFrame:
    primary_path = Path(data_glob)
    cross_frames = []
    for sym in cross_asset_symbols:
        frame = _load_cross_asset_feature(sym, primary_path)
        if not frame.is_empty():
            cross_frames.append(frame)
    if not cross_frames:
        return df_aligned
    cross_combined = cross_frames[0]
    for frame in cross_frames[1:]:
        cross_combined = cross_combined.join(frame, on="ts_event", how="outer")
        right_col = "ts_event_right"
        if right_col in cross_combined.columns:
            cross_combined = cross_combined.drop(right_col)
    df_aligned = df_aligned.join(cross_combined, on="ts_event", how="left")
    cross_cols = [c for c in cross_combined.columns if c != "ts_event"]
    if cross_cols:
        if "session_id" in df_aligned.columns:
            df_aligned = df_aligned.with_columns(
                [
                    pl.col(c)
                    .fill_null(strategy="forward")
                    .over("session_id")
                    for c in cross_cols
                ]
            )
        else:
            df_aligned = df_aligned.with_columns(
                [
                    pl.col(c).fill_null(strategy="forward")
                    for c in cross_cols
                ]
            )
    return df_aligned


# ===========================================================================
# Validation
# ===========================================================================


def _validate_memory_and_integrity(df: pl.DataFrame) -> int:
    import psutil

    if df.is_empty():
        raise ValueError(
            "validate_memory_and_integrity: DataFrame is empty — "
            "upstream processing may have removed all rows."
        )
    if not df["ts_event"].is_sorted():
        raise ValueError("ts_event not strictly increasing.")
    critical_cols = [
        "open", "high", "low", "close", "volume", "session_id",
    ]
    for col in critical_cols:
        if col in df.columns and df[col].null_count() > 0:
            raise ValueError(f"Nulls in column {col}.")
    if all(c in df.columns for c in ("high", "low")):
        if (df["high"] < df["low"]).any():
            raise ValueError("High < Low detected.")
    if all(c in df.columns for c in ("open", "low", "high")):
        if (
            (df["open"] < df["low"]) | (df["open"] > df["high"])
        ).any():
            raise ValueError("Open outside [Low, High].")
    if all(c in df.columns for c in ("close", "low", "high")):
        if (
            (df["close"] < df["low"]) | (df["close"] > df["high"])
        ).any():
            raise ValueError("Close outside [Low, High].")
    est_bytes = df.estimated_size()
    rows = df.height
    logger.info("Memory usage: %.2f GB", est_bytes / 1024**3)
    if est_bytes > config.RAM_CAP_BYTES:
        raise MemoryError(
            f"Data size {est_bytes} exceeds RAM_CAP_BYTES."
        )
    avg_row_bytes = est_bytes / rows if rows > 0 else 0
    rows_per_chunk = min(
        config.ROWS_PER_CHUNK_MAX,
        int(
            config.RAM_CAP_BYTES
            * config.MEMORY_SAFETY_MARGIN
            / (avg_row_bytes + 1)
        ),
    )
    logger.info("Safe rows_per_chunk: %d", rows_per_chunk)
    return rows_per_chunk


# ===========================================================================
# Public stage entry-point
# ===========================================================================


def normalization_stage(state: PipelineState) -> PipelineState:
    """
    Normalization stage.

    Order:
      1.  Session filter + session_id on roll-adjusted raw data.
      2.  Resample to 5m / 1h / 1d frequencies (from adjusted data).
      3.  Align HTF streams (1h + daily → 5-min via join_asof).
      4.  Gap filter (Patch 1 — corrected divisor).
      5.  Cross-asset feature join (if configured).
      6.  Cache aligned output.
    """
    df = state.data
    if df is None or df.is_empty():
        raise ValueError(
            "normalization_stage received empty DataFrame — "
            "check upstream ingestion."
        )

    data_glob: str = state.metadata.get("data_glob", "")
    cache_path: str | None = state.metadata.get("cache_path")
    cross_asset_symbols: list | None = state.metadata.get(
        "cross_asset_symbols"
    )

    # --- 1. Session filter + session_id -----------------------------------
    logger.info("[02_NORM] Applying session filter & session_id…")
    df = filter_session_hours(df)
    if df.is_empty():
        raise ValueError("All rows filtered by session_hours.")
    df = add_session_id(df)

    # --- 2. Resample to 5m / 1h / 1d FROM ADJUSTED DATA ------------------
    logger.info(
        "[02_NORM] Resampling to %s (from adjusted prices)",
        config.RESAMPLE_FREQUENCIES,
    )
    df_5m = resample_to_frequency(df, "5m")
    df_1h = resample_to_frequency(df, "1h")
    df_daily = resample_to_frequency(df, "1d")

    _validate_memory_and_integrity(df_5m)
    logger.info(
        "[02_NORM] Streams: 5m=%d  1h=%d  daily=%d",
        df_5m.height,
        df_1h.height,
        df_daily.height,
    )

    # --- 3. HTF alignment ------------------------------------------------
    logger.info("[02_NORM] Aligning HTF streams…")
    df_aligned = align_htf_streams(df_5m, df_1h, df_daily)
    _validate_memory_and_integrity(df_aligned)

    # --- 4. Gap filter (Patch 1 — fixed divisor) -------------------------
    df_aligned = filter_gaps(df_aligned, max_gap_minutes=30)
    _validate_memory_and_integrity(df_aligned)

    # --- 5. Cross-asset features -----------------------------------------
    if cross_asset_symbols:
        logger.info(
            "[02_NORM] Joining cross-asset features: %s", cross_asset_symbols
        )
        df_aligned = _join_cross_asset_features(
            df_aligned, cross_asset_symbols, data_glob
        )

    # --- 6. Cache --------------------------------------------------------
    if cache_path:
        logger.info("[02_NORM] Caching aligned data to %s", cache_path)
        write_canonical_parquet(df_aligned, cache_path)

    state.data = df_aligned
    logger.info(
        "[02_NORM] Complete: %d rows × %d cols",
        df_aligned.height,
        len(df_aligned.columns),
    )
    return state
