"""
pipeline/01_ingestion/__init__.py
Ingestion stage — loads raw parquet data and applies continuous contract
adjustment BEFORE any session normalization or HTF alignment.

Patch 2 (CRITICAL): build_continuous_series is called on raw data prior to
resampling, so that 1h and daily streams derived downstream reflect
roll-adjusted prices.  This eliminates contract-roll artifacts from the
HTF context features.
"""

import logging
from pathlib import Path

import polars as pl
import yaml

from core.config import config
from pipeline.contracts.continuous import build_continuous_series
from core.market import detect_symbol_from_path, load_market_config

from pipeline.tracking.state import PipelineState

logger = logging.getLogger(__name__)

_OHLCV_PROJECTION = ["ts_event", "open", "high", "low", "close", "volume"]

_OHLCV_ADJUSTED_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def _read_raw_files(data_glob: str) -> pl.DataFrame:
    """Read all parquet files matching the glob into a single DataFrame.

    Only the 6 OHLCV columns are projected to minimise I/O.
    """
    import glob as _glob

    files = sorted(_glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found matching {data_glob}"
        )

    frames: list[pl.DataFrame] = []
    for f in files:
        df = pl.read_parquet(f, columns=_OHLCV_PROJECTION)
        if df["ts_event"].dtype != pl.Datetime:
            df = df.with_columns(
                pl.col("ts_event").cast(
                    pl.Datetime(time_unit="us", time_zone="UTC")
                )
            )
        frames.append(df)

    result = pl.concat(frames).sort("ts_event")
    logger.info(
        "[01_INGESTION] Loaded %d files: %d rows, range %s → %s",
        len(files),
        result.height,
        result["ts_event"].min(),
        result["ts_event"].max(),
    )
    return result


def _apply_continuous_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    """Replace raw OHLCV columns with their continuous (roll-adjusted)
    counterparts after build_continuous_series has populated
    continuous_price / continuous_open / continuous_high / continuous_low.

    Falls back to the original columns when continuous variants are absent
    (e.g. when no roll dates were found).
    """
    mapping: dict[str, str] = {
        "close": "continuous_price",
        "open": "continuous_open",
        "high": "continuous_high",
        "low": "continuous_low",
    }
    exprs = []
    for orig, adj in mapping.items():
        if adj in df.columns:
            exprs.append(pl.col(adj).cast(pl.Float32).alias(orig))
    if exprs:
        df = df.with_columns(exprs)
    return df.select(_OHLCV_ADJUSTED_COLS + ["ts_event"])


def _derive_1h_1d(df_5min_adjusted: pl.DataFrame) -> tuple[
    pl.DataFrame, pl.DataFrame
]:
    """Resample roll-adjusted 5-min data to 1h and 1d frequencies.

    These derived streams inherit the continuous contract adjustment,
    eliminating the roll-artifact leakage identified in the audit
    (CRITICAL finding: HTF alignment used unadjusted prices).
    """
    from pipeline.session.session import (
        add_session_id,
        resample_to_frequency,
    )

    if "session_id" not in df_5min_adjusted.columns:
        df_5min_adjusted = add_session_id(df_5min_adjusted)

    df_1h = resample_to_frequency(df_5min_adjusted.clone(), "1h")
    df_1d = resample_to_frequency(df_5min_adjusted.clone(), "1d")

    logger.info(
        "[01_INGESTION] Derived 1h (%d rows) and 1d (%d rows) from adjusted 5-min",
        df_1h.height,
        df_1d.height,
    )
    return df_1h, df_1d


# ---------------------------------------------------------------------------
# Public stage entry-point
# ---------------------------------------------------------------------------


def ingestion_stage(state: PipelineState) -> PipelineState:
    """
    Ingestion stage (Patch 2 reorder).

    Order:
      1.  Read raw parquet files.
      2.  Detect symbol / load contract multiplier.
      3.  **build_continuous_series ON RAW DATA** (before resampling).
      4.  Replace OHLCV with adjusted (continuous) values.
      5.  Store adjusted data in state (session normalization runs in 02).
    """
    data_glob: str = state.metadata.get("data_glob", "")
    cache_path: str | None = state.metadata.get("cache_path")
    cross_asset_symbols: list | None = state.metadata.get(
        "cross_asset_symbols"
    )

    if not data_glob:
        raise ValueError(
            "PipelineState.metadata must contain 'data_glob' key"
        )

    # --- cache short-circuit (reads fully-processed aligned data) ----------
    if cache_path and Path(cache_path).exists():
        logger.info(
            "[01_INGESTION] Cache hit: %s — loading aligned data", cache_path
        )
        state.data = pl.read_parquet(cache_path)
        state.metadata.setdefault("symbol", detect_symbol_from_path(data_glob))
    return state


def load_and_clean_data(
    data_glob: str,
    cache_path: str = None,
    cross_asset_symbols: list = None,
) -> pl.DataFrame:
    """Compatibility wrapper for code that expects the old quant.ingest API.

    Runs the ingestion stage internally and returns the adjusted DataFrame
    directly, matching the original `load_and_clean_data` contract.
    """
    from pipeline.tracking.state import PipelineState

    state = PipelineState(
        data=pl.DataFrame(),
        metadata={
            "data_glob": data_glob,
            "cache_path": cache_path,
            "cross_asset_symbols": cross_asset_symbols,
        },
    )
    state = ingestion_stage(state)
    return state.data

    # --- 1. Read raw data ---------------------------------------------------
    df_raw = _read_raw_files(data_glob)

    # --- 2. Symbol detection & contract multiplier --------------------------
    symbol = detect_symbol_from_path(data_glob)
    load_market_config(symbol)
    contract_multiplier = 1.0
    market_cfg_yaml = config.MARKET_CONFIGS.get(symbol)
    if market_cfg_yaml and Path(market_cfg_yaml).exists():
        try:
            with open(market_cfg_yaml) as fh:
                mkt = yaml.safe_load(fh)
            contract_multiplier = float(
                mkt.get("metadata", {}).get("contract_multiplier", 1.0)
            )
        except Exception:
            contract_multiplier = 1.0

    state.metadata["symbol"] = symbol
    state.metadata["contract_multiplier"] = contract_multiplier

    # --- 3. Continuous contracts BEFORE resampling (Patch 2) ---------------
    logger.info(
        "[01_INGESTION] Building continuous series for %s (multiplier=%.1f)",
        symbol,
        contract_multiplier,
    )
    df_raw = build_continuous_series(
        df_raw, symbol, contract_multiplier=contract_multiplier
    )

    # --- 4. Replace OHLCV with adjusted values -----------------------------
    df_adjusted = _apply_continuous_ohlcv(df_raw)

    # --- 5. Transfer metadata forward ---------------------------------------
    state.metadata["cross_asset_symbols"] = cross_asset_symbols
    state.metadata["data_glob"] = data_glob
    state.metadata["cache_path"] = cache_path
    state.data = df_adjusted

    logger.info(
        "[01_INGESTION] Complete: %d rows (roll-adjusted)",
        state.data.height,
    )
    return state
