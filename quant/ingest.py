import polars as pl
import logging
import psutil
from pathlib import Path
from quant.config_manager import config
from quant.session import load_all_streams_chunked
from quant.align import align_htf_streams
from quant.continuous_contract import build_continuous_series
from quant.market_config import detect_symbol_from_path, load_market_config
from quant.io.canonical_parquet import write_canonical_parquet

logger = logging.getLogger(__name__)


def validate_memory_and_integrity(df: pl.DataFrame) -> int:
    logger.info('Running memory and integrity validation...')
    if not df['ts_event'].is_sorted():
        raise ValueError('ts_event not strictly increasing.')
    critical_cols = ['open', 'high', 'low', 'close', 'volume', 'session_id']
    for col in critical_cols:
        if df[col].null_count() > 0:
            raise ValueError(f'Nulls in column {col}.')
    if (df['high'] < df['low']).any():
        raise ValueError('High < Low detected.')
    if ((df['open'] < df['low']) | (df['open'] > df['high'])).any():
        raise ValueError('Open outside [Low, High].')
    if ((df['close'] < df['low']) | (df['close'] > df['high'])).any():
        raise ValueError('Close outside [Low, High].')
    est_bytes = df.estimated_size()
    rows = df.height
    logger.info(f'Memory usage: {est_bytes / 1024 ** 3:.2f} GB')
    if est_bytes > config.RAM_CAP_BYTES:
        raise MemoryError(f'Data size {est_bytes} exceeds RAM_CAP_BYTES.')
    avg_row_bytes = est_bytes / rows if rows > 0 else 0
    rows_per_chunk = min(
        config.ROWS_PER_CHUNK_MAX,
        int(config.RAM_CAP_BYTES * config.MEMORY_SAFETY_MARGIN / (avg_row_bytes + 1)),
    )
    logger.info(f'Safe rows_per_chunk: {rows_per_chunk}')
    return rows_per_chunk


def _load_cross_asset_feature(secondary_symbol: str, primary_path: Path) -> pl.DataFrame:
    """
    Load a single cross-asset feature (log return) for one secondary symbol.
    Only reads ts_event and close columns — eliminates SELECT *.
    Returns a DataFrame with [ts_event, {symbol}_ret_1].
    """
    secondary_glob = str(primary_path.parent.parent / secondary_symbol / primary_path.name)
    print(
        f'[INGEST] Loading cross-asset features for {secondary_symbol} from {secondary_glob}',
        flush=True,
    )
    try:
        streams = load_all_streams_chunked(secondary_glob)
        df_5min = streams['5m']
        df_5min = df_5min.with_columns(
            (pl.col('close') / pl.col('close').shift(1)).log().alias(
                f'{secondary_symbol}_ret_1'
            )
        )
        # Column projection: only keep ts_event and the computed feature
        df_5min = df_5min.select(['ts_event', f'{secondary_symbol}_ret_1'])
        logger.info(
            f'Loaded cross-asset features for {secondary_symbol}, {df_5min.height} rows'
        )
        return df_5min
    except Exception as e:
        logger.warning(
            f'Could not load cross-asset features for {secondary_symbol}: {e}'
        )
        return pl.DataFrame()


def _join_cross_asset_features(
    df_aligned: pl.DataFrame,
    cross_asset_symbols: list,
    data_glob: str,
) -> pl.DataFrame:
    """
    Join all cross-asset features in a single batch to eliminate the N+1
    join pattern.  Each secondary symbol is loaded independently, then the
    resulting DataFrames are joined together before a single final left-join
    onto df_aligned, rather than joining one symbol at a time.
    """
    primary_path = Path(data_glob)

    # Load all cross-asset feature frames (each returns [ts_event, {sym}_ret_1])
    cross_frames = []
    for sym in cross_asset_symbols:
        frame = _load_cross_asset_feature(sym, primary_path)
        if not frame.is_empty():
            cross_frames.append(frame)

    if not cross_frames:
        return df_aligned

    # Merge all cross-asset frames on ts_event into a single wide frame
    cross_combined = cross_frames[0]
    for frame in cross_frames[1:]:
        cross_combined = cross_combined.join(frame, on='ts_event', how='outer')

    # Single left-join instead of N+1 individual joins
    df_aligned = df_aligned.join(cross_combined, on='ts_event', how='left')

    # Forward-fill cross-asset columns within session_id groups only,
    # resetting to null at session boundaries to avoid stale data
    # contamination across primary market session gaps (Finding #17).
    cross_cols = [c for c in cross_combined.columns if c != 'ts_event']
    if cross_cols:
        if 'session_id' in df_aligned.columns:
            df_aligned = df_aligned.with_columns(
                [pl.col(c).fill_null(strategy='forward').over('session_id')
                 for c in cross_cols]
            )
        else:
            df_aligned = df_aligned.with_columns(
                [pl.col(c).fill_null(strategy='forward') for c in cross_cols]
            )

    return df_aligned


def load_and_clean_data(
    data_glob: str,
    cache_path: str = None,
    cross_asset_symbols: list = None,
) -> pl.DataFrame:
    if cache_path and Path(cache_path).exists():
        print(
            f'[INGEST] Loading aligned data from cache: {cache_path}', flush=True
        )
        logger.info(f'Loading aligned data from cache: {cache_path}')
        df_aligned = pl.read_parquet(cache_path)
        validate_memory_and_integrity(df_aligned)
        return df_aligned

    print(
        f'[INGEST] No cache found. Loading three streams from: {data_glob}',
        flush=True,
    )
    logger.info(f'Loading three streams from: {data_glob}')
    streams = load_all_streams_chunked(data_glob)
    df_5min = streams['5m']
    df_1h = streams.get('1h')
    df_daily = streams.get('1d')

    print('[INGEST] Aligning HTF streams...', flush=True)
    df_aligned = align_htf_streams(df_5min, df_1h, df_daily)
    validate_memory_and_integrity(df_aligned)

    # ---- Explicit gap filter (Finding #3) -----------------------------------
    from quant.gap_filter import filter_gaps
    df_aligned = filter_gaps(df_aligned, max_gap_minutes=30)
    validate_memory_and_integrity(df_aligned)

    # ---- Continuous contract pipeline (Finding #12) -------------------------
    # Build ratio-adjusted continuous price series with roll-date splicing.
    # Derives symbol from the data_glob path (e.g., 'data/ES/*.parquet' -> 'ES').
    symbol = detect_symbol_from_path(data_glob)
    print(f'[INGEST] Building continuous contract series for {symbol}...',
          flush=True)

    # Load contract_multiplier from per-market YAML if available
    load_market_config(symbol)
    contract_multiplier = 1.0
    market_cfg_yaml = config.MARKET_CONFIGS.get(symbol)
    if market_cfg_yaml and Path(market_cfg_yaml).exists():
        import yaml
        try:
            with open(market_cfg_yaml, 'r') as f:
                mkt = yaml.safe_load(f)
            contract_multiplier = float(
                mkt.get('metadata', {}).get('contract_multiplier', 1.0)
            )
        except Exception:
            contract_multiplier = 1.0

    df_aligned = build_continuous_series(
        df_aligned, symbol, contract_multiplier=contract_multiplier
    )
    validate_memory_and_integrity(df_aligned)

    # Batch join all cross-asset features in one pass (eliminates N+1 pattern)
    if cross_asset_symbols:
        df_aligned = _join_cross_asset_features(
            df_aligned, cross_asset_symbols, data_glob
        )

    if cache_path:
        print(f'[INGEST] Caching aligned data to {cache_path}', flush=True)
        logger.info(f'Caching aligned data to {cache_path}')
        write_canonical_parquet(df_aligned, cache_path)

    if config.MEMORY_LOG_ENABLED:
        logger.info(
            f'RSS after load: {psutil.Process().memory_info().rss / 1024 ** 3:.2f} GB'
        )
    return df_aligned