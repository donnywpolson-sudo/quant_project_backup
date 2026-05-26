pass
import polars as pl
import logging
import psutil
from pathlib import Path
from config import config
from quant.session import load_all_streams_chunked
from quant.align import align_htf_streams
from quant.io.canonical_parquet import write_canonical_parquet
logger = logging.getLogger(__name__)

def validate_memory_and_integrity(df: pl.DataFrame):
    pass
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
    rows_per_chunk = min(config.ROWS_PER_CHUNK_MAX, int(config.RAM_CAP_BYTES * config.MEMORY_SAFETY_MARGIN / (avg_row_bytes + 1)))
    logger.info(f'Safe rows_per_chunk: {rows_per_chunk}')
    return rows_per_chunk

def load_cross_asset_features(data_glob: str, secondary_symbol: str) -> pl.DataFrame:
    pass
    primary_path = Path(data_glob)
    secondary_glob = str(primary_path.parent.parent / secondary_symbol / primary_path.name)
    print(f'[INGEST] Loading cross‑asset features for {secondary_symbol} from {secondary_glob}', flush=True)
    try:
        streams = load_all_streams_chunked(secondary_glob)
        df_5min = streams['5m']
        df_5min = df_5min.with_columns((pl.col('close') / pl.col('close').shift(1)).log().alias(f'{secondary_symbol}_ret_1'))
        df_5min = df_5min.select(['ts_event', f'{secondary_symbol}_ret_1'])
        logger.info(f'Loaded cross‑asset features for {secondary_symbol}, {df_5min.height} rows')
        return df_5min
    except Exception as e:
        logger.warning(f'Could not load cross‑asset features for {secondary_symbol}: {e}')
        return pl.DataFrame()

def load_and_clean_data(data_glob: str, cache_path: str=None, cross_asset_symbols: list=None) -> pl.DataFrame:
    pass
    if cache_path and Path(cache_path).exists():
        print(f'[INGEST] Loading aligned data from cache: {cache_path}', flush=True)
        logger.info(f'Loading aligned data from cache: {cache_path}')
        df_aligned = pl.read_parquet(cache_path)
        validate_memory_and_integrity(df_aligned)
        return df_aligned
    print(f'[INGEST] No cache found. Loading three streams from: {data_glob}', flush=True)
    logger.info(f'Loading three streams from: {data_glob}')
    streams = load_all_streams_chunked(data_glob)
    df_5min = streams['5m']
    df_1h = streams['1h']
    df_daily = streams['1d']
    print('[INGEST] Aligning HTF streams...', flush=True)
    df_aligned = align_htf_streams(df_5min, df_1h, df_daily)
    validate_memory_and_integrity(df_aligned)
    if cross_asset_symbols:
        for sym in cross_asset_symbols:
            print(f'[INGEST] Adding cross‑asset features for {sym}...', flush=True)
            df_cross = load_cross_asset_features(data_glob, sym)
            if not df_cross.is_empty():
                df_aligned = df_aligned.join(df_cross, on='ts_event', how='left')
                for col in df_cross.columns:
                    if col != 'ts_event':
                        df_aligned = df_aligned.with_columns(pl.col(col).fill_null(strategy='forward'))
    if cache_path:
        print(f'[INGEST] Caching aligned data to {cache_path}', flush=True)
        logger.info(f'Caching aligned data to {cache_path}')
        write_canonical_parquet(df_aligned, cache_path)
    if config.MEMORY_LOG_ENABLED:
        logger.info(f'RSS after load: {psutil.Process().memory_info().rss / 1024 ** 3:.2f} GB')
    return df_aligned