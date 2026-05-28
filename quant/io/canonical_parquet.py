pass
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from quant.config_manager import config
import logging
logger = logging.getLogger(__name__)

def write_canonical_parquet(data: pl.DataFrame | pa.Table, path: str):
    pass
    if isinstance(data, pl.DataFrame):
        table = data.to_arrow()
    else:
        table = data
    sorted_column_names = sorted(table.column_names)
    table = table.select(sorted_column_names)
    try:
        pq.write_table(table, path, version='2.6', compression='snappy', row_group_size=getattr(config, 'ROW_GROUP_SIZE', 65536), data_page_version='2.0', use_deprecated_int96_timestamps=False, coerce_timestamps='us')
        logger.info(f'Successfully wrote canonical parquet to {path} with {len(sorted_column_names)} columns.')
    except Exception as e:
        logger.error(f'Failed to write canonical parquet: {e}')
        raise