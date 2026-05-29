pass
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from quant.config_manager import config
from quant.io.atomic import atomic_write_canonical_parquet
import logging
logger = logging.getLogger(__name__)

def write_canonical_parquet(data: pl.DataFrame | pa.Table, path: str):
    pass
    if isinstance(data, pl.DataFrame):
        df = data
    else:
        df = pl.from_arrow(data)
    row_group_size = getattr(config, 'ROW_GROUP_SIZE', 65536)
    atomic_write_canonical_parquet(df, path, row_group_size=row_group_size)
    logger.info('Successfully wrote canonical parquet to %s with %d columns.',
                path, len(df.columns))