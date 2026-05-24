"""
src/io/canonical_parquet.py
Deterministic serialization for canonical feature matrices.
Ensures byte-level reproducibility.
Now uses Parquet format 2.6 (2.0 is deprecated).
"""
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from config import config
import logging

logger = logging.getLogger(__name__)

def write_canonical_parquet(data: pl.DataFrame | pa.Table, path: str):
    """
    Writes a Polars DataFrame or PyArrow Table to a Parquet file with deterministic settings.
    - Format Version: 2.6 (was 2.0, now updated)
    - Compression: snappy
    - Row Group Size: 65536 (from config.ROW_GROUP_SIZE)
    - Column Ordering: Lexicographic (sorted)
    """
    # Convert Polars DataFrame to PyArrow Table if needed
    if isinstance(data, pl.DataFrame):
        table = data.to_arrow()
    else:
        table = data

    # Enforce lexicographic column ordering for determinism
    sorted_column_names = sorted(table.column_names)
    table = table.select(sorted_column_names)

    # Write with fixed parameters
    try:
        pq.write_table(
            table,
            path,
            version="2.6",                         # changed from "2.0"
            compression="snappy",
            row_group_size=getattr(config, "ROW_GROUP_SIZE", 65536),
            data_page_version="2.0",
            use_deprecated_int96_timestamps=False,
            coerce_timestamps="us"
        )
        logger.info(f"Successfully wrote canonical parquet to {path} "
                    f"with {len(sorted_column_names)} columns.")
    except Exception as e:
        logger.error(f"Failed to write canonical parquet: {e}")
        raise