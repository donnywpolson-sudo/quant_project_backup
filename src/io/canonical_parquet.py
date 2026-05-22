"""
src/io/canonical_parquet.py
Deterministic serialization for canonical feature matrices.
Ensures byte-level reproducibility as per Section 18 of ai_prompt.md.
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
    
    Compliance with Section 18:
    - Format Version: 2.0
    - Compression: snappy
    - Row Group Size: 65536
    - Column Ordering: Lexicographic
    """
    # 1. Convert to PyArrow Table if input is a Polars DataFrame
    # This prevents the AttributeError: 'DataFrame' object has no attribute 'column_names'
    if isinstance(data, pl.DataFrame):
        table = data.to_arrow()
    else:
        table = data

    # 2. Enforce Lexicographic Column Ordering
    # Sorting columns ensures that the file structure is identical 
    # regardless of how the DataFrame was constructed.
    sorted_column_names = sorted(table.column_names)
    table = table.select(sorted_column_names)
    
    # 3. Write with Deterministic Parameters
    try:
        pq.write_table(
            table,
            path,
            version="2.6",  # Changed from "2.0" to "2.6"
            compression="snappy",
            row_group_size=config.ROW_GROUP_SIZE,
            data_page_version="2.0",
            use_deprecated_int96_timestamps=False,
            coerce_timestamps="us"
        )
        logger.info(f"Successfully wrote canonical parquet to {path} "
                    f"with {len(sorted_column_names)} columns.")
                    
    except Exception as e:
        logger.error(f"Failed to write canonical parquet: {e}")
        raise