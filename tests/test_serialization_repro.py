"""
tests/test_serialization_repro.py
Performs double write operations on identical feature footprints 
to assert matching cryptographic SHA256 string returns, 
guaranteeing byte-level reproducibility (Section 18).
"""
import pytest
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Import the canonical writer to test its determinism
from src.io.canonical_parquet import write_canonical_parquet


def generate_dummy_table() -> pa.Table:
    """Generates a deterministic PyArrow table for testing."""
    # Create columns out of alphabetical order to test lexicographic sorting enforcement
    data = {
        "zeta_feature": [1.0, 2.0, 3.0, 4.0],
        "alpha_feature": [10.0, 20.0, 30.0, 40.0],
        "beta_feature": [100.0, 200.0, 300.0, 400.0]
    }
    
    # Enforce float32 as per pipeline spec
    schema = pa.schema([
        ("zeta_feature", pa.float32()),
        ("alpha_feature", pa.float32()),
        ("beta_feature", pa.float32())
    ])
    
    return pa.Table.from_pydict(data, schema=schema)


def test_byte_level_reproducibility(tmp_path):
    """
    Ensures deterministic byte-writing using PyArrow schema bounds.
    Writes the same dataframe twice in isolated operations and compares SHA256 hashes.
    """
    file1 = tmp_path / "test_1.parquet"
    file2 = tmp_path / "test_2.parquet"
    
    # Generate and write first instance
    table1 = generate_dummy_table()
    write_canonical_parquet(table1, str(file1))
    
    # Generate and write second instance independently
    table2 = generate_dummy_table()
    write_canonical_parquet(table2, str(file2))
    
    # Calculate SHA256 hashes of the raw bytes
    hash1 = hashlib.sha256(file1.read_bytes()).hexdigest()
    hash2 = hashlib.sha256(file2.read_bytes()).hexdigest()
    
    # Strict byte-level assertion
    assert hash1 == hash2, "Byte-level hash match failed! Serialization is not deterministic."


def test_canonical_parquet_metadata(tmp_path):
    """
    Validates the output file adheres strictly to Section 18 parameters:
    - Format Version: 2.0 (or higher 2.x standard enforcing V2 constructs)
    - Compression: snappy
    - Column Ordering: Lexicographical
    """
    out_file = tmp_path / "metadata_test.parquet"
    table = generate_dummy_table()
    write_canonical_parquet(table, str(out_file))
    
    # Read the parquet metadata
    meta = pq.read_metadata(str(out_file))
    
    # Assert format version
    assert meta.format_version in ["2.0", "2.4", "2.6"], f"Expected Parquet version 2.x, got {meta.format_version}"
    
    # Assert Compression (check first column chunk of the first row group)
    col_chunk = meta.row_group(0).column(0)
    assert col_chunk.compression == "SNAPPY", f"Expected SNAPPY compression, got {col_chunk.compression}"
    
    # Assert Lexicographical Ordering (alpha_feature -> beta_feature -> zeta_feature)
    # The writer should have sorted the columns before saving
    written_columns = [meta.row_group(0).column(i).path_in_schema for i in range(meta.num_columns)]
    expected_columns = sorted(["zeta_feature", "alpha_feature", "beta_feature"])
    
    assert written_columns == expected_columns, "Columns were not lexicographically sorted prior to serialization."