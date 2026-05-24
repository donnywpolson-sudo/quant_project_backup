# src/utils/check_types.py
import polars as pl
import sys
import argparse
from config import config

def validate_dtypes(parquet_path: str):
    """
    Validates that all feature columns are pl.Float32.
    Ignores non-feature columns like ts_event, session_id, etc.
    """
    try:
        # Scan lazily to avoid loading data into memory
        lf = pl.scan_parquet(parquet_path)
        schema = lf.collect_schema()
        
        # Define columns that MUST be Float32
        # Based on features generated in src/features/engine.py
        errors = []
        for col_name, dtype in schema.items():
            # Check if column is a feature/interaction
            if col_name.startswith("feature_") or col_name.startswith("int_"):
                if dtype != pl.Float32:
                    errors.append(f"Column '{col_name}' has incorrect type: {dtype} (Expected Float32)")
            
            # Special check for 'regime' if exists
            if col_name == "regime" and dtype != pl.Int32:
                errors.append(f"Column 'regime' has incorrect type: {dtype} (Expected Int32)")

        if errors:
            print(f"❌ Type Validation Failed for {parquet_path}:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
        
        print(f"✅ Type Validation Passed: All features in {parquet_path} are Float32.")
        sys.exit(0)

    except Exception as e:
        print(f"Error validating file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to features.parquet")
    args = parser.parse_args()
    validate_dtypes(args.path)