from __future__ import annotations
import os
import datetime as dt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from sys import audit

import pandas as pd
import numpy as np
import databento as db

# =========================================
# CONFIG
# =========================================
API_KEY = os.getenv("DATABENTO_API_KEY", "")
DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
STYPE_IN = "continuous"
STYPE_OUT = "instrument_id"

SYMBOLS = [
    "NQ.v.0", "ES.v.0", "YM.v.0", "RTY.v.0",
    "CL.v.0", "NG.v.0",
    "GC.v.0", "SI.v.0", "HG.v.0",
    "ZB.v.0", "ZN.v.0",
    "ZC.v.0",
]

DATA_DIR = Path(r"C:\Users\donny\Desktop\Backtest")
EXPECTED_COLUMNS = ["open", "high", "low", "close", "volume"]
MAX_SESSION_BREAK = dt.timedelta(hours=4)


# =========================================
# HELPERS
# =========================================

def parquet_path(symbol: str) -> Path:
    safe = symbol.replace(".", "_")
    return DATA_DIR / f"{safe}_all.parquet"


def csv_path(symbol: str) -> Path:
    safe = symbol.replace(".", "_")
    return DATA_DIR / f"{safe}_stitched.csv"


def normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert("America/New_York")
    else:
        idx = idx.tz_convert("America/New_York")
    return idx


def load_symbol_history(symbol: str) -> pd.DataFrame:
    parquet_file = parquet_path(symbol)
    csv_file = csv_path(symbol)

    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
    elif csv_file.exists():
        df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
        print(f"Loaded fallback CSV history for {symbol} from {csv_file}")
    else:
        raise FileNotFoundError(f"Missing history for {symbol}: {parquet_file} or {csv_file}")

    df.index = normalize_index(df.index)
    df = df.sort_index()

    if df.index.has_duplicates:
        dup_count = df.index.duplicated(keep="first").sum()
        print(f"Warning: dropping {dup_count} duplicate timestamp rows in existing history for {symbol}")
        df = df[~df.index.duplicated(keep="first")]

    if df.empty:
        raise ValueError(f"{symbol} history is empty after loading and cleaning")

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{symbol} history is missing columns: {missing_cols}")

    return df


def get_client() -> db.Historical:
    if not API_KEY:
        raise RuntimeError("DATABENTO_API_KEY is not set")
    return db.Historical(API_KEY)


def minute_aligned(index: pd.DatetimeIndex) -> bool:
    return ((index.second == 0) & (index.microsecond == 0)).all()


def find_missing_intervals(index: pd.DatetimeIndex, max_break: dt.timedelta = MAX_SESSION_BREAK) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(index) < 2:
        return []

    missing = []
    prev = index[0]
    for current in index[1:]:
        delta = current - prev
        if dt.timedelta(minutes=1) < delta <= max_break:
            missing.append((prev + dt.timedelta(minutes=1), current - dt.timedelta(minutes=1)))
        prev = current
    return missing


def split_interval(start: dt.datetime, end: dt.datetime, max_minutes: int = 1440) -> list[tuple[dt.datetime, dt.datetime]]:
    if start >= end:
        return []

    intervals = []
    current_start = start
    while current_start < end:
        current_end = min(end, current_start + dt.timedelta(minutes=max_minutes))
        intervals.append((current_start, current_end))
        current_start = current_end
    return intervals


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df = df.loc[:, EXPECTED_COLUMNS]

    if df.isna().any(axis=None):
        missing_rows = df[df.isna().any(axis=1)]
        print(f"Warning: dropping {len(missing_rows)} rows with NaNs from fetched data")
        df = df.dropna()

    df.index = normalize_index(df.index)
    df = df.sort_index()

    if df.index.has_duplicates:
        dup_count = df.index.duplicated(keep="first").sum()
        print(f"Warning: dropping {dup_count} duplicate timestamp rows from fetched data")
        df = df[~df.index.duplicated(keep="first")]

    if not minute_aligned(df.index):
        raise ValueError("Fetched data contains non-minute-aligned timestamps")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Fetched data timestamps are not sorted")

    return df


def validate_new_df(df: pd.DataFrame, symbol: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if df.empty:
        return []

    missing = find_missing_intervals(df.index)
    if missing:
        print(f"{symbol}: Detected {len(missing)} missing interval(s) in fetched data:")
        for start, end in missing:
            print(f"  missing {start} → {end}")
    return missing


def calculate_data_coverage(df: pd.DataFrame) -> tuple[int, int, float]:
    """
    Calculate coverage statistics for a complete dataframe.
    Returns (total_span_minutes, missing_minutes, coverage_percentage)
    """
    if df.empty or len(df) < 2:
        return 0, 0, 0.0
    
    total_span_minutes = int((df.index.max() - df.index.min()).total_seconds() / 60)
    if total_span_minutes == 0:
        return 0, 0, 100.0
    
    # Expected rows: one per minute in the span (inclusive of start and end)
    expected_rows = total_span_minutes + 1
    actual_rows = len(df)
    missing_minutes = max(0, expected_rows - actual_rows)
    
    # Coverage percentage
    coverage_pct = 100.0 * (actual_rows / expected_rows)
    
    return total_span_minutes, missing_minutes, coverage_pct


# =========================================
# FETCHING NEW DATA
# =========================================

def fetch_range(symbol: str, start_dt: dt.datetime, end_dt: dt.datetime) -> pd.DataFrame:
    if start_dt >= end_dt:
        return pd.DataFrame()

    client = get_client()
    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"Downloading {symbol}: {start_str} → {end_str}")

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=symbol,
        schema=SCHEMA,
        stype_in=STYPE_IN,
        stype_out=STYPE_OUT,
        start=start_str,
        end=end_str,
    )

    df = data.to_df()
    if df is None or df.empty:
        return pd.DataFrame()

    return normalize_df(df)


def fetch_gap(symbol: str, start_dt: dt.datetime, end_dt: dt.datetime) -> pd.DataFrame:
    intervals = split_interval(start_dt, end_dt)
    parts: list[pd.DataFrame] = []
    for start, end in intervals:
        part = fetch_range(symbol, start, end)
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()


def repair_missing_intervals(symbol: str, missing_intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    if not missing_intervals:
        return pd.DataFrame()

    repaired_parts: list[pd.DataFrame] = []
    for start, end in missing_intervals:
        print(f"{symbol}: repairing missing interval {start} → {end}")
        # fetch_range uses end exclusive semantics, so extend by one minute
        repaired = fetch_gap(symbol, start.tz_convert(dt.timezone.utc), (end + dt.timedelta(minutes=1)).tz_convert(dt.timezone.utc))
        if not repaired.empty:
            repaired_parts.append(repaired)

    if not repaired_parts:
        return pd.DataFrame()

    repaired_df = pd.concat(repaired_parts).sort_index()
    repaired_df = repaired_df[~repaired_df.index.duplicated(keep="first")]
    return repaired_df


# =========================================
# UPDATE WORKER
# =========================================

def update_symbol(symbol: str) -> str:
    try:
        print(f"\n=== Updating {symbol} ===")
        existing = load_symbol_history(symbol)

        if existing.empty:
            return f"{symbol}: ERROR — existing history is empty."

        last_ts = existing.index.max()
        start_dt = last_ts.tz_convert(dt.timezone.utc) + dt.timedelta(minutes=1)
        end_dt = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)

        if start_dt >= end_dt:
            return f"{symbol}: already up to date. No new rows to fetch."

        new_df = fetch_range(symbol, start_dt, end_dt)
        if new_df.empty:
            return f"{symbol}: no new data returned from Databento."

        # Internal gaps within the fetched data
        missing = validate_new_df(new_df, symbol)

        # Boundary gaps between requested range and fetched data
        expected_start_local = pd.Timestamp(start_dt).tz_convert("America/New_York")
        expected_last_local = pd.Timestamp(end_dt - dt.timedelta(minutes=1)).tz_convert("America/New_York")
        first_idx = new_df.index[0]
        last_idx = new_df.index[-1]

        boundary_missing: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if first_idx > expected_start_local:
            boundary_missing.append((expected_start_local, first_idx - dt.timedelta(minutes=1)))
        if last_idx < expected_last_local:
            boundary_missing.append((last_idx + dt.timedelta(minutes=1), expected_last_local))

        if boundary_missing:
            print(f"{symbol}: Detected {len(boundary_missing)} boundary missing interval(s) in fetched data:")
            for start, end in boundary_missing:
                print(f"  boundary missing {start} → {end}")

        missing = missing + boundary_missing

        if missing:
            repaired = repair_missing_intervals(symbol, missing)
            if not repaired.empty:
                new_df = pd.concat([new_df, repaired]).sort_index()
                new_df = new_df[~new_df.index.duplicated(keep="first")]
                missing = validate_new_df(new_df, symbol)

        if missing:
            return f"{symbol}: incomplete fetch: {len(missing)} missing interval(s) remain after repair."

        combined = pd.concat([existing, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        combined.to_parquet(parquet_path(symbol))
        
        # Calculate and report coverage statistics
        total_span, missing_mins, coverage = calculate_data_coverage(combined)
        print(f"{symbol}: saved updated parquet with {len(new_df)} new rows.")
        print(f"{symbol}: dataset span {total_span:,} min, missing {missing_mins:,} min ({100-coverage:.2f}% gaps, {coverage:.2f}% coverage)")

        return f"{symbol}: updated successfully with no detected gaps in fetched range."

    except Exception as exc:
        return f"{symbol}: ERROR — {exc}"

# =========================================
# TOP-LEVEL PARALLEL UPDATER
# =========================================

def update_historical_data():
    print("\n=== Starting parallel historical data update ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(6, len(SYMBOLS))) as exe:
        futures = {exe.submit(update_symbol, symbol): symbol for symbol in SYMBOLS}
        for fut in as_completed(futures):
            print(fut.result())


if __name__ == "__main__":
    update_historical_data()