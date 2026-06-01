from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data import validate_raw_data


def _write_session_config(path: Path) -> None:
    path.write_text(
        """
markets:
  ES:
    timezone: America/Chicago
    week_start_day: Sun
    week_start_time: "17:00"
    week_end_day: Fri
    week_end_time: "16:00"
    daily_break:
      start: "16:00"
      end: "17:00"
    closed_dates: []
    early_closes: {}
    allow_empty_holiday_calendar: true
""".lstrip(),
        encoding="utf-8",
    )


def _write_valid_ohlcv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "ts_event": pd.date_range("2024-01-02 15:00:00Z", periods=3, freq="1min"),
            "open": [100.00, 100.25, 100.50],
            "high": [100.50, 100.75, 101.00],
            "low": [99.75, 100.00, 100.25],
            "close": [100.25, 100.50, 100.75],
            "volume": [10, 11, 12],
        }
    )
    df.to_parquet(path, index=False)


def test_validate_raw_data_smoke_writes_reports_and_validated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_root = tmp_path / "raw"
    config = tmp_path / "market_sessions.yaml"
    out = tmp_path / "reports" / "raw_data"
    validated = tmp_path / "validated"
    _write_valid_ohlcv(raw_root / "ES" / "2024.parquet")
    _write_session_config(config)

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_raw_data.py",
            "--root",
            str(raw_root),
            "--config",
            str(config),
            "--out",
            str(out),
            "--validated-out",
            str(validated),
        ],
    )

    validate_raw_data.main()

    assert (out / "core_summary.csv").exists()
    assert (out / "session_summary.csv").exists()
    assert (validated / "ES" / "2024.parquet").exists()

    core_summary = pd.read_csv(out / "core_summary.csv")
    session_summary = pd.read_csv(out / "session_summary.csv")
    assert len(core_summary) == 1
    assert len(session_summary) == 1
    assert int(session_summary.loc[0, "outside_session_rows"]) == 0
    assert int(session_summary.loc[0, "duplicate_ts"]) == 0


def test_validate_raw_data_smoke_reports_corrupt_parquet_without_crashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_root = tmp_path / "raw"
    config = tmp_path / "market_sessions.yaml"
    out = tmp_path / "reports" / "raw_data"
    bad_file = raw_root / "ES" / "2024.parquet"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_bytes(b"not a parquet file")
    _write_session_config(config)

    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_raw_data.py",
            "--root",
            str(raw_root),
            "--config",
            str(config),
            "--out",
            str(out),
            "--sessions-only",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        validate_raw_data.main()

    assert exc.value.code == 1
    session_summary = pd.read_csv(out / "session_summary.csv")
    assert session_summary.loc[0, "severity"] == "FAIL"
    assert session_summary.loc[0, "check"] == "session_read_parquet"
