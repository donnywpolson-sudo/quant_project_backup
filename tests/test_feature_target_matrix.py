from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from pipeline.features import engine


def _feature_target_df() -> pl.DataFrame:
    start = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    return pl.DataFrame(
        {
            "ts_event": [start + timedelta(minutes=5 * i) for i in range(3)],
            "feature_ret_1": [0.1, 0.2, 0.3],
            "target_sign_4h": [1, 0, 1],
        }
    ).with_columns(
        pl.col("ts_event").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("feature_ret_1").cast(pl.Float32),
        pl.col("target_sign_4h").cast(pl.Int8),
    )


def test_feature_target_matrix_cache_roundtrip(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_generate_features(_df_aligned):
        calls["n"] += 1
        return _feature_target_df()

    monkeypatch.setattr(engine, "generate_features", fake_generate_features)
    cache_path = tmp_path / "full_feature_matrix_test.parquet"

    built = engine.load_or_build_feature_target_matrix(
        pl.DataFrame({"dummy": [1]}),
        cache_path=cache_path,
        target_col="target_sign_4h",
    )

    assert calls["n"] == 1
    assert built.height == 3
    assert cache_path.exists()

    def fail_if_called(_df_aligned):
        raise AssertionError("feature builder should not run when cache exists")

    monkeypatch.setattr(engine, "generate_features", fail_if_called)
    loaded = engine.load_or_build_feature_target_matrix(
        pl.DataFrame({"dummy": [1]}),
        cache_path=cache_path,
        target_col="target_sign_4h",
    )

    assert loaded.height == 3


def test_feature_target_matrix_rejects_null_target():
    bad = _feature_target_df().with_columns(
        pl.Series("target_sign_4h", [1, None, 0], dtype=pl.Int8)
    )
    with pytest.raises(RuntimeError, match="null values in target_sign_4h"):
        engine.validate_feature_target_matrix(bad, target_col="target_sign_4h")
