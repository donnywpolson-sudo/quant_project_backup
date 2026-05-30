# scripts/audit_core_data.py

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQ_COLS = ["ts_event", "open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close"]
NUM_COLS = PRICE_COLS + ["volume"]


def issue(rows, severity, market, year, check, detail, n=None, sample=None):
    rows.append(
        {
            "severity": severity,
            "market": market,
            "year": year,
            "check": check,
            "detail": detail,
            "n": n,
            "sample": sample,
        }
    )


def read_file(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def sample_rows(df: pd.DataFrame, n=5) -> str:
    if df.empty:
        return ""
    cols = [c for c in ["ts_event", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols].head(n).to_json(orient="records", date_format="iso")


def mad_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad


def audit_one(path: Path, market: str, year: int):
    issues = []
    gaps = []
    outliers = []
    summary = []

    try:
        raw = read_file(path)
    except Exception as e:
        issue(issues, "FAIL", market, year, "read_parquet", repr(e))
        return issues, gaps, outliers, summary, None

    missing = [c for c in REQ_COLS if c not in raw.columns]
    extra = [c for c in raw.columns if c not in REQ_COLS]

    if missing:
        issue(issues, "FAIL", market, year, "schema_missing_cols", ",".join(missing))
        return issues, gaps, outliers, summary, None

    if extra:
        issue(issues, "WARN", market, year, "schema_extra_cols", ",".join(extra))

    df = coerce(raw[REQ_COLS])

    n = len(df)
    if n == 0:
        issue(issues, "FAIL", market, year, "empty_file", "no rows")
        return issues, gaps, outliers, summary, None

    bad_ts = df["ts_event"].isna()
    if bad_ts.any():
        issue(issues, "FAIL", market, year, "bad_ts_parse", "unparseable ts_event", int(bad_ts.sum()), sample_rows(df[bad_ts]))

    valid_ts = df["ts_event"].notna()
    not_minute_aligned = valid_ts & (df["ts_event"] != df["ts_event"].dt.floor("min"))
    if not_minute_aligned.any():
        issue(
            issues,
            "FAIL",
            market,
            year,
            "timestamp_not_minute_aligned",
            "ts_event is not aligned to exact 1-minute boundary",
            int(not_minute_aligned.sum()),
            sample_rows(df[not_minute_aligned]),
        )

    parsed_ts = df["ts_event"].dropna()
    if not parsed_ts.is_monotonic_increasing:
        issue(issues, "FAIL", market, year, "not_sorted", "file order is not increasing by parsed UTC ts_event")

    bad_num = df[NUM_COLS].isna()
    if bad_num.any().any():
        for c in NUM_COLS:
            m = bad_num[c]
            if m.any():
                issue(issues, "FAIL", market, year, f"bad_numeric_{c}", "null or non-numeric", int(m.sum()), sample_rows(df[m]))

    finite_mask = np.isfinite(df[NUM_COLS]).all(axis=1)
    if (~finite_mask).any():
        issue(issues, "FAIL", market, year, "non_finite_numeric", "inf/-inf detected", int((~finite_mask).sum()), sample_rows(df[~finite_mask]))

    df = df.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)

    dup = df["ts_event"].duplicated(keep=False)
    if dup.any():
        issue(issues, "FAIL", market, year, "duplicate_ts", "duplicate timestamps", int(dup.sum()), sample_rows(df[dup]))

    wrong_year = df["ts_event"].dt.year != year
    if wrong_year.any():
        issue(
            issues,
            "FAIL",
            market,
            year,
            "timestamp_outside_file_year",
            f"ts_event year != filename year {year}",
            int(wrong_year.sum()),
            sample_rows(df[wrong_year]),
        )

    # OHLC integrity
    checks = {
        "high_lt_low": df["high"] < df["low"],
        "high_lt_open": df["high"] < df["open"],
        "high_lt_close": df["high"] < df["close"],
        "low_gt_open": df["low"] > df["open"],
        "low_gt_close": df["low"] > df["close"],
        "nonpositive_open": df["open"] <= 0,
        "nonpositive_high": df["high"] <= 0,
        "nonpositive_low": df["low"] <= 0,
        "nonpositive_close": df["close"] <= 0,
        "negative_volume": df["volume"] < 0,
    }

    for name, mask in checks.items():
        if mask.any():
            issue(issues, "FAIL", market, year, name, "OHLCV invariant violation", int(mask.sum()), sample_rows(df[mask]))

    zero_vol = df["volume"] == 0
    if zero_vol.any():
        issue(issues, "WARN", market, year, "zero_volume", "zero-volume bars", int(zero_vol.sum()), sample_rows(df[zero_vol]))

    zero_range = df["high"] == df["low"]
    if zero_range.any():
        issue(issues, "WARN", market, year, "zero_range", "high == low bars", int(zero_range.sum()), sample_rows(df[zero_range]))

    # Timestamp gaps.
    # Futures data may have valid session gaps, so record gaps instead of failing by default.
    ts = df["ts_event"]
    delta_min = ts.diff().dt.total_seconds().div(60)
    gap_mask = delta_min > 1

    if gap_mask.any():
        g = df.loc[gap_mask, ["ts_event"]].copy()
        g["prev_ts"] = ts.shift(1).loc[gap_mask].values
        g["gap_minutes"] = delta_min.loc[gap_mask].values
        g["market"] = market
        g["year"] = year
        gaps.append(g[["market", "year", "prev_ts", "ts_event", "gap_minutes"]])

        issue(
            issues,
            "WARN",
            market,
            year,
            "timestamp_gaps_gt_1min",
            "gaps detected; validate against exchange/session calendar",
            int(gap_mask.sum()),
            g.head(5).to_json(orient="records", date_format="iso"),
        )

    reverse_or_same = delta_min <= 0
    if reverse_or_same.any():
        issue(
            issues,
            "FAIL",
            market,
            year,
            "non_increasing_timestamps",
            "timestamp diff <= 0",
            int(reverse_or_same.sum()),
            sample_rows(df[reverse_or_same]),
        )

    # Outlier diagnostics.
    close = df["close"]
    ret = close.pct_change()
    abs_ret = ret.abs()

    big_ret = abs_ret > 0.02
    if big_ret.any():
        tmp = df.loc[big_ret, ["ts_event", "open", "high", "low", "close", "volume"]].copy()
        tmp["market"] = market
        tmp["year"] = year
        tmp["ret"] = ret.loc[big_ret].values
        tmp["check"] = "abs_close_return_gt_2pct"
        outliers.append(tmp)

        issue(
            issues,
            "WARN",
            market,
            year,
            "abs_close_return_gt_2pct",
            "large 1-minute close-to-close move",
            int(big_ret.sum()),
            sample_rows(df[big_ret]),
        )

    hl_range = (df["high"] - df["low"]) / df["close"]
    big_range = hl_range > 0.02
    if big_range.any():
        tmp = df.loc[big_range, ["ts_event", "open", "high", "low", "close", "volume"]].copy()
        tmp["market"] = market
        tmp["year"] = year
        tmp["range_pct"] = hl_range.loc[big_range].values
        tmp["check"] = "hl_range_gt_2pct"
        outliers.append(tmp)

        issue(
            issues,
            "WARN",
            market,
            year,
            "hl_range_gt_2pct",
            "large 1-minute high-low range",
            int(big_range.sum()),
            sample_rows(df[big_range]),
        )

    vol_z = mad_z(df["volume"])
    vol_spike = vol_z.abs() > 25
    if vol_spike.any():
        tmp = df.loc[vol_spike, ["ts_event", "open", "high", "low", "close", "volume"]].copy()
        tmp["market"] = market
        tmp["year"] = year
        tmp["volume_mad_z"] = vol_z.loc[vol_spike].values
        tmp["check"] = "volume_mad_z_gt_25"
        outliers.append(tmp)

        issue(
            issues,
            "WARN",
            market,
            year,
            "volume_mad_z_gt_25",
            "extreme volume spike",
            int(vol_spike.sum()),
            sample_rows(df[vol_spike]),
        )

    summary.append(
        {
            "market": market,
            "year": year,
            "path": str(path),
            "rows": n,
            "first_ts": df["ts_event"].min(),
            "last_ts": df["ts_event"].max(),
            "duplicate_ts": int(dup.sum()),
            "gap_count_gt_1min": int(gap_mask.sum()),
            "zero_volume": int(zero_vol.sum()),
            "zero_range": int(zero_range.sum()),
            "close_min": float(df["close"].min()),
            "close_max": float(df["close"].max()),
            "volume_sum": float(df["volume"].sum()),
        }
    )

    return issues, gaps, outliers, summary, df[["ts_event", "open", "high", "low", "close", "volume"]]


def audit_cross_year(frames_by_market):
    rows = []

    for market, items in frames_by_market.items():
        items = sorted(items, key=lambda x: x[0])

        for (year_a, df_a), (year_b, df_b) in zip(items, items[1:]):
            last_a = df_a["ts_event"].max()
            first_b = df_b["ts_event"].min()

            if first_b <= last_a:
                rows.append(
                    {
                        "severity": "FAIL",
                        "market": market,
                        "check": "cross_year_overlap",
                        "year_a": year_a,
                        "year_b": year_b,
                        "last_ts_a": last_a,
                        "first_ts_b": first_b,
                        "detail": "next year starts before or at prior year end",
                    }
                )
            else:
                gap_min = (first_b - last_a).total_seconds() / 60
                rows.append(
                    {
                        "severity": "INFO",
                        "market": market,
                        "check": "cross_year_gap",
                        "year_a": year_a,
                        "year_b": year_b,
                        "last_ts_a": last_a,
                        "first_ts_b": first_b,
                        "gap_minutes": gap_min,
                        "detail": "cross-year gap; validate against session calendar",
                    }
                )

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="output/audits/data_audit")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_issues = []
    all_gaps = []
    all_outliers = []
    all_summary = []
    frames_by_market = {}

    ohlcv_root = root / "ohlcv_1m"
    files = sorted(ohlcv_root.glob("*/*.parquet")) if ohlcv_root.exists() else []
    if not files:
        files = sorted(root.glob("*/*.parquet"))

    if not files:
        raise SystemExit(
            f"No parquet files found under {root}/ohlcv_1m/{{market}}/{{year}}.parquet "
            f"or {root}/{{market}}/{{year}}.parquet"
        )

    for path in files:
        market = path.parent.name

        try:
            year = int(path.stem)
        except ValueError:
            all_issues.append(
                {
                    "severity": "FAIL",
                    "market": market,
                    "year": path.stem,
                    "check": "bad_filename",
                    "detail": "filename stem is not an integer year",
                    "n": None,
                    "sample": str(path),
                }
            )
            continue

        issues, gaps, outliers, summary, df = audit_one(path, market, year)

        all_issues.extend(issues)
        all_gaps.extend(gaps)
        all_outliers.extend(outliers)
        all_summary.extend(summary)

        if df is not None:
            frames_by_market.setdefault(market, []).append((year, df))

    cross_year = audit_cross_year(frames_by_market)

    issues_df = pd.DataFrame(all_issues)
    summary_df = pd.DataFrame(all_summary)
    gaps_df = pd.concat(all_gaps, ignore_index=True) if all_gaps else pd.DataFrame()
    outliers_df = pd.concat(all_outliers, ignore_index=True) if all_outliers else pd.DataFrame()
    cross_df = pd.DataFrame(cross_year)

    summary_df.to_csv(out / "summary.csv", index=False)
    issues_df.to_csv(out / "issues.csv", index=False)
    gaps_df.to_csv(out / "gaps.csv", index=False)
    outliers_df.to_csv(out / "outliers.csv", index=False)
    cross_df.to_csv(out / "cross_year.csv", index=False)

    fail_count = 0 if issues_df.empty else int((issues_df["severity"] == "FAIL").sum())
    warn_count = 0 if issues_df.empty else int((issues_df["severity"] == "WARN").sum())

    print(f"Wrote reports to: {out}")
    print(f"Files scanned: {len(files)}")
    print(f"FAIL checks: {fail_count}")
    print(f"WARN checks: {warn_count}")

    if fail_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
