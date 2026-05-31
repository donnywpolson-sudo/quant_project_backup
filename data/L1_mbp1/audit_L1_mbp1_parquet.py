from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


MONTHLY_RE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})\.parquet$")
REQ_COLS = ["ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
NUM_COLS = ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
OPTIONAL_COLS = ["instrument_id", "action", "side", "depth", "price", "size", "sequence", "symbol"]
DAY_TO_NUM = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def issue(rows, severity, market, file, check, detail, n=None, sample=None):
    rows.append(
        {
            "severity": severity,
            "market": market,
            "file": file,
            "check": check,
            "detail": detail,
            "n": n,
            "sample": sample,
        }
    )


def sample_rows(df: pd.DataFrame, cols: list[str], n=5) -> str:
    if df.empty:
        return ""
    use_cols = [c for c in cols if c in df.columns]
    return df[use_cols].head(n).to_json(orient="records", date_format="iso")


def parse_monthly_file(path: Path):
    m = MONTHLY_RE.match(path.name)
    if not m:
        return None
    start = pd.Timestamp(year=int(m.group("year")), month=int(m.group("month")), day=1, tz="UTC")
    return path.parent.name, start, start + pd.DateOffset(months=1)


def find_parquet_files(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/*.parquet") if MONTHLY_RE.match(p.name) and not p.name.startswith(".tmp_"))


def parse_hhmm(x: str) -> time:
    h, m = map(int, x.split(":"))
    return time(h, m)


def load_session_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_market_config(market: str, cfg: dict) -> dict | None:
    missing = []
    for key in ("timezone", "week_start_day", "week_start_time", "week_end_day", "week_end_time", "closed_dates", "early_closes"):
        if key not in cfg:
            missing.append(key)
    if "daily_break" not in cfg:
        missing.append("daily_break")
    else:
        for key in ("start", "end"):
            if key not in cfg["daily_break"]:
                missing.append(f"daily_break.{key}")
    if missing:
        return {"severity": "FAIL", "market": market, "file": "", "check": "invalid_market_config", "detail": f"Missing keys: {','.join(sorted(missing))}"}
    return None


def is_active_local_index(local_ts: pd.DatetimeIndex, cfg: dict) -> pd.Series:
    week_start_day = DAY_TO_NUM[cfg["week_start_day"]]
    week_end_day = DAY_TO_NUM[cfg["week_end_day"]]
    week_start_time = parse_hhmm(cfg["week_start_time"])
    week_end_time = parse_hhmm(cfg["week_end_time"])
    break_start = parse_hhmm(cfg["daily_break"]["start"])
    break_end = parse_hhmm(cfg["daily_break"]["end"])

    s = pd.Series(local_ts, index=local_ts)
    wd = s.dt.weekday
    tt = s.dt.time
    dstr = s.dt.strftime("%Y-%m-%d")

    active = pd.Series(False, index=local_ts)
    active |= (wd == week_start_day) & (tt >= week_start_time)
    active |= wd.isin([0, 1, 2, 3])
    active |= (wd == week_end_day) & (tt < week_end_time)
    active &= ~((tt >= break_start) & (tt < break_end))

    closed_dates = set(str(x) for x in cfg.get("closed_dates", []))
    if closed_dates:
        active &= ~dstr.isin(closed_dates)
    for date_str, close_hhmm in (cfg.get("early_closes", {}) or {}).items():
        close_t = parse_hhmm(str(close_hhmm))
        active &= ~((dstr == str(date_str)) & (tt >= close_t))
    return active


def audit_file(path: Path, market_cfg: dict | None):
    parsed = parse_monthly_file(path)
    market = path.parent.name
    issues, outliers, summary = [], [], []

    if parsed is None:
        issue(issues, "FAIL", market, path.name, "bad_filename", "expected YYYY-MM.parquet")
        return issues, outliers, summary, pd.DataFrame(), pd.DataFrame()

    file_market, file_start, file_end = parsed
    if file_market != market:
        issue(issues, "FAIL", market, path.name, "market_folder_mismatch", f"parsed market={file_market} folder={market}")

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        issue(issues, "FAIL", market, path.name, "read_parquet", repr(exc))
        return issues, outliers, summary, pd.DataFrame(), pd.DataFrame()

    n = len(df)
    if n == 0:
        issue(issues, "FAIL", market, path.name, "empty_file", "zero rows")
        return issues, outliers, summary, pd.DataFrame(), pd.DataFrame()

    missing_cols = [c for c in REQ_COLS if c not in df.columns]
    if missing_cols:
        issue(issues, "FAIL", market, path.name, "schema_missing_cols", ",".join(missing_cols), sample=",".join(df.columns))
        return issues, outliers, summary, pd.DataFrame(), pd.DataFrame()

    extra_core_missing = [c for c in OPTIONAL_COLS if c not in df.columns]
    if extra_core_missing:
        issue(issues, "WARN", market, path.name, "optional_cols_missing", ",".join(extra_core_missing))

    df = df.copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad_ts = df["ts_event"].isna()
    if bad_ts.any():
        issue(issues, "FAIL", market, path.name, "bad_ts_parse", "unparseable ts_event", int(bad_ts.sum()), sample_rows(df[bad_ts], REQ_COLS))

    for c in NUM_COLS:
        bad = df[c].isna()
        if bad.any():
            issue(issues, "FAIL", market, path.name, f"bad_numeric_{c}", "null or non-numeric", int(bad.sum()), sample_rows(df[bad], REQ_COLS))

    finite = np.isfinite(df[NUM_COLS]).all(axis=1)
    if (~finite).any():
        issue(issues, "FAIL", market, path.name, "non_finite_numeric", "inf/-inf detected", int((~finite).sum()), sample_rows(df[~finite], REQ_COLS))

    parsed_ts = df["ts_event"].dropna()
    if not parsed_ts.is_monotonic_increasing:
        issue(issues, "FAIL", market, path.name, "not_sorted", "file order is not increasing by ts_event")

    df = df.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)

    outside_file = (df["ts_event"] < file_start) | (df["ts_event"] >= file_end)
    if outside_file.any():
        issue(issues, "FAIL", market, path.name, "timestamp_outside_file_month", f"expected [{file_start}, {file_end})", int(outside_file.sum()), sample_rows(df[outside_file], REQ_COLS))

    dup_all = df.duplicated().sum()
    if dup_all:
        issue(issues, "WARN", market, path.name, "duplicate_full_rows", "duplicate rows", int(dup_all))

    if "sequence" in df.columns:
        dup_seq = df.duplicated(subset=["ts_event", "sequence"]).sum()
        if dup_seq:
            issue(issues, "WARN", market, path.name, "duplicate_ts_sequence", "duplicate ts_event+sequence", int(dup_seq))

    checks = {
        "nonpositive_bid_px_00": df["bid_px_00"] <= 0,
        "nonpositive_ask_px_00": df["ask_px_00"] <= 0,
        "negative_bid_sz_00": df["bid_sz_00"] < 0,
        "negative_ask_sz_00": df["ask_sz_00"] < 0,
        "crossed_book_bid_gt_ask": df["bid_px_00"] > df["ask_px_00"],
    }
    for name, mask in checks.items():
        if mask.any():
            issue(issues, "FAIL", market, path.name, name, "L1 invariant violation", int(mask.sum()), sample_rows(df[mask], REQ_COLS))

    locked = df["bid_px_00"] == df["ask_px_00"]
    if locked.any():
        issue(issues, "WARN", market, path.name, "locked_book_bid_eq_ask", "bid equals ask", int(locked.sum()), sample_rows(df[locked], REQ_COLS))

    zero_size = (df["bid_sz_00"] == 0) | (df["ask_sz_00"] == 0)
    if zero_size.any():
        issue(issues, "WARN", market, path.name, "zero_top_size", "bid or ask size is zero", int(zero_size.sum()), sample_rows(df[zero_size], REQ_COLS))

    spread = df["ask_px_00"] - df["bid_px_00"]
    mid = (df["ask_px_00"] + df["bid_px_00"]) / 2
    spread_pct = spread / mid.replace(0, np.nan)
    wide = spread_pct > 0.01
    if wide.any():
        tmp = df.loc[wide, REQ_COLS].copy()
        tmp["market"] = market
        tmp["file"] = path.name
        tmp["spread_pct"] = spread_pct.loc[wide].values
        tmp["check"] = "spread_gt_1pct_mid"
        outliers.append(tmp)
        issue(issues, "WARN", market, path.name, "spread_gt_1pct_mid", "spread > 1% of mid", int(wide.sum()), sample_rows(tmp, REQ_COLS + ["spread_pct"]))

    delta_sec = df["ts_event"].diff().dt.total_seconds()
    non_increasing = delta_sec <= 0
    if non_increasing.any():
        issue(issues, "FAIL", market, path.name, "non_increasing_timestamps", "timestamp diff <= 0 after sort indicates duplicate timestamps", int(non_increasing.sum()), sample_rows(df[non_increasing], REQ_COLS))

    large_gaps = delta_sec > 3600
    if large_gaps.any():
        issue(issues, "WARN", market, path.name, "event_gaps_gt_1h", "large event gap; validate against session calendar", int(large_gaps.sum()), sample_rows(df[large_gaps], REQ_COLS))

    missing_df = pd.DataFrame()
    outside_df = pd.DataFrame()
    missing_n = outside_n = expected_n = actual_minutes_n = 0
    if market_cfg is not None:
        tz = market_cfg["timezone"]
        ts = pd.DatetimeIndex(df["ts_event"]).sort_values().unique()
        local_ts = ts.tz_convert(tz)
        outside_df = pd.DataFrame({"market": market, "file": path.name, "ts_event": ts, "ts_local": local_ts})[
            ~is_active_local_index(local_ts, market_cfg).to_numpy()
        ]

        expected_utc = pd.date_range(file_start, file_end, freq="1min", inclusive="left")
        expected_utc = expected_utc[is_active_local_index(expected_utc.tz_convert(tz), market_cfg).to_numpy()]
        actual_minutes = pd.DatetimeIndex(ts.floor("min")).unique()
        missing = expected_utc.difference(actual_minutes)
        missing_df = pd.DataFrame(
            {
                "market": market,
                "file": path.name,
                "missing_ts_event": missing,
                "missing_ts_local": missing.tz_convert(tz),
            }
        )
        missing_n = len(missing_df)
        outside_n = len(outside_df)
        expected_n = len(expected_utc)
        actual_minutes_n = len(actual_minutes)

    severity = "PASS" if not any(r["severity"] == "FAIL" for r in issues) and missing_n == 0 else "FAIL"
    summary.append(
        {
            "severity": severity,
            "market": market,
            "file": path.name,
            "rows": n,
            "first_ts": df["ts_event"].min(),
            "last_ts": df["ts_event"].max(),
            "expected_session_minutes": expected_n,
            "actual_event_minutes": actual_minutes_n,
            "missing_expected_minutes": missing_n,
            "outside_session_events": outside_n,
            "locked_book": int(locked.sum()),
            "zero_top_size": int(zero_size.sum()),
            "crossed_book": int((df["bid_px_00"] > df["ask_px_00"]).sum()),
            "median_spread": float(spread.median()),
            "max_spread": float(spread.max()),
        }
    )
    return issues, outliers, summary, missing_df, outside_df


def audit_monthly_coverage(files: list[Path]) -> list[dict]:
    rows = []
    by_market = defaultdict(set)
    for path in files:
        parsed = parse_monthly_file(path)
        if parsed:
            market, start, _ = parsed
            by_market[market].add(start)
    for market, months in by_market.items():
        expected = pd.date_range(min(months), max(months), freq="MS")
        for month in sorted(set(expected) - months):
            issue(rows, "FAIL", market, "", "missing_monthly_file", f"missing {month.strftime('%Y-%m')}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "market_sessions.yaml"))
    ap.add_argument("--out", default="output/audits/L1_mbp1_parquet_audit")
    ap.add_argument("--markets", nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    markets = {m.upper() for m in args.markets} if args.markets else None
    files = find_parquet_files(root)
    if markets:
        files = [p for p in files if p.parent.name.upper() in markets]
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"ERROR: no monthly parquet files found under {root}\\{{market}}\\YYYY-MM.parquet")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    markets_cfg = {}
    config_issues = []
    if Path(args.config).exists():
        markets_cfg = load_session_config(Path(args.config)).get("markets", {})
        for market in sorted({p.parent.name for p in files}):
            if market not in markets_cfg:
                issue(config_issues, "FAIL", market, "", "missing_market_config", f"No session config for {market}")
            else:
                cfg_issue = validate_market_config(market, markets_cfg[market])
                if cfg_issue:
                    config_issues.append(cfg_issue)
    else:
        issue(config_issues, "WARN", "", "", "missing_session_config_file", str(args.config))

    all_issues = audit_monthly_coverage(files) + config_issues
    all_outliers, all_summary, all_missing, all_outside = [], [], [], []

    for path in files:
        market = path.parent.name
        print(f"START L1 parquet audit {market} | {path}", flush=True)
        cfg = markets_cfg.get(market)
        issues, outliers, summary, missing_df, outside_df = audit_file(path, cfg)
        all_issues.extend(issues)
        all_outliers.extend(outliers)
        all_summary.extend(summary)
        if not missing_df.empty:
            all_missing.append(missing_df)
        if not outside_df.empty:
            all_outside.append(outside_df)
        print(f"DONE  L1 parquet audit {market} | {path.name}", flush=True)

    issues_df = pd.DataFrame(all_issues)
    summary_df = pd.DataFrame(all_summary)
    outliers_df = pd.concat(all_outliers, ignore_index=True) if all_outliers else pd.DataFrame()
    missing_df = pd.concat(all_missing, ignore_index=True) if all_missing else pd.DataFrame()
    outside_df = pd.concat(all_outside, ignore_index=True) if all_outside else pd.DataFrame()

    summary_df.to_csv(out / "summary.csv", index=False)
    issues_df.to_csv(out / "issues.csv", index=False)
    outliers_df.to_csv(out / "outliers.csv", index=False)
    missing_df.to_csv(out / "missing_expected_minutes.csv", index=False)
    outside_df.to_csv(out / "outside_session_events.csv", index=False)

    fail_count = 0 if issues_df.empty else int((issues_df["severity"] == "FAIL").sum())
    warn_count = 0 if issues_df.empty else int((issues_df["severity"] == "WARN").sum())
    summary_fail = 0 if summary_df.empty else int((summary_df["severity"] == "FAIL").sum())

    print(f"Wrote reports to: {out}")
    print(f"Files scanned: {len(files)}")
    print(f"FAIL checks: {fail_count + summary_fail}")
    print(f"WARN checks: {warn_count}")

    if fail_count or summary_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
