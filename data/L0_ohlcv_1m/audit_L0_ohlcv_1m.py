from __future__ import annotations

import argparse
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REQ_COLS = ["ts_event", "open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close"]
NUM_COLS = PRICE_COLS + ["volume"]
DAY_TO_NUM = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


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


def sample_rows(df: pd.DataFrame, n=5) -> str:
    if df.empty:
        return ""
    cols = [c for c in REQ_COLS if c in df.columns]
    return df[cols].head(n).to_json(orient="records", date_format="iso")


def mad_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return 0.6745 * (x - med) / mad


def coerce_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def audit_core_file(path: Path, market: str, year: int):
    issues, gaps, outliers, summary = [], [], [], []

    try:
        raw = pd.read_parquet(path)
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

    df = coerce_ohlcv(raw[REQ_COLS])
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
        issue(issues, "FAIL", market, year, "timestamp_not_minute_aligned", "ts_event not aligned to 1-minute boundary", int(not_minute_aligned.sum()), sample_rows(df[not_minute_aligned]))

    parsed_ts = df["ts_event"].dropna()
    if not parsed_ts.is_monotonic_increasing:
        issue(issues, "FAIL", market, year, "not_sorted", "file order is not increasing by parsed UTC ts_event")

    bad_num = df[NUM_COLS].isna()
    for c in NUM_COLS:
        if bad_num[c].any():
            issue(issues, "FAIL", market, year, f"bad_numeric_{c}", "null or non-numeric", int(bad_num[c].sum()), sample_rows(df[bad_num[c]]))

    finite_mask = np.isfinite(df[NUM_COLS]).all(axis=1)
    if (~finite_mask).any():
        issue(issues, "FAIL", market, year, "non_finite_numeric", "inf/-inf detected", int((~finite_mask).sum()), sample_rows(df[~finite_mask]))

    df = df.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)

    dup = df["ts_event"].duplicated(keep=False)
    if dup.any():
        issue(issues, "FAIL", market, year, "duplicate_ts", "duplicate timestamps", int(dup.sum()), sample_rows(df[dup]))

    wrong_year = df["ts_event"].dt.year != year
    if wrong_year.any():
        issue(issues, "FAIL", market, year, "timestamp_outside_file_year", f"ts_event year != filename year {year}", int(wrong_year.sum()), sample_rows(df[wrong_year]))

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
        issue(issues, "WARN", market, year, "timestamp_gaps_gt_1min", "gaps detected; validate against session calendar", int(gap_mask.sum()), g.head(5).to_json(orient="records", date_format="iso"))

    reverse_or_same = delta_min <= 0
    if reverse_or_same.any():
        issue(issues, "FAIL", market, year, "non_increasing_timestamps", "timestamp diff <= 0", int(reverse_or_same.sum()), sample_rows(df[reverse_or_same]))

    close = df["close"]
    ret = close.pct_change()
    big_ret = ret.abs() > 0.02
    if big_ret.any():
        tmp = df.loc[big_ret, REQ_COLS].copy()
        tmp["market"] = market
        tmp["year"] = year
        tmp["ret"] = ret.loc[big_ret].values
        tmp["check"] = "abs_close_return_gt_2pct"
        outliers.append(tmp)
        issue(issues, "WARN", market, year, "abs_close_return_gt_2pct", "large 1-minute close-to-close move", int(big_ret.sum()), sample_rows(df[big_ret]))

    hl_range = (df["high"] - df["low"]) / df["close"]
    big_range = hl_range > 0.02
    if big_range.any():
        tmp = df.loc[big_range, REQ_COLS].copy()
        tmp["market"] = market
        tmp["year"] = year
        tmp["range_pct"] = hl_range.loc[big_range].values
        tmp["check"] = "hl_range_gt_2pct"
        outliers.append(tmp)
        issue(issues, "WARN", market, year, "hl_range_gt_2pct", "large 1-minute high-low range", int(big_range.sum()), sample_rows(df[big_range]))

    vol_z = mad_z(df["volume"])
    vol_spike = vol_z.abs() > 25
    if vol_spike.any():
        tmp = df.loc[vol_spike, REQ_COLS].copy()
        tmp["market"] = market
        tmp["year"] = year
        tmp["volume_mad_z"] = vol_z.loc[vol_spike].values
        tmp["check"] = "volume_mad_z_gt_25"
        outliers.append(tmp)
        issue(issues, "WARN", market, year, "volume_mad_z_gt_25", "extreme volume spike", int(vol_spike.sum()), sample_rows(df[vol_spike]))

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
    return issues, gaps, outliers, summary, df[REQ_COLS]


def audit_cross_year(frames_by_market):
    rows = []
    for market, items in frames_by_market.items():
        items = sorted(items, key=lambda x: x[0])
        for (year_a, df_a), (year_b, df_b) in zip(items, items[1:]):
            last_a = df_a["ts_event"].max()
            first_b = df_b["ts_event"].min()
            rows.append(
                {
                    "severity": "FAIL" if first_b <= last_a else "INFO",
                    "market": market,
                    "check": "cross_year_overlap" if first_b <= last_a else "cross_year_gap",
                    "year_a": year_a,
                    "year_b": year_b,
                    "last_ts_a": last_a,
                    "first_ts_b": first_b,
                    "gap_minutes": (first_b - last_a).total_seconds() / 60,
                    "detail": "next year starts before or at prior year end" if first_b <= last_a else "cross-year gap; validate against session calendar",
                }
            )
    return rows


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
        return {"severity": "FAIL", "market": market, "year": "", "check": "invalid_market_config", "detail": f"Missing required session config keys: {','.join(sorted(missing))}"}
    if not cfg.get("closed_dates") and not cfg.get("early_closes") and not cfg.get("allow_empty_holiday_calendar", False):
        return {"severity": "FAIL", "market": market, "year": "", "check": "incomplete_session_calendar", "detail": "closed_dates and early_closes are empty; expected-minute counts may be false positives"}
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


def audit_session_file(path: Path, market: str, year: int, cfg: dict):
    print(f"START session audit {market} {year} | {path}", flush=True)
    df = pd.read_parquet(path, columns=["ts_event"])
    ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dropna()
    ts = pd.DatetimeIndex(ts).sort_values().unique()
    if len(ts) == 0:
        return ([{"severity": "FAIL", "market": market, "year": year, "check": "empty_or_bad_ts", "n": 0}], pd.DataFrame(), pd.DataFrame())

    tz = cfg["timezone"]
    local_ts = ts.tz_convert(tz)
    outside = pd.DataFrame({"ts_event": ts, "ts_local": local_ts})[~is_active_local_index(local_ts, cfg).to_numpy()]

    expected_utc = pd.date_range(pd.Timestamp(f"{year}-01-01T00:00:00Z"), pd.Timestamp(f"{year + 1}-01-01T00:00:00Z"), freq="1min", inclusive="left")
    expected_utc = expected_utc[is_active_local_index(expected_utc.tz_convert(tz), cfg).to_numpy()]
    missing = expected_utc.difference(pd.DatetimeIndex(ts))

    missing_df = pd.DataFrame({"market": market, "year": year, "missing_ts_event": missing, "missing_ts_local": missing.tz_convert(tz)})
    outside_df = outside.copy()
    outside_df.insert(0, "year", year)
    outside_df.insert(0, "market", market)

    missing_n = len(missing_df)
    outside_n = len(outside_df)
    severity = "PASS" if missing_n == 0 and outside_n == 0 else "FAIL" if missing_n > 0 else "WARN"
    summary = {
        "severity": severity,
        "market": market,
        "year": year,
        "check": "session_calendar",
        "actual_rows": len(ts),
        "expected_session_minutes": len(expected_utc),
        "missing_expected_minutes": missing_n,
        "outside_session_rows": outside_n,
        "first_ts": ts.min(),
        "last_ts": ts.max(),
    }
    print(f"DONE  session audit {market} {year} | actual={len(ts):,} expected={len(expected_utc):,} missing={missing_n:,} outside={outside_n:,}", flush=True)
    return [summary], missing_df, outside_df


def find_parquet_files(root: Path) -> list[Path]:
    for candidate in [root, root / "L0_ohlcv_1m", root / "ohlcv_1m"]:
        if candidate.exists():
            files = sorted(candidate.glob("*/*.parquet"))
            if files:
                return files
    raise SystemExit(f"No parquet files found under {root}/{{market}}/{{year}}.parquet")


def run_core_audit(files: list[Path], out: Path) -> tuple[int, int]:
    all_issues, all_gaps, all_outliers, all_summary = [], [], [], []
    frames_by_market = {}
    for path in files:
        market = path.parent.name
        try:
            year = int(path.stem)
        except ValueError:
            issue(all_issues, "FAIL", market, path.stem, "bad_filename", "filename stem is not an integer year", sample=str(path))
            continue
        issues, gaps, outliers, summary, df = audit_core_file(path, market, year)
        all_issues.extend(issues)
        all_gaps.extend(gaps)
        all_outliers.extend(outliers)
        all_summary.extend(summary)
        if df is not None:
            frames_by_market.setdefault(market, []).append((year, df))

    issues_df = pd.DataFrame(all_issues)
    pd.DataFrame(all_summary).to_csv(out / "core_summary.csv", index=False)
    issues_df.to_csv(out / "core_issues.csv", index=False)
    (pd.concat(all_gaps, ignore_index=True) if all_gaps else pd.DataFrame()).to_csv(out / "core_gaps.csv", index=False)
    (pd.concat(all_outliers, ignore_index=True) if all_outliers else pd.DataFrame()).to_csv(out / "core_outliers.csv", index=False)
    pd.DataFrame(audit_cross_year(frames_by_market)).to_csv(out / "core_cross_year.csv", index=False)

    return (
        0 if issues_df.empty else int((issues_df["severity"] == "FAIL").sum()),
        0 if issues_df.empty else int((issues_df["severity"] == "WARN").sum()),
    )


def run_session_audit(files: list[Path], out: Path, config_path: Path) -> tuple[int, int]:
    markets_cfg = load_session_config(config_path)["markets"]
    summaries, issues, missing_all, outside_all = [], [], [], []
    files_by_market = {}
    for path in files:
        files_by_market.setdefault(path.parent.name, []).append(path)

    valid_markets = set()
    for market in sorted(files_by_market):
        if market not in markets_cfg:
            issues.append({"severity": "FAIL", "market": market, "year": "", "check": "missing_market_config", "detail": f"No session config for market {market}"})
            continue
        config_issue = validate_market_config(market, markets_cfg[market])
        if config_issue is not None:
            issues.append(config_issue)
            continue
        valid_markets.add(market)

    for path in files:
        market = path.parent.name
        if market not in valid_markets:
            continue
        try:
            year = int(path.stem)
        except ValueError:
            issues.append({"severity": "FAIL", "market": market, "year": path.stem, "check": "bad_filename", "detail": "filename stem is not integer year"})
            continue
        summary, missing_df, outside_df = audit_session_file(path, market, year, markets_cfg[market])
        summaries.extend(summary)
        if not missing_df.empty:
            missing_all.append(missing_df)
        if not outside_df.empty:
            outside_all.append(outside_df)

    issues_df = pd.DataFrame(issues)
    summaries_df = pd.DataFrame(summaries)
    summaries_df.to_csv(out / "session_summary.csv", index=False)
    issues_df.to_csv(out / "session_issues.csv", index=False)
    (pd.concat(missing_all, ignore_index=True) if missing_all else pd.DataFrame()).to_csv(out / "session_missing_expected_minutes.csv", index=False)
    (pd.concat(outside_all, ignore_index=True) if outside_all else pd.DataFrame()).to_csv(out / "session_outside_rows.csv", index=False)

    fail_issues = 0 if issues_df.empty else int((issues_df["severity"] == "FAIL").sum())
    fail_summaries = 0 if summaries_df.empty else int((summaries_df["severity"] == "FAIL").sum())
    warn_summaries = 0 if summaries_df.empty else int((summaries_df["severity"] == "WARN").sum())
    return fail_issues + fail_summaries, warn_summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(SCRIPT_DIR))
    ap.add_argument("--config", default=str(PROJECT_ROOT / "data" / "market_sessions.yaml"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "audits" / "L0_ohlcv_1m_audit"))
    ap.add_argument("--core-only", action="store_true")
    ap.add_argument("--sessions-only", action="store_true")
    args = ap.parse_args()

    if args.core_only and args.sessions_only:
        raise SystemExit("ERROR: choose at most one of --core-only or --sessions-only")

    files = find_parquet_files(Path(args.root))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    core_fail = core_warn = session_fail = session_warn = 0
    if not args.sessions_only:
        core_fail, core_warn = run_core_audit(files, out)
    if not args.core_only:
        session_fail, session_warn = run_session_audit(files, out, Path(args.config))

    print(f"Wrote reports to: {out}")
    print(f"Files scanned: {len(files)}")
    print(f"Core FAIL/WARN: {core_fail}/{core_warn}")
    print(f"Session FAIL/WARN: {session_fail}/{session_warn}")
    if core_fail or session_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
