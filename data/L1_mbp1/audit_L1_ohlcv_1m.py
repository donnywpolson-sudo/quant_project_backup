from __future__ import annotations

import argparse
import re
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import databento as db
except ImportError:
    db = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
FILENAME_RE = re.compile(
    r"^(?P<market>[A-Z0-9]+)_front_mbp1_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.dbn\.zst$"
)
REQ_COLS = ["ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
DAY_TO_NUM = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def issue(rows, severity, market, file, check, detail, n=None, sample=None):
    rows.append(
        {
            "severity": severity,
            "market": market,
            "file": str(file),
            "check": check,
            "detail": detail,
            "n": n,
            "sample": sample,
        }
    )


def parse_file(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return (
        m.group("market"),
        pd.Timestamp(m.group("start"), tz="UTC"),
        pd.Timestamp(m.group("end"), tz="UTC"),
    )


def sample_rows(df: pd.DataFrame, cols: list[str], n=5) -> str:
    if df.empty:
        return ""
    return df[[c for c in cols if c in df.columns]].head(n).to_json(orient="records", date_format="iso")


def read_dbn(path: Path) -> pd.DataFrame:
    if db is None:
        raise ImportError("databento is not installed")
    return db.DBNStore.from_file(str(path)).to_df()


def normalize_l1(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[REQ_COLS].copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    for c in REQ_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def audit_core_df(path: Path, raw: pd.DataFrame):
    parsed = parse_file(path)
    market = path.parent.name
    issues, gaps, outliers, summary = [], [], [], []

    if parsed is None:
        issue(issues, "FAIL", market, path.name, "bad_filename", "expected {market}_front_mbp1_YYYY-MM-DD_YYYY-MM-DD.dbn.zst")
        return issues, gaps, outliers, summary, None

    file_market, file_start, file_end = parsed
    if file_market != market:
        issue(issues, "FAIL", market, path.name, "market_folder_mismatch", f"filename market={file_market} folder={market}")

    if raw.empty:
        issue(issues, "FAIL", market, path.name, "empty_decoded_data", "DBN decoded to zero rows")
        return issues, gaps, outliers, summary, None

    missing = [c for c in REQ_COLS if c not in raw.columns]
    if missing:
        issue(issues, "FAIL", market, path.name, "schema_missing_cols", ",".join(missing), sample=",".join(raw.columns))
        return issues, gaps, outliers, summary, None

    df = normalize_l1(raw)
    n = len(df)

    bad_ts = df["ts_event"].isna()
    if bad_ts.any():
        issue(issues, "FAIL", market, path.name, "bad_ts_parse", "unparseable ts_event", int(bad_ts.sum()), sample_rows(df[bad_ts], REQ_COLS))

    bad_num = df[REQ_COLS[1:]].isna()
    for c in REQ_COLS[1:]:
        if bad_num[c].any():
            issue(issues, "FAIL", market, path.name, f"bad_numeric_{c}", "null or non-numeric", int(bad_num[c].sum()), sample_rows(df[bad_num[c]], REQ_COLS))

    finite = np.isfinite(df[REQ_COLS[1:]]).all(axis=1)
    if (~finite).any():
        issue(issues, "FAIL", market, path.name, "non_finite_numeric", "inf/-inf detected", int((~finite).sum()), sample_rows(df[~finite], REQ_COLS))

    if not df["ts_event"].is_monotonic_increasing:
        issue(issues, "FAIL", market, path.name, "not_sorted", "file order is not increasing by ts_event")

    df = df.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)

    outside = (df["ts_event"] < file_start) | (df["ts_event"] >= file_end)
    if outside.any():
        issue(issues, "FAIL", market, path.name, "timestamp_outside_filename_window", f"expected [{file_start}, {file_end})", int(outside.sum()), sample_rows(df[outside], REQ_COLS))

    checks = {
        "nonpositive_bid_px_00": df["bid_px_00"] <= 0,
        "nonpositive_ask_px_00": df["ask_px_00"] <= 0,
        "negative_bid_sz_00": df["bid_sz_00"] < 0,
        "negative_ask_sz_00": df["ask_sz_00"] < 0,
        "crossed_book_bid_gt_ask": df["bid_px_00"] > df["ask_px_00"],
    }
    for name, mask in checks.items():
        if mask.any():
            issue(issues, "FAIL", market, path.name, name, "L1 top-of-book invariant violation", int(mask.sum()), sample_rows(df[mask], REQ_COLS))

    locked = df["bid_px_00"] == df["ask_px_00"]
    if locked.any():
        issue(issues, "WARN", market, path.name, "locked_book_bid_eq_ask", "bid equals ask", int(locked.sum()), sample_rows(df[locked], REQ_COLS))

    zero_size = (df["bid_sz_00"] == 0) | (df["ask_sz_00"] == 0)
    if zero_size.any():
        issue(issues, "WARN", market, path.name, "zero_top_size", "bid or ask size is zero", int(zero_size.sum()), sample_rows(df[zero_size], REQ_COLS))

    mid = (df["bid_px_00"] + df["ask_px_00"]) / 2
    spread = df["ask_px_00"] - df["bid_px_00"]
    spread_pct = spread / mid.replace(0, np.nan)
    wide = spread_pct > 0.01
    if wide.any():
        tmp = df.loc[wide, REQ_COLS].copy()
        tmp["market"] = market
        tmp["file"] = path.name
        tmp["spread_pct"] = spread_pct.loc[wide].values
        tmp["check"] = "spread_gt_1pct_mid"
        outliers.append(tmp)
        issue(issues, "WARN", market, path.name, "spread_gt_1pct_mid", "ask-bid spread > 1% of mid", int(wide.sum()), sample_rows(tmp, REQ_COLS + ["spread_pct"]))

    delta_sec = df["ts_event"].diff().dt.total_seconds()
    gap_mask = delta_sec > 3600
    if gap_mask.any():
        g = df.loc[gap_mask, ["ts_event"]].copy()
        g["prev_ts"] = df["ts_event"].shift(1).loc[gap_mask].values
        g["gap_seconds"] = delta_sec.loc[gap_mask].values
        g["market"] = market
        g["file"] = path.name
        gaps.append(g[["market", "file", "prev_ts", "ts_event", "gap_seconds"]])
        issue(issues, "WARN", market, path.name, "event_gaps_gt_1h", "large event gap; validate against session calendar", int(gap_mask.sum()), g.head(5).to_json(orient="records", date_format="iso"))

    summary.append(
        {
            "market": market,
            "file": path.name,
            "path": str(path),
            "rows": n,
            "file_start": file_start,
            "file_end": file_end,
            "first_ts": df["ts_event"].min(),
            "last_ts": df["ts_event"].max(),
            "crossed_book": int((df["bid_px_00"] > df["ask_px_00"]).sum()),
            "locked_book": int(locked.sum()),
            "zero_top_size": int(zero_size.sum()),
            "median_spread": float(spread.median()),
            "max_spread": float(spread.max()),
        }
    )
    return issues, gaps, outliers, summary, df[["ts_event"]]


def audit_file_coverage(files: list[Path]):
    rows = []
    by_market: dict[str, set[pd.Timestamp]] = {}
    for path in files:
        parsed = parse_file(path)
        if parsed is None:
            continue
        market, start, _ = parsed
        by_market.setdefault(market, set()).add(start)
    for market, dates in by_market.items():
        expected = pd.date_range(min(dates), max(dates), freq="1D")
        for d in sorted(set(expected) - dates):
            issue(rows, "FAIL", market, "", "missing_daily_file", f"missing start date {d.date()}")
    return rows


def parse_hhmm(x: str) -> time:
    h, m = map(int, x.split(":"))
    return time(h, m)


def load_config(path: Path) -> dict:
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


def audit_session_df(path: Path, ts: pd.DatetimeIndex, cfg: dict):
    market = path.parent.name
    parsed = parse_file(path)
    if parsed is None:
        return [{"severity": "FAIL", "market": market, "file": path.name, "check": "bad_filename", "detail": "expected {market}_front_mbp1_YYYY-MM-DD_YYYY-MM-DD.dbn.zst"}], pd.DataFrame(), pd.DataFrame()
    _, file_start, file_end = parsed

    if len(ts) == 0:
        return [{"severity": "FAIL", "market": market, "file": path.name, "check": "empty_or_bad_ts", "detail": "zero valid ts_event"}], pd.DataFrame(), pd.DataFrame()

    tz = cfg["timezone"]
    local_ts = ts.tz_convert(tz)
    outside = pd.DataFrame({"ts_event": ts, "ts_local": local_ts})[~is_active_local_index(local_ts, cfg).to_numpy()]

    expected_utc = pd.date_range(file_start, file_end, freq="1min", inclusive="left")
    expected_utc = expected_utc[is_active_local_index(expected_utc.tz_convert(tz), cfg).to_numpy()]
    actual_minutes = pd.DatetimeIndex(ts.floor("min")).unique()
    missing = expected_utc.difference(actual_minutes)

    missing_df = pd.DataFrame({"market": market, "file": path.name, "missing_ts_event": missing, "missing_ts_local": missing.tz_convert(tz)})
    outside_df = outside.copy()
    outside_df.insert(0, "file", path.name)
    outside_df.insert(0, "market", market)

    missing_n = len(missing_df)
    outside_n = len(outside_df)
    severity = "PASS" if missing_n == 0 and outside_n == 0 else "FAIL" if missing_n > 0 else "WARN"
    summary = {
        "severity": severity,
        "market": market,
        "file": path.name,
        "check": "l1_session_calendar",
        "actual_events": len(ts),
        "actual_event_minutes": len(actual_minutes),
        "expected_session_minutes": len(expected_utc),
        "missing_expected_minutes": missing_n,
        "outside_session_events": outside_n,
        "first_ts": ts.min(),
        "last_ts": ts.max(),
    }
    return [summary], missing_df, outside_df


def find_dbn_files(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/*.dbn.zst") if not p.name.startswith(".tmp_"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(SCRIPT_DIR))
    ap.add_argument("--config", default=str(PROJECT_ROOT / "data" / "market_sessions.yaml"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output" / "audits" / "L1_mbp1_audit"))
    ap.add_argument("--core-only", action="store_true")
    ap.add_argument("--sessions-only", action="store_true")
    ap.add_argument("--max-files", type=int, default=0, help="Optional validation limit. Default: all files.")
    args = ap.parse_args()

    if args.core_only and args.sessions_only:
        raise SystemExit("ERROR: choose at most one of --core-only or --sessions-only")

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = find_dbn_files(root)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No DBN files found under {root}/{{market}}/*.dbn.zst")

    markets_cfg = load_config(Path(args.config))["markets"]
    valid_markets = set()
    session_issues = []
    for market in sorted({p.parent.name for p in files}):
        if market not in markets_cfg:
            session_issues.append({"severity": "FAIL", "market": market, "file": "", "check": "missing_market_config", "detail": f"No session config for market {market}"})
            continue
        cfg_issue = validate_market_config(market, markets_cfg[market])
        if cfg_issue:
            session_issues.append(cfg_issue)
            continue
        valid_markets.add(market)

    core_issues = [] if args.sessions_only else audit_file_coverage(files)
    core_gaps, core_outliers, core_summary = [], [], []
    session_summary, missing_all, outside_all = [], [], []

    for path in files:
        market = path.parent.name
        print(f"START L1 audit {market} | {path}", flush=True)
        if path.stat().st_size <= 0:
            issue(core_issues, "FAIL", market, path.name, "empty_file", "file size is zero")
            continue
        try:
            raw = read_dbn(path)
        except Exception as e:
            issue(core_issues, "FAIL", market, path.name, "read_dbn", repr(e))
            continue

        ts_for_session = pd.DatetimeIndex([])
        if not args.sessions_only:
            issues, gaps, outliers, summary, ts_df = audit_core_df(path, raw)
            core_issues.extend(issues)
            core_gaps.extend(gaps)
            core_outliers.extend(outliers)
            core_summary.extend(summary)
            if ts_df is not None:
                ts_for_session = pd.DatetimeIndex(pd.to_datetime(ts_df["ts_event"], utc=True, errors="coerce").dropna()).sort_values().unique()
        elif "ts_event" in raw.columns:
            ts_for_session = pd.DatetimeIndex(pd.to_datetime(raw["ts_event"], utc=True, errors="coerce").dropna()).sort_values().unique()

        if not args.core_only and market in valid_markets:
            summary, missing_df, outside_df = audit_session_df(path, ts_for_session, markets_cfg[market])
            if summary and summary[0].get("check") != "l1_session_calendar":
                session_issues.extend(summary)
            else:
                session_summary.extend(summary)
            if not missing_df.empty:
                missing_all.append(missing_df)
            if not outside_df.empty:
                outside_all.append(outside_df)
        print(f"DONE  L1 audit {market} | {path.name}", flush=True)

    core_issues_df = pd.DataFrame(core_issues)
    pd.DataFrame(core_summary).to_csv(out / "core_summary.csv", index=False)
    core_issues_df.to_csv(out / "core_issues.csv", index=False)
    (pd.concat(core_gaps, ignore_index=True) if core_gaps else pd.DataFrame()).to_csv(out / "core_gaps.csv", index=False)
    (pd.concat(core_outliers, ignore_index=True) if core_outliers else pd.DataFrame()).to_csv(out / "core_outliers.csv", index=False)

    session_issues_df = pd.DataFrame(session_issues)
    session_summary_df = pd.DataFrame(session_summary)
    session_summary_df.to_csv(out / "session_summary.csv", index=False)
    session_issues_df.to_csv(out / "session_issues.csv", index=False)
    (pd.concat(missing_all, ignore_index=True) if missing_all else pd.DataFrame()).to_csv(out / "session_missing_expected_minutes.csv", index=False)
    (pd.concat(outside_all, ignore_index=True) if outside_all else pd.DataFrame()).to_csv(out / "session_outside_events.csv", index=False)

    core_fail = 0 if core_issues_df.empty else int((core_issues_df["severity"] == "FAIL").sum())
    core_warn = 0 if core_issues_df.empty else int((core_issues_df["severity"] == "WARN").sum())
    session_fail_issues = 0 if session_issues_df.empty else int((session_issues_df["severity"] == "FAIL").sum())
    session_fail_summary = 0 if session_summary_df.empty else int((session_summary_df["severity"] == "FAIL").sum())
    session_warn = 0 if session_summary_df.empty else int((session_summary_df["severity"] == "WARN").sum())

    print(f"Wrote reports to: {out}")
    print(f"Files scanned: {len(files)}")
    print(f"Core FAIL/WARN: {core_fail}/{core_warn}")
    print(f"Session FAIL/WARN: {session_fail_issues + session_fail_summary}/{session_warn}")
    if core_fail or session_fail_issues or session_fail_summary:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
