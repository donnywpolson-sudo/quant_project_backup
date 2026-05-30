from __future__ import annotations

import argparse
from pathlib import Path
from datetime import time

import pandas as pd
import yaml


DAY_TO_NUM = {
    "Mon": 0,
    "Tue": 1,
    "Wed": 2,
    "Thu": 3,
    "Fri": 4,
    "Sat": 5,
    "Sun": 6,
}


def parse_hhmm(x: str) -> time:
    h, m = map(int, x.split(":"))
    return time(h, m)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_market_config(market: str, cfg: dict) -> dict | None:
    missing = []
    for key in (
        "timezone",
        "week_start_day",
        "week_start_time",
        "week_end_day",
        "week_end_time",
        "closed_dates",
        "early_closes",
    ):
        if key not in cfg:
            missing.append(key)
    if "daily_break" not in cfg:
        missing.append("daily_break")
    else:
        for key in ("start", "end"):
            if key not in cfg["daily_break"]:
                missing.append(f"daily_break.{key}")

    if missing:
        return {
            "severity": "FAIL",
            "market": market,
            "year": "",
            "check": "invalid_market_config",
            "detail": f"Missing required session config keys: {','.join(sorted(missing))}",
        }

    if not cfg.get("closed_dates") and not cfg.get("early_closes") and not cfg.get("allow_empty_holiday_calendar", False):
        return {
            "severity": "FAIL",
            "market": market,
            "year": "",
            "check": "incomplete_session_calendar",
            "detail": "closed_dates and early_closes are empty; holidays/early closes are not configured and expected-minute counts may be false positives",
        }

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

    # Sunday after open.
    active |= (wd == week_start_day) & (tt >= week_start_time)

    # Monday-Thursday all day except break.
    active |= wd.isin([0, 1, 2, 3])

    # Friday before close.
    active |= (wd == week_end_day) & (tt < week_end_time)

    # Remove daily maintenance break.
    in_break = (tt >= break_start) & (tt < break_end)
    active &= ~in_break

    # Remove full closed dates.
    closed_dates = set(str(x) for x in cfg.get("closed_dates", []))
    if closed_dates:
        active &= ~dstr.isin(closed_dates)

    # Apply early closes.
    early_closes = cfg.get("early_closes", {}) or {}
    for date_str, close_hhmm in early_closes.items():
        close_t = parse_hhmm(str(close_hhmm))
        active &= ~((dstr == str(date_str)) & (tt >= close_t))

    return active


def audit_file(path: Path, market: str, year: int, cfg: dict):
    print(f"START session audit {market} {year} | {path}", flush=True)

    df = pd.read_parquet(path, columns=["ts_event"])
    ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dropna()
    ts = pd.DatetimeIndex(ts).sort_values().unique()

    if len(ts) == 0:
        return (
            [{"severity": "FAIL", "market": market, "year": year, "check": "empty_or_bad_ts", "n": 0}],
            pd.DataFrame(),
            pd.DataFrame(),
        )

    tz = cfg["timezone"]
    local_ts = ts.tz_convert(tz)

    active_actual = is_active_local_index(local_ts, cfg)
    outside = pd.DataFrame({"ts_event": ts, "ts_local": local_ts})[~active_actual.to_numpy()]

    utc_start = pd.Timestamp(f"{year}-01-01T00:00:00Z")
    utc_end = pd.Timestamp(f"{year + 1}-01-01T00:00:00Z")

    expected_utc = pd.date_range(utc_start, utc_end, freq="1min", inclusive="left")
    expected_local = expected_utc.tz_convert(tz)

    active_expected = is_active_local_index(expected_local, cfg)
    expected_utc = expected_utc[active_expected.to_numpy()]

    actual_set = pd.DatetimeIndex(ts)
    missing = expected_utc.difference(actual_set)

    missing_df = pd.DataFrame(
        {
            "market": market,
            "year": year,
            "missing_ts_event": missing,
            "missing_ts_local": missing.tz_convert(tz),
        }
    )

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

    print(
        f"DONE  session audit {market} {year} | "
        f"actual={len(ts):,} expected={len(expected_utc):,} "
        f"missing={len(missing_df):,} outside={len(outside_df):,}",
        flush=True,
    )

    return [summary], missing_df, outside_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--config", default="data/market_sessions.yaml")
    ap.add_argument("--out", default="output/audits/session_audit")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = load_config(Path(args.config))
    markets_cfg = config["markets"]

    summaries = []
    missing_all = []
    outside_all = []
    issues = []

    ohlcv_root = root / "ohlcv_1m"
    files = sorted(ohlcv_root.glob("*/*.parquet")) if ohlcv_root.exists() else []
    if not files:
        files = sorted(root.glob("*/*.parquet"))

    if not files:
        raise SystemExit(
            f"No parquet files found under {root}/ohlcv_1m/{{market}}/{{year}}.parquet "
            f"or {root}/{{market}}/{{year}}.parquet"
        )

    files_by_market = {}
    valid_markets = set()
    for path in files:
        files_by_market.setdefault(path.parent.name, []).append(path)

    for market in sorted(files_by_market):
        if market not in markets_cfg:
            issues.append(
                {
                    "severity": "FAIL",
                    "market": market,
                    "year": "",
                    "check": "missing_market_config",
                    "detail": f"No session config for market {market}",
                }
            )
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
            issues.append(
                {
                    "severity": "FAIL",
                    "market": market,
                    "year": path.stem,
                    "check": "bad_filename",
                    "detail": "filename stem is not integer year",
                }
            )
            continue

        summary, missing_df, outside_df = audit_file(path, market, year, markets_cfg[market])

        summaries.extend(summary)

        if not missing_df.empty:
            missing_all.append(missing_df)

        if not outside_df.empty:
            outside_all.append(outside_df)

    pd.DataFrame(summaries).to_csv(out / "session_summary.csv", index=False)
    pd.DataFrame(issues).to_csv(out / "session_issues.csv", index=False)

    if missing_all:
        pd.concat(missing_all, ignore_index=True).to_csv(out / "missing_expected_minutes.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "missing_expected_minutes.csv", index=False)

    if outside_all:
        pd.concat(outside_all, ignore_index=True).to_csv(out / "outside_session_rows.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "outside_session_rows.csv", index=False)

    print(f"Wrote reports to: {out}", flush=True)

    fail_issues = 0 if not issues else sum(1 for r in issues if r.get("severity") == "FAIL")
    fail_summaries = sum(1 for r in summaries if r.get("severity") == "FAIL")
    if fail_issues or fail_summaries:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
