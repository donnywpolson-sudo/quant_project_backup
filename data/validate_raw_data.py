from __future__ import annotations

import argparse
from datetime import date, timedelta, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REQ_COLS = ["ts_event", "open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close"]
NUM_COLS = PRICE_COLS + ["volume"]
DAY_TO_NUM = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_project_root() -> Path:
    for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent, Path.cwd()):
        if (candidate / "data" / "raw").exists():
            return candidate
    return SCRIPT_DIR.parent


PROJECT_ROOT = resolve_project_root()
DEFAULT_RAW_ROOT = SCRIPT_DIR / "raw" if (SCRIPT_DIR / "raw").exists() else PROJECT_ROOT / "data" / "raw"
STALE_PRICE_RUN_MINUTES = 240
MAX_GAP_ROWS_PER_FILE = 1000
TICK_SIZE = {
    "CL": 0.01,
    "ES": 0.25,
    "GC": 0.10,
    "HG": 0.0005,
    "NG": 0.001,
    "NQ": 0.25,
    "RTY": 0.10,
    "SI": 0.005,
    "YM": 1.0,
    "ZB": 1.0 / 32.0,
    "ZC": 0.25,
    "ZN": 1.0 / 64.0,
}


def issue(rows, severity, market, year, check, detail, n=None, sample=None):
    rows.append({"severity": severity, "market": market, "year": year, "check": check, "detail": detail, "n": n, "sample": sample})


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


def max_true_run(mask: pd.Series) -> int:
    if mask.empty:
        return 0
    mask = mask.fillna(False).astype(bool)
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    runs = mask[mask].groupby(groups[mask]).sum()
    return int(runs.max()) if not runs.empty else 0


def off_tick_mask(values: pd.Series, tick: float, tol: float = 1e-7) -> pd.Series:
    scaled = values / tick
    return (scaled - np.round(scaled)).abs() > tol


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
    valid_numeric = np.isfinite(df[NUM_COLS]).all(axis=1) & df[NUM_COLS].notna().all(axis=1)
    if not valid_numeric.all():
        df = df.loc[valid_numeric].copy()
        if df.empty:
            issue(issues, "FAIL", market, year, "no_valid_numeric_rows", "all rows invalid after numeric validation")
            return issues, gaps, outliers, summary, None

    non_integer_volume = (df["volume"].dropna() % 1) != 0
    if non_integer_volume.any():
        idx = non_integer_volume[non_integer_volume].index
        issue(issues, "WARN", market, year, "non_integer_volume", "volume is not integer-valued", int(non_integer_volume.sum()), sample_rows(df.loc[idx]))

    dup = df["ts_event"].duplicated(keep=False)
    if dup.any():
        issue(issues, "FAIL", market, year, "duplicate_ts", "duplicate timestamps", int(dup.sum()), sample_rows(df[dup]))

    wrong_year = df["ts_event"].dt.year != year
    if wrong_year.any():
        issue(issues, "FAIL", market, year, "timestamp_outside_file_year", f"ts_event UTC year != filename year {year}; UTC year partitions must not cross walkforward year boundaries", int(wrong_year.sum()), sample_rows(df[wrong_year]))

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

    tick = TICK_SIZE.get(market)
    if tick is not None:
        off_tick = pd.Series(False, index=df.index)
        for c in PRICE_COLS:
            off_tick |= off_tick_mask(df[c], tick)
        if off_tick.any():
            issue(issues, "WARN", market, year, "price_off_tick_grid", f"one or more OHLC prices not aligned to expected tick={tick}", int(off_tick.sum()), sample_rows(df[off_tick]))

    zero_vol = df["volume"] == 0
    if zero_vol.any():
        issue(issues, "WARN", market, year, "zero_volume", "zero-volume bars", int(zero_vol.sum()), sample_rows(df[zero_vol]))

    zero_range = df["high"] == df["low"]
    if zero_range.any():
        issue(issues, "WARN", market, year, "zero_range", "high == low bars", int(zero_range.sum()), sample_rows(df[zero_range]))

    df_nodup = df.drop_duplicates("ts_event", keep=False)
    stale_close = df_nodup["close"].eq(df_nodup["close"].shift(1))
    stale_run = max_true_run(stale_close)
    if stale_run >= STALE_PRICE_RUN_MINUTES:
        issue(issues, "WARN", market, year, "stale_close_run", f"close unchanged for >= {STALE_PRICE_RUN_MINUTES} consecutive minutes", stale_run)

    df_unique_ts = df.drop_duplicates("ts_event", keep=False).sort_values("ts_event").reset_index(drop=True)
    ts = df_unique_ts["ts_event"]
    delta_min = ts.diff().dt.total_seconds().div(60)
    gap_mask = delta_min > 1
    if gap_mask.any():
        g = df_unique_ts.loc[gap_mask, ["ts_event"]].copy()
        g["prev_ts"] = ts.shift(1).loc[gap_mask].values
        g["gap_minutes"] = delta_min.loc[gap_mask].values
        g["market"] = market
        g["year"] = year
        gaps.append(g[["market", "year", "prev_ts", "ts_event", "gap_minutes"]].head(MAX_GAP_ROWS_PER_FILE))
        issue(issues, "WARN", market, year, "timestamp_gaps_gt_1min", "raw timestamp gaps detected on duplicate-free timestamps; validate against session calendar before treating as missing data", int(gap_mask.sum()), g.head(5).to_json(orient="records", date_format="iso"))

    reverse_or_same = delta_min <= 0
    if reverse_or_same.any():
        issue(issues, "FAIL", market, year, "non_increasing_timestamps", "timestamp diff <= 0 after duplicate removal", int(reverse_or_same.sum()), sample_rows(df_unique_ts[reverse_or_same]))

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

    dollar_volume = df["close"] * df["volume"]
    dv_z = mad_z(dollar_volume)
    dv_spike = dv_z.abs() > 25
    if dv_spike.any():
        tmp = df.loc[dv_spike, REQ_COLS].copy()
        tmp["market"] = market
        tmp["year"] = year
        tmp["dollar_volume_mad_z"] = dv_z.loc[dv_spike].values
        tmp["check"] = "dollar_volume_mad_z_gt_25"
        outliers.append(tmp)
        issue(issues, "WARN", market, year, "dollar_volume_mad_z_gt_25", "extreme price*volume spike", int(dv_spike.sum()), sample_rows(df[dv_spike]))

    summary.append({
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
        "stale_close_run_max": stale_run,
        "close_min": float(df["close"].min()),
        "close_max": float(df["close"].max()),
        "volume_sum": float(df["volume"].sum()),
    })
    return issues, gaps, outliers, summary, df[REQ_COLS]


def audit_cross_year(bounds_by_market):
    rows = []
    for market, items in bounds_by_market.items():
        items = sorted(items, key=lambda x: x[0])
        for (year_a, first_a, last_a), (year_b, first_b, last_b) in zip(items, items[1:]):
            rows.append({
                "severity": "FAIL" if first_b <= last_a else "INFO",
                "market": market,
                "check": "cross_year_overlap" if first_b <= last_a else "cross_year_gap",
                "year_a": year_a,
                "year_b": year_b,
                "first_ts_a": first_a,
                "last_ts_a": last_a,
                "first_ts_b": first_b,
                "last_ts_b": last_b,
                "gap_minutes": (first_b - last_a).total_seconds() / 60,
                "detail": "next year starts before or at prior year end" if first_b <= last_a else "cross-year gap; validate against session calendar",
            })
    return rows


def audit_year_file_coverage(files: list[Path]) -> list[dict]:
    rows = []
    observed_years_by_market: dict[str, set[int]] = {}
    for path in files:
        market = path.parent.name
        try:
            df = pd.read_parquet(path, columns=["ts_event"])
            ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dropna()
        except Exception as e:
            issue(rows, "FAIL", market, path.stem, "year_coverage_read_error", repr(e))
            continue
        if ts.empty:
            issue(rows, "FAIL", market, path.stem, "year_coverage_empty_ts", "no valid ts_event values for year coverage inference")
            continue
        for y in range(int(ts.min().year), int(ts.max().year) + 1):
            observed_years_by_market.setdefault(market, set()).add(y)
    for market, years in observed_years_by_market.items():
        for year in range(min(years), max(years) + 1):
            if year not in years:
                issue(rows, "FAIL", market, year, "missing_observed_data_year", f"missing observed ts_event year {year} for {market}")
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
    if not cfg.get("closed_dates") and not cfg.get("early_closes") and not cfg.get("holiday_calendar") and not cfg.get("allow_empty_holiday_calendar", False):
        return {"severity": "FAIL", "market": market, "year": "", "check": "incomplete_session_calendar", "detail": "closed_dates and early_closes are empty; expected-minute counts may be false positives"}
    return None


def is_approximate_session_calendar(cfg: dict) -> bool:
    return str(cfg.get("session_calendar_accuracy", "")).lower() == "approximate"


def easter_date(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    cur = date(year, month, 1)
    cur += timedelta(days=(weekday - cur.weekday()) % 7)
    return cur + timedelta(days=7 * (n - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    cur = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    cur -= timedelta(days=(cur.weekday() - weekday) % 7)
    return cur


def observed_fixed(year: int, month: int, day: int) -> date:
    d = date(year, month, day)
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def generated_cme_globex_us_futures_holidays(years: set[int], early_close_time: str = "12:15") -> tuple[set[str], dict[str, str], dict[str, str]]:
    closed_dates, early_closes, late_opens = set(), {}, {}
    for year in sorted(years):
        new_years = observed_fixed(year, 1, 1)
        christmas = observed_fixed(year, 12, 25)
        good_friday = easter_date(year) - timedelta(days=2)
        closed_dates.add(good_friday.isoformat())
        late_opens[new_years.isoformat()] = "17:00"
        late_opens[christmas.isoformat()] = "17:00"
        early_candidates = [
            nth_weekday(year, 1, 0, 3),
            nth_weekday(year, 2, 0, 3),
            last_weekday(year, 5, 0),
            observed_fixed(year, 7, 4),
            nth_weekday(year, 9, 0, 1),
            nth_weekday(year, 11, 3, 4),
            nth_weekday(year, 11, 3, 4) + timedelta(days=1),
            date(year, 12, 24),
        ]
        if year >= 2022:
            early_candidates.append(observed_fixed(year, 6, 19))
        for d in early_candidates:
            if d.weekday() < 5 and d.isoformat() not in closed_dates:
                early_closes.setdefault(d.isoformat(), early_close_time)
                if not (d.month == 12 and d.day == 24):
                    late_opens.setdefault(d.isoformat(), "17:00")
    return closed_dates, early_closes, late_opens


def resolved_holiday_calendar(local_ts: pd.DatetimeIndex, cfg: dict) -> tuple[set[str], dict[str, str], dict[str, str]]:
    closed_dates = set(str(x) for x in cfg.get("closed_dates", []))
    early_closes = {str(k): str(v) for k, v in (cfg.get("early_closes", {}) or {}).items()}
    late_opens = {str(k): str(v) for k, v in (cfg.get("late_opens", {}) or {}).items()}
    if cfg.get("holiday_calendar") == "cme_globex_us_futures":
        years = set(pd.Series(local_ts).dt.year.dropna().astype(int).tolist())
        gen_closed, gen_early, gen_late = generated_cme_globex_us_futures_holidays(years, str(cfg.get("holiday_early_close_time", "12:15")))
        closed_dates |= gen_closed
        for k, v in gen_early.items():
            early_closes.setdefault(k, v)
        for k, v in gen_late.items():
            late_opens.setdefault(k, v)
    return closed_dates, early_closes, late_opens


def active_weekday_mask(wd: pd.Series, start_day: int, end_day: int) -> pd.Series:
    if start_day <= end_day:
        return (wd > start_day) & (wd < end_day)
    return (wd > start_day) | (wd < end_day)


def is_active_local_index(local_ts: pd.DatetimeIndex, cfg: dict) -> pd.Series:
    week_start_day = DAY_TO_NUM[cfg["week_start_day"]]
    week_end_day = DAY_TO_NUM[cfg["week_end_day"]]
    week_start_time = parse_hhmm(cfg["week_start_time"])
    week_end_time = parse_hhmm(cfg["week_end_time"])
    break_start = parse_hhmm(cfg["daily_break"]["start"])
    break_end = parse_hhmm(cfg["daily_break"]["end"])

    s = pd.Series(local_ts, index=local_ts)
    wd, tt, dstr = s.dt.weekday, s.dt.time, s.dt.strftime("%Y-%m-%d")
    active = pd.Series(False, index=local_ts)
    active |= (wd == week_start_day) & (tt >= week_start_time)
    active |= active_weekday_mask(wd, week_start_day, week_end_day)
    active |= (wd == week_end_day) & (tt < week_end_time)
    active &= ~((tt >= break_start) & (tt < break_end))

    closed_dates, early_closes, late_opens = resolved_holiday_calendar(local_ts, cfg)
    if closed_dates:
        active &= ~dstr.isin(closed_dates)

    for date_str, open_hhmm in late_opens.items():
        if date_str in early_closes:
            continue
        open_t = parse_hhmm(str(open_hhmm))
        active &= ~((dstr == str(date_str)) & (tt < open_t))

    for date_str, close_hhmm in early_closes.items():
        close_t = parse_hhmm(str(close_hhmm))
        reopen_hhmm = late_opens.get(date_str)
        if reopen_hhmm is not None:
            reopen_t = parse_hhmm(str(reopen_hhmm))
            active &= ~((dstr == str(date_str)) & (tt >= close_t) & (tt < reopen_t))
        else:
            active &= ~((dstr == str(date_str)) & (tt >= close_t))
    return active


def audit_session_file(path: Path, market: str, year: int, cfg: dict):
    print(f"START session audit {market} {year} | {path}", flush=True)
    try:
        df = pd.read_parquet(path, columns=["ts_event"])
    except Exception as e:
        return ([{"severity": "FAIL", "market": market, "year": year, "check": "session_read_parquet", "path": str(path), "detail": repr(e)}], pd.DataFrame(), pd.DataFrame())
    raw_rows = len(df)
    ts_raw = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dropna()
    ts_sorted = pd.DatetimeIndex(ts_raw).sort_values()
    duplicate_ts = int(pd.Series(ts_sorted).duplicated(keep=False).sum())
    ts = ts_sorted.unique()
    if len(ts) == 0:
        return ([{"severity": "FAIL", "market": market, "year": year, "check": "empty_or_bad_ts", "n": raw_rows}], pd.DataFrame(), pd.DataFrame())

    tz = cfg["timezone"]
    local_ts = ts.tz_convert(tz)
    outside = pd.DataFrame({"ts_event": ts, "ts_local": local_ts})[~is_active_local_index(local_ts, cfg).to_numpy()]

    utc_start = pd.Timestamp(f"{year}-01-01 00:00", tz="UTC")
    utc_end = pd.Timestamp(f"{year + 1}-01-01 00:00", tz="UTC")
    local_start = utc_start.tz_convert(tz)
    local_end = utc_end.tz_convert(tz)
    full_expected_local = pd.date_range(local_start, local_end, freq="1min", inclusive="left")
    full_expected_local = full_expected_local[is_active_local_index(full_expected_local, cfg).to_numpy()]
    full_expected_utc = full_expected_local.tz_convert("UTC")
    full_expected_utc = full_expected_utc[(full_expected_utc >= utc_start) & (full_expected_utc < utc_end)]

    first_ts = ts.min().floor("min")
    last_ts = ts.max().floor("min")
    expected_utc = full_expected_utc[(full_expected_utc >= first_ts) & (full_expected_utc <= last_ts)]
    missing = expected_utc.difference(pd.DatetimeIndex(ts))

    missing_df = pd.DataFrame({"market": market, "year": year, "missing_ts_event": missing, "missing_ts_local": missing.tz_convert(tz)})
    outside_df = outside.copy()
    outside_df.insert(0, "year", year)
    outside_df.insert(0, "market", market)

    missing_n, outside_n = len(missing_df), len(outside_df)
    partial_start = bool(len(full_expected_utc) and first_ts > full_expected_utc.min())
    partial_end = bool(len(full_expected_utc) and last_ts < full_expected_utc.max())
    partial_bounds = partial_start or partial_end
    coverage_pct = 100.0 * (1.0 - (missing_n / len(expected_utc))) if len(expected_utc) else 0.0
    min_coverage_pct = float(cfg.get("min_session_coverage_pct", 99.0))
    severity = "FAIL" if outside_n > 0 or duplicate_ts > 0 else "WARN" if missing_n > 0 or coverage_pct < min_coverage_pct or partial_bounds else "PASS"

    summary = {
        "severity": severity,
        "market": market,
        "year": year,
        "check": "session_calendar",
        "actual_rows": raw_rows,
        "actual_unique_timestamps": len(ts),
        "duplicate_ts": duplicate_ts,
        "full_year_expected_session_minutes": len(full_expected_utc),
        "expected_session_minutes": len(expected_utc),
        "missing_expected_minutes": missing_n,
        "coverage_pct": coverage_pct,
        "outside_session_rows": outside_n,
        "partial_file_bounds": partial_bounds,
        "partial_start": partial_start,
        "partial_end": partial_end,
        "full_year_expected_first_ts": full_expected_utc.min() if len(full_expected_utc) else pd.NaT,
        "full_year_expected_last_ts": full_expected_utc.max() if len(full_expected_utc) else pd.NaT,
        "expected_first_ts": expected_utc.min() if len(expected_utc) else pd.NaT,
        "expected_last_ts": expected_utc.max() if len(expected_utc) else pd.NaT,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }
    print(f"DONE  session audit {market} {year} | rows={raw_rows:,} unique={len(ts):,} expected={len(expected_utc):,} missing={missing_n:,} outside={outside_n:,} dup={duplicate_ts:,}", flush=True)
    return [summary], missing_df, outside_df


def audit_market_span_session(files_by_market: dict[str, list[Path]], markets_cfg: dict):
    summaries, missing_all, outside_all = [], [], []
    for market, paths in sorted(files_by_market.items()):
        if market not in markets_cfg:
            continue
        cfg = markets_cfg[market]
        if validate_market_config(market, cfg) is not None:
            continue

        ts_parts = []
        raw_rows = 0
        for path in sorted(paths):
            try:
                df = pd.read_parquet(path, columns=["ts_event"])
            except Exception as e:
                summaries.append({"severity": "FAIL", "market": market, "check": "market_span_session", "path": str(path), "detail": repr(e)})
                continue
            raw_rows += len(df)
            ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dropna()
            ts_parts.append(pd.Series(ts))

        if not ts_parts:
            continue
        ts_all = pd.DatetimeIndex(pd.concat(ts_parts, ignore_index=True)).sort_values()
        duplicate_ts = int(pd.Series(ts_all).duplicated(keep=False).sum())
        ts_unique = ts_all.unique()
        if len(ts_unique) == 0:
            summaries.append({"severity": "FAIL", "market": market, "check": "market_span_session", "actual_rows": raw_rows, "actual_unique_timestamps": 0, "detail": "no valid timestamps"})
            continue

        tz = cfg["timezone"]
        local_ts = ts_unique.tz_convert(tz)
        outside = pd.DataFrame({"market": market, "ts_event": ts_unique, "ts_local": local_ts})[~is_active_local_index(local_ts, cfg).to_numpy()]

        first_ts = ts_unique.min().floor("min")
        last_ts = ts_unique.max().floor("min")
        local_start = first_ts.tz_convert(tz).floor("min")
        local_end = last_ts.tz_convert(tz).floor("min")
        expected_local = pd.date_range(local_start, local_end, freq="1min", inclusive="both")
        expected_local = expected_local[is_active_local_index(expected_local, cfg).to_numpy()]
        expected_utc = expected_local.tz_convert("UTC")
        missing = expected_utc.difference(pd.DatetimeIndex(ts_unique))

        missing_df = pd.DataFrame({"market": market, "missing_ts_event": missing, "missing_ts_local": missing.tz_convert(tz)})
        missing_n, outside_n, expected_n = len(missing_df), len(outside), len(expected_utc)
        coverage_pct = 100.0 * (1.0 - missing_n / expected_n) if expected_n else 0.0
        severity = "FAIL" if outside_n > 0 or duplicate_ts > 0 else "WARN" if missing_n > 0 else "PASS"
        if is_approximate_session_calendar(cfg) and severity == "FAIL":
            severity = "WARN"

        summaries.append({
            "severity": severity,
            "market": market,
            "check": "market_span_session",
            "actual_rows": raw_rows,
            "actual_unique_timestamps": len(ts_unique),
            "duplicate_ts": duplicate_ts,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "expected_session_minutes": expected_n,
            "missing_expected_minutes": missing_n,
            "coverage_pct": coverage_pct,
            "outside_session_rows": outside_n,
            "calendar_accuracy": "approximate" if is_approximate_session_calendar(cfg) else "",
            "approximate_reason": cfg.get("approximate_reason", ""),
        })
        if not missing_df.empty:
            missing_all.append(missing_df)
        if not outside.empty:
            outside_all.append(outside)
    return summaries, missing_all, outside_all


def find_parquet_files(root: Path) -> list[Path]:
    files = sorted(p for p in root.glob("*/*.parquet") if p.stem.isdigit() and len(p.stem) == 4)
    if files:
        return files
    raise SystemExit(f"No parquet files found under {root}/{{market}}/{{year}}.parquet")


def filter_files(files: list[Path], markets: list[str] | None, years: list[int] | None) -> list[Path]:
    if markets:
        keep_markets = {m.upper() for m in markets}
        files = [p for p in files if p.parent.name.upper() in keep_markets]
    if years:
        keep_years = {str(y) for y in years}
        files = [p for p in files if p.stem in keep_years]
    if not files:
        raise SystemExit("ERROR: no parquet files matched selected --markets/--years filters")
    return files


def summarize_gaps(gaps: list[pd.DataFrame]) -> pd.DataFrame:
    if not gaps:
        return pd.DataFrame()
    df = pd.concat(gaps, ignore_index=True)
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["market", "year"], as_index=False)
        .agg(
            sampled_gap_rows=("gap_minutes", "size"),
            min_gap_minutes=("gap_minutes", "min"),
            median_gap_minutes=("gap_minutes", "median"),
            max_gap_minutes=("gap_minutes", "max"),
        )
    )


def run_core_audit(files: list[Path], out: Path) -> tuple[int, int]:
    out.mkdir(parents=True, exist_ok=True)
    all_issues, all_gaps, all_outliers, all_summary = [], [], [], []
    all_issues.extend(audit_year_file_coverage(files))
    bounds_by_market = {}
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
            bounds_by_market.setdefault(market, []).append((year, df["ts_event"].min(), df["ts_event"].max()))

    issues_df = pd.DataFrame(all_issues)
    cross_year_df = pd.DataFrame(audit_cross_year(bounds_by_market))
    pd.DataFrame(all_summary).to_csv(out / "core_summary.csv", index=False)
    issues_df.to_csv(out / "core_issues.csv", index=False)
    gaps_df = pd.concat(all_gaps, ignore_index=True) if all_gaps else pd.DataFrame()
    gaps_df.to_csv(out / "core_gaps_sample.csv", index=False)
    summarize_gaps(all_gaps).to_csv(out / "core_gaps_summary.csv", index=False)
    (pd.concat(all_outliers, ignore_index=True) if all_outliers else pd.DataFrame()).to_csv(out / "core_outliers.csv", index=False)
    cross_year_df.to_csv(out / "core_cross_year.csv", index=False)

    issue_fail = 0 if issues_df.empty else int((issues_df["severity"] == "FAIL").sum())
    issue_warn = 0 if issues_df.empty else int((issues_df["severity"] == "WARN").sum())
    cross_year_fail = 0 if cross_year_df.empty else int((cross_year_df["severity"] == "FAIL").sum())
    return issue_fail + cross_year_fail, issue_warn


def run_session_audit(files: list[Path], out: Path, config_path: Path) -> tuple[int, int]:
    out.mkdir(parents=True, exist_ok=True)
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
        if is_approximate_session_calendar(markets_cfg[market]):
            issues.append({"severity": "WARN", "market": market, "year": "", "check": "approximate_session_calendar", "detail": markets_cfg[market].get("approximate_reason", "session calendar marked approximate; excluded from hard session FAIL gating")})
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
        if is_approximate_session_calendar(markets_cfg[market]):
            for row in summary:
                if row.get("severity") == "FAIL":
                    row["severity"] = "WARN"
                row["calendar_accuracy"] = "approximate"
                row["approximate_reason"] = markets_cfg[market].get("approximate_reason", "")
        summaries.extend(summary)
        if not missing_df.empty:
            missing_all.append(missing_df)
        if not outside_df.empty:
            outside_all.append(outside_df)

    span_summaries, span_missing, span_outside = audit_market_span_session({m: ps for m, ps in files_by_market.items() if m in valid_markets}, markets_cfg)

    issues_df = pd.DataFrame(issues)
    summaries_df = pd.DataFrame(summaries)
    span_df = pd.DataFrame(span_summaries)
    summaries_df.to_csv(out / "session_summary.csv", index=False)
    issues_df.to_csv(out / "session_issues.csv", index=False)
    span_df.to_csv(out / "session_market_span_summary.csv", index=False)
    (pd.concat(missing_all, ignore_index=True) if missing_all else pd.DataFrame()).to_csv(out / "session_missing_expected_minutes.csv", index=False)
    (pd.concat(outside_all, ignore_index=True) if outside_all else pd.DataFrame()).to_csv(out / "session_outside_rows.csv", index=False)
    (pd.concat(span_missing, ignore_index=True) if span_missing else pd.DataFrame()).to_csv(out / "session_market_span_missing_expected_minutes.csv", index=False)
    (pd.concat(span_outside, ignore_index=True) if span_outside else pd.DataFrame()).to_csv(out / "session_market_span_outside_rows.csv", index=False)

    fail_issues = 0 if issues_df.empty else int((issues_df["severity"] == "FAIL").sum())
    if not summaries_df.empty and "calendar_accuracy" in summaries_df.columns:
        hard_summary_mask = ~summaries_df["calendar_accuracy"].fillna("").eq("approximate")
        fail_summaries = int(((summaries_df["severity"] == "FAIL") & hard_summary_mask).sum())
    else:
        fail_summaries = 0 if summaries_df.empty else int((summaries_df["severity"] == "FAIL").sum())
    warn_summaries = 0 if summaries_df.empty else int((summaries_df["severity"] == "WARN").sum())

    if not span_df.empty and "calendar_accuracy" in span_df.columns:
        hard_span_mask = ~span_df["calendar_accuracy"].fillna("").eq("approximate")
        fail_spans = int(((span_df["severity"] == "FAIL") & hard_span_mask).sum())
    else:
        fail_spans = 0 if span_df.empty else int((span_df["severity"] == "FAIL").sum())
    warn_spans = 0 if span_df.empty else int((span_df["severity"] == "WARN").sum())
    return fail_issues + fail_summaries + fail_spans, warn_summaries + warn_spans


def write_validated_files(files: list[Path], out_root: Path, audit_out: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for src in files:
        market, year = src.parent.name, int(src.stem)
        raw_rows = len(pd.read_parquet(src, columns=["ts_event"]))
        df = pd.read_parquet(src, columns=REQ_COLS)
        df = coerce_ohlcv(df)
        df = df.dropna(subset=["ts_event"])
        df = df[np.isfinite(df[NUM_COLS]).all(axis=1)]
        df = df[df[NUM_COLS].notna().all(axis=1)]
        df = df.drop_duplicates("ts_event", keep=False)

        valid_ohlcv = (
            (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
            & (df[PRICE_COLS] > 0).all(axis=1)
            & (df["volume"] >= 0)
        )
        df = df.loc[valid_ohlcv].copy()

        tick = TICK_SIZE.get(market)
        off_tick_removed = 0
        if tick is not None:
            off_tick = pd.Series(False, index=df.index)
            for c in PRICE_COLS:
                off_tick |= off_tick_mask(df[c], tick)
            off_tick_removed = int(off_tick.sum())
            df = df.loc[~off_tick].copy()

        df = df.sort_values("ts_event").reset_index(drop=True)[REQ_COLS]
        dst = out_root / market / f"{year}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        manifest.append({
            "market": market,
            "year": year,
            "src": str(src),
            "dst": str(dst),
            "raw_rows": raw_rows,
            "validated_rows": len(df),
            "removed_rows": raw_rows - len(df),
            "off_tick_removed": off_tick_removed,
            "first_ts": df["ts_event"].min() if not df.empty else pd.NaT,
            "last_ts": df["ts_event"].max() if not df.empty else pd.NaT,
        })
    pd.DataFrame(manifest).to_csv(out_root / "_manifest.csv", index=False)

    audit_frames = []
    for name in ("core_summary.csv", "core_cross_year.csv", "session_summary.csv", "session_market_span_summary.csv", "core_issues.csv", "session_issues.csv"):
        p = audit_out / name
        if not p.exists() or p.stat().st_size == 0:
            continue
        try:
            tmp = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if tmp.empty and len(tmp.columns) == 0:
            continue
        tmp.insert(0, "audit_file", name)
        audit_frames.append(tmp)
    (pd.concat(audit_frames, ignore_index=True, sort=False) if audit_frames else pd.DataFrame()).to_csv(out_root / "_audit_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_RAW_ROOT))
    ap.add_argument("--config", default=str(PROJECT_ROOT / "data" / "market_sessions.yaml"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "raw_data"))
    ap.add_argument("--validated-out", default=str(PROJECT_ROOT / "data" / "validated" / "L0_ohlcv_1m"))
    ap.add_argument("--markets", nargs="*", help="Optional market filter, e.g. --markets ES NQ CL")
    ap.add_argument("--years", nargs="*", type=int, help="Optional year filter, e.g. --years 2024 2025")
    ap.add_argument("--core-only", action="store_true")
    ap.add_argument("--sessions-only", action="store_true")
    args = ap.parse_args()

    if args.core_only and args.sessions_only:
        raise SystemExit("ERROR: choose at most one of --core-only or --sessions-only")

    files = filter_files(find_parquet_files(Path(args.root)), args.markets, args.years)
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
    if args.core_only or args.sessions_only:
        print("Validated data not written: run both core and session audits without --core-only/--sessions-only.")
        return

    validated_out = Path(args.validated_out)
    write_validated_files(files, validated_out, out)
    print(f"Wrote validated data to: {validated_out}")


if __name__ == "__main__":
    main()
