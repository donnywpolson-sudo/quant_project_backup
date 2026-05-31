import polars as pl
import logging
import numpy as np
from datetime import timedelta
from core.config import config

from pipeline.features.baseline import compute_baseline_features, load_baseline_feature_names
from pipeline.features.expansion import expand_features, add_cross_timeframe_interactions
from pipeline.features.htf_context import add_htf_context_features
from pipeline.features.volume_profile import add_volume_profile_features
from pipeline.target.target import add_target_5m, add_target_1h, add_target_4h
from pipeline.target.triple_barrier import add_triple_barrier_target
from pipeline.meta.meta_label import add_meta_label_target
logger = logging.getLogger(__name__)


def _stage(df: pl.DataFrame, name: str, fn, *args, **kwargs) -> pl.DataFrame:
    """Track row count and ts_event integrity through a feature stage."""
    before = df.height
    _check_ts_event(df, name, 'before')
    result = fn(df, *args, **kwargs)
    after = result.height if hasattr(result, 'height') else 0
    drop_pct = (before - after) / max(before, 1) * 100
    logger.info('[STAGE] %-22s rows: %d -> %d (%.1f%% drop)', name, before, after, drop_pct)
    if after == 0 and before > 0:
        raise RuntimeError(
            'FEATURE FAIL: stage "%s" collapsed %d rows to 0' % (name, before)
        )
    _check_ts_event(result, name, 'after')
    return result


def _check_ts_event(df: pl.DataFrame, stage: str, point: str):
    """Assert ts_event column integrity."""
    if 'ts_event' not in df.columns:
        raise RuntimeError(
            'FEATURE FAIL: ts_event missing at %s/%s' % (stage, point)
        )
    nulls = df['ts_event'].null_count()
    if nulls > 0:
        raise RuntimeError(
            'FEATURE FAIL: %d null ts_event values at %s/%s' % (nulls, stage, point)
        )
    if df.height == 0:
        return  # empty df is OK at before-checks, caught by _stage for after


def validate_target_feasibility(df: pl.DataFrame, horizon: int, max_gap_minutes: int = 60):
    """Hard-gate before target generation: data sufficiency + time integrity.

    Must be called BEFORE any shift-based target computation.
    Raises RuntimeError on data sufficiency/ts_event violations.
    Warns (does not crash) on intraday gaps — fill_intraday_gaps handles those.
    """
    if df.height < horizon + 10:
        raise RuntimeError(
            'INSUFFICIENT DATA FOR TARGET: %d rows, need at least %d (horizon=%d + 10)' %
            (df.height, horizon + 10, horizon)
        )
    if 'ts_event' not in df.columns:
        raise RuntimeError('MISSING ts_event — cannot validate target feasibility')
    null_count = df['ts_event'].null_count()
    if null_count > 0:
        raise RuntimeError('INVALID TIMESTAMP STATE: %d null ts_event values' % null_count)
    df_sorted = df.sort('ts_event')
    diffs = df_sorted['ts_event'].diff().drop_nulls()
    gap_count = 0
    if len(diffs) > 0:
        diff_minutes = np.array([d.total_seconds() / 60.0 for d in diffs.to_list()])
        closure_mask = diff_minutes > 240.0
        intraday_gaps = diff_minutes[~closure_mask]
        over_threshold = intraday_gaps[intraday_gaps > max_gap_minutes]
        gap_count = len(over_threshold)
        if gap_count > 0:
            logger.warning(
                '[FEASIBILITY] %d intraday gaps > %d min (max=%.0f min) — '
                'will forward-fill in fill_intraday_gaps step',
                gap_count, max_gap_minutes, over_threshold.max()
            )
    logger.info('[FEASIBILITY] OK: %d rows, horizon=%d, time range=%s -> %s, gaps_to_fill=%d',
                df.height, horizon, df['ts_event'].min(), df['ts_event'].max(), gap_count)


def fill_intraday_gaps(df: pl.DataFrame, bar_seconds: int = 300,
                       max_gap_minutes: int = 60, max_fill_minutes: int = 480) -> pl.DataFrame:
    """Forward-fill missing intraday bars within [max_gap_minutes, max_fill_minutes].

    Detects gaps in ts_event, generates fill rows at 5-min intervals, and
    forward-fills OHLCV from the preceding bar. Gaps larger than
    max_fill_minutes (weekends, holidays) are left as genuine closures.
    """
    ts = df['ts_event'].to_list()
    n, new_rows = len(ts), []
    for i in range(n - 1):
        gap_sec = (ts[i + 1] - ts[i]).total_seconds()
        gap_min = gap_sec / 60.0
        if max_gap_minutes < gap_min <= max_fill_minutes:
            missing_steps = int(gap_sec / bar_seconds) - 1
            if missing_steps <= 0:
                continue
            row_i = {col: df[col][i] for col in df.columns}
            for step in range(1, missing_steps + 1):
                fill_ts = ts[i] + timedelta(seconds=step * bar_seconds)
                new_rows.append({**row_i, 'ts_event': fill_ts})
    if not new_rows:
        return df
    fill_df = pl.DataFrame(new_rows).cast(df.schema)
    result = pl.concat([df, fill_df]).sort('ts_event')
    added = len(new_rows)
    gap_count = sum(1 for i in range(n - 1)
                    if max_gap_minutes < (ts[i + 1] - ts[i]).total_seconds() / 60.0 <= max_fill_minutes)
    logger.warning('[FILL-GAPS] Forward-filled %d missing intraday rows across %d gaps',
                   added, gap_count)
    return result


def generate_features(df: pl.DataFrame) -> pl.DataFrame:
    # ---- STEP 1: ts_event (base layer) ----
    ts_dtype = df['ts_event'].dtype
    if ts_dtype != pl.Datetime(time_unit='us', time_zone='UTC'):
        df = df.with_columns(
            pl.col('ts_event').cast(pl.Datetime(time_unit='us', time_zone='UTC'))
        )
    _check_ts_event(df, 'entry', 'base')
    logger.info('[CANONICAL] Step 1 ts_event: %d rows, range=%s -> %s',
                df.height, df['ts_event'].min(), df['ts_event'].max())

    # Hard-gate BEFORE any target computation — validate data sufficiency + time integrity.
    # Triple-barrier uses 64-bar horizon, target_4h uses 48 bars; validate against max.
    validate_target_feasibility(df, horizon=64)

    # Forward-fill missing intraday bars (60-480 min gaps: data feed outages).
    # Gaps > 480 min (weekends, holidays) are left as genuine closures.
    df = _stage(df, 'fill_intraday_gaps', fill_intraday_gaps)

    # ---- STEP 2: HTF context (columns needed by target_1h) ----
    if config.ENABLE_EXPANSION:
        df = _stage(df, 'htf_context', add_htf_context_features)

    # ---- STEP 3: TARGETS (computed FIRST, before features) ----
    df = _stage(df, 'target_5m', add_target_5m)
    df = _stage(df, 'target_1h', add_target_1h)
    df = _stage(df, 'target_4h', add_target_4h)
    df = _stage(df, 'triple_barrier', add_triple_barrier_target)
    logger.info('[CANONICAL] Step 3 targets computed: %d rows', df.height)

    # ---- STEP 4: FEATURES (derived AFTER targets exist) ----
    df = _stage(df, 'baseline_features', compute_baseline_features)
    baseline_names = load_baseline_feature_names()
    baseline_cols = [c for c in baseline_names if c in df.columns]
    if config.ENABLE_EXPANSION:
        df = _stage(df, 'volume_profile', add_volume_profile_features)
        df = _stage(df, 'expand_features', expand_features, baseline_cols)
        htf_cols = [c for c in df.columns if c.startswith('htf_')]
        ltf_candidate = [c for c in df.columns if c.startswith(('feature_', 'ratio_', 'pair_', 'zscore', 'cross_')) and (not c.startswith(('1h_', 'daily_')))]
        ltf_cols = [c for c in ltf_candidate if not c.startswith('cross_')]
        if htf_cols and ltf_cols:
            df = _stage(df, 'cross_timeframe', add_cross_timeframe_interactions, ltf_cols, htf_cols)
    logger.info('[CANONICAL] Step 4 features built: %d rows, %d cols', df.height, len(df.columns))

    # ---- STEP 5: FILTER (explicit mask on target NaNs) ----
    # Build mask from all target columns BEFORE filtering — preserves alignment.
    # Skip target columns that are entirely NaN (e.g., target_1h when HTF data
    # is unavailable) to avoid a single null column collapsing the whole dataset.
    filter_cols = [c for c in ('target_5m', 'target_4h', 'target_1h', 'target_tb') if c in df.columns]
    before = df.height
    mask = None
    full_nan_cols = []
    for tc in filter_cols:
        null_count = df[tc].null_count()
        if null_count == df.height:
            full_nan_cols.append(tc)
            logger.warning('[CANONICAL] Filter: %s is entirely null (%d rows) — excluding from mask', tc, df.height)
            continue
        col_mask = df[tc].is_not_null()
        mask = col_mask if mask is None else mask & col_mask
    if full_nan_cols:
        logger.warning('[CANONICAL] Filter: skipped %d fully-null target columns: %s',
                       len(full_nan_cols), ', '.join(full_nan_cols))
    if mask is not None:
        df = df.filter(mask)
        after = df.height
        dropped = before - after
        logger.info('[CANONICAL] Step 5 filter: %d rows -> %d (dropped %d NaN)', before, after, dropped)
    if before > 0 and df.height == 0:
        raise RuntimeError(
            'FEATURE FAIL: filter collapsed %d rows to 0 — '
            'all target columns fully NaN. Check shift/horizon/cross-asset data.' % before
        )

    # ---- STEP 6: VALIDATE (final contract) ----
    if df.height == 0:
        raise RuntimeError('FEATURE ENGINE FAILURE: empty output after all stages')
    _check_ts_event(df, 'final', 'output')

    feature_cols = [c for c in df.columns if c.startswith(('feature_', 'ratio_', 'pair_', 'zscore', 'cross_', 'htf_'))]
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c in feature_cols])
    logger.info('[CANONICAL] Final: %d rows, %d features (expansion=%s)',
                df.height, len(feature_cols),
                'on' if config.ENABLE_EXPANSION else 'off')
    return df
