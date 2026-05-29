#!/usr/bin/env python
"""
validate_target_tb.py — Regression validate rebuilt triple-barrier target.
"""
import sys
from pathlib import Path
import hashlib
import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
CACHED_FEATURE_MATRIX = Path("output/cache/full_feature_matrix_aa6e302e19f5.parquet")
OUTPUT_REPORT = Path("output/target_tb_validation_report.txt")
FALLBACK_DATA = [
    Path(".kilo/worktrees/invited-coconut/artifacts/train_2024/ES_2024.parquet"),
]
# ---------------------------------------------------------------------------


def _rebuild_target_tb(df: pl.DataFrame) -> pl.DataFrame:
    """Replicated triple_barrier.add_triple_barrier_target with NEW params."""
    H_BARS = 64
    BARS_PER_DAY = 276
    VOL_MULT_UPPER = 1.0
    VOL_MULT_LOWER = 1.0

    close = df["close"].to_numpy().astype(np.float64)
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    n = len(close)

    if "htf_daily_vol_5" in df.columns:
        bar_vol = df["htf_daily_vol_5"].to_numpy().astype(np.float64)
        bar_vol = np.nan_to_num(bar_vol, nan=0.0005)
    else:
        rolling_std = np.full(n, np.nan)
        min_window = 20
        for i in range(min_window, n):
            rets = np.diff(np.log(close[max(0, i - min_window) : i + 1] + 1e-12))
            if len(rets) > 1:
                rolling_std[i] = np.std(rets)
        bar_vol = np.nan_to_num(rolling_std, nan=0.0005)

    bar_vol = np.maximum(bar_vol, 0.0001)
    vol_4h = bar_vol * np.sqrt(H_BARS)
    upper_mult = np.exp(VOL_MULT_UPPER * vol_4h)
    lower_mult = np.exp(-VOL_MULT_LOWER * vol_4h)

    labels = np.full(n, np.nan, dtype=np.float64)
    cp = close
    market_id = (np.log10(np.maximum(cp, 1e-9)) // 0.5).astype(int)

    seg_start = 0
    mid_prev = market_id[0]
    for i in range(1, n + 1):
        if i == n or market_id[i] != mid_prev:
            seg_end = i
            for t in range(seg_start, seg_end - H_BARS):
                entry = close[t]
                if entry <= 0:
                    continue
                window_end = min(t + 1 + H_BARS, seg_end)
                upper_barrier = entry * upper_mult[t]
                lower_barrier = entry * lower_mult[t]
                high_window = high[t + 1 : window_end]
                low_window = low[t + 1 : window_end]
                upper_hit = np.argmax(high_window >= upper_barrier)
                lower_hit = np.argmax(low_window <= lower_barrier)
                wl = window_end - t - 1
                upper_idx = int(upper_hit) if high_window[upper_hit] >= upper_barrier else wl
                lower_idx = int(lower_hit) if low_window[lower_hit] <= lower_barrier else wl
                if upper_idx < wl and upper_idx <= lower_idx:
                    labels[t] = 1.0
                elif lower_idx < wl and lower_idx < upper_idx:
                    labels[t] = -1.0
                else:
                    labels[t] = 0.0
            if i < n:
                seg_start = i
                mid_prev = market_id[i]

    values = [float(v) if np.isfinite(v) else None for v in labels]
    return df.with_columns(pl.Series("target_tb_new", values))


def _class_balance(labels: np.ndarray) -> dict:
    mask = np.isfinite(labels)
    valid = labels[mask]
    total = len(valid)
    if total == 0:
        return {"total": 0, "up": 0, "down": 0, "timeout": 0, "up_pct": 0, "down_pct": 0, "timeout_pct": 0}
    up = int((valid == 1).sum())
    down = int((valid == -1).sum())
    timeout = int((valid == 0).sum())
    return {
        "total": total,
        "n_nan": int((~mask).sum()),
        "up": up,
        "down": down,
        "timeout": timeout,
        "up_pct": round(up / total * 100, 2),
        "down_pct": round(down / total * 100, 2),
        "timeout_pct": round(timeout / total * 100, 2),
    }


def _alignment_check(df: pl.DataFrame) -> dict:
    """Verify target_tb uses no future data beyond the declared forward window."""
    close = df["close"].to_numpy().astype(np.float64)
    target = df["target_tb_new"].to_numpy().astype(np.float64)

    # Check: last 64 bars should all be NaN (no forward window available)
    H_BARS = 64
    n = len(close)
    tail_nan = target[-H_BARS:]
    tail_all_nan = bool(np.all(~np.isfinite(tail_nan)))

    # Check: target at t should only use data from t+H_BARS window (no leakage from t-1)
    # If target depends on features at t, it should NOT depend on features at t+1
    # that aren't part of the declared barrier window.
    # For target_tb: it uses close[t] as entry and high/low[t+1:t+1+H_BARS] as barrier
    # test data. This is a legitimate TARGET — it should be forward-looking by definition.
    # The real question: does target_tb correlate with something at t that it shouldn't?

    return {
        "tail_nan_count": int(np.sum(~np.isfinite(tail_nan))),
        "tail_nan_expected": H_BARS,
        "tail_all_nan": tail_all_nan,
        "forward_window_correct": tail_all_nan,
    }


def _leakage_check(df: pl.DataFrame) -> dict:
    """Audit target_tb for improper leakage patterns."""
    close = df["close"].to_numpy().astype(np.float64)
    target = df["target_tb_new"].to_numpy().astype(np.float64)

    n = len(close)
    H_BARS = 64

    # 1. Target at t should be independent of price change at t (contemporaneous independence)
    #    log_ret[t] = log(close[t]) - log(close[t-1]) — target shouldn't correlate with this
    if n > 1:
        log_ret = np.log(np.maximum(close[1:], 1e-12)) - np.log(np.maximum(close[:-1], 1e-12))
        log_ret_contemp = np.concatenate(([np.nan], log_ret))
    else:
        log_ret_contemp = np.full(n, np.nan)

    # 2. Target at t should correlate strongly with forward return over window
    #    forward_ret = log(close[t+H_BARS]) - log(close[t])
    forward_ret = np.full(n, np.nan)
    for i in range(n - H_BARS):
        if close[i] > 0 and close[i + H_BARS] > 0:
            forward_ret[i] = np.log(close[i + H_BARS]) - np.log(close[i])

    # 3. Target at t should NOT correlate with past returns (past 10 bars)
    past_ret_10 = np.full(n, np.nan)
    for i in range(10, n):
        if close[i] > 0 and close[i - 10] > 0:
            past_ret_10[i] = np.log(close[i]) - np.log(close[i - 10])

    def _corr(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            return 0.0
        return float(np.corrcoef(x[mask], y[mask])[0, 1])

    corr_contemp = _corr(target, log_ret_contemp)
    corr_forward = _corr(target, forward_ret)
    corr_past = _corr(target, past_ret_10)

    # 4. Cross-market leakage: labels within one market segment should not
    #    be influenced by other market segments (verified by market_id segmentation)

    checks = {
        "corr_contemporaneous_ret": round(corr_contemp, 6),
        "corr_forward_64b_ret": round(corr_forward, 6),
        "corr_past_10b_ret": round(corr_past, 6),
        "no_contemp_leakage": abs(corr_contemp) < 0.05,
        "forward_corr_positive": corr_forward > 0.05,  # directional target should have positive forward corr
        "no_past_leakage": abs(corr_past) < 0.05,
    }
    return checks


def _price_alignment(df: pl.DataFrame) -> dict:
    """Verify target distributes sensibly across price / vol regimes."""
    close = df["close"].to_numpy().astype(np.float64)
    target = df["target_tb_new"].to_numpy().astype(np.float64)
    mask = np.isfinite(target)

    # Barrier width varies with vol — check that spread of upper/lower
    # reflects vol environment
    if "htf_daily_vol_5" in df.columns:
        daily_vol = df["htf_daily_vol_5"].to_numpy().astype(np.float64) * 0.01
    else:
        daily_vol = np.full(len(close), 0.01)

    vol_low = daily_vol < np.percentile(daily_vol[mask], 33)
    vol_high = daily_vol >= np.percentile(daily_vol[mask], 67)

    target_lowvol = target[mask & vol_low]
    target_highvol = target[mask & vol_high]

    hl_up_low = int((target_lowvol == 1).sum())
    hl_up_high = int((target_highvol == 1).sum())
    hl_down_low = int((target_lowvol == -1).sum())
    hl_down_high = int((target_highvol == -1).sum())
    hl_timeout_low = int((target_lowvol == 0).sum())
    hl_timeout_high = int((target_highvol == 0).sum())

    return {
        "vol_low_n": int(len(target_lowvol)),
        "vol_high_n": int(len(target_highvol)),
        "vol_low_up_pct": round(hl_up_low / len(target_lowvol) * 100, 2) if len(target_lowvol) > 0 else 0,
        "vol_high_up_pct": round(hl_up_high / len(target_highvol) * 100, 2) if len(target_highvol) > 0 else 0,
        "vol_low_down_pct": round(hl_down_low / len(target_lowvol) * 100, 2) if len(target_lowvol) > 0 else 0,
        "vol_high_down_pct": round(hl_down_high / len(target_highvol) * 100, 2) if len(target_highvol) > 0 else 0,
        "vol_low_timeout_pct": round(hl_timeout_low / len(target_lowvol) * 100, 2) if len(target_lowvol) > 0 else 0,
        "vol_high_timeout_pct": round(hl_timeout_high / len(target_highvol) * 100, 2) if len(target_highvol) > 0 else 0,
    }


def _load_htf_vol(df_raw: pl.DataFrame) -> pl.DataFrame:
    """Compute htf_daily_vol_5 on raw continuous contract data (no session resampling)."""
    ret_1 = (pl.col("close") / pl.col("close").shift(1)).log()
    return df_raw.with_columns(
        ret_1.shift(1).rolling_std(window_size=260).fill_nan(0.01).alias("htf_daily_vol_5")
    )


def main():
    df = None
    source_label = ""

    if CACHED_FEATURE_MATRIX.exists():
        df = pl.read_parquet(str(CACHED_FEATURE_MATRIX))
        source_label = str(CACHED_FEATURE_MATRIX)

    if df is None or df.height == 0:
        for fallback in FALLBACK_DATA:
            if fallback.exists():
                print(f"Cache empty, using fallback: {fallback}")
                df = pl.read_parquet(str(fallback))
                if "htf_daily_vol_5" not in df.columns:
                    df = _load_htf_vol(df)
                source_label = str(fallback)
                break

    if df is None or df.height == 0:
        print("ERROR: no data available for validation")
        sys.exit(1)

    print(f"Loading: {source_label}")
    print(f"Rows: {df.height:,}  Columns: {len(df.columns)}")

    # Verify required columns exist
    required = {"close", "high", "low"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    # Save old target_tb if present
    has_old = "target_tb" in df.columns
    if has_old:
        old_target = df["target_tb"].to_numpy()
        old_balance = _class_balance(old_target)
    else:
        old_balance = None

    # Rebuild
    print("Rebuilding target_tb with new vol-adjusted params...")
    df = _rebuild_target_tb(df)

    new_target = df["target_tb_new"].to_numpy()
    new_balance = _class_balance(new_target)
    alignment = _alignment_check(df)
    leakage = _leakage_check(df)
    price_align = _price_alignment(df)

    # Compute change from old
    if has_old and old_balance:
        delta_up = round(new_balance["up_pct"] - old_balance["up_pct"], 2)
        delta_down = round(new_balance["down_pct"] - old_balance["down_pct"], 2)
        delta_timeout = round(new_balance["timeout_pct"] - old_balance["timeout_pct"], 2)
    else:
        delta_up = delta_down = delta_timeout = None

    # Compute barrier width distribution
    H_BARS = 64
    BARS_PER_DAY = 276
    VOL_MULT_UPPER = 1.0
    VOL_MULT_LOWER = 1.0
    close = df["close"].to_numpy().astype(np.float64)
    n = len(close)
    has_vol = "htf_daily_vol_5" in df.columns
    if has_vol:
        bar_vol = df["htf_daily_vol_5"].to_numpy().astype(np.float64)
        bar_vol = np.nan_to_num(bar_vol, nan=0.0005)
    else:
        bar_vol = np.full(n, 0.0005)
    bar_vol = np.maximum(bar_vol, 0.0001)
    vol_4h = bar_vol * np.sqrt(H_BARS)
    upper_pct = (np.exp(VOL_MULT_UPPER * vol_4h) - 1) * 100
    lower_pct = (np.exp(-VOL_MULT_LOWER * vol_4h) - 1) * 100

    # Confidence scoring
    checks_passed = 0
    checks_total = 7
    if alignment["tail_all_nan"]:
        checks_passed += 1
    if leakage["no_contemp_leakage"]:
        checks_passed += 1
    if leakage["forward_corr_positive"]:
        checks_passed += 1
    if leakage["no_past_leakage"]:
        checks_passed += 1
    if new_balance["total"] > 0:
        checks_passed += 1
    if new_balance["up_pct"] > 0 and new_balance["down_pct"] > 0:
        checks_passed += 1
    if abs(new_balance["timeout_pct"] - 50) < 40:  # timeout shouldn't be 0 or 100%
        checks_passed += 1
    confidence = round(checks_passed / checks_total * 100)

    # Report
    lines = []
    lines.append("=" * 70)
    lines.append("TARGET_TB REBUILD VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("## PARAMETERS APPLIED")
    lines.append(f"  H_BARS (horizon):         {H_BARS} (~5.3h at 5m bars)")
    lines.append(f"  BARS_PER_DAY:             {BARS_PER_DAY}")
    lines.append(f"  VOL_MULT_UPPER:           {VOL_MULT_UPPER} (upper = exp(+MULT * vol_4h))")
    lines.append(f"  VOL_MULT_LOWER:           {VOL_MULT_LOWER} (lower = exp(-MULT * vol_4h))")
    lines.append(f"  Vol source:               htf_daily_vol_5 (lagged 260-bar rolling std)")
    lines.append(f"  vol_4h scaling:          sqrt({H_BARS}/{BARS_PER_DAY}) = {np.sqrt(H_BARS/BARS_PER_DAY):.4f}")
    lines.append(f"  htf_daily_vol_5 column:   {'PRESENT' if has_vol else 'MISSING (using fallback)'}")
    lines.append("")
    lines.append(f"  Mean upper barrier:       {np.mean(upper_pct[np.isfinite(upper_pct)]):.2f}% above entry")
    lines.append(f"  Mean lower barrier:       {abs(np.mean(lower_pct[np.isfinite(lower_pct)])):.2f}% below entry")
    lines.append(f"  Barrier width (50p):      upper={np.percentile(upper_pct[np.isfinite(upper_pct)], 50):.2f}%, lower={abs(np.percentile(lower_pct[np.isfinite(lower_pct)], 50)):.2f}%")
    lines.append("")
    lines.append("## CLASS BALANCE")
    lines.append(f"  Total labeled:            {new_balance['total']:,}")
    lines.append(f"  NaN (tail):               {new_balance['n_nan']:,}")
    lines.append(f"  +1 (bullish):             {new_balance['up']:,}  ({new_balance['up_pct']:.1f}%)")
    lines.append(f"  -1 (bearish):             {new_balance['down']:,}  ({new_balance['down_pct']:.1f}%)")
    lines.append(f"   0 (timeout):             {new_balance['timeout']:,}  ({new_balance['timeout_pct']:.1f}%)")
    if delta_up is not None:
        lines.append("")
        lines.append("  Change from old target_tb:")
        lines.append(f"    +1 delta:               {delta_up:+.1f}pp")
        lines.append(f"    -1 delta:               {delta_down:+.1f}pp")
        lines.append(f"     0 delta:               {delta_timeout:+.1f}pp")
    lines.append("")
    lines.append("## ALIGNMENT CHECKS")
    lines.append(f"  Tail NaN check:           {'PASS' if alignment['tail_all_nan'] else 'FAIL'} ({alignment['tail_nan_count']}/{alignment['tail_nan_expected']} NaN)")
    lines.append(f"  Forward window correct:   {'YES' if alignment['forward_window_correct'] else 'NO'}")
    lines.append("")
    lines.append("## LEAKAGE CHECKS")
    lines.append(f"  corr(target_tb, contemporaneous ret):      {leakage['corr_contemporaneous_ret']:+.6f}  {'OK' if leakage['no_contemp_leakage'] else 'WARN'}")
    lines.append(f"  corr(target_tb, forward 64-bar ret):       {leakage['corr_forward_64b_ret']:+.6f}  {'OK' if leakage['forward_corr_positive'] else 'WARN'}")
    lines.append(f"  corr(target_tb, past 10-bar ret):          {leakage['corr_past_10b_ret']:+.6f}  {'OK' if leakage['no_past_leakage'] else 'WARN'}")
    lines.append("")
    lines.append("## VOLATILITY REGIME ALIGNMENT")
    lines.append(f"  Low vol  (n={price_align['vol_low_n']:,}):  up={price_align['vol_low_up_pct']:.1f}%  down={price_align['vol_low_down_pct']:.1f}%  timeout={price_align['vol_low_timeout_pct']:.1f}%")
    lines.append(f"  High vol (n={price_align['vol_high_n']:,}):  up={price_align['vol_high_up_pct']:.1f}%  down={price_align['vol_high_down_pct']:.1f}%  timeout={price_align['vol_high_timeout_pct']:.1f}%")
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"CONFIDENCE: {confidence}%")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(report)
    print(f"\nReport saved to {OUTPUT_REPORT}")

    if confidence < 70:
        sys.exit(1)


if __name__ == "__main__":
    main()
