#!/usr/bin/env python
"""
feature_leakage_audit.py
Diagnostic script to detect look-ahead bias (data leakage) in the feature matrix.

For every feature X_t, computes Pearson correlation with the target return r_{t+h}
for h in [-5, +5]. Flags any feature where the forward correlation |corr(X_t, r_{t+1})|
significantly exceeds the backward correlation |corr(X_t, r_{t-5})| (p < 0.01),
indicating the feature "sees" into the future.
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path("artifacts/full_feature_matrix_c695d166720c.parquet")
OUTPUT_REPORT = Path("leakage_audit_report.md")

# Columns to exclude from feature analysis (OHLCV, timestamps, targets, metadata)
EXCLUDE_COLUMNS = {
    "open", "high", "low", "close", "volume",
    "ts_event", "session_id",
    "target_5m", "target_sign",
    "target_4h", "target_sign_4h",
    "target_1h", "target_sign_1h",
    "regime",
}

# Feature column prefixes to include in the scan
FEATURE_PREFIXES = ("feature_", "cross_", "pair_", "htf_", "zscore_", "ratio_")

# Horizon range for correlation analysis
HORIZON_RANGE = range(-5, 6)  # -5 to +5 inclusive

# Significance threshold
P_VALUE_THRESHOLD = 0.01

# Minimum forward correlation magnitude to even consider suspicious
MIN_FORWARD_CORR = 0.02


def load_data(path: Path) -> pl.DataFrame:
    """Load the parquet feature matrix."""
    if not path.exists():
        print(f"ERROR: Data file not found: {path}", flush=True)
        sys.exit(1)

    print(f"[AUDIT] Loading data from {path} ...", flush=True)
    df = pl.read_parquet(str(path))

    # Ensure data is sorted by timestamp
    if "ts_event" in df.columns:
        df = df.sort("ts_event")
        print(f"[AUDIT] Rows: {df.height:,}  |  TS range: {df['ts_event'].min()} -> {df['ts_event'].max()}", flush=True)
    else:
        print(f"[AUDIT] Rows: {df.height:,}  (no ts_event column)", flush=True)

    return df


def identify_feature_columns(df: pl.DataFrame) -> list:
    """Identify candidate feature columns to audit."""
    candidate = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue
        if col.startswith(FEATURE_PREFIXES):
            candidate.append(col)
        # Also catch z-score and ratio columns that might not start with a pattern
        elif any(prefix in col for prefix in ("_zscore", "zscore", "_ratio", "ratio_")):
            candidate.append(col)
        # Also catch any column that looks like a feature but was missed
        elif (col.startswith("1h_") or col.startswith("daily_")) and col not in EXCLUDE_COLUMNS:
            candidate.append(col)

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for c in candidate:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def compute_return_series(df: pl.DataFrame) -> np.ndarray:
    """
    Compute the 5-minute log-return series from close price.
    Returns: numpy array of log returns r_t (same length as df).
    First element is NaN.
    """
    close = df["close"].to_numpy().astype(np.float64)
    log_close = np.log(np.maximum(close, 1e-12))
    returns = log_close[1:] - log_close[:-1]
    # Pad with NaN at the start so the series aligns with feature rows
    returns = np.concatenate(([np.nan], returns))
    return returns


def safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Compute Pearson correlation on the joint valid (non-NaN, non-inf) indices.
    Returns (correlation, p_value, n_valid).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < 10:
        return 0.0, 1.0, n
    try:
        r, p = pearsonr(x[mask], y[mask])
        if np.isnan(r) or np.isinf(r):
            return 0.0, 1.0, n
        return float(r), float(p), n
    except Exception:
        return 0.0, 1.0, n


def analyze_feature(feature_values: np.ndarray, returns: np.ndarray) -> dict:
    """
    For a single feature series, compute correlations with returns at all horizons
    h in [-5, +5].

    Returns a dict with corr_at_horizon[h] = (corr, pval, n) for each h.
    """
    result = {}
    n_total = len(feature_values)

    for h in HORIZON_RANGE:
        if h < 0:
            # Backward: feature at t correlated with return at t+h (past return)
            # Align: feature[t] with return[t+h]
            # feature[ abs(h): ] aligns with return[ : n_total-abs(h) ]
            f_slice = feature_values[-h:]  # last N+h elements
            r_slice = returns[:h]  # first N+h elements (h is negative, so :h goes up to N+h)
        elif h > 0:
            # Forward: feature at t correlated with return at t+h (future return)
            # feature[ : n_total-h ] aligns with return[ h: ]
            f_slice = feature_values[:-h]
            r_slice = returns[h:]
        else:
            # h == 0: contemporaneous
            f_slice = feature_values
            r_slice = returns

        min_len = min(len(f_slice), len(r_slice))
        if min_len < 10:
            result[h] = (0.0, 1.0, 0)
            continue

        cr, cp, cn = safe_pearson(f_slice[:min_len], r_slice[:min_len])
        result[h] = (cr, cp, cn)

    return result


def classify_feature(horizon_corrs: dict) -> tuple:
    """
    Classify a feature as 'clean' or 'cheating' based on the correlation profile.

    A feature is flagged as cheating if:
      |corr(X_t, r_{t+1})| > |corr(X_t, r_{t-5})|
      AND the forward corr p-value < 0.01
      AND |corr(X_t, r_{t+1})| >= MIN_FORWARD_CORR (to avoid noise)

    Returns (classification, forward_corr, forward_pval, backward_corr_h5, backward_pval_h5)
    """
    fwd_corr, fwd_pval, _ = horizon_corrs.get(1, (0.0, 1.0, 0))
    bwd_corr, bwd_pval, _ = horizon_corrs.get(-5, (0.0, 1.0, 0))
    bwd_corr_h1, bwd_pval_h1, _ = horizon_corrs.get(-1, (0.0, 1.0, 0))

    abs_fwd = abs(fwd_corr)
    abs_bwd = abs(bwd_corr)

    # Cheating test: forward correlation magnitude exceeds backward by a meaningful margin
    # and forward correlation is statistically significant
    is_cheating = (
        abs_fwd > abs_bwd
        and fwd_pval < P_VALUE_THRESHOLD
        and abs_fwd >= MIN_FORWARD_CORR
    )

    classification = "CHEATING" if is_cheating else "CLEAN"

    return (
        classification,
        fwd_corr,
        fwd_pval,
        bwd_corr,
        bwd_pval,
        bwd_corr_h1,
        bwd_pval_h1,
    )


def generate_hypothesis(feature_name: str, horizon_corrs: dict) -> str:
    """
    Generate a hypothesis about the source of look-ahead bias based on the
    feature name pattern and its correlation profile.
    """
    fwd_corr, _, _ = horizon_corrs.get(1, (0.0, 1.0, 0))

    hypotheses = []

    # Check for HTF features that might have forward-looking joins
    if feature_name.startswith("htf_"):
        if "daily_return" in feature_name or "hourly_return" in feature_name:
            hypotheses.append(
                "HTF return feature may be computed from future-joined daily/hourly bars. "
                "The `join_asof(strategy='backward')` should prevent this, but if the join key or "
                "sort order is wrong, future HTF data could leak into the 5-min row."
            )
        elif "distance_to_daily" in feature_name:
            hypotheses.append(
                "HTF distance-to-daily-high/low may use the current day's final high/low before the day "
                "is complete (peeking at the daily bar's eventual close). Daily H/L for an incomplete "
                "day should be NaN or forward-filled from the prior day, not the current unfinished bar."
            )
        elif "volatility_ratio" in feature_name:
            hypotheses.append(
                "HTF volatility ratio divides 1h vol by daily vol. The daily vol may be computed "
                "from the current (incomplete) daily bar, leaking future intraday information."
            )
        elif "trend_alignment" in feature_name:
            hypotheses.append(
                "HTF trend alignment sign-check may reference the current incomplete daily bar's "
                "trend slope, which contains future close information."
            )
        else:
            hypotheses.append(
                "HTF feature may be derived from a join that uses the current day's incomplete bar "
                "data (daily_high/daily_low/daily_close) before those values are final."
            )

    # Check for cross-features (HTF × LTF interactions)
    elif feature_name.startswith("cross_"):
        if any(kw in feature_name for kw in ["daily_return", "daily_trend", "daily_vol"]):
            hypotheses.append(
                "Cross-feature multiplies an LTF feature by an HTF feature that may contain "
                "future-leaked daily data (incomplete daily bar OHLC)."
            )
        elif "htf_distance" in feature_name:
            hypotheses.append(
                "Cross-feature involving distance-to-daily-high/low multiplies by a value that "
                "may reference the current day's ultimate high/low before the day ends."
            )
        elif "htf_hourly_trend_alignment" in feature_name:
            hypotheses.append(
                "Cross-feature involving htf_hourly_trend_alignment uses a sign comparison that "
                "may incorporate the current (incomplete) period's realized values."
            )
        else:
            hypotheses.append(
                "Cross-feature propagates leakage from its HTF component. Check the upstream "
                "HTF feature generation for look-ahead bias."
            )

    # Check for pair features
    elif feature_name.startswith("pair_"):
        if "vwap" in feature_name:
            hypotheses.append(
                "Pairwise interaction involving VWAP deviation. VWAP for an incomplete 5-min bar "
                "may use the bar's final close/volume, leaking the bar's own outcome into the feature."
            )
        elif any(kw in feature_name for kw in ["regime", "_regime"]):
            hypotheses.append(
                "Pairwise interaction involving regime. The regime feature is computed from "
                "smoothed volatility which may have a look-ahead component if rolling windows "
                "are centered or if the vol computation uses future returns."
            )
        else:
            # Check if both components might be suspicious
            components = feature_name.replace("pair_", "").split("_x_")
            for comp in components:
                if "daily_return" in comp or "htf_" in comp:
                    hypotheses.append(
                        f"Pair feature includes HTF component '{comp}' which may leak future "
                        "daily bar information."
                    )
                    break
            if not hypotheses:
                hypotheses.append(
                    "Pairwise feature may contain a component with rolling-window look-ahead "
                    "bias. Check the upstream feature generation for improper shift() or "
                    "rolling() with center=True."
                )

    # Check for rolling-quantile / rolling-moment features
    elif any(kw in feature_name for kw in ["quantile", "skew", "kurt"]):
        hypotheses.append(
            "Rolling moment/quantile feature. If the rolling window includes the current bar's "
            "full data (including close) before it's used as a lagged feature, the computation "
            "may be quasi-forward-looking. Rolling_std/skew/kurt with the observation included "
            "is standard, but if the window is uncentered and includes t=0, this is generally "
            "acceptable - check if min_periods or clip values are leaking future info."
        )

    # Check for z-score features
    elif "zscore" in feature_name:
        hypotheses.append(
            "Z-score feature uses rolling mean/std. If the rolling window includes the current "
            "observation in its normalization, this can create a subtle forward bias via the "
            "standard deviation denominator. Consider using expanding windows or shift(1) before "
            "computing z-scores."
        )

    # Check for features with regime interactions
    elif "regime" in feature_name:
        hypotheses.append(
            "Regime-interacted feature. The regime variable is computed from smoothed volatility "
            "which depends on a rolling median/mean of realized returns. If these rolling "
            "computations include the contemporaneous or future period, leakage occurs."
        )

    else:
        # Generic catch-all hypothesis
        abs_fwd = abs(fwd_corr)
        if abs_fwd > 0.5:
            hypotheses.append(
                "Extremely strong forward correlation (>0.5) suggests direct future-price leakage. "
                "Likely cause: a `shift(-k)` or forward-looking operation in feature construction, "
                "or a join that matches on future timestamps."
            )
        elif abs_fwd > 0.1:
            hypotheses.append(
                "Moderate forward correlation suggests indirect leakage. Possible causes: "
                "(1) Rolling window computations that include future bars, "
                "(2) HTF daily bar data from the current (incomplete) day, "
                "(3) Forward-fill of data that hasn't been released yet, "
                "(4) Improperly aligned multi-timeframe joins."
            )
        else:
            hypotheses.append(
                "Weak but statistically significant forward correlation. May be a false positive "
                "from multiple testing. Verify by examining the full correlation profile across "
                "all horizons."
            )

    return " ".join(hypotheses)


def format_corr_profile(horizon_corrs: dict) -> str:
    """Format the correlation profile across all horizons as a markdown table row."""
    parts = []
    for h in sorted(horizon_corrs.keys()):
        cr, cp, cn = horizon_corrs[h]
        sig = "***" if cp < 0.001 else ("**" if cp < 0.01 else ("*" if cp < 0.05 else ""))
        parts.append(f"h={h:+d}: {cr:+.4f}{sig} (p={cp:.1e}, n={cn:,})")
    return "<br>".join(parts)


def run_audit(data_path: Path) -> dict:
    """Main audit routine."""
    # Load data
    df = load_data(data_path)

    # Compute returns from close
    print("[AUDIT] Computing 5-min return series from close price...", flush=True)
    returns = compute_return_series(df)
    n_valid_returns = np.isfinite(returns).sum()
    print(f"[AUDIT] Valid return observations: {n_valid_returns:,}", flush=True)

    # Identify feature columns
    feature_cols = identify_feature_columns(df)
    print(f"[AUDIT] Identified {len(feature_cols)} feature columns to audit.", flush=True)

    # Audit each feature
    clean_features = []
    cheating_features = []

    total = len(feature_cols)
    for i, col in enumerate(feature_cols):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[AUDIT] Progress: {i+1}/{total} features analyzed...", flush=True)

        # Extract feature values
        try:
            vals = df[col].to_numpy().astype(np.float64)
        except Exception:
            # Skip non-numeric columns
            continue

        # Replace inf with NaN
        vals = np.where(np.isfinite(vals), vals, np.nan)
        n_valid_feature = np.isfinite(vals).sum()
        if n_valid_feature < 50:
            continue  # too few valid observations

        # Analyze correlations at all horizons
        horizon_corrs = analyze_feature(vals, returns)

        # Classify
        classification, fwd_corr, fwd_pval, bwd_corr, bwd_pval, bwd_corr_h1, bwd_pval_h1 = classify_feature(horizon_corrs)

        # Build profile string for reporting
        profile_str = format_corr_profile(horizon_corrs)

        entry = {
            "name": col,
            "forward_corr_h1": fwd_corr,
            "forward_pval_h1": fwd_pval,
            "backward_corr_h5": bwd_corr,
            "backward_pval_h5": bwd_pval,
            "backward_corr_h1": bwd_corr_h1,
            "backward_pval_h1": bwd_pval_h1,
            "correlation_profile": profile_str,
            "hypothesis": generate_hypothesis(col, horizon_corrs) if classification == "CHEATING" else "",
        }

        if classification == "CHEATING":
            cheating_features.append(entry)
        else:
            clean_features.append(entry)

    print(f"\n[AUDIT] COMPLETE.", flush=True)
    print(f"  Clean features:    {len(clean_features)}", flush=True)
    print(f"  Cheating features:  {len(cheating_features)}", flush=True)

    return {
        "clean": clean_features,
        "cheating": cheating_features,
        "total_features": len(feature_cols),
        "total_rows": df.height,
    }


def generate_report(results: dict, output_path: Path):
    """Generate the markdown audit report."""
    clean = results["clean"]
    cheating = results["cheating"]
    total = results["total_features"]
    rows = results["total_rows"]

    lines = []
    lines.append("# Feature Leakage Audit Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Total feature columns audited:** {total}")
    lines.append(f"- **Data rows analyzed:** {rows:,}")
    lines.append(f"- **Clean features:** {len(clean)}")
    lines.append(f"- **Cheating features (look-ahead bias detected):** {len(cheating)}")
    lines.append("")
    lines.append(
        "**Methodology:** For each feature $X_t$, we computed the Pearson correlation "
        "with the 5-minute log-return $r_{t+h}$ at horizons $h \\in [-5, +5]$. "
        "A feature is flagged as \"cheating\" if:"
    )
    lines.append("")
    lines.append("1. $|\\text{corr}(X_t, r_{t+1})| > |\\text{corr}(X_t, r_{t-5})|$")
    lines.append(f"2. The forward correlation is statistically significant ($p < {P_VALUE_THRESHOLD}$)")
    lines.append(f"3. $|corr(X_t, r_{{t+1}})| \\geq {MIN_FORWARD_CORR}$ (minimum magnitude threshold)")
    lines.append("")
    lines.append("This pattern — strong forward correlation with weak backward correlation — "
                "is the telltale signature of look-ahead bias: the feature \"knows\" about "
                "future returns at $t+1$ but not past returns at $t-5$.")
    lines.append("")

    # --- Cheating Features Section ---
    lines.append("---")
    lines.append("")
    lines.append("## 🚨 Cheating Features (Look-Ahead Bias Detected)")
    lines.append("")

    if cheating:
        # Sort by forward correlation magnitude (descending)
        cheating_sorted = sorted(cheating, key=lambda x: abs(x["forward_corr_h1"]), reverse=True)

        lines.append(f"| # | Feature | Forward Corr (h=+1) | Forward p-value | Backward Corr (h=-5) | Backward Corr (h=-1) | Verdict |")
        lines.append(f"|---|---------|---------------------|-----------------|----------------------|----------------------|---------|")
        for idx, entry in enumerate(cheating_sorted, 1):
            fwd = entry["forward_corr_h1"]
            fwd_p = entry["forward_pval_h1"]
            bwd = entry["backward_corr_h5"]
            bwd_h1 = entry["backward_corr_h1"]
            name = entry["name"]
            lines.append(
                f"| {idx} | `{name}` | {fwd:+.4f} | {fwd_p:.2e} | {bwd:+.4f} | {bwd_h1:+.4f} | CHEATING |"
            )

        lines.append("")
        lines.append("### Detailed Correlation Profiles & Hypotheses")
        lines.append("")

        for idx, entry in enumerate(cheating_sorted, 1):
            lines.append(f"#### {idx}. `{entry['name']}`")
            lines.append("")
            lines.append(f"**Correlation Profile (all horizons):**")
            lines.append(f"> {entry['correlation_profile']}")
            lines.append("")
            lines.append(f"**Bias Hypothesis:** {entry['hypothesis']}")
            lines.append("")
    else:
        lines.append("> ✅ No cheating features detected.")
        lines.append("")

    # --- Clean Features Summary ---
    lines.append("---")
    lines.append("")
    lines.append("## ✅ Clean Features")
    lines.append("")
    if clean:
        lines.append(f"The following {len(clean)} features passed the look-ahead bias test.")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary>Click to expand clean feature list</summary>")
        lines.append("")
        for entry in sorted(clean, key=lambda x: x["name"]):
            lines.append(f"- `{entry['name']}`")
        lines.append("")
        lines.append("</details>")
    else:
        lines.append("> No clean features (all features appear to have some level of leakage)")
    lines.append("")

    # --- Recommendations ---
    lines.append("---")
    lines.append("")
    lines.append("## Remediation Recommendations")
    lines.append("")
    lines.append("Based on the audit findings, the following actions are recommended:")
    lines.append("")
    lines.append("### 1. HTF Feature Leakage (htf_* and cross_* involving HTF)")
    lines.append("")
    lines.append(
        "The `htf_*` features are constructed from daily and 1-hour bar data joined via "
        "`join_asof(strategy='backward')`. While `strategy='backward'` is correct in principle, "
        "if the daily bar timestamps are set to the bar's **close time** (e.g., 16:00 ET) rather "
        "than the bar's **open time** (e.g., 09:30 ET), then during the trading day the \"current\" "
        "daily bar's data leaks its eventual close price into the 5-minute rows."
    )
    lines.append("")
    lines.append(
        "**Fix:** Ensure daily bar timestamps represent the bar's **start** time (or the "
        "prior day's close timestamp). The `join_asof` should match each 5-min row to the "
        "**previous** completed daily bar, not the current incomplete one."
    )
    lines.append("")
    lines.append("### 2. Rolling Window Computations")
    lines.append("")
    lines.append(
        "Features using `rolling_std`, `rolling_mean`, `rolling_quantile`, `rolling_skew`, "
        "or `rolling_kurt` include the current observation (t) in the window, which is standard "
        "practice. However, if the window center is misconfigured or the computation uses future "
        "data (e.g., `shift(-1)` before rolling), leakage occurs."
    )
    lines.append("")
    lines.append(
        "**Fix:** Audit all `rolling_*` calls. If any feature is intended to be strictly "
        "lagged (not including t=0), apply `.shift(1)` before the rolling computation."
    )
    lines.append("")
    lines.append("### 3. Z-Score Features")
    lines.append("")
    lines.append(
        "Z-scores computed with rolling mean/std that include the current observation can create "
        "subtle forward bias because the denominator (std) is influenced by the current return. "
        "This is particularly problematic if the z-score is then used to predict the next return."
    )
    lines.append("")
    lines.append(
        "**Fix:** Compute rolling mean and std on `shift(1)`-lagged values to exclude the "
        "current observation, then apply the z-score formula to the current value using those "
        "lagged statistics."
    )
    lines.append("")
    lines.append("### 4. Regime Features")
    lines.append("")
    lines.append(
        "The regime variable is derived from smoothed volatility which itself depends on "
        "realized returns. If the volatility computation window includes the current "
        "observation's return (which contributes to the regime classification), then the "
        "regime implicitly contains information about the current bar's outcome."
    )
    lines.append("")
    lines.append(
        "**Fix:** Lag the volatility estimate by at least 1 bar before computing the regime "
        "flag: `regime_t = f(vol_{t-1})`."
    )
    lines.append("")
    lines.append("### 5. VWAP Deviation")
    lines.append("")
    lines.append(
        "VWAP for an incomplete bar uses the bar's current close/volume, meaning the feature "
        "contains the bar's own price action. While this is standard for execution algos, it "
        "creates look-ahead bias in a predictive context."
    )
    lines.append("")
    lines.append(
        "**Fix:** Use the previous bar's completed VWAP, or compute VWAP from t-1 to t-window."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by `feature_leakage_audit.py` using data from `{DATA_PATH}`.*")

    report_content = "\n".join(lines)

    output_path.write_text(report_content, encoding="utf-8")
    print(f"\n[AUDIT] Report written to {output_path}", flush=True)


def main():
    print("=" * 70, flush=True)
    print(" FEATURE LEAKAGE AUDIT", flush=True)
    print("=" * 70, flush=True)
    print(f" Data: {DATA_PATH}", flush=True)
    print(f" Output: {OUTPUT_REPORT}", flush=True)
    print(f" P-value threshold: {P_VALUE_THRESHOLD}", flush=True)
    print(f" Min forward corr: {MIN_FORWARD_CORR}", flush=True)
    print(f" Horizon range: {list(HORIZON_RANGE)}", flush=True)
    print("=" * 70, flush=True)

    results = run_audit(DATA_PATH)
    generate_report(results, OUTPUT_REPORT)

    num_cheating = len(results["cheating"])
    if num_cheating > 0:
        print(f"\n⚠️  WARNING: {num_cheating} cheating features detected!", flush=True)
        print("  Review the report for detailed findings and remediation advice.", flush=True)
        sys.exit(1)
    else:
        print("\n✅ All features appear clean. No look-ahead bias detected.", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()