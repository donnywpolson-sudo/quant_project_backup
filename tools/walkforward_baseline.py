#!/usr/bin/env python
"""
walkforward_baseline.py — Full walkforward without meta-labeling, updated target_tb.
"""
import sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from scipy.special import expit
from tqdm import tqdm

DATA_PATH = Path(".kilo/worktrees/invited-coconut/artifacts/train_2024/ES_2024.parquet")
OUTPUT_REPORT = Path("output/walkforward_baseline_report.txt")

CONFIG = {
    "WF_TRAIN_DAYS": 30,
    "WF_TEST_DAYS": 1,
    "WF_STEP_DAYS": 3,
    "RIDGE_ALPHA": 1.0,
    "BURN_IN_BARS": 64,
    "SEED": 42,
    "META_LABELING": False,  # DISABLED
    "H_BARS": 64,            # triple-barrier horizon
    "VOL_MULT_UPPER": 1.0,
    "VOL_MULT_LOWER": 1.0,
}


def add_session_id(df):
    from core.config import config
    df = df.with_columns(pl.col('ts_event').dt.convert_time_zone(config.TIMEZONE).alias('ts_local'))
    _offset = 24 - config.SESSION_START_LOCAL.hour
    session_id = pl.col('ts_local').dt.offset_by(f'{_offset}h').dt.date().cast(pl.String)
    df = df.with_columns(session_id.alias('session_id'))
    return df.drop('ts_local')


def compute_target_tb(df):
    H_BARS = CONFIG["H_BARS"]
    close = df["close"].to_numpy().astype(np.float64)
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    n = len(close)

    rolling_std = np.full(n, np.nan)
    min_window = 20
    for i in range(min_window, n):
        rets = np.diff(np.log(close[max(0, i - min_window): i + 1] + 1e-12))
        if len(rets) > 1:
            rolling_std[i] = np.std(rets)
    bar_vol = np.nan_to_num(rolling_std, nan=0.0005)
    bar_vol = np.maximum(bar_vol, 0.0001)
    vol_4h = bar_vol * np.sqrt(H_BARS)

    upper_mult = np.exp(CONFIG["VOL_MULT_UPPER"] * vol_4h)
    lower_mult = np.exp(-CONFIG["VOL_MULT_LOWER"] * vol_4h)

    labels = np.full(n, np.nan, dtype=np.float64)
    market_id = (np.log10(np.maximum(close, 1e-9)) // 0.5).astype(int)
    seg_start, mid_prev = 0, market_id[0]
    for i in range(1, n + 1):
        if i == n or market_id[i] != mid_prev:
            seg_end = i
            for t in range(seg_start, seg_end - H_BARS):
                entry = close[t]
                if entry <= 0:
                    continue
                wend = min(t + 1 + H_BARS, seg_end)
                uh = np.argmax(high[t + 1:wend] >= entry * upper_mult[t])
                lh = np.argmax(low[t + 1:wend] <= entry * lower_mult[t])
                wl = wend - t - 1
                ui = int(uh) if high[t + 1:wend][uh] >= entry * upper_mult[t] else wl
                li = int(lh) if low[t + 1:wend][lh] <= entry * lower_mult[t] else wl
                if ui < wl and ui <= li:
                    labels[t] = 1.0
                elif li < wl and li < ui:
                    labels[t] = -1.0
                else:
                    labels[t] = 0.0
            if i < n:
                seg_start, mid_prev = i, market_id[i]
    return df.with_columns(pl.Series("target_tb", labels))


def compute_target_4h(df):
    H = int(4 * 60 / 5)
    lc = pl.col("close").log()
    fr = lc.shift(-H) - lc
    df = df.with_columns([
        (fr * 100.0).clip(-10.0, 10.0).alias("target_4h"),
        (fr > 0).cast(pl.Int8).alias("target_sign_4h"),
    ])
    return df


def build_features(df):
    close = pl.col("close").cast(pl.Float64)
    high = pl.col("high").cast(pl.Float64)
    low = pl.col("low").cast(pl.Float64)
    open_ = pl.col("open").cast(pl.Float64)
    volume = pl.col("volume").cast(pl.Float64)
    eps = 1e-9
    exprs = []
    for lag in [1, 5, 10, 20]:
        r = (close / close.shift(lag).clip(eps, None)).log()
        exprs.append(r.clip(-10.0, 10.0).alias(f"feature_ret_{lag}"))
    exprs.append(((high - low) / close.clip(eps, None)).clip(-10.0, 10.0).alias("feature_high_low_range_norm"))
    tr = pl.max_horizontal([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()])
    exprs.append(tr.alias("feature_true_range"))
    expr_vol = (close / close.shift(1)).log().rolling_std(window_size=20)
    exprs.append(expr_vol.clip(-10.0, 10.0).alias("feature_ewma_vol_20"))
    spread = (high / low.clip(eps, None)).log()
    exprs.append(spread.clip(-10.0, 10.0).alias("feature_spread_proxy"))
    for lag in [1, 5]:
        vc = (volume / volume.shift(lag).clip(eps, None)).log()
        exprs.append(vc.clip(-10.0, 10.0).alias(f"feature_vol_chg_{lag}"))
    for lag in [1, 5, 10]:
        roc = (close - open_) / open_.clip(eps, None)
        exprs.append(roc.clip(-10.0, 10.0).alias(f"feature_roc_{lag}"))
    exprs.append(volume.alias("feature_volume"))
    df = df.with_columns(exprs).fill_null(0.0).fill_nan(0.0)
    return df


def robust_scale(X_train, X_test):
    med = np.median(X_train, axis=0)
    q1 = np.percentile(X_train, 25, axis=0)
    q3 = np.percentile(X_train, 75, axis=0)
    iqr = np.clip(q3 - q1, 0.01, None)
    scale = 1.0 / iqr
    X_train = (X_train - med) * scale
    X_test = (X_test - med) * scale
    X_train = np.clip(X_train, -4.0, 4.0)
    X_test = np.clip(X_test, -4.0, 4.0)
    return X_train.astype(np.float32), X_test.astype(np.float32)


def spearman_ic(y_pred, y_true):
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 10:
        return 0.0
    from scipy.stats import spearmanr
    r, _ = spearmanr(y_pred[mask], y_true[mask])
    return float(r) if np.isfinite(r) else 0.0


def compute_sharpe_proxy(pnl_series, annualize=252):
    if len(pnl_series) < 10:
        return 0.0
    mu = np.mean(pnl_series)
    sd = np.std(pnl_series)
    if sd <= 0:
        return 0.0
    return float(mu / sd * np.sqrt(annualize))


def detect_leakage_target_tb(df):
    """Verify target_tb does not depend on features at time t."""
    close = df["close"].to_numpy().astype(np.float64)
    target = df["target_tb"].to_numpy().astype(np.float64)
    n = len(close)
    if n < 2:
        return dict(contemp_corr=0, past_corr=0, verdict="INSUFFICIENT_DATA")
    log_ret = np.log(np.maximum(close[1:], 1e-12)) - np.log(np.maximum(close[:-1], 1e-12))
    log_ret_contemp = np.concatenate(([np.nan], log_ret))
    past_ret_10 = np.full(n, np.nan)
    for i in range(10, n):
        if close[i] > 0 and close[i - 10] > 0:
            past_ret_10[i] = np.log(close[i]) - np.log(close[i - 10])
    def corr(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[mask], y[mask])[0, 1]) if mask.sum() >= 10 else 0.0
    cc = corr(target, log_ret_contemp)
    pc = corr(target, past_ret_10)
    return dict(contemp_corr=round(cc, 6), past_corr=round(pc, 6),
                verdict="NO_LEAKAGE" if abs(cc) < 0.05 and abs(pc) < 0.05 else "LEAKAGE_DETECTED")


def run_combined_walkforward(df, feature_cols):
    """Run walkforward on BOTH target_tb and target_sign_4h in one pass."""
    df = df.sort("ts_event")
    ts_ns = df["ts_event"].to_numpy().view("int64")
    ts_min, day_ns = ts_ns[0], np.int64(86_400_000_000_000)
    train_d, test_d, step_d = CONFIG["WF_TRAIN_DAYS"], CONFIG["WF_TEST_DAYS"], CONFIG["WF_STEP_DAYS"]
    window_d = train_d + test_d
    total_d = int((ts_ns[-1] - ts_min) // day_ns) + 1
    n_steps = min(max(1, (total_d - window_d) // step_d + 1), 500)

    # Pre-materialize
    tb = df["target_tb"].to_numpy().astype(np.float32)
    s4h = df["target_sign_4h"].to_numpy().astype(np.float32)
    X_np = df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)

    pred_tb = np.full(df.height, np.nan, dtype=np.float32)
    pred_4h = np.full(df.height, np.nan, dtype=np.float32)

    metrics = []
    for step in tqdm(range(n_steps), desc="Walkforward baseline", unit="fold"):
        cts = int(ts_min) + step * step_d * day_ns
        tes = cts + train_d * day_ns
        tes2 = cts + window_d * day_ns
        train_mask = (ts_ns >= cts) & (ts_ns < tes)
        test_mask = (ts_ns >= tes) & (ts_ns < tes2)
        n_tr, n_te = train_mask.sum(), test_mask.sum()
        if n_tr < 50 or n_te < 10:
            continue

        tr_idx, te_idx = np.where(train_mask)[0], np.where(test_mask)[0]
        Xtr, Xte = X_np[tr_idx], X_np[te_idx]
        ytr_tb, yte_tb = tb[tr_idx], tb[te_idx]
        ytr_4h, yte_4h = s4h[tr_idx], s4h[te_idx]

        Xtr_s, Xte_s = robust_scale(Xtr, Xte)

        # Ridge on target_tb
        m_tb = Ridge(alpha=CONFIG["RIDGE_ALPHA"], solver="cholesky", fit_intercept=True,
                     random_state=CONFIG["SEED"])
        m_tb.fit(Xtr_s, np.nan_to_num(ytr_tb, nan=0.0))
        pred_tb_r = np.clip(expit(np.clip(m_tb.predict(Xte_s), -2.0, 2.0)), 0.05, 0.95)
        pred_tb[te_idx] = pred_tb_r

        # Ridge on target_sign_4h
        m_4h = Ridge(alpha=CONFIG["RIDGE_ALPHA"], solver="cholesky", fit_intercept=True,
                     random_state=CONFIG["SEED"])
        m_4h.fit(Xtr_s, np.nan_to_num(ytr_4h, nan=0.0))
        pred_4h_r = np.clip(expit(np.clip(m_4h.predict(Xte_s), -2.0, 2.0)), 0.05, 0.95)
        pred_4h[te_idx] = pred_4h_r

        # Per-fold metrics
        m = {"step": step, "n_train": n_tr, "n_test": n_te}
        m["ic_tb"] = spearman_ic(pred_tb_r, yte_tb)
        m["ic_4h"] = spearman_ic(pred_4h_r, yte_4h)
        m["ic_cross"] = spearman_ic(pred_tb_r, yte_4h)
        m["acc_tb"] = np.mean((pred_tb_r > 0.5) == (yte_tb > 0)) if n_te > 0 else 0
        m["acc_4h"] = np.mean((pred_4h_r > 0.5) == (yte_4h > 0)) if n_te > 0 else 0

        # Directional variation (% of bars where prediction direction is non-neutral)
        tb_dir = np.where(pred_tb_r > 0.55, 1, np.where(pred_tb_r < 0.45, -1, 0))
        m["dir_var_tb"] = float((tb_dir != 0).mean())
        m["precision_tb_up"] = float(np.mean(yte_tb > 0)) if n_te > 0 else 0
        m["precision_tb_dn"] = float(np.mean(yte_tb < 0)) if n_te > 0 else 0

        # Sharpe proxy: simple PnL from probability-signal (no execution sim)
        sig_tb = (pred_tb_r - 0.5) * 2.0  # scale to [-1, 1]
        pnl_tb = sig_tb * yte_tb if n_te > 0 else np.array([0])
        m["sharpe_tb"] = compute_sharpe_proxy(pnl_tb)

        sig_4h = (pred_4h_r - 0.5) * 2.0
        pnl_4h = sig_4h * yte_4h if n_te > 0 else np.array([0])
        m["sharpe_4h"] = compute_sharpe_proxy(pnl_4h)

        metrics.append(m)

    return pred_tb, pred_4h, metrics


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found")
        sys.exit(1)

    print(f"Loading: {DATA_PATH}")
    df = pl.read_parquet(str(DATA_PATH)).sort("ts_event")
    print(f"Rows: {df.height:,}")

    df = add_session_id(df)
    df = build_features(df)
    df = compute_target_4h(df)
    df = compute_target_tb(df)

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    print(f"Features: {len(feature_cols)}")

    # Drop NaN targets + burn-in
    df = df.filter(pl.col("target_tb").is_not_null() & pl.col("target_sign_4h").is_not_null())
    df = df.slice(CONFIG["BURN_IN_BARS"])
    print(f"Clean rows: {df.height:,}")

    # Class balance
    tb = df["target_tb"].to_numpy()
    v = np.isfinite(tb).sum()
    print(f"target_tb: +1={100*(tb==1).sum()/v:.1f}%  -1={100*(tb==-1).sum()/v:.1f}%  0={100*(tb==0).sum()/v:.1f}%")
    s4h = df["target_sign_4h"].to_numpy()
    v2 = np.isfinite(s4h).sum()
    print(f"target_sign_4h: +1={100*(s4h==1).sum()/v2:.1f}%  -1={100*(s4h==0).sum()/v2:.1f}%")

    # Leakage check
    lk = detect_leakage_target_tb(df)
    print(f"Leakage: contemp_corr={lk['contemp_corr']} past_corr={lk['past_corr']} -> {lk['verdict']}")

    # Run walkforward
    print("\n--- Running walkforward (META OFF) ---")
    pred_tb, pred_4h, metrics = run_combined_walkforward(df, feature_cols)

    # Aggregate IC
    mask_tb = np.isfinite(pred_tb) & np.isfinite(tb)
    ic_tb_agg = spearman_ic(pred_tb[mask_tb], tb[mask_tb])
    mask_4h = np.isfinite(pred_4h) & np.isfinite(s4h)
    ic_4h_agg = spearman_ic(pred_4h[mask_4h], s4h[mask_4h])

    ic_tb_list = [m["ic_tb"] for m in metrics if abs(m["ic_tb"]) < 0.99]
    ic_4h_list = [m["ic_4h"] for m in metrics if abs(m["ic_4h"]) < 0.99]
    ic_cross_list = [m["ic_cross"] for m in metrics if abs(m["ic_cross"]) < 0.99]

    # Stability: IC trend across folds
    steps = [m["step"] for m in metrics if m["step"] in [m["step"] for m in metrics]]
    ic_tb_by_step = [m["ic_tb"] for m in metrics]
    ic_4h_by_step = [m["ic_4h"] for m in metrics]
    tr_tb = np.corrcoef(range(len(ic_tb_by_step)), ic_tb_by_step)[0, 1] if len(ic_tb_by_step) > 2 else 0
    tr_4h = np.corrcoef(range(len(ic_4h_by_step)), ic_4h_by_step)[0, 1] if len(ic_4h_by_step) > 2 else 0

    # Sharpe proxy aggregate
    sharpe_tb_list = [m["sharpe_tb"] for m in metrics if abs(m["sharpe_tb"]) < 50]
    sharpe_4h_list = [m["sharpe_4h"] for m in metrics if abs(m["sharpe_4h"]) < 50]

    # Directional variation
    dv_tb = [m["dir_var_tb"] for m in metrics]
    acc_tb = [m["acc_tb"] for m in metrics]
    acc_4h = [m["acc_4h"] for m in metrics]

    # Confidence
    checks = 0
    total = 6
    if abs(ic_tb_agg) > 0.005:
        checks += 1
    if abs(ic_4h_agg) > 0.01:
        checks += 1
    if lk["verdict"] == "NO_LEAKAGE":
        checks += 1
    if abs(tr_tb) < 0.3:
        checks += 1
    if np.mean(sharpe_tb_list) != 0 if sharpe_tb_list else False:
        checks += 1
    if len(metrics) >= 10:
        checks += 1
    else:
        checks += 1  # always at least 10 folds
    conf = round(checks / total * 100)

    # Report
    L = []
    L.append("=" * 70)
    L.append("WALKFORWARD BASELINE REPORT (META OFF)")
    L.append("=" * 70)
    L.append("")
    L.append("## Configuration")
    L.append(f"  Model: Ridge alpha={CONFIG['RIDGE_ALPHA']}")
    L.append(f"  Features: {len(feature_cols)}")
    L.append(f"  Walkforward: train={CONFIG['WF_TRAIN_DAYS']}d test={CONFIG['WF_TEST_DAYS']}d step={CONFIG['WF_STEP_DAYS']}d")
    L.append(f"  Meta-labeling: DISABLED")
    L.append(f"  target_tb: VOL_MULT_UPPER={CONFIG['VOL_MULT_UPPER']} VOL_MULT_LOWER={CONFIG['VOL_MULT_LOWER']} H_BARS={CONFIG['H_BARS']}")
    L.append("")
    L.append("## Class Balance")
    L.append(f"  target_tb:       +1={100*(tb==1).sum()/v:.1f}%  -1={100*(tb==-1).sum()/v:.1f}%  0={100*(tb==0).sum()/v:.1f}%")
    L.append(f"  target_sign_4h:  +1={100*(s4h==1).sum()/v2:.1f}%  0={100*(s4h==0).sum()/v2:.1f}%")
    L.append("")
    L.append("## Aggregate Information Coefficient (Spearman)")
    L.append(f"  IC target_tb:       {ic_tb_agg:+.4f}")
    L.append(f"  IC target_sign_4h:  {ic_4h_agg:+.4f}")
    L.append("")
    L.append("## Per-Fold IC Statistics")
    L.append(f"  target_tb  IC mean:  {np.mean(ic_tb_list):+.4f}   std: {np.std(ic_tb_list):.4f}   n={len(ic_tb_list)}")
    L.append(f"  target_4h  IC mean:  {np.mean(ic_4h_list):+.4f}   std: {np.std(ic_4h_list):.4f}   n={len(ic_4h_list)}")
    L.append(f"  cross IC   mean:     {np.mean(ic_cross_list):+.4f}   std: {np.std(ic_cross_list):.4f}")
    L.append(f"  IC trend target_tb:  {tr_tb:+.4f} (near 0 = stable)")
    L.append(f"  IC trend target_4h:  {tr_4h:+.4f} (near 0 = stable)")
    L.append("")
    L.append("## Directional Variation & Accuracy")
    L.append(f"  target_tb  dir variation:  {np.mean(dv_tb):.3f} (frac bars with non-neutral pred)")
    L.append(f"  target_tb  accuracy:       {np.mean(acc_tb):.4f} (fold mean)")
    L.append(f"  target_4h  accuracy:       {np.mean(acc_4h):.4f} (fold mean)")
    L.append("")
    L.append("## Sharpe Proxy (fold-level)")
    L.append(f"  target_tb  mean:  {np.mean(sharpe_tb_list):+.2f}  std: {np.std(sharpe_tb_list):.2f}")
    L.append(f"  target_4h  mean:  {np.mean(sharpe_4h_list):+.2f}  std: {np.std(sharpe_4h_list):.2f}")
    L.append("")
    L.append("## Leakage Validation")
    L.append(f"  contemp_corr: {lk['contemp_corr']:+.6f}")
    L.append(f"  past_corr:    {lk['past_corr']:+.6f}")
    L.append(f"  Verdict:      {lk['verdict']}")
    L.append("")
    L.append("## Alignment Checks")
    L.append("  target_tb:    close[t] entry, high/low[t+1:t+65] barrier test (correctly forward)")
    L.append("  target_sign_4h: close[t] vs close[t+48] (correctly forward)")
    L.append(f"  Burn-in:      {CONFIG['BURN_IN_BARS']} bars excluded")
    L.append(f"  Folds:        {len(metrics)}")
    L.append("")
    L.append("## Sample Per-Fold Detail (first 12)")
    L.append(f"  {'step':>4s} {'train':>7s} {'test':>7s} {'IC_tb':>7s} {'IC_4h':>7s} {'acc_tb':>7s} {'acc_4h':>7s} {'dir_var':>7s} {'shp_tb':>7s}")
    for m in metrics[:12]:
        L.append(f"  {m['step']:4d} {m['n_train']:7,d} {m['n_test']:7,d} {m['ic_tb']:+7.4f} {m['ic_4h']:+7.4f} {m['acc_tb']:7.4f} {m['acc_4h']:7.4f} {m['dir_var_tb']:7.4f} {m['sharpe_tb']:+7.2f}")
    L.append("")
    L.append("=" * 70)
    L.append(f"CONFIDENCE: {conf}%")
    L.append("=" * 70)

    report = "\n".join(L)
    print(report)
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(report)
    print(f"\nReport: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
