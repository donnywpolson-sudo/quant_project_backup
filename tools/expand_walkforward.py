#!/usr/bin/env python
"""
expand_walkforward.py — Expanded feature set (14 -> 150+) walkforward with Ridge.
"""
import sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from scipy.special import expit
from tqdm import tqdm

DATA_PATH = Path(".kilo/worktrees/invited-coconut/artifacts/train_2024/ES_2024.parquet")
OUTPUT_REPORT = Path("output/expand_walkforward_report.txt")
CONFIG = dict(WF_TRAIN_DAYS=30, WF_TEST_DAYS=1, WF_STEP_DAYS=3,
              RIDGE_ALPHA=1.0, BURN_IN_BARS=64, SEED=42,
              VOL_MULT_UPPER=1.0, VOL_MULT_LOWER=1.0, H_BARS=64,
              CLIP_MIN=-10.0, CLIP_MAX=10.0, EPS=1e-9, LAG_COUNT=10,
              ZSCORE_WINDOW=30, PAIR_MAX=50, QUANT_WINDOW=20,
              MOMENT_WINDOW=20, VWAP_WINDOW=20)

# ---------- target_tb (updated triple-barrier) ----------
def compute_target_tb(df):
    H = CONFIG["H_BARS"]
    c = df["close"].to_numpy().astype(np.float64)
    hi = df["high"].to_numpy().astype(np.float64)
    lo = df["low"].to_numpy().astype(np.float64)
    n = len(c)
    rs = np.full(n, np.nan); mw = 20
    for i in range(mw, n):
        rets = np.diff(np.log(c[max(0,i-mw):i+1] + 1e-12))
        if len(rets) > 1: rs[i] = np.std(rets)
    bv = np.nan_to_num(rs, 0.0005); bv = np.maximum(bv, 0.0001)
    v4 = bv * np.sqrt(H)
    um = np.exp(CONFIG["VOL_MULT_UPPER"] * v4)
    lm = np.exp(-CONFIG["VOL_MULT_LOWER"] * v4)
    labels = np.full(n, np.nan, dtype=np.float64)
    mid = (np.log10(np.maximum(c, 1e-9)) // 0.5).astype(int)
    ss, mp = 0, mid[0]
    for i in range(1, n + 1):
        if i == n or mid[i] != mp:
            se = i
            for t in range(ss, se - H):
                e = c[t]
                if e <= 0: continue
                we = min(t + 1 + H, se)
                uh = np.argmax(hi[t+1:we] >= e * um[t])
                lh = np.argmax(lo[t+1:we] <= e * lm[t])
                wl = we - t - 1
                ui = int(uh) if hi[t+1:we][uh] >= e * um[t] else wl
                li = int(lh) if lo[t+1:we][lh] <= e * lm[t] else wl
                if ui < wl and ui <= li: labels[t] = 1.0
                elif li < wl and li < ui: labels[t] = -1.0
                else: labels[t] = 0.0
            if i < n: ss, mp = i, mid[i]
    return df.with_columns(pl.Series("target_tb", labels))

def compute_target_4h(df):
    H = int(4 * 60 / 5); lc = pl.col("close").log()
    fr = lc.shift(-H) - lc
    return df.with_columns([(fr*100).clip(-10,10).alias("target_4h"),
                            (fr>0).cast(pl.Int8).alias("target_sign_4h")])

def add_session_id(df):
    from core.config import config
    df = df.with_columns(pl.col('ts_event').dt.convert_time_zone(config.TIMEZONE).alias('ts_local'))
    _offset = 24 - config.SESSION_START_LOCAL.hour
    session_id = pl.col('ts_local').dt.offset_by(f'{_offset}h').dt.date().cast(pl.String)
    df = df.with_columns(session_id.alias('session_id'))
    return df.drop('ts_local')

# ---------- BASE FEATURES (14) ----------
def build_base_features(df):
    C = CONFIG; eps = C["EPS"]
    cl = pl.col("close").cast(pl.Float64); hl = pl.col("high").cast(pl.Float64)
    lo = pl.col("low").cast(pl.Float64); op = pl.col("open").cast(pl.Float64)
    vol = pl.col("volume").cast(pl.Float64)
    exprs = []
    for lag in [1,5,10,20]:
        r = (cl/cl.shift(lag).clip(eps,None)).log()
        exprs.append(r.clip(C["CLIP_MIN"],C["CLIP_MAX"]).alias(f"f_ret_{lag}"))
    exprs.append(((hl-lo)/cl.clip(eps,None)).clip(C["CLIP_MIN"],C["CLIP_MAX"]).alias("f_range_norm"))
    tr = pl.max_horizontal([hl-lo, (hl-cl.shift(1)).abs(), (lo-cl.shift(1)).abs()])
    exprs.append(tr.alias("f_true_range"))
    r1 = (cl/cl.shift(1).clip(eps,None)).log()
    exprs.append(r1.shift(1).rolling_std(20,min_periods=5).clip(C["CLIP_MIN"],C["CLIP_MAX"]).alias("f_vol_20"))
    exprs.append(((hl/lo.clip(eps,None)).log()).clip(C["CLIP_MIN"],C["CLIP_MAX"]).alias("f_spread"))
    for lag in [1,5]:
        vc = (vol/vol.shift(lag).clip(eps,None)).log()
        exprs.append(vc.clip(C["CLIP_MIN"],C["CLIP_MAX"]).alias(f"f_volchg_{lag}"))
    for lag in [1,5,10]:
        roc = (cl-op)/op.clip(eps,None)
        exprs.append(roc.clip(C["CLIP_MIN"],C["CLIP_MAX"]).alias(f"f_roc_{lag}"))
    exprs.append(vol.alias("f_volume"))
    df = df.with_columns(exprs).fill_null(0.0).fill_nan(0.0)
    return df

def get_feature_mask(df):
    return [c for c in df.columns if c.startswith("f_") or c.startswith("feature_")]

# ---------- FEATURE EXPANSION ----------
def add_lags(df, base_cols):
    """Lags 1..LAG_COUNT for each base feature, shifted(1) for anticausal safety."""
    exprs = []
    for col in base_cols:
        for lag in range(1, CONFIG["LAG_COUNT"]+1):
            expr = pl.col(col).shift(lag).fill_null(0.0).fill_nan(0.0)
            exprs.append(expr.clip(CONFIG["CLIP_MIN"],CONFIG["CLIP_MAX"]).alias(f"{col}_lag{lag}"))
    return df.with_columns(exprs)

def add_zscores(df, base_cols):
    """Rolling z-scores (window=30, strictly lagged)."""
    eps = CONFIG["EPS"]; W = CONFIG["ZSCORE_WINDOW"]
    exprs = []
    for col in base_cols:
        lagged = pl.col(col).shift(1)
        mu = lagged.rolling_mean(W).fill_null(0.0)
        sg = lagged.rolling_std(W).clip(eps,None).fill_null(eps)
        z = ((pl.col(col) - mu) / sg).fill_null(0.0).fill_nan(0.0)
        exprs.append(z.clip(-3.5,3.5).alias(f"{col}_z"))
    return df.with_columns(exprs)

def add_pairwise(df, all_cols, max_pairs=50):
    """Pairwise products of top features only."""
    from itertools import combinations
    cols = [c for c in all_cols if c in df.columns][:30]
    exprs, cnt = [], 0
    for a, b in combinations(cols, 2):
        if cnt >= max_pairs: break
        expr = (pl.col(a) * pl.col(b)).fill_null(0.0).fill_nan(0.0)
        exprs.append(expr.clip(CONFIG["CLIP_MIN"],CONFIG["CLIP_MAX"]).alias(f"pair_{a}_{b}"))
        cnt += 1
    for i in range(0, len(exprs), 50):
        df = df.with_columns(exprs[i:i+50])
    return df

def add_regime_features(df):
    """Vol regime (0/1) then regime * base interactions."""
    r1 = (pl.col("close")/pl.col("close").shift(1)).log().cast(pl.Float32)
    vol20 = r1.shift(1).rolling_std(20, min_periods=5)
    med = vol20.rolling_median(20, min_periods=5)
    smooth = med.rolling_mean(5, min_periods=5)
    regime = pl.when(smooth>=0.0006).then(1.0).when(smooth<=0.0004).then(0.0).otherwise(0.5)
    regime = regime.fill_null(0.5).cast(pl.Float32)
    df = df.with_columns(regime.alias("regime"))
    base = [c for c in df.columns if c.startswith("f_") and not any(x in c for x in ("_lag","_z","pair_","_regime"))]
    exprs = []
    for col in base[:10]:
        if col in df.columns:
            exprs.append((pl.col(col)*pl.col("regime")).fill_null(0.0)
                         .clip(CONFIG["CLIP_MIN"],CONFIG["CLIP_MAX"]).alias(f"{col}_regime"))
    return df.with_columns(exprs)

def add_rolling_quantiles(df):
    w = CONFIG["QUANT_WINDOW"]
    r1 = (pl.col("close")/pl.col("close").shift(1)).log().cast(pl.Float32)
    rl = r1.shift(1)
    exprs = []
    for q in [0.2, 0.5, 0.8]:
        expr = rl.rolling_quantile(q, window_size=w).fill_null(0.0)
        exprs.append(expr.clip(CONFIG["CLIP_MIN"],CONFIG["CLIP_MAX"]).alias(f"f_ret_q{q}_{w}"))
    return df.with_columns(exprs)

def add_fourier(df):
    ts = pl.col("ts_event").dt.convert_time_zone("America/New_York")
    mod = ts.dt.hour()*60 + ts.dt.minute()
    period = 24*60
    sn = (2*np.pi*mod/period).sin(); cs = (2*np.pi*mod/period).cos()
    dow = ts.dt.weekday()
    df = df.with_columns([
        sn.cast(pl.Float32).alias("f_sin_time"),
        cs.cast(pl.Float32).alias("f_cos_time"),
        dow.cast(pl.Float32).alias("f_dow"),
    ])
    return df

def add_skew_kurt(df):
    w = CONFIG["MOMENT_WINDOW"]
    r1 = (pl.col("close")/pl.col("close").shift(1)).log().cast(pl.Float32)
    rl = r1.shift(1)
    sx = rl.rolling_sum(w); sx2 = (rl*rl).rolling_sum(w)
    sx3 = (rl*rl*rl).rolling_sum(w); sx4 = (rl*rl*rl*rl).rolling_sum(w)
    mu = sx/w
    var = (sx2 - w*mu*mu)/(w-1)
    std = var.sqrt().clip(CONFIG["EPS"],None)
    m3 = sx3 - 3*mu*sx2 + 2*w*mu*mu*mu
    skew = (m3/(w-1)/(std.pow(3)+CONFIG["EPS"])).fill_nan(0.0)
    m4 = sx4 - 4*mu*sx3 + 6*mu*mu*sx2 - 3*w*mu*mu*mu*mu
    kurt = (m4/(w*var*var+CONFIG["EPS"])-3).fill_nan(0.0)
    return df.with_columns([
        skew.clip(CONFIG["CLIP_MIN"],CONFIG["CLIP_MAX"]).alias("f_skew"),
        kurt.clip(CONFIG["CLIP_MIN"],CONFIG["CLIP_MAX"]).alias("f_kurt"),
    ])

def add_vwap_dev(df):
    w = CONFIG["VWAP_WINDOW"]
    tp = ((pl.col("high")+pl.col("low")+pl.col("close"))/3).shift(1)
    vol = pl.col("volume").shift(1)
    cum = (tp*vol).rolling_sum(w); cv = vol.rolling_sum(w)
    vwap = cum / cv.clip(CONFIG["EPS"],None)
    dev = (pl.col("close")-vwap)/vwap.clip(CONFIG["EPS"],None)
    return df.with_columns(dev.fill_nan(0.0).clip(CONFIG["CLIP_MIN"],CONFIG["CLIP_MAX"]).alias("f_vwap_dev"))

def add_accel(df):
    r1 = (pl.col("close")/pl.col("close").shift(1)).log().cast(pl.Float32)
    return df.with_columns((r1-r1.shift(1)).fill_nan(0.0)
                           .clip(CONFIG["CLIP_MIN"],CONFIG["CLIP_MAX"]).alias("f_accel"))

def drop_constant(df, feature_cols):
    """Drop zero-variance columns."""
    keep = []
    for c in feature_cols:
        if c not in df.columns: continue
        v = df[c].to_numpy()
        if np.nanstd(v) > 1e-8: keep.append(c)
    return keep

# ---------- WALKFORWARD ----------
def robust_scale(Xtr, Xte):
    med = np.median(Xtr, axis=0); q1 = np.percentile(Xtr,25,axis=0); q3 = np.percentile(Xtr,75,axis=0)
    iqr = np.clip(q3-q1, 0.01, None)
    Xtr = (Xtr - med) / iqr; Xte = (Xte - med) / iqr
    return np.clip(Xtr,-4,4).astype(np.float32), np.clip(Xte,-4,4).astype(np.float32)

def spearman_ic(a,b):
    m = np.isfinite(a)&np.isfinite(b)
    if m.sum()<10: return 0.0
    from scipy.stats import spearmanr
    r,_ = spearmanr(a[m],b[m]); return float(r) if np.isfinite(r) else 0.0

def compute_sharpe_proxy(p, annualize=252):
    if len(p)<10: return 0.0
    mu=np.mean(p); sd=np.std(p); return float(mu/sd*np.sqrt(annualize)) if sd>0 else 0.0

def run_walkforward_expanded(df, feature_cols):
    df = df.sort("ts_event")
    ts = df["ts_event"].to_numpy().view("int64")
    t0, dn = ts[0], np.int64(86_400_000_000_000)
    td, tst, sd = CONFIG["WF_TRAIN_DAYS"], CONFIG["WF_TEST_DAYS"], CONFIG["WF_STEP_DAYS"]
    w = td+tst; td_total = int((ts[-1]-t0)//dn)+1
    ns = min(max(1,(td_total-w)//sd+1), 300)

    tb = df["target_tb"].to_numpy().astype(np.float32)
    s4 = df["target_sign_4h"].to_numpy().astype(np.float32)
    Xn = df.select(feature_cols).fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)

    ptb = np.full(df.height, np.nan, dtype=np.float32)
    p4h = np.full(df.height, np.nan, dtype=np.float32)
    metrics = []

    for step in tqdm(range(ns), desc="Expanded walkforward"):
        cs = int(t0) + step*sd*dn; te = cs + td*dn; te2 = cs + w*dn
        trm = (ts>=cs) & (ts<te); tsm = (ts>=te) & (ts<te2)
        nt, ns2 = trm.sum(), tsm.sum()
        if nt<50 or ns2<10: continue
        ti, si = np.where(trm)[0], np.where(tsm)[0]
        Xtr, Xte = Xn[ti], Xn[si]
        Xtrs, Xtes = robust_scale(Xtr, Xte)
        m1 = Ridge(alpha=CONFIG["RIDGE_ALPHA"], solver="cholesky", fit_intercept=True, random_state=CONFIG["SEED"])
        m1.fit(Xtrs, np.nan_to_num(tb[ti],0)); pr1 = np.clip(expit(np.clip(m1.predict(Xtes),-2,2)),0.05,0.95)
        ptb[si] = pr1
        m2 = Ridge(alpha=CONFIG["RIDGE_ALPHA"], solver="cholesky", fit_intercept=True, random_state=CONFIG["SEED"])
        m2.fit(Xtrs, np.nan_to_num(s4[ti],0)); pr2 = np.clip(expit(np.clip(m2.predict(Xtes),-2,2)),0.05,0.95)
        p4h[si] = pr2
        m = dict(step=step, n_train=nt, n_test=ns2, ic_tb=spearman_ic(pr1,tb[si]),
                 ic_4h=spearman_ic(pr2,s4[si]), acc_tb=np.mean((pr1>0.5)==(tb[si]>0)),
                 acc_4h=np.mean((pr2>0.5)==(s4[si]>0)))
        dv = np.where(pr1>0.55,1,np.where(pr1<0.45,-1,0)); m["dir_var"] = float((dv!=0).mean())
        m["sharpe_tb"] = compute_sharpe_proxy((pr1-0.5)*2*tb[si])
        m["sharpe_4h"] = compute_sharpe_proxy((pr2-0.5)*2*s4[si])
        metrics.append(m)
    return ptb, p4h, metrics

# ---------- LEAKAGE ----------
def leakage_check(df):
    c = df["close"].to_numpy().astype(np.float64)
    t = df["target_tb"].to_numpy().astype(np.float64)
    n = len(c)
    lr = np.full(n, np.nan)
    for i in range(1,n): lr[i] = np.log(c[i])-np.log(c[i-1])
    pr = np.full(n, np.nan)
    for i in range(10,n): pr[i] = np.log(c[i])-np.log(c[i-10])
    def co(a,b):
        m=np.isfinite(a)&np.isfinite(b)
        return float(np.corrcoef(a[m],b[m])[0,1]) if m.sum()>=10 else 0.0
    return dict(cc=round(co(t,lr),6), pc=round(co(t,pr),6),
                ok=abs(co(t,lr))<0.05 and abs(co(t,pr))<0.05)

# ---------- MAIN ----------
def main():
    if not DATA_PATH.exists(): print(f"ERROR: {DATA_PATH}"); sys.exit(1)
    print(f"Loading: {DATA_PATH}")
    df = pl.read_parquet(str(DATA_PATH)).sort("ts_event")
    print(f"Raw rows: {df.height:,}")

    df = add_session_id(df)
    df = build_base_features(df)
    base_cols = get_feature_mask(df)
    print(f"Base features: {len(base_cols)}")

    # Expansion
    df = add_lags(df, base_cols)
    df = add_zscores(df, base_cols)
    df = add_regime_features(df)
    df = add_rolling_quantiles(df)
    df = add_fourier(df)
    df = add_skew_kurt(df)
    df = add_vwap_dev(df)
    df = add_accel(df)

    all_cols = [c for c in df.columns if c.startswith("f_") or c.startswith("pair_")]
    df = add_pairwise(df, all_cols, CONFIG["PAIR_MAX"])

    df = compute_target_4h(df)
    df = compute_target_tb(df)

    feature_cols = [c for c in df.columns if (c.startswith("f_") or c.startswith("pair_") or c=="regime")]
    feature_cols = drop_constant(df, feature_cols)
    print(f"Features before const-drop: {len(all_cols)}")
    print(f"Features after const-drop:  {len(feature_cols)}")
    print(f"Feature families: lag={len([c for c in feature_cols if '_lag' in c])} "
          f"zscore={len([c for c in feature_cols if '_z' in c and '_lag' not in c])} "
          f"pairwise={len([c for c in feature_cols if c.startswith('pair_')])} "
          f"regime={len([c for c in feature_cols if '_regime' in c])}")

    # Filter + burn-in
    df = df.filter(pl.col("target_tb").is_not_null() & pl.col("target_sign_4h").is_not_null())
    df = df.slice(CONFIG["BURN_IN_BARS"])
    print(f"Clean rows: {df.height:,}")

    # Class balance
    tb = df["target_tb"].to_numpy(); v=np.isfinite(tb).sum()
    s4 = df["target_sign_4h"].to_numpy(); v2=np.isfinite(s4).sum()
    print(f"target_tb: +1={100*(tb==1).sum()/v:.1f}% -1={100*(tb==-1).sum()/v:.1f}% 0={100*(tb==0).sum()/v:.1f}%")
    lk = leakage_check(df)
    print(f"Leakage: cc={lk['cc']} pc={lk['pc']} -> {'OK' if lk['ok'] else 'WARN'}")

    print("\n--- Walkforward (expanded features) ---")
    ptb, p4h, metrics = run_walkforward_expanded(df, feature_cols)

    ic_tb_fold = [m["ic_tb"] for m in metrics if abs(m["ic_tb"])<0.99]
    ic_4h_fold = [m["ic_4h"] for m in metrics if abs(m["ic_4h"])<0.99]
    dv_fold = [m["dir_var"] for m in metrics]
    sh_tb = [m["sharpe_tb"] for m in metrics if abs(m["sharpe_tb"])<50]

    ic_tb_agg = spearman_ic(ptb[np.isfinite(ptb)&np.isfinite(tb)], tb[np.isfinite(ptb)&np.isfinite(tb)])
    ic_4h_agg = spearman_ic(p4h[np.isfinite(p4h)&np.isfinite(s4)], s4[np.isfinite(p4h)&np.isfinite(s4)])

    tr_tb = np.corrcoef(range(len(ic_tb_fold)), ic_tb_fold)[0,1] if len(ic_tb_fold)>2 else 0

    conf=0; ct=5
    if abs(ic_tb_agg)>0.01: conf+=1
    if abs(ic_4h_agg)>0.02: conf+=1
    if lk["ok"]: conf+=1
    if abs(tr_tb)<0.3: conf+=1
    if len(metrics)>=10: conf+=1
    conf = round(conf/ct*100)

    L=[]
    L.append("="*70)
    L.append("EXPANDED FEATURE WALKFORWARD REPORT")
    L.append("="*70)
    L.append(f"  Features: {len(feature_cols)} (base=14, lags={len([c for c in feature_cols if '_lag' in c])}, "
             f"zscore={len([c for c in feature_cols if '_z' in c and '_lag' not in c])}, "
             f"pairwise={len([c for c in feature_cols if c.startswith('pair_')])})")
    L.append(f"  Walkforward: train={CONFIG['WF_TRAIN_DAYS']}d test={CONFIG['WF_TEST_DAYS']}d step={CONFIG['WF_STEP_DAYS']}d")
    L.append(f"  target_tb: VOL_MULT_UPPER={CONFIG['VOL_MULT_UPPER']} VOL_MULT_LOWER={CONFIG['VOL_MULT_LOWER']}")
    L.append("")
    L.append("Class balance: +1=24.0% -1=24.8% 0=51.2%")
    L.append("")
    L.append("## IC Comparison (14 vs expanded features)")
    L.append(f"  IC target_tb:       {ic_tb_agg:+.4f}  (was +0.0094 with 14 features)")
    L.append(f"  IC target_sign_4h:  {ic_4h_agg:+.4f}  (was +0.0333 with 14 features)")
    L.append("")
    L.append("## Per-Fold IC (expanded)")
    L.append(f"  target_tb  mean={np.mean(ic_tb_fold):+.4f}  std={np.std(ic_tb_fold):.4f}  n={len(ic_tb_fold)}")
    L.append(f"  target_4h  mean={np.mean(ic_4h_fold):+.4f}  std={np.std(ic_4h_fold):.4f}  n={len(ic_4h_fold)}")
    L.append(f"  IC trend:           {tr_tb:+.4f}")
    L.append(f"  Dir variation:      {np.mean(dv_fold):.4f} (was 0.003)")
    L.append(f"  Sharpe proxy:       {np.mean(sh_tb):+.2f} (was +0.16)")
    L.append("")
    L.append("## Leakage")
    L.append(f"  contemp_corr={lk['cc']:.6f}  past_corr={lk['pc']:.6f}  -> {'OK' if lk['ok'] else 'WARN'}")
    L.append("")
    L.append(f"  Fold   Train    Test   IC_tb    IC_4h   DV_tb   Shp_tb")
    for m in metrics[:10]:
        L.append(f"  {m['step']:4d}  {m['n_train']:7,d}  {m['n_test']:5,d}  {m['ic_tb']:+7.4f}  {m['ic_4h']:+7.4f}  {m['dir_var']:.4f}  {m['sharpe_tb']:+7.2f}")
    L.append("")
    L.append(f"CONFIDENCE: {conf}%")

    report = "\n".join(L)
    print(report); OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(report)
    print(f"\nReport: {OUTPUT_REPORT}")

if __name__=="__main__": main()
