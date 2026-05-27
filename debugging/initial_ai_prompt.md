# ⚙️ Quant Flowchart — Three‑Stream HTF‑Aware Pipeline (Consolidated)
Deterministic CPU‑only intraday futures pipeline: 1‑min → three streams (5min, 1h, Daily) → mixed‑timeframe features → HTF context → ExtraTrees discovery → frozen features → walkforward Ridge → top‑down execution.

---

## SECTION 1: CORE OBJECTIVE

Strict intraday Globex 23/5 18:00 America/New_York → 16:00 America/New_York, with the CME settlement gap excluded from 17:00–18:00 ET, no overnight holds. Zero leakage, memory <16 GB (safe cap 14 GB), seed 42, float32 as primary compute dtype. Polars (no pandas), pytz, chunked processing.

**MVP Scope — Minimal Viable Product (single-market)**
- Goal: find a repeatable alpha for next 4-hour direction using 1-minute history.
- Limit: one market (pick CL/ES/ZB). Run locally, deterministic, single-threaded.
- I/O schema & entrypoint:
  - Implement `run_pipeline(data_path, --prod-mode)` entrypoint.
  - Minimal schema: input parquet (1-min UTC ts_event, open/high/low/close, volume).
  - Optional flags: `--prod-mode` to enable canonical writer and full manifest.
- Small baseline:
  - Select 20 core features (from existing 40) computed per session.
  - Clip, cast to `float32`, persist baseline matrix to `artifacts/baseline_feature_matrix.parquet` for reproducibility.
- Discovery (fast loop):
  - Single ExtraTrees run, 5–10 bootstrap folds in-process.
  - Compute selection frequency and mean importance; freeze top K (e.g., K=20).
- Stability checks:
  - Sign-consistency check across folds, correlation filter on frozen set. Reduce K if unstable.
- Modeling:
  - Train `Ridge` on 5-min resampled data using frozen features.
  - Deterministic scaler per fold, walkforward slices evaluate Sharpe, hit rate, turnover.
- Execution simulation (minimal):
  - Simple slippage + commission model, require flat N minutes before session close.
- Iterate:
  - If signal passes walkforward and checks, scale discovery folds, enable pairwise interactions, subprocess isolation, and `--prod-mode` serialization.

**Pipeline Flowchart:**
1. **Clean Data** — Collect raw data, normalize timezone, build sessions, resample to bars (5m + HTF), ensure no future leakage (causal alignment).
2. **Build HTF Context** — Compute HTF indicators (ATR, trends, slopes, etc.), select useful HTF features, freeze HTF features (no peeking forward).
3. **Build 5m Features** — Create baseline 5m features (returns, volatility, microstructure), join frozen HTF features → conditions the 5m model.
4. **Expand + Select Features** — Use ExtraTrees to generate feature importance, expand interactions/nonlinear signals, prune weak features → keep only signal.
5. **Train Model** — Ridge Regression with walkforward training: train on past, test on next slice (OOS), repeat across time, use per-fold scaling (no leakage).
6. **Backtest** — Turn predictions → trades, simulate position sizing, slippage + latency, output PnL, Sharpe/drawdown, stability across folds.
7. **Validate** — Reproducibility checks, no-leakage tests, CI-style sanity checks.
8. **Deploy + Monitor** — Run live with same pipeline, track performance drift, gate updates via CI.

---

## SECTION 2: IMPLEMENTATION STATUS

### 2.1 Global Environment & Determinism

| Item | Status | Location |
|---|---|---|
| SEED=42 for numpy/random/sklearn | ✅ Implemented | `quant/config.py` L67, `quant/cli.py` L18-19 |
| OMP/OPENBLAS/MKL/POLARS single-threaded | ✅ Configured | `quant/config.py` (env vars set at runtime) |
| Libs: polars, numpy, sklearn, pyarrow, joblib, pytz | ✅ In use | `requirements.txt`, all modules |
| CLIP_MIN=-10.0, CLIP_MAX=10.0, EPS=1e-9 | ✅ Implemented | `quant/config.py` L68-70 |
| NaN/inf → 0.0 replacement | ✅ Implemented | `quant/walkforward.py` L18-19, `quant/features/expansion.py` L205 |
| No pandas dependency | ✅ Compliant | Zero pandas imports across all source files |

### 2.2 Data Load — Three‑Stream Resampling

| Item | Status | Location |
|---|---|---|
| Lazy scan_parquet → session filter | ✅ Implemented | `quant/session.py` L56-69 (`process_one_file`) |
| Resample to 5m, 1h, 1d simultaneously | ✅ Implemented | `quant/session.py` L72-102 (`process_frequency`, `load_all_streams_chunked`) |
| Session definition (18:00–16:00 ET, 17:00–18:00 gap excluded) | ✅ Implemented | `quant/session.py` L25-31 (`filter_session_hours`) |
| Session ID via +6h shift → date | ✅ Implemented | `quant/session.py` L19-23 (`add_session_id`) |
| Drop incomplete bars (config.DROP_INCOMPLETE_ROWS) | ✅ Implemented | `quant/session.py` L37-42 |
| OHLC integrity validation | ✅ Implemented | `quant/ingest.py` L21-26 (`validate_memory_and_integrity`) |
| Memory abort check (>RAM_CAP_BYTES) | ✅ Implemented | `quant/ingest.py` L30-31 |
| Chunked processing safety margins | ✅ Implemented | `quant/ingest.py` L32-35 |
| Cleaned data cache (aligned_data_*.parquet) | ✅ Implemented | `quant/ingest.py` L55-59, L79-82 |

### 2.3 HTF Alignment (Causal Join)

| Item | Status | Location |
|---|---|---|
| 1h bars: timestamp shifted +1h for backward join | ✅ Implemented | `quant/align.py` L28-37 |
| Daily bars: timestamp shifted +1d for backward join | ✅ Implemented | `quant/align.py` L45-56 |
| Forward/backward fill for daily boundary bars | ✅ Implemented | `quant/align.py` L61-65 |
| join_asof(strategy='backward') — strictly no lookahead | ✅ Implemented | `quant/align.py` L37, L56 |

### 2.4 Base Features & HTF Context

| Item | Status | Location |
|---|---|---|
| 5-min baseline: 40 features from YAML | ✅ Implemented | `quant/features/baseline.py` |
| HTF context features (distance to daily high/low, daily return, daily trend slope, daily vol) | ✅ Implemented | `quant/features/htf_context.py` L11-102 |
| Hourly trend alignment (1h return × daily trend sign) | ✅ Implemented | `quant/features/htf_context.py` L74-80 |
| Volatility ratio (1h vol / daily vol) | ✅ Implemented | `quant/features/htf_context.py` L82-89 |
| All HTF features clipped + cast to float32 | ✅ Implemented | `quant/features/htf_context.py` L92-96 |

### 2.5 Feature Expansion

| Item | Status | Location |
|---|---|---|
| Intra-timeframe pairwise products (capped 500) | ✅ Implemented | `quant/features/expansion.py` L63-87 |
| Cross-timeframe interactions (5min × HTF, capped 200) | ✅ Implemented | `quant/features/expansion.py` L90-116 |
| Ratios and z-scores | ✅ Implemented | `quant/features/expansion.py` L23-36 |
| Regime-conditioned transforms | ✅ Implemented | `quant/features/expansion.py` L39-60 |
| Rolling quantiles, Fourier features, moments (skew/kurt), acceleration | ✅ Implemented | `quant/features/expansion.py` L119-180 |
| VWAP deviation | ✅ Implemented | `quant/features/expansion.py` L169-180 |
| Feature engine orchestrates all expansions | ✅ Implemented | `quant/features/engine.py` L13-33 |

### 2.6 Target Construction

| Item | Status | Location |
|---|---|---|
| Target 5m (open[t+1] → close[t+1] execution-aligned) | ✅ Implemented | `quant/features/target.py` L4-18 |
| Target 1h (from aligned 1h bars) | ✅ Implemented | `quant/features/target.py` L30-36 |
| Target 4h (log-return over 48 bars) | ✅ Implemented | `quant/features/target.py` L38-42 |
| target_sign (binary direction) for all horizons | ✅ Implemented | `quant/features/target.py` L16, L35, L42 |
| Drop incomplete targets (bars without future for any horizon) | ✅ Implemented | `quant/features/target.py` L20-28 |

### 2.7 Discovery (ExtraTrees)

| Item | Status | Location |
|---|---|---|
| ExtraTreesRegressor with deterministic seed per fold | ✅ Implemented | `quant/discovery.py` L17-19, L62-65 |
| Bootstrap folds (BOOTSTRAP_FOLDS=30) | ✅ Implemented | `quant/discovery.py` L50-75 |
| Selection frequency threshold (0.75) | ✅ Implemented | `quant/discovery.py` L95 |
| Sign consistency threshold (0.8) | ✅ Implemented | `quant/discovery.py` L95 |
| Cumulative importance threshold (0.95) | ✅ Implemented | `quant/discovery.py` L97-104 |
| Fallback to min features (MIN_SELECTED_FEATURES=10) | ✅ Implemented | `quant/discovery.py` L105-124 |
| Manifest JSON with feature names, hashes, stability stats | ✅ Implemented | `quant/discovery.py` L128-131 |
| RSS memory abort per fold | ✅ Implemented | `quant/discovery.py` L55-56 |
| Feature list includes HTF features when present | ✅ Implemented | `quant/discovery.py` L128 (`htf_features_included` flag) |

### 2.8 Walkforward Modeling

| Item | Status | Location |
|---|---|---|
| Ridge regression (config.RIDGE_PARAMS) | ✅ Implemented | `quant/walkforward.py` L50-56 |
| RandomForestClassifier as alternative MODEL_TYPE | ✅ Implemented | `quant/walkforward.py` L58-61 |
| Robust IQR-based per-fold scaling (no leakage) | ✅ Implemented | `quant/walkforward.py` L21-32 |
| Correlation pruning on first train fold (threshold 0.95) | ✅ Implemented | `quant/walkforward.py` L118, `quant/features/corr_prune.py` |
| Variance filter (remove constant features) | ✅ Implemented | `quant/walkforward.py` L39, `quant/features/variance_filter.py` |
| Walkforward 60/1 day rolling with configurable step | ✅ Implemented | `quant/walkforward.py` L109-146 |
| Sigmoid probability conversion (expit) | ✅ Implemented | `quant/walkforward.py` L57 |
| EMA probability smoothing across sessions | ✅ Implemented | `quant/walkforward.py` L67-83 |
| Target stabilization (clip y to [-1, 1]) | ✅ Implemented | `quant/walkforward.py` L34-36 |

### 2.9 Execution Simulation

| Item | Status | Location |
|---|---|---|
| Signal generation from prediction probability (>0.6 long, <0.4 short) | ✅ Implemented | `quant/execution/simulator.py` L31-38 |
| HTF directional bias gating (suppress signals against HTF trend) | ✅ Implemented | `quant/execution/simulator.py` L52-68 |
| Session break filter (flatten during 17:00–18:00 ET) | ✅ Implemented | `quant/execution/simulator.py` L73-87 |
| Flat before close (5 min before 16:00 ET) | ✅ Implemented | `quant/execution/simulator.py` L92-107 |
| HTF volatility scaling (1 / daily_vol × leverage cap) | ✅ Implemented | `quant/execution/simulator.py` L115-133 |
| Fixed contract size multiplier | ✅ Implemented | `quant/execution/simulator.py` L138-142 |
| HTF trend alignment filter (suppress counter-trend trades) | ✅ Implemented | `quant/execution/simulator.py` L150-166 |
| Slippage + commission + vol penalty cost model | ✅ Implemented | `quant/execution/simulator.py` L188-193 |
| Execution at open[t+1], exit at close[t+1] | ✅ Implemented | `quant/execution/simulator.py` L202-206 |
| Position change tracking (turnover calculation) | ✅ Implemented | `quant/execution/simulator.py` L212-226 |
| Liquidity filter (high-low spread proxy) | ✅ Implemented | `quant/execution/simulator.py` L178-181 |

### 2.10 Analytics & Benchmark

| Item | Status | Location |
|---|---|---|
| Sharpe, Sortino, Calmar ratios | ✅ Implemented | `quant/analytics/aggregate.py` L33-54 |
| Max drawdown, win rate, profit factor, avg win/loss | ✅ Implemented | `quant/analytics/aggregate.py` L48-105 |
| Turnover, avg holding bars, trade count | ✅ Implemented | `quant/analytics/aggregate.py` L65-80 |
| Benchmark (20-period SMA crossover, long only) | ✅ Implemented | `quant/walkforward.py` L85-98 |
| Benchmark comparison (Sharpe, max DD, correlation) | ✅ Implemented | `quant/analytics/aggregate.py` L108-129 |
| Year-by-year breakdown | ✅ Implemented | `quant/analytics/aggregate.py` L177-187 |
| Multi-market aggregation | ✅ Implemented | `quant/analytics/aggregate.py` L219-236 |

### 2.11 CLI & Manifest

| Item | Status | Location |
|---|---|---|
| `discover` subcommand (Phase 1: feature discovery) | ✅ Implemented | `quant/cli.py` L62-77 |
| `run` subcommand (Phase 2: walkforward + execution) | ✅ Implemented | `quant/cli.py` L78-113 |
| `aggregate` subcommand (cross-market reporting) | ✅ Implemented | `quant/cli.py` L114-116 |
| Manifest-based feature pruning | ✅ Implemented | `quant/cli.py` L30-38 |
| Data caching (aligned + feature matrix parquet) | ✅ Implemented | `quant/cli.py` L67-74, L83-92 |
| Memory safety check on startup | ✅ Implemented | `quant/cli.py` L22-28 |
| Market config detection + loading | ✅ Implemented | `quant/cli.py` L59-61, `quant/market_config.py` |

### 2.12 Tests

| Item | Status | Location |
|---|---|---|
| Memory abort tests | ✅ Implemented | `tests/` |
| Serialization reproducibility | ✅ Implemented | `tests/` |
| Walkforward integrity | ✅ Implemented | `tests/` |
| Dtype validation | ✅ Implemented | `tests/` |
| Causal audit (no lookahead) | ✅ Implemented | `tests/test_causal_audit.py` |
| HTF feature tests | ✅ Implemented | `tests/test_htf_features.py` |
| Session streaming tests | ✅ Implemented | `tests/test_session_streaming.py` |
| Alignment tests | ✅ Implemented | `tests/test_alignment.py` |
| Leakage audit tool | ✅ Implemented | `tools/leakage_audit.py` |

---

## SECTION 3: KNOWN DEVIATIONS & TRADEOFFS

### 3.1 ExtraTrees max_depth=8 (Not Spec's 12)

| | Spec Value | Actual Value | Rationale |
|---|---|---|---|
| `max_depth` | 12 | 8 | Reduced tree depth prevents overfitting on noisy intraday data and keeps memory footprint lower per fold. Shallower trees generalize better across 30 bootstrap folds. |

**Config reference:** `quant/config.py` L41 (`EXTRA_TREES_PARAMS`)

### 3.2 float64 in Metrics Computation

| Issue | Detail |
|---|---|
| Location | `quant/analytics/aggregate.py` L13-148 (`compute_pro_metrics`) |
| Nature | PnL arrays are cast to `np.float32` (L18), but metric values returned to the caller are Python native `float` (float64). JSON serialization uses `round()` which preserves double precision. |
| Impact | Negligible — these are final scalar statistics, not part of the feature pipeline or model inference. Memory impact is trivial. |
| Resolution | Low priority. Could enforce float32 return types if strict compliance is required. |

### 3.3 Ridge alpha Override

| Issue | Detail |
|---|---|
| Spec value | `alpha=1.0` in `quant/config.py` L50 (`RIDGE_PARAMS`) |
| Potential override | Planned `alpha=50.0` for increased regularization in noisy regimes |
| Status | Config holds 1.0; walkforward applies `ridge_params.get('alpha', 1.0)` (L53). Override not yet active in code — config change or runtime flag needed. |

### 3.4 Classification-Based Execution Signal

| Issue | Detail |
|---|---|
| Spec description | Ridge regression → continuous predictions → position sizing |
| Actual behavior | Ridge outputs are converted to probabilities via `expit()` (sigmoid), clamped to [0.3, 0.7], then discretized: prob > 0.6 → long, prob < 0.4 → short, otherwise flat. |
| Rationale | Binary classification framework proved more stable in walkforward than raw regression outputs. The sigmoid squashing and probability clamping prevent extreme position allocations. |
| Code reference | `quant/walkforward.py` L55-65, `quant/execution/simulator.py` L31-38 |

### 3.5 RandomForestClassifier as Alternative Model

| Issue | Detail |
|---|---|
| Spec | Only Ridge regression |
| Actual | `config.MODEL_TYPE` supports `'Ridge'` (default) and `'RandomForestClassifier'` |
| Rationale | RF provides a non-linear baseline for comparison. Both models use the same frozen feature set. |
| Code reference | `quant/walkforward.py` L50-63 |

### 3.6 Target for Walkforward

| Issue | Detail |
|---|---|
| Spec | Target is `target_5m` (short-horizon) |
| Actual | `target_sign_4h` is the default target column in CLI `run` command (L80) |
| Rationale | The 4-hour horizon aligns with the MVP goal of "next 4-hour direction." Multiple targets (5m, 1h, 4h) are computed; the 4h sign is selected for the primary backtest. |

---

## SECTION 4: REMAINING ROADMAP

### Priority 1 — High Impact (Core Completeness)

| # | Task | Detail | Target Files |
|---|---|---|---|
| 1 | `--prod-mode` flag | Full canonical writer pipeline, manifest versioning, run metadata logging | `quant/cli.py`, `quant/io/` |
| 2 | Subprocess isolation for discovery folds | `joblib` or `multiprocessing` with memory-per-worker caps; currently in-process only | `quant/discovery.py` |
| 3 | Overnight hold restriction enforcement | Session-level position tracking to ensure flat at each session close (already flattening intra-session, but no cross-session position carry check) | `quant/execution/simulator.py` |

### Priority 2 — Medium Impact (Robustness & Scale)

| # | Task | Detail | Target Files |
|---|---|---|---|
| 4 | Multi-market orchestration | Run pipeline across ES/CL/ZB in sequence or parallel batches, aggregate results. Market configs already loaded (`quant/market_config.py`). | `quant/cli.py`, new orchestration module |
| 5 | CI-style reproducibility checks | Assert deterministic output across runs: same seed + same data → identical manifest, identical PnL series hash. | New `tests/test_reproducibility.py`, CI config |
| 6 | Leakage audit hardening | Extend `tools/leakage_audit.py` to validate HTF join logic, regime label leakage, and target construction across fold boundaries | `tools/leakage_audit.py` |
| 7 | Full float32 enforcement on metrics output | Cast `compute_pro_metrics` return values to float32 before JSON serialization | `quant/analytics/aggregate.py` |

### Priority 3 — Lower Impact (Live & Monitoring)

| # | Task | Detail | Target Files |
|---|---|---|---|
| 8 | Live pipeline execution mode | Same `run` entrypoint operating on streaming data instead of historical parquet files | `quant/execution/live.py` (new) |
| 9 | Performance drift monitoring | Compare live signal distribution vs. walkforward OOS distribution; alert on KL divergence threshold breach | New module |
| 10 | Model update gating via CI | Automated re-discovery and walkforward on schedule; gate deployment on Sharpe/drawdown thresholds | CI pipeline config |

---

## APPENDIX: Configuration Reference

<details>
<summary>Global Constants (quant/config.py)</summary>

```
SEED = 42
TIMEZONE = 'America/New_York'
SESSION_START_LOCAL = time(18,0)
SESSION_END_LOCAL = time(16,0)
SESSION_BREAK_START_LOCAL = time(17,0)
SESSION_BREAK_END_LOCAL = time(18,0)
RESAMPLE_FREQUENCIES = ['5m', '1h', '1d']
DROP_INCOMPLETE_ROWS = True

RAM_CAP_BYTES = 14 * 1024**3
RSS_STOP_BYTES = 13.5 * 1024**3
ROWS_PER_CHUNK_MAX = 5_000_000
MEMORY_SAFETY_MARGIN = 0.95

CLIP_MIN = -10.0
CLIP_MAX = 10.0
EPS = 1e-9
REPLACE_INF_NAN_WITH = 0.0
ROW_GROUP_SIZE = 65536

# Window parameters
ROLL_WINDOWS = [5,10,20,50]
ROLL_WINDOWS_1H = [2,4,6,12]
ROLL_WINDOWS_DAILY = [5,10,20]

# Feature expansion
FEATURE_TRANSFORMS = ['lags','ratios','z_scores','pairwise_products_limited','cross_timeframe_ratios']
MAX_PAIRWISE_INTERACTIONS = 500
MAX_CROSS_TIMEFRAME_INTERACTIONS = 200

# HTF Context
HTF_TREND_WINDOWS = [5,10,20]
HTF_VOLATILITY_WINDOWS = [5,10,20]
HTF_ALIGNMENT_FILTER = True
HTF_TREND_THRESHOLD = 0.1

# Regime
VOL_MEDIAN_WINDOW = 20
VOL_SMOOTH_WINDOW = 5
REGIME_HIGH_THRESH = 0.6
REGIME_LOW_THRESH = 0.4

# Target & Discovery
TARGET_5M_HORIZON = 1
TARGET_SCALE_FACTOR = 100.0
DISCOVERY_WINDOW_DAYS = 60
BOOTSTRAP_FOLDS = 30
EXTRA_TREES_PARAMS = {'random_state':42,'n_jobs':1,'n_estimators':100,'max_depth':8,'max_features':0.3,'bootstrap':False}
SELECTION_FREQ_THRESHOLD = 0.75
SIGN_CONSISTENCY_THRESHOLD = 0.8
CUMULATIVE_IMPORTANCE_THRESHOLD = 0.95
MIN_SELECTED_FEATURES = 10
MAX_SELECTED_FEATURES = 1000

# Walkforward & Ridge
WF_TRAIN_DAYS = 60
WF_TEST_DAYS = 1
WF_STEP_DAYS = 1
RIDGE_PARAMS = {'alpha':1.0,'solver':'cholesky','fit_intercept':True,'random_state':42}
MODEL_TYPE = 'Ridge'
CORR_THRESHOLD = 0.95

# Execution
EXECUTE_AT = 'open[t+1]'
SLIPPAGE_K = 0.001
VOL_PENALTY = 0.005
COMMISSION_PER_TRADE = 0.00002
TARGET_VOL = 0.01
MAX_LEVERAGE = 3.0
MAX_POS_CHANGE_PER_MIN = 0.1
FLAT_BEFORE_CLOSE_MINUTES = 5
HTF_TREND_ALIGNMENT = True
HTF_VOL_SCALING = True
HTF_VOL_WINDOW = 10

# Metrics
ANNUALIZATION_FACTOR = 66528  # 5-min bars/year
```
</details>

<details>
<summary>Key File Map</summary>

| Module | File | Purpose |
|---|---|---|
| Config | `quant/config.py` | All constants and parameters |
| Session | `quant/session.py` | 3-stream resampling, session filtering |
| Ingest | `quant/ingest.py` | Load, clean, validate, cache aligned data |
| Alignment | `quant/align.py` | Causal HTF join (shifted backward asof) |
| Baseline Features | `quant/features/baseline.py` | 40 core 5-min features from YAML |
| HTF Context | `quant/features/htf_context.py` | Daily/hourly context features |
| Expansion | `quant/features/expansion.py` | Pairwise, cross-TF, regime, moments |
| Target | `quant/features/target.py` | 5m, 1h, 4h targets |
| Engine | `quant/features/engine.py` | Orchestrates all feature generation |
| Discovery | `quant/discovery.py` | ExtraTrees bootstrap + freeze |
| Walkforward | `quant/walkforward.py` | Ridge/RF walkforward with scaling |
| Execution | `quant/execution/simulator.py` | HTF-gated execution simulation |
| Analytics | `quant/analytics/aggregate.py` | Sharpe, drawdown, trade stats |
| CLI | `quant/cli.py` | discover, run, aggregate commands |
| Market Config | `quant/market_config.py` | Per-symbol configuration loader |
</details>