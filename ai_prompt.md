⚙️ Quant Flowchart — Three‑Stream HTF‑Aware Pipeline (Consolidated)
Deterministic CPU‑only intraday futures pipeline: 1‑min → three streams (5min, 1h, Daily) → mixed‑timeframe features → HTF context → ExtraTrees discovery → frozen features → walkforward Ridge → top‑down execution.
Implementation status: Snapshot implements 5min stream fully; 1h/Daily streams, cross‑timeframe features, HTF context, HTF execution filters missing (flagged below). Config values below match config/config.py.

Hardware: RAM 16 GB, storage 500 GB, single‑threaded (OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, POLARS_MAX_THREADS=1), CPU only (AMD Ryzen 5 2600, 6 cores / 12 logical processors), Python 3.10+, pytz (not zoneinfo).

"Pipeline Flowchart: 

1) Clean Data (get it usable)
Collect raw data
Normalize timezone
Build sessions
Resample to bars (5m + HTF)
Ensure no future leakage (causal alignment)

2) Build HTF Context (slow signals)
Compute HTF indicators (ATR, trends, slopes, etc.)
Select useful HTF features
Freeze HTF features (no peeking forward)

3) Build 5m Features (fast signals)
Create baseline 5m features (returns, volatility, microstructure)
Join frozen HTF features → conditions the 5m model

4) Expand + Select Features
Use ExtraTrees to:
Generate feature importance
Expand interactions / nonlinear signals
Prune weak features → keep only signal

5) Train Model (simple + robust)
Model = Ridge Regression

Use walkforward training:
Train on past
Test on next slice (OOS)
Repeat across time

Use per-fold scaling (no leakage)

6) Backtest (does it make money?)
Turn predictions → trades

Simulate:
position sizing (HTF volatility-based)
slippage + latency

Output:
PnL
Sharpe / drawdown
stability across folds

7) Validate (don’t fool yourself)
Reproducibility checks
No leakage tests
CI-style sanity checks

8) Deploy + Monitor
Run live with same pipeline
Track performance drift
Gate updates via CI"

MVP Scope — Minimal Viable Product (single-market)
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
	- Params: `n_estimators=100`, `max_depth=8`.
	- Compute selection frequency and mean importance; freeze top K (e.g., K=20).
- Stability checks:
	- Sign-consistency check across folds, correlation filter on frozen set. Reduce K if unstable.
- Modeling:
	- Train `Ridge` on 5-min resampled data using frozen features.
	- Deterministic scaler per fold, walkforward slices evaluate Sharpe, hit rate, turnover.
- Execution simulation (minimal):
	- Simple slippage + commission model, require flat N minutes before session close.
	- Keep position sizing fixed for MVP (no HTF scaling).
- Iterate:
	- If signal passes walkforward and checks, scale discovery folds, enable pairwise interactions, subprocess isolation, and `--prod-mode` serialization.


1. OBJECTIVE
Strict intraday Globex 23/5 18:00 America/New_York → 16:00 America/New_York, with the CME settlement gap excluded from 17:00–18:00 ET, no overnight holds. Zero leakage, memory <16GB (safe cap 14GB), seed 42, float32 only. Polars (no pandas), pytz, chunked processing.

2. GLOBAL ENV
SEED=42 for numpy/random/sklearn

OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=POLARS_MAX_THREADS=1

Libs: polars, numpy, sklearn, pyarrow, joblib, pytz

CLIP_MIN=-10.0, CLIP_MAX=10.0, EPS=1e-9

No NaN/inf → replace with 0.0

3. CONFIG (actual values from snapshot)
Paths
text
DATA_GLOB = "data/futures/*.parquet"
MANIFEST_PATH = "artifacts/manifest.json"
BASELINE_FEATURES_FILE = "config/baseline_features.yaml"
BASELINE_FEATURES_PERSIST_PATH = "artifacts/baseline_feature_matrix.parquet"
TRADES_OUT = "artifacts/trades.csv"
LOG_DIR = "logs/"
Memory & determinism
text
RAM_CAP_BYTES = 14 * 1024**3   # still leaves 2GB headroom on 16GB system
RSS_STOP_BYTES = 13.5 * 1024**3
ROWS_PER_CHUNK_MAX = 5_000_000
MEMORY_SAFETY_MARGIN = 0.95
Resampling (three streams – 1h/Daily not yet implemented)
text
SESSION_START_LOCAL = time(18,0)
SESSION_END_LOCAL = time(16,0)
SESSION_BREAK_START_LOCAL = time(17,0)
SESSION_BREAK_END_LOCAL = time(18,0)
RESAMPLE_FREQUENCIES = ["5m", "1h", "1d"]
DROP_INCOMPLETE_ROWS = True
Baseline windows (5min only – HTF windows exist but unused)
text
ROLL_WINDOWS = [5,10,20,50]
ROLL_WINDOWS_1H = [2,4,6,12]       # not implemented
ROLL_WINDOWS_DAILY = [5,10,20]     # not implemented
Feature expansion (cross‑timeframe not implemented)
text
FEATURE_TRANSFORMS = ["lags","ratios","z_scores","pairwise_products_limited","cross_timeframe_ratios"]
MAX_PAIRWISE_INTERACTIONS = 500
MAX_CROSS_TIMEFRAME_INTERACTIONS = 200   # not implemented
HTF Context (not implemented)
text
HTF_TREND_WINDOWS = [5,10,20]
HTF_VOLATILITY_WINDOWS = [5,10,20]
HTF_ALIGNMENT_FILTER = True
HTF_TREND_THRESHOLD = 0.1
Regime (implemented)
text
VOL_MEDIAN_WINDOW = 20
VOL_SMOOTH_WINDOW = 5
REGIME_HIGH_THRESH = 0.6
REGIME_LOW_THRESH = 0.4
Target & discovery
text
TARGET_5M_HORIZON = 1
DISCOVERY_WINDOW_DAYS = 60
BOOTSTRAP_FOLDS = 30
EXTRA_TREES_PARAMS = {"random_state":42,"n_jobs":1,"n_estimators":100,"max_depth":12,"max_features":0.3,"bootstrap":False}
SELECTION_FREQ_THRESHOLD = 0.75
SIGN_CONSISTENCY_THRESHOLD = 0.8
CUMULATIVE_IMPORTANCE_THRESHOLD = 0.95
MIN_SELECTED_FEATURES = 10
MAX_SELECTED_FEATURES = 1000
Walkforward & Ridge
text
WF_TRAIN_DAYS = 60
WF_TEST_DAYS = 1
WF_STEP_DAYS = 1
RIDGE_PARAMS = {"alpha":1.0,"solver":"cholesky","fit_intercept":True,"random_state":42}
Execution (HTF bias and fixed sizing not implemented)
text
EXECUTE_AT = "open[t+1]"
SLIPPAGE_K = 0.001
VOL_PENALTY = 0.005
COMMISSION_PER_TRADE = 0.00002
TARGET_VOL = 0.01
MAX_LEVERAGE = 3.0
MAX_POS_CHANGE_PER_MIN = 0.1
FLAT_BEFORE_CLOSE_MINUTES = 5
HTF_TREND_ALIGNMENT = True      # missing in simulator
HTF_VOL_SCALING = True          # missing
HTF_VOL_WINDOW = 10
FIXED_CONTRACT_SIZE = True       # execution should always use the same contract size
HTF_DIRECTIONAL_BIAS = True      # HTF should bias 5m execution to long/short/no trade
Metrics & constants
text
METRICS_TO_COMPUTE = ["Sharpe","MaxDrawdown","Turnover","HitRate","AvgWin","AvgLoss","MAE"]
ANNUALIZATION_FACTOR = 66528   # 5‑min bars/year
ROW_GROUP_SIZE = 65536
4. HARDWARE LIMITS
RSS stop at 13.5GB, hard cap 14GB (leaving 2GB for OS and other processes on 16GB system).

Check memory before each chunk.

5. DATA LOAD – THREE‑STREAM RESAMPLING (MEMORY SAFE)
Lazy scan_parquet → session filter → resample to 5min, 1h, Daily simultaneously.

Store each stream to temp files.

For each 5‑min bar, align most recent 1h bar (closed ≤ 5min timestamp) and most recent Daily bar (closed before session).

Status: Only 5min implemented in src/session.py.

6. POST‑RESAMPLE VALIDATION
OHLC integrity (high≥low, open/close within range).

Estimate combined feature matrix memory; abort if > RAM_CAP_BYTES.

7. SESSION DEFINITION & FILTERING (as in snapshot)
Add session_id by shifting timestamps +6h → date.

Keep rows where local time in [18:00, 16:00).

8. CLEANING RULES
Drop zero‑volume rows (optional).

Drop incomplete bars (config).

Replace inf/nan with 0.0.

9. BASE FEATURES & HTF CONTEXT
5min baseline: 40 features from YAML – implemented (src/features/baseline.py).

HTF state features (daily_return_1, distance_to_daily_high, hourly_trend_alignment, etc.) – not implemented.

10. FEATURE EXPANSION
Intra‑timeframe: pairwise products (capped 500), ratios, z‑scores, regime‑conditioned – implemented.

Cross‑timeframe (5min×1h, etc.) – not implemented.

11. NUMERIC GUARDS – fully implemented (clipping, float32, NaN→0).
12. REGIME – implemented (volatility‑based high/medium/low).
13. TARGET CONSTRUCTION – implemented (target_5m).
14. CORRELATION FILTER – implemented on first train fold (threshold 0.95).
15. NONLINEAR DISCOVERY (ExtraTrees) – implemented, but only 5min features (no HTF).
16. MANIFEST & CANONICAL PARQUET – implemented (src/io/canonical_parquet.py).
17. FEATURE FREEZE – implemented (manifest stores selected names).
18. WALKFORWARD RIDGE – implemented (60/1 day rolling), but only frozen 5min features.
19. TOP‑DOWN EXECUTION – partially implemented (position sizing, costs, max change, flatten). Missing: fixed contract sizing, HTF directional bias gating, HTF vol scaling, trend alignment filter.
20. NAIVE BENCHMARK – implemented (20‑period SMA crossover, long only).
21. TESTS – memory abort, serialisation reproducibility, walkforward, dtypes, manifest format.
22. ENTRYPOINT – src/cli.py discover and src/cli.py run.
23. MISSING IMPLEMENTATIONS CHECKLIST (to complete three‑stream HTF)
Resample 1h and Daily streams in src/session.py

Create HTF context features (src/features/htf_context.py)

Extend expand_features with cross‑timeframe interactions

Modify discovery to use all three streams

Update walkforward to align HTF features without lookahead

Add HTF volatility scaling and trend alignment to simulate_execution

24. FINAL DIRECTIVE
Generate a complete, runnable package that implements the three‑stream HTF‑aware pipeline as described above. Use the config values and file structure from the snapshot as the baseline. Implement all missing pieces flagged “not implemented”. Ensure determinism, memory safety, and zero leakage.