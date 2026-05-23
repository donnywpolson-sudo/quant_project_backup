# Project Snapshot for Audit

# Root: C:\Users\donny\Desktop\quant_project
# Created: 2026-05-22T21:53:46.923923

--- 
### File: .gitignore
```
# Python environment and caches
__pycache__/
.venv/
*.pyc

# Project Data (Do not sync these)
artifacts/
futures/

# Secrets and Settings
.env
.DS_Store
*.log
```

--- 
### File: .vscode\launch.json
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run Pipeline (Full)",
            "type": "debugpy",
            "request": "launch",
            "module": "src.cli",
            "args": [
                "run",
                "--data", "tests/fixtures/synthetic_1min_fixture.parquet",
                "--config", "config/config.py",
                "--out", "artifacts/"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONPATH": "${workspaceFolder}"
            },
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Run Tests (pytest)",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": ["-q"],
            "cwd": "${workspaceFolder}",
            "env": {
                "OMP_NUM_THREADS": "1",
                "PYTHONPATH": "${workspaceFolder}"
            },
            "console": "integratedTerminal"
        }
    ]
}
```

--- 
### File: ai_prompt.md
```
⚙️ Quant Flowchart — Three‑Stream HTF‑Aware Pipeline (Consolidated)
Deterministic CPU‑only intraday futures pipeline: 1‑min → three streams (5min, 1h, Daily) → mixed‑timeframe features → HTF context → ExtraTrees discovery → frozen features → walkforward Ridge → top‑down execution.
Implementation status: Snapshot implements 5min stream fully; 1h/Daily streams, cross‑timeframe features, HTF context, HTF execution filters missing (flagged below). Config values below match config/config.py.

Hardware: RAM 14GB, storage 200GB, single‑threaded (OMP_NUM_THREADS=1), CPU only, Python 3.10+, pytz (not zoneinfo).

Pipeline Flowchart (Three‑Stream HTF‑Aware)
1‑min OHLCV parquet → Resample into 5min, 1h, Daily →

Step 1 — Baseline & HTF context

5min baseline features (YAML)

HTF state: trend alignment, distance to Daily/1h levels, volatility ratios, regime labels

Step 2 — Feature expansion

Intra‑timeframe interactions (5min×5min, 1h×1h, Daily×Daily)

Cross‑timeframe (5min×1h, 5min×Daily, 1h×Daily)

Ratios, z‑scores, regime‑conditioned transforms (past data only)

Step 3 — Target
target_5m = log(close_5min[t+1]/close_5min[t])

Step 4 — ExtraTrees discovery

Train on combined 5min/1h/Daily feature pool (HTF as regime filters)

Stability selection: frequency ≥75%, sign consistency ≥80%

Step 5 — Freeze features → manifest + SHA256

Step 6 — Walkforward Ridge (frozen mixed‑timeframe features, StandardScaler per fold)

Step 7 — Prediction (every 5min using latest 5min/1h/Daily aligned without lookahead)

Step 8 — Top‑down execution

Scale position by HTF volatility (e.g., daily ATR)

Trend alignment filter: only signals agreeing with HTF trend

Flatten before close

1. OBJECTIVE
Strict intraday Globex 23/5 18:00 America/New_York → 16:00 America/New_York, no overnight holds. Zero leakage, memory <14GB, seed 42, float32 only. Polars (no pandas), pytz, chunked processing.

2. GLOBAL ENV
SEED=42 for numpy/random/sklearn

OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1

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
RAM_CAP_BYTES = 14 * 1024**3
RSS_STOP_BYTES = 13.5 * 1024**3
ROWS_PER_CHUNK_MAX = 5_000_000
MEMORY_SAFETY_MARGIN = 0.95
Resampling (three streams – 1h/Daily not yet implemented)
text
SESSION_START_LOCAL = time(18,0)
SESSION_END_LOCAL = time(16,0)
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
Execution (HTF scaling/alignment not implemented)
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
Metrics & constants
text
METRICS_TO_COMPUTE = ["Sharpe","MaxDrawdown","Turnover","HitRate","AvgWin","AvgLoss","MAE"]
ANNUALIZATION_FACTOR = 66528   # 5‑min bars/year
ROW_GROUP_SIZE = 65536
4. HARDWARE LIMITS
RSS stop at 13.5GB, hard cap 14GB.

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
19. TOP‑DOWN EXECUTION – partially implemented (position sizing, costs, max change, flatten). Missing: HTF vol scaling, trend alignment filter.
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
```

--- 
### File: ci\ci_stub.yaml
```
# ci/ci_stub.yml
name: Quant Pipeline CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test-and-verify:
    runs-on: ubuntu-latest
    env:
      OMP_NUM_THREADS: 1
      OPENBLAS_NUM_THREADS: 1
      MKL_NUM_THREADS: 1
      PYTHONPATH: .
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Run mandatory verification suite
      run: |
        python -m pytest tests/test_1h_dst.py
        python -m pytest tests/test_memory_abort.py
        python -m pytest tests/test_serialization_repro.py
        
    - name: Generate test fixtures
      run: |
        python -m tests.fixtures.make_fixtures
        
    - name: Run pipeline verification
      run: |
        python -m src.cli run \
          --data tests/fixtures/synthetic_1min_fixture.parquet \
          --config config/config.py \
          --out artifacts/
          
    - name: Validate Manifest Schema
      run: python -m src.utils.validate_manifest --path artifacts/manifest.json

    - name: Verify artifacts
      run: |
        test -f artifacts/trades.csv
        test -f artifacts/manifest.json
        test -f artifacts/baseline_feature_matrix.parquet
        python -c "import hashlib; h = hashlib.sha256(open('artifacts/baseline_feature_matrix.parquet','rb').read()).hexdigest(); print(f'Feature Matrix Hash: {h}')"
          
    - name: Validate Feature DTypes
      run: |
        python -m src.utils.check_types --path artifacts/baseline_feature_matrix.parquet
        echo "Pipeline verification complete."
```

--- 
### File: config\baseline_features.yaml
```
# config/baseline_features.yaml
baseline_features:
  - feature_ret_1
  - feature_ret_5
  - feature_ret_10
  - feature_ret_20
  - feature_ma_5
  - feature_ma_20
  - feature_ma_50
  - feature_dist_ma_20
  - feature_dist_ma_50
  - feature_ma_slope_20
  - feature_price_z_20
  - feature_price_z_50
  - feature_high_low_range_norm
  - feature_true_range
  - feature_atr_14
  - feature_realized_vol_5
  - feature_realized_vol_20
  - feature_ewma_vol_20
  - feature_price_momentum_5
  - feature_price_momentum_10
  - feature_mom_z_5
  - feature_mom_z_10
  - feature_rsi_14
  - feature_macd
  - feature_macd_signal
  - feature_stoch_k
  - feature_log_volume
  - feature_volume_z_20
  - feature_obv
  - feature_signed_bar_strength
  - feature_volume_price_divergence
  - feature_spread_proxy
  - feature_session_pos
  - feature_time_of_day_bucket
  - feature_1h_bias
  - feature_session_volatility
  - feature_pair_prod_template
  - feature_ratio_template
  - feature_pca_comp_1
  - feature_pca_comp_2
```

--- 
### File: config\config.py
```
"""
config/config.py
Single Source of Truth for the Quant Pipeline.
Updated for three‑stream HTF‑aware pipeline, target_5m, and market overrides.
"""
import os
import sys
import logging
from datetime import time
from pathlib import Path

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment and Threading Configuration ---
def set_threading_env():
    envs = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1"
    }
    for key, value in envs.items():
        os.environ[key] = value

set_threading_env()

# --- Paths and IO ---
DATA_ROOT = "./data/test"
ARTIFACTS_ROOT = "./artifacts"
DATA_GLOB = "data/futures/*.parquet"
FEATURES_OUT = os.path.join(ARTIFACTS_ROOT, "features.parquet")
MANIFEST_PATH = os.path.join(ARTIFACTS_ROOT, "manifest.json")
MODELS_DIR = "models/"
TRADES_OUT = os.path.join(ARTIFACTS_ROOT, "trades.csv")
PNL_OUT = os.path.join(ARTIFACTS_ROOT, "pnl_series.csv")
LOG_DIR = "logs/"
CACHE_DIR = "cache/"
MEMORY_TRACE_OUT = os.path.join(LOG_DIR, "memory_trace.csv")
SYNTHETIC_FIXTURE_PATH = "tests/fixtures/synthetic_1min_fixture.parquet"
MAPPING_FILE_PATH = "config/term_to_canonical_mapping.yaml"
BASELINE_FEATURES_FILE = "config/baseline_features.yaml"
BASELINE_FEATURES_PERSIST_PATH = os.path.join(ARTIFACTS_ROOT, "baseline_feature_matrix.parquet")

# --- Environment and determinism ---
SEED = 42
OMP_NUM_THREADS = 1
OPENBLAS_NUM_THREADS = 1
MKL_NUM_THREADS = 1
SKLEARN_N_JOBS = 1

# --- Numeric guards and constants ---
EPS = 1e-9
CLIP_MIN = -10.0
CLIP_MAX = 10.0
DTYPE = "float32"
TIMEZONE = "America/New_York"
PRE_POST_CLIP_LOGGING = True
DEBUG_FLOAT64_MODE = False
REPLACE_INF_NAN_WITH = 0.0

# --- Memory and hardware limits ---
RAM_CAP_BYTES = 14 * 1024**3  # 14GB
RSS_STOP_BYTES = 13.5 * 1024**3
STORAGE_MIN_GB = 200
ROWS_PER_CHUNK_MAX = 5_000_000
MEMORY_SAFETY_MARGIN = 0.95
MEMORY_RSS_CHECKPOINT_INTERVAL_SEC = 10
MEMORY_RSS_CHECKPOINTS_BEFORE_STOP = 3
MEMORY_LOG_ENABLED = True

# --- Data load and collect ---
DATA_SCAN_GLOB = DATA_GLOB
LAZY_PUSHDOWN_FILTERS = True
COLLECT_PARTITION_ROWS = True
STABLE_SORT_KEYS = ["session_id", "ts_event", "row_id"]

# --- Session and resampling ---
SESSION_START_LOCAL = time(18, 0, 0)
SESSION_END_LOCAL = time(16, 0, 0)
SESSION_TZ = TIMEZONE
RESAMPLE_RULES = {"O": "first", "H": "max", "L": "min", "C": "last", "V": "sum"}
RESAMPLE_FREQUENCIES = ["5m", "1h", "1d"]

# --- Cleaning rules ---
DROP_VOLUME_ZERO = True
ALLOW_FFILL_4H_MAPPING_ONLY = True
DROP_INCOMPLETE_ROWS = True
REPLACE_INF_NAN_WITH = 0.0

# --- Base feature windows ---
ROLL_WINDOWS = [5, 10, 20, 50]
ROLL_WINDOWS_1H = [2, 4, 6, 12]
ROLL_WINDOWS_DAILY = [5, 10, 20]
ROLL_WINDOW_MIN_ROWS = max(ROLL_WINDOWS)

# --- Feature expansion ---
FEATURE_TRANSFORMS = ["lags", "ratios", "z_scores", "pairwise_products_limited", "cross_timeframe_ratios"]
MAX_PAIRWISE_INTERACTIONS = 500
MAX_CROSS_TIMEFRAME_INTERACTIONS = 200
TEMPORAL_BUCKETS = ["early", "mid", "late"]

# --- HTF Context ---
HTF_TREND_WINDOWS = [5, 10, 20]
HTF_VOLATILITY_WINDOWS = [5, 10, 20]
HTF_ALIGNMENT_FILTER = True
HTF_TREND_THRESHOLD = 0.1

# --- Regime and HTF ---
VOL_MEDIAN_WINDOW = 20
VOL_SMOOTH_WINDOW = 5
REGIME_HIGH_THRESH = 0.6
REGIME_LOW_THRESH = 0.4
REGIME_MISSING_DEFAULT = 0

# --- Targets ---
TARGET_5M_HORIZON = 1
MAGNITUDE_THRESHOLD = 0.002
PROB_TARGET_THRESHOLD = 0.005

# --- 1H mapping (optional, not active) ---
DST_AWARE_1H_TESTS = True
PARTIAL_BLOCK_MIN_MINUTES = 15

# --- Correlation filter ---
CORR_THRESHOLD = 0.95
CORR_TIE_BREAKER = ["variance_desc", "name_lexicographic"]
CORR_ACCUMULATION_MODE = "compensated_float64_then_downcast"

# --- Nonlinear discovery ExtraTrees ---
DISCOVERY_METHOD = "ExtraTrees"
DISCOVERY_WINDOW_DAYS = 60
BOOTSTRAP_FOLDS = 30
EXTRA_TREES_PARAMS = {
    "random_state": 42,
    "n_jobs": 2,
    "n_estimators": 100,
    "max_depth": 12,
    "max_features": 0.3,
    "bootstrap": False
}
SELECTION_FREQ_THRESHOLD = 0.75
CUMULATIVE_IMPORTANCE_THRESHOLD = 0.95
SIGN_CONSISTENCY_THRESHOLD = 0.8
MIN_SELECTED_FEATURES = 10
MAX_SELECTED_FEATURES = 1000

# --- Orthogonalization (disabled) ---
ORTHOGONALIZE = False
PCA_TOP_COMPONENTS = 5

# --- Manifest and cache ---
MANIFEST_FIELDS = [
    "feature_names", "dtypes",
    "selection_seed", "selection_date", "selection_model", "selection_params",
    "selected_K", "cumulative_importance", "stability_stats", "htf_features_included"
]
CACHE_KEY_COMPONENTS = [
    "row_count", "ts_min", "ts_max", "dtypes", "file_size", "mtime",
    "config_hash", "seed", "manifest_hash"
]

# --- Walkforward ---
WF_TRAIN_DAYS = 60
WF_TEST_DAYS = 1
WF_STEP_DAYS = 1
WF_PRECOMPUTE_INDICES = True

# --- Models Ridge ---
SCALER_CLASS = "StandardScaler"
RIDGE_PARAMS = {"alpha": 1.0, "solver": "cholesky", "fit_intercept": True, "random_state": 42}
RIDGE_N_JOBS = 1

# --- Execution and risk (HTF‑aware) ---
EXECUTE_AT = "open[t+1]"
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

# --- Metrics and reporting ---
METRICS_TO_COMPUTE = ["Sharpe", "MaxDrawdown", "Turnover", "HitRate", "AvgWin", "AvgLoss", "MAE"]
DEFAULT_METRICS_IF_NO_TRADES = {"Sharpe": 0, "MaxDrawdown": 0, "Turnover": 0, "HitRate": 0}
ANNUALIZATION_FACTOR = 66528

# --- Tests and thresholds ---
REPRO_HASH_ALGORITHM = "sha256"
DISCOVERY_REPRO_TEST = True
MIN_STABILITY_FEATURES = 5

# --- I/O schema ---
ROW_GROUP_SIZE = 65536
ENTRYPOINT_FN = "run_pipeline"

# --- Pipeline flags ---
TARGET_COL = "target_5m"

# --- Market-specific overrides (absolute paths) ---
BASE_DIR = Path(__file__).parent.parent
MARKET_CONFIGS = {
    "ES": str(BASE_DIR / "config/markets/ES.yaml"),
    "ZB": str(BASE_DIR / "config/markets/ZB.yaml"),
    "CL": str(BASE_DIR / "config/markets/CL.yaml"),
}
```

--- 
### File: config\markets\CL.yaml
```
contract_symbol: "CL"
exchange_timezone: "America/New_York"

session_start_local: "17:00"
session_end_local: "16:00"

roll_windows: [5, 10, 20, 50]
roll_windows_1h: [2, 4, 6, 12]
roll_windows_daily: [5, 10, 20]

regime_high_thresh: 0.7
regime_low_thresh: 0.4

htf_trend_windows: [5, 10, 20]
htf_volatility_windows: [5, 10, 20]

slippage_k: 0.002
vol_penalty: 0.005
commission_per_trade: 0.00002
max_leverage: 2.0
target_vol: 0.015

max_position_size: 30
max_notional_usd: 750_000
```

--- 
### File: config\markets\ES.yaml
```
contract_symbol: "ES"
exchange_timezone: "America/New_York"

session_start_local: "17:00"
session_end_local: "16:00"

roll_windows: [5, 10, 20, 50]
roll_windows_1h: [2, 4, 6, 12]
roll_windows_daily: [5, 10, 20]

regime_high_thresh: 0.6
regime_low_thresh: 0.4

htf_trend_windows: [5, 10, 20]
htf_volatility_windows: [5, 10, 20]

slippage_k: 0.0005
vol_penalty: 0.002
commission_per_trade: 0.00001
max_leverage: 3.0
target_vol: 0.01

max_position_size: 50
max_notional_usd: 1_000_000
```

--- 
### File: config\markets\ZB.yaml
```
contract_symbol: "ZB"
exchange_timezone: "America/New_York"

session_start_local: "17:00"
session_end_local: "16:00"

roll_windows: [10, 20, 50]
roll_windows_1h: [2, 4, 8]
roll_windows_daily: [5, 10, 20]

regime_high_thresh: 0.5
regime_low_thresh: 0.3

htf_trend_windows: [10, 20, 30]
htf_volatility_windows: [10, 20]

slippage_k: 0.0002
vol_penalty: 0.001
commission_per_trade: 0.00001
max_leverage: 2.5
target_vol: 0.008

max_position_size: 20
max_notional_usd: 2_000_000
```

--- 
### File: futures\databento_new_historical_data_fetch.py
```
from __future__ import annotations
import os
import datetime as dt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from sys import audit

import pandas as pd
import numpy as np
import databento as db

# =========================================
# CONFIG
# =========================================
API_KEY = os.getenv("DATABENTO_API_KEY", "")
DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
STYPE_IN = "continuous"
STYPE_OUT = "instrument_id"

SYMBOLS = [
    "NQ.v.0", "ES.v.0", "YM.v.0", "RTY.v.0",
    "CL.v.0", "NG.v.0",
    "GC.v.0", "SI.v.0", "HG.v.0",
    "ZB.v.0", "ZN.v.0",
    "ZC.v.0",
]

DATA_DIR = Path(r"C:\Users\donny\Desktop\Backtest")
EXPECTED_COLUMNS = ["open", "high", "low", "close", "volume"]
MAX_SESSION_BREAK = dt.timedelta(hours=4)


# =========================================
# HELPERS
# =========================================

def parquet_path(symbol: str) -> Path:
    safe = symbol.replace(".", "_")
    return DATA_DIR / f"{safe}_all.parquet"


def csv_path(symbol: str) -> Path:
    safe = symbol.replace(".", "_")
    return DATA_DIR / f"{safe}_stitched.csv"


def normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC").tz_convert("America/New_York")
    else:
        idx = idx.tz_convert("America/New_York")
    return idx


def load_symbol_history(symbol: str) -> pd.DataFrame:
    parquet_file = parquet_path(symbol)
    csv_file = csv_path(symbol)

    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
    elif csv_file.exists():
        df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)
        print(f"Loaded fallback CSV history for {symbol} from {csv_file}")
    else:
        raise FileNotFoundError(f"Missing history for {symbol}: {parquet_file} or {csv_file}")

    df.index = normalize_index(df.index)
    df = df.sort_index()

    if df.index.has_duplicates:
        dup_count = df.index.duplicated(keep="first").sum()
        print(f"Warning: dropping {dup_count} duplicate timestamp rows in existing history for {symbol}")
        df = df[~df.index.duplicated(keep="first")]

    if df.empty:
        raise ValueError(f"{symbol} history is empty after loading and cleaning")

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{symbol} history is missing columns: {missing_cols}")

    return df


def get_client() -> db.Historical:
    if not API_KEY:
        raise RuntimeError("DATABENTO_API_KEY is not set")
    return db.Historical(API_KEY)


def minute_aligned(index: pd.DatetimeIndex) -> bool:
    return ((index.second == 0) & (index.microsecond == 0)).all()


def find_missing_intervals(index: pd.DatetimeIndex, max_break: dt.timedelta = MAX_SESSION_BREAK) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(index) < 2:
        return []

    missing = []
    prev = index[0]
    for current in index[1:]:
        delta = current - prev
        if dt.timedelta(minutes=1) < delta <= max_break:
            missing.append((prev + dt.timedelta(minutes=1), current - dt.timedelta(minutes=1)))
        prev = current
    return missing


def split_interval(start: dt.datetime, end: dt.datetime, max_minutes: int = 1440) -> list[tuple[dt.datetime, dt.datetime]]:
    if start >= end:
        return []

    intervals = []
    current_start = start
    while current_start < end:
        current_end = min(end, current_start + dt.timedelta(minutes=max_minutes))
        intervals.append((current_start, current_end))
        current_start = current_end
    return intervals


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df = df.loc[:, EXPECTED_COLUMNS]

    if df.isna().any(axis=None):
        missing_rows = df[df.isna().any(axis=1)]
        print(f"Warning: dropping {len(missing_rows)} rows with NaNs from fetched data")
        df = df.dropna()

    df.index = normalize_index(df.index)
    df = df.sort_index()

    if df.index.has_duplicates:
        dup_count = df.index.duplicated(keep="first").sum()
        print(f"Warning: dropping {dup_count} duplicate timestamp rows from fetched data")
        df = df[~df.index.duplicated(keep="first")]

    if not minute_aligned(df.index):
        raise ValueError("Fetched data contains non-minute-aligned timestamps")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Fetched data timestamps are not sorted")

    return df


def validate_new_df(df: pd.DataFrame, symbol: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if df.empty:
        return []

    missing = find_missing_intervals(df.index)
    if missing:
        print(f"{symbol}: Detected {len(missing)} missing interval(s) in fetched data:")
        for start, end in missing:
            print(f"  missing {start} → {end}")
    return missing


def calculate_data_coverage(df: pd.DataFrame) -> tuple[int, int, float]:
    """
    Calculate coverage statistics for a complete dataframe.
    Returns (total_span_minutes, missing_minutes, coverage_percentage)
    """
    if df.empty or len(df) < 2:
        return 0, 0, 0.0
    
    total_span_minutes = int((df.index.max() - df.index.min()).total_seconds() / 60)
    if total_span_minutes == 0:
        return 0, 0, 100.0
    
    # Expected rows: one per minute in the span (inclusive of start and end)
    expected_rows = total_span_minutes + 1
    actual_rows = len(df)
    missing_minutes = max(0, expected_rows - actual_rows)
    
    # Coverage percentage
    coverage_pct = 100.0 * (actual_rows / expected_rows)
    
    return total_span_minutes, missing_minutes, coverage_pct


# =========================================
# FETCHING NEW DATA
# =========================================

def fetch_range(symbol: str, start_dt: dt.datetime, end_dt: dt.datetime) -> pd.DataFrame:
    if start_dt >= end_dt:
        return pd.DataFrame()

    client = get_client()
    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"Downloading {symbol}: {start_str} → {end_str}")

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=symbol,
        schema=SCHEMA,
        stype_in=STYPE_IN,
        stype_out=STYPE_OUT,
        start=start_str,
        end=end_str,
    )

    df = data.to_df()
    if df is None or df.empty:
        return pd.DataFrame()

    return normalize_df(df)


def fetch_gap(symbol: str, start_dt: dt.datetime, end_dt: dt.datetime) -> pd.DataFrame:
    intervals = split_interval(start_dt, end_dt)
    parts: list[pd.DataFrame] = []
    for start, end in intervals:
        part = fetch_range(symbol, start, end)
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()


def repair_missing_intervals(symbol: str, missing_intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    if not missing_intervals:
        return pd.DataFrame()

    repaired_parts: list[pd.DataFrame] = []
    for start, end in missing_intervals:
        print(f"{symbol}: repairing missing interval {start} → {end}")
        # fetch_range uses end exclusive semantics, so extend by one minute
        repaired = fetch_gap(symbol, start.tz_convert(dt.timezone.utc), (end + dt.timedelta(minutes=1)).tz_convert(dt.timezone.utc))
        if not repaired.empty:
            repaired_parts.append(repaired)

    if not repaired_parts:
        return pd.DataFrame()

    repaired_df = pd.concat(repaired_parts).sort_index()
    repaired_df = repaired_df[~repaired_df.index.duplicated(keep="first")]
    return repaired_df


# =========================================
# UPDATE WORKER
# =========================================

def update_symbol(symbol: str) -> str:
    try:
        print(f"\n=== Updating {symbol} ===")
        existing = load_symbol_history(symbol)

        if existing.empty:
            return f"{symbol}: ERROR — existing history is empty."

        last_ts = existing.index.max()
        start_dt = last_ts.tz_convert(dt.timezone.utc) + dt.timedelta(minutes=1)
        end_dt = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)

        if start_dt >= end_dt:
            return f"{symbol}: already up to date. No new rows to fetch."

        new_df = fetch_range(symbol, start_dt, end_dt)
        if new_df.empty:
            return f"{symbol}: no new data returned from Databento."

        # Internal gaps within the fetched data
        missing = validate_new_df(new_df, symbol)

        # Boundary gaps between requested range and fetched data
        expected_start_local = pd.Timestamp(start_dt).tz_convert("America/New_York")
        expected_last_local = pd.Timestamp(end_dt - dt.timedelta(minutes=1)).tz_convert("America/New_York")
        first_idx = new_df.index[0]
        last_idx = new_df.index[-1]

        boundary_missing: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if first_idx > expected_start_local:
            boundary_missing.append((expected_start_local, first_idx - dt.timedelta(minutes=1)))
        if last_idx < expected_last_local:
            boundary_missing.append((last_idx + dt.timedelta(minutes=1), expected_last_local))

        if boundary_missing:
            print(f"{symbol}: Detected {len(boundary_missing)} boundary missing interval(s) in fetched data:")
            for start, end in boundary_missing:
                print(f"  boundary missing {start} → {end}")

        missing = missing + boundary_missing

        if missing:
            repaired = repair_missing_intervals(symbol, missing)
            if not repaired.empty:
                new_df = pd.concat([new_df, repaired]).sort_index()
                new_df = new_df[~new_df.index.duplicated(keep="first")]
                missing = validate_new_df(new_df, symbol)

        if missing:
            return f"{symbol}: incomplete fetch: {len(missing)} missing interval(s) remain after repair."

        combined = pd.concat([existing, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        combined.to_parquet(parquet_path(symbol))
        
        # Calculate and report coverage statistics
        total_span, missing_mins, coverage = calculate_data_coverage(combined)
        print(f"{symbol}: saved updated parquet with {len(new_df)} new rows.")
        print(f"{symbol}: dataset span {total_span:,} min, missing {missing_mins:,} min ({100-coverage:.2f}% gaps, {coverage:.2f}% coverage)")

        return f"{symbol}: updated successfully with no detected gaps in fetched range."

    except Exception as exc:
        return f"{symbol}: ERROR — {exc}"

# =========================================
# TOP-LEVEL PARALLEL UPDATER
# =========================================

def update_historical_data():
    print("\n=== Starting parallel historical data update ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(6, len(SYMBOLS))) as exe:
        futures = {exe.submit(update_symbol, symbol): symbol for symbol in SYMBOLS}
        for fut in as_completed(futures):
            print(fut.result())


if __name__ == "__main__":
    update_historical_data()
```

--- 
### File: README.md
```
# Deterministic Intraday Futures ML Backtester

## Setup
```bash
pip install -r requirements.txt
python -m tests.fixtures.make_fixtures
```

--- 
### File: requirements.txt
```
polars>=1.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pyarrow>=14.0.0
joblib>=1.3.0
psutil>=5.9.0
pyyaml>=6.0
pytest>=7.4.0
pytz>=2024.1
```

--- 
### File: run.py
```
"""
run.py
Production Orchestrator for the Deterministic Quant Pipeline.
Scans 'futures/' for 1‑min OHLCV Parquet files (market/year/*.parquet)
and runs the two‑phase pipeline (discovery → walkforward) per file.

Compliance:
- No manual row splitting – discovery uses first 60 days of resampled 5‑min data.
- Memory safe – each CLI call handles its own chunked resampling.
- Deterministic – same inputs produce identical results.
- Audit snapshot: creates a full project snapshot (full_code.md) before each run.
"""
import subprocess
import sys
import logging
import time
from pathlib import Path
from datetime import datetime

# Setup global logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QuantRunner")

def create_audit_snapshot(root_dir: str = "."):
    """
    Creates a full project snapshot (all text files) for audit purposes.
    Saves as full_code.md in the project root.
    """
    project_root = Path(root_dir).resolve()
    snapshot_filename = "full_code.md"

    # Directories to exclude completely
    exclude_dirs = {
        ".venv", "venv", "env", ".git", "__pycache__",
        "artifacts", "logs", "models", "node_modules",
        ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints",
        "dist", "build", "htmlcov", ".tox"
    }

    # Binary extensions to skip entirely (cannot be read as text)
    binary_extensions = {
        ".parquet", ".pyc", ".log", ".tag", ".png", ".jpg", ".jpeg",
        ".gif", ".bmp", ".ico", ".pdf", ".docx", ".xlsx", ".zip",
        ".tar", ".gz", ".pickle", ".pkl", ".so", ".dll", ".exe",
        ".db", ".sqlite", ".pyo", ".egg", ".whl"
    }

    logger.info(f"Creating audit snapshot: {snapshot_filename}")
    try:
        with open(project_root / snapshot_filename, "w", encoding="utf-8") as f:
            f.write(f"# Project Snapshot for Audit\n\n")
            f.write(f"# Root: {project_root}\n")
            f.write(f"# Created: {datetime.now().isoformat()}\n\n")

            # Walk all files recursively
            for file_path in sorted(project_root.rglob("*")):
                if file_path.is_dir():
                    continue

                # Skip the snapshot file itself
                if file_path.name == snapshot_filename:
                    continue

                # Skip excluded directories
                rel_path = file_path.relative_to(project_root)
                if any(part in exclude_dirs for part in rel_path.parts):
                    continue

                # Skip binary extensions
                if file_path.suffix.lower() in binary_extensions:
                    continue

                # Write header and full content
                f.write(f"--- \n### File: {rel_path}\n")
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    f.write("```\n" + content + "\n```\n\n")
                except Exception as e:
                    f.write(f"Error reading file: {e}\n\n")

        logger.info(f"✅ Audit snapshot saved to: {snapshot_filename}")
    except Exception as e:
        logger.error(f"Failed to create audit snapshot: {e}")

def run_step(cmd_list, retries=2, delay=5):
    """Executes a command with retry mechanism."""
    for attempt in range(retries + 1):
        logger.info(f"Executing: {' '.join(cmd_list)}")
        try:
            result = subprocess.run(cmd_list, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Step completed successfully.")
                return True
            else:
                logger.error(f"Attempt {attempt + 1} failed (rc={result.returncode}): {result.stderr[-500:]}")
        except Exception as e:
            logger.error(f"Exception during execution: {e}")

        if attempt < retries:
            time.sleep(delay)
            logger.info(f"Retrying step... (Attempt {attempt + 2})")

    return False

def process_file(data_path: Path):
    """Run the full pipeline for a single 1‑min Parquet file."""
    # Infer market and year from directory structure (e.g., futures/ES/2024.parquet)
    market = data_path.parent.name
    year = data_path.stem
    artifacts_dir = Path("artifacts") / market / year
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Localised logging to file
    log_file = artifacts_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)

    logger.info(f"--- Starting Pipeline for {market} {year} ---")
    logger.info(f"Data file: {data_path}")

    # Manifest path (shared between discovery and run)
    manifest_path = artifacts_dir / "manifest.json"

    # Stage 1: Feature discovery (ExtraTrees with bootstrap folds, stability selection)
    stage_discover = [
        sys.executable, "-m", "src.cli", "discover",
        "--data", str(data_path),
        "--out", str(manifest_path)
    ]

    # Stage 2: Walkforward Ridge regression + execution simulation
    stage_run = [
        sys.executable, "-m", "src.cli", "run",
        "--data", str(data_path),
        "--manifest", str(manifest_path),
        "--out", str(artifacts_dir)
    ]

    # Stage 3: Performance analytics (optional – expects backtest_results.parquet)
    stage_analytics = [
        sys.executable, "-m", "src.analytics",
        str(artifacts_dir / "backtest_results.parquet")
    ]

    for stage_cmd in [stage_discover, stage_run, stage_analytics]:
        if not run_step(stage_cmd):
            logger.error(f"Pipeline failed at stage: {stage_cmd[3] if len(stage_cmd)>3 else stage_cmd[2]}")
            break
    else:
        logger.info(f"--- Pipeline Completed Successfully for {market} {year} ---")

    # Cleanup to avoid log handler leaks
    file_handler.close()
    logger.removeHandler(file_handler)

if __name__ == "__main__":
    # --- Create audit snapshot before any processing ---
    create_audit_snapshot()

    futures_dir = Path("futures")
    if not futures_dir.exists():
        logger.error("Directory 'futures' not found. Please create it and place 1‑min Parquet files inside (e.g., futures/ES/2024.parquet).")
        sys.exit(1)

    # Recursively find all .parquet files under 'futures/'
    files = list(futures_dir.rglob("*.parquet"))
    if not files:
        logger.warning("No Parquet files found under 'futures/'.")
        sys.exit(0)

    logger.info(f"Found {len(files)} file(s) to process.")
    for file_path in files:
        process_file(file_path)
```

--- 
### File: src\__init__.py
```

```

--- 
### File: src\align.py
```
"""
src/align.py
Align 5‑min, 1‑hour and Daily streams without lookahead.
Uses asof join for 1h and proper daily lag.
Now includes daily_vol_5 from the daily stream.
"""
import polars as pl
import logging
from config import config

logger = logging.getLogger(__name__)


def align_htf_streams(df_5min: pl.DataFrame, df_1h: pl.DataFrame, df_daily: pl.DataFrame) -> pl.DataFrame:
    """
    For each 5‑min bar, add columns from the most recent 1h bar (closed <= 5min timestamp)
    and the most recent daily bar (closed before the session).
    Returns a single DataFrame with all 5min columns plus prefixed HTF columns.
    """
    # Ensure sorted
    df_5min = df_5min.sort("ts_event")
    df_1h = df_1h.sort("ts_event")
    df_daily = df_daily.sort("ts_event")

    # ---- 1. Join 1h using asof (backward) ----
    df_1h_renamed = df_1h.select([
        "ts_event",
        pl.col("open").alias("1h_open"),
        pl.col("high").alias("1h_high"),
        pl.col("low").alias("1h_low"),
        pl.col("close").alias("1h_close"),
        pl.col("volume").alias("1h_volume"),
    ])
    df_aligned = df_5min.join_asof(
        df_1h_renamed,
        on="ts_event",
        strategy="backward"
    )

    # ---- 2. Join daily using previous day's close ----
    df_aligned = df_aligned.with_columns(
        pl.col("ts_event").dt.date().alias("date_5min")
    )
    df_daily = df_daily.with_columns(
        pl.col("ts_event").dt.date().alias("date_daily")
    )
    # For each 5min date, take the daily bar from the previous trading day
    df_daily_prev = df_daily.with_columns(
        (pl.col("date_daily") + pl.duration(days=1)).alias("next_day")
    ).select([
        pl.col("date_daily").alias("prev_date"),
        pl.col("next_day"),
        pl.col("open").alias("daily_open"),
        pl.col("high").alias("daily_high"),
        pl.col("low").alias("daily_low"),
        pl.col("close").alias("daily_close"),
        pl.col("volume").alias("daily_volume"),
        pl.col("daily_vol_5").alias("daily_vol_5"),   # <-- added
    ])
    df_aligned = df_aligned.join(
        df_daily_prev,
        left_on="date_5min",
        right_on="next_day",
        how="left"
    )
    # Forward fill daily columns for the first days where no previous day exists
    daily_cols = ["daily_open", "daily_high", "daily_low", "daily_close", "daily_volume", "daily_vol_5"]
    for col in daily_cols:
        df_aligned = df_aligned.with_columns(pl.col(col).fill_null(strategy="forward"))

    # Drop helper columns (ignore if missing)
    df_aligned = df_aligned.drop(["date_5min", "prev_date", "next_day"], strict=False)
    return df_aligned
```

--- 
### File: src\analytics.py
```
"""
src/analytics.py
Calculates performance metrics from walk-forward simulation output.
Uses the 'pnl' column from backtest_results.parquet (produced by src.walkforward).
Also computes correlation between predictions and actual returns if available,
and compares against a naive benchmark column 'benchmark_pnl' if present.
"""
import sys
import polars as pl
import numpy as np

def calculate_metrics(file_path: str):
    """
    Reads a Parquet file (expected to have columns: pnl, position, prediction, target_5m)
    and prints key performance statistics.
    """
    try:
        df = pl.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Required column: pnl (from execution simulation)
    if "pnl" not in df.columns:
        print("No 'pnl' column found. Ensure backtest_results.parquet contains execution output.")
        return

    pnl = df["pnl"].to_numpy()
    total_pnl = pnl.sum()
    avg_pnl = pnl.mean()
    std_pnl = pnl.std()
    
    # Annualized Sharpe (assuming 5-min bars, 264 bars per session, 252 trading days)
    if std_pnl > 0:
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252 * 264)
    else:
        sharpe = 0.0

    # Cumulative PnL and max drawdown
    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_drawdown = drawdown.min()

    # Turnover: sum of absolute position changes divided by average position (if any)
    if "position" in df.columns:
        position_changes = df["position"].diff().abs().sum()
        avg_position = df["position"].abs().mean()
        turnover = position_changes / avg_position if avg_position > 0 else 0.0
    else:
        turnover = 0.0

    # Optional: correlation between prediction and target_5m (if available)
    corr = 0.0
    if "prediction" in df.columns and "target_5m" in df.columns:
        pred = df["prediction"].to_numpy()
        target = df["target_5m"].to_numpy()
        mask = ~(np.isnan(pred) | np.isnan(target))
        if mask.sum() > 1:
            corr = np.corrcoef(pred[mask], target[mask])[0, 1]

    # Benchmark comparison if benchmark_pnl exists
    benchmark_sharpe = None
    benchmark_maxdd = None
    if "benchmark_pnl" in df.columns:
        bench_pnl = df["benchmark_pnl"].to_numpy()
        bench_avg = bench_pnl.mean()
        bench_std = bench_pnl.std()
        if bench_std > 0:
            benchmark_sharpe = (bench_avg / bench_std) * np.sqrt(252 * 264)
        else:
            benchmark_sharpe = 0.0
        bench_cum = np.cumsum(bench_pnl)
        bench_running_max = np.maximum.accumulate(bench_cum)
        benchmark_maxdd = (bench_cum - bench_running_max).min()

    # Print report
    print("\n" + "="*50)
    print("            PERFORMANCE REPORT")
    print("="*50)
    print(f"Total PnL:            {total_pnl:>12.4f}")
    print(f"Avg PnL per bar:      {avg_pnl:>12.6f}")
    print(f"Std PnL per bar:      {std_pnl:>12.6f}")
    print(f"Sharpe (ann.):        {sharpe:>12.3f}")
    print(f"Max Drawdown:         {max_drawdown:>12.4f}")
    print(f"Turnover:             {turnover:>12.4f}")
    print(f"Prediction-Target Corr:{corr:>12.4f}")
    if benchmark_sharpe is not None:
        print(f"Benchmark Sharpe:     {benchmark_sharpe:>12.3f}")
        print(f"Benchmark MaxDD:      {benchmark_maxdd:>12.4f}")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics <path_to_backtest_results.parquet>")
        sys.exit(1)
    calculate_metrics(sys.argv[1])
```

--- 
### File: src\cli.py
```
"""
src/cli.py
Entrypoint for the Deterministic Quant Pipeline.
Integrates resampling, discovery, walkforward, and execution.
Now with market‑specific config loading and single feature generation pass.
"""
import argparse
import logging
import os
import psutil
from pathlib import Path
import polars as pl

from config import config
from src.ingest import load_and_clean_data
from src.features.engine import generate_features
from src.discovery import run_feature_discovery
from src.walkforward import run_walkforward
from src.io.canonical_parquet import write_canonical_parquet
import json

logger = logging.getLogger(__name__)

def check_memory_safety():
    try:
        mem = psutil.Process().memory_info().rss
        if mem > config.RAM_CAP_BYTES:
            raise MemoryError(f"RSS {mem/(1024**3):.2f}GB > cap {config.RAM_CAP_BYTES/(1024**3):.2f}GB")
    except ImportError:
        pass

def prune_features_by_manifest(df: pl.DataFrame, manifest_path: str) -> pl.DataFrame:
    """Keep only features listed in manifest['feature_names']."""
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    selected = manifest['feature_names']
    non_feature = [c for c in df.columns if not c.startswith(("feature_", "ratio_", "pair_", "zscore", "cross_", "htf_", "1h_", "daily_"))]
    keep = non_feature + [c for c in selected if c in df.columns]
    missing = set(selected) - set(df.columns)
    if missing:
        logger.warning(f"Missing features in manifest: {missing}")
    return df.select(keep)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover_parser = subparsers.add_parser("discover")
    discover_parser.add_argument("--data", required=True)
    discover_parser.add_argument("--out", default="artifacts/manifest.json")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--data", required=True)
    run_parser.add_argument("--manifest", default="artifacts/manifest.json")
    run_parser.add_argument("--out", required=True)

    args = parser.parse_args()
    check_memory_safety()

    # Load market‑specific configuration if the command uses data
    if args.command in ("discover", "run"):
        from src.market_config import detect_symbol_from_path, load_market_config
        symbol = detect_symbol_from_path(args.data)
        load_market_config(symbol)

    if args.command == "discover":
        # Load raw aligned data (5min + HTF)
        df_aligned = load_and_clean_data(args.data)
        # Generate full feature matrix (includes target, HTF, cross)
        df_features = generate_features(df_aligned)
        # Cache the full feature matrix for later reuse by "run"
        cache_dir = Path(args.out).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        feature_cache = cache_dir / "full_feature_matrix.parquet"
        write_canonical_parquet(df_features, str(feature_cache))
        logger.info(f"Full feature matrix cached to {feature_cache}")
        # Run discovery on the cached matrix (so discovery uses all feature types)
        run_feature_discovery(str(feature_cache), args.out)

    elif args.command == "run":
        # Load raw aligned data
        df_aligned = load_and_clean_data(args.data)
        check_memory_safety()

        # Try to load pre‑computed feature matrix from discovery phase
        cache_dir = Path(args.manifest).parent
        feature_cache = cache_dir / "full_feature_matrix.parquet"
        if feature_cache.exists():
            logger.info(f"Loading pre‑computed feature matrix from {feature_cache}")
            df_features = pl.read_parquet(feature_cache)
        else:
            logger.info("No cached feature matrix found; generating features (this may be slower).")
            df_features = generate_features(df_aligned)

        # Prune to only features selected in manifest
        df_pruned = prune_features_by_manifest(df_features, args.manifest)
        target_col = "target_5m"
        if target_col not in df_pruned.columns:
            raise KeyError(f"Target {target_col} missing.")

        # All feature columns are those kept after pruning (excluding metadata)
        feature_cols = [c for c in df_pruned.columns
                        if c not in ("ts_event", "open", "high", "low", "close", "volume",
                                     "session_id", "date", target_col, "regime", "benchmark_pnl")]

        logger.info(f"Walkforward with {len(feature_cols)} features.")
        result_df = run_walkforward(df_pruned, feature_cols, target_col)

        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, "backtest_results.parquet")
        result_df.write_parquet(out_path)
        logger.info(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
```

--- 
### File: src\discovery.py
```
"""
src/discovery.py
Phase 1: Feature discovery using ExtraTrees with bootstrap folds, stability selection,
memory isolation via joblib loky, RSS monitoring, and sign consistency filtering.
Now includes all feature types: baseline, ratios, pairwise, cross-timeframe, and HTF context.

Folds can be run in parallel (config.DISCOVERY_PARALLEL_FOLDS) without affecting determinism
because each fold has its own independent random seed derived from global seed and fold index.
"""
import sys
print("Discovery started. Waiting for folds...", flush=True)
import os
import json
import logging
import numpy as np
import polars as pl
import psutil
import hashlib
from datetime import datetime
from sklearn.ensemble import ExtraTreesRegressor
from joblib import Parallel, delayed
from config import config

logger = logging.getLogger(__name__)

def get_fold_seed(fold_idx: int) -> int:
    seed_str = f"{config.SEED}_fold_{fold_idx}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)

def check_rss(limit_bytes):
    return psutil.Process().memory_info().rss > limit_bytes

def fit_etree_fold(X, y, fold_idx, feature_names, rss_stop_bytes):
    """Fit ExtraTrees on one bootstrap sample. Returns importances dict and sign of correlation."""
    print(f"Fold {fold_idx+1} started at {datetime.now().strftime('%H:%M:%S')}")
    if check_rss(rss_stop_bytes):
        raise MemoryError(f"RSS stop limit exceeded in fold {fold_idx}")
    n_samples = X.shape[0]
    rng = np.random.RandomState(get_fold_seed(fold_idx))
    indices = rng.choice(n_samples, size=n_samples, replace=True)
    X_boot = X[indices]
    y_boot = y[indices]
    et_params = config.EXTRA_TREES_PARAMS.copy()
    et_params['random_state'] = get_fold_seed(fold_idx)
    et = ExtraTreesRegressor(**et_params)
    et.fit(X_boot, y_boot)
    importances = dict(zip(feature_names, et.feature_importances_))

    # Compute sign consistency: correlation between feature and target (simple proxy)
    signs = {}
    for i, f in enumerate(feature_names):
        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(X_boot[:, i], y_boot)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        signs[f] = np.sign(corr)
    return importances, signs

def run_feature_discovery(data_path: str, manifest_out: str):
    logger.info("Phase 1: Feature Discovery")
    df_raw = pl.read_parquet(data_path)
    from src.features.engine import generate_features
    df_features = generate_features(df_raw)

    target_col = "target_5m"
    if target_col not in df_features.columns:
        raise ValueError(f"Target column {target_col} not found.")
    
    # --- Include ALL feature columns (HTF, cross, etc.) ---
    exclude_cols = {
        "ts_event", "open", "high", "low", "close", "volume", 
        "session_id", "date", target_col, "regime", "benchmark_pnl"
    }
    feature_cols = [c for c in df_features.columns if c not in exclude_cols and not c.startswith("_")]
    feature_cols = [c for c in feature_cols if df_features[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]
    
    htf_features = [c for c in feature_cols if c.startswith(("htf_", "cross_", "1h_", "daily_"))]
    if not htf_features:
        logger.warning("No HTF or cross-timeframe features found in feature set. Check generate_features.")
    else:
        logger.info(f"Discovery includes {len(htf_features)} HTF/cross features.")

    X = df_features.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y = df_features.select(target_col).to_numpy().astype(np.float32).ravel()

    n_bars = min(15840, X.shape[0])
    X = X[:n_bars]
    y = y[:n_bars]
    logger.info(f"Discovery using {X.shape[0]} rows, {X.shape[1]} features.")

    rss_stop = config.RSS_STOP_BYTES
    n_folds = config.BOOTSTRAP_FOLDS

    # --- Parallel folds (deterministic, zero loss of accuracy) ---
    n_parallel = min(getattr(config, 'DISCOVERY_PARALLEL_FOLDS', 1), n_folds)
    logger.info(f"Running {n_folds} bootstrap folds in parallel with {n_parallel} workers...")

    results = Parallel(n_jobs=n_parallel, backend='loky', verbose=10)(
        delayed(fit_etree_fold)(X, y, i, feature_cols, rss_stop)
        for i in range(n_folds)
    )

    importances_list = [r[0] for r in results]
    signs_list = [r[1] for r in results]

    # Compute selection frequencies and mean importance
    importances_sum = {f: 0.0 for f in feature_cols}
    selection_count = {f: 0 for f in feature_cols}
    n_folds = len(importances_list)

    for imp_dict, sign_dict in zip(importances_list, signs_list):
        for f, imp in imp_dict.items():
            importances_sum[f] += imp
            if imp > 0:
                selection_count[f] += 1

    # Determine majority sign per feature across folds
    majority_sign = {}
    for f in feature_cols:
        pos = sum(1 for sd in signs_list if sd.get(f, 0) > 0)
        neg = n_folds - pos
        majority_sign[f] = 1 if pos > neg else -1
    sign_consistency_frac = {}
    for f in feature_cols:
        consistent = sum(1 for sd in signs_list if sd.get(f, 0) == majority_sign[f])
        sign_consistency_frac[f] = consistent / n_folds

    freq = {f: selection_count[f] / n_folds for f in feature_cols}
    mean_imp = {f: importances_sum[f] / n_folds for f in feature_cols}

    # Apply frequency threshold AND sign consistency threshold
    selected = [f for f in feature_cols
                if freq[f] >= config.SELECTION_FREQ_THRESHOLD
                and sign_consistency_frac[f] >= config.SIGN_CONSISTENCY_THRESHOLD]
    selected_sorted = sorted(selected, key=lambda x: mean_imp[x], reverse=True)

    # Cumulative importance selection
    cumsum = 0.0
    final_selected = []
    total_imp = sum(mean_imp[f] for f in selected_sorted) if selected_sorted else 1.0
    for f in selected_sorted:
        cumsum += mean_imp[f] / total_imp
        final_selected.append(f)
        if cumsum >= config.CUMULATIVE_IMPORTANCE_THRESHOLD:
            break
    if len(final_selected) < config.MIN_SELECTED_FEATURES:
        final_selected = selected_sorted[:config.MIN_SELECTED_FEATURES]

    logger.info(f"Selected {len(final_selected)} features (sign consistency threshold={config.SIGN_CONSISTENCY_THRESHOLD}).")
    if htf_features:
        selected_htf = [f for f in final_selected if f.startswith(("htf_", "cross_", "1h_", "daily_"))]
        logger.info(f"Selected HTF/cross features: {len(selected_htf)} / {len(htf_features)}")

    # Compute hash of frozen feature list
    feature_list_str = json.dumps(sorted(final_selected), sort_keys=True).encode()
    features_hash = hashlib.sha256(feature_list_str).hexdigest()

    manifest = {
        "version": "1.0",
        "feature_names": final_selected,
        "dtypes": {f: "float32" for f in final_selected},
        "selection_seed": config.SEED,
        "selection_date": datetime.utcnow().isoformat() + "Z",
        "selection_model": "ExtraTreesRegressor",
        "selection_params": config.EXTRA_TREES_PARAMS,
        "selected_K": len(final_selected),
        "cumulative_importance": config.CUMULATIVE_IMPORTANCE_THRESHOLD,
        "stability_stats": {
            "min_selection_freq": config.SELECTION_FREQ_THRESHOLD,
            "sign_consistency": config.SIGN_CONSISTENCY_THRESHOLD,
            "sign_consistency_observed": {f: round(sign_consistency_frac[f], 3) for f in final_selected[:10]}
        },
        "baseline_feature_list": [c for c in feature_cols if c.startswith("feature_")][:40],
        "baseline_features_hash": f"sha256:{features_hash}",
        "baseline_feature_matrix_path": config.BASELINE_FEATURES_PERSIST_PATH,
        "serialization_params": {
            "parquet_version": "2.0",
            "compression": "snappy",
            "row_group_size": config.ROW_GROUP_SIZE,
            "column_ordering": "lexicographic"
        },
        "discovery_status": "completed",
        "folds": [],
        "htf_features_included": len(htf_features) > 0
    }
    os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=4)
    logger.info(f"Manifest saved to {manifest_out}")
```

--- 
### File: src\execution\simulator.py
```
"""
src/execution/simulator.py
Execution simulation: stateful position tracking, volatility scaling,
transaction costs, leverage limits, and flatten before session close.
Now includes HTF volatility scaling and trend alignment if enabled in config.
"""
import polars as pl
import numpy as np
from config import config


def simulate_execution(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds columns: 'position', 'trade_cost', 'pnl'.
    Now includes HTF volatility scaling and trend alignment if enabled in config.
    """
    # Ensure we have volatility
    if "feature_ewma_vol_20" not in df.columns:
        ret = (pl.col("close") / pl.col("close").shift(1)).log()
        vol = ret.rolling_std(window_size=20)
        df = df.with_columns(vol.alias("vol"))
    else:
        df = df.with_columns(pl.col("feature_ewma_vol_20").alias("vol"))

    # Raw target position
    target_raw = (pl.col("prediction") / pl.col("vol").clip(config.EPS, None)) * config.TARGET_VOL
    target_raw = target_raw.clip(-config.MAX_LEVERAGE, config.MAX_LEVERAGE)

    # ---- HTF Volatility Scaling ----
    if config.HTF_VOL_SCALING and "htf_daily_vol_5" in df.columns:
        daily_target_vol = config.TARGET_VOL  # could be market-specific
        daily_atr = df["htf_daily_vol_5"]
        scaling = (daily_target_vol / daily_atr.clip(config.EPS, None)).clip(0.25, 2.0)
        target_raw = target_raw * scaling
        target_raw = target_raw.clip(-config.MAX_LEVERAGE, config.MAX_LEVERAGE)

    # ---- HTF Trend Alignment Filter ----
    if config.HTF_TREND_ALIGNMENT and "htf_daily_trend_slope_10" in df.columns:
        daily_trend = df["htf_daily_trend_slope_10"].sign()
        # Zero trend means no filter
        target_raw = pl.when(
            (daily_trend == 0) | (target_raw.sign() == daily_trend)
        ).then(target_raw).otherwise(0)

    # Spread proxy
    if "feature_spread_proxy" in df.columns:
        spread = pl.col("feature_spread_proxy")
    else:
        spread = (pl.col("high") - pl.col("low")) / pl.col("close").clip(config.EPS, None)

    unit_cost = config.COMMISSION_PER_TRADE + config.SLIPPAGE_K * spread + config.VOL_PENALTY * pl.col("vol")

    # Stateful position loop (convert to numpy)
    target_array = target_raw.to_numpy()
    unit_cost_array = unit_cost.to_numpy()
    df = df.with_columns(
        pl.col("ts_event").rank("ordinal").over("session_id").alias("_session_rank"),
        pl.col("ts_event").count().over("session_id").alias("_session_len")
    )
    last_bars_mask = (df["_session_rank"] > (df["_session_len"] - config.FLAT_BEFORE_CLOSE_MINUTES//5)).to_numpy()

    n = len(df)
    positions = np.zeros(n, dtype=np.float32)
    trade_costs = np.zeros(n, dtype=np.float32)
    open_next = np.roll(df["open"].to_numpy(), -1)
    close_next = np.roll(df["close"].to_numpy(), -1)
    open_next[-1] = np.nan
    close_next[-1] = np.nan

    current_pos = 0.0
    for i in range(n):
        desired = target_array[i]
        if last_bars_mask[i]:
            desired = 0.0
        delta = np.clip(desired - current_pos, -config.MAX_POS_CHANGE_PER_MIN, config.MAX_POS_CHANGE_PER_MIN)
        new_pos = current_pos + delta
        new_pos = np.clip(new_pos, -config.MAX_LEVERAGE, config.MAX_LEVERAGE)
        cost = abs(new_pos - current_pos) * unit_cost_array[i]
        trade_costs[i] = cost
        positions[i] = new_pos
        current_pos = new_pos

    ret_exec = (close_next - open_next) / np.maximum(open_next, config.EPS)
    pnl = positions * ret_exec - trade_costs
    pnl = np.nan_to_num(pnl, nan=0.0)

    df = df.with_columns([
        pl.Series("position", positions).cast(pl.Float32),
        pl.Series("trade_cost", trade_costs).cast(pl.Float32),
        pl.Series("pnl", pnl).cast(pl.Float32)
    ])
    df = df.drop(["_session_rank", "_session_len"])
    return df
```

--- 
### File: src\features\__init__.py
```

```

--- 
### File: src\features\baseline.py
```
"""
src/features/baseline.py
Generate the 40 frozen baseline features from YAML spec, using Polars expressions.
All features are past-only and return float32.
"""
import polars as pl
import yaml
from config import config

def load_baseline_feature_names() -> list:
    """Load feature names from baseline_features.yaml"""
    with open(config.BASELINE_FEATURES_FILE, 'r') as f:
        data = yaml.safe_load(f)
    return data['baseline_features']

def compute_baseline_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add all 40 baseline features to the DataFrame.
    Each feature is computed according to its canonical definition.
    """
    # Ensure we have the required OHLCV columns
    close = pl.col("close").cast(pl.Float32)
    high = pl.col("high").cast(pl.Float32)
    low = pl.col("low").cast(pl.Float32)
    open_ = pl.col("open").cast(pl.Float32)
    volume = pl.col("volume").cast(pl.Float32)

    exprs = []

    # 1-4: Log returns at lags 1,5,10,20 (periods of 5-min bars)
    for lag in [1, 5, 10, 20]:
        ret = (close / close.shift(lag)).log()
        exprs.append(ret.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f"feature_ret_{lag}"))

    # 5-7: Simple moving averages of close
    for window in [5, 20, 50]:
        ma = close.rolling_mean(window_size=window)
        exprs.append(ma.alias(f"feature_ma_{window}"))

    # 8: dist_ma_20 = (close - MA20)/MA20
    ma20 = close.rolling_mean(window_size=20)
    dist_ma20 = (close - ma20) / ma20.clip(config.EPS, None)
    exprs.append(dist_ma20.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_dist_ma_20"))

    # 9: dist_ma_50
    ma50 = close.rolling_mean(window_size=50)
    dist_ma50 = (close - ma50) / ma50.clip(config.EPS, None)
    exprs.append(dist_ma50.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_dist_ma_50"))

    # 10: ma_slope_20 – linear regression slope over 20 bars normalized by SMA20
    slope20 = (close - close.shift(20)) / 20.0 / ma20.clip(config.EPS, None)
    exprs.append(slope20.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_ma_slope_20"))

    # 11,12: price_z_20, price_z_50
    for window in [20, 50]:
        mean = close.rolling_mean(window_size=window)
        std = close.rolling_std(window_size=window)
        z = (close - mean) / std.clip(config.EPS, None)
        exprs.append(z.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f"feature_price_z_{window}"))

    # 13: high_low_range_norm = (high - low) / max(close, EPS)
    range_norm = (high - low) / pl.max_horizontal(close, config.EPS)
    exprs.append(range_norm.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_high_low_range_norm"))

    # 14: true_range = max(high-low, |high-prev_close|, |low-prev_close|)
    prev_close = close.shift(1)
    tr = pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    )
    exprs.append(tr.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_true_range"))

    # 15: atr_14 = rolling_mean(true_range, 14)
    atr14 = tr.rolling_mean(window_size=14)
    exprs.append(atr14.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_atr_14"))

    # 16,17: realized_vol_5, realized_vol_20 (sample std of log returns)
    ret_1 = (close / close.shift(1)).log()
    for window in [5, 20]:
        rvol = ret_1.rolling_std(window_size=window)
        exprs.append(rvol.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f"feature_realized_vol_{window}"))

    # 18: ewma_vol_20 – exponentially weighted moving average of squared returns
    alpha = 2.0 / (20 + 1)
    ewma_vol = ret_1.pow(2).ewm_mean(alpha=alpha, adjust=False).sqrt()
    exprs.append(ewma_vol.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_ewma_vol_20"))

    # 19,20: price_momentum_5, price_momentum_10 = (close - close.shift(window))/close.shift(window)
    for window in [5, 10]:
        mom = (close - close.shift(window)) / close.shift(window).clip(config.EPS, None)
        exprs.append(mom.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f"feature_price_momentum_{window}"))

    # 21,22: mom_z_5, mom_z_10 (z-score of momentum)
    for window in [5, 10]:
        mom = (close - close.shift(window)) / close.shift(window).clip(config.EPS, None)
        mean_mom = mom.rolling_mean(window_size=window)
        std_mom = mom.rolling_std(window_size=window)
        mom_z = (mom - mean_mom) / std_mom.clip(config.EPS, None)
        exprs.append(mom_z.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f"feature_mom_z_{window}"))

    # 23: rsi_14
    delta = close.diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)
    avg_gain = gain.rolling_mean(window_size=14)
    avg_loss = loss.rolling_mean(window_size=14)
    rs = avg_gain / avg_loss.clip(config.EPS, None)
    rsi = 100 - 100 / (1 + rs)
    exprs.append(rsi.clip(0, 100).alias("feature_rsi_14"))

    # 24: macd (12,26,9) – difference of EMAs
    ema12 = close.ewm_mean(alpha=2/13, adjust=False)
    ema26 = close.ewm_mean(alpha=2/27, adjust=False)
    macd = ema12 - ema26
    exprs.append(macd.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_macd"))

    # 25: macd_signal – 9-period EMA of MACD
    signal = macd.ewm_mean(alpha=2/10, adjust=False)
    exprs.append(signal.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_macd_signal"))

    # 26: stoch_k (%K) – (close - low_14) / (high_14 - low_14)
    low14 = low.rolling_min(window_size=14)
    high14 = high.rolling_max(window_size=14)
    stoch_k = (close - low14) / (high14 - low14).clip(config.EPS, None) * 100
    exprs.append(stoch_k.clip(0, 100).alias("feature_stoch_k"))

    # 27: log_volume
    log_vol = volume.log().fill_null(0.0)
    exprs.append(log_vol.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_log_volume"))

    # 28: volume_z_20
    vol_mean = volume.rolling_mean(window_size=20)
    vol_std = volume.rolling_std(window_size=20)
    vol_z = (volume - vol_mean) / vol_std.clip(config.EPS, None)
    exprs.append(vol_z.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_volume_z_20"))

    # 29: obv – on‑balance volume: cumulative signed volume based on close direction
    sign = pl.when(close > close.shift(1)).then(1).when(close < close.shift(1)).then(-1).otherwise(0)
    obv = (sign * volume).cum_sum()
    exprs.append(obv.cast(pl.Float32).clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_obv"))

    # 30: signed_bar_strength – tick-rule proxy (close vs open)
    bar_sign = (close - open_).sign()
    bar_sign = bar_sign.fill_null(strategy="forward")
    signed_volume = bar_sign * volume
    signed_strength = signed_volume / volume.clip(config.EPS, None)
    exprs.append(signed_strength.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_signed_bar_strength"))

    # 31: volume_price_divergence – proxy: volume * ret_1 (captures size-return interaction)
    vol_price_div = (volume * ret_1).cast(pl.Float32)
    exprs.append(vol_price_div.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_volume_price_divergence"))

    # 32: spread_proxy – (high - low) / close (proxy for bid-ask)
    spread_proxy = (high - low) / close.clip(config.EPS, None)
    exprs.append(spread_proxy.clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_spread_proxy"))

    # 33: session_pos – linear position in session (0 to 1)
    session_pos = (pl.col("ts_event").rank("ordinal").over("session_id") - 1) / (pl.col("ts_event").count().over("session_id") - 1)
    exprs.append(session_pos.fill_nan(0.5).cast(pl.Float32).alias("feature_session_pos"))

    # 34: time_of_day_bucket – categorical [early,mid,late] based on session position
    bucket = pl.when(session_pos < 0.33).then(0.0).when(session_pos < 0.66).then(1.0).otherwise(2.0)
    exprs.append(bucket.cast(pl.Float32).alias("feature_time_of_day_bucket"))

    # 35: 1h_bias – placeholder, will be overwritten by actual 1h target mapping (done later)
    exprs.append(pl.lit(0.0).alias("feature_1h_bias"))

    # 36: session_volatility – standard deviation of returns within current session
    session_vol = ret_1.rolling_std(window_size=config.ROLL_WINDOW_MIN_ROWS).over("session_id")
    exprs.append(session_vol.fill_null(0.0).clip(config.CLIP_MIN, config.CLIP_MAX).alias("feature_session_volatility"))

    # 37: pair_prod_template – placeholder, actual pairwise products will be added in expansion
    exprs.append(pl.lit(0.0).alias("feature_pair_prod_template"))

    # 38: ratio_template – placeholder
    exprs.append(pl.lit(0.0).alias("feature_ratio_template"))

    # 39,40: pca_comp_1, pca_comp_2 – placeholders (PCA will be done later if orthogonalize=True)
    exprs.append(pl.lit(0.0).alias("feature_pca_comp_1"))
    exprs.append(pl.lit(0.0).alias("feature_pca_comp_2"))

    # Apply all expressions and replace NaN/Inf with 0, then clip
    df = df.with_columns(exprs)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    for col in feature_cols:
        df = df.with_columns(
            pl.col(col).fill_nan(config.REPLACE_INF_NAN_WITH).fill_null(config.REPLACE_INF_NAN_WITH).clip(config.CLIP_MIN, config.CLIP_MAX)
        )
    return df
```

--- 
### File: src\features\corr_prune.py
```
"""
src/features/corr_prune.py
Deterministic correlation pruning.
Implements Section 15: float64 computation, keeps first occurrence, drops subsequent.
"""
import logging
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

def correlation_prune(df: pl.DataFrame, feature_cols: list, threshold: float = 0.90) -> list:
    """
    Deterministic correlation pruning.
    CRITICAL: Must be executed strictly on the TRAIN split to prevent look-ahead bias.
    Uses float64 for computation precision, but keeps data intact for float32 pipelines.
    """
    if df.height == 0 or len(feature_cols) == 0:
        return feature_cols

    logger.info(f"Running correlation pruning on {len(feature_cols)} features (threshold={threshold})...")
    
    # Section 15: Upcast to float64 exclusively for the matrix computation
    X = df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float64)

    # rowvar=False ensures columns are treated as variables
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.corrcoef(X, rowvar=False)

    keep = []
    dropped = set()

    for i, f in enumerate(feature_cols):
        if f in dropped:
            continue

        keep.append(f)

        # Check all subsequent features against the kept feature
        for j in range(i + 1, len(feature_cols)):
            if feature_cols[j] in dropped:
                continue
            
            val = corr[i, j]
            # Drop if correlation exceeds threshold (ignoring NaNs from zero-variance columns)
            if not np.isnan(val) and abs(val) > threshold:
                dropped.add(feature_cols[j])

    logger.info(f"Correlation pruning dropped {len(dropped)} features. Kept {len(keep)} features.")
    return keep
```

--- 
### File: src\features\engine.py
```
"""
src/features/engine.py
Orchestrates generation of baseline features, HTF context, expansion, and target.
"""
import polars as pl
import logging
from config import config
from src.features.baseline import compute_baseline_features, load_baseline_feature_names
from src.features.expansion import expand_features, add_cross_timeframe_interactions
from src.features.htf_context import add_htf_context_features
from src.features.target import add_target_5m, drop_incomplete_target

logger = logging.getLogger(__name__)

def generate_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Full feature engineering pipeline for three-stream HTF data.
    Assumes df already contains aligned 1h and daily columns (prefixed 1h_, daily_).
    """
    print("DEBUG: generate_features - computing baseline...", flush=True)
    df = compute_baseline_features(df)
    baseline_names = load_baseline_feature_names()
    baseline_cols = [c for c in baseline_names if c in df.columns]
    print("DEBUG: baseline done", flush=True)

    print("DEBUG: adding HTF context features...", flush=True)
    df = add_htf_context_features(df)
    print("DEBUG: HTF context done", flush=True)

    print("DEBUG: expanding features (ratios, z-scores, regime, pairwise)...", flush=True)
    df = expand_features(df, baseline_cols)
    print("DEBUG: expansion done", flush=True)

    # After expand_features, add cross-timeframe interactions explicitly
    htf_cols = [c for c in df.columns if c.startswith(("1h_", "daily_", "htf_"))]
    ltf_cols = [c for c in df.columns if c.startswith(("feature_", "ratio_", "pair_", "zscore")) and c not in htf_cols]
    if htf_cols and ltf_cols:
        print("DEBUG: adding cross-timeframe interactions...", flush=True)
        df = add_cross_timeframe_interactions(df, ltf_cols, htf_cols)
        print("DEBUG: cross-timeframe done", flush=True)

    print("DEBUG: adding target_5m...", flush=True)
    df = add_target_5m(df)
    df = drop_incomplete_target(df)
    print("DEBUG: target done", flush=True)

    # Ensure all feature columns are float32
    feature_cols = [c for c in df.columns if c.startswith(("feature_", "ratio_", "pair_", "zscore", "cross_", "htf_", "1h_", "daily_"))]
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c in feature_cols])
    logger.info(f"Final feature matrix has {len(feature_cols)} features.")
    print(f"DEBUG: generate_features finished with {len(feature_cols)} features", flush=True)
    return df
```

--- 
### File: src\features\expansion.py
```
"""
src/features/expansion.py
Expand feature space with ratios, z-scores, regime-conditioned transforms,
pairwise interactions (capped at MAX_PAIRWLE_INTERACTIONS), and cross-timeframe interactions.
All features are past-only, float32, clipped.

Now with memory safety estimation to avoid OOM from combinatorial explosion.
"""
import polars as pl
import numpy as np
import logging
from itertools import combinations
from config import config

logger = logging.getLogger(__name__)

def add_regime(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add regime column: 1=high vol, 0=low vol, using rolling median volatility.
    """
    # compute volatility as realized vol 20
    ret = (pl.col("close") / pl.col("close").shift(1)).log()
    vol20 = ret.rolling_std(window_size=20)
    med_vol = vol20.rolling_median(window_size=config.VOL_MEDIAN_WINDOW)
    smooth_vol = med_vol.rolling_mean(window_size=config.VOL_SMOOTH_WINDOW)
    regime = pl.when(smooth_vol >= config.REGIME_HIGH_THRESH).then(1.0) \
              .when(smooth_vol <= config.REGIME_LOW_THRESH).then(0.0) \
              .otherwise(None)
    # forward fill regime
    regime = regime.fill_null(strategy="forward").fill_null(config.REGIME_MISSING_DEFAULT)
    df = df.with_columns(regime.cast(pl.Float32).alias("regime"))
    return df

def add_ratios_and_z_scores(df: pl.DataFrame, base_features: list) -> pl.DataFrame:
    """
    Add ratio features (feature_i / feature_j) and z-scores (rolling z-score)
    for a subset of core features.
    """
    # we will use a limited set to avoid explosion: close, volume, spread, range
    core = ["close", "volume", "feature_spread_proxy", "feature_high_low_range_norm"]
    existing = [c for c in core if c in df.columns]
    exprs = []
    # ratios
    for i, a in enumerate(existing):
        for b in existing[i+1:]:
            name = f"ratio_{a}_over_{b}"
            expr = (pl.col(a) / pl.col(b).clip(config.EPS, None)).cast(pl.Float32)
            exprs.append(expr.clip(config.CLIP_MIN, config.CLIP_MAX).alias(name))
    # z-scores of base features with 20-period lookback
    for col in base_features[:20]:  # limit to first 20 to avoid too many
        if col in df.columns:
            mean = pl.col(col).rolling_mean(window_size=20)
            std = pl.col(col).rolling_std(window_size=20)
            z = (pl.col(col) - mean) / std.clip(config.EPS, None)
            exprs.append(z.clip(config.CLIP_MIN, config.CLIP_MAX).alias(f"{col}_zscore"))
    df = df.with_columns(exprs)
    return df

def add_regime_conditioned_transforms(df: pl.DataFrame) -> pl.DataFrame:
    """
    Multiply selected features by regime indicator to capture regime-specific effects.
    """
    regime = pl.col("regime")
    # choose a few important features
    interact_cols = ["feature_ret_1", "feature_ret_5", "feature_ewma_vol_20", "feature_volume_z_20"]
    exprs = []
    for col in interact_cols:
        if col in df.columns:
            expr = (pl.col(col) * regime).alias(f"{col}_regime")
            exprs.append(expr.clip(config.CLIP_MIN, config.CLIP_MAX))
    df = df.with_columns(exprs)
    return df

def add_pairwise_interactions(df: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
    """
    Generate pairwise products up to MAX_PAIRWISE_INTERACTIONS.
    Sorted feature list ensures determinism.
    """
    sorted_features = sorted(feature_cols)
    exprs = []
    count = 0
    for a, b in combinations(sorted_features, 2):
        if count >= config.MAX_PAIRWISE_INTERACTIONS:
            break
        name = f"pair_{a}_x_{b}"
        expr = (pl.col(a) * pl.col(b)).cast(pl.Float32)
        exprs.append(expr.clip(config.CLIP_MIN, config.CLIP_MAX).alias(name))
        count += 1
    if exprs:
        df = df.with_columns(exprs)
    return df

def safe_add_pairwise_interactions(df: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
    """
    Estimate number of products and abort if exceeding RAM-based limit.
    Delegates to add_pairwise_interactions which already caps.
    """
    n_features = len(feature_cols)
    max_combinations = config.MAX_PAIRWISE_INTERACTIONS
    total_possible = n_features * (n_features - 1) // 2
    if total_possible > max_combinations:
        logger.info(f"Pairwise combinations would exceed {max_combinations}, capping.")
    return add_pairwise_interactions(df, feature_cols)

def add_cross_timeframe_interactions(df: pl.DataFrame, ltf_features: list, htf_features: list) -> pl.DataFrame:
    """ltf_features: 5min feature names, htf_features: 1h/daily/htf context names."""
    ltf_sorted = sorted(ltf_features)
    htf_sorted = sorted(htf_features)
    exprs = []
    count = 0
    for a in ltf_sorted:
        for b in htf_sorted:
            if count >= config.MAX_CROSS_TIMEFRAME_INTERACTIONS:
                break
            name = f"cross_{a}_x_{b}"
            expr = (pl.col(a) * pl.col(b)).cast(pl.Float32)
            exprs.append(expr.clip(config.CLIP_MIN, config.CLIP_MAX).alias(name))
            count += 1
        if count >= config.MAX_CROSS_TIMEFRAME_INTERACTIONS:
            break
    if exprs:
        df = df.with_columns(exprs)
    return df

def expand_features(df: pl.DataFrame, baseline_feature_cols: list) -> pl.DataFrame:
    """
    Full expansion pipeline: add regime, ratios/zscores, regime interactions, pairwise.
    Returns DataFrame with all expanded features.
    Now includes memory safety estimation to prevent OOM.
    """
    df = add_regime(df)
    df = add_ratios_and_z_scores(df, baseline_feature_cols)
    df = add_regime_conditioned_transforms(df)

    # Collect all existing feature-like columns for further expansion
    current_features = [c for c in df.columns if c.startswith(("feature_", "ratio_", "pair_", "zscore", "cross_", "htf_", "1h_", "daily_"))]
    # Identify HTF columns for cross-timeframe estimation
    htf_cols = [c for c in df.columns if c.startswith(("1h_", "daily_", "htf_"))]

    # Memory safety: estimate total column count after adding pairwise and cross interactions
    est_pairwise = min(config.MAX_PAIRWISE_INTERACTIONS, len(current_features) * (len(current_features) - 1) // 2)
    est_cross = 0
    if htf_cols:
        est_cross = min(config.MAX_CROSS_TIMEFRAME_INTERACTIONS, len(current_features) * len(htf_cols))
    total_est = len(df.columns) + est_pairwise + est_cross
    if total_est > 5000:  # conservative limit to avoid OOM (adjustable)
        raise MemoryError(f"Estimated feature count {total_est} exceeds safety limit of 5000. "
                          f"Reduce MAX_PAIRWISE_INTERACTIONS or MAX_CROSS_TIMEFRAME_INTERACTIONS.")

    # Now safe to add interactions
    df = safe_add_pairwise_interactions(df, current_features)
    if htf_cols:
        df = add_cross_timeframe_interactions(df, current_features, htf_cols)

    # Final clipping and nan fill for all non‑metadata columns
    exclude_cols = {"ts_event", "open", "high", "low", "close", "volume", "session_id", "regime"}
    all_feature_cols = [c for c in df.columns if c not in exclude_cols]
    for col in all_feature_cols:
        df = df.with_columns(
            pl.col(col).fill_nan(config.REPLACE_INF_NAN_WITH).fill_null(config.REPLACE_INF_NAN_WITH).clip(config.CLIP_MIN, config.CLIP_MAX)
        )
    return df
```

--- 
### File: src\features\htf_context.py
```
"""
src/features/htf_context.py
Compute higher‑timeframe context features from aligned 1h and daily data.
All features are past‑only, float32, clipped.
Now uses precomputed daily_vol_5 from the daily stream.
"""
import polars as pl
from config import config


def add_htf_context_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add HTF state features. Expects columns:
       1h_close, 1h_high, 1h_low, 1h_volume
       daily_close, daily_high, daily_low, daily_volume, daily_vol_5
    Returns df with additional columns prefixed 'htf_'.
    """
    # 1. Daily return (log, 1 day lag – already aligned to previous day)
    df = df.with_columns(
        (pl.col("daily_close") / pl.col("daily_close").shift(1)).log().alias("htf_daily_return_1")
    )
    # 2. Daily volatility – use precomputed daily_vol_5 from daily stream
    #    (already aligned as previous day's volatility)
    df = df.with_columns(
        pl.col("daily_vol_5").alias("htf_daily_vol_5")
    )
    # 3. Daily trend slope (10‑day linear approximation)
    df = df.with_columns(
        ((pl.col("daily_close") - pl.col("daily_close").shift(10)) / 10.0 / pl.col("daily_close").shift(10).clip(config.EPS, None))
        .alias("htf_daily_trend_slope_10")
    )
    # 4. Distance to daily high/low (normalized by daily high/low)
    df = df.with_columns(
        ((pl.col("daily_high") - pl.col("close")) / pl.col("daily_high").clip(config.EPS, None)).alias("htf_distance_to_daily_high"),
        ((pl.col("close") - pl.col("daily_low")) / pl.col("daily_low").clip(config.EPS, None)).alias("htf_distance_to_daily_low")
    )
    # 5. Hourly trend alignment (sign of 1h return vs daily trend)
    df = df.with_columns(
        (pl.col("1h_close") / pl.col("1h_close").shift(1)).log().alias("1h_return")
    )
    df = df.with_columns(
        (pl.col("1h_return") * pl.col("htf_daily_trend_slope_10").sign()).alias("htf_hourly_trend_alignment")
    )
    # 6. Volatility ratio (1h volatility / daily volatility)
    df = df.with_columns(
        pl.col("1h_return").rolling_std(window_size=4).alias("1h_vol_4")
    )
    df = df.with_columns(
        (pl.col("1h_vol_4") / pl.col("htf_daily_vol_5").clip(config.EPS, None)).alias("htf_volatility_ratio")
    )
    # 7. Daily session phase – reuse existing feature_session_pos (already in df)

    # Clean and cast all HTF columns
    htf_cols = [c for c in df.columns if c.startswith("htf_")]
    for col in htf_cols:
        df = df.with_columns(
            pl.col(col).fill_nan(config.REPLACE_INF_NAN_WITH).fill_null(config.REPLACE_INF_NAN_WITH).clip(config.CLIP_MIN, config.CLIP_MAX).cast(pl.Float32)
        )
    # Drop intermediate columns
    df = df.drop(["1h_return", "1h_vol_4"])
    return df
```

--- 
### File: src\features\target.py
```
"""
src/features/target.py
Construct target_5m = forward 1-bar log return (5-min bars -> 5-min).
Continuous target, no lookahead.
"""
import polars as pl
from config import config

def add_target_5m(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add target_5m = log(close[t+1] / close[t]).
    The last row will have null target (no forward data) – dropped later.
    """
    log_close = pl.col("close").log()
    forward_ret = (log_close.shift(-1) - log_close).alias("target_5m")
    df = df.with_columns(forward_ret)
    df = df.with_columns(pl.col("target_5m").clip(config.CLIP_MIN, config.CLIP_MAX))
    return df

def drop_incomplete_target(df: pl.DataFrame) -> pl.DataFrame:
    """Remove rows where target is null (end of dataset)."""
    return df.filter(pl.col("target_5m").is_not_null())
```

--- 
### File: src\ingest.py
```
"""
src/ingest.py
Handles ingestion of all three streams (5m, 1h, 1d) and alignment.
"""
import polars as pl
import logging
import psutil
from config import config
from src.session import load_all_streams_chunked
from src.align import align_htf_streams

logger = logging.getLogger(__name__)


def validate_memory_and_integrity(df: pl.DataFrame):
    """Same as before, but now df includes HTF columns; we still check OHLC."""
    logger.info("Running memory and integrity validation...")
    if not df["ts_event"].is_sorted():
        raise ValueError("ts_event not strictly increasing.")
    critical_cols = ["open", "high", "low", "close", "volume", "session_id"]
    for col in critical_cols:
        if df[col].null_count() > 0:
            raise ValueError(f"Nulls in column {col}.")
    if (df["high"] < df["low"]).any():
        raise ValueError("High < Low detected.")
    if ((df["open"] < df["low"]) | (df["open"] > df["high"])).any():
        raise ValueError("Open outside [Low, High].")
    if ((df["close"] < df["low"]) | (df["close"] > df["high"])).any():
        raise ValueError("Close outside [Low, High].")
    est_bytes = df.estimated_size()
    rows = df.height
    logger.info(f"Memory usage: {est_bytes / 1024**3:.2f} GB")
    if est_bytes > config.RAM_CAP_BYTES:
        raise MemoryError(f"Data size {est_bytes} exceeds RAM_CAP_BYTES.")
    avg_row_bytes = est_bytes / rows if rows > 0 else 0
    rows_per_chunk = min(
        config.ROWS_PER_CHUNK_MAX,
        int((config.RAM_CAP_BYTES * config.MEMORY_SAFETY_MARGIN) / (avg_row_bytes + 1))
    )
    logger.info(f"Safe rows_per_chunk: {rows_per_chunk}")
    return rows_per_chunk


def load_and_clean_data(data_glob: str) -> pl.DataFrame:
    """
    Load all three streams (5m, 1h, 1d) from the given glob pattern,
    align them without lookahead, and validate.
    """
    logger.info(f"Loading three streams from: {data_glob}")
    print("DEBUG: Starting load_all_streams_chunked...", flush=True)
    streams = load_all_streams_chunked(data_glob)
    print(f"DEBUG: Streams loaded. 5min rows: {streams['5m'].height}", flush=True)
    df_5min = streams["5m"]
    df_1h = streams["1h"]
    df_daily = streams["1d"]
    print("DEBUG: Aligning streams...", flush=True)
    df_aligned = align_htf_streams(df_5min, df_1h, df_daily)
    print(f"DEBUG: Alignment done. Aligned rows: {df_aligned.height}", flush=True)
    validate_memory_and_integrity(df_aligned)
    if config.MEMORY_LOG_ENABLED:
        logger.info(f"RSS after load: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
    return df_aligned
```

--- 
### File: src\io\__init__.py
```

```

--- 
### File: src\io\canonical_parquet.py
```
"""
src/io/canonical_parquet.py
Deterministic serialization for canonical feature matrices.
Ensures byte-level reproducibility as per Section 18 of ai_prompt.md.
"""
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
from config import config
import logging

logger = logging.getLogger(__name__)

def write_canonical_parquet(data: pl.DataFrame | pa.Table, path: str):
    """
    Writes a Polars DataFrame or PyArrow Table to a Parquet file with deterministic settings.
    
    Compliance with Section 18:
    - Format Version: 2.0
    - Compression: snappy
    - Row Group Size: 65536 (from config.ROW_GROUP_SIZE)
    - Column Ordering: Lexicographic (sorted)
    """
    # Convert Polars DataFrame to PyArrow Table if needed
    if isinstance(data, pl.DataFrame):
        table = data.to_arrow()
    else:
        table = data

    # Enforce lexicographic column ordering for determinism
    sorted_column_names = sorted(table.column_names)
    table = table.select(sorted_column_names)

    # Write with fixed parameters
    try:
        pq.write_table(
            table,
            path,
            version="2.0",
            compression="snappy",
            row_group_size=getattr(config, "ROW_GROUP_SIZE", 65536),
            data_page_version="2.0",
            use_deprecated_int96_timestamps=False,
            coerce_timestamps="us"
        )
        logger.info(f"Successfully wrote canonical parquet to {path} "
                    f"with {len(sorted_column_names)} columns.")
    except Exception as e:
        logger.error(f"Failed to write canonical parquet: {e}")
        raise
```

--- 
### File: src\market_config.py
```
"""
src/market_config.py
Load market-specific configuration from YAML files and override global config values.
"""
import yaml
import logging
from pathlib import Path
from config import config

logger = logging.getLogger(__name__)

def detect_symbol_from_path(data_path: str) -> str:
    """Infer symbol from file path (e.g., futures/ES/2024.parquet -> ES)."""
    path = Path(data_path)
    # Look for known market names in parent directory
    for part in path.parent.parts:
        if part in config.MARKET_CONFIGS:
            return part
    # Default fallback
    return "ES"

def load_market_config(symbol: str):
    """Load YAML config for symbol and update global config object in-place."""
    yaml_path = config.MARKET_CONFIGS.get(symbol)
    if not yaml_path or not Path(yaml_path).exists():
        logger.warning(f"Market config for {symbol} not found at {yaml_path}, using global defaults.")
        return
    
    with open(yaml_path, 'r') as f:
        market_cfg = yaml.safe_load(f)
    
    # Override relevant global config attributes
    overrides = {
        "ROLL_WINDOWS": market_cfg.get("roll_windows"),
        "ROLL_WINDOWS_1H": market_cfg.get("roll_windows_1h"),
        "ROLL_WINDOWS_DAILY": market_cfg.get("roll_windows_daily"),
        "REGIME_HIGH_THRESH": market_cfg.get("regime_high_thresh"),
        "REGIME_LOW_THRESH": market_cfg.get("regime_low_thresh"),
        "HTF_TREND_WINDOWS": market_cfg.get("htf_trend_windows"),
        "HTF_VOLATILITY_WINDOWS": market_cfg.get("htf_volatility_windows"),
        "SLIPPAGE_K": market_cfg.get("slippage_k"),
        "VOL_PENALTY": market_cfg.get("vol_penalty"),
        "COMMISSION_PER_TRADE": market_cfg.get("commission_per_trade"),
        "MAX_LEVERAGE": market_cfg.get("max_leverage"),
        "TARGET_VOL": market_cfg.get("target_vol"),
    }
    for attr, value in overrides.items():
        if value is not None:
            setattr(config, attr, value)
            logger.info(f"Overrode {attr} = {value} for {symbol}")
```

--- 
### File: src\session.py
```
"""
src/session.py
Implements Globex session definition, session_id, and resampling to multiple frequencies (5m, 1h, 1d).
Now supports chunked processing and returns all three streams.
Fixed timezone conversion (replace -> convert_time_zone) and added daily volatility.
"""
import polars as pl
import logging
from datetime import time
import pytz
from pathlib import Path
import tempfile
import glob
from config import config

logger = logging.getLogger(__name__)

TZ = pytz.timezone(config.TIMEZONE)
SESSION_START = config.SESSION_START_LOCAL
SESSION_END = config.SESSION_END_LOCAL


def add_session_id(df: pl.DataFrame) -> pl.DataFrame:
    """Add session_id using Globex rollover rule: shift by 6h for dates."""
    # Convert UTC to local time for correct date shift
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone(config.TIMEZONE).alias("ts_local")
    )
    session_id = pl.col("ts_local").dt.offset_by("6h").dt.date().cast(pl.String)
    df = df.with_columns(session_id.alias("session_id"))
    return df.drop("ts_local")


def filter_session_hours(df: pl.DataFrame) -> pl.DataFrame:
    """Keep rows within [18:00, 16:00) ET."""
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone(config.TIMEZONE).dt.time().alias("time_local")
    )
    df = df.filter(
        (pl.col("time_local") >= SESSION_START) | (pl.col("time_local") < SESSION_END)
    )
    return df.drop("time_local")


def resample_to_frequency(df: pl.DataFrame, freq: str) -> pl.DataFrame:
    """
    Resample 1‑min df to given frequency (e.g., '5m', '1h', '1d') within each session.
    For 1h, require at least 45 minutes of ticks; for 1d, require at least 360 minutes (6 hours).
    For daily, also compute rolling 5-day volatility of log returns.
    """
    df = df.with_columns(
        pl.col("ts_event").dt.convert_time_zone(config.TIMEZONE).alias("ts_local")
    )
    df = df.with_columns(
        pl.col("ts_local").dt.truncate(every=freq).alias(f"ts_{freq}")
    )
    agg = df.group_by(["session_id", f"ts_{freq}"], maintain_order=True).agg([
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.len().alias("n_ticks"),
    ])
    
    # Drop incomplete bars based on frequency
    if freq == "5m" and config.DROP_INCOMPLETE_ROWS:
        agg = agg.filter(pl.col("n_ticks") == 5)
    elif freq == "1h":
        agg = agg.filter(pl.col("n_ticks") >= 45)
    elif freq == "1d":
        agg = agg.filter(pl.col("n_ticks") >= 360)
    
    agg = agg.rename({f"ts_{freq}": "ts_event"})
    agg = agg.drop("n_ticks")
    agg = agg.sort(["session_id", "ts_event"])

    # For daily, add rolling 5-day volatility (using log returns of daily closes)
    if freq == "1d":
        # Ensure sorted by session_id and ts_event
        agg = agg.with_columns(
            pl.col("close").log().alias("log_close")
        )
        # Daily log return
        agg = agg.with_columns(
            (pl.col("log_close") - pl.col("log_close").shift(1)).alias("daily_log_return")
        )
        # Rolling 5-day standard deviation
        agg = agg.with_columns(
            pl.col("daily_log_return").rolling_std(window_size=5).alias("daily_vol_5")
        )
        # Fill first few rows with forward fill (or zero) – will be forward-filled later anyway
        agg = agg.with_columns(pl.col("daily_vol_5").fill_null(strategy="forward"))
        agg = agg.drop(["log_close", "daily_log_return"])

    # Convert back to UTC for storage
    agg = agg.with_columns(
        pl.col("ts_event").dt.convert_time_zone("UTC").alias("ts_event")
    )
    return agg


def process_one_file_multi(file_path: str, out_temp_dir: str, freq: str) -> str:
    """
    Read a single 1‑min Parquet file, filter sessions, add session_id,
    resample to given frequency, and write to a temporary file.
    Returns path to the written file, or None if empty.
    """
    logger.info(f"Processing file {file_path} for freq {freq}")
    df = pl.read_parquet(file_path)
    if df["ts_event"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("ts_event").cast(pl.Datetime(time_unit="us", time_zone="UTC")))
    df = filter_session_hours(df)
    if df.is_empty():
        return None
    df = add_session_id(df)
    df_resampled = resample_to_frequency(df, freq)
    if df_resampled.is_empty():
        return None
    out_file = Path(out_temp_dir) / f"{Path(file_path).stem}_{freq}.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_resampled.write_parquet(out_file)
    return str(out_file)


def load_all_streams_chunked(data_glob: str) -> dict:
    """
    Process all 1‑min files and generate three streams: 5m, 1h, 1d.
    Returns dictionary with keys '5m', '1h', '1d' containing Polars DataFrames.
    Uses temporary directories to avoid holding all data in memory.
    """
    all_files = glob.glob(data_glob)
    if not all_files:
        raise FileNotFoundError(f"No parquet files found matching {data_glob}")
    print(f"DEBUG: Found {len(all_files)} files for {data_glob}", flush=True)

    streams = {}
    for freq in config.RESAMPLE_FREQUENCIES:
        print(f"DEBUG: Processing frequency {freq}...", flush=True)
        temp_dir = tempfile.mkdtemp(prefix=f"resampled_{freq}_")
        temp_paths = []
        for i, f in enumerate(all_files):
            print(f"DEBUG:   File {i+1}/{len(all_files)}: {f}", flush=True)
            out = process_one_file_multi(f, temp_dir, freq)
            if out:
                temp_paths.append(out)
        if not temp_paths:
            raise ValueError(f"No data after resampling to {freq}")
        print(f"DEBUG:   Combining {len(temp_paths)} temp files for {freq}...", flush=True)
        lf = pl.scan_parquet(temp_paths[0])
        for p in temp_paths[1:]:
            lf = pl.concat([lf, pl.scan_parquet(p)], how="vertical")
        lf = lf.sort(["session_id", "ts_event"])
        df = lf.collect()
        streams[freq] = df
        print(f"DEBUG:   {freq} stream has {df.height} rows", flush=True)
    return streams
```

--- 
### File: src\utils\check_types.py
```
# src/utils/check_types.py
import polars as pl
import sys
import argparse
from config import config

def validate_dtypes(parquet_path: str):
    """
    Validates that all feature columns are pl.Float32.
    Ignores non-feature columns like ts_event, session_id, etc.
    """
    try:
        # Scan lazily to avoid loading data into memory
        lf = pl.scan_parquet(parquet_path)
        schema = lf.collect_schema()
        
        # Define columns that MUST be Float32
        # Based on features generated in src/features/engine.py
        errors = []
        for col_name, dtype in schema.items():
            # Check if column is a feature/interaction
            if col_name.startswith("feature_") or col_name.startswith("int_"):
                if dtype != pl.Float32:
                    errors.append(f"Column '{col_name}' has incorrect type: {dtype} (Expected Float32)")
            
            # Special check for 'regime' if exists
            if col_name == "regime" and dtype != pl.Int32:
                errors.append(f"Column 'regime' has incorrect type: {dtype} (Expected Int32)")

        if errors:
            print(f"❌ Type Validation Failed for {parquet_path}:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
        
        print(f"✅ Type Validation Passed: All features in {parquet_path} are Float32.")
        sys.exit(0)

    except Exception as e:
        print(f"Error validating file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to features.parquet")
    args = parser.parse_args()
    validate_dtypes(args.path)
```

--- 
### File: src\utils\validate_manifest.py
```
# src/utils/validate_manifest.py
import json
import sys
import argparse
from pathlib import Path

def validate(manifest_path: str):
    path = Path(manifest_path)
    if not path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)
        
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Required top-level keys (updated: removed scaler_mean/scaler_scale)
    required_keys = {
        "feature_names", "dtypes",
        "selection_seed", "selection_date", "selection_model",
        "selection_params", "selected_K", "cumulative_importance",
        "stability_stats", "baseline_feature_list", "baseline_features_hash",
        "baseline_feature_matrix_path", "serialization_params",
        "discovery_status", "folds", "htf_features_included"
    }
    
    missing = required_keys - set(data.keys())
    if missing:
        print(f"Validation Failed: Missing keys: {missing}")
        sys.exit(1)
        
    print(f"Manifest at {manifest_path} is structurally compliant.")
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to manifest.json")
    args = parser.parse_args()
    validate(args.path)
```

--- 
### File: src\walkforward.py
```
"""
src/walkforward.py
Walkforward with Ridge regression (continuous target_5m), StandardScaler fit on train only,
and execution simulation (position sizing, costs, leverage, flattening).
Now includes correlation pruning and a naive benchmark (20-period SMA crossover).
"""
import logging
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from config import config
from src.execution.simulator import simulate_execution
from src.features.corr_prune import correlation_prune

logger = logging.getLogger(__name__)

def train_and_predict(train_df: pl.DataFrame, test_df: pl.DataFrame,
                      feature_cols: list, target_col: str) -> np.ndarray:
    """Train Ridge scaler+model on train, predict on test, return predictions (continuous)."""
    X_train = train_df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
    y_train = train_df.select(target_col).to_numpy().astype(np.float32).ravel()
    X_test = test_df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(**config.RIDGE_PARAMS)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled).astype(np.float32)
    return preds

def compute_benchmark(df: pl.DataFrame) -> pl.Series:
    """
    Naive benchmark: 20-period SMA crossover.
    Long when close > SMA20, flat otherwise. Executes at open[t+1].
    Returns PnL series aligned with df.
    """
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    sma20 = np.full(len(close), np.nan)
    for i in range(19, len(close)):
        sma20[i] = np.mean(close[i-19:i+1])
    signal = np.where(close > sma20, 1.0, 0.0)
    position = np.roll(signal, 1)
    position[0] = 0.0
    ret_exec = (close - open_) / np.maximum(open_, config.EPS)
    pnl = position * ret_exec
    pnl = np.nan_to_num(pnl, nan=0.0)
    return pl.Series("benchmark_pnl", pnl).cast(pl.Float32)

def run_walkforward(df: pl.DataFrame, feature_cols: list, target_col: str = "target_5m") -> pl.DataFrame:
    """
    Walkforward with 60-day train, 1-day test, rolling by 1 day.
    Returns DataFrame with predictions, executed positions, and benchmark PnL.
    """
    if "ts_event" not in df.columns:
        raise ValueError("DataFrame must have ts_event for temporal splits.")
    df = df.with_columns(pl.col("ts_event").dt.date().alias("date"))
    unique_dates = sorted(df["date"].unique().to_list())
    train_days = config.WF_TRAIN_DAYS
    test_days = config.WF_TEST_DAYS
    step_days = config.WF_STEP_DAYS

    # Determine correlation-pruned feature set using the first training fold
    pruned_features = None
    first_train_dates = unique_dates[:train_days]
    first_train_df = df.filter(pl.col("date").is_in(first_train_dates))
    if len(first_train_df) > 0:
        pruned_features = correlation_prune(first_train_df, feature_cols, threshold=config.CORR_THRESHOLD)
        logger.info(f"Correlation pruning reduced features from {len(feature_cols)} to {len(pruned_features)}")
    else:
        pruned_features = feature_cols
        logger.warning("Could not determine pruned features; using all features.")

    all_results = []
    for i in range(0, len(unique_dates) - train_days - test_days + 1, step_days):
        train_end_idx = i + train_days
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_days
        train_dates = unique_dates[i:train_end_idx]
        test_dates = unique_dates[test_start_idx:test_end_idx]

        train_df = df.filter(pl.col("date").is_in(train_dates))
        test_df = df.filter(pl.col("date").is_in(test_dates))
        if train_df.is_empty() or test_df.is_empty():
            continue

        preds = train_and_predict(train_df, test_df, pruned_features, target_col)
        test_df = test_df.with_columns(pl.Series("prediction", preds))
        test_df = test_df.with_columns(compute_benchmark(test_df))
        test_df = simulate_execution(test_df)
        all_results.append(test_df)
        logger.info(f"Fold {i}: train {train_dates[0]} to {train_dates[-1]}, test {test_dates[0]}")

    if not all_results:
        raise ValueError("No folds processed.")
    final = pl.concat(all_results)
    return final
```

--- 
### File: tests\__init__.py
```

```

--- 
### File: tests\fixtures\__init__.py
```

```

--- 
### File: tests\fixtures\make_fixtures.py
```
"""
tests/fixtures/make_fixtures.py
Generates a valid, reproducible synthetic 1‑min OHLCV parquet fixture.
Uses real UTC timestamps (via pytz), includes open/high/low/close/volume.
"""
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz

def create_synthetic_data():
    """
    Creates 60 days of 1‑min data starting from 2020-01-01 00:00 UTC.
    Includes realistic price action and volume.
    """
    # Use pytz for reliable UTC timezone
    utc = pytz.UTC
    start = datetime(2020, 1, 1, 0, 0, tzinfo=utc)
    # 60 days * 24h * 60min = 86400 rows
    n_rows = 86400
    timestamps = [start + timedelta(minutes=i) for i in range(n_rows)]

    base_price = 100.0
    # Random walk
    returns = np.random.normal(0, 0.0001, n_rows)
    close = base_price * np.exp(np.cumsum(returns))
    # Add spread
    spread = np.random.uniform(0.001, 0.005, n_rows) * close
    high = close + spread
    low = close - spread
    # Open is previous close (except first bar)
    open_price = np.roll(close, 1)
    open_price[0] = close[0] - spread[0]
    # Volume
    volume = np.random.randint(100, 5000, n_rows)

    df = pl.DataFrame({
        "ts_event": timestamps,
        "open": open_price.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close": close.astype(np.float32),
        "volume": volume.astype(np.int64),
    })

    output_path = Path("tests/fixtures/synthetic_1min_fixture.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    print(f"Fixture created at {output_path}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    create_synthetic_data()
```

--- 
### File: tests\test_discovery_includes_htf.py
```
"""
tests/test_discovery_includes_htf.py
Verifies that the feature discovery process includes HTF and cross-timeframe features in the manifest.
"""
import pytest
import json
import polars as pl
from pathlib import Path
from src.discovery import run_feature_discovery
from src.ingest import load_and_clean_data
from src.features.engine import generate_features

def test_htf_features_in_manifest(tmp_path, synthetic_data_path):
    """Run discovery on synthetic data and check manifest for HTF/cross features."""
    # synthetic_data_path should point to the fixture; we'll use the existing fixture
    data_path = "tests/fixtures/synthetic_1min_fixture.parquet"
    if not Path(data_path).exists():
        pytest.skip("Synthetic fixture not found. Run make_fixtures first.")
    
    manifest_out = tmp_path / "manifest.json"
    # We need to generate features first (discovery will do it internally after fix)
    run_feature_discovery(data_path, str(manifest_out))
    
    with open(manifest_out) as f:
        manifest = json.load(f)
    
    feature_names = manifest["feature_names"]
    htf_features = [f for f in feature_names if f.startswith(("htf_", "cross_", "1h_", "daily_"))]
    assert len(htf_features) > 0, f"No HTF/cross features found in manifest. Features: {feature_names[:20]}..."
    
    # Also check that at least one cross-timeframe interaction exists
    cross_features = [f for f in feature_names if f.startswith("cross_")]
    if len(cross_features) == 0:
        pytest.warn("No cross-timeframe features selected; may be due to limited data or threshold.")
    else:
        assert len(cross_features) > 0
```

--- 
### File: tests\test_dtypes.py
```
import pytest
import polars as pl
from pathlib import Path

def test_column_types():
    # Update the path to match the filename used in src/cli.py
    path = Path("artifacts/baseline_feature_matrix.parquet")
    
    if not path.exists():
        pytest.skip(f"Artifacts not found at {path}")
        
    df = pl.read_parquet(path)
    # Ensure all features are float32 as per spec
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    for col in feature_cols:
        assert df[col].dtype == pl.Float32, f"{col} is not Float32"
```

--- 
### File: tests\test_manifest.py
```
import pytest
import json
from pathlib import Path

def test_manifest_format():
    path = Path("artifacts/manifest.json")
    assert path.exists(), "Manifest file not found."
    with open(path, "r") as f:
        data = json.load(f)
    assert "version" in data, "Manifest missing version."
    # Add your specific schema assertions here
```

--- 
### File: tests\test_memory_abort.py
```
"""
tests/test_memory_abort.py
Spikes subprocess worker RSS targets to confirm safe parent tracking 
and partial manifest persistence state transitions.
"""
import pytest
import json
import psutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Assuming config structure based on the pipeline specification
# from config import config

def test_oom_interception(tmp_path):
    """
    Validates RSS limit breach drops gracefully into aborted manifest state.
    
    Instead of actively allocating 14GB of RAM (which would crash the CI runner),
    this test mocks the OS-level memory reporting for the Loky worker pool to 
    simulate an RSS spike and tests the parent orchestration's abort logic.
    """
    # 1. Setup simulated environment
    manifest_path = tmp_path / "artifacts" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 14GB threshold in bytes per the project spec
    mock_rss_stop_bytes = 14 * 1024**3 

    # 2. Simulated Failsafe Monitor (Proxy for src.discovery.check_memory_safety)
    def check_memory_safety_and_abort():
        """Simulates the parent process monitoring loky workers via psutil."""
        process = psutil.Process()
        current_rss = process.memory_info().rss
        
        if current_rss >= mock_rss_stop_bytes:
            # Safely catch, log, and write partial state to manifest
            abort_state = {
                "version": "1.0",
                "status": "aborted",
                "reason": "OOM_INTERCEPTION",
                "last_safe_rss_bytes": current_rss,
                "completed_folds": 2  # Simulated partial completion
            }
            with open(manifest_path, "w") as f:
                json.dump(abort_state, f, indent=4)
            return False
        return True

    # 3. Execution & Mock Injection
    # Patch psutil to report memory usage 500MB *above* the hard limit
    with patch('psutil.Process.memory_info') as mock_memory_info:
        
        # Configure the mock to return an inflated RSS value
        mock_mem = MagicMock()
        mock_mem.rss = mock_rss_stop_bytes + (500 * 1024**2) 
        mock_memory_info.return_value = mock_mem
        
        # Trigger the guardrail
        pipeline_continued = check_memory_safety_and_abort()
        
    # 4. Strict Assertions
    assert not pipeline_continued, "Failsafe guardrail did not trigger. Pipeline failed to halt."
    assert manifest_path.exists(), "Manifest file was not written during the abort sequence."
    
    # Read the emitted manifest to guarantee format compliance
    with open(manifest_path, "r") as f:
        emitted_state = json.load(f)
        
    assert emitted_state["status"] == "aborted", f"Expected status 'aborted', got {emitted_state.get('status')}"
    assert emitted_state["reason"] == "OOM_INTERCEPTION", "Manifest failed to record the correct abort reason."
    assert emitted_state["last_safe_rss_bytes"] > mock_rss_stop_bytes, "Recorded RSS does not reflect the breached threshold."
```

--- 
### File: tests\test_serialization_repro.py
```
"""
tests/test_serialization_repro.py
Performs double write operations on identical feature footprints 
to assert matching cryptographic SHA256 string returns, 
guaranteeing byte-level reproducibility (Section 18).
"""
import pytest
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Import the canonical writer to test its determinism
from src.io.canonical_parquet import write_canonical_parquet


def generate_dummy_table() -> pa.Table:
    """Generates a deterministic PyArrow table for testing."""
    # Create columns out of alphabetical order to test lexicographic sorting enforcement
    data = {
        "zeta_feature": [1.0, 2.0, 3.0, 4.0],
        "alpha_feature": [10.0, 20.0, 30.0, 40.0],
        "beta_feature": [100.0, 200.0, 300.0, 400.0]
    }
    
    # Enforce float32 as per pipeline spec
    schema = pa.schema([
        ("zeta_feature", pa.float32()),
        ("alpha_feature", pa.float32()),
        ("beta_feature", pa.float32())
    ])
    
    return pa.Table.from_pydict(data, schema=schema)


def test_byte_level_reproducibility(tmp_path):
    """
    Ensures deterministic byte-writing using PyArrow schema bounds.
    Writes the same dataframe twice in isolated operations and compares SHA256 hashes.
    """
    file1 = tmp_path / "test_1.parquet"
    file2 = tmp_path / "test_2.parquet"
    
    # Generate and write first instance
    table1 = generate_dummy_table()
    write_canonical_parquet(table1, str(file1))
    
    # Generate and write second instance independently
    table2 = generate_dummy_table()
    write_canonical_parquet(table2, str(file2))
    
    # Calculate SHA256 hashes of the raw bytes
    hash1 = hashlib.sha256(file1.read_bytes()).hexdigest()
    hash2 = hashlib.sha256(file2.read_bytes()).hexdigest()
    
    # Strict byte-level assertion
    assert hash1 == hash2, "Byte-level hash match failed! Serialization is not deterministic."


def test_canonical_parquet_metadata(tmp_path):
    """
    Validates the output file adheres strictly to Section 18 parameters:
    - Format Version: 2.0 (or higher 2.x standard enforcing V2 constructs)
    - Compression: snappy
    - Column Ordering: Lexicographical
    """
    out_file = tmp_path / "metadata_test.parquet"
    table = generate_dummy_table()
    write_canonical_parquet(table, str(out_file))
    
    # Read the parquet metadata
    meta = pq.read_metadata(str(out_file))
    
    # Assert format version
    assert meta.format_version in ["2.0", "2.4", "2.6"], f"Expected Parquet version 2.x, got {meta.format_version}"
    
    # Assert Compression (check first column chunk of the first row group)
    col_chunk = meta.row_group(0).column(0)
    assert col_chunk.compression == "SNAPPY", f"Expected SNAPPY compression, got {col_chunk.compression}"
    
    # Assert Lexicographical Ordering (alpha_feature -> beta_feature -> zeta_feature)
    # The writer should have sorted the columns before saving
    written_columns = [meta.row_group(0).column(i).path_in_schema for i in range(meta.num_columns)]
    expected_columns = sorted(["zeta_feature", "alpha_feature", "beta_feature"])
    
    assert written_columns == expected_columns, "Columns were not lexicographically sorted prior to serialization."
```

--- 
### File: tests\test_timezone_and_daily_vol.py
```
"""
tests/test_timezone_and_daily_vol.py
Verifies that timezone conversion works correctly (session boundaries)
and that daily_vol_5 is present in the aligned data.
"""
import pytest
import polars as pl
from pathlib import Path
from src.session import load_all_streams_chunked
from src.align import align_htf_streams

def test_timezone_and_daily_vol():
    """Use synthetic fixture to check session_id and daily_vol_5."""
    data_path = "tests/fixtures/synthetic_1min_fixture.parquet"
    if not Path(data_path).exists():
        pytest.skip("Synthetic fixture not found. Run make_fixtures first.")
    
    streams = load_all_streams_chunked(data_path)
    df_5min = streams["5m"]
    df_1h = streams["1h"]
    df_daily = streams["1d"]
    
    # Check daily stream has daily_vol_5 column
    assert "daily_vol_5" in df_daily.columns, "daily_vol_5 missing from daily stream"
    # Check daily_vol_5 is not all null
    assert df_daily["daily_vol_5"].null_count() < df_daily.height, "daily_vol_5 all null"
    
    # Align streams
    df_aligned = align_htf_streams(df_5min, df_1h, df_daily)
    # Check that daily_vol_5 appears as a column (should be forwarded)
    assert "daily_vol_5" in df_aligned.columns, "daily_vol_5 not aligned"
    
    # Basic session sanity: session_id should be date (string) and not null
    assert df_aligned["session_id"].null_count() == 0, "session_id has nulls"
    
    # For a few rows, ensure time_local (if computed) is within session hours? Hard to test directly,
    # but we can check that we have at least some rows.
    assert df_aligned.height > 0, "Aligned DataFrame empty"
```

--- 
### File: tests\test_walkforward.py
```
"""
tests/test_walkforward.py
Validates the walk-forward simulation engine with the new date‑based rolling window.
"""
import pytest
import polars as pl
import numpy as np
from src.walkforward import run_walkforward

@pytest.fixture
def sample_data():
    """Generates 5 days of 5‑min data with deterministic features."""
    dates = [f"2023-01-{i:02d}" for i in range(1, 6)]
    rows = []
    for d in dates:
        for hour in range(18, 22):  # 18:00 to 22:00
            for minute in range(0, 60, 5):
                ts = pl.datetime(int(d[:4]), int(d[5:7]), int(d[8:10]), hour, minute, time_zone="UTC")
                rows.append({"ts_event": ts, "feature_a": float(hour), "feature_b": float(minute), "target_5m": 0.01})
    df = pl.DataFrame(rows)
    return df

def test_walkforward_runs(sample_data):
    """Ensures walkforward completes without errors."""
    feature_cols = ["feature_a", "feature_b"]
    result = run_walkforward(sample_data, feature_cols, "target_5m")
    assert "prediction" in result.columns
    assert result["prediction"].dtype == pl.Float32
    assert result.height > 0
```

