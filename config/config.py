"""
config/config.py
Single Source of Truth for the Quant Pipeline.
Optimised for Ryzen 5 2600 (6 cores / 12 threads) – high performance, deterministic.
"""
import os
import logging
from datetime import time
from pathlib import Path

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment (let BLAS/NumPy use all cores, still deterministic) ---
# Uncomment to explicitly set thread count for BLAS (optional)
# os.environ["OMP_NUM_THREADS"] = "6"
# os.environ["OPENBLAS_NUM_THREADS"] = "6"
# os.environ["MKL_NUM_THREADS"] = "6"

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

# --- Determinism seed ---
SEED = 42

# --- Numeric guards ---
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

# --- Feature expansion (moderate – still fast, prop‑firm quality) ---
FEATURE_TRANSFORMS = ["lags", "ratios", "z_scores", "pairwise_products_limited", "cross_timeframe_ratios"]
MAX_PAIRWISE_INTERACTIONS = 200      # reduced from 500 – still powerful, much faster
MAX_CROSS_TIMEFRAME_INTERACTIONS = 100  # reduced from 200
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
TARGET_5M_HORIZON = 10          # increased from 1 to 10 bars (50 minutes)
MAGNITUDE_THRESHOLD = 0.002
PROB_TARGET_THRESHOLD = 0.005
TARGET_SCALE_FACTOR = 1         # not used for classification

# --- 1H mapping (optional, not active) ---
DST_AWARE_1H_TESTS = True
PARTIAL_BLOCK_MIN_MINUTES = 15

# --- Correlation filter ---
CORR_THRESHOLD = 0.95
CORR_TIE_BREAKER = ["variance_desc", "name_lexicographic"]
CORR_ACCUMULATION_MODE = "compensated_float64_then_downcast"

# --- Nonlinear discovery ExtraTrees (deterministic but parallel folds) ---
DISCOVERY_METHOD = "ExtraTrees"
DISCOVERY_WINDOW_DAYS = 60
BOOTSTRAP_FOLDS = 10
EXTRA_TREES_PARAMS = {
    "random_state": 42,
    "n_jobs": 1,
    "n_estimators": 30,
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
RIDGE_PARAMS = {"alpha": 0.01, "solver": "cholesky", "fit_intercept": True, "random_state": 42}
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
PROBABILITY_SMOOTHING_ALPHA = 0.1   # EMA smoothing for predictions (0 = no smoothing, 1 = instant)

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

# ===================================================
# PARALLELISM FLAGS (Optimised for Ryzen 5 2600 – 6 cores / 12 threads)
# ===================================================
DISCOVERY_PARALLEL_FOLDS = 6
WF_PARALLEL_FOLDS = 6

# --- Market-specific overrides (absolute paths) ---
BASE_DIR = Path(__file__).parent.parent
MARKET_CONFIGS = {
    "ES": str(BASE_DIR / "config/markets/ES.yaml"),
    "ZB": str(BASE_DIR / "config/markets/ZB.yaml"),
    "CL": str(BASE_DIR / "config/markets/CL.yaml"),
}