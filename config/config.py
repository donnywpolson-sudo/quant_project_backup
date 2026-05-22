"""
config/config.py
Single Source of Truth for the Quant Pipeline.
All tunable parameters centralized here. Change values here only; code imports this module.
"""
import os
from datetime import time

# --- 1. Environment and Threading Configuration ---
# Setting environment variables before library imports is critical.
# These values are set at the OS level to ensure deterministic behavior.
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

# --- 2. Paths and IO ---
DATA_GLOB = "data/futures/*.parquet"
FEATURES_OUT = "artifacts/features.parquet"
MANIFEST_PATH = "artifacts/manifest.json"
MODELS_DIR = "models/"
TRADES_OUT = "artifacts/trades.csv"
PNL_OUT = "artifacts/pnl_series.csv"
LOG_DIR = "logs/"
CACHE_DIR = "cache/"
MEMORY_TRACE_OUT = "logs/memory_trace.csv"
SYNTHETIC_FIXTURE_PATH = "tests/fixtures/synthetic_1min_fixture.parquet"
MAPPING_FILE_PATH = "config/term_to_canonical_mapping.yaml"
BASELINE_FEATURES_FILE = "config/baseline_features.yaml"

# --- 3. Schema Definitions ---
# Fixed to match discovered columns: 
# ['session_id', 'ts_event', 'row_id', 'close', 'volume', 'ask_price', 'bid_price', 'target']
ALL_COLUMNS = ['session_id', 'ts_event', 'row_id', 'close', 'volume', 'ask_price', 'bid_price', 'target']

# Defining features (excluding metadata and target)
BASELINE_FEATURES = ["close", "volume", "ask_price", "bid_price"]
TARGET_COL = "target"
ID_COLS = ["session_id", "ts_event", "row_id"]

BASELINE_FEATURES_PERSIST_PATH = "artifacts/baseline_feature_matrix.parquet"
MEMORY_LOG_ENABLED = True

# --- 4. Determinism ---
SEED = 42
SKLEARN_N_JOBS = 1

# --- 5. Numeric Guards and constants ---
EPS = 1e-9
CLIP_MIN = -10.0
CLIP_MAX = 10.0
DTYPE = "float32"
TIMEZONE = "America/New_York"
PRE_POST_CLIP_LOGGING = True
DEBUG_FLOAT64_MODE = False
REPLACE_INF_NAN_WITH = 0.0

# --- 6. Memory and hardware limits ---
RAM_CAP_BYTES = 14 * 1024**3  # 14GB
RSS_STOP_BYTES = 13.5 * 1024**3  # 13.5GB runtime stop threshold
STORAGE_MIN_GB = 200
ROWS_PER_CHUNK_MAX = 5_000_000
MEMORY_SAFETY_MARGIN = 0.95
MEMORY_RSS_CHECKPOINT_INTERVAL_SEC = 10
MEMORY_RSS_CHECKPOINTS_BEFORE_STOP = 3

# --- 7. Data load and collect ---
DATA_SCAN_GLOB = DATA_GLOB
LAZY_PUSHDOWN_FILTERS = True
COLLECT_PARTITION_ROWS = True
STABLE_SORT_KEYS = ["session_id", "ts_event", "row_id"]
ROW_GROUP_SIZE = 65536

# --- 8. Session and resampling ---
SESSION_START_LOCAL = time(18, 0, 0)
SESSION_END_LOCAL = time(16, 0, 0)
SESSION_TZ = TIMEZONE
RESAMPLE_RULES = {"O": "first", "H": "max", "L": "min", "C": "last", "V": "sum"}

# --- 9. Cleaning rules ---
DROP_VOLUME_ZERO = True
ALLOW_FFILL_4H_MAPPING_ONLY = True
DROP_INCOMPLETE_ROWS = True

# --- 10. Base feature windows ---
ROLL_WINDOWS = [5, 10, 20, 50]
ROLL_WINDOW_MIN_ROWS = max(ROLL_WINDOWS)

# --- 11. Feature expansion ---
FEATURE_TRANSFORMS = ["lags", "ratios", "z_scores", "pairwise_products_limited"]
MAX_PAIRWISE_INTERACTIONS = 500
TEMPORAL_BUCKETS = ["early", "mid", "late"]

# --- 12. Regime and HTF ---
VOL_MEDIAN_WINDOW = 20
VOL_SMOOTH_WINDOW = 5
REGIME_HIGH_THRESH = 0.6
REGIME_LOW_THRESH = 0.4
REGIME_MISSING_DEFAULT = 0

# --- 13. Targets ---
TARGET_5M_HORIZON = 1
TARGET_1H_RESAMPLE_RULE = "1H"
MAGNITUDE_THRESHOLD = 0.002
PROB_TARGET_THRESHOLD = 0.005

# --- 14. H1 mapping ---
H1_BLOCK_DEFINITION = "explicit_per_session"
H1_FORWARD_FILL_ALLOWED = True
DST_AWARE_H1_TESTS = True
H1_PARTIAL_BLOCK_MIN_MINUTES = 15

# --- 15. Correlation filter ---
CORR_THRESHOLD = 0.95
CORR_TIE_BREAKER = ["variance_desc", "name_lexicographic"]
CORR_ACCUMULATION_MODE = "compensated_float64_then_downcast"

# --- 16. Nonlinear discovery ExtraTrees ---
DISCOVERY_METHOD = "ExtraTrees"
DISCOVERY_WINDOW_DAYS = 60
BOOTSTRAP_FOLDS = 30
EXTRA_TREES_PARAMS = {
    "random_state": 42,
    "n_jobs": 1,
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
DISCOVERY_SENSITIVITY_REPORT = True

# --- 17. Orthogonalization and PCA ---
ORTHOGONALIZE = False
PCA_TOP_COMPONENTS = 5

# --- 18. Manifest and cache ---
MANIFEST_FIELDS = ["feature_names", "dtypes", "scaler_mean", "scaler_scale", "selection_seed", 
                   "selection_date", "selection_model", "selection_params", "selected_K", 
                   "cumulative_importance", "stability_stats"]
CACHE_KEY_COMPONENTS = ["row_count", "ts_min", "ts_max", "dtypes", "file_size", "mtime", 
                        "config_hash", "seed", "manifest_hash"]
CI_SMOKE_DISCOVERY = True

# --- 19. Walkforward ---
WF_TRAIN_DAYS = 60
WF_TEST_DAYS = 1
WF_STEP_DAYS = 1
WF_PRECOMPUTE_INDICES = True

# --- 20. Models Ridge ---
SCALER_CLASS = "StandardScaler"
RIDGE_PARAMS = {"alpha": 1.0, "solver": "cholesky", "fit_intercept": True, "random_state": 42}
RIDGE_N_JOBS = 1
CLASS_WEIGHT_METHOD = "deterministic_from_counts"

# --- 21. Stacking and calibration ---
ENABLE_STACKING = False
STACKER_MODEL = "Ridge"
CALIBRATION_METHOD = "platt"
CALIBRATION_CV_FOLDS = 5

# --- 22. Execution and risk ---
EXECUTE_AT = "open[t+1]"
SLIPPAGE_K = 0.001
VOL_PENALTY = 0.005
SLIPPAGE_STRESS_PCT = 0.5
COMMISSION_PER_TRADE = 0.00002
TARGET_VOL = 0.01
MAX_LEVERAGE = 3.0
MAX_POS_CHANGE_PER_MIN = 0.1
FLAT_BEFORE_CLOSE_MINUTES = 5

# --- 23. Metrics and reporting ---
METRICS_TO_COMPUTE = ["Sharpe", "MaxDrawdown", "Turnover", "HitRate", "AvgWin", "AvgLoss", "MAE"]
DEFAULT_METRICS_IF_NO_TRADES = {"Sharpe": 0, "MaxDrawdown": 0, "Turnover": 0, "HitRate": 0}
ANNUALIZATION_FACTOR = 66528

# --- 24. Tests and thresholds ---
REPRO_HASH_ALGORITHM = "sha256"
DISCOVERY_REPRO_TEST = True
MIN_STABILITY_FEATURES = 5

# --- 25. Trading terminology mapping ---
ENFORCE_MAPPING_AT_INGEST = True
MAPPING_HASH_IN_MANIFEST = True
MAPPING_VERSION_FIELD = "mapping_version"
MAPPING_VALIDATION_CI_TESTS = True

# --- 26. Baseline features and discovery pool ---
DISCOVERY_INITIAL_POOL = "baseline_plus_generated"
BASELINE_FEATURES_FROZEN = True
BASELINE_FEATURES_HASH_FIELD = "baseline_features_hash"

# --- 27. Feature Discovery Configuration ---
FEATURE_DISCOVERY_SPLIT_PCT = 0.50  # Use only the first 50% of historical rows for ExtraTrees discovery
NUM_TOP_FEATURES_TO_SELECT = 15     # Number of best features to export to the manifest
MANIFEST_PATH = "artifacts/manifest.json"

# --- Entrypoint ---
ENTRYPOINT_FN = "run_pipeline"