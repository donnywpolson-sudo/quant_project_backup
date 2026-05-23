"""
config/config.py
Single Source of Truth for the Quant Pipeline.
Professional Implementation: Includes Schema Validation and Circuit Breakers.
"""
import os
import sys
import yaml
import logging
from datetime import time

# --- 0. Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Environment and Threading Configuration ---
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

# --- 2. Dynamic Configuration Loading & Validation ---
# Schema enforcement ensures all markets provide necessary risk/specs
REQUIRED_KEYS = [
    "contract_symbol", "exchange_timezone", "SESSION_START_LOCAL", 
    "SESSION_END_LOCAL", "ROLL_WINDOWS", "REGIME_HIGH_THRESH", 
    "TARGET_5M_HORIZON", "SLIPPAGE_K", "MAX_POSITION_SIZE", "MAX_NOTIONAL_USD"
]

def load_market_config(market_name: str):
    """
    Applies and validates market-specific YAML overrides.
    """
    market_path = f"config/markets/{market_name}.yaml"
    
    if not os.path.exists(market_path):
        logging.error(f"Market config file not found: {market_path}")
        raise FileNotFoundError(f"Config for {market_name} missing.")
        
    with open(market_path, 'r') as f:
        overrides = yaml.safe_load(f)
        
    # Schema Validation
    missing_keys = [k for k in REQUIRED_KEYS if k not in overrides]
    if missing_keys:
        error_msg = f"Market config {market_path} is missing required keys: {missing_keys}"
        logging.error(error_msg)
        raise ValueError(error_msg)
            
    current_module = sys.modules[__name__]
    for key, value in overrides.items():
        if hasattr(current_module, key) or key not in REQUIRED_KEYS:
            setattr(current_module, key, value)
            logging.info(f"Overridden {key} with {value}")
        else:
            logging.warning(f"Config key '{key}' from {market_path} is being set for the first time.")
    
    logging.info(f"Successfully validated and applied overrides for {market_name}")

# --- 3. Environment & Infrastructure ---
ENV = os.getenv("QUANT_ENV", "DEVELOPMENT") 
DEBUG_MODE = (ENV == "DEVELOPMENT")

DATA_ROOT = "/mnt/data/prod" if not DEBUG_MODE else "./data/test"
ARTIFACTS_ROOT = "/mnt/artifacts/prod" if not DEBUG_MODE else "./artifacts"

# --- 4. Global Risk & Circuit Breakers ---
GLOBAL_STOP_LOSS_PCT = 0.05
MAX_DAILY_DRAWDOWN_USD = 10000.0
CIRCUIT_BREAKER_RESET_TIME = time(17, 0, 0) 

# --- 5. Execution & Microstructure ---
MIN_ORDER_INTERVAL_MS = 50
SLIPPAGE_BPS_DEFAULT = 0.5
MAX_NOTIONAL_EXPOSURE = 5000000

# --- 6. Paths and IO ---
DATA_GLOB = os.path.join(DATA_ROOT, "futures/*.parquet")
FEATURES_OUT = os.path.join(ARTIFACTS_ROOT, "features.parquet")
MANIFEST_PATH = os.path.join(ARTIFACTS_ROOT, "manifest.json")
MODELS_DIR = "models/"
TRADES_OUT = os.path.join(ARTIFACTS_ROOT, "trades.csv")
PNL_OUT = os.path.join(ARTIFACTS_ROOT, "pnl_series.csv")
LOG_DIR = "logs/"
CACHE_DIR = "cache/"
SYNTHETIC_FIXTURE_PATH = "tests/fixtures/synthetic_1min_fixture.parquet"
MAPPING_FILE_PATH = "config/term_to_canonical_mapping.yaml"
BASELINE_FEATURES_FILE = "config/baseline_features.yaml"

# --- 7. Schema Definitions ---
ALL_COLUMNS = ['session_id', 'ts_event', 'row_id', 'close', 'volume', 'ask_price', 'bid_price', 'target']
BASELINE_FEATURES = ["close", "volume", "ask_price", "bid_price"]
TARGET_COL = "target"
ID_COLS = ["session_id", "ts_event", "row_id"]
BASELINE_FEATURES_PERSIST_PATH = os.path.join(ARTIFACTS_ROOT, "baseline_feature_matrix.parquet")
MEMORY_LOG_ENABLED = True

# --- 8. Determinism ---
SEED = 42
SKLEARN_N_JOBS = 1

# --- 9. Numeric Guards ---
EPS = 1e-9
CLIP_MIN = -10.0
CLIP_MAX = 10.0
DTYPE = "float32"
TIMEZONE = "America/New_York"
PRE_POST_CLIP_LOGGING = True
DEBUG_FLOAT64_MODE = False
REPLACE_INF_NAN_WITH = 0.0

# --- 10. Memory and Hardware ---
RAM_CAP_BYTES = 14 * 1024**3
RSS_STOP_BYTES = 13.5 * 1024**3
STORAGE_MIN_GB = 200
ROWS_PER_CHUNK_MAX = 5_000_000
MEMORY_SAFETY_MARGIN = 0.95
MEMORY_RSS_CHECKPOINT_INTERVAL_SEC = 10
MEMORY_RSS_CHECKPOINTS_BEFORE_STOP = 3

# --- 11. Data Load & Session ---
DATA_SCAN_GLOB = DATA_GLOB
LAZY_PUSHDOWN_FILTERS = True
COLLECT_PARTITION_ROWS = True
STABLE_SORT_KEYS = ["session_id", "ts_event", "row_id"]
ROW_GROUP_SIZE = 65536
SESSION_START_LOCAL = time(18, 0, 0)
SESSION_END_LOCAL = time(16, 0, 0)
SESSION_TZ = TIMEZONE
RESAMPLE_RULES = {"O": "first", "H": "max", "L": "min", "C": "last", "V": "sum"}

# --- 12. Cleaning & Features ---
DROP_VOLUME_ZERO = True
ALLOW_FFILL_4H_MAPPING_ONLY = True
DROP_INCOMPLETE_ROWS = True
ROLL_WINDOWS = [5, 10, 20, 50]
ROLL_WINDOW_MIN_ROWS = max(ROLL_WINDOWS)
FEATURE_TRANSFORMS = ["lags", "ratios", "z_scores", "pairwise_products_limited"]
MAX_PAIRWISE_INTERACTIONS = 500
TEMPORAL_BUCKETS = ["early", "mid", "late"]

# --- 13. Regime and Targets ---
VOL_MEDIAN_WINDOW = 20
VOL_SMOOTH_WINDOW = 5
REGIME_HIGH_THRESH = 0.6
REGIME_LOW_THRESH = 0.4
REGIME_MISSING_DEFAULT = 0
TARGET_5M_HORIZON = 1
TARGET_1H_RESAMPLE_RULE = "1H"
MAGNITUDE_THRESHOLD = 0.002
PROB_TARGET_THRESHOLD = 0.005

# --- 14. Discovery & Modeling ---
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
RIDGE_PARAMS = {"alpha": 1.0, "solver": "cholesky", "fit_intercept": True, "random_state": 42}

# --- 15. Execution and Risk ---
EXECUTE_AT = "open[t+1]"
SLIPPAGE_K = 0.001
VOL_PENALTY = 0.005
COMMISSION_PER_TRADE = 0.00002
TARGET_VOL = 0.01
MAX_LEVERAGE = 3.0
MAX_POS_CHANGE_PER_MIN = 0.1
FLAT_BEFORE_CLOSE_MINUTES = 5

# --- 16. Metrics ---
METRICS_TO_COMPUTE = ["Sharpe", "MaxDrawdown", "Turnover", "HitRate", "AvgWin", "AvgLoss", "MAE"]
ANNUALIZATION_FACTOR = 66528

# --- 17. Entrypoint ---
ENTRYPOINT_FN = "run_pipeline"