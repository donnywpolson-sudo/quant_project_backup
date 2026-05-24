# config.py
from datetime import time
from types import SimpleNamespace

config = SimpleNamespace()

# Paths
config.DATA_GLOB = "data/futures/*.parquet"
config.MANIFEST_PATH = "artifacts/manifest.json"
config.BASELINE_FEATURES_FILE = "config/baseline_features.yaml"
config.BASELINE_FEATURES_PERSIST_PATH = "artifacts/baseline_feature_matrix.parquet"
config.TRADES_OUT = "artifacts/trades.csv"
config.LOG_DIR = "logs/"

# Memory & determinism
config.RAM_CAP_BYTES = 14 * 1024**3
config.RSS_STOP_BYTES = 13.5 * 1024**3
config.ROWS_PER_CHUNK_MAX = 5_000_000
config.MEMORY_SAFETY_MARGIN = 0.95

# Time & session
config.TIMEZONE = "America/New_York"
config.SESSION_START_LOCAL = time(18, 0)
config.SESSION_END_LOCAL = time(16, 0)
config.RESAMPLE_FREQUENCIES = ["5m", "1h", "1d"]
config.DROP_INCOMPLETE_ROWS = True

# Feature windows
config.ROLL_WINDOWS = [5, 10, 20, 50]
config.ROLL_WINDOWS_1H = [2, 4, 6, 12]
config.ROLL_WINDOWS_DAILY = [5, 10, 20]
config.ROLL_WINDOW_MIN_ROWS = 20          # <--- ADDED (used in baseline.py)

# Feature expansion
config.FEATURE_TRANSFORMS = ["lags","ratios","z_scores","pairwise_products_limited","cross_timeframe_ratios"]
config.MAX_PAIRWISE_INTERACTIONS = 500
config.MAX_CROSS_TIMEFRAME_INTERACTIONS = 200

# HTF context
config.HTF_TREND_WINDOWS = [5, 10, 20]
config.HTF_VOLATILITY_WINDOWS = [5, 10, 20]
config.HTF_ALIGNMENT_FILTER = True
config.HTF_TREND_THRESHOLD = 0.1

# Regime
config.VOL_MEDIAN_WINDOW = 20
config.VOL_SMOOTH_WINDOW = 5
config.REGIME_HIGH_THRESH = 0.6
config.REGIME_LOW_THRESH = 0.4
config.REGIME_MISSING_DEFAULT = 0.0

# Target
config.TARGET_5M_HORIZON = 1
config.TARGET_SCALE_FACTOR = 100.0

# Discovery
config.DISCOVERY_WINDOW_DAYS = 60
config.BOOTSTRAP_FOLDS = 30
config.EXTRA_TREES_PARAMS = {"random_state":42, "n_jobs":1, "n_estimators":100,
                             "max_depth":12, "max_features":0.3, "bootstrap":False}
config.SELECTION_FREQ_THRESHOLD = 0.75
config.SIGN_CONSISTENCY_THRESHOLD = 0.8
config.CUMULATIVE_IMPORTANCE_THRESHOLD = 0.95
config.MIN_SELECTED_FEATURES = 10
config.MAX_SELECTED_FEATURES = 1000

# Walkforward & model
config.WF_TRAIN_DAYS = 60
config.WF_TEST_DAYS = 1
config.WF_STEP_DAYS = 1
config.RIDGE_PARAMS = {"alpha":1.0, "solver":"cholesky", "fit_intercept":True, "random_state":42}
config.MODEL_TYPE = "Ridge"
config.PROBABILITY_SMOOTHING_ALPHA = 0.1
config.CORR_THRESHOLD = 0.95
config.WF_PARALLEL_FOLDS = 1

# Execution
config.EXECUTE_AT = "open[t+1]"
config.SLIPPAGE_K = 0.001
config.VOL_PENALTY = 0.005
config.COMMISSION_PER_TRADE = 0.00002
config.TARGET_VOL = 0.01
config.MAX_LEVERAGE = 3.0
config.MAX_POS_CHANGE_PER_MIN = 0.1
config.FLAT_BEFORE_CLOSE_MINUTES = 5
config.HTF_TREND_ALIGNMENT = True
config.HTF_VOL_SCALING = True
config.HTF_VOL_WINDOW = 10
config.REMOVE_PREDICTION_BIAS = False

# Misc
config.SEED = 42
config.CLIP_MIN = -10.0
config.CLIP_MAX = 10.0
config.EPS = 1e-9
config.REPLACE_INF_NAN_WITH = 0.0
config.ROW_GROUP_SIZE = 65536
config.MEMORY_LOG_ENABLED = True

# Market config paths
config.MARKET_CONFIGS = {
    "ES": "config/markets/ES.yaml",
    "CL": "config/markets/CL.yaml",
    "ZB": "config/markets/ZB.yaml",
}