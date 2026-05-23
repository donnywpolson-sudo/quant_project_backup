# ⚙️ Quant Flowchart — End-to-End Spec (Optimized)

1‑min OHLCV parquet (partitioned by market/year)
   ↓
Resample → 5‑min bars
   ↓

Step 1 — Baseline Features
Small, human-designed (YAML)

Step 2 — Feature Expansion (critical)
Create large candidate space:
- interactions
- ratios
- normalized versions
- regime-conditioned transforms

(ALL features must use past data only)
   ↓

Step 3 — Target Construction
target_1h = forward 12-bar return (5m → 1h)
(optionally convert to long / neutral / short)
   ↓

Step 4 — ExtraTrees (discovery)
- Train on target_1h
- Rank feature importance
- Select top features
- Require stability across folds (e.g. ≥60%)

(hundreds → few dozen)
   ↓

Step 5 — Freeze Features
- Lock selected feature list
- Persist schema + hash
   ↓

Step 6 — Walkforward Ridge Training
- Train only on frozen features
- Target = target_1h
- StandardScaler fit on train only
   ↓

Step 7 — Prediction
- Predict every 5 minutes
   ↓

Step 8 — Execution
- Predictions recomputed every 5 minutes
- Convert prediction → position
- Trade on 5‑min bars


> **Single Source of Truth**  
> Deterministic CPU-only intraday futures ML pipeline  
>  
> data → features → ExtraTrees discovery → frozen features → walkforward Ridge → execution simulation → artifacts

---

# 0. PURPOSE

This document defines the complete production architecture.

The system MUST:
- be deterministic
- enforce zero lookahead leakage
- stay under strict memory limits (<14GB)
- produce reproducible artifacts

---

# 1. OBJECTIVE

Deterministic CPU-only intraday futures ML backtester.

### Trading Constraints
- Globex 23/5
- Session window: 18:00 → 16:00 (America/New_York)
- No overnight holdings

### Core Requirements
- Seed = 42
- Float32 everywhere
- No NaN / inf
- Memory safe (<14GB)
- Deterministic across runs

---

## Critical Rules

### No-Leak Rule
- All features must be computed from t‑1 or earlier
- No training data may contain information from the target bar

---

### Implementation Style
- Polars expression API ONLY
- Lazy execution preferred
- No Python loops unless strictly necessary

---

### Memory Guarantee
Must use streaming:
scan_parquet → transform → sink_parquet

If non-vectorizable:
- MUST respect chunk size from Section 22

---

# 2. GLOBAL ENVIRONMENT

### Seeds
SEED = 42

Initialize:
- numpy
- random
- sklearn

---

### Thread Control
OMP_NUM_THREADS=1  
OPENBLAS_NUM_THREADS=1  
MKL_NUM_THREADS=1  
NUMEXPR_NUM_THREADS=1  

---

### Libraries
- polars >= 1.0
- numpy
- scikit-learn
- pyarrow
- joblib

NO pandas

---

### Numeric Rules
- dtype = float32
- clip to [CLIP_MIN, CLIP_MAX]
- no NaN or inf allowed

---

# 3. CONFIG (SOURCE OF TRUTH)

## Paths
DATA_GLOB = data/futures/*.parquet  
FEATURES_OUT = artifacts/features.parquet  
MANIFEST_PATH = artifacts/manifest.json  
MODELS_DIR = models/  
TRADES_OUT = artifacts/trades.csv  
PNL_OUT = artifacts/pnl_series.csv  
LOG_DIR = logs/  
CACHE_DIR = cache/  
MEMORY_TRACE_OUT = logs/memory_trace.csv  

---

## Determinism
SEED = 42  
SKLEARN_N_JOBS = 1  

---

## Numeric Guards
EPS = 1e-9  
CLIP_MIN = -10.0  
CLIP_MAX = 10.0  
DTYPE = float32  
TIMEZONE = America/New_York  

REPLACE_INF_NAN_WITH = 0.0  
DEBUG_FLOAT64_MODE = False  

---

## Memory Limits
RAM_CAP_BYTES = 14GB  
RSS_STOP_BYTES = 13.5GB  

ROWS_PER_CHUNK_MAX = 5_000_000  
MEMORY_SAFETY_MARGIN = 0.95  

---

## Data Handling
STABLE_SORT_KEYS = ["session_id", "ts_event", "row_id"]

---

# 4. HARDWARE LIMITS

- RAM strictly < 14GB  
- Storage ≥ 200GB  

Abort immediately on violation.

---

# 5. DATA PIPELINE

## Lazy Stage
scan_parquet → pushdown filters → light transforms

---

## Eager Stage
- deduplicate  
- compute session boundaries  
- validate schema  

---

## Sorting
["session_id", "ts_event", "row_id"]

---

# 6. POST-COLLECT VALIDATION

### Integrity Checks
- ts_event strictly increasing  
- no nulls  
- high ≥ low  
- prices within [low, high]  
- volume ≥ 0  

---

### Memory Estimation
Compute:
- total bytes  
- avg row size  
- rows_per_chunk  

Abort if:
estimated_memory > RAM_CAP_BYTES

---

# 7. SESSION LOGIC (CRITICAL)

## Globex Rollover
If hour ≥ 18:00:
session_date = ts_event + 6 hours

Ensures:
- continuous trading day  
- no split sessions  

---

## Session Filtering
Keep only 18:00 → 16:00

---

## Resampling
5-minute bars

Rules:
O = first  
H = max  
L = min  
C = last  
V = sum  

---

# 8. CLEANING RULES

- No forward/back fill (except allowed cases)  
- Drop volume == 0  
- Drop incomplete rows  
- Replace NaN / inf → 0.0  

---

# 9. BASE FEATURES

- log returns (t-1)  
- rolling statistics per session  
- float32 only  
- clipped  

---

# 10. FEATURE EXPANSION

All features MUST be:
- deterministic  
- session-aware  
- float32  
- clipped  

---

## Core Groups

Range / Vol:
- high_low_range_norm  
- true_range  
- atr_14  
- price_z_20  

Trend:
- dist_ma_20 / 50  
- pos_in_range_20  

Volume / Microstructure:
- log_volume  
- volume_z_20  
- signed volume imbalance  
- spread proxy  

Session:
- session_pos  
- session_len  
- time_of_day_bucket  

Momentum:
- rolling returns  
- rolling std  
- EWMA volatility  

---

## Pairwise Interactions
- max 500  
- lexicographic order  
- deterministic  
- stop exactly at cap  

---

# 11. NUMERIC GUARDS

- no NaN / inf  
- clip everything  
- optional float64 debug pass  

---

# 12. REGIME

vol → median → smoothed  

Assign:
- 1 = high vol  
- 0 = low vol  
- else carry forward  

---

# 13. TARGETS

## A. 5m Target
target_5m[t] = sign(log(close[t+1] / open[t+1]))

Set to 0 if crossing session boundary.

---

## B. 1H Target
- session-aware  
- forward shift after block closes  
- DST-safe  

---

## C. Magnitude Target
|log_return| > threshold  

---

## D. Probabilistic Target
- used for calibration  

---

# 14. 1H MAPPING RULES

- start inclusive  
- end exclusive  
- drop partial blocks < 15 min  
- forward fill only within block  
- DST-safe  

---

# 15. CORRELATION FILTER

- train split only  
- float64 compute → float32 store  
- drop features > threshold  

Exclude:
- targets  
- regime  

---

# 16. FEATURE DISCOVERY

- joblib loky parallel  
- bootstrap folds = 30  

---

## Deterministic Seeds
seed = HMAC_SHA256(SEED:fold_index)

---

## Abort Protocol
- monitor RSS  
- terminate if exceeded  
- persist partial manifest  

---

## Baseline Feature Pool
- load YAML (40 features)  
- cannot prune pre-discovery  

---

# 17. PCA / ORTHOGONALIZATION

Optional:
top components = 5

---

# 18. PARQUET WRITING (STRICT)

version = 2.0  
compression = snappy  
row_group_size = 65536  
column_order = lexicographic  

---

# 19. FEATURE FREEZE

- identical schema everywhere  
- assert hash equality  

---

# 20. WALKFORWARD

train = 60 days  
test = 1 day  
step = 1 day  

---

## Model
Ridge + StandardScaler

---

## Execution
position = clip(prediction / vol * TARGET_VOL, -MAX_LEVERAGE, MAX_LEVERAGE)

---

## Costs
cost = commission + slippage_k * spread + vol_penalty * volatility

---

# 21. DATA SCHEMA

Required:
- ts_event  
- open  
- high  
- low  
- close  
- volume  
- row_id  
- session_id  

---

# 22. CHUNK FORMULA

rows_per_chunk =  
min(ROWS_PER_CHUNK_MAX, floor(RAM * margin / avg_row_bytes))

---

# 23. TESTS

Must include:
- DST test  
- memory abort test  
- serialization reproducibility test  

---

# 24. REPOSITORY STRUCTURE

Unchanged from original spec.

---

# 25. BENCHMARK

Must include:
- naive baseline  
- compare Sharpe, Drawdown, Turnover  

---

# 26. ZERO TRUNCATION RULE

- no pseudocode  
- no placeholders  
- full implementations only  

---

# 27. FIXTURE

Synthetic parquet with DST edge case required.

---

# 28. MANIFEST SCHEMA

Strict JSON format enforced exactly as specified.

---

# 29. ENTRYPOINT

def run_pipeline(data_glob, config_path, out_dir)

---

# 30. DEV CHECKLIST

- generate fixtures  
- run pipeline  
- verify hashes  
- run pytest  
- validate memory  

---

# 31. FINAL DIRECTIVE

Priority order:

1. Determinism  
2. No leakage  
3. Memory safety  
4. Reproducibility  

All other considerations are secondary.