# ⚙️ Quant Flowchart — End-to-End Spec (Optimized, Stability-Enhanced)

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
target_1h = forward 12-bar return normalized by volatility
(optionally convert to long / neutral / short)
   ↓

Step 4 — ExtraTrees (discovery)
- Train on target_1h
- Rank feature importance
- Select top features
- Require stability across folds (≥70%)

(hundreds → few dozen)
   ↓

Step 4.5 — Correlation Pruning (NEW)
- Remove redundant features (|corr| > 0.90)
   ↓

Step 5 — Freeze Features
- Lock selected feature list
- Persist schema + hash
   ↓

Step 6 — Walkforward Ridge Training
- Train only on frozen features
- Target = normalized target_1h
- StandardScaler fit on train only
   ↓

Step 7 — Prediction
- Predict every 5 minutes
- Apply rolling smoothing (3 bars)
   ↓

Step 8 — Execution
- Predictions recomputed every 5 minutes
- Use smoothed volatility
- Convert prediction → position
- Trade on 5‑min bars

---

> **Single Source of Truth**  
> Deterministic CPU-only intraday futures ML pipeline  
>  
> data → features → ExtraTrees → correlation prune → frozen features → walkforward Ridge → smoothed prediction → execution → artifacts

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
Abort if:
estimated_memory > RAM_CAP_BYTES

---

# 7. SESSION LOGIC (CRITICAL)

(unchanged)

---

# 8. CLEANING RULES

(unchanged)

---

# 9. BASE FEATURES

(unchanged)

---

# 10. FEATURE EXPANSION

(unchanged)

---

# 10A. FEATURE STABILITY HARDENING

- Feature must appear in ≥70% of folds
- Median importance preferred

---

# 10B. CORRELATION PRUNING

- Happens AFTER discovery, BEFORE freeze
- Use train split only
- Use float64 correlation

Rule:
- If |corr| > 0.90:
  - keep earlier feature (deterministic)
  - drop later feature

---

# 11. NUMERIC GUARDS

(unchanged)

---

# 12. REGIME

(unchanged)

---

# 13. TARGETS

## 1H Target (UPDATED)

target_1h =
forward log return / rolling volatility

Rules:
- backward-looking volatility
- window ≥ 20
- no overlap with forecast horizon
- session-aware
- clipped

---

# 14. 1H MAPPING RULES

(unchanged)

---

# 15. CORRELATION FILTER

- Runs after discovery
- before freeze
- train-only data
- deterministic output

---

# 16. FEATURE DISCOVERY

(stability ≥70%)

---

# 17. PCA / ORTHOGONALIZATION

(optional)

---

# 18. PARQUET WRITING

(unchanged)

---

# 19. FEATURE FREEZE

(unchanged)

---

# 20. WALKFORWARD

train = 60 days  
test = 1 day  
step = 1 day  

---

## Model

Ridge

alpha = 5.0

---

## 20A. REGULARIZATION

- Higher alpha preferred
- Default = 5.0

---

## 20B. PREDICTION SMOOTHING

- Rolling mean
- window = 3
- strictly backward-looking

---

## 20C. VOLATILITY-STABILIZED EXECUTION

Use smoothed volatility (≥5 bars)

position =
clip(prediction / smoothed_vol * TARGET_VOL, -MAX_LEVERAGE, MAX_LEVERAGE)

---

## Costs

(unchanged)

---

# 21. DATA SCHEMA

(unchanged)

---

# 22. CHUNK FORMULA

(unchanged)

---

# 23. TESTS

(unchanged)

---

# 24. REPOSITORY STRUCTURE

(unchanged)

---

# 25. BENCHMARK

(unchanged)

---

# 26. ZERO TRUNCATION RULE

(unchanged)

---

# 27. FIXTURE

(unchanged)

---

# 28. MANIFEST SCHEMA

(unchanged)

---

# 29. ENTRYPOINT

def run_pipeline(data_glob, config_path, out_dir)

---

# 30. DEV CHECKLIST

(unchanged)

---

# 31. FINAL DIRECTIVE

Priority:

1. Determinism  
2. No leakage  
3. Memory safety  
4. Stability across folds  
5. Reproducibility  