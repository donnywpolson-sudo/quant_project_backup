# ⚙️ Quant Flowchart — End‑to‑End Updated Objective and Spec (Gemini Pro Extended Optimized)

> **One file. One source of truth.** > Deterministic, CPU‑only intraday futures pipeline: data → features → one‑time ExtraTrees discovery → frozen features → walkforward Ridge models → execution simulation → artifacts & tests.

> **From the attached spec:** > “Deterministic CPU-only intraday futures ML backtester. Strict intraday only Globex 23/5 18:00 America_New_York → 16:00 America_New_York no overnight holds.”  
> “The following canonical names and formulas are the frozen baseline candidate pool.”

---

## 0. Purpose of this file
This is the **final Gemini Pro Extended prompt** (`ai_prompt.md`) used to generate a complete, production-ready runnable package implementing the pipeline described in the spec. It leverages Gemini's large context window to generate robust, fully implemented modules across the entire repository structure, guaranteeing determinism, zero lookahead leakage, and rigorous memory safety.

---

## 1. OBJECTIVE
Deterministic CPU-only intraday futures ML backtester.  
Strict intraday only Globex 23/5 18:00 America_New_York → 16:00 America_New_York no overnight holds.  
Zero leakage strict memory less than 14GB reproducible seed 42 float32 only.  
Primary goal: produce robust deployable alpha while guaranteeing determinism, no lookahead, and safe memory usage.

**Source of truth:** change behavior only via `# 3. CONFIG` and this objective block. Include the objective hash in cache keys and manifests to prevent silent drift.

Implicit Constraints:
- **The "No-Leak" Rule:** All feature computations MUST be t-1 relative to the target. No training on data that includes the target bar's close.
- **Implementation Style:** Use polars expression-based lazy evaluation for all feature engineering. Avoid standard iterative loops. Use `map_batches` only if absolutely necessary for custom rolling logic.
- **Memory Guarantee:** The system must use a "Chunked Processing" approach via Polars' streaming API (`scan_parquet` -> `sink_parquet`) to ensure total RSS never exceeds `RAM_CAP_BYTES`. If a computation cannot be natively vectorized, it must be performed in sub-blocks defined strictly by the dynamic `ROWS_PER_CHUNK` formula in Section 22.

---

## 2. GLOBAL ENV
- **seed** = 42 set at process start for numpy, Python `random`, and sklearn.  
- **Threads:** `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.  
- **Libs:** `polars>=1.0.0` (strictly use modern expression API, no deprecated syntax), `numpy`, `scikit-learn`, `pyarrow`, `joblib`. NO pandas. CPU-only.
- **Timezone handling:** Use Python's native `zoneinfo` explicitly for all timezone-aware datetime arithmetic to prevent Daylight Saving Time (DST) bugs.
- **Numeric clipping:** use `CLIP_MIN` `CLIP_MAX` everywhere.  
- All numeric arrays `float32`. No NaN or inf anywhere.

---

## 3. CONFIG
All tunable parameters centralized here. Change values here only; code should import this config module.

### Paths and IO
- **DATA_GLOB** = `"data/futures/*.parquet"`  
- **FEATURES_OUT** = `"artifacts/features.parquet"`  
- **MANIFEST_PATH** = `"artifacts/manifest.json"`  
- **MODELS_DIR** = `"models/"`  
- **TRADES_OUT** = `"artifacts/trades.csv"`  
- **PNL_OUT** = `"artifacts/pnl_series.csv"`  
- **LOG_DIR** = `"logs/"`  
- **CACHE_DIR** = `"cache/"`  
- **MEMORY_TRACE_OUT** = `"logs/memory_trace.csv"`  

Logging: Add `MEMORY_LOG_ENABLED=True`. The system must print the peak RSS at the end of every major pipeline stage (Data Look, Feature Gen, Train, Backtest).

### Environment and determinism
- **SEED** = `42`  
- **OMP_NUM_THREADS** = `1`  
- **OPENBLAS_NUM_THREADS** = `1`  
- **MKL_NUM_THREADS** = `1`  
- **SKLEARN_N_JOBS** = `1`

### Numeric guards and constants
- **EPS** = `1e-9`  
- **CLIP_MIN** = `-10.0`  
- **CLIP_MAX** = `10.0`  
- **DTYPE** = `float32`  
- **TIMEZONE** = `"America/New_York"`  
- **PRE_POST_CLIP_LOGGING** = `True`  
- **DEBUG_FLOAT64_MODE** = `False`  

### Memory and hardware limits
- **RAM_CAP_BYTES** = `14 * 1024**3`  # 14GB  
- **RSS_STOP_BYTES** = `13.5 * 1024**3`  # 13.5GB runtime stop threshold  
- **STORAGE_MIN_GB** = `200`  
- **ROWS_PER_CHUNK_MAX** = `5_000_000`  
- **MEMORY_SAFETY_MARGIN** = `0.95`  
- **MEMORY_RSS_CHECKPOINT_INTERVAL_SEC** = `10`  
- **MEMORY_RSS_CHECKPOINTS_BEFORE_STOP** = `3`  

### Data load and collect
- **DATA_SCAN_GLOB** = `DATA_GLOB`  
- **LAZY_PUSHDOWN_FILTERS** = `True`  
- **COLLECT_PARTITION_ROWS** = `True`  
- **STABLE_SORT_KEYS** = `["session_id", "ts_event", "row_id"]`

### Session and resampling
- **SESSION_START_LOCAL** = `"18:00"`  
- **SESSION_END_LOCAL** = `"16:00"`  
- **SESSION_TZ** = `TIMEZONE`  
- **RESAMPLE_RULES** = `{"O":"first","H":"max","L":"min","C":"last","V":"sum"}`

### Cleaning rules
- **DROP_VOLUME_ZERO** = `True`  
- **ALLOW_FFILL_4H_MAPPING_ONLY** = `True`  
- **DROP_INCOMPLETE_ROWS** = `True`  
- **REPLACE_INF_NAN_WITH** = `0.0`

### Base feature windows
- **ROLL_WINDOWS** = `[5, 10, 20, 50]`  
- **ROLL_WINDOW_MIN_ROWS** = `max(ROLL_WINDOWS)`  

### Feature expansion
- **FEATURE_TRANSFORMS** = `["lags", "ratios", "z_scores", "pairwise_products_limited"]`  
- **MAX_PAIRWISE_INTERACTIONS** = `500`  
- **TEMPORAL_BUCKETS** = `["early","mid","late"]`  

### Regime and HTF
- **VOL_MEDIAN_WINDOW** = `20`  
- **VOL_SMOOTH_WINDOW** = `5`  
- **REGIME_HIGH_THRESH** = `0.6`  
- **REGIME_LOW_THRESH** = `0.4`  
- **REGIME_MISSING_DEFAULT** = `0`

### Targets
- **TARGET_5M_HORIZON** = `1`  
- **TARGET_1H_RESAMPLE_RULE** = `"1H"`  
- **MAGNITUDE_THRESHOLD** = `0.002`  
- **PROB_TARGET_THRESHOLD** = `0.005`  

### 1H mapping
- **1H_BLOCK_DEFINITION** = `"explicit_per_session"`  
- **1H_FORWARD_FILL_ALLOWED** = `True`  
- **DST_AWARE_1H_TESTS** = `True`  
- **1H_PARTIAL_BLOCK_MIN_MINUTES** = `15`  

### Correlation filter
- **CORR_THRESHOLD** = `0.95`  
- **CORR_TIE_BREAKER** = `["variance_desc", "name_lexicographic"]`  
- **CORR_ACCUMULATION_MODE** = `"compensated_float64_then_downcast"`  

### Nonlinear discovery ExtraTrees
- **DISCOVERY_METHOD** = `"ExtraTrees"`  
- **DISCOVERY_WINDOW_DAYS** = `60`  
- **BOOTSTRAP_FOLDS** = `30`  
- **EXTRA_TREES_PARAMS** = {  
  `"random_state": 42,`  
  `"n_jobs": 1,`  
  `"n_estimators": 100,`  
  `"max_depth": 12,`  
  `"max_features": 0.3,`  
  `"bootstrap": False`  
}  
- **SELECTION_FREQ_THRESHOLD** = `0.75`  
- **CUMULATIVE_IMPORTANCE_THRESHOLD** = `0.95`  
- **SIGN_CONSISTENCY_THRESHOLD** = `0.8`  
- **MIN_SELECTED_FEATURES** = `10`  
- **MAX_SELECTED_FEATURES** = `1000`  
- **DISCOVERY_SENSITIVITY_REPORT** = `True`  

### Orthogonalization and PCA
- **ORTHOGONALIZE** = `False`  
- **PCA_TOP_COMPONENTS** = `5`  

### Manifest and cache
- **MANIFEST_FIELDS** = `["feature_names","dtypes","scaler_mean","scaler_scale","selection_seed","selection_date","selection_model","selection_params","selected_K","cumulative_importance","stability_stats"]`  
- **CACHE_KEY_COMPONENTS** = `["row_count","ts_min","ts_max","dtypes","file_size","mtime","config_hash","seed","manifest_hash"]`  
- **CI_SMOKE_DISCOVERY** = `True`  

### Walkforward
- **WF_TRAIN_DAYS** = `60`  
- **WF_TEST_DAYS** = `1`  
- **WF_STEP_DAYS** = `1`  
- **WF_PRECOMPUTE_INDICES** = `True`  

### Models Ridge
- **SCALER_CLASS** = `"StandardScaler"`  
- **RIDGE_PARAMS** = `{"alpha": 1.0, "solver": "cholesky", "fit_intercept": True, "random_state": 42}`  
- **RIDGE_N_JOBS** = `1`  
- **CLASS_WEIGHT_METHOD** = `"deterministic_from_counts"`  

### Stacking and calibration
- **ENABLE_STACKING** = `False`  
- **STACKER_MODEL** = `"Ridge"`  
- **CALIBRATION_METHOD** = `"platt"`  
- **CALIBRATION_CV_FOLDS** = `5`

### Execution and risk
- **EXECUTE_AT** = `"open[t+1]"`  
- **SLIPPAGE_K** = `0.001`  
- **VOL_PENALTY** = `0.005` 
- **SLIPPAGE_STRESS_PCT** = `0.5`  
- **COMMISSION_PER_TRADE** = `0.00002`  
- **TARGET_VOL** = `0.01`  
- **MAX_LEVERAGE** = `3.0`  
- **MAX_POS_CHANGE_PER_MIN** = `0.1`  
- **FLAT_BEFORE_CLOSE_MINUTES** = `5`  

### Metrics and reporting
- **METRICS_TO_COMPUTE** = `["Sharpe","MaxDrawdown","Turnover","HitRate","AvgWin","AvgLoss","MAE"]`  
- **DEFAULT_METRICS_IF_NO_TRADES** = `{"Sharpe":0,"MaxDrawdown":0,"Turnover":0,"HitRate":0}`  
- **ANNUALIZATION_FACTOR** = `66528`  # 264 5-min bars per 22h session * 252 days

### Tests and thresholds
- **REPRO_HASH_ALGORITHM** = `"sha256"`  
- **DISCOVERY_REPRO_TEST** = `True`  
- **MIN_STABILITY_FEATURES** = `5`  

### Trading terminology mapping
- **MAPPING_FILE_PATH** = `"config/term_to_canonical_mapping.yaml"`  
- **ENFORCE_MAPPING_AT_INGEST** = `True`  
- **MAPPING_HASH_IN_MANIFEST** = `True`  
- **MAPPING_VERSION_FIELD** = `"mapping_version"`  
- **MAPPING_VALIDATION_CI_TESTS** = `True`  
- **MAPPING_SCHEMA** = `{term: {canonical_name, dtype, formula_ref, clip, unit_tests}}`

### Baseline features and discovery pool
- **BASELINE_FEATURES_FILE** = `"config/baseline_features.yaml"`  
- **DISCOVERY_INITIAL_POOL** = `"baseline_plus_generated"`  
- **BASELINE_FEATURES_FROZEN** = `True`  
- **BASELINE_FEATURES_PERSIST_PATH** = `"artifacts/baseline_feature_matrix.parquet"`  
- **BASELINE_FEATURES_HASH_FIELD** = `"baseline_features_hash"`

### New concrete constants
- **ROW_GROUP_SIZE** = `65536`  
- **SYNTHETIC_FIXTURE_PATH** = `"tests/fixtures/synthetic_1min_fixture.parquet"`  
- **ENTRYPOINT_FN** = `run_pipeline(data_glob: str, config_path: str, out_dir: str) -> None`  

---

## 4. HARDWARE LIMITS AND FAILSAFE
- Process constraints: RAM cap < 14GB, Storage 200GB.  
- Load per-contract or per-year dynamically to stay safely below 14GB. Any single-chunk memory footprint violation triggers immediate pipeline abort.
- Set thread envs at process initialization context before loading math/ML libraries.

---

## 5. DATA LOAD POLARS LAZY AND LAZY TO EAGER BOUNDARY
- `scan_parquet(DATA_SCAN_GLOB)` lazy execution.  
- **LAZY:** initial scan, pushdown filters, light transformations.  
- **EAGER collect:** dedupe on `ts_event`, compute session boundaries per-partition integrity checks. Collect exactly once per partition.  
- Precompute session boundaries explicitly before any rolling or lookback operations.  
- **Stable sort** by `session_id, ts_event, row_id` before any aggregation to guarantee deterministic resampling.

---

## 6. POST COLLECT VALIDATION AND MEMORY PRECHECK
- After `collect()`:
  - Assert `ts_event` strictly increasing, no nulls. Ensure high >= low, open & close in range [low, high], volume >= 0.  
  - Compute `estimated_memory_bytes = sum(approx_bytes_per_column)` and `avg_row_bytes = estimated_memory_bytes / row_count`. Log metrics via standardized logger.  
  - If `estimated_memory_bytes > RAM_CAP_BYTES`, STOP and throw memory-safe error.  
  - Calculate `rows_per_chunk = min(ROWS_PER_CHUNK_MAX, floor(RAM_CAP_BYTES / avg_row_bytes))`.  
  - Log metadata. Persist trace headers to `MEMORY_TRACE_OUT`.

---

## 7. SESSION DEFINITION AND FILTERING
- **Globex Session Rollover Logic:** To prevent slicing a single 23/5 session across calendar boundaries, compute a logical `session_id`. If `ts_event` (localized to `America/New_York`) has an hour >= 18:00, shift the date component forward by 6 hours (`date = ts_event + 6 hours`) to map Sunday night/Monday morning to the same logical trading day. No cross-session mixing.
- Filter and retain rows strictly within `SESSION_START_LOCAL` to `SESSION_END_LOCAL`. Drop out-of-session tails.  
- **Base Resampling:** Resample raw metrics into canonical 5-minute RTH bars (`5m`) per session. Use session-aware boundaries. 
- **Aggregation Rules:** O first, H max, L min, C last, V sum. Enforce `stable_sort` by `session_id, ts_event, row_id` prior to resampling.

---

## 8. CLEANING RULES
- No lookahead forward-fills or backward-fills except where explicitly permitted for 4H mapping.  
- Drop incomplete rows and rows where `volume == 0`.  
- Replace `inf` and `NaN` with `REPLACE_INF_NAN_WITH` (0.0) immediately post-computation to stabilize numeric arrays.

---

## 9. BASE FEATURES EAGER
- `ret = log(close / close.shift(1))` past-only, executed strictly per-session block.  
- Rolling windows past-only per-session: windows [5, 10, 20, 50] tracking mean, std.  
- Drop zero-variance columns. Cast matrix to `float32`. Clip values to `CLIP_MIN`, `CLIP_MAX`.

---

## 10. FEATURE EXPANSION (DETERMINISTIC CONSTRAINTS)
All features float32, deterministic, past-only, session-aware, clipped. Post-process NaN/inf with 0.0, drop zero-variance features, and lexicographically sort column names.

### Range and vol
- `high_low_range_norm = (high - low) / max(close, EPS)`  
- `true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))`  
- `atr_14 = rolling_mean(true_range, 14)` per session  
- `price_z_20 = (close - mean20) / std20`

### Trend
- `dist_ma_20 = (close - MA20) / MA20`  
- `dist_ma_50 = (close - MA50) / MA50`  
- `pos_in_range_20 = (close - min20) / (max20 - min20 + EPS)`

### Volume and microstructure (Tick-Rule Proxy)
- `log_volume = log(volume)` if `volume > 0` else `0`  
- `volume_z_20 = (volume - mean20) / std20`  
- **Signed Volume Imbalance Proxy:** Approximate via the Tick Rule: If `close > open`, assign `volume` as `buy_volume`. If `close < open`, assign as `sell_volume`. If `close == open`, carry forward the sign of the previous bar's return. Compute proxy: `(buy_volume - sell_volume) / max(volume, EPS)`.
- **Spread Proxy:** `(ask - bid) / max(mid, EPS)`. If bid/ask explicitly missing, proxy as `(high - low) / max(close, EPS)`.
- **Pairwise Interactions Capping Guardrail:** To generate pairwise products up to `MAX_PAIRWISE_INTERACTIONS` (500) safely without memory spikes, you MUST use a Python `itertools.combinations` generator over lexicographically sorted baseline features ($A < B$). Append features to a lazy evaluation list one by one, and `break` the generation loop the exact millisecond the unique count hits 500. Pass only this strictly sized definition list to Polars to avoid eager matrix explosion.

### Session
- `session_pos = index_in_session / (session_length - 1)`  
- `session_len = session_length`  
- Temporal buckets session segment hash: early/mid/late via deterministic hashing to isolate intraday seasonality regimes.

### Momentum and volatility
- `ret_1, ret_mean_5, 10, 20, 50, ret_std_5, 10, 20, 50`  
- Short-term realized volatility and EWMA volatility (HAR-lite structure).

---

## 11. NUMERIC GUARDS
Assert zero NaN/inf visibility post-clean. If `DEBUG_FLOAT64_MODE` is True, execute feature engineering logic in float64, downcast, and log precision drift statistics.

---

## 12. REGIME
- `vol_20` rolling_median `VOL_MEDIAN_WINDOW` smoothed via rolling_mean `VOL_SMOOTH_WINDOW`.  
- `regime = 1` if `smooth >= REGIME_HIGH_THRESH`, `0` if `smooth < REGIME_LOW_THRESH`, else carry forward `lag1` value. Missing defaults to `REGIME_MISSING_DEFAULT`.

---

## 13. TARGETS
- **A 5m Target Execution Alignment:** To eliminate trade execution friction mismatch where `EXECUTE_AT = "open[t+1]"`, define `target_5m[t] = sign(log(close[t+1] / open[t+1]))`. Set to 0 if $t+1$ crosses session boundaries or lands outside session.
- **B 1H_target No-Leak Rule:** Resample to 1H session-aware blocks. `1H_target = sign(log(1H_close[t+1] / 1H_close[t]))`. Map 1H_target back to intraday rows by forward-shift and forward-fill *only* within explicit 1H blocks. To prevent lookahead leakage, the 1H return calculated inside an hour block cannot be visible to intraday rows until that 1H block has fully closed and its calculation timestamp is <= $t$. Values mapped: -1, 0, 1.  
- **C Magnitude target:** Indicator function tracking thresholded velocity shifts: `I(|log_ret| > MAGNITUDE_THRESHOLD)`.  
- **D Probabilistic target:** Evaluates multi-period probability density scaling for post-fit model calibration.

---

## 14. 1H MAPPING RULES (NO LEAK / DST GUARD)
- Define explicit 1H blocks `(1H_start_ts, 1H_end_ts)` per session. Blocks computed deterministically from session start using timezone-aware arithmetic: **block start inclusive, block end exclusive**.  
- **Inclusion rule:** a 1H block is valid if duration >= `1H_PARTIAL_BLOCK_MIN_MINUTES`. Partial blocks shorter than this (e.g., due to DST adjustments) are invalid; mapped 1H_target is set to `0`.  
- **Forward-fill rule:** forward-fill a 1H_target into intraday rows **only if** both the current 1H close and the next 1H close exist within the same session boundaries. Otherwise default to `0`.  
- **Deterministic tie‑breakers:** when an intraday row falls exactly on an hourly boundary, assign it to the block whose start timestamp equals the row timestamp (start-inclusive).

---

## 15. CORRELATION FILTER (TRAIN SPLIT ONLY)
- On train partition, compute variances, sorting features by variance descending, then name lexicographically.  
- **Numeric accumulation:** Cast candidate feature columns to `pl.Float64` natively. Compute Pearson correlations utilizing Polars' native `.corr()` expressions (which employ numerically stable two-pass variance/covariance algorithms in Rust). Denominators must be clamped with `EPS`. Downcast the resulting correlation matrix back to `pl.Float32` for storage.
- Drop features exceeding `CORR_THRESHOLD` adhering to deterministic variance rules.
- **Exclusion zone:** Target columns (`target_5m`, `1H_target`) and regime identifiers MUST be excluded from the correlation filter and pairwise interaction logic to prevent extreme lookahead bias.

---

## 16. NONLINEAR FEATURE DISCOVERY (JOBLIB LOKY ISOLATION)
- Run ONLY on first walkforward train window slice (`DISCOVERY_WINDOW_DAYS`). No leak allowed.
- **Execution model (Process Safety):** You MUST run each bootstrap fold `fit` inside a dedicated subprocess utilizing `joblib.Parallel` with `backend='loky'` and `max_nbytes=None`. This ensures complete memory isolation between folds, circumventing Python GIL compilation leaks or shared reference corruption.
- **RNG Seed Inheritance:** Inherit a deterministic RNG seed derived by running an `HMAC-SHA256` digest over the string sequence: `f"{SEED}:{fold_index}"`. Persist the fold seed map directly into the manifest.
- **Abort protocol:** Parent process samples RSS at `MEMORY_RSS_CHECKPOINT_INTERVAL_SEC`. If `RSS_STOP_BYTES` is breached for `MEMORY_RSS_CHECKPOINTS_BEFORE_STOP` consecutive samples, signal immediate worker termination. Write out partial `manifest.json` flag tracking state as `aborted`. Pipeline must support deterministic resume functionality using checkpoint manifests.
- **Baseline Feature candidates:** Load and validate `config/baseline_features.yaml` containing the canonical 40 features listed below. They are frozen and cannot be pruned from the pool prior to fitting.

### Baseline 40 OHLCV Feature Definitions
1. `feature_ret_1` to 4. `feature_ret_20` — Log returns across lags [1, 5, 10, 20].
5. `feature_ma_5` to 7. `feature_ma_50` — Simple moving averages of close price.
8. `feature_dist_ma_20`, 9. `feature_dist_ma_50` — Normalized price distance to moving averages.
10. `feature_ma_slope_20` — Linear regression slope of close over 20 bars normalized by SMA20.
11. `feature_price_z_20`, 12. `feature_price_z_50` — Z-score scalers for pricing tracking.
13. `feature_high_low_range_norm` — (high - low) / max(close, EPS).
14. `feature_true_range` — Volatility metrics tracking daily range gaps.
15. `feature_atr_14` — Average True Range tracking window.
16. `feature_realized_vol_5`, 17. `feature_realized_vol_20` — Sample standard deviation arrays.
18. `feature_ewma_vol_20` — Exponentially Weighted Moving Average volatility vector.
19. `feature_price_momentum_5`, 20. `feature_price_momentum_10` — Pure directional momentum velocity.
21. `feature_mom_z_5`, 22. `feature_mom_z_10` — Normalized momentum shifts.
23. `feature_rsi_14` — Relative Strength Index via Wilder smoothing.
24. `feature_macd`, 25. `feature_macd_signal` — Convergence-divergence signal loops.
26. `feature_stoch_k` — Stochastic range bounded track over 14 bars.
27. `feature_log_volume` — Log transformed size profiles.
28. `feature_volume_z_20` — Volume activity normalization.
29. `feature_obv` — Session localized On-Balance Volume calculations.
30. `feature_signed_bar_strength` — Intraday price location ratio tracking.
31. `feature_volume_price_divergence` — Vector tracking size vs return velocity.
32. `feature_spread_proxy` — Microstructure proxy calculation framework.
33. `feature_session_pos` — Linear temporal position scalar within trading session.
34. `feature_time_of_day_bucket` — Categorical season maps [early, mid, late].
35. `feature_1h_bias` — Safe resampled target return mappings without leak metrics.
36. `feature_session_volatility` — Dynamic rolling risk component per session boundary.
37. `feature_pair_prod_template`, 38. `feature_ratio_template` — Deterministic generation placeholders.
39. `feature_pca_comp_1`, 40. `feature_pca_comp_2` — Principal components trained strictly on in-sample folds.

---

## 17. FEATURE ORTHOGONALIZATION AND STABILITY
- Optionally project selected features onto the orthogonal basis of the top `PCA_TOP_COMPONENTS` components computed on the in-sample training split. Enforce strict frequency selection criteria across folds.

---

## 18. MANIFEST AND CANONICAL PARQUET WRITER SETTINGS
- Persist structured metadata into `manifest.json`. Include schema, scaling definitions, baseline hashes, and processing state indicators.
- **Canonical Feature Matrix Serialization:** To guarantee byte-level SHA256 reproducibility across different operating systems, all frozen feature matrices written to disk MUST employ the following explicit settings using `pyarrow.parquet.write_table`:
  - **Format Version:** `2.0` 
  - **Endianness:** Little-endian execution arrays
  - **Compression:** `snappy`
  - **Row Group Size:** Fixed exactly to the integer value of `ROW_GROUP_SIZE` (65536)
  - **Column Ordering:** Alphabetical/Lexicographic sorting forced across column arrays before serialization.
  - **Metadata Pruning:** Explicitly strip non-essential structural metadata before calculating filesystem hashes.

---

## 19. FEATURE FREEZE
- Finalize selection matrix definitions. Enforce matching feature names across train, test, and out-of-sample execution engines. Assert identity matrix hashes via `test_serialization_repro.py`.

---

## 20. WALKFORWARD SETUP & VOLATILITY-SCALED TCA
- **Window Mechanics:** Implement a strict rolling-window walkforward cross-validation. `WF_TRAIN_DAYS` = 60 is a fixed-size rolling window; at each step, the train window advances by `WF_STEP_DAYS` = 1 day, dropping the oldest day's indices. **The walkforward Ridge model MUST use `target_5m` as its exclusive `y` variable for training and prediction.**
- **Feature Scaling Pipeline:** For each walkforward step, fit the `StandardScaler` strictly on the training partition data. Transform both the train slice and the out-of-sample `WF_TEST_DAYS` = 1 day test slice using those coefficients. **CRITICAL EXCLUSIONS:** You must explicitly exclude `target_5m`, `1H_target`, `regime`, and any structural metadata columns from scaling.
- **Execution Simulation & Volatility-Scaled TCA:** At the open of bar `t+1`, execute the model's target position allocation change.
  - **Position Sizing Rule:** Calculate size explicitly using: `target_position = clip((ridge_prediction / max(feature_ewma_vol_20, EPS)) * TARGET_VOL, -MAX_LEVERAGE, MAX_LEVERAGE)`. Failsafe: if feature generation fails for an out-of-sample bar `t`, immediately force `target_position = 0.0`.
  - **Dynamic Slippage & Costs:** Transaction costs must be modeled dynamically to simulate realistic high-variance regime liquidity drains: 
    $$\text{cost} = \text{COMMISSION\_PER\_TRADE} + (\text{SLIPPAGE\_K} \times \text{spread\_proxy}) + (\text{VOL\_PENALTY} \times \text{feature\_ewma\_vol\_20})$$
    Subtract this cost metric on every position modification or adjustment. Force risk unwinds (`FLAT_BEFORE_CLOSE_MINUTES`) before session close.

---

## 21. I/O SCHEMA DEFINITIONS
- **Expected Parquet Schema Columns:** `ts_event (int64/string)`, `open (float32)`, `high (float32)`, `low (float32)`, `close (float32)`, `volume (int64)`, `row_id (int64)`, `session_id (string)`. Optionals: `buy_volume (int64)`, `sell_volume (int64)`, `bid (float32)`, `ask (float32)`.

---

## 22. ROWS PER CHUNK FORMULA
- Compute `estimated_memory_bytes = sum(column_count * dtype_size_bytes * row_count_estimate_per_column)`. Integer/Float scales: float32=4, int64=8, string estimate=32 bytes.
- Dynamic row limits calculated via: `rows_per_chunk = min(ROWS_PER_CHUNK_MAX, floor(RAM_CAP_BYTES * MEMORY_SAFETY_MARGIN / avg_row_bytes))`. Include step log outputs in execution stream.

---

## 23. TESTING AND VALIDATION HARNESS (CI REQUISITES)
Provide unit testing verification loops covering:
1. **tests/test_1h_dst.py:** Validates timezone offsets, inclusive start/exclusive end mappings, partial block truncation drops, and boundary conditions.
2. **tests/test_memory_abort.py:** Spikes subprocess worker RSS targets to confirm safe parent tracking and partial manifest persistence state transitions.
3. **tests/test_serialization_repro.py:** Performs double write operations on identical feature footprints to assert matching cryptographic SHA256 string returns.

---

## 24. REPOSITORY LAYOUT MANDATE
Generate complete file implementations for the layout described below:
```text
/config
  config.py                   # Centralized configuration parameters from Section 3
  baseline_features.yaml      # Structured canonical baseline 40 features setup
/src
  /io
    canonical_parquet.py      # PyArrow custom serialization writer logic
  ingest.py                   # Lazy data ingestion, streaming chunks, session sorting
  features.py                 # Modern Polars expression matrix for baseline & expansion
  discovery.py                # Joblib loky fold isolation, discovery, and abort tracking
  walkforward.py              # Rolling window partitions, Ridge model, volatility TCA simulation
  cli.py                      # Production CLI entrypoint execution loop
/tests
  fixtures/make_fixtures.py   # Synthetic minute parquet file generator script
  test_1h_dst.py              # Timezone validation loop
  test_memory_abort.py        # OOM intercept loop
  test_serialization_repro.py # Structural reproducibility check
/ci
  ci_stub.yml                 # Automation instruction pipeline matrix
README.md                     # Run environment overview documentation
requirements.txt              # Pinned packages: polars, pyarrow, numpy, scikit-learn, joblib, pytest

## 25. SCIENTIFIC RIGOR BENCHMARK REQUIREMENT
In src/walkforward.py and the final reporting engine, you MUST compute and output a "Naive Benchmark" tracking series along with the Ridge execution metrics (e.g., a simple 20-period moving average crossover or buy-and-hold rule executing exclusively during session RTH blocks). Output all out-of-sample evaluation metrics (Sharpe, MaxDrawdown, Turnover) relative to this naive benchmark to prove genuine alpha generation.

## 26. GEMINI DIRECTIVE: ZERO TRUNCATION POLICY
You are a Principal Quantitative Software Engineer. Because you possess an extended context window, you are required to generate the full, production-grade codebase for this entire architecture.

Do NOT output pseudo-code.

Do NOT use ... or // TODO placeholders.

Generate complete, fully syntax-validated file blocks matching the schema, data frames, and algorithmic constraints dictated in this specification file. Start generating the complete file matrix now.

## 27. EXPLICIT FIXTURE PAYLOAD
Fixture path: tests/fixtures/synthetic_1min_fixture.parquet

To ensure generated CI tests properly capture DST shifts, use this exact structural schema context when generating the synthetic parquet testing files:

Code snippet
ts_event,open,high,low,close,volume,row_id
2020-03-08T06:00:00Z,100,101,99.5,100.5,1000,1
2020-03-08T06:01:00Z,100.5,101.2,100.2,100.8,1100,2
Include a helper module script tests/fixtures/make_fixtures.py that serializes this dataframe array natively using pyarrow bounded to the section 18 settings.

## 28. MANIFEST JSON ENFORCEMENT SCHEMA
The generated architecture must output and validate artifacts/manifest.json against this precise structural signature format:

JSON
{
  "feature_names": ["feature_ret_1","feature_ma_20","feature_price_z_20"],
  "dtypes": {"feature_ret_1":"float32","feature_ma_20":"float32","feature_price_z_20":"float32"},
  "scaler_mean": {"feature_ret_1":0.0,"feature_ma_20":100.0,"feature_price_z_20":0.0},
  "scaler_scale": {"feature_ret_1":1.0,"feature_ma_20":1.0,"feature_price_z_20":1.0},
  "selection_seed": 42,
  "selection_date": "2026-05-20T00:00:00Z",
  "selection_model": "ExtraTrees",
  "selection_params": {"n_estimators":100,"max_depth":12},
  "selected_K": 120,
  "cumulative_importance": 0.95,
  "stability_stats": {"min_selection_freq":0.75,"sign_consistency":0.85},
  "baseline_feature_list": ["feature_ret_1","feature_ret_5"],
  "baseline_features_hash": "sha256:abcdef...",
  "baseline_feature_matrix_path": "artifacts/baseline_feature_matrix.parquet",
  "serialization_params": {"parquet_version":"2.0","compression":"snappy","row_group_size":65536,"column_ordering":"lexicographic"},
  "discovery_status": "completed",
  "folds": [
    {"fold_index":0,"fold_seed":"hmac:...","fold_status":"completed","importance_file":"fold_0_importances.parquet","rss_before":12345678,"rss_after":23456789}
  ]
}
## 29. ENTRYPOINT SIGNATURE
The core processing script (src/cli.py) and underlying runner frameworks must implement and expose this explicit Python method interface:

Python
def run_pipeline(data_glob: str, config_path: str, out_dir: str) -> None:
    """
    Deterministic pipeline entrypoint.
    - data_glob: glob for input parquet files (e.g., 'data/futures/*.parquet')
    - config_path: path to the config module or JSON
    - out_dir: directory to write artifacts (artifacts/, models/, logs/)
    """
## 30. POST-GENERATION DEVELOPER CHECKLIST
(Generate a DEV_RUN.md file containing these exact verification steps during generation):

Run python -m tests.fixtures.make_fixtures to generate baseline system test fixtures.

Run python -m src.cli run --data tests/fixtures/synthetic_1min_fixture.parquet --config config/config.py --out artifacts/ and explicitly verify:

artifacts/baseline_feature_matrix.parquet matches baseline_features_hash registered inside the manifest.

artifacts/manifest.json maps without format exception to the enforcement schema definitions.

Cryptographic SHA256 matches identically across sequential execution passes over unchanging inputs.

Execute pytest -q to verify the framework pass criteria on the localized testing fixtures.

Run a single test production year slice to guarantee execution stays bounded under RAM_CAP_BYTES.

Advance to full production deployment scaling arrays across the 15-year target window block.

##31. FINAL NOTE TO GEMINI PRO EXTENDED
Generate a complete, fully functioning runnable package implementing the exact framework described above. Adhere strictly to the explicit function structures, directory setups, Polars expressions, chunking pipelines, and dynamic TCA requirements mapped here. Prioritize determinism, memory isolation containment via loky, zero-lookahead tracking rules, and byte-level write reproducibility over all other secondary parameters. Ensure absolute compliance across all file modules.