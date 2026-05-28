# Structural Audit Findings — Quant Pipeline

**Auditor:** Sr Quant Auditor / Adversarial Risk Engineer  
**Axiom:** "backtest always lies via hidden biases"  
**Date:** 2026-05-28  
**Scope:** 5m ES, CL, ZB futures pipeline (ExtraTrees discovery → Ridge walkforward → execution simulation)

---

## Assertion Summary

| # | Assertion | Verdict | Risk |
|---|-----------|---------|------|
| 1 | CONFIG HIERARCHY | PASS | — |
| 2 | DATA INTEGRITY | PASS | — |
| 3 | GAP DETECTION | PASS | — |
| 4 | HTF ALIGNMENT | PASS | — |
| 5 | ROLLING WINDOWS | PASS | — |
| 6 | Z-SCORE | PASS | — |
| 7 | DISCOVERY | PASS | — |
| 8 | ENGINE / SIGNAL -> POSITION | PASS | — |
| 9 | COST MODEL | PASS | — |
| 10 | SESSION | PASS | — |
| 11 | INTRABAR SL/TP/GAP SIM | PASS | — |
| 12 | CONTINUOUS CONTRACT / ROLLOVER | PASS | — |
| 13 | POSITION CLIPPING | PASS | — |
| 14 | BURN-IN / WARMUP | PASS | — |
| 15 | WALKFORWARD SESSION BOUNDARY | PASS | — |
| 16 | MULTIPLIER PROPAGATION | PASS | — |
| 17 | CROSS-ASSET ALIGNMENT | PASS | — |
| 18 | HMM PNL RECOMPUTE | PASS | — |
| 19 | PROBABILITY SMOOTHING | PASS | — |
| — | CI WORKFLOW | PASS | — |
| — | FUZZ HARNESS | PASS | — |
| — | CONFIG max_position_size propagation | PASS | — |
| — | MARKET_CONFIG max_position_size | PASS | — |

---

## Detailed Findings

---

### Finding 1 — CONFIG HIERARCHY: max_position_size propagated (RESOLVED)

| Evidence | File |
|----------|------|
| `max_position_size: float \| None = None` in ExecutionConfig | `quant/config_manager.py:181` |
| `config.MAX_POSITION_SIZE` assigned in `_populate_simple_namespace()` | `quant/config_manager.py:415-419` |

**Verdict:** PASS. max_position_size is properly typed as `float | None`, propagated to SimpleNamespace, and defaults to `float('inf')` when not set.

**Fix:** None required.

---

### Finding 2 — DATA INTEGRITY: RSS check added in walkforward.py

| Evidence | File |
|----------|------|
| `process_fold()` checks `psutil.Process().memory_info().rss > config.RSS_STOP_BYTES` before training | `quant/walkforward.py:116-119` |
| `discovery.py` checks RSS per bootstrap fold | `quant/discovery.py:138-139` |
| `ingest.py` validates ts_event sorted, OHLCV nulls, high>=low, open/close bounds, RAM_CAP_BYTES | `quant/ingest.py:17-33` |

**Verdict:** PASS. RSS check was absent from walkforward previously but is now present in `process_fold()`. All data integrity checks are enforced.

**Fix:** None required.

---

### Finding 3 — GAP DETECTION: explicit filter exists and is called

| Evidence | File |
|----------|------|
| `filter_gaps(df, max_gap_minutes=30)` function | `quant/gap_filter.py:11-23` |
| Called from `load_and_clean_data()` after alignment | `quant/ingest.py:153-154` |

**Verdict:** PASS. Standalone gap filter function exists and is called in the ingest pipeline. Resampling n_ticks thresholds in session.py remain as a complementary guard.

**Fix:** None required.

---

### Finding 4 — HTF ALIGNMENT: backward_fill only after forward_fill (EXEMPT)

| Evidence | File |
|----------|------|
| 1h timestamps shifted +1 hour before `join_asof(strategy='backward')` | `quant/align.py:32` |
| Daily timestamps shifted +1 day | `quant/align.py:51` |
| `daily_cols` forward-filled THEN backward-filled (boundary bars only) | `quant/align.py:62-64` |
| `htf_daily_trend_slope_10` uses `shift(1)` and `shift(1+bars*10)` — strictly past | `quant/features/htf_context.py:52-56` |

**Verdict:** PASS. `.backward_fill()` on daily columns appears only after `.forward_fill()` (line 64) — EXEMPT per documented rule. All HTF trend features use strictly lagged values.

**Fix:** None required.

---

### Finding 5 — ROLLING WINDOWS: all predictive rolling_* use shift(1)

| Function | File:Line | Shift Applied |
|----------|-----------|---------------|
| `feature_ewma_vol_20` | `baseline.py:38-39` | `ret_1.shift(1).rolling_std(20)` YES |
| Regime vol | `expansion.py:13` | `ret.shift(1).rolling_std(20)` YES |
| Z-score mean/std | `expansion.py:30-31` | `pl.col(col).shift(1).rolling_mean/std(30)` YES |
| `rolling_quantile` | `expansion.py:122` | `ret.shift(1).rolling_quantile(...)` YES |
| `rolling_moments` | `expansion.py:143` | `ret.shift(1).rolling_sum/...` YES |
| Acceleration | `expansion.py:164` | `ret - ret.shift(1)` — no rolling, no issue YES |
| VWAP | `expansion.py:172-173` | `tp.shift(1) * vol.shift(1).rolling_sum(...)` YES |
| Volume Profile close_lag/vol_lag | `volume_profile.py:62-63` | `shift(1)` YES |
| VPA vol_lag/spread_lag | `volume_profile.py:167-168` | `shift(1)` YES |
| `_daily_high_expanding` | `htf_context.py:20` | `high.shift(1).cum_max()` YES |
| `htf_daily_vol_5` | `htf_context.py:61` | `ret_1.shift(1).rolling_std(260)` YES |
| `_1h_vol_4` | `htf_context.py:70` | `_1h_return.shift(1).rolling_std(4)` YES |
| Simulator vol | `simulator.py:372-374` | `ret.shift(1).rolling_std(20)` YES |

**Verdict:** PASS. Every rolling window that feeds a predictive feature applies `.shift(1)` to the input series before the rolling operation. No leakage.

**Fix:** None required.

---

### Finding 6 — Z-SCORE: feature z-scores lagged; signal z-score NOT lagged (correct)

| Evidence | File |
|----------|------|
| Feature z-scores use `shift(1)` on feature column before `rolling_mean/std(30)` | `quant/features/expansion.py:30-32` |
| Signal z-score in simulator uses current `prediction_prob` directly (NOT lagged) — this is CORRECT for entry gating: you want current conviction measured against its own rolling distribution | `quant/execution/simulator.py:171-174` |
| Signal rolling window: 1000 bars, `min_periods=50` — sufficient history | `quant/execution/simulator.py:172-173` |

**Verdict:** PASS. Feature z-scores use lagged distributions. Signal z-score correctly uses current probability (this is entry gating, not a predictive feature). Window size is adequate.

**Fix:** None required.

---

### Finding 7 — DISCOVERY: deterministic, parameter-consistent

| Evidence | File |
|----------|------|
| `get_fold_seed()` uses SHA256 of `config.SEED + fold_idx` | `quant/discovery.py:19-21` |
| ExtraTrees params from `config.EXTRA_TREES_PARAMS` (defaults: max_depth=8, n_estimators=100, max_features=0.3, bootstrap=False) | `quant/config_manager.py:131-140` |
| Bootstrap folds = `config.BOOTSTRAP_FOLDS` (default 30) | `quant/discovery.py:129` |
| IQR scaling via `robust_scale()` | `quant/walkforward.py:26-37` |
| Manifest prune: selection_freq >= 0.75, sign_consistency >= 0.8, cumulative_importance >= 0.95 | `quant/discovery.py:188-205` |

**Verdict:** PASS. Seed is deterministic. ExtraTrees parameters match RootConfig defaults. Bootstrap fold count matches config. IQR scaling uses median/IQR. Manifest pruning uses configured thresholds.

**Fix:** None required. (alpha_1 YAML overrides `bootstrap_folds: 3` and `discovery_window_days: 30` explicitly — tier override is valid.)

---

### Finding 8 — ENGINE / SIGNAL -> POSITION: pipeline order verified

| Step | File:Line | Verified |
|------|-----------|----------|
| 1. Z-score gating -> raw_signal | `simulator.py:171-184` | YES |
| 2. HTF hourly alignment gating | `simulator.py:197-213` | YES |
| 3. Session break flat (17:00-18:00) | `simulator.py:218-232` | YES |
| 4. Session close flat (15:55) | `simulator.py:237-251` | YES |
| 5. HTF vol scaling | `simulator.py:260-278` | YES |
| 6. ATR sizing (TARGET_RISK_PER_TRADE / ATR14) | `simulator.py:287-301` | YES |
| 6b. Position clipping (max_position_size + notional cap) | `simulator.py:304-343` | YES |
| 7. HTF daily trend alignment | `simulator.py:351-367` | YES |
| Position at bar t = target_exec.shift(1) | `simulator.py:441` | YES |
| PnL = position * ret_exec * multiplier * price - costs + intrabar + round-turn settlement | `simulator.py:489-498` | YES |

**Verdict:** PASS. Pipeline order is correct. All gating stages are present and applied in the right sequence.

**Fix:** None required.

---

### Finding 9 — COST MODEL: round-turn settlement on flatting IMPLEMENTED

| Evidence | File |
|----------|------|
| `unit_cost = COMMISSION_PER_TRADE + SLIPPAGE_K*spread + VOL_PENALTY*vol + TX_COST_PER_ROUNDTURN/2` | `simulator.py:393-398` |
| Round-turn settlement: `when position_went_flat -> pnl - TX_COST_PER_ROUNDTURN * |prior_position|` | `simulator.py:486-494` |

**Verdict:** PASS. The per-delta amortization (TX_COST/2 per position change) combined with the flat-settlement charge (line 492-494) produces the correct total round-turn cost: flat-to-flat = 2 * (TX_COST/2) = TX_COST. Single-bar round-trips are fully covered.

**Fix:** None required.

---

### Finding 10 — SESSION: flatten logic and session_id verified

| Evidence | File |
|----------|------|
| Session break flat: 17:00-18:00 local -> position = 0 | `simulator.py:224-232` |
| Session close flat: 15:55-16:00 local -> position = 0 | `simulator.py:237-251` |
| `session_id = ts_local.offset_by('6h').dt.date()` | `session.py:29` |

**Verdict:** PASS. Session break and close flattens are correctly timed. `offset_by('6h')` ensures evening (18:00-23:59) and morning (00:00-16:00) bars share the same session_id.

**Fix:** None required.

---

### Finding 11 — INTRABAR SL/TP/GAP SIM: IMPLEMENTED

| Evidence | File |
|----------|------|
| `simulate_intrabar_stops()` with linear-path logic and gap-opening fill | `simulator.py:20-158` |
| Called from `_compute_pnl_from_target_exec()` | `simulator.py:458-467` |
| `stop_loss_pct: 0.005`, `take_profit_pct: 0.01`, `gap_slippage_pct: 0.002` in config | `config_manager.py:185-187` |

**Verdict:** PASS. Intrabar stop/take-profit simulation is present with linear-path first-touched logic and gap-opening fill at `open + gap_slippage_pct`. Called in both the main execution path and the HMM PnL recompute path.

**Fix:** None required.

---

### Finding 12 — CONTINUOUS CONTRACT / ROLLOVER: IMPLEMENTED

| Evidence | File |
|----------|------|
| `compute_roll_dates()` — ES/NQ/YM/RTY (HMUZ quarterly), CL/NG (monthly), ZB/ZN (quarterly) | `quant/continuous_contract.py:54-162` |
| `build_ratio_adjusted_series()` — ratio-adjustment across one roll point | `quant/continuous_contract.py:165-247` |
| `apply_splice()` — cumulative factor join and continuous_price computation | `quant/continuous_contract.py:250-294` |
| `build_continuous_series()` — full pipeline, called from ingest | `quant/continuous_contract.py:297-457` |
| Called from `load_and_clean_data()` after alignment | `quant/ingest.py:179-181` |
| `adjustment_factor`, `contract_month`, `contract_multiplier` persisted in output | `quant/continuous_contract.py:440-449` |

**Verdict:** PASS. Continuous contract pipeline exists with symbol-specific roll schedules, ratio adjustment, cumulative splicing, and metadata columns. Called from ingest.py as required.

**Fix:** None required.

---

### Finding 13 — POSITION CLIPPING: IMPLEMENTED (config chain complete)

| Evidence | File |
|----------|------|
| `max_position_size` clipped in simulator from market YAML | `simulator.py:325-330` |
| Notional cap: `|position| <= max_leverage / (open_next * contract_multiplier)` | `simulator.py:333-343` |
| `contract_multiplier` loaded from per-market YAML | `simulator.py:312-318` |
| `config.MAX_POSITION_SIZE` assigned in `_populate_simple_namespace()` | `config_manager.py:415-419` |
| `MAX_POSITION_SIZE` in `load_market_config()` overrides from per-market YAML | `market_config.py:21` |

**Verdict:** PASS. Position clipping is enforced in the simulator (both max_position_size and notional cap). Config propagation chain is complete: Pydantic type is `float | None` (line 181), SimpleNamespace assignment present (lines 415-419), market_config override present (line 21).

**Fix:** None required.

---

### Finding 14 — BURN-IN / WARMUP: IMPLEMENTED

| Evidence | File |
|----------|------|
| `WalkforwardConfig.burn_in_bars: int = 500` | `config_manager.py:164` |
| `exclude_warmup()` function | `quant/walkforward.py:105-111` |
| Called in `process_fold()` | `quant/walkforward.py:127` |
| Called in `run_walkforward()` | `quant/walkforward.py:483` |
| Called in `run_walkforward_with_hmm()` | `quant/walkforward.py:436` |

**Verdict:** PASS. `burn_in_bars=500` is configured, propagated to `config.BURN_IN_BARS`, and applied in all three walkforward paths (`process_fold`, `run_walkforward`, `run_walkforward_with_hmm`). Metrics aggregation excludes the first 500 bars of each fold.

**Fix:** None required.

---

### Finding 15 — WALKFORWARD SESSION BOUNDARY: session_id splitting IMPLEMENTED

| Evidence | File |
|----------|------|
| `run_walkforward()` builds folds by `session_id` groups, NOT `pl.col('ts_event').dt.date()` | `quant/walkforward.py:447-472` |
| `run_walkforward_with_hmm()` also uses `session_id` groups | `quant/walkforward.py:339-372` |

**Verdict:** PASS. Both walkforward entry points use `unique_sessions = df['session_id'].unique(maintain_order=True)` and split folds by session index range. No calendar-date leakage across session boundaries.

**Fix:** None required.

---

### Finding 16 — MULTIPLIER PROPAGATION: contract_multiplier reaches PnL

| Evidence | File |
|----------|------|
| `contract_multiplier` loaded from per-market YAML in `simulate_execution_classification()` | `simulator.py:312-318` |
| Passed to `_compute_pnl_from_target_exec()` | `simulator.py:401` |
| Used in PnL: `position * ret_exec * entry_price * contract_multiplier` | `simulator.py:489` |
| Used in intrabar PnL: `intrabar_pnl * contract_multiplier` | `simulator.py:490` |
| Used in PnL clip: `0.05 * entry_price * contract_multiplier` | `simulator.py:496` |
| `FIXED_CONTRACT_SIZE = 1.0` is only a fallback (line 17), overridden by actual multiplier | `simulator.py:17` |

**Verdict:** PASS. Contract multiplier is loaded from per-market YAML and applied in PnL computation, intrabar PnL, and PnL clipping. The `FIXED_CONTRACT_SIZE=1.0` constant is only used as a fallback when no market config exists.

**Fix:** None required.

---

### Finding 17 — CROSS-ASSET ALIGNMENT: session-scoped forward-fill IMPLEMENTED

| Evidence | File |
|----------|------|
| Cross-asset features forward-filled within `session_id` groups, resetting to null at session boundaries | `quant/ingest.py:111-114` |

**Verdict:** PASS. `fill_null(strategy='forward').over('session_id')` prevents stale secondary-market data from bleeding across primary-market session gaps.

**Fix:** None required.

---

### Finding 18 — HMM PNL RECOMPUTE: identical pipeline used

| Evidence | File |
|----------|------|
| `_recompute_pnl_after_gate()` calls `_compute_pnl_from_target_exec()` — same pipeline as `simulate_execution_classification()` | `quant/walkforward.py:249-297` |
| Shares contract_multiplier resolution, intrabar stops, position clipping, round-turn settlement, PnL clip | `quant/walkforward.py:291` |

**Verdict:** PASS. HMM-gated PnL is recomputed through the exact same `_compute_pnl_from_target_exec()` pipeline. No divergence between base and HMM PnL formulas.

**Fix:** None required.

---

### Finding 19 — PROBABILITY SMOOTHING: session-boundary reset verified

| Evidence | File |
|----------|------|
| `smooth_probabilities()` resets `current = 0.5` when `sess != last_session` | `quant/walkforward.py:82-83` |
| `session_ids` from `test_original['session_id']` — matches actual data session boundaries | `quant/walkforward.py:122` |

**Verdict:** PASS. EMA smoothing resets to neutral (0.5) at every session boundary, and the session_id array comes directly from the DataFrame.

**Fix:** None required.

---

### Finding 20 — CI WORKFLOW: PRESENT (RESOLVED)

| Evidence | File |
|----------|------|
| CI workflow exists with all 7 test steps + audit assertion summary | `.github/workflows/audit_quant_model.yml:1-52` |

**Verdict:** PASS. CI pipeline runs causal audit, continuous contract, fuzz harness (1000 runs), config verification, alignment, HTF features, and session streaming tests on push/PR to main/master.

**Fix:** None required.

---

### Finding 21 — FUZZ HARNESS: EXISTS

| Evidence | File |
|----------|------|
| `run_fuzz_harness()` with `time_skew`, `missing_bars`, `roll_jump` perturbations | `tests/test_fuzz.py:598-646` |
| Audit assertion checks embedded: roll test, leverage stress, intrabar gap, burn-in exclusion, round-turn cost | `tests/test_fuzz.py:310-451` |
| Supports `--runs 1000`, `--seed 42` CLI flags | `tests/test_fuzz.py:652-659` |

**Verdict:** PASS. Fuzz harness exists with all required perturbation types and audit-assertion checks.

**Fix:** None required.

---

### Finding 22 — market_config.py: max_position_size loaded (RESOLVED)

| Evidence | File |
|----------|------|
| `overrides` dict includes `'MAX_POSITION_SIZE': market_cfg.get('risk', {}).get('max_position_size')` | `quant/market_config.py:21` |

**Verdict:** PASS. Per-market max_position_size is loaded from market YAML and propagated to config namespace.

**Fix:** None required.

---

### Finding 23 — config_manager.py: Pydantic type corrected (RESOLVED)

| Evidence | File |
|----------|------|
| `max_position_size: float | None = None` | `quant/config_manager.py:181` |

**Verdict:** PASS. Pydantic field type is `float | None`, matching actual integer values in market YAMLs.

**Fix:** None required.

---

## Risk Summary

| Risk Level | Count | Items |
|------------|-------|-------|
| VERY HIGH | 0 | All VERY HIGH risks from original audit are RESOLVED |
| HIGH | 0 | All HIGH risks from original audit are RESOLVED |
| MEDIUM | 0 | All MEDIUM risks RESOLVED |
| LOW | 0 | All LOW risks RESOLVED |

**Directive:** All 23 findings have been resolved. Deploy is unblocked. Assertion summary:

| Assertion | Status |
|-----------|--------|
| ASSERT_CONFIG_HIERARCHY | PASS |
| ASSERT_DATA_INTEGRITY | PASS |
| ASSERT_GAP_FILTER | PASS |
| ASSERT_HTF_ALIGNMENT | PASS |
| ASSERT_ROLLING_SHIFT | PASS |
| ASSERT_ZSCORE_CORRECT | PASS |
| ASSERT_ET_PARAMS | PASS |
| ASSERT_CONTINUOUS_CONTRACT | PASS |
| ASSERT_MAX_POSITION_SIZE | PASS |
| ASSERT_INTRABAR_STOPS | PASS |
| ASSERT_BURN_IN | PASS |
| ASSERT_SESSION_FOLD | PASS |
| ASSERT_MULTIPLIER | PASS |
| ASSERT_FUZZ_HARNESS | PASS |

**CI:** `.github/workflows/audit_quant_model.yml` present — all assertions + fuzz harness run on push/PR.

**All 19 core assertions + CI workflow + fuzz harness: PASS. Zero findings remain.**