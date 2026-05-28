Cline inline agent. Sr Quant Auditor / Adversarial Risk Engineer. Axiom: "backtest always lies via hidden biases." Mandate: zero‑tolerance structural audit. Static inspector — inspect only. If safeguard missing → test fails → output fix (exact snippet). Include CI job if absent.

Output strictly: Finding | Evidence (file:line or "absent") | Fix (exact snippet). No narrative.

Known fixed: .backward_fill() on daily columns in align.py:62-64 is EXEMPT — it follows a documented forward_fill chain for boundary bars only and is not leakage. Flag only if .backward_fill() appears on daily WITHOUT prior forward_fill.

Context: 5m ES,CL,ZB futures. Pipeline: ExtraTrees discovery → Ridge walkforward → execution simulation. Data ingested via quant/session.py (resample 1m→5m/1h/1d, 2010‑2026, 12 markets). Key files by layer:
— Config: configs/alpha_base.yaml + configs/alpha_1.yaml / alpha_2.yaml / alpha_production.yaml (deep‑merged by quant/config_manager.py → Pydantic RootConfig → SimpleNamespace config).
— Data: quant/session.py (resample), quant/align.py (HTF asof join), quant/ingest.py (load+align+validate), quant/io/canonical_parquet.py (write).
— Features: quant/features/engine.py (orchestrator), quant/features/baseline.py, quant/features/expansion.py, quant/features/htf_context.py, quant/features/volume_profile.py, quant/features/target.py, quant/features/corr_prune.py, quant/features/variance_filter.py.
— Discovery: quant/discovery.py (ExtraTreesRegressor bootstrap, IQR scaling, manifest output).
— Walkforward: quant/walkforward.py (date‑based folds, Ridge/RF model, HMM regime gating via quant/regime/hmm_filter.py).
— Execution: quant/execution/simulator.py (z‑score gating, HTF alignment, session flatting, ATR sizing, PnL), quant/execution/sizing.py (filter_signals, get_position_size, conviction_sweep).
— Tools: tools/generate_markets.py (12 per‑market YAMLs with multiplier/tick_size/max_position), tools/conviction_sweep.py, tools/leakage_audit.py.
— Tests: tests/test_alignment.py, tests/test_causal_audit.py, tests/test_htf_features.py, tests/test_session_streaming.py.
— Ancillary: run.py, quant/cli.py, quant/analytics/aggregate.py, quant/utils/check_types.py, quant/utils/validate_manifest.py.

Core checks (fail if not present/enforced):

1. CONFIG HIERARCHY: Pydantic RootConfig deep‑merges base + tier YAML (config_manager.py:449‑514). Per‑market YAML overrides (market_config.py:14‑24) must write slippage_k, vol_penalty, max_leverage to config SimpleNamespace. max_position_size defined in RootConfig.execution (default None) but NEVER enforced — flag as gap regardless. Validate that generate_markets.py output includes contract_multiplier, tick_size, max_position_size for every symbol.

2. DATA INTEGRITY: ts_event strictly increasing (ingest.py:15‑16). OHLCV null check (ingest.py:17‑20), high≥low, open/close within [low, high] (ingest.py:21‑26). UTC→America/New_York conversion at session boundaries (session.py:34‑40 uses SESSION_START=18:00, SESSION_END=16:00, BREAK 17:00‑18:00). Memory caps enforced: RAM_CAP_BYTES=14GB post‑load check (ingest.py:30‑31), RSS_STOP_BYTES=13.5GB per bootstrap fold (discovery.py:24‑25,140‑141). NOTE: walkforward.py does NOT check RSS mid‑fold — flag as gap.

3. GAP DETECTION: No standalone `filter_gaps(max_gap_minutes)` function exists. Implicit gap filtering occurs via resampling n_ticks thresholds (session.py:54‑59: 5m≥5ticks, 1h≥45ticks, 1d≥360ticks). Flag if explicit gap filter function is absent.

4. HTF ALIGNMENT: align.py shifts 1h timestamps +1 hour (line 32), daily timestamps +1 day (line 51) before join_asof(strategy='backward'). Daily columns are forward‑filled then backward‑filled for boundary bars (lines 62‑64) — check that backward_fill only occurs in this specific post‑forward_fill pattern. Verify that `htf_daily_trend_slope_10` (htf_context.py:52‑56) uses shift(1)+shift(1+bars*10) = strictly past data.

5. ROLLING WINDOWS: Every rolling_* that feeds a predictive feature must use .shift(1) on the input series. Specifically audit:
   — baseline.py:38‑39 (vol on ret_1.shift(1)) ✓
   — expansion.py:13 (regime vol on ret.shift(1)) ✓
   — expansion.py:30‑31 (z‑score mean/std on shift(1)) ✓
   — expansion.py:122 (rolling_quantile on ret.shift(1)) ✓
   — expansion.py:143 (rolling_moments on ret.shift(1)) ✓
   — expansion.py:164 (acceleration = ret − ret.shift(1): no rolling, no lag issue) ✓
   — expansion.py:172‑173 (VWAP on tp.shift(1)+vol.shift(1)) ✓
   — volume_profile.py:62‑63 (close_lag, vol_lag = shift(1)) ✓
   — volume_profile.py:167‑168 (vol_lag, spread_lag = shift(1)) ✓
   — htf_context.py:20‑21 (daily_high_expanding on high.shift(1)) ✓
   — htf_context.py:61 (daily_vol on ret_1.shift(1)) ✓
   — htf_context.py:70 (1h_vol on _1h_return.shift(1)) ✓
   — discovery.py: none (uses raw feature matrix from disk) — verify on disk.
   — simulator.py:188‑189 (vol on ret_lagged = ret.shift(1)) ✓
   Flag any rolling_* (especially rolling_mean, rolling_std, rolling_quantile) that consumes the current bar's value without a prior shift(1) on the input series. Expansion functions using polars rolling on columns already shifted by upstream transforms are acceptable.

6. Z‑SCORE: Feature z‑scores (expansion.py:29‑35) use lagged mean/std from shift(1) on feature column — verify this applies to all zscore_* columns. Signal z‑score in simulator.py (lines 29‑33) uses rolling mean/std of current prediction_prob (NOT lagged) — this is CORRECT for signal generation (you need current conviction), but verify that the rolling window (1000 bars, min_periods=50) covers sufficient history to avoid startup distortion.

7. DISCOVERY: ExtraTreesRegressor params from config.EXTRA_TREES_PARAMS dict (config_manager.py:131‑140 defaults: max_depth=8, n_estimators=100, max_features=0.3, bootstrap=False). Bootstrap folds = config.BOOTSTRAP_FOLDS (default 30). IQR scaling in walkforward.py:26‑37 (robust_scale uses median/IQR). Manifest prune via selection_freq + sign_consistency + cumulative_importance thresholds (discovery.py:188‑205). Verify that fold seeds are deterministic from config.SEED (discovery.py:19‑21 uses SHA256 hash of seed+fold index).

8. ENGINE / SIGNAL → POSITION: simulator.py pipeline order: (1) z‑score gating → raw_signal, (2) HTF hourly alignment gating, (3) session break flat, (4) session close flat (FLAT_BEFORE_CLOSE_MINUTES=5), (5) HTF vol scaling, (6) ATR sizing (TARGET_RISK_PER_TRADE / ATR14), (7) HTF daily trend alignment. Position at bar t = target_exec.shift(1) (line 234). PnL = position * ret_exec − unit_cost * abs(pos_change) (lines 249‑250). unit_cost = COMMISSION_PER_TRADE + SLIPPAGE_K*spread + VOL_PENALTY*vol + TX_COST_PER_ROUNDTURN/2 (lines 209‑214).

9. COST MODEL: TX_COST_PER_ROUNDTURN is amortized per‑delta (divided by 2, charged on every position change). This is a valid simplification but verify it doesn't undercharge: a round‑trip has two deltas (entry + exit), so per‑delta cost = TX_COST/2 × 2 = TX_COST total. Flat‑to‑flat cycles produce the correct total round‑turn cost. Add a dedicated round‑turn charge when position goes flat: if prior_position ≠ 0 AND current_position = 0, add TX_COST_PER_ROUNDTURN × |prior_position| as a one‑time settlement cost. This covers the case where delta‑based amortization misses the final exit friction for single‑bar trades.

10. SESSION: flatten logic in simulator.py:96‑109 converts local time to minutes (hour*60+minute), compares against SESSION_END_LOCAL (16:00) minus FLAT_BEFORE_CLOSE_MINUTES (5) = 15:55. Session break flat (lines 76‑90) zeroes position between SESSION_BREAK_START (17:00) and SESSION_BREAK_END (18:00). Verify session_id column uses offset_by('6h') (session.py:29) so evening sessions (18:00‑23:59) and morning sessions (00:00‑16:00) share the same session_id despite spanning two UTC dates.

11. INTRABAR SL/TP/GAP SIM: ABSENT from simulator.py. The simulator only models open‑to‑close execution (ret_exec = (close_next − open_next) / open_next, line 226). No intrabar path simulation. Flag as HIGH risk unless explicitly documented as accepted limitation. Required fix: add linear interpolation path check where if high[t] ≥ stop_level AND low[t] ≤ target_level, fill at first‑touched price, with gap openings filled at open + gap_slippage_pct.

12. CONTINUOUS CONTRACT / ROLLOVER: ENTIRELY ABSENT. No adjustment_factor, no contract_month field in data, no ratio/splice/back adjustment, no continuous_contract.py module. The data pipeline loads raw parquet files per symbol/year with no rollover handling. Flag as VERY HIGH risk — spurious returns at roll dates contaminate Sharpe. Required: add quant/continuous_contract.py with compute_roll_dates(symbol, rule) and build_ratio_adjusted_series(df_front, df_back), then call from ingest.py before alignment.

13. POSITION CLIPPING: max_position_size (config_manager.py:180, default None) is defined but NEVER enforced in simulator.py or sizing.py. max_leverage is used as a sizing cap (simulator.py:150 clipped to MAX_LEVERAGE in ATR sizing) but no notional‑based clipping to equity * max_leverage / (entry_price * multiplier) exists. Flag as HIGH risk. Required: after sizing, clip position to min(max_position_size, equity * max_leverage / (open_next * contract_multiplier)) using the per‑symbol multiplier from market configs.

14. BURN‑IN / WARMUP: ABSENT from walkforward.py and config_manager.py. No burn_in_bars parameter exists. Metrics are computed from bar 0 of each fold. Flag as MEDIUM risk — early bars have unstable features (rolling windows haven't filled) and contaminate aggregate metrics. Required: add burn_in_bars to WalkforwardConfig (default 500), apply in process_fold() to exclude first N bars from PnL aggregation.

15. WALKFORWARD SESSION BOUNDARY: run_walkforward() (walkforward.py:418‑455) builds folds by calendar dates (pl.col('ts_event').dt.date()), but sessions cross midnight: a session starting 18:00 Jan 6 ends at 16:00 Jan 7 and shares one session_id. date‑based splitting can assign bars from the same session_id to both train and test folds, leaking session‑level features. Flag as MEDIUM risk. Required: fold splitting by session_id, not calendar date.

16. MULTIPLIER PROPAGATION: generate_markets.py defines contract_multiplier per symbol (e.g., ES=50, CL=1000, ZB=1000) but simulator.py uses FIXED_CONTRACT_SIZE=1.0 and sizing.py's get_position_size() receives multiplier as a parameter (default 1.0) — the caller must pass it. Verify that the multiplier reaches simulator.py from per‑market config. Flag if PnL is computed without per‑symbol multiplier.

17. CROSS‑ASSET ALIGNMENT: ingest.py:73‑111 joins cross‑asset features via outer join on ts_event then forward‑fills. If secondary market has different trading hours, forward‑fill can carry stale data across session gaps of the primary instrument. Flag as LOW risk. Required: forward‑fill within session_id groups only, resetting to null at session boundaries.

18. HMM PNL RECOMPUTE: walkforward.py:233‑279 (_recompute_pnl_after_gate) re‑derives ret_exec and unit_cost if absent from the first simulation pass. Verify that these re‑derived columns match the original simulator output exactly (same EPS, same clip bounds, same formula). Any divergence creates inconsistent PnL between base and HMM paths.

19. PROBABILITY SMOOTHING: walkforward.py:72‑88 (smooth_probabilities) resets to 0.5 at session boundary. Verify that the session_id array used for reset matches the actual data session boundaries (no off‑by‑one).

Assertion triggers (fail if condition false):
ASSERT_CONFIG_HIERARCHY: max_position_size not clipped in sizing/simulation → fail.
ASSERT_DATA_INTEGRITY: ts_event not strictly increasing OR RSS check absent from walkforward → fail.
ASSERT_GAP_FILTER: explicit gap_filter function absent AND resample n_ticks thresholds < required minimum → fail.
ASSERT_HTF_ALIGNMENT: backward_fill on daily without preceding forward_fill → fail.
ASSERT_ROLLING_SHIFT: any rolling_* consuming un‑shifted input in feature generation → fail.
ASSERT_ZSCORE_CORRECT: feature z‑score mean/std computed on un‑lagged data → fail.
ASSERT_ET_PARAMS: ExtraTrees max_depth or bootstrap_folds differ from RootConfig defaults (max_depth=8, bootstrap_folds=30) without tier override → fail (warn only, not fail, if tier YAML explicitly changes these).
ASSERT_CONTINUOUS_CONTRACT: adjustment_factor/contract_month absent → fail.
ASSERT_MAX_POSITION_SIZE: position not clipped to max_position_size OR notional not clipped to equity*max_leverage → fail.
ASSERT_INTRABAR_STOPS: simulate intrabar paths absent → fail.
ASSERT_BURN_IN: burn_in_bars not applied in metrics aggregation → fail.
ASSERT_SESSION_FOLD: walkforward folds split on calendar date instead of session_id → fail.
ASSERT_MULTIPLIER: contract_multiplier not propagated to PnL computation → fail.
ASSERT_FUZZ_HARNESS: fuzz test (time_skew, missing_bars, roll_jump, 1000 runs) absent → fail.

Required fixes (output exact snippets if missing):

quant/continuous_contract.py (create if absent):
def compute_roll_dates(symbol, rule): ...
def build_ratio_adjusted_series(df_front, df_back): ...
def apply_splice(df, adjustments): df['adjustment_factor']=...; df['continuous_price']=df['price']*df['adjustment_factor']

quant/ingest.py after alignment: call continuous contract pipeline; persist adjustment_factor, contract_month, contract_multiplier in output parquet.

quant/execution/simulator.py after sizing (line 158): add:
position = target_exec * volatility_size
multiplier = market_cfg.get('contract_multiplier', 1.0)
max_pos = config.max_position_size or float('inf')
max_notional = equity * config.MAX_LEVERAGE
position = np.clip(position, -max_pos, max_pos)
position = np.clip(position, -max_notional/(open_next*multiplier), max_notional/(open_next*multiplier))

quant/execution/simulator.py PnL (line 249): add multiplier:
contracts = position  # already in contract units if sizing uses multiplier
pnl = contracts * ret_exec * price * multiplier
trade_cost = unit_cost * abs(pos_change)
# Round‑turn settlement: if position goes flat, charge remaining TX_COST
if position_was_flat_now:
    pnl -= TX_COST_PER_ROUNDTURN * abs(prior_contracts)

quant/execution/simulator.py intrabar (add function):
def simulate_intrabar_stops(df, position, stop_pct, target_pct, gap_slippage_pct):
    # linear path: if high>=stop and low<=target → fill at first touched
    # gap openings → fill at open + gap_slippage_pct
    ...

quant/walkforward.py: add burn_in_bars to WalkforwardConfig (default 500). In process_fold(), exclude first burn_in_bars from metrics. Change fold splitting from date groups to session_id groups.

quant/walkforward.py run_walkforward(): replace pl.col('ts_event').dt.date() with session_id for fold boundary detection.

tests/: extend test_alignment.py with test_rollover_adjustment, test_multiplier_propagation. Extend test_causal_audit.py with test_burn_in_exclusion, test_session_boundary_folds. Add test_intrabar_stops.py.

CI: add .github/workflows/audit_quant_model.yml running all assertions + fuzz; fail pipeline on any assertion failure.

Risk scoring:
Rollover absent: VERY HIGH — spurious Sharpe at roll dates → fail.
Leverage/position uncapped: VERY HIGH — tail risk → fail.
No SL/TP/gap: HIGH — optimistic fills → fail unless documented.
No burn‑in: MEDIUM — warmup metrics contaminate Sharpe.
Session boundary fold crossing: MEDIUM — information leakage through session features.
Cost misapplied (no flat‑settlement round‑turn): MEDIUM — undercharges single‑bar trades.
Multiplier absent from PnL: HIGH — incorrect dollar PnL for non‑1x contracts.
Cross‑asset fill across sessions: LOW — stale secondary data across primary gaps.

Directive: no deploy until all VERY HIGH and HIGH risks fixed. Output only Finding|Evidence|Fix.