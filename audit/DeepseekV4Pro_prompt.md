MODEL: DeepSeek V4 Pro
MODE: TARGETED SAFE CODEBASE REPAIR

You are a senior quant systems repair agent. Audit and patch this Python futures ML backtester with minimal edits only.

PRIMARY GOAL:
Make the codebase run safely and preserve causal correctness.

TOKEN RULES:
- Be brief.
- Read targeted files only.
- Do not summarize whole files.
- Do not inspect unrelated code unless needed.
- Prefer grep/search before opening files.
- Patch small ranges only.
- Stop after validation.

GLOBAL SAFETY:
- No destructive edits.
- No file moves/renames.
- No architecture rewrites.
- No strategy redesign.
- No alpha/threshold tuning.
- No formatting-only churn.
- No dependency additions unless required to fix a crash.
- No infinite loops/exploration.
- No broad refactor.
- Preserve existing public APIs, CLI commands, configs, outputs, and cache paths.
- Preserve temporal causality: features(t-1) -> prediction(t) -> execution(t+1).
- Never introduce lookahead, target leakage, same-bar fills, full-sample scaler leakage, or cross-fold contamination.
- Never silently fallback futures metadata to ES.

FIRST PASS — MUST FIX COMPILE/RUNTIME BREAKS:
1. Run/inspect syntax quickly:
   - python -m py_compile run.py
   - python -m py_compile core/config.py
   - python -m py_compile pipeline/**/*.py where practical
2. Fix only real syntax/import/runtime blockers.
3. Visible known issue:
   - run.py appears to have broken indentation/missing function around `_log_failures` and `_print_split_dashboard`.
   - Restore `_print_split_dashboard(split_idx, total_splits, per_symbol)` as a proper function.
   - Keep dashboard behavior unchanged.

SECOND PASS — TARGETED AUDIT FIXES:
A) Session ID offset consistency
- Search for hardcoded session offsets like `offset_by('6h')`, `"6h"`, `timedelta(hours=6)`.
- Replace hardcoded session offset with canonical logic derived from config.SESSION_START_LOCAL.
- Default 18:00 session start must still equal 6h offset.
- Ensure session.py and walkforward.py use equivalent session grouping logic.

B) Contract multiplier safety
- Search `_recompute_pnl_after_gate`, `CURRENT_SYMBOL`, multiplier fallback, `.get(..., 'ES')`, `.get(..., 50)`.
- Remove silent ES fallback.
- Require explicit symbol and valid multiplier.
- Raise clear RuntimeError if symbol/multiplier missing.
- Non-ES symbols must never use ES multiplier accidentally.

C) Subprocess failure logs
- Keep console concise.
- Persist full stdout/stderr for failed subprocesses under existing output/log path.
- Include command, return code, symbol, split_idx, stage.
- Do not change fallback behavior.

D) Continuous contract factor guard
- Search cumulative_factor logic.
- Do not redesign contract construction.
- Add fail-fast guard for non-finite, <=0 factor.
- Add warning for extreme but finite cumulative_factor.
- Keep output schema unchanged.

E) Adjustment factor semantics
- Do not break output columns.
- Add narrow comments/internal variable names only if needed.
- No schema changes.

TESTS:
Add/update only minimal tests for touched behavior:
- session offset consistency
- no ES multiplier fallback
- run.py compiles
- subprocess failure log persistence if easy

VALIDATION ORDER:
1. python -m py_compile run.py
2. python -m py_compile core/config.py
3. python -m py_compile pipeline/...
4. pytest tests/test_session_streaming.py -q
5. pytest tests/test_continuous_contract.py -q
6. pytest tests/test_causal_audit.py -q
7. pytest -q only if fast/practical

IF TESTS FAIL:
- Fix only failures caused by your edits.
- If unrelated failure exists, report it and stop.

OUTPUT ONLY:
1. Files changed
2. Fixes made
3. Commands run + pass/fail
4. Remaining risks

STOP CONDITION:
Stop immediately once compile passes, targeted fixes are patched, and relevant tests pass.