FULL PYTEST PASS: 19 passed

QUARANTINED:
unused/core/utils/manifest.py
unused/core/utils/types.py
unused/pipeline/orchestrator.py
unused/pipeline/runner.py
unused/pipeline/ingest/stage.py
unused/pipeline/session/normalization.py
unused/pipeline/tracking/state.py

FIXED:
- discovery target bug
- HMM logging visibility
- session offset helper
- multiplier fail-fast validation
- causal audit harness
- synthetic path test

CURRENT GOAL:
Clean flowchart-style logging for run.py

DO NOT MODIFY:
- strategy logic
- HMM logic
- targets
- execution
- metrics