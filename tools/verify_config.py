"""
verify_config.py — Print active configuration parameters for Lean Alpha Discovery.

Validates that the active config.yaml matches the expected Lean Alpha Discovery
values before any backtest is run.

Usage:
    python tools/verify_config.py

Expected output (Lean Alpha Discovery mode):
    markets:                       ['ES', 'CL', 'ZB']
    discovery.bootstrap_folds:     5
    walkforward.wf_step_days:      5
    discovery.extra_trees_params.max_depth: 6

Exit code 0 = all values match.  Exit code 1 = mismatch detected.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so we can import from quant/
_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

from quant.config import config
from quant.utils.config_loader import load_config

# Expected values for Lean Alpha Discovery mode
EXPECTED = {
    "markets": ["ES", "CL", "ZB"],
    "BOOTSTRAP_FOLDS": 5,
    "WF_STEP_DAYS": 5,
    "EXTRA_TREES_PARAMS.max_depth": 6,
}


def verify() -> int:
    """Load config and compare active values against expected values.

    Returns 0 if all checks pass, 1 if any check fails.
    """
    load_config()

    checks = [
        ("markets", config.MARKETS, EXPECTED["markets"]),
        ("discovery.bootstrap_folds", config.BOOTSTRAP_FOLDS, EXPECTED["BOOTSTRAP_FOLDS"]),
        ("walkforward.wf_step_days", config.WF_STEP_DAYS, EXPECTED["WF_STEP_DAYS"]),
        (
            "discovery.extra_trees_params.max_depth",
            config.EXTRA_TREES_PARAMS["max_depth"],
            EXPECTED["EXTRA_TREES_PARAMS.max_depth"],
        ),
    ]

    print("=" * 60)
    print("  LEAN ALPHA DISCOVERY — Config Verification")
    print("=" * 60)
    print()

    all_ok = True
    for label, actual, expected in checks:
        status = "PASS" if actual == expected else "FAIL"
        if status == "FAIL":
            all_ok = False

        # Format lists nicely
        actual_str = str(actual)
        expected_str = str(expected)
        print(f"  [{status}] {label}")
        print(f"         actual:   {actual_str}")
        print(f"         expected: {expected_str}")
        print()

    print("=" * 60)
    if all_ok:
        print("  RESULT: ALL CHECKS PASSED — Ready for backtest.")
    else:
        print("  RESULT: MISMATCH DETECTED — Do NOT proceed with backtest.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(verify())