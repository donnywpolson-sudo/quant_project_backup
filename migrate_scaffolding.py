#!/usr/bin/env python3
"""
migrate_scaffolding.py
Automatically restructures the quant project from src/ -> quant/,
creates config/ folder, updates imports, and adjusts run.py.
Run from project root.
"""

import os
import re
import shutil
import sys
from pathlib import Path
from tkinter import FALSE, TRUE

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DRY_RUN = FALSE          # Set to True to preview changes without writing
BACKUP_SUFFIX = ".bak"   # Create backups of modified files (if not dry run)

PROJECT_ROOT = Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"
QUANT_DIR = PROJECT_ROOT / "quant"
CONFIG_DIR = PROJECT_ROOT / "config"
RUN_PY = PROJECT_ROOT / "run.py"

# Files/folders to move into config/
YAML_FILES = ["baseline_features.yaml"]
MARKETS_DIR = PROJECT_ROOT / "config" / "markets"  # original location? check both
# We'll assume original yaml is at PROJECT_ROOT / "config/baseline_features.yaml"
# and markets at PROJECT_ROOT / "config/markets/". If not, we'll create and copy.

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def backup_file(path: Path):
    if DRY_RUN:
        return
    if path.exists() and not path.with_suffix(BACKUP_SUFFIX).exists():
        shutil.copy2(path, path.with_suffix(BACKUP_SUFFIX))
        print(f"   Backed up: {path} -> {path.with_suffix(BACKUP_SUFFIX)}")

def log_action(action: str, target: str):
    print(f"{'[DRY RUN] ' if DRY_RUN else ''}{action}: {target}")

def apply_replacements(file_path: Path, replacements: list):
    """Apply list of (pattern, replacement) tuples to file content."""
    if not file_path.exists():
        return
    backup_file(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    new_content = content
    for pattern, repl in replacements:
        new_content = re.sub(pattern, repl, new_content)
    if new_content != content:
        if not DRY_RUN:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            log_action("Modified", str(file_path))
        else:
            log_action("Would modify", str(file_path))
    else:
        log_action("No changes needed", str(file_path))

def ensure_init_py(directory: Path):
    """Create empty __init__.py in directory if not present."""
    init_file = directory / "__init__.py"
    if not init_file.exists():
        if not DRY_RUN:
            init_file.touch()
            log_action("Created", str(init_file))
        else:
            log_action("Would create", str(init_file))

# ----------------------------------------------------------------------
# Step 1: Rename src/ to quant/ (if not already done)
# ----------------------------------------------------------------------
def step_rename_src_to_quant():
    if QUANT_DIR.exists() and not SRC_DIR.exists():
        log_action("INFO", "quant/ already exists and src/ missing – assuming already migrated")
        return
    if not SRC_DIR.exists():
        log_action("ERROR", "src/ directory not found. Nothing to rename.")
        return
    if QUANT_DIR.exists():
        raise FileExistsError("Both src/ and quant/ exist. Please remove quant/ manually.")
    if not DRY_RUN:
        shutil.move(str(SRC_DIR), str(QUANT_DIR))
        log_action("Renamed", f"src/ -> quant/")
    else:
        log_action("Would rename", "src/ -> quant/")

# ----------------------------------------------------------------------
# Step 2: Ensure config/ directory and move YAML files
# ----------------------------------------------------------------------
def step_setup_config_dir():
    CONFIG_DIR.mkdir(exist_ok=True)
    log_action("Ensured", f"config/ directory exists")
    # Move baseline_features.yaml if it exists in root config/ (old location) or in new config/
    src_yaml = PROJECT_ROOT / "config" / "baseline_features.yaml"
    if not src_yaml.exists():
        # try root level
        src_yaml = PROJECT_ROOT / "baseline_features.yaml"
    dest_yaml = CONFIG_DIR / "baseline_features.yaml"
    if src_yaml.exists() and not dest_yaml.exists():
        if not DRY_RUN:
            shutil.move(str(src_yaml), str(dest_yaml))
            log_action("Moved", f"{src_yaml} -> {dest_yaml}")
        else:
            log_action("Would move", f"{src_yaml} -> {dest_yaml}")
    # Move markets folder
    src_markets = PROJECT_ROOT / "config" / "markets"
    if not src_markets.exists():
        src_markets = PROJECT_ROOT / "markets"   # old location?
    dest_markets = CONFIG_DIR / "markets"
    if src_markets.exists() and not dest_markets.exists():
        if not DRY_RUN:
            shutil.move(str(src_markets), str(dest_markets))
            log_action("Moved", f"{src_markets} -> {dest_markets}")
        else:
            log_action("Would move", f"{src_markets} -> {dest_markets}")
    else:
        # ensure markets folder exists
        dest_markets.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Step 3: Update imports in all .py files under quant/
# ----------------------------------------------------------------------
def step_update_imports():
    replacements = [
        (r'\bfrom src\.', r'from quant.'),
        (r'\bimport src\.', r'import quant.'),
        # Also handle from . import xxx? no, keep absolute
        # But careful: do not change 'from config import config'
    ]
    for py_file in QUANT_DIR.rglob("*.py"):
        if py_file.name == "__init__.py":
            # still need to update content if any imports inside
            pass
        apply_replacements(py_file, replacements)

# ----------------------------------------------------------------------
# Step 4: Update run.py (change src.cli to quant.cli)
# ----------------------------------------------------------------------
def step_update_run_py():
    if not RUN_PY.exists():
        log_action("WARNING", "run.py not found, skipping")
        return
    replacements = [
        (r'-m src\.cli', r'-m quant.cli'),
        (r'from src\.cli', r'from quant.cli'),
        (r'import src\.cli', r'import quant.cli'),
    ]
    apply_replacements(RUN_PY, replacements)

# ----------------------------------------------------------------------
# Step 5: Add __init__.py files
# ----------------------------------------------------------------------
def step_add_init_files():
    for dirpath in [QUANT_DIR] + [d for d in QUANT_DIR.rglob("*") if d.is_dir()]:
        # skip __pycache__ and other non-package dirs
        if dirpath.name == "__pycache__":
            continue
        ensure_init_py(dirpath)

# ----------------------------------------------------------------------
# Step 6: (Optional) Update quant/market_config.py if path changed
#   Not needed if config.py already points to "config/markets/..."
# ----------------------------------------------------------------------
def step_check_market_config():
    mcfg = QUANT_DIR / "market_config.py"
    if not mcfg.exists():
        return
    # Ensure that MARKET_CONFIGS in config.py points to correct relative paths
    # We'll not modify automatically; user may need to adjust.
    log_action("INFO", "Please verify that config.MARKET_CONFIGS points to 'config/markets/...'")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Quant Project Scaffolding Migration Tool")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dry run mode: {DRY_RUN}")
    print("=" * 60)

    if not DRY_RUN:
        response = input("This will modify files. Ensure you have a backup. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    steps = [
        ("Rename src/ -> quant/", step_rename_src_to_quant),
        ("Setup config/ directory", step_setup_config_dir),
        ("Update imports in quant/", step_update_imports),
        ("Update run.py", step_update_run_py),
        ("Add __init__.py files", step_add_init_files),
        ("Check market config", step_check_market_config),
    ]

    for name, func in steps:
        print(f"\n--- {name} ---")
        try:
            func()
        except Exception as e:
            print(f"ERROR in step {name}: {e}")
            if not DRY_RUN:
                print("Aborting due to error.")
                sys.exit(1)

    print("\n" + "=" * 60)
    if DRY_RUN:
        print("Dry run complete. Run with DRY_RUN = False to apply changes.")
    else:
        print("Migration complete. Please test the pipeline with: python run.py")
        print("If any issues, restore from .bak files or backup.")

if __name__ == "__main__":
    main()