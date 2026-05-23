"""
run.py
Production Orchestrator for the Deterministic Quant Pipeline.
Scans 'futures/' for 1‑min OHLCV Parquet files (market/year/*.parquet)
and runs the two‑phase pipeline (discovery → walkforward) per file.

Compliance:
- No manual row splitting – discovery uses first 60 days of resampled 5‑min data.
- Memory safe – each CLI call handles its own chunked resampling.
- Deterministic – same inputs produce identical results.
- Audit snapshot: creates a full project snapshot (full_code.md) before each run.
"""
import subprocess
import sys
import logging
import time
from pathlib import Path
from datetime import datetime

# Setup global logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QuantRunner")

def create_audit_snapshot(root_dir: str = "."):
    """
    Creates a full project snapshot (all text files) for audit purposes.
    Saves as full_code.md in the project root.
    """
    project_root = Path(root_dir).resolve()
    snapshot_filename = "full_code.md"

    # Directories to exclude completely
    exclude_dirs = {
        ".venv", "venv", "env", ".git", "__pycache__",
        "artifacts", "logs", "models", "node_modules",
        ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints",
        "dist", "build", "htmlcov", ".tox"
    }

    # Binary extensions to skip entirely (cannot be read as text)
    binary_extensions = {
        ".parquet", ".pyc", ".log", ".tag", ".png", ".jpg", ".jpeg",
        ".gif", ".bmp", ".ico", ".pdf", ".docx", ".xlsx", ".zip",
        ".tar", ".gz", ".pickle", ".pkl", ".so", ".dll", ".exe",
        ".db", ".sqlite", ".pyo", ".egg", ".whl"
    }

    logger.info(f"Creating audit snapshot: {snapshot_filename}")
    try:
        with open(project_root / snapshot_filename, "w", encoding="utf-8") as f:
            f.write(f"# Project Snapshot for Audit\n\n")
            f.write(f"# Root: {project_root}\n")
            f.write(f"# Created: {datetime.now().isoformat()}\n\n")

            # Walk all files recursively
            for file_path in sorted(project_root.rglob("*")):
                if file_path.is_dir():
                    continue

                # Skip the snapshot file itself
                if file_path.name == snapshot_filename:
                    continue

                # Skip excluded directories
                rel_path = file_path.relative_to(project_root)
                if any(part in exclude_dirs for part in rel_path.parts):
                    continue

                # Skip binary extensions
                if file_path.suffix.lower() in binary_extensions:
                    continue

                # Write header and full content
                f.write(f"--- \n### File: {rel_path}\n")
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    f.write("```\n" + content + "\n```\n\n")
                except Exception as e:
                    f.write(f"Error reading file: {e}\n\n")

        logger.info(f"✅ Audit snapshot saved to: {snapshot_filename}")
    except Exception as e:
        logger.error(f"Failed to create audit snapshot: {e}")

def run_step(cmd_list, retries=2, delay=5):
    """Executes a command with retry mechanism."""
    for attempt in range(retries + 1):
        logger.info(f"Executing: {' '.join(cmd_list)}")
        try:
            result = subprocess.run(cmd_list, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Step completed successfully.")
                return True
            else:
                logger.error(f"Attempt {attempt + 1} failed (rc={result.returncode}): {result.stderr[-500:]}")
        except Exception as e:
            logger.error(f"Exception during execution: {e}")

        if attempt < retries:
            time.sleep(delay)
            logger.info(f"Retrying step... (Attempt {attempt + 2})")

    return False

def process_file(data_path: Path):
    """Run the full pipeline for a single 1‑min Parquet file."""
    # Infer market and year from directory structure (e.g., futures/ES/2024.parquet)
    market = data_path.parent.name
    year = data_path.stem
    artifacts_dir = Path("artifacts") / market / year
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Localised logging to file
    log_file = artifacts_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)

    logger.info(f"--- Starting Pipeline for {market} {year} ---")
    logger.info(f"Data file: {data_path}")

    # Manifest path (shared between discovery and run)
    manifest_path = artifacts_dir / "manifest.json"

    # Stage 1: Feature discovery (ExtraTrees with bootstrap folds, stability selection)
    stage_discover = [
        sys.executable, "-m", "src.cli", "discover",
        "--data", str(data_path),
        "--out", str(manifest_path)
    ]

    # Stage 2: Walkforward Ridge regression + execution simulation
    stage_run = [
        sys.executable, "-m", "src.cli", "run",
        "--data", str(data_path),
        "--manifest", str(manifest_path),
        "--out", str(artifacts_dir)
    ]

    # Stage 3: Performance analytics (optional – expects backtest_results.parquet)
    stage_analytics = [
        sys.executable, "-m", "src.analytics",
        str(artifacts_dir / "backtest_results.parquet")
    ]

    for stage_cmd in [stage_discover, stage_run, stage_analytics]:
        if not run_step(stage_cmd):
            logger.error(f"Pipeline failed at stage: {stage_cmd[3] if len(stage_cmd)>3 else stage_cmd[2]}")
            break
    else:
        logger.info(f"--- Pipeline Completed Successfully for {market} {year} ---")

    # Cleanup to avoid log handler leaks
    file_handler.close()
    logger.removeHandler(file_handler)

if __name__ == "__main__":
    # --- Create audit snapshot before any processing ---
    create_audit_snapshot()

    futures_dir = Path("futures")
    if not futures_dir.exists():
        logger.error("Directory 'futures' not found. Please create it and place 1‑min Parquet files inside (e.g., futures/ES/2024.parquet).")
        sys.exit(1)

    # Recursively find all .parquet files under 'futures/'
    files = list(futures_dir.rglob("*.parquet"))
    if not files:
        logger.warning("No Parquet files found under 'futures/'.")
        sys.exit(0)

    logger.info(f"Found {len(files)} file(s) to process.")
    for file_path in files:
        process_file(file_path)