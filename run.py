"""
run.py
Production Orchestrator for the Deterministic Quant Pipeline.
Scans 'futures/' for 1‑min OHLCV Parquet files (market/year/*.parquet)
and runs the two‑phase pipeline (discovery → walkforward) per file.

After all files are processed, it runs aggregate_metrics.py to produce
consolidated performance reports per market and across all markets.
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

            for file_path in sorted(project_root.rglob("*")):
                if file_path.is_dir():
                    continue
                if file_path.name == snapshot_filename:
                    continue
                rel_path = file_path.relative_to(project_root)
                if any(part in exclude_dirs for part in rel_path.parts):
                    continue
                if file_path.suffix.lower() in binary_extensions:
                    continue
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
    market = data_path.parent.name
    year = data_path.stem
    artifacts_dir = Path("artifacts") / market / year
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log_file = artifacts_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)

    logger.info(f"--- Starting Pipeline for {market} {year} ---")
    logger.info(f"Data file: {data_path}")

    manifest_path = artifacts_dir / "manifest.json"

    stage_discover = [
        sys.executable, "-m", "src.cli", "discover",
        "--data", str(data_path),
        "--out", str(manifest_path)
    ]

    stage_run = [
        sys.executable, "-m", "src.cli", "run",
        "--data", str(data_path),
        "--manifest", str(manifest_path),
        "--out", str(artifacts_dir)
    ]

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

    file_handler.close()
    logger.removeHandler(file_handler)

if __name__ == "__main__":
    create_audit_snapshot()

    futures_dir = Path("futures")
    if not futures_dir.exists():
        logger.error("Directory 'futures' not found.")
        sys.exit(1)

    files = list(futures_dir.rglob("*.parquet"))
    if not files:
        logger.warning("No Parquet files found under 'futures/'.")
        sys.exit(0)

    logger.info(f"Found {len(files)} file(s) to process.")
    for file_path in files:
        process_file(file_path)

    # --- POST-PROCESSING: Run aggregator to consolidate all results ---
    logger.info("All files processed. Running aggregate_metrics.py to generate consolidated reports...")
    agg_result = subprocess.run([sys.executable, "aggregate_metrics.py"], capture_output=True, text=True)
    if agg_result.returncode == 0:
        logger.info("✅ Aggregated metrics saved to artifacts/aggregated/")
        # Print the output for visibility
        print("\n" + agg_result.stdout)
    else:
        logger.error(f"Aggregation failed: {agg_result.stderr}")