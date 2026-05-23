"""
run_full_pipeline.py
Automated Orchestrator: Scans 'futures/' and processes all found parquet files.
Includes: Retry logic, File-based logging, and robust temporal splitting.
"""
import subprocess
import sys
import logging
import time
import polars as pl
from pathlib import Path

# Setup global logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineOrchestrator")

def run_step(cmd_list, retries=2, delay=5):
    """Executes a command with a retry mechanism for robustness."""
    for attempt in range(retries + 1):
        logger.info(f"Executing: {' '.join(cmd_list)}")
        try:
            result = subprocess.run(cmd_list, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Step completed successfully.")
                return True
            else:
                logger.error(f"Attempt {attempt + 1} failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Exception during execution: {e}")
        
        if attempt < retries:
            time.sleep(delay)
            logger.info(f"Retrying step... (Attempt {attempt + 2})")
            
    return False

def process_file(data_path: Path):
    """Orchestrates the pipeline with localized logging per run."""
    market = data_path.parent.name
    year = data_path.stem
    artifacts_dir = Path("artifacts") / market / year
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Localized Logging ---
    log_file = artifacts_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)
    
    logger.info(f"--- Starting Pipeline for {market} {year} ---")
    
    # --- Robust Temporal Partitioning ---
    train_subset_path = artifacts_dir / "discovery_subset.parquet"
    df = pl.read_parquet(data_path)
    # Split index (50%) - ensures discovery only sees first half
    split_idx = int(len(df) * 0.5)
    df.slice(0, split_idx).write_parquet(train_subset_path)
    logger.info(f"Discovery subset created: {len(df.slice(0, split_idx))} rows.")

    # --- Pipeline Stages ---
    # NOTE: The analytics stage is now pointed to 'backtest_results.parquet'
    stages = [
        [sys.executable, "-m", "src.cli", "discover", "--data", str(train_subset_path), "--out", str(artifacts_dir / "manifest.json")],
        [sys.executable, "-m", "src.cli", "run", "--data", str(data_path), "--manifest", str(artifacts_dir / "manifest.json"), "--out", str(artifacts_dir)],
        [sys.executable, "-m", "src.analytics", str(artifacts_dir / "backtest_results.parquet")]
    ]

    for stage_cmd in stages:
        if not run_step(stage_cmd):
            logger.error(f"Pipeline failed at stage: {stage_cmd[2]}")
            break
    else:
        logger.info(f"--- Pipeline Completed Successfully for {market} {year} ---")

    # Cleanup log handler to prevent leaking handles
    file_handler.close()
    logger.removeHandler(file_handler)

if __name__ == "__main__":
    futures_dir = Path("futures")
    if not futures_dir.exists():
        logger.error("Directory 'futures' not found.")
        sys.exit(1)

    files = list(futures_dir.rglob("*.parquet"))
    for file_path in files:
        process_file(file_path)