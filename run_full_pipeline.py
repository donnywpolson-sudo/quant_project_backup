"""
run_full_pipeline.py
Automated Orchestrator: Scans 'futures/' and processes all found parquet files.
"""
import subprocess
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineOrchestrator")

def run_step(cmd_list):
    """Executes a command and logs success/failure."""
    logger.info(f"Executing: {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        logger.error(f"Pipeline FAILED at command: {' '.join(cmd_list)}")
        return False
    logger.info("Step completed successfully.")
    return True

def process_file(data_path: Path):
    """Orchestrates the pipeline for a single market/year parquet file."""
    # Derive metadata from path: futures/{market}/{year}.parquet
    market = data_path.parent.name
    year = data_path.stem
    
    logger.info(f"--- Starting Pipeline for {market} {year} ---")
    
    # Define paths
    artifacts_dir = Path("artifacts") / market / year
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts_dir / "manifest.json"
    results_path = artifacts_dir / "final_features.parquet"

    # Stage 1: Feature Discovery
    success = run_step([
        sys.executable, "-m", "src.cli", "discover",
        "--data", str(data_path),
        "--out", str(manifest_path)
    ])
    if not success: return

    # Stage 2: Walk-Forward Simulation
    success = run_step([
        sys.executable, "-m", "src.cli", "run",
        "--data", str(data_path),
        "--manifest", str(manifest_path),
        "--out", str(artifacts_dir)
    ])
    if not success: return

    # Stage 3: Analytics
    success = run_step([
        sys.executable, "-m", "src.analytics", str(results_path)
    ])
    
    if success:
        logger.info(f"--- Pipeline Completed Successfully for {market} {year} ---")

if __name__ == "__main__":
    futures_dir = Path("futures")
    
    # Ensure futures directory exists
    if not futures_dir.exists():
        logger.error(f"Directory 'futures' not found. Please create it.")
        sys.exit(1)

    # Find all parquet files recursively (e.g., futures/ES/2026.parquet)
    files = list(futures_dir.rglob("*.parquet"))
    
    if not files:
        logger.warning(f"No parquet files found in {futures_dir.absolute()}.")
        sys.exit(0)
        
    logger.info(f"Found {len(files)} files. Starting batch processing...")
    
    for file in files:
        process_file(file)
        
    logger.info("All tasks completed.")