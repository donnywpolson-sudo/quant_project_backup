"""
run.py
Production Orchestrator for the Deterministic Quant Pipeline.
Scans 'futures/' for 1‑min OHLCV Parquet files (market/year/*.parquet)
and runs the two‑phase pipeline (discovery → walkforward) per file.

After all files are processed, it runs aggregate_metrics.py to produce
consolidated performance reports per market and across all markets.

NOW WITH REAL‑TIME OUTPUT STREAMING – NO MORE HIDDEN PROGRESS BARS.
"""
import subprocess
import sys
import logging
import time
import re
from pathlib import Path
from datetime import datetime

# Setup global logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QuantRunner")

def create_audit_snapshot(root_dir: str = "."):
    """
    Creates a full project snapshot (all text files) for audit purposes.
    Saves as full_code.md in the project root.
    Excludes directories and files that may contain secrets,
    and redacts any remaining API keys or credentials.
    """
    project_root = Path(root_dir).resolve()
    snapshot_filename = "full_code.md"

    # Directories to exclude completely
    exclude_dirs = {
        ".venv", "venv", "env", ".git", "__pycache__",
        "artifacts", "logs", "models", "node_modules",
        ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints",
        "dist", "build", "htmlcov", ".tox",
        ".config", "secrets", "credentials",
    }

    # Explicitly skip files that often hold secrets
    exclude_files = {
        ".env", ".env.local", ".env.production", ".env.secret",
        "config.py", "secrets.py", "credentials.py",
        "databento_key.txt", "api_key.txt",
        ".netrc", ".aws/credentials", ".gcloud/credentials.json",
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
                if file_path.name in exclude_files:
                    continue
                if any(keyword in file_path.name.lower() for keyword in ("key", "secret", "token", "credential", "password")):
                    continue
                if file_path.suffix.lower() in binary_extensions:
                    continue

                f.write(f"--- \n### File: {rel_path}\n")
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    # Redact Databento API keys and other common credential patterns
                    content = re.sub(
                        r'(DATABENTO_API_KEY\s*=\s*["\'])([^"\']+)(["\'])',
                        r'\1[REDACTED]\3',
                        content,
                        flags=re.IGNORECASE
                    )
                    content = re.sub(
                        r'(api_key\s*=\s*["\'])([^"\']+)(["\'])',
                        r'\1[REDACTED]\3',
                        content,
                        flags=re.IGNORECASE
                    )
                    content = re.sub(
                        r'(API_KEY\s*=\s*["\'])([^"\']+)(["\'])',
                        r'\1[REDACTED]\3',
                        content
                    )
                    content = re.sub(
                        r'(DATABENTO_API_KEY\s*=\s*)([^\s]+)',
                        r'\1[REDACTED]',
                        content
                    )
                    f.write("```\n" + content + "\n```\n\n")
                except Exception as e:
                    f.write(f"Error reading file: {e}\n\n")
        logger.info(f"✅ Audit snapshot saved to: {snapshot_filename}")
    except Exception as e:
        logger.error(f"Failed to create audit snapshot: {e}")

def run_step(cmd_list, retries=2, delay=5):
    """
    Executes a command with real‑time output streaming.
    Uses subprocess.Popen to show stdout/stderr immediately.
    """
    for attempt in range(retries + 1):
        logger.info(f"Executing: {' '.join(cmd_list)}")
        try:
            proc = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for line in proc.stdout:
                print(line, end='', flush=True)
            proc.wait()
            if proc.returncode == 0:
                logger.info("Step completed successfully.")
                return True
            else:
                logger.error(f"Attempt {attempt + 1} failed with return code {proc.returncode}")
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

    # CHANGED: src.cli -> quant.cli, src.analytics -> quant.analytics
    stage_discover = [
        sys.executable, "-m", "quant.cli", "discover",
        "--data", str(data_path),
        "--out", str(manifest_path)
    ]

    stage_run = [
        sys.executable, "-m", "quant.cli", "run",
        "--data", str(data_path),
        "--manifest", str(manifest_path),
        "--out", str(artifacts_dir)
    ]

    stage_analytics = [
        sys.executable, "-m", "quant.analytics",
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
    import argparse

    create_audit_snapshot()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="Path to root folder containing market/year parquet files (overrides 'futures')")
    args = parser.parse_args()

    # Allow env var fallback
    data_dir = args.data_dir or os.environ.get("DATA_DIR") or "futures"
    futures_dir = Path(data_dir)
    if not futures_dir.exists():
        logger.error("Directory '%s' not found.", data_dir)
        sys.exit(1)

    files = list(futures_dir.rglob("*.parquet"))
    if not files:
        logger.warning("No Parquet files found under '%s'.", data_dir)
        sys.exit(0)

    logger.info(f"Found {len(files)} file(s) to process in '{data_dir}'.")
    for file_path in files:
        process_file(file_path)

    logger.info("All files processed. Running aggregate_metrics.py to generate consolidated reports...")
    agg_result = subprocess.run([sys.executable, "aggregate_metrics.py"], capture_output=True, text=True)
    if agg_result.returncode == 0:
        logger.info("✅ Aggregated metrics saved to artifacts/aggregated/")
        print("\n" + agg_result.stdout)
    else:
        logger.error(f"Aggregation failed: {agg_result.stderr}")