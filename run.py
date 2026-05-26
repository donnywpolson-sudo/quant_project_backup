import os
import sys
import subprocess
import logging
from pathlib import Path
import yaml
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QuantRunner")

# =========================
# LOAD CONFIG
# =========================
def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config")
        return config
    return {}

# =========================
# GET FILE LIST
# =========================
def get_files(data_dir, config):
    files = list(Path(data_dir).rglob("*.parquet"))

    start = config.get("start_year")
    end = config.get("end_year")

    valid = []
    for f in files:
        try:
            year = int(f.stem)
        except:
            continue

        if start and year < start:
            continue
        if end and year > end:
            continue

        valid.append(f)

    return valid

# =========================
# WALKFORWARD SPLITS
# =========================
def generate_walkforward_splits(files, config):
    train_years = config.get("training_years", 3)
    wf_years = config.get("walkforward_years", 1)

    years = sorted({int(f.stem) for f in files})
    splits = []

    for i in range(len(years)):
        train_start = i
        train_end = i + train_years
        test_end = train_end + wf_years

        if test_end > len(years):
            break

        train_range = years[train_start:train_end]
        test_range = years[train_end:test_end]

        splits.append((train_range, test_range))

    logger.info(f"Generated {len(splits)} walkforward splits")
    return splits


# ✅ ✅ FIXED CORE FUNCTION
def process_split(train_years, test_years, files):

    train_files = [f for f in files if int(f.stem) in train_years]
    test_files = [f for f in files if int(f.stem) in test_years]

    if not train_files or not test_files:
        logger.warning("Empty train/test split — skipping")
        return

    # ✅ STEP 1: BUILD COMBINED TRAIN DATASET
    combined_train_path = Path("artifacts") / f"train_{'_'.join(map(str, train_years))}.parquet"
    combined_train_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building TRAIN dataset for years {train_years}")

    dfs = []
    for f in train_files:
        logger.info(f"Adding train file: {f}")
        dfs.append(pl.read_parquet(f))

    train_df = pl.concat(dfs).sort("ts_event")
    train_df.write_parquet(combined_train_path)

    # ✅ STEP 2: DISCOVERY (TRAIN ONLY)
    manifest_path = Path("artifacts") / f"manifest_{'_'.join(map(str, train_years))}.json"

    logger.info("Running feature discovery on TRAIN data...")

    subprocess.run([
        sys.executable, "-m", "quant.cli", "discover",
        "--data", str(combined_train_path),
        "--out", str(manifest_path)
    ], check=True)

    # ✅ STEP 3: EVALUATION (TEST ONLY)
    for f in test_files:
        logger.info(f"Evaluating TEST file: {f}")

        out_dir = Path("artifacts") / f.parent.name / f.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run([
            sys.executable, "-m", "quant.cli", "run",
            "--data", str(f),
            "--manifest", str(manifest_path),
            "--out", str(out_dir)
        ], check=True)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    config = load_config()
    data_dir = config.get("data_dir", "data")

    files = get_files(data_dir, config)

    if not files:
        logger.warning("No files found after filtering")
        sys.exit(0)

    splits = generate_walkforward_splits(files, config)

    for i, (train, test) in enumerate(splits, 1):
        logger.info(f"[Split {i}] Train: {train} | Test: {test}")
        process_split(train, test, files)

    logger.info("Done")