import os
import sys
import subprocess
import logging
import pandas as pd
from pathlib import Path
import yaml

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
# CORRELATION FILTER
# =========================
def select_uncorrelated_markets(files, config):
    if not config.get("use_correlation_filter", False):
        return files

    logger.info("Running correlation filtering...")

    market_returns = {}

    for f in files:
        market = f.parent.name

        df = pd.read_parquet(f, columns=["close"])
        returns = df["close"].pct_change().dropna()

        if market not in market_returns:
            market_returns[market] = returns
        else:
            market_returns[market] = pd.concat([market_returns[market], returns])

    df = pd.DataFrame({m: r for m, r in market_returns.items()}).dropna()

    corr = df.corr().abs()

    selected = []
    threshold = config.get("correlation_threshold", 0.75)

    for m in corr.columns:
        if all(corr.loc[m, s] < threshold for s in selected):
            selected.append(m)

    max_markets = config.get("max_markets")
    if max_markets:
        selected = selected[:max_markets]

    logger.info(f"Selected markets: {selected}")

    return [f for f in files if f.parent.name in selected]

# =========================
# CREATE WALKFORWARD WINDOWS
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

# =========================
# PROCESS SPLITS
# =========================
def process_split(train_years, test_years, files):

    subset = [
        f for f in files
        if int(f.stem) in train_years + test_years
    ]

    for f in subset:
        logger.info(f"Processing {f}")

        subprocess.run([
            sys.executable, "-m", "quant.cli", "discover",
            "--data", str(f)
        ])

        subprocess.run([
            sys.executable, "-m", "quant.cli", "run",
            "--data", str(f)
        ])

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    config = load_config()

    data_dir = config.get("data_dir", "data")

    files = get_files(data_dir, config)

    files = select_uncorrelated_markets(files, config)

    if not files:
        logger.warning("No files found after filtering")
        sys.exit(0)

    splits = generate_walkforward_splits(files, config)

    for i, (train, test) in enumerate(splits, 1):
        logger.info(f"[Split {i}] Train: {train} | Test: {test}")
        process_split(train, test, files)

    logger.info("Done")