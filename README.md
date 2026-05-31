# Deterministic Intraday Futures ML Backtester

`run.py` is the main entrypoint for the full walk-forward pipeline.

## Required layout

```text
configs/              profile and market configuration
pipeline/common/      config, market metadata, atomic/canonical IO helpers
pipeline/             ingest, features, targets, walk-forward, execution, analytics
data/                 local data utilities, calendars, raw/vendor data
run.py                top-level pipeline runner
requirements.txt      Python dependencies
```

Runtime artifacts are written under `output/` and are intentionally git-ignored.
Reports now live under `output/reports/`; do not use a separate root `reports/` folder.

Expected core data layout:

```text
data/
  L0_ohlcv_1m/{market}/{year}.parquet
  L1_mbp1/{market}/*.dbn.zst
```

Example:

```text
data/L0_ohlcv_1m/ES/2024.parquet
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the pipeline

```powershell
$env:CONFIG_ENV="alpha_0"
python run.py
```

Useful logging modes:

```powershell
$env:LOG_MODE="clean"    # concise default
$env:LOG_MODE="verbose"  # stream child process details
$env:LOG_MODE="debug"    # maximum diagnostics
```

Main logs are written to:

```text
output/logs/YYYY-MM-DD_HH-MM-SS_MARKET_PROFILE_RUNID.log
```

## Data audits

```powershell
python data/L0_ohlcv_1m/audit_L0_ohlcv_1m.py --root data/L0_ohlcv_1m --out output/reports/data_audit
python -m pipeline.data_gate.manifest build --root data/L0_ohlcv_1m --audit-dir output/reports/data_audit --out output/reports/data_audit/audit_manifest.json
```

Audit commands may exit nonzero when they detect real data/calendar issues, but they still write CSV reports before exiting.

## L1 Databento helper

```powershell
python data/L1_mbp1/download_L1_mbp1_dbn.py --market ES --dry-run-cost
python data/L1_mbp1/download_L1_mbp1_dbn.py --market ES
python data/L1_mbp1/audit_L1_mbp1_parquet.py --out output/reports/L1_mbp1_parquet_audit
```

Raw Databento `.dbn.zst` is already losslessly compressed. Keep it immutable under
`data/L1_mbp1/{market}/`; store derived monthly Parquet separately from raw DBN files.

Do not commit API keys. Use environment variables or a local `.env` file.

## Validation

```powershell
python -m py_compile run.py pipeline/data_gate/manifest.py code_to_text.py git_sync.py
python -m pytest -q
```
