# Deterministic Intraday Futures ML Backtester

`run.py` is the main entrypoint for the full walk-forward pipeline.

## Required layout

```text
configs/              profile and market configuration
core/                 config, market metadata, atomic/canonical IO helpers
data/                 local data utilities and session calendars
pipeline/             feature, target, walk-forward, execution, analytics code
run.py                top-level pipeline runner
requirements.txt      Python dependencies
```

Runtime outputs are written under `output/` and are intentionally git-ignored.

Expected local data layout:

```text
data/
  ohlcv_1m/{market}/{year}.parquet
  l1_mbp1/
    raw_dbn/{market}/...
    parquet/{market}/...
    features/{market}/...
    manifests/
```

Example:

```text
data/ohlcv_1m/ES/2024.parquet
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
output/logs/run_YYYYMMDD_HHMMSS_{profile}_{symbols}_{run_id}.log
```

## Data audits

```powershell
python data/audit_core_data.py --root data --out output/audits/data_audit
python data/audit_sessions.py --root data --config data/market_sessions.yaml --out output/audits/session_audit
```

Audit commands may exit nonzero when they detect real data or calendar issues, but they still write CSV reports before exiting.

## L1 Databento helper

```powershell
python data/download_databento_L1_mbp1.py --dry-run-cost
python data/download_databento_L1_mbp1.py
```

Raw Databento `.dbn.zst` is already losslessly compressed. Keep it immutable under
`data/l1_mbp1/raw_dbn/`; store derived/reduced Parquet under `data/l1_mbp1/parquet/`
or features under `data/l1_mbp1/features/`.

Do not commit API keys. Use environment variables or a local `.env` file.

## Validation

```powershell
python -m py_compile run.py
python -m py_compile data/audit_core_data.py data/audit_sessions.py
python -m pytest -q
```
