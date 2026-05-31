# Project Layout / Runtime Map

This repo is organized by code responsibility. Runtime order is controlled by
`run.py` and `pipeline/cli.py`; do not create parallel modules just to match a
flowchart label.

| Runtime step | Source location | Runtime artifacts |
|---|---|---|
| 0. Raw data ingestion | `data/L0_ohlcv_1m/`, `data/L1_mbp1/` | local files under `data/` |
| 1. Dataset gate | `pipeline/data_gate/` | `output/reports/data_audit/audit_manifest.json` |
| 2. Canonical build/load | `pipeline/ingest/`, `pipeline/session/` | `output/cache/canonical_data_*.parquet` |
| 3. Alignment / continuous contracts | `pipeline/align/`, `pipeline/contracts/` | `output/cache/aligned_data_*.parquet` |
| 4. Feature + target matrix | `pipeline/features/`, `pipeline/target/` | `output/cache/full_feature_matrix_*.parquet` |
| 5. Train-only feature discovery | `pipeline/features/discovery.py` | `output/manifest_*_<profile>.json` |
| 6. Frozen feature manifest | `pipeline/features/discovery.py` | same manifest applied to test windows |
| 7. Walk-forward modeling | `pipeline/walkforward/` | OOS result frame |
| 8. OOS predictions | `pipeline/walkforward/`, `pipeline/cli.py` | `oos_predictions*.parquet` |
| 9. Execution simulation | `pipeline/execution/` | `backtest_results*.parquet` |
| 10. Risk gates | `pipeline/risk/` | `risk_report*.json` |
| 11. Metrics / analytics | `pipeline/analytics/` | `metrics_report*.json`, `output/aggregated/` |

## Shared infrastructure

- `pipeline/common/config.py`: profile config loading from `configs/alpha.yaml`.
- `pipeline/common/market.py`: market metadata and contract multiplier lookup.
- `pipeline/common/io/atomic.py`: atomic writes for Parquet/JSON outputs.
- `pipeline/common/io/canonical.py`: canonical Parquet write wrapper.

## Output policy

Use one runtime root:

```text
output/
  cache/
  logs/
  reports/
  aggregated/
  <market>/
```

Do not write new runtime files to a root-level `reports/` directory.

## Keep in Git

- `configs/`
- `pipeline/`
- `tests/`
- `data/*/*.py`
- `data/market_sessions.yaml`
- `README.md`
- `project_layout.md`
- `code_to_text.py`
- `codex_custom_instructions.md`

## Do not keep in Git

- `output/`
- root `reports/` legacy artifacts
- `full_code.txt`
- raw/downloaded `.parquet`, `.dbn.zst`, cache, model, and log files
