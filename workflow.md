### Full end to end workflow

| **Step** | **Purpose** | **Key output** |
|---|---:|---|
| **1. Data ingest and sessionization** | Read 1‑min parquet, apply session filter and timezone rules | Raw 1‑min table with `session_id`, `ts_event` |
| **2. Resample to 5m, 1h, daily** | Aggregate OHLCV using O=first H=max L=min C=last V=sum | Three streams: `5m`, `1h`, `daily` parquet files |
| **3. HTF alignment** | As‑of join: attach most‑recent closed `1h` and `daily` values to each 5m row | 5m rows with `htf_ts`, `1h_*`, `daily_*` columns |
| **4. HTF feature engineering** | Compute HTF features: trend slopes, distance to levels, ATR/vol ratios, regime labels | `htf_*` feature columns (float32, clipped) |
| **5. HTF discovery (ExtraTrees)** | Run ExtraTrees on HTF streams only; stability selection → freeze HTF features | `manifest_htf.json` with selected HTF features and SHA256 |
| **6. 5m baseline features** | Compute YAML baseline features for each 5m row | Baseline feature matrix (float32, clipped) |
| **7. Join frozen HTF to 5m** | Attach frozen HTF columns (no lookahead) to 5m matrix | Full 5m feature matrix with HTF columns |
| **8. Feature expansion and pruning** | Intra‑timeframe and capped cross‑timeframe interactions; variance/corr pruning on train fold | Expanded matrix; correlation pruned list |
| **9. 5m discovery (ExtraTrees conditioned)** | Run ExtraTrees on 5m matrix conditioned on frozen HTF; stability selection → freeze features | `manifest_5m.json` referencing `manifest_htf.json` |
| **10. Walkforward Ridge training** | Per fold: fit scaler on train, train Ridge on frozen features, evaluate OOS | Fold models, OOS metrics, aggregated report |
| **11. Execution simulation and HTF scaling** | Simulate execution using HTF vol scaling and trend alignment; produce trades/PnL | Trades CSV, PnL series, metrics |
| **12. Monitoring, manifest hash checks, CI** | Repro tests, memory safety, no‑leakage tests, manifest hash equality | CI pass/fail, reproducible manifests, alerts |
