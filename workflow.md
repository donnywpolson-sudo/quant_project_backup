1. Data Collection & Engineering
Step 1 = data sourcing, timezone normalization, sessionization, raw manifest.
Step 2 = resampling and explicit ts_close.
Step 3 = HTF alignment enforcing causality.
Step 4 = HTF feature engineering (indicators, ATR, slopes).

2. Alpha Research & Modeling
Step 5 = HTF discovery and feature selection.
Step 6 = 5m baseline features.
Step 7 = joining frozen HTF to 5m (conditioning).
Step 8 = feature expansion and pruning.
Step 9 = 5m discovery conditioned on HTF.

3. Backtesting & Validation
Step 10 = walkforward Ridge training, per‑fold scalers, OOS metrics.
Step 11 = execution simulation produces trades and PnL for validation.
Step 12 = CI tests, reproducibility, no‑leakage checks.

4. Portfolio Optimization & Execution
Step 11 = execution simulation, HTF volatility sizing, slippage/latency modeling.
Step 12 = monitoring and CI gating for deploy.