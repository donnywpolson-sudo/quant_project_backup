0. RAW DATA INGESTION

1. DATASET GATE
   raw data exists
   audit manifest exists
   selected files passed audit
   file size/mtime/hash unchanged

2. CANONICAL DATASET BUILD / LOAD
   schema normalization
   timezone/session normalization
   gap filtering
   cache canonical result

3. ALIGNMENT / CONTINUOUS CONTRACTS
   continuous contracts
   HTF alignment if enabled
   cache aligned result

4. FEATURE + TARGET BUILD / LOAD
   baseline features
   optional expansion
   targets
   cache feature matrix

5. TRAIN-ONLY FEATURE DISCOVERY
   use train window only
   select stable features

6. FROZEN FEATURE MANIFEST
   save selected feature list
   apply same feature list to test window

7. WALKFORWARD MODELING
   train on train window
   predict test window only

8. OOS PREDICTIONS
   save prediction_prob / raw signals

9. EXECUTION SIMULATION
   signal shift
   sizing
   costs/slippage
   PnL

10. RISK
    risk gates

11. METRICS REPORT
    risk gates
    Sharpe
    IC
    hit rate
    turnover
    gross vs net
    diagnostics