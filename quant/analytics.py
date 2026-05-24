"""
src/analytics.py
Calculates performance metrics from walk-forward simulation output.
Uses the 'pnl' column from backtest_results.parquet (produced by src.walkforward).
Also computes correlation between predictions and actual returns if available,
and compares against a naive benchmark column 'benchmark_pnl' if present.
"""
import sys
import polars as pl
import numpy as np

def calculate_metrics(file_path: str):
    """
    Reads a Parquet file (expected to have columns: pnl, position, prediction, target_5m)
    and prints key performance statistics.
    """
    try:
        df = pl.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Required column: pnl (from execution simulation)
    if "pnl" not in df.columns:
        print("No 'pnl' column found. Ensure backtest_results.parquet contains execution output.")
        return

    pnl = df["pnl"].to_numpy()
    total_pnl = pnl.sum()
    avg_pnl = pnl.mean()
    std_pnl = pnl.std()
    
    # Annualized Sharpe (assuming 5-min bars, 264 bars per session, 252 trading days)
    if std_pnl > 0:
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252 * 264)
    else:
        sharpe = 0.0

    # Cumulative PnL and max drawdown
    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_drawdown = drawdown.min()

    # Turnover: sum of absolute position changes divided by average position (if any)
    if "position" in df.columns:
        position_changes = df["position"].diff().abs().sum()
        avg_position = df["position"].abs().mean()
        turnover = position_changes / avg_position if avg_position > 0 else 0.0
    else:
        turnover = 0.0

    # Optional: correlation between prediction and target_5m (if available)
    corr = 0.0
    if "prediction" in df.columns and "target_5m" in df.columns:
        pred = df["prediction"].to_numpy()
        target = df["target_5m"].to_numpy()
        mask = ~(np.isnan(pred) | np.isnan(target))
        if mask.sum() > 1:
            corr = np.corrcoef(pred[mask], target[mask])[0, 1]

    # Benchmark comparison if benchmark_pnl exists
    benchmark_sharpe = None
    benchmark_maxdd = None
    if "benchmark_pnl" in df.columns:
        bench_pnl = df["benchmark_pnl"].to_numpy()
        bench_avg = bench_pnl.mean()
        bench_std = bench_pnl.std()
        if bench_std > 0:
            benchmark_sharpe = (bench_avg / bench_std) * np.sqrt(252 * 264)
        else:
            benchmark_sharpe = 0.0
        bench_cum = np.cumsum(bench_pnl)
        bench_running_max = np.maximum.accumulate(bench_cum)
        benchmark_maxdd = (bench_cum - bench_running_max).min()

    # Print report
    print("\n" + "="*50)
    print("            PERFORMANCE REPORT")
    print("="*50)
    print(f"Total PnL:            {total_pnl:>12.4f}")
    print(f"Avg PnL per bar:      {avg_pnl:>12.6f}")
    print(f"Std PnL per bar:      {std_pnl:>12.6f}")
    print(f"Sharpe (ann.):        {sharpe:>12.3f}")
    print(f"Max Drawdown:         {max_drawdown:>12.4f}")
    print(f"Turnover:             {turnover:>12.4f}")
    print(f"Prediction-Target Corr:{corr:>12.4f}")
    if benchmark_sharpe is not None:
        print(f"Benchmark Sharpe:     {benchmark_sharpe:>12.3f}")
        print(f"Benchmark MaxDD:      {benchmark_maxdd:>12.4f}")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics <path_to_backtest_results.parquet>")
        sys.exit(1)
    calculate_metrics(sys.argv[1])