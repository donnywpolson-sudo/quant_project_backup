"""
src/analytics.py
Calculates performance metrics from walk-forward simulation output.
"""
import sys
import polars as pl
import numpy as np

def calculate_metrics(file_path: str):
    """
    Computes Hit Rate, Sharpe Ratio, and Drawdown.
    
    Expects a parquet file with the following columns:
    - 'prediction': The model's binary/class prediction
    - 'target': The actual ground truth
    - 'close': Price data (used to derive returns if 'returns' is missing)
    """
    try:
        df = pl.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    # 1. Ensure 'returns' column exists
    # If not present, calculate daily/minute returns from 'close' price
    if "returns" not in df.columns:
        if "close" in df.columns:
            df = df.with_columns(
                (pl.col("close").diff().shift(-1) / pl.col("close")).fill_null(0).alias("returns")
            )
        else:
            print("Error: Could not calculate returns. 'returns' or 'close' column missing.")
            return

    # 2. Hit Rate (Accuracy)
    # Comparison of prediction vs target
    hit_rate = (df["prediction"] == df["target"]).mean()
    
    # 3. Strategy Returns
    # Assuming prediction 1 = Long, 0 = Flat
    df = df.with_columns(
        (df["prediction"] * df["returns"]).alias("strategy_returns")
    )
    
    # 4. Sharpe Ratio (Annualized proxy)
    # Assuming 1-minute data frequency, 390 minutes per trading day, 252 trading days
    excess_returns = df["strategy_returns"].mean()
    volatility = df["strategy_returns"].std()
    
    # Avoid division by zero
    sharpe = (excess_returns / volatility) * np.sqrt(252 * 390) if volatility > 0 else 0
    
    # 5. Equity Curve & Drawdown
    equity_curve = (1 + df["strategy_returns"]).cum_prod()
    peak = equity_curve.cum_max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()
    
    # Output Report
    print("\n" + "="*30)
    print("      PERFORMANCE REPORT")
    print("="*30)
    print(f"Hit Rate:        {hit_rate:.2%}")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Max Drawdown:    {max_drawdown:.2%}")
    print(f"Final Equity:    {equity_curve[-1]:.4f}x")
    print("="*30 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.analytics <path_to_parquet>")
        sys.exit(1)
        
    calculate_metrics(sys.argv[1])