import json
import polars as pl
import numpy as np
import sys
from pathlib import Path

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def main():
    # Allow command line argument for different market/year
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
    else:
        results_path = Path("artifacts/ES/2026/backtest_results.parquet")
    
    manifest_path = results_path.parent / "manifest.json"

    print_section(f"DIAGNOSTIC REPORT - {results_path.parent.name}")

    # 1. Check manifest
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        selected_features = manifest["feature_names"]
        print(f"\n1. Selected features: {len(selected_features)}")
        if len(selected_features) == 0:
            print("   WARNING: No features selected! Discovery thresholds may be too strict.")
        else:
            print(f"   First 5 features: {selected_features[:5]}")
    else:
        print("\n1. Manifest not found – skipping feature list.")

    # 2. Load backtest results
    if not results_path.exists():
        print(f"\nERROR: {results_path} not found.")
        return
    df = pl.read_parquet(results_path)
    print(f"\n2. Data shape: {df.shape}")
    print(f"   Columns: {df.columns[:10]}...")

    # Check for target_sign leakage in backtest results
    if "target_sign" in df.columns:
        print("   WARNING: target_sign present in backtest results (should be only in feature matrix)")

    # 3. Check predictions / probabilities
    if "prediction_prob" in df.columns:
        pred = df["prediction_prob"].to_numpy()
        print(f"\n3. Prediction probabilities (class 1):")
        print(f"   Mean: {pred.mean():.6f}")
        print(f"   Std:  {pred.std():.6f}")
        print(f"   Min:  {pred.min():.6f}")
        print(f"   Max:  {pred.max():.6f}")
        print(f"   Fraction >0.5: {(pred > 0.5).mean():.4f}")
    elif "prediction" in df.columns:
        pred = df["prediction"].to_numpy()
        print(f"\n3. Predictions (regression):")
        print(f"   Mean: {pred.mean():.8f}")
        print(f"   Std:  {pred.std():.8f}")
        print(f"   Min:  {pred.min():.8f}")
        print(f"   Max:  {pred.max():.8f}")
        print(f"   Non-zero: {np.count_nonzero(pred)} / {len(pred)}")
        if np.all(pred == 0):
            print("   WARNING: All predictions are zero!")
    else:
        print("\n3. No prediction column found.")

    # 4. Check PnL
    if "pnl" in df.columns:
        pnl = df["pnl"].to_numpy()
        total_pnl = pnl.sum()
        print(f"\n4. PnL:")
        print(f"   Total: {total_pnl:.8f}")
        print(f"   Mean:  {pnl.mean():.8f}")
        print(f"   Std:   {pnl.std():.8f}")
        print(f"   Min:   {pnl.min():.8f}")
        print(f"   Max:   {pnl.max():.8f}")
        if total_pnl == 0:
            print("   WARNING: Total PnL is exactly zero.")
            if np.all(pnl == 0):
                print("   All PnL entries are zero.")
            else:
                print("   Non-zero PnL entries exist but sum to zero.")
    else:
        print("\n4. No 'pnl' column found.")

    # 5. Check positions
    if "position" in df.columns:
        pos = df["position"].to_numpy()
        print(f"\n5. Positions:")
        print(f"   Mean absolute: {np.abs(pos).mean():.4f}")
        print(f"   Max absolute:  {np.abs(pos).max():.4f}")
        print(f"   Unique values (first 10): {np.unique(pos)[:10]}")
        if np.all(pos == 0):
            print("   WARNING: All positions are zero (no trades executed).")
    else:
        print("\n5. No 'position' column found.")

    # --- FIRST 10 ROWS CHECK ---
    print("\n--- FIRST 10 ROWS CHECK ---")
    select_cols = ["ts_event"]
    if "prediction_prob" in df.columns:
        select_cols.append("prediction_prob")
    if "target_sign" in df.columns:
        select_cols.append("target_sign")
    if "position" in df.columns:
        select_cols.append("position")
    if "pnl" in df.columns:
        select_cols.append("pnl")
    if "benchmark_pnl" in df.columns:
        select_cols.append("benchmark_pnl")
    small = df.head(10).select(select_cols)
    print(small)

    # Check correlation between prediction_prob and target_sign (lookahead detection)
    if "prediction_prob" in df.columns and "target_sign" in df.columns:
        pred_prob = df["prediction_prob"].to_numpy()
        target_sign = df["target_sign"].to_numpy()
        # Remove rows where either is NaN
        mask = ~(np.isnan(pred_prob) | np.isnan(target_sign))
        if mask.sum() > 1:
            corr = np.corrcoef(pred_prob[mask], target_sign[mask])[0, 1]
            print(f"\nCorrelation prediction_prob vs target_sign: {corr:.4f}")
            if corr > 0.9:
                print("WARNING: prediction_prob is almost identical to target_sign – lookahead detected!")
        else:
            print("\nNot enough valid rows to compute correlation.")

    # 6. Feature-target correlation (if full feature matrix exists)
    feature_cache = results_path.parent / "full_feature_matrix.parquet"
    if feature_cache.exists():
        print(f"\n6. Loading feature matrix from {feature_cache}...")
        df_features = pl.read_parquet(feature_cache)
        if "target_sign" in df_features.columns:
            target = df_features["target_sign"].to_numpy()
            target_name = "target_sign"
        elif "target_5m" in df_features.columns:
            target = df_features["target_5m"].to_numpy()
            target_name = "target_5m"
        else:
            target = None
            target_name = None

        if target is not None:
            exclude = {"ts_event", "open", "high", "low", "close", "volume", "session_id",
                       "date", "regime", "benchmark_pnl", "target_5m", "target_sign",
                       "prediction", "prediction_prob", "position", "trade_cost", "pnl"}
            feature_cols = [c for c in df_features.columns if c not in exclude and not c.startswith("_")]
            feature_cols = feature_cols[:50]  # limit to first 50 for speed

            print(f"   Computing correlation with {target_name} for first {len(feature_cols)} features...")
            corrs = []
            for col in feature_cols:
                feat = df_features[col].to_numpy()
                mask = ~(np.isnan(feat) | np.isnan(target))
                if mask.sum() > 10:
                    corr = np.corrcoef(feat[mask], target[mask])[0, 1]
                else:
                    corr = 0.0
                corrs.append((col, corr))
            corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            print("\n   Top 10 absolute correlations with target:")
            for col, c in corrs[:10]:
                if not np.isnan(c):
                    print(f"      {col}: {c:.4f}")
                else:
                    print(f"      {col}: nan")
            print("\n   Bottom 10 (most negative) correlations:")
            for col, c in corrs[-10:]:
                if not np.isnan(c):
                    print(f"      {col}: {c:.4f}")
                else:
                    print(f"      {col}: nan")
        else:
            print("\n6. No target column found in feature matrix.")
    else:
        print("\n6. Full feature matrix not found – skipping correlation analysis.")
        print("   (Looked for:", feature_cache, ")")

    print_section("END OF DIAGNOSTIC REPORT")

if __name__ == "__main__":
    main()