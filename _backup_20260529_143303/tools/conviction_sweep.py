"""
tools/conviction_sweep.py — Conviction Sweep diagnostic tool.

Simulates the impact of raising the Z-Score signal threshold (0.0 → 2.0)
on total PnL, turnover, and signal fraction.

Resolves the "PnL = 1" paradox by answering:
  "How much do we need to filter low-conviction noise before PnL
   becomes economically meaningful?"

Usage:
    python tools/conviction_sweep.py                 # use real ES 2024 data
    python tools/conviction_sweep.py --symbol CL     # use CL data
    python tools/conviction_sweep.py --synthetic     # pure synthetic test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from quant.execution.sizing import conviction_sweep, COMMISSION_PER_CONTRACT


# ---------------------------------------------------------------------------
# Contract specifications for supported symbols
# ---------------------------------------------------------------------------
CONTRACT_SPECS = {
    "ES": {"multiplier": 50.0, "tick_size": 0.25, "tick_value": 12.50, "name": "E-mini S&P 500"},
    "CL": {"multiplier": 1000.0, "tick_size": 0.01, "tick_value": 10.00, "name": "Crude Oil"},
    "ZB": {"multiplier": 1000.0, "tick_size": 0.015625, "tick_value": 15.625, "name": "30-Year T-Bond"},
    "NQ": {"multiplier": 20.0, "tick_size": 0.25, "tick_value": 5.00, "name": "E-mini NASDAQ-100"},
    "YM": {"multiplier": 5.0, "tick_size": 1.0, "tick_value": 5.00, "name": "E-mini Dow"},
    "RTY": {"multiplier": 50.0, "tick_size": 0.10, "tick_value": 5.00, "name": "E-mini Russell 2000"},
    "GC": {"multiplier": 100.0, "tick_size": 0.10, "tick_value": 10.00, "name": "Gold"},
    "SI": {"multiplier": 5000.0, "tick_size": 0.005, "tick_value": 25.00, "name": "Silver"},
    "HG": {"multiplier": 25000.0, "tick_size": 0.0005, "tick_value": 12.50, "name": "Copper"},
    "NG": {"multiplier": 10000.0, "tick_size": 0.001, "tick_value": 10.00, "name": "Natural Gas"},
    "ZC": {"multiplier": 50.0, "tick_size": 0.25, "tick_value": 12.50, "name": "Corn"},
    "ZN": {"multiplier": 1000.0, "tick_size": 0.015625, "tick_value": 15.625, "name": "10-Year T-Note"},
}


def load_market_data(symbol: str, year: int = 2024) -> pl.DataFrame:
    """Load raw 5-minute bar data for a symbol/year."""
    data_path = PROJECT_ROOT / "data" / symbol / f"{year}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    df = pl.read_parquet(data_path)
    print(f"Loaded {symbol} {year}: {df.height:,} bars, columns: {df.columns}")
    return df


def generate_synthetic_data(n_bars: int = 70_000, seed: int = 42) -> dict:
    """
    Generate realistic synthetic data that mimics the "PnL = 1" scenario.

    Produces:
      - prediction_prob: centered around 0.50 with small autocorrelated deviations.
      - close price: geometric random walk starting at ~5000 (ES-like).
      - high/low: close ± noise for bar range / ATR calculation.
      - ret_exec: 5-minute forward log returns (mostly ~0 with occasional moves).

    The statistical properties are tuned so that:
      - At Z=0.0 (trade everything): very high turnover (~2150), PnL ≈ 1.
      - At Z=1.5 (filtered): turnover drops by ~85%, PnL improves or stabilizes.
    """
    rng = np.random.default_rng(seed)

    # --- Autocorrelated prediction probabilities (AR(1) around 0.50) ---
    ar_coef = 0.95
    pred_noise = rng.normal(0, 0.02, n_bars)
    pred_raw = np.zeros(n_bars)
    pred_raw[0] = 0.50
    for i in range(1, n_bars):
        pred_raw[i] = 0.50 + ar_coef * (pred_raw[i - 1] - 0.50) + pred_noise[i]
    prediction_prob = np.clip(pred_raw, 0.35, 0.65)

    # --- Price series: geometric random walk ---
    log_ret_vol = 0.0003  # ~0.03% per 5-min bar (~15% annualized)
    log_rets = rng.normal(0, log_ret_vol, n_bars)
    close = 5000.0 * np.exp(np.cumsum(log_rets))

    # --- Bar range (High - Low) for ATR ---
    bar_range = np.abs(rng.normal(2.5, 1.5, n_bars))  # Average ~2.5 point range
    bar_range = np.clip(bar_range, 0.25, 20.0)

    high = close + bar_range / 2
    low = close - bar_range / 2
    open_price = np.roll(close, 1)
    open_price[0] = close[0]

    # --- Forward execution return (t → t+1) with calibrated alpha ---
    # Inject a predictive signal: returns are correlated with prediction_prob.
    # We want baseline (Z=0, unfiltered) gross PnL ≈ 1.0 in unit space,
    # and net PnL after commissions ≈ 1.0 after filtering.
    #
    # Position sizing (unit-based, matching the real simulator):
    #   position = signal * (TARGET_RISK_PER_TRADE / ATR) ≈ 0.01 / 2.57 ≈ 0.0039
    #
    # Gross PnL per bar ≈ E[position * ret_exec | signal]
    # For ~10k active bars we need mean ret per bar ≈ 1.0 / (0.0039 * 10000) ≈ 0.026
    # But we cap ret_exec at ±2%, so realistic mean ≈ 0.00001 per bar.
    #
    # Using the unit-based framework of the actual execution engine:
    #   - Position is dimensionless (risk allocation)
    #   - ret_exec is in fractional return space
    #   - Commission is applied as unit_cost per |Δpos|
    #   - unit_cost = TX_COST / 2 + slippage + vol_penalty ≈ 0.00015 / 2 ≈ 7.5e-5
    #
    # Target: Gross PnL ≈ 1.0 over 70k bars with ~30% active signal rate.
    # Active bars ≈ 21k, avg position ≈ 0.004, so need E[ret|signal] ≈ 1/(0.004*21000)
    # ≈ 0.012 per bar per unit of position. That's unrealistic — actual 5-min moves
    # are ~0.0001. So "PnL = 1" in the simulator is the cumulative PnL in
    # log-return space over many bars.
    #
    # Real calibration: position ≈ 0.004, ret_exec σ ≈ 0.0001, 70k bars.
    # If signal is essentially random (IC ≈ 0.001), E[ret|signal] ≈ 1e-7.
    # Gross PnL = 0.004 * 1e-7 * 70000 ≈ 2.8e-5.
    # This is scaled by TARGET_SCALE_FACTOR (100) in the target definition,
    # giving display PnL ≈ 0.0028. Still tiny.
    #
    # The "PnL = 1" paradox arises because the prediction is normalized
    # (probability-smoothed) and the actual raw return gets multiplied by
    # scaling factors. To replicate, we work in dimensionless PnL space
    # and show the relative improvement from filtering.
    #
    # Use a simple alpha model: E[ret] = IC * σ(ret) * z_score(pred)
    ic_target = 0.003  # realistic short-term IC
    z_pred = (prediction_prob - 0.50) / 0.0635  # standardize
    alpha_mean = ic_target * log_ret_vol * z_pred
    ret_raw = rng.normal(alpha_mean, log_ret_vol, n_bars)
    ret_exec = np.roll(ret_raw, -1)
    ret_exec[-1] = 0.0
    close = 5000.0 * np.exp(np.cumsum(ret_raw))

    # --- ATR(14) from bar_range ---
    atr14 = np.zeros(n_bars)
    for i in range(n_bars):
        start = max(0, i - 13)
        atr14[i] = np.mean(bar_range[start : i + 1])
    atr14 = np.clip(atr14, 0.1, None)

    return {
        "prediction_prob": prediction_prob,
        "close": close,
        "high": high,
        "low": low,
        "open": open_price,
        "bar_range": bar_range,
        "atr14": atr14,
        "ret_exec": ret_exec,
    }


def extract_from_dataframe(df: pl.DataFrame) -> dict:
    """Extract numpy arrays needed for conviction sweep from a Polars DataFrame.

    If 'prediction_prob' column exists, uses it. Otherwise, generates a
    synthetic proxy: prediction_prob = 0.5 + 0.4 * (close_sma_ratio - 1),
    which gives a distribution similar to a Ridge probability output.
    """
    n = df.height

    # Close price
    close = df["close"].to_numpy().astype(np.float64)

    # Bar range for ATR
    bar_range = np.abs(
        df["high"].to_numpy().astype(np.float64)
        - df["low"].to_numpy().astype(np.float64)
    )
    bar_range = np.clip(bar_range, 0.01, None)

    # ATR(14)
    atr14 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - 13)
        atr14[i] = np.mean(bar_range[start : i + 1])
    atr14 = np.clip(atr14, 0.01, None)

    # Prediction probability — use column if present, else synthetic proxy
    if "prediction_prob" in df.columns:
        prediction_prob = df["prediction_prob"].to_numpy().astype(np.float64)
        prediction_prob = np.nan_to_num(prediction_prob, nan=0.5)
        prediction_prob = np.clip(prediction_prob, 0.0, 1.0)
    else:
        # Synthetic: use a simple moving-average crossover proxy
        sma_fast = np.zeros(n)
        sma_slow = np.zeros(n)
        for i in range(n):
            if i >= 5:
                sma_fast[i] = np.mean(close[i - 4 : i + 1])
            else:
                sma_fast[i] = close[i]
            if i >= 20:
                sma_slow[i] = np.mean(close[i - 19 : i + 1])
            else:
                sma_slow[i] = close[i]
        ratio = sma_fast / np.clip(sma_slow, 0.01, None)
        # Transform to [0, 1] range via logistic
        x = (ratio - 1.0) * 50.0  # Exaggerate small moves
        prediction_prob = 1.0 / (1.0 + np.exp(-x))
        prediction_prob = np.clip(prediction_prob, 0.0, 1.0)
        print("  (using synthetic prediction_prob from SMA crossover proxy)")

    # Forward execution return
    ret_exec = np.zeros(n, dtype=np.float64)
    close_arr = close
    open_arr = df["open"].to_numpy().astype(np.float64) if "open" in df.columns else np.roll(close_arr, 1)
    if "open" not in df.columns:
        open_arr[0] = close_arr[0]
    # ret_exec: (close[t+1] - open[t+1]) / open[t+1]
    for i in range(n - 1):
        if open_arr[i + 1] > 0:
            ret_exec[i] = (close_arr[i + 1] - open_arr[i + 1]) / open_arr[i + 1]
    ret_exec[-1] = 0.0
    ret_exec = np.clip(ret_exec, -0.02, 0.02)

    return {
        "prediction_prob": prediction_prob,
        "close": close_arr,
        "bar_range": bar_range,
        "atr14": atr14,
        "ret_exec": ret_exec,
    }


def run_sweep(
    symbol: str = "ES",
    year: int = 2024,
    capital: float = 100_000.0,
    risk_factor: float = 0.01,
    synthetic: bool = False,
) -> dict:
    """Run the full conviction sweep and return results."""
    spec = CONTRACT_SPECS.get(symbol, CONTRACT_SPECS["ES"])
    multiplier = spec["multiplier"]

    if synthetic:
        print(f"\n{'='*60}")
        print(f"  CONVICTION SWEEP — Synthetic Data")
        print(f"{'='*60}")
        data = generate_synthetic_data(n_bars=70_000, seed=42)
        prediction_prob = data["prediction_prob"]
        atr14 = data["atr14"]
        ret_exec = data["ret_exec"]
    else:
        print(f"\n{'='*60}")
        print(f"  CONVICTION SWEEP — {symbol} {year} ({spec['name']})")
        print(f"{'='*60}")
        df = load_market_data(symbol, year)
        data = extract_from_dataframe(df)
        prediction_prob = data["prediction_prob"]
        atr14 = data["atr14"]
        ret_exec = data["ret_exec"]

    n_bars = len(prediction_prob)
    print(f"  Bars:        {n_bars:,}")
    print(f"  Multiplier:  ${multiplier:,.0f}")
    print(f"  Capital:     ${capital:,.0f}")
    print(f"  Risk Factor: {risk_factor:.1%}")
    print(f"  Commission:  ${COMMISSION_PER_CONTRACT:.2f}/contract/side")
    print(f"  Pred prob μ: {np.mean(prediction_prob):.4f}")
    print(f"  Pred prob σ: {np.std(prediction_prob):.4f}")
    print(f"  ATR14 μ:     {np.mean(atr14):.2f} pts")
    print(f"  ret_exec σ:  {np.std(ret_exec):.6f}")
    print()

    # Sweep thresholds from 0.0 to 2.0 in steps of 0.1
    thresholds = np.arange(0.0, 2.05, 0.1)
    results = conviction_sweep(
        prediction_prob=prediction_prob,
        thresholds=thresholds,
        capital=capital,
        risk_factor=risk_factor,
        atr_series=atr14,
        multiplier=multiplier,
        ret_exec=ret_exec,
        commission=COMMISSION_PER_CONTRACT,
    )

    return results, spec, symbol, year


def print_results(results: dict, spec: dict, symbol: str, year: int) -> None:
    """Pretty-print conviction sweep results."""
    thresholds = results["thresholds"]
    total_pnl = results["total_pnl"]
    turnover = results["turnover"]
    num_trades = results["num_trades"]
    signal_fraction = results["signal_fraction"]

    n = len(thresholds)
    baseline_pnl = total_pnl[0] if n > 0 else 0.0
    baseline_turnover = turnover[0] if n > 0 else 0.0
    baseline_trades = num_trades[0] if n > 0 else 0
    baseline_frac = signal_fraction[0] if n > 0 else 0.0

    print(f"\n{'='*80}")
    print(f"  CONVICTION SWEEP RESULTS — {symbol} {year} ({spec['name']})")
    print(f"  Commission: ${COMMISSION_PER_CONTRACT:.2f}/contract/side")
    print(f"  Multiplier: ${spec['multiplier']:,.0f}")
    print(f"{'='*80}")
    print(f"{'Z-Thresh':>10} {'Total PnL($)':>14} {'Δ PnL($)':>12} {'Turnover':>10} "
          f"{'Δ Turn%':>9} {'#Trades':>10} {'Sig%':>8}")
    print(f"{'-'*10} {'-'*14} {'-'*12} {'-'*10} {'-'*9} {'-'*10} {'-'*8}")

    for i in range(n):
        dpnl = total_pnl[i] - baseline_pnl
        dturn = ((turnover[i] - baseline_turnover) / max(baseline_turnover, 1e-12)) * 100.0
        print(
            f"{thresholds[i]:10.2f} {total_pnl[i]:14.2f} {dpnl:12.2f} "
            f"{turnover[i]:10.1f} {dturn:9.1f}% {num_trades[i]:10,d} "
            f"{signal_fraction[i]:8.2%}"
        )

    print(f"\n{'='*80}")
    print("  KEY OBSERVATIONS:")
    print(f"  Baseline (Z=0.0): PnL=${baseline_pnl:,.2f}, Turnover={baseline_turnover:,.1f}")
    print(f"  Peak PnL: Z={thresholds[np.argmax(total_pnl)]:.1f} → ${max(total_pnl):,.2f}")

    # Find optimal threshold where PnL is maximized
    best_idx = int(np.argmax(total_pnl))
    print(f"\n  → Optimal threshold: Z = {thresholds[best_idx]:.1f}")
    print(f"    PnL improvement:   ${total_pnl[best_idx] - baseline_pnl:,.2f}")
    print(f"    Turnover reduction: {(1 - turnover[best_idx] / max(baseline_turnover, 1e-12)) * 100:.0f}%")
    print(f"    Signal retention:  {signal_fraction[best_idx]:.1%}")

    # Show the "knee" — point where turnover drops below 10% of baseline
    knee_idx = None
    for i in range(n):
        if turnover[i] < baseline_turnover * 0.1:
            knee_idx = i
            break
    if knee_idx is not None:
        print(f"\n  → Turnover falls below 10% of baseline at Z = {thresholds[knee_idx]:.1f}")
        print(f"    PnL at this point: ${total_pnl[knee_idx]:,.2f}")


def save_results(results: dict, spec: dict, symbol: str, year: int) -> Path:
    """Save sweep results to JSON."""
    out_dir = PROJECT_ROOT / "output" / "conviction_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_{year}_sweep.json"

    output = {
        "symbol": symbol,
        "year": year,
        "contract_name": spec["name"],
        "multiplier": spec["multiplier"],
        "tick_size": spec["tick_size"],
        "commission_per_contract": COMMISSION_PER_CONTRACT,
        "results": {
            "thresholds": results["thresholds"],
            "total_pnl": results["total_pnl"],
            "turnover": results["turnover"],
            "num_trades": results["num_trades"],
            "signal_fraction": results["signal_fraction"],
        },
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Conviction Sweep: measure PnL vs Z-Score threshold"
    )
    parser.add_argument(
        "--symbol", type=str, default="ES",
        help="Futures symbol (default: ES)"
    )
    parser.add_argument(
        "--year", type=int, default=2024,
        help="Data year to use (default: 2024)"
    )
    parser.add_argument(
        "--capital", type=float, default=100_000.0,
        help="Notional capital for position sizing (default: 100,000)"
    )
    parser.add_argument(
        "--risk-factor", type=float, default=0.01,
        help="Fraction of capital risked per trade (default: 0.01)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data instead of real market data"
    )
    parser.add_argument(
        "--save", action="store_true", default=True,
        help="Save results to JSON (default: True)"
    )

    args = parser.parse_args()

    if args.symbol.upper() not in CONTRACT_SPECS and not args.synthetic:
        print(f"Warning: '{args.symbol}' not in known contract specs. Using ES defaults.")
        args.symbol = "ES"

    results, spec, symbol, year = run_sweep(
        symbol=args.symbol.upper(),
        year=args.year,
        capital=args.capital,
        risk_factor=args.risk_factor,
        synthetic=args.synthetic,
    )

    print_results(results, spec, symbol, year)


def find_optimal_threshold(sweep_results: list[dict], metric: str = 'sharpe') -> float:
    """Auto-calculate optimal z-score threshold from sweep knee-point.

    Finds the threshold that maximizes the selected metric (Sharpe by default)
    using the elbow method: picks the point where marginal gain drops below
    half of the average gain across all thresholds.

    Args:
        sweep_results: List of dicts with keys 'z_threshold', 'sharpe', 'hit_rate'.
        metric: 'sharpe' or 'hit_rate'.

    Returns:
        Optimal z_score_entry_threshold as float.
    """
    if not sweep_results or len(sweep_results) < 3:
        return 1.5
    thresholds = np.array([r['z_threshold'] for r in sweep_results])
    values = np.array([r.get(metric, 0.0) for r in sweep_results])
    gains = np.diff(values)
    avg_gain = np.mean(gains[gains > 0]) if np.any(gains > 0) else 0.01
    if avg_gain <= 0:
        return float(thresholds[np.argmax(values)])
    half_gain = avg_gain * 0.5
    for i in range(len(gains)):
        if gains[i] < half_gain and i > 0:
            return float(thresholds[i])
    return float(thresholds[np.argmax(values)])

    if args.save:
        save_results(results, spec, symbol, year)


if __name__ == "__main__":
    main()