"""
sizing.py — Conviction-based position sizing and signal filtering.

Provides:
  - filter_signals():   Z-score gating — only trade when |z| > threshold.
  - get_position_size(): Volatility-adjusted position sizing using ATR.

Resolves the "PnL = 1" paradox by:
  1. Suppressing low-conviction noise (z-score filter).
  2. Scaling position inversely with volatility (risk-parity sizing).
  3. Supporting per-contract commission modeling ($1.50/contract).
"""
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Per-contract commission — realistic retail futures cost
# ---------------------------------------------------------------------------
COMMISSION_PER_CONTRACT: float = 1.50  # $1.50 per contract per side


def filter_signals(
    prediction_prob: np.ndarray,
    z_score_threshold: float = 1.5,
    window: int = 1000,
    min_periods: int = 50,
) -> np.ndarray:
    """
    Signal Significance Filter.

    Only generate trading signals when the rolling z-score of prediction_prob
    exceeds the absolute threshold.  Below threshold → zero position (no trade).

    This directly reduces turnover by ignoring low-conviction predictions
    where the model is indecisive (near its rolling mean).

    Args:
        prediction_prob: 1-D array of prediction probabilities [0, 1].
        z_score_threshold: Minimum |z-score| to trigger a trade (default 1.5).
        window: Rolling window for mean/std computation (default 1000 ≈ 3.5d of 5m bars).
        min_periods: Minimum observations before z-score is valid.

    Returns:
        1-D array of trading signals: +1 (long), -1 (short), 0 (no trade).
    """
    prob = np.asarray(prediction_prob, dtype=np.float64)
    n = len(prob)

    if n < min_periods:
        return np.zeros(n, dtype=np.float64)

    # Rolling mean & std using cumulative sums for vectorised speed
    cumsum = np.cumsum(np.insert(prob, 0, 0.0))
    cumsum2 = np.cumsum(np.insert(prob ** 2, 0, 0.0))

    roll_mean = np.full(n, np.nan)
    roll_std = np.full(n, np.nan)

    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        count = i - start + 1
        roll_mean[i] = (cumsum[i + 1] - cumsum[start]) / count
        variance = (cumsum2[i + 1] - cumsum2[start]) / count - roll_mean[i] ** 2
        roll_std[i] = np.sqrt(max(variance, 1e-12))

    # Z-score
    z_score = np.where(roll_std > 0, (prob - roll_mean) / roll_std, 0.0)

    # Discrete signal
    signals = np.zeros(n, dtype=np.float64)
    signals[z_score > z_score_threshold] = 1.0
    signals[z_score < -z_score_threshold] = -1.0

    # Pre-fill early bars with 0 (insufficient history)
    signals[:min_periods] = 0.0

    return signals


def get_position_size(
    signal: float,
    capital: float,
    risk_factor: float,
    atr: float,
    multiplier: float = 1.0,
    max_leverage: float = 3.0,
    min_size: float = 0.0,
    eps: float = 1e-12,
) -> float:
    """
    Volatility-Adjusted Position Sizing.

    Position = (Capital × RiskFactor) / (ATR × Multiplier)

    Rationale:
      - When volatility (ATR) is high → position shrinks (risk controlled).
      - When volatility is low → position expands (up to max_leverage cap).
      - RiskFactor determines how much capital is exposed per unit of ATR.

    Args:
        signal:     Directional signal (+1, -1, or 0 for no trade).
        capital:    Notional account capital (e.g., 100_000).
        risk_factor:Fraction of capital risked per ATR unit (e.g., 0.01 = 1%).
        atr:        Current 14-period Average True Range (in price units).
        multiplier: Contract multiplier (e.g., $50/point for ES, $1,000/point for ZB).
        max_leverage:Maximum absolute position size cap.
        min_size:   Minimum position size (positions below this are floored to 0).
        eps:        Small constant to prevent division by zero.

    Returns:
        Position size in contracts (float).  Zero when signal is 0 or ATR invalid.

    Example:
        # ES: capital=$100k, risk=1%, ATR=50pts, multiplier=$50
        # Position = (100000 * 0.01) / (50 * 50) = 1000 / 2500 = 0.4 contracts
        >>> get_position_size(1.0, 100_000, 0.01, 50.0, 50.0)
        0.4
    """
    if signal == 0.0 or atr <= eps:
        return 0.0

    risk_capital = capital * risk_factor
    atr_dollar = atr * multiplier

    if atr_dollar <= eps:
        return 0.0

    raw_size = risk_capital / atr_dollar
    signed_size = np.sign(signal) * raw_size

    # Apply leverage cap
    signed_size = np.clip(signed_size, -max_leverage, max_leverage)

    # Floor tiny positions to zero (avoids micro-positions that only generate costs)
    if abs(signed_size) < min_size:
        signed_size = 0.0

    return float(signed_size)


def compute_commission_cost(
    position_change: float,
    commission_per_contract: float = COMMISSION_PER_CONTRACT,
) -> float:
    """
    Compute per-bar commission cost from position changes.

    Each contract traded costs commission_per_contract (default $1.50).
    A round-trip (enter + exit) costs 2 × commission_per_contract.

    Args:
        position_change: Absolute change in position |pos_t - pos_{t-1}|.
        commission_per_contract: Cost per contract per side.

    Returns:
        Dollar cost to subtract from PnL.
    """
    return float(abs(position_change) * commission_per_contract)


def conviction_sweep(
    prediction_prob: np.ndarray,
    thresholds: np.ndarray,
    capital: float = 100_000.0,
    risk_factor: float = 0.01,
    atr_series: Optional[np.ndarray] = None,
    multiplier: float = 50.0,
    ret_exec: Optional[np.ndarray] = None,
    commission: float = COMMISSION_PER_CONTRACT,
    max_leverage: float = 3.0,
) -> dict:
    """
    Sweep across z-score thresholds and measure PnL + turnover impact.

    This is the core diagnostic for the "Conviction Sweep" — it shows how
    increasing the signal significance threshold (being more selective)
    affects total PnL, turnover, and win rate.

    Fully vectorised for speed over 70k+ bar datasets.

    Args:
        prediction_prob: 1-D array of prediction probabilities.
        thresholds:       Array of z-score thresholds to test.
        capital:          Notional capital for position sizing.
        risk_factor:      Fraction of capital risked per trade.
        atr_series:       ATR values aligned with prediction_prob.
        multiplier:       Contract multiplier for dollar ATR conversion.
        ret_exec:         Realized forward returns for PnL computation.
        commission:       Per-contract per-side commission.

    Returns:
        dict with keys: thresholds, total_pnl, turnover, num_trades, signal_fraction.
    """
    prob = np.asarray(prediction_prob, dtype=np.float64)
    n = len(prob)

    if atr_series is None:
        atr_series = np.ones(n, dtype=np.float64)
    atr = np.asarray(atr_series, dtype=np.float64)

    if ret_exec is None:
        ret_exec_arr = np.zeros(n, dtype=np.float64)
    else:
        ret_exec_arr = np.asarray(ret_exec, dtype=np.float64)

    # Pre-compute rolling z-score once (independent of threshold)
    window, min_periods = 1000, 50
    cumsum = np.cumsum(np.insert(prob, 0, 0.0))
    cumsum2 = np.cumsum(np.insert(prob ** 2, 0, 0.0))
    roll_mean = np.full(n, np.nan, dtype=np.float64)
    roll_std = np.full(n, np.nan, dtype=np.float64)
    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        count = i - start + 1
        roll_mean[i] = (cumsum[i + 1] - cumsum[start]) / count
        variance = (cumsum2[i + 1] - cumsum2[start]) / count - roll_mean[i] ** 2
        roll_std[i] = np.sqrt(max(variance, 1e-12))
    z_scores = np.where(roll_std > 0, (prob - roll_mean) / roll_std, 0.0)
    z_scores[:min_periods] = 0.0

    # Pre-compute raw position size (before sign): risk_capital / (ATR * multiplier)
    # Vectorised for all bars simultaneously
    risk_capital = capital * risk_factor
    atr_dollar = np.clip(atr * multiplier, 1e-12, None)
    raw_size = risk_capital / atr_dollar
    raw_size = np.clip(raw_size, 0.0, max_leverage)

    # Pre-compute ret_exec shifted (position at t-1 * return at t)
    ret_shifted = np.roll(ret_exec_arr, 1)
    ret_shifted[0] = 0.0

    results = {
        'thresholds': [],
        'total_pnl': [],
        'turnover': [],
        'num_trades': [],
        'signal_fraction': [],
    }

    for threshold in thresholds:
        t = float(threshold)

        # Vectorised signal generation
        signals = np.zeros(n, dtype=np.float64)
        signals[z_scores > t] = 1.0
        signals[z_scores < -t] = -1.0

        # Vectorised position sizing: size = sign * raw_size
        positions = np.sign(signals) * raw_size

        # PnL: pos_{t-1} * ret_exec_t
        pos_shifted = np.roll(positions, 1)
        pos_shifted[0] = 0.0
        pnl_bars = pos_shifted * ret_exec_arr

        # Commission: |Δpos| * commission
        pos_changes = np.abs(np.diff(positions, prepend=0.0))
        pnl_bars -= pos_changes * commission

        total_pnl = float(np.sum(pnl_bars))
        turnover_val = float(np.sum(pos_changes) / max(n, 1))
        num_trades_val = int(np.sum(np.abs(signals) > 0))
        signal_fraction_val = float(num_trades_val / max(n, 1))

        results['thresholds'].append(t)
        results['total_pnl'].append(total_pnl)
        results['turnover'].append(turnover_val)
        results['num_trades'].append(num_trades_val)
        results['signal_fraction'].append(signal_fraction_val)

    return results
