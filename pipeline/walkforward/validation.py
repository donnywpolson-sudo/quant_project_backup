"""
validation.py — HMM Regime Validation & Comparison Metrics.

Provides:
  - Probabilistic Sharpe Ratio (PSR) for statistical significance testing.
  - Strategy comparison: Base vs HMM-filtered.
  - ValidationReport dataclass for structured output.
  - Regime-specific performance attribution.

All metrics are computed from PnL series — no look-ahead, no data leakage.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Annualization factor for 5-minute bars
# ~23 hours * 12 bars/hour * 252 days = 69,552 bars/year
ANNUAL_FACTOR: float = 69552.0
RISK_FREE_RATE: float = 0.0
EPS: float = 1e-12


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class StrategyMetrics:
    """Performance metrics for a single strategy variant."""

    label: str
    annualized_sharpe: float
    annualized_sortino: float
    max_drawdown: float
    calmar_ratio: float
    total_return_pct: float
    win_rate: float
    profit_factor: float
    number_of_trades: int
    avg_trade_pnl: float
    turnover: float
    volatility_annualized: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "annualized_sharpe": round(self.annualized_sharpe, 4),
            "annualized_sortino": round(self.annualized_sortino, 4),
            "max_drawdown": round(self.max_drawdown, 6),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "total_return_percent": round(self.total_return_pct, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "number_of_trades": self.number_of_trades,
            "avg_trade_pnl": round(self.avg_trade_pnl, 8),
            "turnover": round(self.turnover, 4),
            "volatility_annualized": round(self.volatility_annualized, 6),
        }


@dataclass
class PSRResult:
    """Probabilistic Sharpe Ratio test result."""

    psr: float  # P(SR > SR_benchmark)
    base_sharpe: float
    filtered_sharpe: float
    sharpe_difference: float
    confidence_level: float  # e.g. 0.95
    significant: bool  # True if PSR >= confidence_level
    p_value: float  # 1 - PSR, approximate

    def to_dict(self) -> dict:
        return {
            "psr": round(self.psr, 4),
            "base_sharpe": round(self.base_sharpe, 4),
            "filtered_sharpe": round(self.filtered_sharpe, 4),
            "sharpe_difference": round(self.sharpe_difference, 4),
            "confidence_level": self.confidence_level,
            "significant": self.significant,
            "p_value": round(self.p_value, 6),
        }


@dataclass
class ValidationReport:
    """Complete validation report comparing base vs HMM-filtered strategies."""

    base_metrics: StrategyMetrics
    hmm_metrics: StrategyMetrics
    psr_result: PSRResult
    regime_attribution: Dict[str, dict] = field(default_factory=dict)
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "base_metrics": self.base_metrics.to_dict(),
            "hmm_metrics": self.hmm_metrics.to_dict(),
            "psr_result": self.psr_result.to_dict(),
            "regime_attribution": self.regime_attribution,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason,
            "recommendation": self.recommendation,
        }

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "=" * 60,
            "HMM REGIME FILTER VALIDATION REPORT",
            "=" * 60,
            "",
            f"{'Metric':<25} {'Base':>12} {'HMM-Filtered':>15} {'Δ':>10}",
            "-" * 62,
            f"{'Annualized Sharpe':<25} {self.base_metrics.annualized_sharpe:>12.4f} "
            f"{self.hmm_metrics.annualized_sharpe:>15.4f} "
            f"{self.psr_result.sharpe_difference:>10.4f}",
            f"{'Max Drawdown':<25} {self.base_metrics.max_drawdown:>12.6f} "
            f"{self.hmm_metrics.max_drawdown:>15.6f}",
            f"{'Calmar Ratio':<25} {self.base_metrics.calmar_ratio:>12.4f} "
            f"{self.hmm_metrics.calmar_ratio:>15.4f}",
            f"{'Win Rate':<25} {self.base_metrics.win_rate:>12.4f} "
            f"{self.hmm_metrics.win_rate:>15.4f}",
            f"{'Profit Factor':<25} {self.base_metrics.profit_factor:>12.4f} "
            f"{self.hmm_metrics.profit_factor:>15.4f}",
            f"{'# Trades':<25} {self.base_metrics.number_of_trades:>12d} "
            f"{self.hmm_metrics.number_of_trades:>15d}",
            f"{'Turnover':<25} {self.base_metrics.turnover:>12.4f} "
            f"{self.hmm_metrics.turnover:>15.4f}",
            "-" * 62,
            "",
            f"PSR (Probabilistic Sharpe Ratio): {self.psr_result.psr:.4f}",
            f"  Confidence Level: {self.psr_result.confidence_level:.0%}",
            f"  Significant Improvement: {self.psr_result.significant}",
            f"  p-value (approx): {self.psr_result.p_value:.6f}",
            "",
            f"Fallback Triggered: {self.fallback_triggered}",
        ]
        if self.fallback_reason:
            lines.append(f"Fallback Reason: {self.fallback_reason}")
        lines.append("")
        lines.append(f"Recommendation: {self.recommendation}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# Core Metric Computation
# ============================================================================


def _compute_strategy_metrics(
    pnl_series: np.ndarray,
    label: str,
    positions: Optional[np.ndarray] = None,
) -> StrategyMetrics:
    """
    Compute standard strategy metrics from PnL array.

    Args:
        pnl_series: 1D numpy array of per-bar PnL values.
        label: Strategy name/label.
        positions: Optional position array for trade counting and turnover.

    Returns:
        StrategyMetrics dataclass.
    """
    pnl = np.asarray(pnl_series, dtype=np.float64)
    pnl = np.nan_to_num(pnl, nan=0.0, posinf=0.0, neginf=0.0)

    n = len(pnl)
    if n < 2:
        return StrategyMetrics(
            label=label,
            annualized_sharpe=0.0,
            annualized_sortino=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            total_return_pct=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            number_of_trades=0,
            avg_trade_pnl=0.0,
            turnover=0.0,
            volatility_annualized=0.0,
        )

    total_pnl = float(pnl.sum())
    avg_pnl = float(pnl.mean())
    std_pnl = float(pnl.std())

    # Annualized Sharpe
    sharpe = (avg_pnl / (std_pnl + EPS)) * np.sqrt(ANNUAL_FACTOR)

    # Sortino
    downside = pnl[pnl < 0]
    if len(downside) > 0:
        downside_std = float(downside.std())
        sortino = (avg_pnl / (downside_std + EPS)) * np.sqrt(ANNUAL_FACTOR)
    else:
        sortino = np.inf if avg_pnl > 0 else 0.0

    # Max Drawdown
    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_dd = float(drawdown.min())

    # Calmar
    annualized_return = avg_pnl * ANNUAL_FACTOR
    calmar = annualized_return / (abs(max_dd) + EPS)

    # Total return
    total_return_pct = float(total_pnl * 100.0)

    # Annualized volatility
    vol_annualized = std_pnl * np.sqrt(ANNUAL_FACTOR)

    # Trade statistics from positions
    trades = 0
    win_rate = 0.0
    profit_factor = 0.0
    avg_trade_pnl = 0.0
    turnover = 0.0

    if positions is not None and len(positions) > 0:
        pos = np.asarray(positions, dtype=np.float64)
        pos_changes = np.abs(np.diff(pos, prepend=0.0))
        turnover = float(pos_changes.sum() / max(len(pos), 1))

        # Trade-level PnL
        trade_pnl_list = []
        current_pos = 0.0
        entry_bar = 0
        cum_pnl_local = np.cumsum(pnl)

        for i in range(len(pnl)):
            if pos[i] != current_pos:
                if current_pos != 0.0:
                    trade_pnl_list.append(cum_pnl_local[i] - cum_pnl_local[entry_bar])
                current_pos = pos[i]
                entry_bar = i
        if current_pos != 0.0:
            trade_pnl_list.append(cum_pnl_local[-1] - cum_pnl_local[entry_bar])

        if trade_pnl_list:
            trade_pnl_arr = np.array(trade_pnl_list, dtype=np.float64)
            trades = len(trade_pnl_arr)
            gains = trade_pnl_arr[trade_pnl_arr > 0]
            losses = trade_pnl_arr[trade_pnl_arr < 0]
            win_rate = float(len(gains) / max(trades, 1))
            avg_trade_pnl = float(trade_pnl_arr.mean())

            if len(losses) > 0 and abs(losses.sum()) > EPS:
                profit_factor = float(gains.sum() / abs(losses.sum()))
            else:
                profit_factor = np.inf if gains.sum() > 0 else 0.0

    return StrategyMetrics(
        label=label,
        annualized_sharpe=sharpe if np.isfinite(sharpe) else 0.0,
        annualized_sortino=sortino if np.isfinite(sortino) else 0.0,
        max_drawdown=max_dd,
        calmar_ratio=calmar if np.isfinite(calmar) else 0.0,
        total_return_pct=total_return_pct,
        win_rate=win_rate,
        profit_factor=profit_factor if np.isfinite(profit_factor) else 0.0,
        number_of_trades=int(trades),
        avg_trade_pnl=avg_trade_pnl,
        turnover=turnover,
        volatility_annualized=vol_annualized if np.isfinite(vol_annualized) else 0.0,
    )


# ============================================================================
# Probabilistic Sharpe Ratio (PSR)
# ============================================================================


def probabilistic_sharpe_ratio(
    pnl_base: np.ndarray,
    pnl_hmm: np.ndarray,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
) -> PSRResult:
    """
    Compute the Probabilistic Sharpe Ratio (PSR) to test whether the
    HMM-filtered strategy's Sharpe ratio is statistically significantly
    better than the base strategy's Sharpe ratio.

    PSR formulation (Bailey & López de Prado, 2012):
        PSR = Φ( (SR_hat - SR_benchmark) * sqrt(T - 1) / sqrt(1 - skew * SR_hat + (kurt - 1)/4 * SR_hat^2) )

    where:
        - SR_hat = estimated annualized Sharpe ratio of HMM-filtered strategy
        - SR_benchmark = base strategy's Sharpe ratio
        - T = number of observations
        - skew = skewness of HMM-filtered returns
        - kurt = excess kurtosis of HMM-filtered returns
        - Φ = standard normal CDF

    Args:
        pnl_base: Per-bar PnL of the base (unfiltered) strategy.
        pnl_hmm: Per-bar PnL of the HMM-filtered strategy.
        benchmark_sharpe: Sharpe ratio to test against (default: base Sharpe).
        confidence: Confidence level for significance (default: 0.95).

    Returns:
        PSRResult with PSR value and significance determination.
    """
    hmm = np.asarray(pnl_hmm, dtype=np.float64)
    hmm = np.nan_to_num(hmm, nan=0.0, posinf=0.0, neginf=0.0)

    base = np.asarray(pnl_base, dtype=np.float64)
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)

    n_hmm = len(hmm)
    n_base = len(base)

    if n_hmm < 30 or n_base < 30:
        logger.warning(
            f"Insufficient observations for PSR: base={n_base}, hmm={n_hmm}. "
            f"Need >= 30."
        )
        return PSRResult(
            psr=0.0,
            base_sharpe=0.0,
            filtered_sharpe=0.0,
            sharpe_difference=0.0,
            confidence_level=confidence,
            significant=False,
            p_value=1.0,
        )

    # Compute base Sharpe
    base_avg = float(base.mean())
    base_std = float(base.std())
    base_sharpe = (base_avg / (base_std + EPS)) * np.sqrt(ANNUAL_FACTOR)

    # Compute HMM-filtered Sharpe
    hmm_avg = float(hmm.mean())
    hmm_std = float(hmm.std())
    hmm_sharpe = (hmm_avg / (hmm_std + EPS)) * np.sqrt(ANNUAL_FACTOR)

    # If benchmark not provided, use base Sharpe
    if benchmark_sharpe == 0.0:
        benchmark_sharpe = base_sharpe

    sharpe_diff = hmm_sharpe - benchmark_sharpe

    if sharpe_diff <= 0:
        # HMM didn't improve Sharpe — PSR is effectively 0
        return PSRResult(
            psr=0.0,
            base_sharpe=float(base_sharpe),
            filtered_sharpe=float(hmm_sharpe),
            sharpe_difference=float(sharpe_diff),
            confidence_level=confidence,
            significant=False,
            p_value=1.0,
        )

    # Compute higher moments for PSR
    # We need per-bar returns (not PnL) — but PnL IS the per-bar return in log-space
    # Compute skewness and kurtosis on the HMM-filtered returns
    hmm_centered = hmm - hmm_avg

    # Skewness
    skew = np.mean(hmm_centered ** 3) / (hmm_std ** 3 + EPS)

    # Excess kurtosis (kurtosis - 3)
    kurt = np.mean(hmm_centered ** 4) / (hmm_std ** 4 + EPS) - 3.0

    # PSR formula
    sr_hat = hmm_sharpe
    sr_bench = benchmark_sharpe
    T = n_hmm

    # Denominator: sqrt of asymptotic variance of Sharpe ratio estimator
    variance_adjustment = 1.0 - skew * sr_hat + (kurt / 4.0) * (sr_hat ** 2)
    if variance_adjustment <= 0:
        logger.warning(
            f"PSR variance adjustment is non-positive ({variance_adjustment:.6f}). "
            f"skew={skew:.4f}, kurt={kurt:.4f}, SR={sr_hat:.4f}. "
            f"Clipping to small positive value."
        )
        variance_adjustment = 1e-6

    denominator = np.sqrt(variance_adjustment / (T - 1))
    if denominator < EPS:
        denominator = EPS

    z_score = (sr_hat - sr_bench) / denominator

    # PSR = Φ(z_score) = probability that true Sharpe > benchmark
    psr = float(norm.cdf(z_score))

    significant = psr >= confidence
    p_value = 1.0 - psr

    logger.info(
        f"PSR: {psr:.4f} (significant={significant} at {confidence:.0%} CI) | "
        f"SR_base={base_sharpe:.4f} | SR_hmm={hmm_sharpe:.4f} | "
        f"ΔSR={sharpe_diff:.4f} | z={z_score:.4f} | "
        f"skew={skew:.4f} | kurt={kurt:.4f}"
    )

    return PSRResult(
        psr=float(psr),
        base_sharpe=float(base_sharpe),
        filtered_sharpe=float(hmm_sharpe),
        sharpe_difference=float(sharpe_diff),
        confidence_level=confidence,
        significant=significant,
        p_value=float(p_value),
    )


# ============================================================================
# Regime Attribution
# ============================================================================


def compute_regime_attribution(
    df_5min: "pl.DataFrame",
    pnl_column: str = "pnl",
    regime_prob_columns: Optional[list] = None,
) -> Dict[str, dict]:
    """
    Attribute PnL performance to individual HMM regimes.

    Each bar is assigned to the dominant regime (argmax over probabilities),
    and metrics are computed per regime.

    Args:
        df_5min: 5-min DataFrame with PnL and hmm_regime_* columns.
        pnl_column: Column name for PnL.
        regime_prob_columns: List of 'hmm_regime_*' column names.

    Returns:
        Dict mapping regime label to dict of metrics.
    """
    import polars as pl

    if regime_prob_columns is None:
        # Auto-detect hmm_regime_* columns
        regime_prob_columns = [
            c for c in df_5min.columns if c.startswith("hmm_regime_")
        ]
        regime_prob_columns = sorted(regime_prob_columns)

    if not regime_prob_columns:
        logger.warning("No hmm_regime_* columns for attribution.")
        return {}

    if pnl_column not in df_5min.columns:
        logger.warning(f"PnL column '{pnl_column}' not found.")
        return {}

    # Get dominant regime per bar
    probs = np.column_stack([
        df_5min[c].to_numpy() for c in regime_prob_columns
    ])
    dominant = np.argmax(probs, axis=1)
    pnl = df_5min[pnl_column].to_numpy().astype(np.float64)

    regime_labels = {
        0: "LowVol_Bullish",
        1: "LowVol_Range",
        2: "HighVol_Correction",
        3: "HighVol_Expansion",
    }

    attribution = {}
    n_regimes = len(regime_prob_columns)

    for r in range(n_regimes):
        mask = dominant == r
        regime_pnl = pnl[mask]
        n_bars = int(mask.sum())

        if n_bars < 5:
            attribution[regime_labels.get(r, f"regime_{r}")] = {
                "bars": n_bars,
                "pct_time": round(float(n_bars / max(len(pnl), 1)) * 100, 2),
                "total_pnl": 0.0,
                "avg_pnl_per_bar": 0.0,
                "sharpe_contribution": 0.0,
            }
            continue

        total = float(regime_pnl.sum())
        avg = float(regime_pnl.mean())
        std = float(regime_pnl.std())
        regime_sharpe = (avg / (std + EPS)) * np.sqrt(ANNUAL_FACTOR)

        attribution[regime_labels.get(r, f"regime_{r}")] = {
            "bars": n_bars,
            "pct_time": round(float(n_bars / max(len(pnl), 1)) * 100, 2),
            "total_pnl": round(total, 6),
            "avg_pnl_per_bar": round(avg, 8),
            "sharpe_contribution": round(regime_sharpe, 4) if np.isfinite(regime_sharpe) else 0.0,
        }

    return attribution


# ============================================================================
# Main Comparison Function
# ============================================================================


def compare_strategies(
    df_base: "pl.DataFrame",
    df_hmm: "pl.DataFrame",
    pnl_column: str = "pnl",
    position_column: str = "position",
    confidence: float = 0.95,
    hmm_active: bool = True,
    fallback_reason: Optional[str] = None,
) -> ValidationReport:
    """
    Compare base vs HMM-filtered strategy performance.

    Computes full metrics for both strategies, runs PSR significance test,
    attributes performance by regime, and generates a recommendation.

    Args:
        df_base: 5-min DataFrame with base strategy results (pnl, position).
        df_hmm: 5-min DataFrame with HMM-filtered results (pnl, position,
                hmm_regime_* columns).
        pnl_column: Column name for per-bar PnL.
        position_column: Column name for position (for trade stats).
        confidence: PSR confidence level (default 0.95).
        hmm_active: Whether HMM filter was active (False if fallback).
        fallback_reason: Reason for fallback, if triggered.

    Returns:
        ValidationReport dataclass.
    """
    import polars as pl

    # Extract PnL arrays
    base_pnl = df_base[pnl_column].to_numpy().astype(np.float64) if pnl_column in df_base.columns else np.array([])
    hmm_pnl = df_hmm[pnl_column].to_numpy().astype(np.float64) if pnl_column in df_hmm.columns else np.array([])

    # Extract positions
    base_pos = None
    hmm_pos = None
    if position_column in df_base.columns:
        base_pos = df_base[position_column].to_numpy().astype(np.float64)
    if position_column in df_hmm.columns:
        hmm_pos = df_hmm[position_column].to_numpy().astype(np.float64)

    # Compute metrics
    base_metrics = _compute_strategy_metrics(base_pnl, "Base", base_pos)
    hmm_metrics = _compute_strategy_metrics(hmm_pnl, "HMM-Filtered", hmm_pos)

    # PSR test
    psr_result = probabilistic_sharpe_ratio(
        pnl_base=base_pnl,
        pnl_hmm=hmm_pnl,
        benchmark_sharpe=base_metrics.annualized_sharpe,
        confidence=confidence,
    )

    # Regime attribution (only for HMM)
    regime_attribution = {}
    hmm_regime_cols = [c for c in df_hmm.columns if c.startswith("hmm_regime_")]
    if hmm_regime_cols:
        regime_attribution = compute_regime_attribution(
            df_hmm, pnl_column=pnl_column,
            regime_prob_columns=sorted(hmm_regime_cols),
        )

    # Generate recommendation
    recommendation = _generate_recommendation(
        psr_result=psr_result,
        hmm_active=hmm_active,
        fallback_reason=fallback_reason,
        sharpe_diff=psr_result.sharpe_difference,
        calmar_base=base_metrics.calmar_ratio,
        calmar_hmm=hmm_metrics.calmar_ratio,
        maxdd_base=base_metrics.max_drawdown,
        maxdd_hmm=hmm_metrics.max_drawdown,
    )

    report = ValidationReport(
        base_metrics=base_metrics,
        hmm_metrics=hmm_metrics,
        psr_result=psr_result,
        regime_attribution=regime_attribution,
        fallback_triggered=not hmm_active,
        fallback_reason=fallback_reason,
        recommendation=recommendation,
    )

    # Log summary
    logger.info(f"\n{report.summary()}")

    return report


def _generate_recommendation(
    psr_result: PSRResult,
    hmm_active: bool,
    fallback_reason: Optional[str],
    sharpe_diff: float,
    calmar_base: float,
    calmar_hmm: float,
    maxdd_base: float,
    maxdd_hmm: float,
) -> str:
    """Generate a human-readable recommendation based on validation results."""

    if not hmm_active:
        return (
            f"HMM filter is INACTIVE (fallback: {fallback_reason or 'unknown'}). "
            f"Results reflect pass-through execution. Consider: "
            f"(1) increasing min_train_bars, "
            f"(2) adding more feature transforms for stationarity, "
            f"(3) reducing retrain frequency."
        )

    if psr_result.significant:
        if sharpe_diff > 0.5:
            return (
                f"STRONG ACCEPT: HMM regime filter significantly improves risk-adjusted "
                f"performance (PSR={psr_result.psr:.4f}, ΔSR={sharpe_diff:+.4f}). "
                f"Calmar improved from {calmar_base:.4f} to {calmar_hmm:.4f}. "
                f"Deploy with regime-aware position sizing."
            )
        else:
            return (
                f"ACCEPT: HMM regime filter shows statistically significant improvement "
                f"(PSR={psr_result.psr:.4f}, ΔSR={sharpe_diff:+.4f}). "
                f"Consider tuning trade-gate thresholds for further optimization."
            )
    else:
        if sharpe_diff > 0:
            return (
                f"WEAK SIGNAL: HMM filter shows positive but non-significant improvement "
                f"(PSR={psr_result.psr:.4f} < {psr_result.confidence_level:.0%} CI, "
                f"ΔSR={sharpe_diff:+.4f}). Consider: "
                f"(1) expanding feature set (add higher moments, cross-asset features), "
                f"(2) increasing training data, "
                f"(3) tuning state mapping thresholds."
            )
        elif sharpe_diff < -0.1:
            return (
                f"REJECT: HMM regime filter DEGRADES performance "
                f"(ΔSR={sharpe_diff:+.4f}). The trade-gating logic is removing "
                f"profitable trades. Review regime mapping and allowed/prohibited "
                f"regime lists. Consider more granular state definitions."
            )
        else:
            return (
                f"NEUTRAL: HMM filter shows negligible impact on performance "
                f"(ΔSR={sharpe_diff:+.4f}). The regime detection may not add "
                f"value over the base strategy's existing HTF filters. "
                f"Consider removing redundant time-based gating."
            )