import json
import polars as pl
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

# Number of 5-minute bars in a trading year (~23h * 12 bars/h * 252 trading days)
# Overridable via config.ANNUAL_FACTOR if set before aggregate import.
ANNUAL_FACTOR = 69552
RISK_FREE_RATE = 0.0

# Column projection for loading backtest results: only read columns that
# are actually used downstream.  Eliminates SELECT * on parquet reads.
_BACKTEST_COLUMNS_OF_INTEREST = [
    'ts_event', 'pnl', 'position', 'benchmark_pnl',
    'prediction_prob', 'ret_exec',
]


def compute_ic(predictions: pl.Series, targets: pl.Series) -> dict:
    """Compute Information Coefficient using Spearman rank correlation.

    Args:
        predictions: Model prediction probabilities (or any continuous signal).
        targets: Realized forward returns or target labels.

    Returns:
        dict with 'spearman_ic' (float) rounded to 4 decimals, or None if
        the computation fails (e.g., constant arrays).
    """
    try:
        pred = predictions.to_numpy().astype(np.float64)
        targ = targets.to_numpy().astype(np.float64)
        mask = np.isfinite(pred) & np.isfinite(targ)
        if mask.sum() < 3:
            return {'spearman_ic': None}
        corr, pvalue = spearmanr(pred[mask], targ[mask])
        return {
            'spearman_ic': round(float(corr), 4),
            'spearman_ic_pvalue': round(float(pvalue), 6),
        }
    except Exception:
        return {'spearman_ic': None}


def compute_pro_metrics(
    pnl_series: pl.Series,
    positions_series: pl.Series = None,
    benchmark_series: pl.Series = None,
    predictions_series: pl.Series = None,
    targets_series: pl.Series = None,
) -> dict:
    pnl = pnl_series.to_numpy().astype(np.float32)
    eps = 1e-12

    # --- Core PnL metrics ---
    total_pnl = float(pnl.sum())

    # Proper return-on-capital: use the instrument's typical notional value.
    # PnL is measured in log-return space (unitless). The sum of log returns
    # approximates cumulative log return. We normalize by number of observations
    # to get average log return, then annualize.
    avg_pnl = float(pnl.mean())
    std_pnl = float(pnl.std())

    # Sharpe: annualized mean return / annualized volatility
    # mean(pnl) * ANNUAL_FACTOR / (std(pnl) * sqrt(ANNUAL_FACTOR)) = mean/std * sqrt(ANNUAL)
    sharpe = (avg_pnl / (std_pnl + eps)) * np.sqrt(ANNUAL_FACTOR)

    # Total return as cumulative log return (sum of log returns = log(total_return))
    total_return_pct = float(total_pnl * 100.0)  # Convert to percentage

    # --- Sortino ratio ---
    downside = pnl[pnl < 0]
    if len(downside) > 0:
        downside_std = float(downside.std())
        sortino = (avg_pnl / (downside_std + eps)) * np.sqrt(ANNUAL_FACTOR)
    else:
        sortino = np.inf if avg_pnl > 0 else 0.0

    # --- Drawdown ---
    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_drawdown = float(drawdown.min())

    # --- Calmar ratio ---
    annualized_return = avg_pnl * ANNUAL_FACTOR
    calmar = annualized_return / (abs(max_drawdown) + eps)

    # --- Trade statistics ---
    trades = 0
    win_rate = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    profit_factor = 0.0
    avg_holding_bars = 0.0
    turnover = 0.0

    if positions_series is not None and len(positions_series) > 0:
        positions = positions_series.to_numpy().astype(np.float32)
        # Turnover: sum of absolute position changes divided by total bars
        # This measures how many times the position turns over relative to the sample
        pos_changes = np.abs(np.diff(positions, prepend=0))
        turnover = float(pos_changes.sum() / max(len(positions), 1))

        # Count trades: each time position changes
        entry_idx = np.where(pos_changes > 0)[0]
        trades = len(entry_idx)

        # Average holding bars between entry events
        if trades > 1:
            holding = np.diff(entry_idx)
            avg_holding_bars = float(holding.mean())

        # Trade-level PnL attribution
        trade_pnl = []
        current_pos = 0
        entry_bar = 0
        cum_pnl_local = np.cumsum(pnl)
        for i in range(len(pnl)):
            if positions[i] != current_pos:
                if current_pos != 0:
                    trade_pnl.append(cum_pnl_local[i] - cum_pnl_local[entry_bar])
                current_pos = positions[i]
                entry_bar = i
        if current_pos != 0:
            trade_pnl.append(cum_pnl_local[-1] - cum_pnl_local[entry_bar])

        trade_pnl = np.array(trade_pnl, dtype=np.float32)
        if len(trade_pnl) > 0:
            gains = trade_pnl[trade_pnl > 0]
            losses = trade_pnl[trade_pnl < 0]
            win_rate = float(len(gains) / max(len(trade_pnl), 1))
            avg_win = float(gains.mean()) if len(gains) > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
            if len(losses) > 0 and abs(losses.sum()) > eps:
                profit_factor = float(gains.sum() / abs(losses.sum()))
            else:
                profit_factor = np.inf if gains.sum() > 0 else 0.0

    # --- Benchmark metrics ---
    benchmark_sharpe = None
    benchmark_maxdd = None
    correlation = None
    if benchmark_series is not None:
        bench = benchmark_series.to_numpy().astype(np.float32)
        bench_avg = float(bench.mean())
        bench_std = float(bench.std())
        if bench_std > eps:
            benchmark_sharpe = (bench_avg / bench_std) * np.sqrt(ANNUAL_FACTOR)
        else:
            benchmark_sharpe = 0.0

        bench_cum = np.cumsum(bench)
        bench_max = np.maximum.accumulate(bench_cum)
        benchmark_maxdd = float((bench_cum - bench_max).min())

        if len(pnl) == len(bench) and np.isfinite(pnl).all() and np.isfinite(bench).all():
            try:
                corr_matrix = np.corrcoef(pnl, bench)
                correlation = float(corr_matrix[0, 1])
            except Exception:
                correlation = 0.0

    # --- Information Coefficient ---
    ic_result = {}
    if predictions_series is not None and targets_series is not None:
        ic_result = compute_ic(predictions_series, targets_series)

    return {
        'total_pnl': round(total_pnl, 6),
        'total_return_percent': round(total_return_pct, 4),
        'sharpe_annualized': round(sharpe, 3) if np.isfinite(sharpe) else 0.0,
        'sortino_annualized': round(sortino, 3) if np.isfinite(sortino) else 0.0,
        'calmar_ratio': round(calmar, 3),
        'max_drawdown': round(max_drawdown, 6),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 4) if np.isfinite(profit_factor) else 'inf',
        'avg_win': round(avg_win, 8),
        'avg_loss': round(avg_loss, 8),
        'ratio_avg_win_loss': round(avg_win / (abs(avg_loss) + eps), 3),
        'number_of_trades': int(trades),
        'avg_holding_bars': round(avg_holding_bars, 1),
        'turnover': round(turnover, 4),
        'spearman_ic': ic_result.get('spearman_ic'),
        'spearman_ic_pvalue': ic_result.get('spearman_ic_pvalue'),
        'benchmark_sharpe': round(benchmark_sharpe, 3) if benchmark_sharpe is not None else None,
        'benchmark_max_drawdown': round(benchmark_maxdd, 6) if benchmark_maxdd is not None else None,
        'correlation_with_benchmark': round(correlation, 4) if correlation is not None else None,
    }


def load_all_backtests(artifacts_root='output') -> dict:
    """Load all backtest parquet files with explicit column projection.

    Uses the ``columns`` parameter of ``pl.read_parquet`` to read only the
    six columns of interest rather than SELECT *, and filters to files that
    actually contain the required columns.
    """
    root = Path(artifacts_root)
    results = {}

    # Search for backtest_results.parquet nested two levels deep:
    #   <root>/<market>/<year>/backtest_results.parquet
    for f in root.glob('*/*/backtest_results.parquet'):
        try:
            # Projection: read only columns we'll actually use downstream.
            # Polars will read the parquet footer to discover which of the
            # requested columns exist; missing columns are silently ignored
            # when ``columns`` is a list (pre-0.20 behaviour), so we verify
            # afterwards instead.
            df = pl.read_parquet(f, columns=_BACKTEST_COLUMNS_OF_INTEREST)

            if 'pnl' not in df.columns or 'ts_event' not in df.columns:
                continue

            keep = ['ts_event']
            for col in ['pnl', 'position', 'benchmark_pnl',
                        'prediction_prob', 'ret_exec']:
                if col in df.columns:
                    keep.append(col)
            df = df.select(keep).sort('ts_event')
            market = f.parent.parent.name
            year = f.parent.name
            results.setdefault(market, []).append((year, df))
        except Exception:
            continue

    return results


def aggregate_market(dfs: list) -> pl.DataFrame:
    return pl.concat([df for _, df in dfs]).sort('ts_event')


def compute_year_breakdown(year_dfs: list) -> list:
    breakdown = []
    for year, df in year_dfs:
        predictions = df['prediction_prob'] if 'prediction_prob' in df.columns else None
        targets = df['ret_exec'] if 'ret_exec' in df.columns else None
        metrics = compute_pro_metrics(
            df['pnl'],
            df['position'] if 'position' in df.columns else None,
            df['benchmark_pnl'] if 'benchmark_pnl' in df.columns else None,
            predictions_series=predictions,
            targets_series=targets,
        )
        metrics['year'] = year
        breakdown.append(metrics)
    return breakdown


def _build_combined_pnl(results: dict) -> pl.Series:
    """
    Build a combined-market total-PnL series by concatenating all market
    PnL series and summing per-timestamp using group_by aggregation.

    This avoids N+1 join chains and the suffix/collision issues inherent
    in outer joins over 3+ frames.

    Returns a pl.Series of combined pnl values, or None if fewer than 2
    market series are available.
    """
    pnl_frames = []
    for market, year_dfs in results.items():
        market_df = aggregate_market(year_dfs).select(['ts_event', 'pnl'])
        pnl_frames.append(market_df)

    if len(pnl_frames) < 2:
        return None

    # Stack all market series and sum PnL per timestamp
    stacked = pl.concat(pnl_frames)
    combined = stacked.group_by('ts_event', maintain_order=True).agg(
        pl.col('pnl').sum().alias('total_pnl')
    )
    return combined['total_pnl']


def run_aggregation(artifacts_root='output'):
    root = Path(artifacts_root)
    output_dir = root / 'aggregated'
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_all_backtests(artifacts_root)
    if not results:
        print('No backtest results found.')
        return

    all_series = []

    for market, year_dfs in results.items():
        combined = aggregate_market(year_dfs)
        if combined.is_empty():
            continue
        predictions = combined['prediction_prob'] if 'prediction_prob' in combined.columns else None
        targets = combined['ret_exec'] if 'ret_exec' in combined.columns else None
        metrics = compute_pro_metrics(
            combined['pnl'],
            combined['position'] if 'position' in combined.columns else None,
            combined['benchmark_pnl'] if 'benchmark_pnl' in combined.columns else None,
            predictions_series=predictions,
            targets_series=targets,
        )
        metrics.update({
            'market': market,
            'num_years': len(year_dfs),
            'total_rows': combined.height,
            'years_breakdown': compute_year_breakdown(year_dfs),
        })
        out = output_dir / f'{market}_metrics.json'
        with open(out, 'w') as f:
            json.dump(metrics, f, indent=2)
        ic_str = (
            f"IC={metrics['spearman_ic']}"
            if metrics.get('spearman_ic') is not None
            else "IC=N/A"
        )
        print(
            f"Saved {out} | Sharpe={metrics['sharpe_annualized']} | "
            f"HitRate={metrics['win_rate']} | {ic_str} | "
            f"PnL={metrics['total_pnl']}"
        )
        all_series.append(combined['pnl'])

    # Combined multi-market report — single outer-join chain (eliminates N+1)
    if len(results) > 1:
        total_pnl = _build_combined_pnl(results)
        if total_pnl is not None:
            total_metrics = compute_pro_metrics(total_pnl)
            total_metrics.update({
                'description': 'Sum of all markets',
                'markets': sorted(results.keys()),
            })
            out = output_dir / 'all_markets.json'
            with open(out, 'w') as f:
                json.dump(total_metrics, f, indent=2)
            print(
                f"Saved combined report | Sharpe={total_metrics['sharpe_annualized']}"
            )