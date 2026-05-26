import json
import polars as pl
import numpy as np
from pathlib import Path
ANNUAL_FACTOR = 66528
RISK_FREE_RATE = 0.0

def compute_pro_metrics(pnl_series: pl.Series, positions_series: pl.Series=None, benchmark_series: pl.Series=None) -> dict:
    pnl = pnl_series.to_numpy().astype(np.float64)
    total_pnl = float(pnl.sum())
    total_return = total_pnl / (np.abs(pnl).mean() + 1e-09) * 100
    avg_pnl = float(pnl.mean())
    std_pnl = float(pnl.std())
    sharpe = avg_pnl / std_pnl * np.sqrt(ANNUAL_FACTOR) if std_pnl > 0 else 0.0
    downside = pnl[pnl < 0]
    if len(downside) > 0:
        downside_std = downside.std()
        sortino = avg_pnl / (downside_std + 1e-09) * np.sqrt(ANNUAL_FACTOR)
    else:
        sortino = np.inf if avg_pnl > 0 else 0.0
    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max
    max_drawdown = float(drawdown.min())
    annualized_return = avg_pnl * ANNUAL_FACTOR
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    trades = 0
    win_rate = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    profit_factor = 0.0
    avg_holding_bars = 0.0
    turnover = 0.0
    if positions_series is not None and len(positions_series) > 0:
        positions = positions_series.to_numpy()
        pos_changes = np.diff(positions, prepend=0)
        entry_idx = np.where(pos_changes != 0)[0]
        trades = len(entry_idx)
        if trades > 1:
            holding = np.diff(entry_idx)
            avg_holding_bars = float(holding.mean())
        trade_pnl = []
        current_pos = 0
        entry_bar = 0
        for i in range(len(pnl)):
            if positions[i] != current_pos:
                if current_pos != 0:
                    trade_pnl.append(cum_pnl[i] - cum_pnl[entry_bar])
                current_pos = positions[i]
                entry_bar = i
        if current_pos != 0:
            trade_pnl.append(cum_pnl[-1] - cum_pnl[entry_bar])
        trade_pnl = np.array(trade_pnl)
        if len(trade_pnl) > 0:
            gains = trade_pnl[trade_pnl > 0]
            losses = trade_pnl[trade_pnl < 0]
            win_rate = len(gains) / len(trade_pnl)
            avg_win = gains.mean() if len(gains) > 0 else 0.0
            avg_loss = losses.mean() if len(losses) > 0 else 0.0
            if losses.sum() != 0:
                profit_factor = gains.sum() / abs(losses.sum())
            else:
                profit_factor = np.inf
        turnover = float(np.abs(positions).sum() / (len(positions) + 1))
    benchmark_sharpe = None
    benchmark_maxdd = None
    correlation = None
    if benchmark_series is not None:
        bench = benchmark_series.to_numpy()
        bench_avg = bench.mean()
        bench_std = bench.std()
        benchmark_sharpe = bench_avg / bench_std * np.sqrt(ANNUAL_FACTOR) if bench_std > 0 else 0.0
        bench_cum = np.cumsum(bench)
        bench_max = np.maximum.accumulate(bench_cum)
        benchmark_maxdd = float((bench_cum - bench_max).min())
        if len(pnl) == len(bench):
            correlation = float(np.corrcoef(pnl, bench)[0, 1])
    return {'total_pnl': round(total_pnl, 4), 'total_return_percent': round(total_return, 2), 'sharpe_annualized': round(sharpe, 3), 'sortino_annualized': round(sortino if np.isfinite(sortino) else 0.0, 3), 'calmar_ratio': round(calmar, 3), 'max_drawdown': round(max_drawdown, 4), 'win_rate': round(win_rate, 4), 'profit_factor': round(profit_factor, 4) if np.isfinite(profit_factor) else 'inf', 'avg_win': round(avg_win, 6), 'avg_loss': round(avg_loss, 6), 'ratio_avg_win_loss': round(avg_win / abs(avg_loss), 3) if avg_loss != 0 else 0.0, 'number_of_trades': trades, 'avg_holding_bars': round(avg_holding_bars, 1), 'turnover': round(turnover, 4), 'benchmark_sharpe': round(benchmark_sharpe, 3) if benchmark_sharpe is not None else None, 'benchmark_max_drawdown': round(benchmark_maxdd, 4) if benchmark_maxdd is not None else None, 'correlation_with_benchmark': round(correlation, 4) if correlation is not None else None}

def load_all_backtests(artifacts_root='artifacts') -> dict:
    root = Path(artifacts_root)
    results = {}
    for f in root.glob('*/*/backtest_results.parquet'):
        try:
            market = f.parent.parent.name
            year = f.parent.name
            df = pl.read_parquet(f)
            if 'pnl' not in df.columns or 'ts_event' not in df.columns:
                continue
            keep = ['ts_event']
            for col in ['pnl', 'position', 'benchmark_pnl']:
                if col in df.columns:
                    keep.append(col)
            df = df.select(keep).sort('ts_event')
            results.setdefault(market, []).append((year, df))
        except Exception:
            continue
    return results

def aggregate_market(dfs: list) -> pl.DataFrame:
    return pl.concat([df for _, df in dfs]).sort('ts_event')

def compute_year_breakdown(year_dfs: list) -> list:
    breakdown = []
    for year, df in year_dfs:
        metrics = compute_pro_metrics(df['pnl'], df['position'] if 'position' in df.columns else None, df['benchmark_pnl'] if 'benchmark_pnl' in df.columns else None)
        metrics['year'] = year
        breakdown.append(metrics)
    return breakdown

def run_aggregation(artifacts_root='artifacts'):
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
        metrics = compute_pro_metrics(combined['pnl'], combined['position'] if 'position' in combined.columns else None, combined['benchmark_pnl'] if 'benchmark_pnl' in combined.columns else None)
        metrics.update({'market': market, 'num_years': len(year_dfs), 'total_rows': combined.height, 'years_breakdown': compute_year_breakdown(year_dfs)})
        out = output_dir / f'{market}_metrics.json'
        with open(out, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'Saved {out} | Sharpe={metrics['sharpe_annualized']} | PnL={metrics['total_pnl']}')
        all_series.append(combined['pnl'])
    if len(all_series) > 1:
        combined = None
        for market, year_dfs in results.items():
            df = aggregate_market(year_dfs).select(['ts_event', 'pnl'])
            if combined is None:
                combined = df
            else:
                combined = combined.join(df, on='ts_event', how='outer', suffix=f'_{market}')
        pnl_cols = [c for c in combined.columns if c.startswith('pnl')]
        total = combined.with_columns(sum((pl.col(c).fill_null(0) for c in pnl_cols)).alias('total_pnl'))['total_pnl']
        total_metrics = compute_pro_metrics(total)
        total_metrics.update({'description': 'Sum of all markets', 'markets': list(results.keys())})
        out = output_dir / 'all_markets.json'
        with open(out, 'w') as f:
            json.dump(total_metrics, f, indent=2)
        print(f'Saved combined report | Sharpe={total_metrics['sharpe_annualized']}')