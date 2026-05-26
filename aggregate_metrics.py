pass
import json
import polars as pl
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
ARTIFACTS_ROOT = Path('artifacts')
OUTPUT_DIR = ARTIFACTS_ROOT / 'aggregated'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNUAL_FACTOR = 66528
RISK_FREE_RATE = 0.0

def compute_pro_metrics(pnl_series: pl.Series, positions_series: pl.Series=None, benchmark_series: pl.Series=None) -> dict:
    pass
    pnl = pnl_series.to_numpy().astype(np.float64)
    total_pnl = float(pnl.sum())
    total_return = total_pnl / (np.abs(pnl).mean() + 1e-09) * 100
    avg_pnl = float(pnl.mean())
    std_pnl = float(pnl.std())
    if std_pnl > 0:
        sharpe = avg_pnl / std_pnl * np.sqrt(ANNUAL_FACTOR)
    else:
        sharpe = 0.0
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
            holding_bars = np.diff(entry_idx)
            avg_holding_bars = float(holding_bars.mean())
        cum_pnl_arr = cum_pnl
        trade_pnl = []
        current_pos = 0
        entry_bar = 0
        for i in range(len(pnl)):
            if positions[i] != current_pos:
                if current_pos != 0:
                    trade_pnl.append(cum_pnl_arr[i] - cum_pnl_arr[entry_bar])
                current_pos = positions[i]
                entry_bar = i
        if current_pos != 0:
            trade_pnl.append(cum_pnl_arr[-1] - cum_pnl_arr[entry_bar])
        trade_pnl = np.array(trade_pnl)
        if len(trade_pnl) > 0:
            gains = trade_pnl[trade_pnl > 0]
            losses = trade_pnl[trade_pnl < 0]
            win_rate = len(gains) / len(trade_pnl)
            avg_win = gains.mean() if len(gains) > 0 else 0.0
            avg_loss = losses.mean() if len(losses) > 0 else 0.0
            profit_factor = gains.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
        turnover = float(np.abs(positions).sum() / (len(positions) + 1))
    benchmark_sharpe = None
    benchmark_maxdd = None
    correlation_with_benchmark = None
    if benchmark_series is not None:
        bench = benchmark_series.to_numpy()
        bench_avg = bench.mean()
        bench_std = bench.std()
        if bench_std > 0:
            benchmark_sharpe = bench_avg / bench_std * np.sqrt(ANNUAL_FACTOR)
        else:
            benchmark_sharpe = 0.0
        bench_cum = np.cumsum(bench)
        bench_max = np.maximum.accumulate(bench_cum)
        benchmark_maxdd = float((bench_cum - bench_max).min())
        if len(pnl) == len(bench):
            correlation_with_benchmark = float(np.corrcoef(pnl, bench)[0, 1])
    return {'total_pnl': round(total_pnl, 4), 'total_return_percent': round(total_return, 2), 'sharpe_annualized': round(sharpe, 3), 'sortino_annualized': round(sortino if np.isfinite(sortino) else 0.0, 3), 'calmar_ratio': round(calmar, 3), 'max_drawdown': round(max_drawdown, 4), 'win_rate': round(win_rate, 4), 'profit_factor': round(profit_factor, 4) if np.isfinite(profit_factor) else 'inf', 'avg_win': round(avg_win, 6), 'avg_loss': round(avg_loss, 6), 'ratio_avg_win_loss': round(avg_win / abs(avg_loss), 3) if avg_loss != 0 else 0.0, 'number_of_trades': trades, 'avg_holding_bars': round(avg_holding_bars, 1), 'turnover': round(turnover, 4), 'benchmark_sharpe': round(benchmark_sharpe, 3) if benchmark_sharpe is not None else None, 'benchmark_max_drawdown': round(benchmark_maxdd, 4) if benchmark_maxdd is not None else None, 'correlation_with_benchmark': round(correlation_with_benchmark, 4) if correlation_with_benchmark is not None else None}

def load_all_backtests() -> dict:
    pass
    results_by_market = {}
    essential_cols = ['pnl', 'position', 'benchmark_pnl']
    for parquet_file in ARTIFACTS_ROOT.glob('*/*/backtest_results.parquet'):
        parts = parquet_file.parts
        if len(parts) >= 3:
            market = parts[1]
            year = parts[2]
            try:
                df = pl.read_parquet(parquet_file)
                if 'ts_event' not in df.columns:
                    continue
                cols_to_keep = ['ts_event']
                for col in essential_cols:
                    if col in df.columns:
                        cols_to_keep.append(col)
                cols_to_keep = list(dict.fromkeys(cols_to_keep))
                if 'pnl' not in cols_to_keep:
                    continue
                df = df.select(cols_to_keep).sort('ts_event')
                results_by_market.setdefault(market, []).append((year, df))
            except Exception:
                pass
    return results_by_market

def aggregate_market(market: str, dfs: list) -> pl.DataFrame:
    pass
    combined = pl.concat([df for _, df in dfs])
    return combined.sort('ts_event')

def compute_year_breakdown(market: str, year_dfs: list) -> list:
    pass
    breakdown = []
    for year, df in year_dfs:
        pnl = df['pnl']
        positions = df['position'] if 'position' in df.columns else None
        bench = df['benchmark_pnl'] if 'benchmark_pnl' in df.columns else None
        metrics = compute_pro_metrics(pnl, positions, bench)
        metrics['year'] = year
        breakdown.append(metrics)
    return breakdown

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    results = load_all_backtests()
    if not results:
        print('No backtest results found under artifacts/. Run the pipeline first.')
        sys.exit(0)
    all_pnl_series = []
    for market, year_dfs in results.items():
        combined_df = aggregate_market(market, year_dfs)
        if combined_df.is_empty():
            continue
        pnl_series = combined_df['pnl']
        positions_series = combined_df['position'] if 'position' in combined_df.columns else None
        bench_series = combined_df['benchmark_pnl'] if 'benchmark_pnl' in combined_df.columns else None
        market_metrics = compute_pro_metrics(pnl_series, positions_series, bench_series)
        market_metrics['market'] = market
        market_metrics['num_years'] = len(year_dfs)
        market_metrics['total_rows'] = combined_df.height
        market_metrics['years_breakdown'] = compute_year_breakdown(market, year_dfs)
        out_file = OUTPUT_DIR / f'{market}_metrics.json'
        save_json(market_metrics, out_file)
        print(f'Saved {out_file}')
        print(f'  Sharpe: {market_metrics['sharpe_annualized']} | Total PnL: {market_metrics['total_pnl']} | Trades: {market_metrics['number_of_trades']}')
        all_pnl_series.append(pnl_series)
    if len(all_pnl_series) > 1:
        combined_across = None
        for market, year_dfs in results.items():
            df = aggregate_market(market, year_dfs).select(['ts_event', 'pnl'])
            if combined_across is None:
                combined_across = df
            else:
                combined_across = combined_across.join(df, on='ts_event', how='outer', suffix=f'_{market}')
        pnl_cols = [c for c in combined_across.columns if c == 'pnl' or c.startswith('pnl_')]
        total_pnl_series = combined_across.with_columns(sum((pl.col(c).fill_null(0) for c in pnl_cols)).alias('total_pnl'))['total_pnl']
        total_metrics = compute_pro_metrics(total_pnl_series, positions_series=None, benchmark_series=None)
        total_metrics['description'] = 'Sum of PnL across all markets (equal capital allocation, no correlation adjustment)'
        total_metrics['markets_included'] = list(results.keys())
        out_all = OUTPUT_DIR / 'all_markets_combined.json'
        save_json(total_metrics, out_all)
        print(f'\nSaved across-markets report: {out_all}')
        print(f'  Combined Sharpe: {total_metrics['sharpe_annualized']} | Total PnL: {total_metrics['total_pnl']}')
    else:
        print('\nOnly one market found. No across-markets aggregation performed.')
if __name__ == '__main__':
    main()