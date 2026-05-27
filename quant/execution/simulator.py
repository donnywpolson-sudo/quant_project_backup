"""
simulator.py
Realistic execution simulator with economic friction.
- 1.5 bps fixed cost per round-turn (applied to position changes)
- Signal generated at bar t, executed at bar t+1 open
- Volatility computed from lagged returns [t-N, t-1]
- Position changes tracked for turnover calculation
"""
import polars as pl
from config import config


# Fixed transaction cost: 1.5 basis points (0.00015) per round-turn
TX_COST_PER_ROUNDTURN = 0.00015


def simulate_execution_classification(df: pl.DataFrame) -> pl.DataFrame:
    eps = config.EPS

    # 1. Generate trading signal from prediction probability
    #    (prediction_prob is based on features at time t, so signal is generated at t)
    signal_expr = (
        pl.when(pl.col('prediction_prob').fill_null(0.5) > 0.6)
        .then(1.0)
        .when(pl.col('prediction_prob').fill_null(0.5) < 0.4)
        .then(-1.0)
        .otherwise(0.0)
    )
    df = df.with_columns(signal_expr.alias('raw_signal'))

    # 2. Session break filter (flatten position during exchange break)
    df = df.with_columns(
        pl.col('ts_event').dt.convert_time_zone(config.TIMEZONE).dt.time().alias('t_local')
    )
    df = df.with_columns(
        pl.when(
            (pl.col('t_local') >= pl.lit(config.SESSION_BREAK_START_LOCAL))
            & (pl.col('t_local') < pl.lit(config.SESSION_BREAK_END_LOCAL))
        )
        .then(0.0)
        .otherwise(pl.col('raw_signal'))
        .alias('target_exec')
    )
    df = df.drop('t_local')

    # 3. Volatility: strictly lagged, computed from returns [t-N, t-1]
    ret = (pl.col('close') / pl.col('close').shift(1)).log()
    ret_lagged = ret.shift(1)
    vol = ret_lagged.rolling_std(window_size=20).clip(eps, None)
    df = df.with_columns(vol.fill_null(1e-06).alias('vol'))

    # 4. Spread proxy (bid-ask approximation)
    spread = (pl.col('high') - pl.col('low')) / pl.col('close').clip(eps, None)
    spread = spread.clip(0.0, 0.05)
    df = df.with_columns(spread.alias('spread'))

    # 5. Per-bar cost: 1.5 bps applied to position CHANGE (round-turn cost)
    #    First compute position changes, then apply cost per round-turn
    #    unit_cost is per-bar but only charged when position changes
    unit_cost = (
        config.COMMISSION_PER_TRADE
        + config.SLIPPAGE_K * pl.col('spread')
        + config.VOL_PENALTY * pl.col('vol')
    ).clip(0.0, 0.01)
    df = df.with_columns(unit_cost.alias('unit_cost'))

    # 6. Execution return: buy/sell at bar t+1 open, exit at bar t+1 close
    #    This is forward-looking but acceptable because:
    #    - target_exec was computed from features at time t
    #    - trade is assumed executed at next bar's open
    #    - trade is unwound at next bar's close
    open_next = pl.col('open').shift(-1)
    close_next = pl.col('close').shift(-1)
    ret_exec = ((close_next - open_next) / open_next.clip(eps, None)).fill_null(0.0)
    ret_exec = ret_exec.clip(-0.02, 0.02)
    df = df.with_columns(ret_exec.alias('ret_exec'))

    # 7. Compute position: signal is generated at t, executed at t+1
    #    So the position for PnL at bar t uses the signal from bar t-1
    position = pl.col('target_exec').shift(1).fill_null(0.0)
    df = df.with_columns(position.alias('position'))

    # 8. Position change (for transaction cost): |pos_t - pos_t-1|
    pos_change = (pl.col('position') - pl.col('position').shift(1)).abs()
    df = df.with_columns(pos_change.fill_null(0.0).alias('pos_change'))

    # 9. Transaction cost in pnl terms: pos_change * TX_COST_PER_ROUNDTURN
    #    This is the friction per trade (entry + exit combined as 1.5 bps)
    tx_cost = pl.col('pos_change') * TX_COST_PER_ROUNDTURN
    df = df.with_columns(tx_cost.alias('tx_cost'))

    # 10. PnL: position * forward return - transaction costs
    #     position is from t-1 signal, ret_exec is t->t+1 return
    #     This is a standard t+1 execution model
    pnl = (pl.col('position') * pl.col('ret_exec') - pl.col('tx_cost'))

    # Additional per-bar micro cost (commission, slippage proportional to position)
    pnl = pnl - pl.col('unit_cost') * pl.col('position').abs()

    pnl = pnl.fill_nan(0.0).clip(-0.05, 0.05)
    df = df.with_columns(pnl.alias('pnl'))

    return df


def simulate_execution(df: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError('Use simulate_execution_classification for new pipeline.')