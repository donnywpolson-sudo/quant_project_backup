"""
simulator.py
Realistic execution simulator with economic friction.
- 1.5 bps fixed cost per round-turn (applied to position changes)
- Signal generated at bar t, executed at bar t+1 open
- Volatility computed from lagged returns [t-N, t-1]
- Position changes tracked for turnover calculation
- HTF context-aware trade gating (directional bias, vol scaling, trend alignment)
- Session close flatting (zero position within 5 min of 16:00 ET)
"""
import numpy as np
import polars as pl
from quant.config import config

# Fixed contract size multiplier for position sizing
FIXED_CONTRACT_SIZE = 1.0


def simulate_execution_classification(df: pl.DataFrame) -> pl.DataFrame:
    eps = config.EPS

    # ------------------------------------------------------------------------
    # 1. Generate trading signal from prediction probability
    #    (prediction_prob is based on features at time t, so signal is
    #     generated at t)
    # ------------------------------------------------------------------------
    signal_expr = (
        pl.when(pl.col('prediction_prob').fill_null(0.5) > 0.6)
        .then(1.0)
        .when(pl.col('prediction_prob').fill_null(0.5) < 0.4)
        .then(-1.0)
        .otherwise(0.0)
    )
    df = df.with_columns(signal_expr.cast(pl.Float32).alias('raw_signal'))

    # Initialize target_exec from raw_signal as Float32
    df = df.with_columns(
        pl.col('raw_signal').cast(pl.Float32).alias('target_exec')
    )

    # ------------------------------------------------------------------------
    # 2. HTF Directional Bias: gate trades against HTF hourly trend alignment.
    #    Long only if htf_hourly_trend_alignment > 0 (suppress longs otherwise)
    #    Short only if htf_hourly_trend_alignment < 0 (suppress shorts otherwise)
    #    Strict zero-boundary gating — no neutral zone threshold.
    # ------------------------------------------------------------------------
    col_names = df.columns
    if 'htf_hourly_trend_alignment' in col_names:
        htf_align = (
            pl.col('htf_hourly_trend_alignment')
            .fill_null(0.0)
            .cast(pl.Float32)
        )
        target = pl.col('target_exec')

        target_gated = (
            pl.when((target > 0.0) & (htf_align <= 0.0))
            .then(pl.lit(0.0, dtype=pl.Float32))
            .when((target < 0.0) & (htf_align >= 0.0))
            .then(pl.lit(0.0, dtype=pl.Float32))
            .otherwise(target)
        )
        df = df.with_columns(target_gated.alias('target_exec'))

    # ------------------------------------------------------------------------
    # 3. Session break filter (flatten position during exchange break)
    # ------------------------------------------------------------------------
    df = df.with_columns(
        pl.col('ts_event')
        .dt.convert_time_zone(config.TIMEZONE)
        .dt.time()
        .alias('t_local')
    )
    df = df.with_columns(
        pl.when(
            (pl.col('t_local') >= pl.lit(config.SESSION_BREAK_START_LOCAL))
            & (pl.col('t_local') < pl.lit(config.SESSION_BREAK_END_LOCAL))
        )
        .then(pl.lit(0.0, dtype=pl.Float32))
        .otherwise(pl.col('target_exec'))
        .alias('target_exec')
    )

    # ------------------------------------------------------------------------
    # 4. Session Close Flatting: zero out position within
    #    FLAT_BEFORE_CLOSE_MINUTES of SESSION_END_LOCAL (16:00 ET).
    # ------------------------------------------------------------------------
    close_time = config.SESSION_END_LOCAL  # time(16, 0)
    flat_minutes = config.FLAT_BEFORE_CLOSE_MINUTES  # 5
    t_minutes = (
        pl.col('t_local').dt.hour().cast(pl.Int32) * 60
        + pl.col('t_local').dt.minute().cast(pl.Int32)
    )
    close_minutes = close_time.hour * 60 + close_time.minute  # 960
    flat_start_minutes = close_minutes - flat_minutes          # 955 (15:55)
    df = df.with_columns(
        pl.when(t_minutes >= pl.lit(flat_start_minutes, dtype=pl.Int32))
        .then(pl.lit(0.0, dtype=pl.Float32))
        .otherwise(pl.col('target_exec'))
        .alias('target_exec')
    )
    df = df.drop('t_local')

    # ------------------------------------------------------------------------
    # 5. HTF Volatility Scaling: scale position size by
    #    1 / (htf_daily_vol_5 * HTF_VOL_WINDOW), capped at MAX_LEVERAGE.
    #    Only applied when config.HTF_VOL_SCALING is enabled and the
    #    column exists in the DataFrame.
    # ------------------------------------------------------------------------
    if config.HTF_VOL_SCALING and ('htf_daily_vol_5' in col_names):
        htf_vol = (
            pl.col('htf_daily_vol_5')
            .fill_null(1e-06)
            .cast(pl.Float32)
            .clip(eps, None)
        )
        vol_scaler = (
            pl.lit(1.0, dtype=pl.Float32)
            / (htf_vol * pl.lit(config.HTF_VOL_WINDOW, dtype=pl.Float32))
        ).clip(
            pl.lit(-config.MAX_LEVERAGE, dtype=pl.Float32),
            pl.lit(config.MAX_LEVERAGE, dtype=pl.Float32),
        )
        df = df.with_columns(
            (pl.col('target_exec') * vol_scaler)
            .cast(pl.Float32)
            .alias('target_exec')
        )

    # ------------------------------------------------------------------------
    # 6. Fixed Contract Sizing: apply constant multiplier.
    # ------------------------------------------------------------------------
    df = df.with_columns(
        (pl.col('target_exec') * pl.lit(FIXED_CONTRACT_SIZE, dtype=pl.Float32))
        .cast(pl.Float32)
        .alias('target_exec')
    )

    # ------------------------------------------------------------------------
    # 7. HTF Trend Alignment: if config.HTF_TREND_ALIGNMENT is enabled,
    #    suppress signals that disagree with htf_daily_trend_slope_10.
    #    - Suppress longs when daily trend is strongly down (< -threshold)
    #    - Suppress shorts when daily trend is strongly up (> +threshold)
    # ------------------------------------------------------------------------
    if config.HTF_TREND_ALIGNMENT and ('htf_daily_trend_slope_10' in col_names):
        daily_trend = (
            pl.col('htf_daily_trend_slope_10')
            .fill_null(0.0)
            .cast(pl.Float32)
        )
        trend_threshold = pl.lit(config.HTF_TREND_THRESHOLD, dtype=pl.Float32)
        target = pl.col('target_exec')

        target_aligned = (
            pl.when((target > 0.0) & (daily_trend < -trend_threshold))
            .then(pl.lit(0.0, dtype=pl.Float32))
            .when((target < 0.0) & (daily_trend > trend_threshold))
            .then(pl.lit(0.0, dtype=pl.Float32))
            .otherwise(target)
        )
        df = df.with_columns(target_aligned.alias('target_exec'))

    # ------------------------------------------------------------------------
    # 8. Volatility: strictly lagged, computed from returns [t-N, t-1]
    # ------------------------------------------------------------------------
    ret = (pl.col('close') / pl.col('close').shift(1)).log()
    ret_lagged = ret.shift(1)
    vol = ret_lagged.rolling_std(window_size=20).clip(eps, None)
    df = df.with_columns(vol.fill_null(1e-06).alias('vol'))

    # ------------------------------------------------------------------------
    # 9. Spread proxy (bid-ask approximation)
    # ------------------------------------------------------------------------
    spread = (pl.col('high') - pl.col('low')) / pl.col('close').clip(eps, None)
    spread = spread.clip(0.0, 0.05)
    df = df.with_columns(spread.alias('spread'))

    # ------------------------------------------------------------------------
    # 10. Unified per-bar cost charged on position CHANGE (turnover).
    #     Commission, slippage, vol penalty, and tx_cost_per_roundturn
    #     are all consolidated into a single cost rate applied to
    #     absolute position changes (entry + exit friction).
    # ------------------------------------------------------------------------
    # TX_COST_PER_ROUNDTURN (1.5 bps) covers the full round-trip (entry + exit).
    # unit_cost is charged per position change, and a round-trip has two
    # position changes (enter and exit), so we divide by 2 to avoid double-charging.
    unit_cost = (
        config.COMMISSION_PER_TRADE
        + config.SLIPPAGE_K * pl.col('spread')
        + config.VOL_PENALTY * pl.col('vol')
        + config.TX_COST_PER_ROUNDTURN / 2.0
    ).clip(0.0, 0.01)
    df = df.with_columns(unit_cost.alias('unit_cost'))

    # ------------------------------------------------------------------------
    # 11. Execution return: buy/sell at bar t+1 open, exit at bar t+1 close
    #     This is forward-looking but acceptable because:
    #     - target_exec was computed from features at time t
    #     - trade is assumed executed at next bar's open
    #     - trade is unwound at next bar's close
    # ------------------------------------------------------------------------
    open_next = pl.col('open').shift(-1)
    close_next = pl.col('close').shift(-1)
    ret_exec = ((close_next - open_next) / open_next.clip(eps, None)).fill_null(0.0)
    ret_exec = ret_exec.clip(-0.02, 0.02)
    df = df.with_columns(ret_exec.alias('ret_exec'))

    # ------------------------------------------------------------------------
    # 12. Compute position: signal is generated at t, executed at t+1.
    #     So the position for PnL at bar t uses the signal from bar t-1.
    # ------------------------------------------------------------------------
    position = pl.col('target_exec').shift(1).fill_null(0.0)
    df = df.with_columns(position.alias('position'))

    # ------------------------------------------------------------------------
    # 13. Position change (for transaction cost): |pos_t - pos_t-1|
    # ------------------------------------------------------------------------
    pos_change = (pl.col('position') - pl.col('position').shift(1)).abs()
    df = df.with_columns(pos_change.fill_null(0.0).alias('pos_change'))

    # ------------------------------------------------------------------------
    # 14. PnL: position * forward return - unified transaction costs.
    #     position is from t-1 signal, ret_exec is t->t+1 return.
    #     Friction (commission, slippage, vol penalty, tx_cost) is charged
    #     on position changes (turnover) only, not on held position.
    # ------------------------------------------------------------------------
    pnl = pl.col('position') * pl.col('ret_exec')
    pnl = pnl - pl.col('unit_cost') * pl.col('pos_change')
    pnl = pnl.fill_nan(0.0).clip(-0.05, 0.05)
    df = df.with_columns(pnl.alias('pnl'))

    return df


def simulate_execution(df: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError(
        'Use simulate_execution_classification for new pipeline.'
    )