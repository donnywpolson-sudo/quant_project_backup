"""
src/execution/simulator.py
Execution simulation: stateful position tracking, volatility scaling,
transaction costs, leverage limits, and flatten before session close.
Now with safe fallbacks for missing HTF columns.
"""
import polars as pl
import numpy as np
from config import config

def simulate_execution_classification(df: pl.DataFrame) -> pl.DataFrame:
    """Classification version: uses 'prediction_prob'. Uses fixed unit sizing and HTF trend bias."""
    # Always use fixed contract sizing: signal direction is binary, size is constant.
    signal_expr = pl.when((pl.col("prediction_prob").fill_null(0.5) - 0.5) > 0).then(1.0)
    signal_expr = signal_expr.when((pl.col("prediction_prob").fill_null(0.5) - 0.5) < 0).then(-1.0).otherwise(0.0)

    # HTF directional bias: daily trend gives a long/short/neutral bias, hourly alignment sharpens it.
    if config.HTF_TREND_ALIGNMENT and "htf_daily_trend_slope_10" in df.columns:
        daily_bias = pl.when(
            pl.col("htf_daily_trend_slope_10").abs() >= config.HTF_TREND_THRESHOLD
        ).then(pl.col("htf_daily_trend_slope_10").sign()).otherwise(0.0)
        if "htf_hourly_trend_alignment" in df.columns:
            hourly_align = pl.col("htf_hourly_trend_alignment").sign()
            daily_bias = pl.when(
                (daily_bias != 0.0) & (hourly_align == daily_bias)
            ).then(daily_bias).otherwise(0.0)
        target_raw_expr = pl.when(
            (signal_expr == daily_bias) & (daily_bias != 0.0)
        ).then(daily_bias).otherwise(0.0)
    else:
        target_raw_expr = signal_expr

    target_series = df.select(target_raw_expr).to_series().fill_nan(0.0).fill_null(0.0)
    target_array = target_series.to_numpy()

    # Volatility column for cost estimation only.
    if "feature_ewma_vol_20" not in df.columns:
        ret = (pl.col("close") / pl.col("close").shift(1)).log()
        vol = ret.rolling_std(window_size=20)
    else:
        vol = pl.col("feature_ewma_vol_20")
    df = df.with_columns(
        vol.fill_null(strategy="forward").fill_null(1e-6).alias("vol")
    )

    # Spread proxy
    if "feature_spread_proxy" in df.columns:
        spread_expr = pl.col("feature_spread_proxy")
    else:
        spread_expr = (pl.col("high") - pl.col("low")) / pl.col("close").clip(config.EPS, None)
    spread_series = df.select(spread_expr).to_series().fill_nan(0.0).fill_null(0.0)
    unit_cost_array = (config.COMMISSION_PER_TRADE + config.SLIPPAGE_K * spread_series + config.VOL_PENALTY * df["vol"]).to_numpy()

    # Session flatten mask
    df = df.with_columns(
        pl.col("ts_event").rank("ordinal").over("session_id").alias("_session_rank"),
        pl.col("ts_event").count().over("session_id").alias("_session_len")
    )
    last_bars_mask = (df["_session_rank"] > (df["_session_len"] - config.FLAT_BEFORE_CLOSE_MINUTES // 5)).to_numpy()

    # Pre‑compute returns
    open_next = np.roll(df["open"].to_numpy().astype(np.float64), -1)
    close_next = np.roll(df["close"].to_numpy().astype(np.float64), -1)
    open_next[-1] = np.nan
    close_next[-1] = np.nan
    ret_exec = (close_next - open_next) / np.maximum(open_next, config.EPS)
    ret_exec = np.nan_to_num(ret_exec, nan=0.0)

    n = len(df)
    positions = np.zeros(n, dtype=np.float32)
    trade_costs = np.zeros(n, dtype=np.float32)
    current_pos = 0.0
    for i in range(n):
        desired = target_array[i]
        if last_bars_mask[i]:
            desired = 0.0
        delta = np.clip(desired - current_pos, -config.MAX_POS_CHANGE_PER_MIN, config.MAX_POS_CHANGE_PER_MIN)
        new_pos = current_pos + delta
        new_pos = np.clip(new_pos, -config.MAX_LEVERAGE, config.MAX_LEVERAGE)
        cost = abs(new_pos - current_pos) * unit_cost_array[i]
        trade_costs[i] = cost
        positions[i] = new_pos
        current_pos = new_pos

    pnl = positions * ret_exec - trade_costs
    pnl = np.nan_to_num(pnl, nan=0.0)

    df = df.with_columns([
        pl.Series("position", positions).cast(pl.Float32),
        pl.Series("trade_cost", trade_costs).cast(pl.Float32),
        pl.Series("pnl", pnl).cast(pl.Float32)
    ]).drop(["_session_rank", "_session_len"])
    return df

def simulate_execution(df: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError("Use simulate_execution_classification for new pipeline.")