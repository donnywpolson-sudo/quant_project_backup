"""
src/execution/simulator.py
Execution simulation: stateful position tracking, volatility scaling,
transaction costs, leverage limits, and flatten before session close.
Now includes HTF volatility scaling and trend alignment if enabled in config.
"""
import polars as pl
import numpy as np
from config import config


def simulate_execution(df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds columns: 'position', 'trade_cost', 'pnl'.
    Now includes HTF volatility scaling and trend alignment if enabled in config.
    """
    # Ensure we have volatility
    if "feature_ewma_vol_20" not in df.columns:
        ret = (pl.col("close") / pl.col("close").shift(1)).log()
        vol = ret.rolling_std(window_size=20)
        df = df.with_columns(vol.alias("vol"))
    else:
        df = df.with_columns(pl.col("feature_ewma_vol_20").alias("vol"))

    # Raw target position
    target_raw = (pl.col("prediction") / pl.col("vol").clip(config.EPS, None)) * config.TARGET_VOL
    target_raw = target_raw.clip(-config.MAX_LEVERAGE, config.MAX_LEVERAGE)

    # ---- HTF Volatility Scaling ----
    if config.HTF_VOL_SCALING and "htf_daily_vol_5" in df.columns:
        daily_target_vol = config.TARGET_VOL  # could be market-specific
        daily_atr = df["htf_daily_vol_5"]
        scaling = (daily_target_vol / daily_atr.clip(config.EPS, None)).clip(0.25, 2.0)
        target_raw = target_raw * scaling
        target_raw = target_raw.clip(-config.MAX_LEVERAGE, config.MAX_LEVERAGE)

    # ---- HTF Trend Alignment Filter ----
    if config.HTF_TREND_ALIGNMENT and "htf_daily_trend_slope_10" in df.columns:
        daily_trend = df["htf_daily_trend_slope_10"].sign()
        # Zero trend means no filter
        target_raw = pl.when(
            (daily_trend == 0) | (target_raw.sign() == daily_trend)
        ).then(target_raw).otherwise(0)

    # Spread proxy
    if "feature_spread_proxy" in df.columns:
        spread = pl.col("feature_spread_proxy")
    else:
        spread = (pl.col("high") - pl.col("low")) / pl.col("close").clip(config.EPS, None)

    unit_cost = config.COMMISSION_PER_TRADE + config.SLIPPAGE_K * spread + config.VOL_PENALTY * pl.col("vol")

    # Stateful position loop (convert to numpy)
    target_array = target_raw.to_numpy()
    unit_cost_array = unit_cost.to_numpy()
    df = df.with_columns(
        pl.col("ts_event").rank("ordinal").over("session_id").alias("_session_rank"),
        pl.col("ts_event").count().over("session_id").alias("_session_len")
    )
    last_bars_mask = (df["_session_rank"] > (df["_session_len"] - config.FLAT_BEFORE_CLOSE_MINUTES//5)).to_numpy()

    n = len(df)
    positions = np.zeros(n, dtype=np.float32)
    trade_costs = np.zeros(n, dtype=np.float32)
    open_next = np.roll(df["open"].to_numpy(), -1)
    close_next = np.roll(df["close"].to_numpy(), -1)
    open_next[-1] = np.nan
    close_next[-1] = np.nan

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

    ret_exec = (close_next - open_next) / np.maximum(open_next, config.EPS)
    pnl = positions * ret_exec - trade_costs
    pnl = np.nan_to_num(pnl, nan=0.0)

    df = df.with_columns([
        pl.Series("position", positions).cast(pl.Float32),
        pl.Series("trade_cost", trade_costs).cast(pl.Float32),
        pl.Series("pnl", pnl).cast(pl.Float32)
    ])
    df = df.drop(["_session_rank", "_session_len"])
    return df