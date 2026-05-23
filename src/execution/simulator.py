"""
src/execution/simulator.py
Execution simulation: stateful position tracking, volatility scaling,
transaction costs, leverage limits, and flatten before session close.
Now includes per‑session bias removal to eliminate spurious drift profits.
"""
import polars as pl
import numpy as np
from config import config


def simulate_execution_classification(df: pl.DataFrame) -> pl.DataFrame:
    """
    Classification version: uses column 'prediction_prob' (probability of up move).
    Raw signal = 2*prob - 1 (range -1 to 1). Then scaled by TARGET_VOL / vol.
    Bias removal: centres probabilities per session (mean becomes 0.5) to remove
    any systematic long/short bias that could profit from a trending market.
    Adds columns: 'position', 'trade_cost', 'pnl'.
    """
    # ---- 1. Volatility column ----
    if "feature_ewma_vol_20" not in df.columns:
        ret = (pl.col("close") / pl.col("close").shift(1)).log()
        vol = ret.rolling_std(window_size=20)
    else:
        vol = pl.col("feature_ewma_vol_20")
    df = df.with_columns(
        vol.fill_null(strategy="forward").fill_null(1e-6).alias("vol")
    )

    # ---- 2. Bias removal (per session) ----
    prob_series = df["prediction_prob"].fill_null(0.5).clip(0.0, 1.0)
    if getattr(config, 'REMOVE_PREDICTION_BIAS', False):
        # Convert to writable numpy array
        probs = prob_series.to_numpy().copy()   # .copy() makes it writable
        sess_ids = df["session_id"].to_numpy()
        unique_sessions = np.unique(sess_ids)
        for sess in unique_sessions:
            mask = (sess_ids == sess)
            sess_mean = probs[mask].mean()
            # Center around 0.5, keeping the same range
            probs[mask] = probs[mask] - sess_mean + 0.5
        # Clip again to [0,1] after adjustment
        probs = np.clip(probs, 0.0, 1.0)
        prob_series = pl.Series(probs)
        print(f"Per‑session bias removed: each session's mean probability → 0.5")

    # ---- 3. Raw signal from probability ----
    raw_signal = (prob_series - 0.5) * 2.0   # maps [0,1] -> [-1,1]

    target_raw_expr = (raw_signal / pl.col("vol")) * config.TARGET_VOL
    target_raw_expr = target_raw_expr.clip(-config.MAX_LEVERAGE, config.MAX_LEVERAGE)

    # HTF volatility scaling
    if config.HTF_VOL_SCALING and "htf_daily_vol_5" in df.columns:
        daily_atr = pl.col("htf_daily_vol_5").fill_null(strategy="forward").fill_null(1e-6)
        scaling = (config.TARGET_VOL / daily_atr).clip(0.25, 2.0)
        target_raw_expr = (target_raw_expr * scaling).clip(-config.MAX_LEVERAGE, config.MAX_LEVERAGE)

    # HTF trend alignment filter
    if config.HTF_TREND_ALIGNMENT and "htf_daily_trend_slope_10" in df.columns:
        daily_trend = pl.col("htf_daily_trend_slope_10").sign()
        target_raw_expr = pl.when(
            (daily_trend == 0) | (target_raw_expr.sign() == daily_trend)
        ).then(target_raw_expr).otherwise(0)

    # ---- 4. Rate limit (loop) ----
    target_series = df.select(target_raw_expr).to_series().fill_nan(0.0).fill_null(0.0)
    target_array = target_series.to_numpy()

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
    open_next = np.roll(df["open"].to_numpy(), -1)
    close_next = np.roll(df["close"].to_numpy(), -1)
    open_next[-1] = np.nan
    close_next[-1] = np.nan
    ret_exec = (close_next - open_next) / np.maximum(open_next, config.EPS)
    ret_exec = np.nan_to_num(ret_exec, nan=0.0)

    # ---- 5. Loop ----
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
    """Legacy regression version (uses 'prediction' column)."""
    raise NotImplementedError("Use simulate_execution_classification for new pipeline.")