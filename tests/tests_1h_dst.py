"""
src/features/engine.py
Dynamic feature engineering engine.
Implements memory-safe, deterministic pairwise feature interaction
and baseline structural mapping per the specification.
"""
import logging
import polars as pl
from itertools import combinations
from config import config

logger = logging.getLogger(__name__)

def _get_base_feature_exprs() -> list[pl.Expr]:
    """
    Generates the core baseline features per Section 10.
    All features strictly past-only, deterministic, and float32.
    """
    exprs = []
    
    # Range and Volatility (feature_high_low_range_norm)
    # (high - low) / max(close, EPS)
    exprs.append(
        ((pl.col("high") - pl.col("low")) / 
         pl.max_horizontal(pl.col("close"), config.EPS))
        .cast(pl.Float32)
        .alias("feature_high_low_range_norm")
    )
    
    # Volume Tick-Rule Proxy (feature_signed_bar_strength)
    # If close > open: buy_volume (+1), if close < open: sell_volume (-1)
    bar_sign = (pl.col("close") - pl.col("open")).sign()
    
    # Zero lookahead forward fill for doji bars (close == open)
    lag1_ret_sign = (pl.col("close") / pl.col("close").shift(1)).log().sign()
    final_sign = pl.when(bar_sign != 0).then(bar_sign).otherwise(lag1_ret_sign)
    
    exprs.append(
        ((final_sign * pl.col("volume")) / 
         pl.max_horizontal(pl.col("volume"), config.EPS))
        .cast(pl.Float32)
        .alias("feature_signed_bar_strength")
    )
    
    # Log Volume (feature_log_volume)
    exprs.append(
        pl.when(pl.col("volume") > 0)
        .then(pl.col("volume").log())
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("feature_log_volume")
    )

    # Apply clipping and NaN replacements to base expressions
    cleaned_exprs = [
        e.fill_nan(config.REPLACE_INF_NAN_WITH)
         .fill_null(config.REPLACE_INF_NAN_WITH)
         .clip(config.CLIP_MIN, config.CLIP_MAX) 
        for e in exprs
    ]
    
    return cleaned_exprs


def generate_features(df: pl.LazyFrame | pl.DataFrame, baseline_features: list[str]) -> pl.LazyFrame | pl.DataFrame:
    """
    Generates baseline and dynamic pairwise interaction features.
    
    Compliance Requirements:
    - Adheres to MAX_PAIRWISE_INTERACTIONS exactly (break limit).
    - Strict float32 calculations and downcasting.
    - Zero lookahead (strict t-1 or rolling past boundaries).
    - Values clipped to [CLIP_MIN, CLIP_MAX].
    """
    logger.info("Starting deterministic feature generation...")
    
    # 1. Apply baseline feature expressions natively
    base_exprs = _get_base_feature_exprs()
    df = df.with_columns(base_exprs)
    
    # 2. Pairwise expansion strictly bounded by MAX_PAIRWISE_INTERACTIONS
    # We lexicographically sort baseline features before mapping to ensure deterministic behavior across systems
    sorted_features = sorted(baseline_features)
    
    pairwise_exprs = []
    count = 0
    
    # Generate pairwise combinations safely using Python generator
    for f1, f2 in combinations(sorted_features, 2):
        if count >= config.MAX_PAIRWISE_INTERACTIONS:
            break
            
        new_col_name = f"feature_pair_prod_{f1}_x_{f2}"
        
        # Append bounded definition list to polars lazy expressions
        expr = (
            (pl.col(f1) * pl.col(f2))
            .fill_nan(config.REPLACE_INF_NAN_WITH)
            .fill_null(config.REPLACE_INF_NAN_WITH)
            .clip(config.CLIP_MIN, config.CLIP_MAX)
            .cast(pl.Float32)
            .alias(new_col_name)
        )
        pairwise_exprs.append(expr)
        
        count += 1
        
    logger.info(f"Generated {count} pairwise interactions. Appending to dataframe...")
    
    # Apply all expressions simultaneously to prevent iterative memory bloating
    df = df.with_columns(pairwise_exprs)
    
    # 3. Final global pass: Lexicographically sort all columns for serialization determinism (Section 18)
    sorted_cols = sorted(df.collect_schema().names())
    df = df.select(sorted_cols)
    
    return df