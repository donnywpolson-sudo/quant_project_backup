import numpy as np
import polars as pl
from quant.config_manager import config


def add_meta_label_target(
    df: pl.DataFrame,
    primary_prediction_col: str = 'primary_prediction',
    primary_target_col: str = 'target_tb',
) -> pl.DataFrame:
    """
    Meta-labeling target — predicts whether the primary model's
    directional forecast was correct.

    After the primary model (triple-barrier) assigns a direction
    (+1 / -1), the meta-label answers: "Would a trade in that
    direction have been profitable?"

    Label:
      1 — primary prediction direction matched realized outcome
      0 — primary prediction was wrong
     -1 — primary prediction was neutral (no trade)

    This is trained AFTER the primary model, using frozen features +
    the primary model's prediction as an additional input. The
    resulting meta-label probability gates the Ridge execution layer.
    """
    n = df.height
    labels = np.full(n, -1, dtype=np.int8)

    if primary_prediction_col not in df.columns:
        return df.with_columns(pl.Series('target_meta', labels))

    if primary_target_col not in df.columns:
        return df.with_columns(pl.Series('target_meta', labels))

    pred = df[primary_prediction_col].to_numpy().astype(np.int8)
    actual = df[primary_target_col].to_numpy().astype(np.int8)

    for i in range(n):
        p = pred[i]
        a = actual[i]

        if p == 0:
            labels[i] = -1  # primary model was neutral — meta-label abstains
        elif p == a:
            labels[i] = 1   # primary model was correct
        else:
            labels[i] = 0   # primary model was wrong

    return df.with_columns(pl.Series('target_meta', labels))
