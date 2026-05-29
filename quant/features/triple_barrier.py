import polars as pl
import numpy as np


def add_triple_barrier_target(df: pl.DataFrame) -> pl.DataFrame:
    """
    Volatility-adjusted 4h triple-barrier label.

    For each bar t, constructs a 48-bar forward window and tests
    which of three barriers is touched first:

      Upper:  entry_price * exp(+2.0 * vol_4h)
      Lower:  entry_price * exp(-1.0 * vol_4h)
      Time:   48 bars (expiry)

    vol_4h = daily_vol_5 * sqrt(48 / 276)  (daily vol scaled to 4h)

    Labels:
      +1   — upper barrier touched first (bullish)
      -1   — lower barrier touched first (bearish)
       0   — time barrier expired (neutral / sideways)
      NaN  — no forward window available (last 48 bars of dataset)

    The daily volatility is read from the ``daily_vol_5`` column which
    is computed during session resampling (5-day rolling std of daily
    log returns).  Falls back to a 20-bar rolling std when the daily
    stream is unavailable.
    """
    H_BARS = 48
    BARS_PER_DAY = 276

    close = df['close'].to_numpy().astype(np.float64)
    high = df['high'].to_numpy().astype(np.float64)
    low = df['low'].to_numpy().astype(np.float64)
    n = len(close)

    # Daily volatility — use aligned column when available
    if 'daily_vol_5' in df.columns:
        daily_vol = df['daily_vol_5'].to_numpy().astype(np.float64)
        daily_vol = np.nan_to_num(daily_vol, nan=0.01)
    else:
        rolling_std = np.full(n, np.nan)
        min_window = 20
        for i in range(min_window, n):
            rets = np.diff(np.log(close[max(0, i - min_window):i + 1] + 1e-12))
            if len(rets) > 1:
                rolling_std[i] = np.std(rets)
        daily_vol = np.nan_to_num(rolling_std, nan=0.01)

    daily_vol = np.maximum(daily_vol, 0.001)

    # Barrier scaling — 4h vol = daily_vol * sqrt(48 / 276)
    vol_4h = daily_vol * np.sqrt(H_BARS / BARS_PER_DAY)
    upper_mult = np.exp(+2.0 * vol_4h)
    lower_mult = np.exp(-1.0 * vol_4h)

    labels = np.full(n, np.nan, dtype=np.float64)

    for t in range(n - H_BARS):
        entry = close[t]
        if entry <= 0:
            continue

        upper_barrier = entry * upper_mult[t]
        lower_barrier = entry * lower_mult[t]

        high_window = high[t + 1:min(t + 1 + H_BARS, n)]
        low_window = low[t + 1:min(t + 1 + H_BARS, n)]

        upper_hit = np.argmax(high_window >= upper_barrier)
        lower_hit = np.argmax(low_window <= lower_barrier)

        upper_idx = int(upper_hit) if high_window[upper_hit] >= upper_barrier else H_BARS
        lower_idx = int(lower_hit) if low_window[lower_hit] <= lower_barrier else H_BARS

        if upper_idx < H_BARS and upper_idx <= lower_idx:
            labels[t] = 1.0
        elif lower_idx < H_BARS and lower_idx < upper_idx:
            labels[t] = -1.0
        else:
            labels[t] = 0.0

    values = [float(v) if np.isfinite(v) else None for v in labels]
    df = df.with_columns(pl.Series('target_tb', values))
    return df
