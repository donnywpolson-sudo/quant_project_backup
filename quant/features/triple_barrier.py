import polars as pl
import numpy as np


def add_triple_barrier_target(df: pl.DataFrame) -> pl.DataFrame:
    """
    Volatility-adjusted triple-barrier label.

    For each bar t, constructs a 64-bar (~5.3h) forward window and tests
    which of three barriers is touched first:

      Upper:  entry_price * exp(+0.08)  (≈ 8.33% above entry)
      Lower:  entry_price * exp(-0.04)  (≈ 3.92% below entry)
      Time:   64 bars (expiry)

    vol_4h = daily_vol * sqrt(64 / 276)  (daily vol scaled to window)

    Labels:
      +1   — upper barrier touched first (bullish)
      -1   — lower barrier touched first (bearish)
       0   — time barrier expired (neutral / sideways)
      NaN  — no forward window available (last 64 bars of dataset)

    The daily volatility is read from the ``htf_daily_vol_5`` column which
    is computed during session resampling (5-day rolling std of daily
    log returns).  Falls back to a 20-bar rolling std when the
    column is unavailable.
    """
    H_BARS = 64
    BARS_PER_DAY = 276

    close = df['close'].to_numpy().astype(np.float64)
    high = df['high'].to_numpy().astype(np.float64)
    low = df['low'].to_numpy().astype(np.float64)
    n = len(close)

    # Daily volatility — use aligned column when available.
    # htf_daily_vol_5 is in percentage form (e.g. 3.37 = 3.37%).
    # Convert to decimal before scaling.
    if 'htf_daily_vol_5' in df.columns:
        daily_vol = df['htf_daily_vol_5'].to_numpy().astype(np.float64) * 0.01
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

    # Fixed percentage barriers per market.
    # +8.33% upper / -3.92% lower over 64-bar window.
    UPPER_BARRIER_PCT = 0.20
    LOWER_BARRIER_PCT = 0.12
    upper_mult = np.full(n, np.exp(UPPER_BARRIER_PCT))
    lower_mult = np.full(n, np.exp(-LOWER_BARRIER_PCT))

    labels = np.full(n, np.nan, dtype=np.float64)

    # Use raw close for market segmentation — continuous_price has zero
    # variance in the merged multi-market feature matrix.
    cp = close
    market_id = (np.log10(np.maximum(cp, 1e-9)) // 0.5).astype(int)

    # Process each contiguous market segment separately
    seg_start = 0
    mid_prev = market_id[0]
    for i in range(1, n + 1):
        if i == n or market_id[i] != mid_prev:
            seg_end = i
            _compute_barriers_segment(
                seg_start, seg_end, H_BARS, close, high, low,
                upper_mult, lower_mult, labels
            )
            if i < n:
                seg_start = i
                mid_prev = market_id[i]

    values = [float(v) if np.isfinite(v) else None for v in labels]
    df = df.with_columns(pl.Series('target_tb', values))
    return df


def _compute_barriers_segment(start, end, H_BARS, close, high, low,
                              upper_mult, lower_mult, labels):
    """Compute triple-barrier labels for a contiguous single-market segment."""
    for t in range(start, end - H_BARS):
        entry = close[t]
        if entry <= 0:
            continue
        window_end = min(t + 1 + H_BARS, end)

        upper_barrier = entry * upper_mult[t]
        lower_barrier = entry * lower_mult[t]

        high_window = high[t + 1:window_end]
        low_window = low[t + 1:window_end]

        upper_hit = np.argmax(high_window >= upper_barrier)
        lower_hit = np.argmax(low_window <= lower_barrier)

        wl = window_end - t - 1
        upper_idx = int(upper_hit) if high_window[upper_hit] >= upper_barrier else wl
        lower_idx = int(lower_hit) if low_window[lower_hit] <= lower_barrier else wl

        if upper_idx < wl and upper_idx <= lower_idx:
            labels[t] = 1.0
        elif lower_idx < wl and lower_idx < upper_idx:
            labels[t] = -1.0
        else:
            labels[t] = 0.0
