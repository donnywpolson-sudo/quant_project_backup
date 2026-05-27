# Feature Leakage Audit Report

## Executive Summary

- **Total feature columns audited:** 751
- **Data rows analyzed:** 67,811
- **Clean features:** 709
- **Cheating features (look-ahead bias detected):** 40

**Methodology:** For each feature $X_t$, we computed the Pearson correlation with the 5-minute log-return $r_{t+h}$ at horizons $h \in [-5, +5]$. A feature is flagged as "cheating" if:

1. $|\text{corr}(X_t, r_{t+1})| > |\text{corr}(X_t, r_{t-5})|$
2. The forward correlation is statistically significant ($p < 0.01$)
3. $|corr(X_t, r_{t+1})| \geq 0.02$ (minimum magnitude threshold)

This pattern — strong forward correlation with weak backward correlation — is the telltale signature of look-ahead bias: the feature "knows" about future returns at $t+1$ but not past returns at $t-5$.

---

## 🚨 Cheating Features (Look-Ahead Bias Detected)

| # | Feature | Forward Corr (h=+1) | Forward p-value | Backward Corr (h=-5) | Backward Corr (h=-1) | Verdict |
|---|---------|---------------------|-----------------|----------------------|----------------------|---------|
| 1 | `cross_feature_high_low_range_norm_x_htf_distance_to_daily_low` | -0.0995 | 7.12e-149 | +0.0082 | -0.0005 | CHEATING |
| 2 | `cross_feature_spread_proxy_x_htf_distance_to_daily_low` | -0.0995 | 7.12e-149 | +0.0082 | -0.0005 | CHEATING |
| 3 | `pair_feature_high_low_range_norm_x_htf_distance_to_daily_low` | -0.0995 | 7.12e-149 | +0.0082 | -0.0005 | CHEATING |
| 4 | `cross_feature_ewma_vol_20_x_htf_distance_to_daily_low` | -0.0954 | 9.80e-137 | +0.0098 | +0.0053 | CHEATING |
| 5 | `pair_feature_ewma_vol_20_x_htf_distance_to_daily_low` | -0.0954 | 9.80e-137 | +0.0098 | +0.0053 | CHEATING |
| 6 | `cross_feature_ret_quantile_0.2_20_x_htf_distance_to_daily_low` | +0.0913 | 1.72e-125 | +0.0231 | +0.0174 | CHEATING |
| 7 | `cross_feature_high_low_range_norm_x_htf_distance_to_daily_high` | +0.0900 | 5.36e-122 | -0.0244 | -0.0098 | CHEATING |
| 8 | `cross_feature_spread_proxy_x_htf_distance_to_daily_high` | +0.0900 | 5.36e-122 | -0.0244 | -0.0098 | CHEATING |
| 9 | `pair_feature_high_low_range_norm_x_htf_distance_to_daily_high` | +0.0900 | 5.36e-122 | -0.0244 | -0.0098 | CHEATING |
| 10 | `cross_feature_true_range_x_htf_distance_to_daily_high` | +0.0870 | 4.45e-114 | -0.0259 | -0.0231 | CHEATING |
| 11 | `cross_feature_ret_quantile_0.8_20_x_htf_distance_to_daily_low` | -0.0828 | 1.58e-103 | +0.0383 | +0.0378 | CHEATING |
| 12 | `cross_feature_ewma_vol_20_x_htf_distance_to_daily_high` | +0.0806 | 4.18e-98 | -0.0294 | -0.0230 | CHEATING |
| 13 | `pair_feature_ewma_vol_20_x_htf_distance_to_daily_high` | +0.0806 | 4.18e-98 | -0.0294 | -0.0230 | CHEATING |
| 14 | `cross_feature_ret_quantile_0.8_20_x_htf_distance_to_daily_high` | +0.0805 | 6.20e-98 | +0.0121 | +0.0120 | CHEATING |
| 15 | `htf_distance_to_daily_low` | -0.0765 | 1.23e-88 | +0.0086 | +0.0087 | CHEATING |
| 16 | `cross_feature_cos_time_x_htf_distance_to_daily_low` | -0.0760 | 1.95e-87 | +0.0085 | +0.0088 | CHEATING |
| 17 | `pair_feature_cos_time_x_htf_distance_to_daily_low` | -0.0760 | 1.95e-87 | +0.0085 | +0.0088 | CHEATING |
| 18 | `cross_feature_ret_quantile_0.2_20_x_htf_distance_to_daily_high` | -0.0738 | 1.79e-82 | +0.0500 | +0.0476 | CHEATING |
| 19 | `htf_distance_to_daily_high` | +0.0734 | 1.19e-81 | -0.0164 | -0.0160 | CHEATING |
| 20 | `cross_feature_cos_time_x_htf_distance_to_daily_high` | +0.0726 | 6.37e-80 | -0.0164 | -0.0162 | CHEATING |
| 21 | `pair_feature_cos_time_x_htf_distance_to_daily_high` | +0.0726 | 6.37e-80 | -0.0164 | -0.0162 | CHEATING |
| 22 | `cross_feature_ret_kurt_20_x_htf_distance_to_daily_low` | +0.0712 | 6.16e-77 | -0.0062 | -0.0090 | CHEATING |
| 23 | `cross_feature_ret_kurt_20_x_htf_distance_to_daily_high` | -0.0668 | 7.82e-68 | +0.0095 | +0.0122 | CHEATING |
| 24 | `cross_feature_dow_x_htf_distance_to_daily_low` | -0.0656 | 1.78e-65 | +0.0075 | +0.0090 | CHEATING |
| 25 | `pair_feature_dow_x_htf_distance_to_daily_low` | -0.0656 | 1.78e-65 | +0.0075 | +0.0090 | CHEATING |
| 26 | `cross_feature_dow_x_htf_distance_to_daily_high` | +0.0596 | 2.17e-54 | -0.0187 | -0.0194 | CHEATING |
| 27 | `pair_feature_dow_x_htf_distance_to_daily_high` | +0.0596 | 2.17e-54 | -0.0187 | -0.0194 | CHEATING |
| 28 | `cross_feature_high_low_range_norm_zscore_x_htf_distance_to_daily_low` | -0.0367 | 1.14e-21 | -0.0045 | -0.0140 | CHEATING |
| 29 | `cross_feature_spread_proxy_zscore_x_htf_distance_to_daily_low` | -0.0367 | 1.14e-21 | -0.0045 | -0.0140 | CHEATING |
| 30 | `pair_feature_high_low_range_norm_zscore_x_htf_distance_to_daily_low` | -0.0367 | 1.14e-21 | -0.0045 | -0.0140 | CHEATING |
| 31 | `cross_feature_ret_5_x_htf_volatility_ratio` | +0.0344 | 2.97e-19 | +0.0063 | +0.1882 | CHEATING |
| 32 | `pair_feature_ret_5_x_htf_volatility_ratio` | +0.0344 | 2.96e-19 | +0.0063 | +0.1882 | CHEATING |
| 33 | `cross_feature_ewma_vol_20_zscore_x_htf_distance_to_daily_low` | -0.0329 | 1.07e-17 | -0.0056 | -0.0017 | CHEATING |
| 34 | `pair_feature_ewma_vol_20_zscore_x_htf_distance_to_daily_low` | -0.0329 | 1.07e-17 | -0.0056 | -0.0017 | CHEATING |
| 35 | `cross_feature_ret_5_zscore_x_htf_volatility_ratio` | +0.0320 | 7.12e-17 | -0.0260 | +0.1795 | CHEATING |
| 36 | `cross_feature_high_low_range_norm_zscore_x_htf_distance_to_daily_high` | +0.0292 | 2.84e-14 | -0.0082 | -0.0024 | CHEATING |
| 37 | `cross_feature_spread_proxy_zscore_x_htf_distance_to_daily_high` | +0.0292 | 2.84e-14 | -0.0082 | -0.0024 | CHEATING |
| 38 | `pair_feature_high_low_range_norm_zscore_x_htf_distance_to_daily_high` | +0.0292 | 2.84e-14 | -0.0082 | -0.0024 | CHEATING |
| 39 | `cross_feature_ewma_vol_20_zscore_x_htf_distance_to_daily_high` | +0.0259 | 1.64e-11 | -0.0059 | -0.0081 | CHEATING |
| 40 | `pair_feature_ewma_vol_20_zscore_x_htf_distance_to_daily_high` | +0.0259 | 1.64e-11 | -0.0059 | -0.0081 | CHEATING |

### Detailed Correlation Profiles & Hypotheses

#### 1. `cross_feature_high_low_range_norm_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: +0.0082* (p=3.2e-02, n=67,805)<br>h=-4: -0.0007 (p=8.6e-01, n=67,806)<br>h=-3: +0.0041 (p=2.8e-01, n=67,807)<br>h=-2: +0.0076* (p=4.8e-02, n=67,808)<br>h=-1: -0.0005 (p=9.0e-01, n=67,809)<br>h=+0: +0.0123** (p=1.3e-03, n=67,810)<br>h=+1: -0.0995*** (p=7.1e-149, n=67,810)<br>h=+2: -0.0919*** (p=4.2e-127, n=67,809)<br>h=+3: -0.0880*** (p=1.4e-116, n=67,808)<br>h=+4: -0.0883*** (p=2.1e-117, n=67,807)<br>h=+5: -0.0829*** (p=1.1e-103, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 2. `cross_feature_spread_proxy_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: +0.0082* (p=3.2e-02, n=67,805)<br>h=-4: -0.0007 (p=8.6e-01, n=67,806)<br>h=-3: +0.0041 (p=2.8e-01, n=67,807)<br>h=-2: +0.0076* (p=4.8e-02, n=67,808)<br>h=-1: -0.0005 (p=9.0e-01, n=67,809)<br>h=+0: +0.0123** (p=1.3e-03, n=67,810)<br>h=+1: -0.0995*** (p=7.1e-149, n=67,810)<br>h=+2: -0.0919*** (p=4.2e-127, n=67,809)<br>h=+3: -0.0880*** (p=1.4e-116, n=67,808)<br>h=+4: -0.0883*** (p=2.1e-117, n=67,807)<br>h=+5: -0.0829*** (p=1.1e-103, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 3. `pair_feature_high_low_range_norm_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: +0.0082* (p=3.2e-02, n=67,805)<br>h=-4: -0.0007 (p=8.6e-01, n=67,806)<br>h=-3: +0.0041 (p=2.8e-01, n=67,807)<br>h=-2: +0.0076* (p=4.8e-02, n=67,808)<br>h=-1: -0.0005 (p=9.0e-01, n=67,809)<br>h=+0: +0.0123** (p=1.3e-03, n=67,810)<br>h=+1: -0.0995*** (p=7.1e-149, n=67,810)<br>h=+2: -0.0919*** (p=4.2e-127, n=67,809)<br>h=+3: -0.0880*** (p=1.4e-116, n=67,808)<br>h=+4: -0.0883*** (p=2.1e-117, n=67,807)<br>h=+5: -0.0829*** (p=1.1e-103, n=67,806)

**Bias Hypothesis:** Pair feature includes HTF component 'htf_distance_to_daily_low' which may leak future daily bar information.

#### 4. `cross_feature_ewma_vol_20_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: +0.0098* (p=1.1e-02, n=67,805)<br>h=-4: +0.0073 (p=5.6e-02, n=67,806)<br>h=-3: +0.0061 (p=1.1e-01, n=67,807)<br>h=-2: +0.0064 (p=9.5e-02, n=67,808)<br>h=-1: +0.0053 (p=1.6e-01, n=67,809)<br>h=+0: +0.0070 (p=6.9e-02, n=67,810)<br>h=+1: -0.0954*** (p=9.8e-137, n=67,810)<br>h=+2: -0.0905*** (p=2.4e-123, n=67,809)<br>h=+3: -0.0887*** (p=1.9e-118, n=67,808)<br>h=+4: -0.0869*** (p=9.0e-114, n=67,807)<br>h=+5: -0.0873*** (p=7.6e-115, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 5. `pair_feature_ewma_vol_20_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: +0.0098* (p=1.1e-02, n=67,805)<br>h=-4: +0.0073 (p=5.6e-02, n=67,806)<br>h=-3: +0.0061 (p=1.1e-01, n=67,807)<br>h=-2: +0.0064 (p=9.5e-02, n=67,808)<br>h=-1: +0.0053 (p=1.6e-01, n=67,809)<br>h=+0: +0.0070 (p=6.9e-02, n=67,810)<br>h=+1: -0.0954*** (p=9.8e-137, n=67,810)<br>h=+2: -0.0905*** (p=2.4e-123, n=67,809)<br>h=+3: -0.0887*** (p=1.9e-118, n=67,808)<br>h=+4: -0.0869*** (p=9.0e-114, n=67,807)<br>h=+5: -0.0873*** (p=7.6e-115, n=67,806)

**Bias Hypothesis:** Pair feature includes HTF component 'htf_distance_to_daily_low' which may leak future daily bar information.

#### 6. `cross_feature_ret_quantile_0.2_20_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: +0.0231*** (p=1.9e-09, n=67,805)<br>h=-4: +0.0239*** (p=4.6e-10, n=67,806)<br>h=-3: +0.0212*** (p=3.6e-08, n=67,807)<br>h=-2: +0.0201*** (p=1.6e-07, n=67,808)<br>h=-1: +0.0174*** (p=5.9e-06, n=67,809)<br>h=+0: +0.0148*** (p=1.2e-04, n=67,810)<br>h=+1: +0.0913*** (p=1.7e-125, n=67,810)<br>h=+2: +0.0874*** (p=4.6e-115, n=67,809)<br>h=+3: +0.0837*** (p=8.7e-106, n=67,808)<br>h=+4: +0.0824*** (p=2.2e-102, n=67,807)<br>h=+5: +0.0814*** (p=5.9e-100, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 7. `cross_feature_high_low_range_norm_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0244*** (p=2.1e-10, n=67,805)<br>h=-4: -0.0182*** (p=2.0e-06, n=67,806)<br>h=-3: -0.0166*** (p=1.6e-05, n=67,807)<br>h=-2: -0.0224*** (p=5.3e-09, n=67,808)<br>h=-1: -0.0098* (p=1.1e-02, n=67,809)<br>h=+0: -0.0449*** (p=1.2e-31, n=67,810)<br>h=+1: +0.0900*** (p=5.4e-122, n=67,810)<br>h=+2: +0.0767*** (p=6.2e-89, n=67,809)<br>h=+3: +0.0772*** (p=4.5e-90, n=67,808)<br>h=+4: +0.0763*** (p=4.5e-88, n=67,807)<br>h=+5: +0.0791*** (p=1.8e-94, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 8. `cross_feature_spread_proxy_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0244*** (p=2.1e-10, n=67,805)<br>h=-4: -0.0182*** (p=2.0e-06, n=67,806)<br>h=-3: -0.0166*** (p=1.6e-05, n=67,807)<br>h=-2: -0.0224*** (p=5.3e-09, n=67,808)<br>h=-1: -0.0098* (p=1.1e-02, n=67,809)<br>h=+0: -0.0449*** (p=1.2e-31, n=67,810)<br>h=+1: +0.0900*** (p=5.4e-122, n=67,810)<br>h=+2: +0.0767*** (p=6.2e-89, n=67,809)<br>h=+3: +0.0772*** (p=4.5e-90, n=67,808)<br>h=+4: +0.0763*** (p=4.5e-88, n=67,807)<br>h=+5: +0.0791*** (p=1.8e-94, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 9. `pair_feature_high_low_range_norm_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0244*** (p=2.1e-10, n=67,805)<br>h=-4: -0.0182*** (p=2.0e-06, n=67,806)<br>h=-3: -0.0166*** (p=1.6e-05, n=67,807)<br>h=-2: -0.0224*** (p=5.3e-09, n=67,808)<br>h=-1: -0.0098* (p=1.1e-02, n=67,809)<br>h=+0: -0.0449*** (p=1.2e-31, n=67,810)<br>h=+1: +0.0900*** (p=5.4e-122, n=67,810)<br>h=+2: +0.0767*** (p=6.2e-89, n=67,809)<br>h=+3: +0.0772*** (p=4.5e-90, n=67,808)<br>h=+4: +0.0763*** (p=4.5e-88, n=67,807)<br>h=+5: +0.0791*** (p=1.8e-94, n=67,806)

**Bias Hypothesis:** Pair feature includes HTF component 'htf_distance_to_daily_high' which may leak future daily bar information.

#### 10. `cross_feature_true_range_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0259*** (p=1.4e-11, n=67,805)<br>h=-4: -0.0235*** (p=9.2e-10, n=67,806)<br>h=-3: -0.0230*** (p=2.0e-09, n=67,807)<br>h=-2: -0.0228*** (p=2.7e-09, n=67,808)<br>h=-1: -0.0231*** (p=1.9e-09, n=67,809)<br>h=+0: -0.0275*** (p=8.4e-13, n=67,810)<br>h=+1: +0.0870*** (p=4.4e-114, n=67,810)<br>h=+2: +0.0842*** (p=5.0e-107, n=67,809)<br>h=+3: +0.0796*** (p=8.8e-96, n=67,808)<br>h=+4: +0.0810*** (p=4.2e-99, n=67,807)<br>h=+5: +0.0843*** (p=2.9e-107, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 11. `cross_feature_ret_quantile_0.8_20_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: +0.0383*** (p=1.7e-23, n=67,805)<br>h=-4: +0.0396*** (p=6.6e-25, n=67,806)<br>h=-3: +0.0374*** (p=1.9e-22, n=67,807)<br>h=-2: +0.0375*** (p=1.6e-22, n=67,808)<br>h=-1: +0.0378*** (p=7.3e-23, n=67,809)<br>h=+0: +0.0425*** (p=2.0e-28, n=67,810)<br>h=+1: -0.0828*** (p=1.6e-103, n=67,810)<br>h=+2: -0.0790*** (p=2.9e-94, n=67,809)<br>h=+3: -0.0796*** (p=1.0e-95, n=67,808)<br>h=+4: -0.0784*** (p=5.4e-93, n=67,807)<br>h=+5: -0.0809*** (p=9.2e-99, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 12. `cross_feature_ewma_vol_20_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0294*** (p=1.8e-14, n=67,805)<br>h=-4: -0.0263*** (p=7.9e-12, n=67,806)<br>h=-3: -0.0240*** (p=4.2e-10, n=67,807)<br>h=-2: -0.0235*** (p=9.7e-10, n=67,808)<br>h=-1: -0.0230*** (p=2.2e-09, n=67,809)<br>h=+0: -0.0238*** (p=5.4e-10, n=67,810)<br>h=+1: +0.0806*** (p=4.2e-98, n=67,810)<br>h=+2: +0.0758*** (p=5.8e-87, n=67,809)<br>h=+3: +0.0754*** (p=4.1e-86, n=67,808)<br>h=+4: +0.0733*** (p=1.6e-81, n=67,807)<br>h=+5: +0.0740*** (p=6.0e-83, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 13. `pair_feature_ewma_vol_20_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0294*** (p=1.8e-14, n=67,805)<br>h=-4: -0.0263*** (p=7.9e-12, n=67,806)<br>h=-3: -0.0240*** (p=4.2e-10, n=67,807)<br>h=-2: -0.0235*** (p=9.7e-10, n=67,808)<br>h=-1: -0.0230*** (p=2.2e-09, n=67,809)<br>h=+0: -0.0238*** (p=5.4e-10, n=67,810)<br>h=+1: +0.0806*** (p=4.2e-98, n=67,810)<br>h=+2: +0.0758*** (p=5.8e-87, n=67,809)<br>h=+3: +0.0754*** (p=4.1e-86, n=67,808)<br>h=+4: +0.0733*** (p=1.6e-81, n=67,807)<br>h=+5: +0.0740*** (p=6.0e-83, n=67,806)

**Bias Hypothesis:** Pair feature includes HTF component 'htf_distance_to_daily_high' which may leak future daily bar information.

#### 14. `cross_feature_ret_quantile_0.8_20_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: +0.0121** (p=1.7e-03, n=67,805)<br>h=-4: +0.0108** (p=4.8e-03, n=67,806)<br>h=-3: +0.0119** (p=1.9e-03, n=67,807)<br>h=-2: +0.0118** (p=2.1e-03, n=67,808)<br>h=-1: +0.0120** (p=1.7e-03, n=67,809)<br>h=+0: +0.0114** (p=3.0e-03, n=67,810)<br>h=+1: +0.0805*** (p=6.2e-98, n=67,810)<br>h=+2: +0.0801*** (p=5.5e-97, n=67,809)<br>h=+3: +0.0787*** (p=1.1e-93, n=67,808)<br>h=+4: +0.0774*** (p=1.6e-90, n=67,807)<br>h=+5: +0.0792*** (p=8.5e-95, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 15. `htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: +0.0086* (p=2.6e-02, n=67,805)<br>h=-4: +0.0085* (p=2.6e-02, n=67,806)<br>h=-3: +0.0076* (p=4.7e-02, n=67,807)<br>h=-2: +0.0076* (p=4.7e-02, n=67,808)<br>h=-1: +0.0087* (p=2.4e-02, n=67,809)<br>h=+0: +0.0102** (p=7.7e-03, n=67,810)<br>h=+1: -0.0765*** (p=1.2e-88, n=67,810)<br>h=+2: -0.0754*** (p=3.7e-86, n=67,809)<br>h=+3: -0.0751*** (p=2.7e-85, n=67,808)<br>h=+4: -0.0754*** (p=4.8e-86, n=67,807)<br>h=+5: -0.0766*** (p=7.3e-89, n=67,806)

**Bias Hypothesis:** HTF distance-to-daily-high/low may use the current day's final high/low before the day is complete (peeking at the daily bar's eventual close). Daily H/L for an incomplete day should be NaN or forward-filled from the prior day, not the current unfinished bar.

#### 16. `cross_feature_cos_time_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: +0.0085* (p=2.7e-02, n=67,805)<br>h=-4: +0.0085* (p=2.7e-02, n=67,806)<br>h=-3: +0.0077* (p=4.6e-02, n=67,807)<br>h=-2: +0.0077* (p=4.6e-02, n=67,808)<br>h=-1: +0.0088* (p=2.3e-02, n=67,809)<br>h=+0: +0.0103** (p=7.5e-03, n=67,810)<br>h=+1: -0.0760*** (p=1.9e-87, n=67,810)<br>h=+2: -0.0749*** (p=5.2e-85, n=67,809)<br>h=+3: -0.0746*** (p=3.3e-84, n=67,808)<br>h=+4: -0.0749*** (p=4.8e-85, n=67,807)<br>h=+5: -0.0762*** (p=7.5e-88, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 17. `pair_feature_cos_time_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: +0.0085* (p=2.7e-02, n=67,805)<br>h=-4: +0.0085* (p=2.7e-02, n=67,806)<br>h=-3: +0.0077* (p=4.6e-02, n=67,807)<br>h=-2: +0.0077* (p=4.6e-02, n=67,808)<br>h=-1: +0.0088* (p=2.3e-02, n=67,809)<br>h=+0: +0.0103** (p=7.5e-03, n=67,810)<br>h=+1: -0.0760*** (p=1.9e-87, n=67,810)<br>h=+2: -0.0749*** (p=5.2e-85, n=67,809)<br>h=+3: -0.0746*** (p=3.3e-84, n=67,808)<br>h=+4: -0.0749*** (p=4.8e-85, n=67,807)<br>h=+5: -0.0762*** (p=7.5e-88, n=67,806)

**Bias Hypothesis:** Pair feature includes HTF component 'htf_distance_to_daily_low' which may leak future daily bar information.

#### 18. `cross_feature_ret_quantile_0.2_20_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: +0.0500*** (p=9.4e-39, n=67,805)<br>h=-4: +0.0491*** (p=2.1e-37, n=67,806)<br>h=-3: +0.0472*** (p=1.1e-34, n=67,807)<br>h=-2: +0.0461*** (p=3.2e-33, n=67,808)<br>h=-1: +0.0476*** (p=2.3e-35, n=67,809)<br>h=+0: +0.0502*** (p=4.4e-39, n=67,810)<br>h=+1: -0.0738*** (p=1.8e-82, n=67,810)<br>h=+2: -0.0711*** (p=8.9e-77, n=67,809)<br>h=+3: -0.0680*** (p=3.2e-70, n=67,808)<br>h=+4: -0.0662*** (p=9.3e-67, n=67,807)<br>h=+5: -0.0667*** (p=9.0e-68, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 19. `htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0164*** (p=2.0e-05, n=67,805)<br>h=-4: -0.0160*** (p=3.2e-05, n=67,806)<br>h=-3: -0.0148*** (p=1.1e-04, n=67,807)<br>h=-2: -0.0149*** (p=1.0e-04, n=67,808)<br>h=-1: -0.0160*** (p=3.3e-05, n=67,809)<br>h=+0: -0.0176*** (p=4.5e-06, n=67,810)<br>h=+1: +0.0734*** (p=1.2e-81, n=67,810)<br>h=+2: +0.0722*** (p=5.2e-79, n=67,809)<br>h=+3: +0.0719*** (p=2.0e-78, n=67,808)<br>h=+4: +0.0720*** (p=1.6e-78, n=67,807)<br>h=+5: +0.0731*** (p=6.1e-81, n=67,806)

**Bias Hypothesis:** HTF distance-to-daily-high/low may use the current day's final high/low before the day is complete (peeking at the daily bar's eventual close). Daily H/L for an incomplete day should be NaN or forward-filled from the prior day, not the current unfinished bar.

#### 20. `cross_feature_cos_time_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0164*** (p=1.9e-05, n=67,805)<br>h=-4: -0.0161*** (p=2.8e-05, n=67,806)<br>h=-3: -0.0150*** (p=9.3e-05, n=67,807)<br>h=-2: -0.0151*** (p=8.1e-05, n=67,808)<br>h=-1: -0.0162*** (p=2.3e-05, n=67,809)<br>h=+0: -0.0179*** (p=3.1e-06, n=67,810)<br>h=+1: +0.0726*** (p=6.4e-80, n=67,810)<br>h=+2: +0.0715*** (p=1.9e-77, n=67,809)<br>h=+3: +0.0712*** (p=5.7e-77, n=67,808)<br>h=+4: +0.0713*** (p=3.8e-77, n=67,807)<br>h=+5: +0.0724*** (p=1.4e-79, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 21. `pair_feature_cos_time_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0164*** (p=1.9e-05, n=67,805)<br>h=-4: -0.0161*** (p=2.8e-05, n=67,806)<br>h=-3: -0.0150*** (p=9.3e-05, n=67,807)<br>h=-2: -0.0151*** (p=8.1e-05, n=67,808)<br>h=-1: -0.0162*** (p=2.3e-05, n=67,809)<br>h=+0: -0.0179*** (p=3.1e-06, n=67,810)<br>h=+1: +0.0726*** (p=6.4e-80, n=67,810)<br>h=+2: +0.0715*** (p=1.9e-77, n=67,809)<br>h=+3: +0.0712*** (p=5.7e-77, n=67,808)<br>h=+4: +0.0713*** (p=3.8e-77, n=67,807)<br>h=+5: +0.0724*** (p=1.4e-79, n=67,806)

**Bias Hypothesis:** Pair feature includes HTF component 'htf_distance_to_daily_high' which may leak future daily bar information.

#### 22. `cross_feature_ret_kurt_20_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: -0.0062 (p=1.1e-01, n=67,805)<br>h=-4: -0.0073 (p=5.8e-02, n=67,806)<br>h=-3: -0.0065 (p=9.1e-02, n=67,807)<br>h=-2: -0.0068 (p=7.7e-02, n=67,808)<br>h=-1: -0.0090* (p=1.9e-02, n=67,809)<br>h=+0: -0.0114** (p=2.9e-03, n=67,810)<br>h=+1: +0.0712*** (p=6.2e-77, n=67,810)<br>h=+2: +0.0712*** (p=7.1e-77, n=67,809)<br>h=+3: +0.0702*** (p=7.5e-75, n=67,808)<br>h=+4: +0.0708*** (p=3.6e-76, n=67,807)<br>h=+5: +0.0722*** (p=4.6e-79, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 23. `cross_feature_ret_kurt_20_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: +0.0095* (p=1.4e-02, n=67,805)<br>h=-4: +0.0106** (p=5.7e-03, n=67,806)<br>h=-3: +0.0097* (p=1.2e-02, n=67,807)<br>h=-2: +0.0101** (p=8.4e-03, n=67,808)<br>h=-1: +0.0122** (p=1.5e-03, n=67,809)<br>h=+0: +0.0147*** (p=1.3e-04, n=67,810)<br>h=+1: -0.0668*** (p=7.8e-68, n=67,810)<br>h=+2: -0.0669*** (p=4.9e-68, n=67,809)<br>h=+3: -0.0659*** (p=3.3e-66, n=67,808)<br>h=+4: -0.0664*** (p=4.5e-67, n=67,807)<br>h=+5: -0.0675*** (p=3.0e-69, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 24. `cross_feature_dow_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: +0.0075 (p=5.0e-02, n=67,805)<br>h=-4: +0.0078* (p=4.3e-02, n=67,806)<br>h=-3: +0.0073 (p=5.6e-02, n=67,807)<br>h=-2: +0.0076* (p=4.7e-02, n=67,808)<br>h=-1: +0.0090* (p=1.9e-02, n=67,809)<br>h=+0: +0.0107** (p=5.3e-03, n=67,810)<br>h=+1: -0.0656*** (p=1.8e-65, n=67,810)<br>h=+2: -0.0645*** (p=1.9e-63, n=67,809)<br>h=+3: -0.0642*** (p=6.2e-63, n=67,808)<br>h=+4: -0.0644*** (p=2.7e-63, n=67,807)<br>h=+5: -0.0654*** (p=4.2e-65, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 25. `pair_feature_dow_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: +0.0075 (p=5.0e-02, n=67,805)<br>h=-4: +0.0078* (p=4.3e-02, n=67,806)<br>h=-3: +0.0073 (p=5.6e-02, n=67,807)<br>h=-2: +0.0076* (p=4.7e-02, n=67,808)<br>h=-1: +0.0090* (p=1.9e-02, n=67,809)<br>h=+0: +0.0107** (p=5.3e-03, n=67,810)<br>h=+1: -0.0656*** (p=1.8e-65, n=67,810)<br>h=+2: -0.0645*** (p=1.9e-63, n=67,809)<br>h=+3: -0.0642*** (p=6.2e-63, n=67,808)<br>h=+4: -0.0644*** (p=2.7e-63, n=67,807)<br>h=+5: -0.0654*** (p=4.2e-65, n=67,806)

**Bias Hypothesis:** Pair feature includes HTF component 'htf_distance_to_daily_low' which may leak future daily bar information.

#### 26. `cross_feature_dow_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0187*** (p=1.1e-06, n=67,805)<br>h=-4: -0.0184*** (p=1.6e-06, n=67,806)<br>h=-3: -0.0179*** (p=3.3e-06, n=67,807)<br>h=-2: -0.0182*** (p=2.2e-06, n=67,808)<br>h=-1: -0.0194*** (p=4.2e-07, n=67,809)<br>h=+0: -0.0213*** (p=3.0e-08, n=67,810)<br>h=+1: +0.0596*** (p=2.2e-54, n=67,810)<br>h=+2: +0.0583*** (p=3.2e-52, n=67,809)<br>h=+3: +0.0582*** (p=6.9e-52, n=67,808)<br>h=+4: +0.0581*** (p=7.8e-52, n=67,807)<br>h=+5: +0.0589*** (p=3.6e-53, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 27. `pair_feature_dow_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0187*** (p=1.1e-06, n=67,805)<br>h=-4: -0.0184*** (p=1.6e-06, n=67,806)<br>h=-3: -0.0179*** (p=3.3e-06, n=67,807)<br>h=-2: -0.0182*** (p=2.2e-06, n=67,808)<br>h=-1: -0.0194*** (p=4.2e-07, n=67,809)<br>h=+0: -0.0213*** (p=3.0e-08, n=67,810)<br>h=+1: +0.0596*** (p=2.2e-54, n=67,810)<br>h=+2: +0.0583*** (p=3.2e-52, n=67,809)<br>h=+3: +0.0582*** (p=6.9e-52, n=67,808)<br>h=+4: +0.0581*** (p=7.8e-52, n=67,807)<br>h=+5: +0.0589*** (p=3.6e-53, n=67,806)

**Bias Hypothesis:** Pair feature includes HTF component 'htf_distance_to_daily_high' which may leak future daily bar information.

#### 28. `cross_feature_high_low_range_norm_zscore_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: -0.0045 (p=2.4e-01, n=67,805)<br>h=-4: -0.0125** (p=1.1e-03, n=67,806)<br>h=-3: -0.0087* (p=2.4e-02, n=67,807)<br>h=-2: -0.0087* (p=2.4e-02, n=67,808)<br>h=-1: -0.0140*** (p=2.6e-04, n=67,809)<br>h=+0: +0.0042 (p=2.7e-01, n=67,810)<br>h=+1: -0.0367*** (p=1.1e-21, n=67,810)<br>h=+2: -0.0322*** (p=5.5e-17, n=67,809)<br>h=+3: -0.0244*** (p=2.1e-10, n=67,808)<br>h=+4: -0.0286*** (p=8.6e-14, n=67,807)<br>h=+5: -0.0259*** (p=1.5e-11, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 29. `cross_feature_spread_proxy_zscore_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: -0.0045 (p=2.4e-01, n=67,805)<br>h=-4: -0.0125** (p=1.1e-03, n=67,806)<br>h=-3: -0.0087* (p=2.4e-02, n=67,807)<br>h=-2: -0.0087* (p=2.4e-02, n=67,808)<br>h=-1: -0.0140*** (p=2.6e-04, n=67,809)<br>h=+0: +0.0042 (p=2.7e-01, n=67,810)<br>h=+1: -0.0367*** (p=1.1e-21, n=67,810)<br>h=+2: -0.0322*** (p=5.5e-17, n=67,809)<br>h=+3: -0.0244*** (p=2.1e-10, n=67,808)<br>h=+4: -0.0286*** (p=8.6e-14, n=67,807)<br>h=+5: -0.0259*** (p=1.5e-11, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 30. `pair_feature_high_low_range_norm_zscore_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: -0.0045 (p=2.4e-01, n=67,805)<br>h=-4: -0.0125** (p=1.1e-03, n=67,806)<br>h=-3: -0.0087* (p=2.4e-02, n=67,807)<br>h=-2: -0.0087* (p=2.4e-02, n=67,808)<br>h=-1: -0.0140*** (p=2.6e-04, n=67,809)<br>h=+0: +0.0042 (p=2.7e-01, n=67,810)<br>h=+1: -0.0367*** (p=1.1e-21, n=67,810)<br>h=+2: -0.0322*** (p=5.5e-17, n=67,809)<br>h=+3: -0.0244*** (p=2.1e-10, n=67,808)<br>h=+4: -0.0286*** (p=8.6e-14, n=67,807)<br>h=+5: -0.0259*** (p=1.5e-11, n=67,806)

**Bias Hypothesis:** Pair feature includes HTF component 'htf_distance_to_daily_low' which may leak future daily bar information.

#### 31. `cross_feature_ret_5_x_htf_volatility_ratio`

**Correlation Profile (all horizons):**
> h=-5: +0.0063 (p=1.0e-01, n=67,805)<br>h=-4: +0.1259*** (p=1.4e-237, n=67,806)<br>h=-3: +0.1786*** (p=0.0e+00, n=67,807)<br>h=-2: +0.1860*** (p=0.0e+00, n=67,807)<br>h=-1: +0.1882*** (p=0.0e+00, n=67,807)<br>h=+0: +0.2106*** (p=0.0e+00, n=67,807)<br>h=+1: +0.0344*** (p=3.0e-19, n=67,806)<br>h=+2: +0.0310*** (p=6.6e-16, n=67,805)<br>h=+3: +0.0332*** (p=5.0e-18, n=67,804)<br>h=+4: +0.0409*** (p=1.6e-26, n=67,803)<br>h=+5: +0.0205*** (p=8.8e-08, n=67,802)

**Bias Hypothesis:** Cross-feature propagates leakage from its HTF component. Check the upstream HTF feature generation for look-ahead bias.

#### 32. `pair_feature_ret_5_x_htf_volatility_ratio`

**Correlation Profile (all horizons):**
> h=-5: +0.0063 (p=1.0e-01, n=67,805)<br>h=-4: +0.1259*** (p=1.4e-237, n=67,806)<br>h=-3: +0.1786*** (p=0.0e+00, n=67,807)<br>h=-2: +0.1860*** (p=0.0e+00, n=67,808)<br>h=-1: +0.1882*** (p=0.0e+00, n=67,809)<br>h=+0: +0.2106*** (p=0.0e+00, n=67,810)<br>h=+1: +0.0344*** (p=3.0e-19, n=67,810)<br>h=+2: +0.0310*** (p=6.6e-16, n=67,809)<br>h=+3: +0.0332*** (p=5.0e-18, n=67,808)<br>h=+4: +0.0409*** (p=1.6e-26, n=67,807)<br>h=+5: +0.0205*** (p=8.8e-08, n=67,806)

**Bias Hypothesis:** Pair feature includes HTF component 'htf_volatility_ratio' which may leak future daily bar information.

#### 33. `cross_feature_ewma_vol_20_zscore_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: -0.0056 (p=1.5e-01, n=67,805)<br>h=-4: -0.0068 (p=7.6e-02, n=67,806)<br>h=-3: -0.0053 (p=1.7e-01, n=67,807)<br>h=-2: -0.0035 (p=3.7e-01, n=67,808)<br>h=-1: -0.0017 (p=6.6e-01, n=67,809)<br>h=+0: +0.0008 (p=8.4e-01, n=67,810)<br>h=+1: -0.0329*** (p=1.1e-17, n=67,810)<br>h=+2: -0.0294*** (p=1.8e-14, n=67,809)<br>h=+3: -0.0259*** (p=1.6e-11, n=67,808)<br>h=+4: -0.0263*** (p=7.2e-12, n=67,807)<br>h=+5: -0.0237*** (p=6.8e-10, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 34. `pair_feature_ewma_vol_20_zscore_x_htf_distance_to_daily_low`

**Correlation Profile (all horizons):**
> h=-5: -0.0056 (p=1.5e-01, n=67,805)<br>h=-4: -0.0068 (p=7.6e-02, n=67,806)<br>h=-3: -0.0053 (p=1.7e-01, n=67,807)<br>h=-2: -0.0035 (p=3.7e-01, n=67,808)<br>h=-1: -0.0017 (p=6.6e-01, n=67,809)<br>h=+0: +0.0008 (p=8.4e-01, n=67,810)<br>h=+1: -0.0329*** (p=1.1e-17, n=67,810)<br>h=+2: -0.0294*** (p=1.8e-14, n=67,809)<br>h=+3: -0.0259*** (p=1.6e-11, n=67,808)<br>h=+4: -0.0263*** (p=7.2e-12, n=67,807)<br>h=+5: -0.0237*** (p=6.8e-10, n=67,806)

**Bias Hypothesis:** Pair feature includes HTF component 'htf_distance_to_daily_low' which may leak future daily bar information.

#### 35. `cross_feature_ret_5_zscore_x_htf_volatility_ratio`

**Correlation Profile (all horizons):**
> h=-5: -0.0260*** (p=1.3e-11, n=67,805)<br>h=-4: +0.0971*** (p=9.4e-142, n=67,806)<br>h=-3: +0.1445*** (p=3.3e-313, n=67,807)<br>h=-2: +0.1598*** (p=0.0e+00, n=67,807)<br>h=-1: +0.1795*** (p=0.0e+00, n=67,807)<br>h=+0: +0.2165*** (p=0.0e+00, n=67,807)<br>h=+1: +0.0320*** (p=7.1e-17, n=67,806)<br>h=+2: +0.0295*** (p=1.4e-14, n=67,805)<br>h=+3: +0.0364*** (p=2.7e-21, n=67,804)<br>h=+4: +0.0423*** (p=3.0e-28, n=67,803)<br>h=+5: +0.0223*** (p=6.7e-09, n=67,802)

**Bias Hypothesis:** Cross-feature propagates leakage from its HTF component. Check the upstream HTF feature generation for look-ahead bias.

#### 36. `cross_feature_high_low_range_norm_zscore_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0082* (p=3.2e-02, n=67,805)<br>h=-4: -0.0038 (p=3.2e-01, n=67,806)<br>h=-3: -0.0057 (p=1.4e-01, n=67,807)<br>h=-2: -0.0064 (p=9.6e-02, n=67,808)<br>h=-1: -0.0024 (p=5.4e-01, n=67,809)<br>h=+0: -0.0199*** (p=2.1e-07, n=67,810)<br>h=+1: +0.0292*** (p=2.8e-14, n=67,810)<br>h=+2: +0.0242*** (p=2.8e-10, n=67,809)<br>h=+3: +0.0193*** (p=5.2e-07, n=67,808)<br>h=+4: +0.0195*** (p=3.7e-07, n=67,807)<br>h=+5: +0.0305*** (p=2.0e-15, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 37. `cross_feature_spread_proxy_zscore_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0082* (p=3.2e-02, n=67,805)<br>h=-4: -0.0038 (p=3.2e-01, n=67,806)<br>h=-3: -0.0057 (p=1.4e-01, n=67,807)<br>h=-2: -0.0064 (p=9.6e-02, n=67,808)<br>h=-1: -0.0024 (p=5.4e-01, n=67,809)<br>h=+0: -0.0199*** (p=2.1e-07, n=67,810)<br>h=+1: +0.0292*** (p=2.8e-14, n=67,810)<br>h=+2: +0.0242*** (p=2.8e-10, n=67,809)<br>h=+3: +0.0193*** (p=5.2e-07, n=67,808)<br>h=+4: +0.0195*** (p=3.7e-07, n=67,807)<br>h=+5: +0.0305*** (p=2.0e-15, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 38. `pair_feature_high_low_range_norm_zscore_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0082* (p=3.2e-02, n=67,805)<br>h=-4: -0.0038 (p=3.2e-01, n=67,806)<br>h=-3: -0.0057 (p=1.4e-01, n=67,807)<br>h=-2: -0.0064 (p=9.6e-02, n=67,808)<br>h=-1: -0.0024 (p=5.4e-01, n=67,809)<br>h=+0: -0.0199*** (p=2.1e-07, n=67,810)<br>h=+1: +0.0292*** (p=2.8e-14, n=67,810)<br>h=+2: +0.0242*** (p=2.8e-10, n=67,809)<br>h=+3: +0.0193*** (p=5.2e-07, n=67,808)<br>h=+4: +0.0195*** (p=3.7e-07, n=67,807)<br>h=+5: +0.0305*** (p=2.0e-15, n=67,806)

**Bias Hypothesis:** Pair feature includes HTF component 'htf_distance_to_daily_high' which may leak future daily bar information.

#### 39. `cross_feature_ewma_vol_20_zscore_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0059 (p=1.2e-01, n=67,805)<br>h=-4: -0.0051 (p=1.8e-01, n=67,806)<br>h=-3: -0.0058 (p=1.3e-01, n=67,807)<br>h=-2: -0.0060 (p=1.2e-01, n=67,808)<br>h=-1: -0.0081* (p=3.6e-02, n=67,809)<br>h=+0: -0.0078* (p=4.2e-02, n=67,810)<br>h=+1: +0.0259*** (p=1.6e-11, n=67,810)<br>h=+2: +0.0235*** (p=8.7e-10, n=67,809)<br>h=+3: +0.0219*** (p=1.1e-08, n=67,808)<br>h=+4: +0.0228*** (p=3.0e-09, n=67,807)<br>h=+5: +0.0198*** (p=2.6e-07, n=67,806)

**Bias Hypothesis:** Cross-feature involving distance-to-daily-high/low multiplies by a value that may reference the current day's ultimate high/low before the day ends.

#### 40. `pair_feature_ewma_vol_20_zscore_x_htf_distance_to_daily_high`

**Correlation Profile (all horizons):**
> h=-5: -0.0059 (p=1.2e-01, n=67,805)<br>h=-4: -0.0051 (p=1.8e-01, n=67,806)<br>h=-3: -0.0058 (p=1.3e-01, n=67,807)<br>h=-2: -0.0060 (p=1.2e-01, n=67,808)<br>h=-1: -0.0081* (p=3.6e-02, n=67,809)<br>h=+0: -0.0078* (p=4.2e-02, n=67,810)<br>h=+1: +0.0259*** (p=1.6e-11, n=67,810)<br>h=+2: +0.0235*** (p=8.7e-10, n=67,809)<br>h=+3: +0.0219*** (p=1.1e-08, n=67,808)<br>h=+4: +0.0228*** (p=3.0e-09, n=67,807)<br>h=+5: +0.0198*** (p=2.6e-07, n=67,806)

**Bias Hypothesis:** Pair feature includes HTF component 'htf_distance_to_daily_high' which may leak future daily bar information.

---

## ✅ Clean Features

The following 709 features passed the look-ahead bias test.

<details>
<summary>Click to expand clean feature list</summary>

- `1h_close`
- `1h_high`
- `1h_low`
- `1h_open`
- `1h_volume`
- `cross_feature_cos_time_x_htf_daily_return_1`
- `cross_feature_cos_time_x_htf_daily_trend_slope_10`
- `cross_feature_cos_time_x_htf_daily_vol_5`
- `cross_feature_cos_time_x_htf_hourly_trend_alignment`
- `cross_feature_cos_time_x_htf_volatility_ratio`
- `cross_feature_dow_x_htf_daily_return_1`
- `cross_feature_dow_x_htf_daily_trend_slope_10`
- `cross_feature_dow_x_htf_daily_vol_5`
- `cross_feature_dow_x_htf_hourly_trend_alignment`
- `cross_feature_dow_x_htf_volatility_ratio`
- `cross_feature_ewma_vol_20_regime_x_htf_daily_return_1`
- `cross_feature_ewma_vol_20_regime_x_htf_daily_trend_slope_10`
- `cross_feature_ewma_vol_20_regime_x_htf_daily_vol_5`
- `cross_feature_ewma_vol_20_regime_x_htf_distance_to_daily_high`
- `cross_feature_ewma_vol_20_regime_x_htf_distance_to_daily_low`
- `cross_feature_ewma_vol_20_regime_x_htf_hourly_trend_alignment`
- `cross_feature_ewma_vol_20_regime_x_htf_volatility_ratio`
- `cross_feature_ewma_vol_20_x_htf_daily_return_1`
- `cross_feature_ewma_vol_20_x_htf_daily_trend_slope_10`
- `cross_feature_ewma_vol_20_x_htf_daily_vol_5`
- `cross_feature_ewma_vol_20_x_htf_hourly_trend_alignment`
- `cross_feature_ewma_vol_20_x_htf_volatility_ratio`
- `cross_feature_ewma_vol_20_zscore_x_htf_daily_return_1`
- `cross_feature_ewma_vol_20_zscore_x_htf_daily_trend_slope_10`
- `cross_feature_ewma_vol_20_zscore_x_htf_daily_vol_5`
- `cross_feature_ewma_vol_20_zscore_x_htf_hourly_trend_alignment`
- `cross_feature_ewma_vol_20_zscore_x_htf_volatility_ratio`
- `cross_feature_high_low_range_norm_x_htf_daily_return_1`
- `cross_feature_high_low_range_norm_x_htf_daily_trend_slope_10`
- `cross_feature_high_low_range_norm_x_htf_daily_vol_5`
- `cross_feature_high_low_range_norm_x_htf_hourly_trend_alignment`
- `cross_feature_high_low_range_norm_x_htf_volatility_ratio`
- `cross_feature_high_low_range_norm_zscore_x_htf_daily_return_1`
- `cross_feature_high_low_range_norm_zscore_x_htf_daily_trend_slope_10`
- `cross_feature_high_low_range_norm_zscore_x_htf_daily_vol_5`
- `cross_feature_high_low_range_norm_zscore_x_htf_hourly_trend_alignment`
- `cross_feature_high_low_range_norm_zscore_x_htf_volatility_ratio`
- `cross_feature_ret_10_regime_x_htf_daily_return_1`
- `cross_feature_ret_10_regime_x_htf_daily_trend_slope_10`
- `cross_feature_ret_10_regime_x_htf_daily_vol_5`
- `cross_feature_ret_10_regime_x_htf_distance_to_daily_high`
- `cross_feature_ret_10_regime_x_htf_distance_to_daily_low`
- `cross_feature_ret_10_regime_x_htf_hourly_trend_alignment`
- `cross_feature_ret_10_regime_x_htf_volatility_ratio`
- `cross_feature_ret_10_x_htf_daily_return_1`
- `cross_feature_ret_10_x_htf_daily_trend_slope_10`
- `cross_feature_ret_10_x_htf_daily_vol_5`
- `cross_feature_ret_10_x_htf_distance_to_daily_high`
- `cross_feature_ret_10_x_htf_distance_to_daily_low`
- `cross_feature_ret_10_x_htf_hourly_trend_alignment`
- `cross_feature_ret_10_x_htf_volatility_ratio`
- `cross_feature_ret_10_zscore_x_htf_daily_return_1`
- `cross_feature_ret_10_zscore_x_htf_daily_trend_slope_10`
- `cross_feature_ret_10_zscore_x_htf_daily_vol_5`
- `cross_feature_ret_10_zscore_x_htf_distance_to_daily_high`
- `cross_feature_ret_10_zscore_x_htf_distance_to_daily_low`
- `cross_feature_ret_10_zscore_x_htf_hourly_trend_alignment`
- `cross_feature_ret_10_zscore_x_htf_volatility_ratio`
- `cross_feature_ret_1_regime_x_htf_daily_return_1`
- `cross_feature_ret_1_regime_x_htf_daily_trend_slope_10`
- `cross_feature_ret_1_regime_x_htf_daily_vol_5`
- `cross_feature_ret_1_regime_x_htf_distance_to_daily_high`
- `cross_feature_ret_1_regime_x_htf_distance_to_daily_low`
- `cross_feature_ret_1_regime_x_htf_hourly_trend_alignment`
- `cross_feature_ret_1_regime_x_htf_volatility_ratio`
- `cross_feature_ret_1_x_htf_daily_return_1`
- `cross_feature_ret_1_x_htf_daily_trend_slope_10`
- `cross_feature_ret_1_x_htf_daily_vol_5`
- `cross_feature_ret_1_x_htf_distance_to_daily_high`
- `cross_feature_ret_1_x_htf_distance_to_daily_low`
- `cross_feature_ret_1_x_htf_hourly_trend_alignment`
- `cross_feature_ret_1_x_htf_volatility_ratio`
- `cross_feature_ret_1_zscore_x_htf_daily_return_1`
- `cross_feature_ret_1_zscore_x_htf_daily_trend_slope_10`
- `cross_feature_ret_1_zscore_x_htf_daily_vol_5`
- `cross_feature_ret_1_zscore_x_htf_distance_to_daily_high`
- `cross_feature_ret_1_zscore_x_htf_distance_to_daily_low`
- `cross_feature_ret_1_zscore_x_htf_hourly_trend_alignment`
- `cross_feature_ret_1_zscore_x_htf_volatility_ratio`
- `cross_feature_ret_20_x_htf_daily_return_1`
- `cross_feature_ret_20_x_htf_daily_trend_slope_10`
- `cross_feature_ret_20_x_htf_daily_vol_5`
- `cross_feature_ret_20_x_htf_distance_to_daily_high`
- `cross_feature_ret_20_x_htf_distance_to_daily_low`
- `cross_feature_ret_20_x_htf_hourly_trend_alignment`
- `cross_feature_ret_20_x_htf_volatility_ratio`
- `cross_feature_ret_20_zscore_x_htf_daily_return_1`
- `cross_feature_ret_20_zscore_x_htf_daily_trend_slope_10`
- `cross_feature_ret_20_zscore_x_htf_daily_vol_5`
- `cross_feature_ret_20_zscore_x_htf_distance_to_daily_high`
- `cross_feature_ret_20_zscore_x_htf_distance_to_daily_low`
- `cross_feature_ret_20_zscore_x_htf_hourly_trend_alignment`
- `cross_feature_ret_20_zscore_x_htf_volatility_ratio`
- `cross_feature_ret_5_regime_x_htf_daily_return_1`
- `cross_feature_ret_5_regime_x_htf_daily_trend_slope_10`
- `cross_feature_ret_5_regime_x_htf_daily_vol_5`
- `cross_feature_ret_5_regime_x_htf_distance_to_daily_high`
- `cross_feature_ret_5_regime_x_htf_distance_to_daily_low`
- `cross_feature_ret_5_regime_x_htf_hourly_trend_alignment`
- `cross_feature_ret_5_regime_x_htf_volatility_ratio`
- `cross_feature_ret_5_x_htf_daily_return_1`
- `cross_feature_ret_5_x_htf_daily_trend_slope_10`
- `cross_feature_ret_5_x_htf_daily_vol_5`
- `cross_feature_ret_5_x_htf_distance_to_daily_high`
- `cross_feature_ret_5_x_htf_distance_to_daily_low`
- `cross_feature_ret_5_x_htf_hourly_trend_alignment`
- `cross_feature_ret_5_zscore_x_htf_daily_return_1`
- `cross_feature_ret_5_zscore_x_htf_daily_trend_slope_10`
- `cross_feature_ret_5_zscore_x_htf_daily_vol_5`
- `cross_feature_ret_5_zscore_x_htf_distance_to_daily_high`
- `cross_feature_ret_5_zscore_x_htf_distance_to_daily_low`
- `cross_feature_ret_5_zscore_x_htf_hourly_trend_alignment`
- `cross_feature_ret_acceleration_x_htf_daily_return_1`
- `cross_feature_ret_acceleration_x_htf_daily_trend_slope_10`
- `cross_feature_ret_acceleration_x_htf_daily_vol_5`
- `cross_feature_ret_acceleration_x_htf_distance_to_daily_high`
- `cross_feature_ret_acceleration_x_htf_distance_to_daily_low`
- `cross_feature_ret_acceleration_x_htf_hourly_trend_alignment`
- `cross_feature_ret_acceleration_x_htf_volatility_ratio`
- `cross_feature_ret_kurt_20_x_htf_daily_return_1`
- `cross_feature_ret_kurt_20_x_htf_daily_trend_slope_10`
- `cross_feature_ret_kurt_20_x_htf_daily_vol_5`
- `cross_feature_ret_kurt_20_x_htf_hourly_trend_alignment`
- `cross_feature_ret_kurt_20_x_htf_volatility_ratio`
- `cross_feature_ret_quantile_0.2_20_x_htf_daily_return_1`
- `cross_feature_ret_quantile_0.2_20_x_htf_daily_trend_slope_10`
- `cross_feature_ret_quantile_0.2_20_x_htf_daily_vol_5`
- `cross_feature_ret_quantile_0.2_20_x_htf_hourly_trend_alignment`
- `cross_feature_ret_quantile_0.2_20_x_htf_volatility_ratio`
- `cross_feature_ret_quantile_0.5_20_x_htf_daily_return_1`
- `cross_feature_ret_quantile_0.5_20_x_htf_daily_trend_slope_10`
- `cross_feature_ret_quantile_0.5_20_x_htf_daily_vol_5`
- `cross_feature_ret_quantile_0.5_20_x_htf_distance_to_daily_high`
- `cross_feature_ret_quantile_0.5_20_x_htf_distance_to_daily_low`
- `cross_feature_ret_quantile_0.5_20_x_htf_hourly_trend_alignment`
- `cross_feature_ret_quantile_0.5_20_x_htf_volatility_ratio`
- `cross_feature_ret_quantile_0.8_20_x_htf_daily_return_1`
- `cross_feature_ret_quantile_0.8_20_x_htf_daily_trend_slope_10`
- `cross_feature_ret_quantile_0.8_20_x_htf_daily_vol_5`
- `cross_feature_ret_quantile_0.8_20_x_htf_hourly_trend_alignment`
- `cross_feature_ret_quantile_0.8_20_x_htf_volatility_ratio`
- `cross_feature_ret_skew_20_x_htf_daily_return_1`
- `cross_feature_ret_skew_20_x_htf_daily_trend_slope_10`
- `cross_feature_ret_skew_20_x_htf_daily_vol_5`
- `cross_feature_ret_skew_20_x_htf_distance_to_daily_high`
- `cross_feature_ret_skew_20_x_htf_distance_to_daily_low`
- `cross_feature_ret_skew_20_x_htf_hourly_trend_alignment`
- `cross_feature_ret_skew_20_x_htf_volatility_ratio`
- `cross_feature_sin_time_x_htf_daily_return_1`
- `cross_feature_sin_time_x_htf_daily_trend_slope_10`
- `cross_feature_sin_time_x_htf_daily_vol_5`
- `cross_feature_sin_time_x_htf_distance_to_daily_high`
- `cross_feature_sin_time_x_htf_distance_to_daily_low`
- `cross_feature_sin_time_x_htf_hourly_trend_alignment`
- `cross_feature_sin_time_x_htf_volatility_ratio`
- `cross_feature_spread_proxy_regime_x_htf_daily_return_1`
- `cross_feature_spread_proxy_regime_x_htf_daily_trend_slope_10`
- `cross_feature_spread_proxy_regime_x_htf_daily_vol_5`
- `cross_feature_spread_proxy_regime_x_htf_distance_to_daily_high`
- `cross_feature_spread_proxy_regime_x_htf_distance_to_daily_low`
- `cross_feature_spread_proxy_regime_x_htf_hourly_trend_alignment`
- `cross_feature_spread_proxy_regime_x_htf_volatility_ratio`
- `cross_feature_spread_proxy_x_htf_daily_return_1`
- `cross_feature_spread_proxy_x_htf_daily_trend_slope_10`
- `cross_feature_spread_proxy_x_htf_daily_vol_5`
- `cross_feature_spread_proxy_x_htf_hourly_trend_alignment`
- `cross_feature_spread_proxy_x_htf_volatility_ratio`
- `cross_feature_spread_proxy_zscore_x_htf_daily_return_1`
- `cross_feature_spread_proxy_zscore_x_htf_daily_trend_slope_10`
- `cross_feature_spread_proxy_zscore_x_htf_daily_vol_5`
- `cross_feature_spread_proxy_zscore_x_htf_hourly_trend_alignment`
- `cross_feature_spread_proxy_zscore_x_htf_volatility_ratio`
- `cross_feature_true_range_x_htf_daily_return_1`
- `cross_feature_true_range_x_htf_daily_trend_slope_10`
- `cross_feature_true_range_x_htf_daily_vol_5`
- `daily_close`
- `daily_high`
- `daily_low`
- `daily_open`
- `daily_vol_5`
- `daily_volume`
- `feature_cos_time`
- `feature_dow`
- `feature_ewma_vol_20`
- `feature_ewma_vol_20_regime`
- `feature_ewma_vol_20_zscore`
- `feature_high_low_range_norm`
- `feature_high_low_range_norm_zscore`
- `feature_ret_1`
- `feature_ret_10`
- `feature_ret_10_regime`
- `feature_ret_10_zscore`
- `feature_ret_1_regime`
- `feature_ret_1_zscore`
- `feature_ret_20`
- `feature_ret_20_zscore`
- `feature_ret_5`
- `feature_ret_5_regime`
- `feature_ret_5_zscore`
- `feature_ret_acceleration`
- `feature_ret_kurt_20`
- `feature_ret_quantile_0.2_20`
- `feature_ret_quantile_0.5_20`
- `feature_ret_quantile_0.8_20`
- `feature_ret_skew_20`
- `feature_sin_time`
- `feature_spread_proxy`
- `feature_spread_proxy_regime`
- `feature_spread_proxy_zscore`
- `feature_true_range`
- `feature_true_range_zscore`
- `feature_vwap_deviation`
- `htf_daily_return_1`
- `htf_daily_trend_slope_10`
- `htf_daily_vol_5`
- `htf_hourly_trend_alignment`
- `htf_volatility_ratio`
- `pair_feature_cos_time_x_feature_dow`
- `pair_feature_cos_time_x_feature_ewma_vol_20`
- `pair_feature_cos_time_x_feature_ewma_vol_20_regime`
- `pair_feature_cos_time_x_feature_ewma_vol_20_zscore`
- `pair_feature_cos_time_x_feature_high_low_range_norm`
- `pair_feature_cos_time_x_feature_high_low_range_norm_zscore`
- `pair_feature_cos_time_x_feature_ret_1`
- `pair_feature_cos_time_x_feature_ret_10`
- `pair_feature_cos_time_x_feature_ret_10_regime`
- `pair_feature_cos_time_x_feature_ret_10_zscore`
- `pair_feature_cos_time_x_feature_ret_1_regime`
- `pair_feature_cos_time_x_feature_ret_1_zscore`
- `pair_feature_cos_time_x_feature_ret_20`
- `pair_feature_cos_time_x_feature_ret_20_zscore`
- `pair_feature_cos_time_x_feature_ret_5`
- `pair_feature_cos_time_x_feature_ret_5_regime`
- `pair_feature_cos_time_x_feature_ret_5_zscore`
- `pair_feature_cos_time_x_feature_ret_acceleration`
- `pair_feature_cos_time_x_feature_ret_kurt_20`
- `pair_feature_cos_time_x_feature_ret_quantile_0.2_20`
- `pair_feature_cos_time_x_feature_ret_quantile_0.5_20`
- `pair_feature_cos_time_x_feature_ret_quantile_0.8_20`
- `pair_feature_cos_time_x_feature_ret_skew_20`
- `pair_feature_cos_time_x_feature_sin_time`
- `pair_feature_cos_time_x_feature_spread_proxy`
- `pair_feature_cos_time_x_feature_spread_proxy_regime`
- `pair_feature_cos_time_x_feature_spread_proxy_zscore`
- `pair_feature_cos_time_x_feature_true_range`
- `pair_feature_cos_time_x_feature_true_range_zscore`
- `pair_feature_cos_time_x_feature_vwap_deviation`
- `pair_feature_cos_time_x_htf_daily_return_1`
- `pair_feature_cos_time_x_htf_daily_trend_slope_10`
- `pair_feature_cos_time_x_htf_daily_vol_5`
- `pair_feature_cos_time_x_htf_hourly_trend_alignment`
- `pair_feature_cos_time_x_htf_volatility_ratio`
- `pair_feature_dow_x_feature_ewma_vol_20`
- `pair_feature_dow_x_feature_ewma_vol_20_regime`
- `pair_feature_dow_x_feature_ewma_vol_20_zscore`
- `pair_feature_dow_x_feature_high_low_range_norm`
- `pair_feature_dow_x_feature_high_low_range_norm_zscore`
- `pair_feature_dow_x_feature_ret_1`
- `pair_feature_dow_x_feature_ret_10`
- `pair_feature_dow_x_feature_ret_10_regime`
- `pair_feature_dow_x_feature_ret_10_zscore`
- `pair_feature_dow_x_feature_ret_1_regime`
- `pair_feature_dow_x_feature_ret_1_zscore`
- `pair_feature_dow_x_feature_ret_20`
- `pair_feature_dow_x_feature_ret_20_zscore`
- `pair_feature_dow_x_feature_ret_5`
- `pair_feature_dow_x_feature_ret_5_regime`
- `pair_feature_dow_x_feature_ret_5_zscore`
- `pair_feature_dow_x_feature_ret_acceleration`
- `pair_feature_dow_x_feature_ret_kurt_20`
- `pair_feature_dow_x_feature_ret_quantile_0.2_20`
- `pair_feature_dow_x_feature_ret_quantile_0.5_20`
- `pair_feature_dow_x_feature_ret_quantile_0.8_20`
- `pair_feature_dow_x_feature_ret_skew_20`
- `pair_feature_dow_x_feature_sin_time`
- `pair_feature_dow_x_feature_spread_proxy`
- `pair_feature_dow_x_feature_spread_proxy_regime`
- `pair_feature_dow_x_feature_spread_proxy_zscore`
- `pair_feature_dow_x_feature_true_range`
- `pair_feature_dow_x_feature_true_range_zscore`
- `pair_feature_dow_x_feature_vwap_deviation`
- `pair_feature_dow_x_htf_daily_return_1`
- `pair_feature_dow_x_htf_daily_trend_slope_10`
- `pair_feature_dow_x_htf_daily_vol_5`
- `pair_feature_dow_x_htf_hourly_trend_alignment`
- `pair_feature_dow_x_htf_volatility_ratio`
- `pair_feature_ewma_vol_20_regime_x_feature_ewma_vol_20_zscore`
- `pair_feature_ewma_vol_20_regime_x_feature_high_low_range_norm`
- `pair_feature_ewma_vol_20_regime_x_feature_high_low_range_norm_zscore`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_1`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_10`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_10_regime`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_10_zscore`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_1_regime`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_1_zscore`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_20`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_20_zscore`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_5`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_5_regime`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_5_zscore`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_acceleration`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_kurt_20`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_quantile_0.2_20`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_quantile_0.5_20`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_quantile_0.8_20`
- `pair_feature_ewma_vol_20_regime_x_feature_ret_skew_20`
- `pair_feature_ewma_vol_20_regime_x_feature_sin_time`
- `pair_feature_ewma_vol_20_regime_x_feature_spread_proxy`
- `pair_feature_ewma_vol_20_regime_x_feature_spread_proxy_regime`
- `pair_feature_ewma_vol_20_regime_x_feature_spread_proxy_zscore`
- `pair_feature_ewma_vol_20_regime_x_feature_true_range`
- `pair_feature_ewma_vol_20_regime_x_feature_true_range_zscore`
- `pair_feature_ewma_vol_20_regime_x_feature_vwap_deviation`
- `pair_feature_ewma_vol_20_regime_x_htf_daily_return_1`
- `pair_feature_ewma_vol_20_regime_x_htf_daily_trend_slope_10`
- `pair_feature_ewma_vol_20_regime_x_htf_daily_vol_5`
- `pair_feature_ewma_vol_20_regime_x_htf_distance_to_daily_high`
- `pair_feature_ewma_vol_20_regime_x_htf_distance_to_daily_low`
- `pair_feature_ewma_vol_20_regime_x_htf_hourly_trend_alignment`
- `pair_feature_ewma_vol_20_regime_x_htf_volatility_ratio`
- `pair_feature_ewma_vol_20_x_feature_ewma_vol_20_regime`
- `pair_feature_ewma_vol_20_x_feature_ewma_vol_20_zscore`
- `pair_feature_ewma_vol_20_x_feature_high_low_range_norm`
- `pair_feature_ewma_vol_20_x_feature_high_low_range_norm_zscore`
- `pair_feature_ewma_vol_20_x_feature_ret_1`
- `pair_feature_ewma_vol_20_x_feature_ret_10`
- `pair_feature_ewma_vol_20_x_feature_ret_10_regime`
- `pair_feature_ewma_vol_20_x_feature_ret_10_zscore`
- `pair_feature_ewma_vol_20_x_feature_ret_1_regime`
- `pair_feature_ewma_vol_20_x_feature_ret_1_zscore`
- `pair_feature_ewma_vol_20_x_feature_ret_20`
- `pair_feature_ewma_vol_20_x_feature_ret_20_zscore`
- `pair_feature_ewma_vol_20_x_feature_ret_5`
- `pair_feature_ewma_vol_20_x_feature_ret_5_regime`
- `pair_feature_ewma_vol_20_x_feature_ret_5_zscore`
- `pair_feature_ewma_vol_20_x_feature_ret_acceleration`
- `pair_feature_ewma_vol_20_x_feature_ret_kurt_20`
- `pair_feature_ewma_vol_20_x_feature_ret_quantile_0.2_20`
- `pair_feature_ewma_vol_20_x_feature_ret_quantile_0.5_20`
- `pair_feature_ewma_vol_20_x_feature_ret_quantile_0.8_20`
- `pair_feature_ewma_vol_20_x_feature_ret_skew_20`
- `pair_feature_ewma_vol_20_x_feature_sin_time`
- `pair_feature_ewma_vol_20_x_feature_spread_proxy`
- `pair_feature_ewma_vol_20_x_feature_spread_proxy_regime`
- `pair_feature_ewma_vol_20_x_feature_spread_proxy_zscore`
- `pair_feature_ewma_vol_20_x_feature_true_range`
- `pair_feature_ewma_vol_20_x_feature_true_range_zscore`
- `pair_feature_ewma_vol_20_x_feature_vwap_deviation`
- `pair_feature_ewma_vol_20_x_htf_daily_return_1`
- `pair_feature_ewma_vol_20_x_htf_daily_trend_slope_10`
- `pair_feature_ewma_vol_20_x_htf_daily_vol_5`
- `pair_feature_ewma_vol_20_x_htf_hourly_trend_alignment`
- `pair_feature_ewma_vol_20_x_htf_volatility_ratio`
- `pair_feature_ewma_vol_20_zscore_x_feature_high_low_range_norm`
- `pair_feature_ewma_vol_20_zscore_x_feature_high_low_range_norm_zscore`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_1`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_10`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_10_regime`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_10_zscore`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_1_regime`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_1_zscore`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_20`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_20_zscore`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_5`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_5_regime`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_5_zscore`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_acceleration`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_kurt_20`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_quantile_0.2_20`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_quantile_0.5_20`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_quantile_0.8_20`
- `pair_feature_ewma_vol_20_zscore_x_feature_ret_skew_20`
- `pair_feature_ewma_vol_20_zscore_x_feature_sin_time`
- `pair_feature_ewma_vol_20_zscore_x_feature_spread_proxy`
- `pair_feature_ewma_vol_20_zscore_x_feature_spread_proxy_regime`
- `pair_feature_ewma_vol_20_zscore_x_feature_spread_proxy_zscore`
- `pair_feature_ewma_vol_20_zscore_x_feature_true_range`
- `pair_feature_ewma_vol_20_zscore_x_feature_true_range_zscore`
- `pair_feature_ewma_vol_20_zscore_x_feature_vwap_deviation`
- `pair_feature_ewma_vol_20_zscore_x_htf_daily_return_1`
- `pair_feature_ewma_vol_20_zscore_x_htf_daily_trend_slope_10`
- `pair_feature_ewma_vol_20_zscore_x_htf_daily_vol_5`
- `pair_feature_ewma_vol_20_zscore_x_htf_hourly_trend_alignment`
- `pair_feature_ewma_vol_20_zscore_x_htf_volatility_ratio`
- `pair_feature_high_low_range_norm_x_feature_high_low_range_norm_zscore`
- `pair_feature_high_low_range_norm_x_feature_ret_1`
- `pair_feature_high_low_range_norm_x_feature_ret_10`
- `pair_feature_high_low_range_norm_x_feature_ret_10_regime`
- `pair_feature_high_low_range_norm_x_feature_ret_10_zscore`
- `pair_feature_high_low_range_norm_x_feature_ret_1_regime`
- `pair_feature_high_low_range_norm_x_feature_ret_1_zscore`
- `pair_feature_high_low_range_norm_x_feature_ret_20`
- `pair_feature_high_low_range_norm_x_feature_ret_20_zscore`
- `pair_feature_high_low_range_norm_x_feature_ret_5`
- `pair_feature_high_low_range_norm_x_feature_ret_5_regime`
- `pair_feature_high_low_range_norm_x_feature_ret_5_zscore`
- `pair_feature_high_low_range_norm_x_feature_ret_acceleration`
- `pair_feature_high_low_range_norm_x_feature_ret_kurt_20`
- `pair_feature_high_low_range_norm_x_feature_ret_quantile_0.2_20`
- `pair_feature_high_low_range_norm_x_feature_ret_quantile_0.5_20`
- `pair_feature_high_low_range_norm_x_feature_ret_quantile_0.8_20`
- `pair_feature_high_low_range_norm_x_feature_ret_skew_20`
- `pair_feature_high_low_range_norm_x_feature_sin_time`
- `pair_feature_high_low_range_norm_x_feature_spread_proxy`
- `pair_feature_high_low_range_norm_x_feature_spread_proxy_regime`
- `pair_feature_high_low_range_norm_x_feature_spread_proxy_zscore`
- `pair_feature_high_low_range_norm_x_feature_true_range`
- `pair_feature_high_low_range_norm_x_feature_true_range_zscore`
- `pair_feature_high_low_range_norm_x_feature_vwap_deviation`
- `pair_feature_high_low_range_norm_x_htf_daily_return_1`
- `pair_feature_high_low_range_norm_x_htf_daily_trend_slope_10`
- `pair_feature_high_low_range_norm_x_htf_daily_vol_5`
- `pair_feature_high_low_range_norm_x_htf_hourly_trend_alignment`
- `pair_feature_high_low_range_norm_x_htf_volatility_ratio`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_1`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_10`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_10_regime`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_10_zscore`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_1_regime`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_1_zscore`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_20`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_20_zscore`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_5`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_5_regime`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_5_zscore`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_acceleration`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_kurt_20`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_quantile_0.2_20`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_quantile_0.5_20`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_quantile_0.8_20`
- `pair_feature_high_low_range_norm_zscore_x_feature_ret_skew_20`
- `pair_feature_high_low_range_norm_zscore_x_feature_sin_time`
- `pair_feature_high_low_range_norm_zscore_x_feature_spread_proxy`
- `pair_feature_high_low_range_norm_zscore_x_feature_spread_proxy_regime`
- `pair_feature_high_low_range_norm_zscore_x_feature_spread_proxy_zscore`
- `pair_feature_high_low_range_norm_zscore_x_feature_true_range`
- `pair_feature_high_low_range_norm_zscore_x_feature_true_range_zscore`
- `pair_feature_high_low_range_norm_zscore_x_feature_vwap_deviation`
- `pair_feature_high_low_range_norm_zscore_x_htf_daily_return_1`
- `pair_feature_high_low_range_norm_zscore_x_htf_daily_trend_slope_10`
- `pair_feature_high_low_range_norm_zscore_x_htf_daily_vol_5`
- `pair_feature_high_low_range_norm_zscore_x_htf_hourly_trend_alignment`
- `pair_feature_high_low_range_norm_zscore_x_htf_volatility_ratio`
- `pair_feature_ret_10_regime_x_feature_ret_10_zscore`
- `pair_feature_ret_10_regime_x_feature_ret_1_regime`
- `pair_feature_ret_10_regime_x_feature_ret_1_zscore`
- `pair_feature_ret_10_regime_x_feature_ret_20`
- `pair_feature_ret_10_regime_x_feature_ret_20_zscore`
- `pair_feature_ret_10_regime_x_feature_ret_5`
- `pair_feature_ret_10_regime_x_feature_ret_5_regime`
- `pair_feature_ret_10_regime_x_feature_ret_5_zscore`
- `pair_feature_ret_10_regime_x_feature_ret_acceleration`
- `pair_feature_ret_10_regime_x_feature_ret_kurt_20`
- `pair_feature_ret_10_regime_x_feature_ret_quantile_0.2_20`
- `pair_feature_ret_10_regime_x_feature_ret_quantile_0.5_20`
- `pair_feature_ret_10_regime_x_feature_ret_quantile_0.8_20`
- `pair_feature_ret_10_regime_x_feature_ret_skew_20`
- `pair_feature_ret_10_regime_x_feature_sin_time`
- `pair_feature_ret_10_regime_x_feature_spread_proxy`
- `pair_feature_ret_10_regime_x_feature_spread_proxy_regime`
- `pair_feature_ret_10_regime_x_feature_spread_proxy_zscore`
- `pair_feature_ret_10_regime_x_feature_true_range`
- `pair_feature_ret_10_regime_x_feature_true_range_zscore`
- `pair_feature_ret_10_regime_x_feature_vwap_deviation`
- `pair_feature_ret_10_regime_x_htf_daily_return_1`
- `pair_feature_ret_10_regime_x_htf_daily_trend_slope_10`
- `pair_feature_ret_10_regime_x_htf_daily_vol_5`
- `pair_feature_ret_10_regime_x_htf_distance_to_daily_high`
- `pair_feature_ret_10_regime_x_htf_distance_to_daily_low`
- `pair_feature_ret_10_regime_x_htf_hourly_trend_alignment`
- `pair_feature_ret_10_regime_x_htf_volatility_ratio`
- `pair_feature_ret_10_x_feature_ret_10_regime`
- `pair_feature_ret_10_x_feature_ret_10_zscore`
- `pair_feature_ret_10_x_feature_ret_1_regime`
- `pair_feature_ret_10_x_feature_ret_1_zscore`
- `pair_feature_ret_10_x_feature_ret_20`
- `pair_feature_ret_10_x_feature_ret_20_zscore`
- `pair_feature_ret_10_x_feature_ret_5`
- `pair_feature_ret_10_x_feature_ret_5_regime`
- `pair_feature_ret_10_x_feature_ret_5_zscore`
- `pair_feature_ret_10_x_feature_ret_acceleration`
- `pair_feature_ret_10_x_feature_ret_kurt_20`
- `pair_feature_ret_10_x_feature_ret_quantile_0.2_20`
- `pair_feature_ret_10_x_feature_ret_quantile_0.5_20`
- `pair_feature_ret_10_x_feature_ret_quantile_0.8_20`
- `pair_feature_ret_10_x_feature_ret_skew_20`
- `pair_feature_ret_10_x_feature_sin_time`
- `pair_feature_ret_10_x_feature_spread_proxy`
- `pair_feature_ret_10_x_feature_spread_proxy_regime`
- `pair_feature_ret_10_x_feature_spread_proxy_zscore`
- `pair_feature_ret_10_x_feature_true_range`
- `pair_feature_ret_10_x_feature_true_range_zscore`
- `pair_feature_ret_10_x_feature_vwap_deviation`
- `pair_feature_ret_10_x_htf_daily_return_1`
- `pair_feature_ret_10_x_htf_daily_trend_slope_10`
- `pair_feature_ret_10_x_htf_daily_vol_5`
- `pair_feature_ret_10_x_htf_distance_to_daily_high`
- `pair_feature_ret_10_x_htf_distance_to_daily_low`
- `pair_feature_ret_10_x_htf_hourly_trend_alignment`
- `pair_feature_ret_10_x_htf_volatility_ratio`
- `pair_feature_ret_10_zscore_x_feature_ret_1_regime`
- `pair_feature_ret_10_zscore_x_feature_ret_1_zscore`
- `pair_feature_ret_10_zscore_x_feature_ret_20`
- `pair_feature_ret_10_zscore_x_feature_ret_20_zscore`
- `pair_feature_ret_10_zscore_x_feature_ret_5`
- `pair_feature_ret_10_zscore_x_feature_ret_5_regime`
- `pair_feature_ret_10_zscore_x_feature_ret_5_zscore`
- `pair_feature_ret_10_zscore_x_feature_ret_acceleration`
- `pair_feature_ret_10_zscore_x_feature_ret_kurt_20`
- `pair_feature_ret_10_zscore_x_feature_ret_quantile_0.2_20`
- `pair_feature_ret_10_zscore_x_feature_ret_quantile_0.5_20`
- `pair_feature_ret_10_zscore_x_feature_ret_quantile_0.8_20`
- `pair_feature_ret_10_zscore_x_feature_ret_skew_20`
- `pair_feature_ret_10_zscore_x_feature_sin_time`
- `pair_feature_ret_10_zscore_x_feature_spread_proxy`
- `pair_feature_ret_10_zscore_x_feature_spread_proxy_regime`
- `pair_feature_ret_10_zscore_x_feature_spread_proxy_zscore`
- `pair_feature_ret_10_zscore_x_feature_true_range`
- `pair_feature_ret_10_zscore_x_feature_true_range_zscore`
- `pair_feature_ret_10_zscore_x_feature_vwap_deviation`
- `pair_feature_ret_10_zscore_x_htf_daily_return_1`
- `pair_feature_ret_10_zscore_x_htf_daily_trend_slope_10`
- `pair_feature_ret_10_zscore_x_htf_daily_vol_5`
- `pair_feature_ret_10_zscore_x_htf_distance_to_daily_high`
- `pair_feature_ret_10_zscore_x_htf_distance_to_daily_low`
- `pair_feature_ret_10_zscore_x_htf_hourly_trend_alignment`
- `pair_feature_ret_10_zscore_x_htf_volatility_ratio`
- `pair_feature_ret_1_regime_x_feature_ret_1_zscore`
- `pair_feature_ret_1_regime_x_feature_ret_20`
- `pair_feature_ret_1_regime_x_feature_ret_20_zscore`
- `pair_feature_ret_1_regime_x_feature_ret_5`
- `pair_feature_ret_1_regime_x_feature_ret_5_regime`
- `pair_feature_ret_1_regime_x_feature_ret_5_zscore`
- `pair_feature_ret_1_regime_x_feature_ret_acceleration`
- `pair_feature_ret_1_regime_x_feature_ret_kurt_20`
- `pair_feature_ret_1_regime_x_feature_ret_quantile_0.2_20`
- `pair_feature_ret_1_regime_x_feature_ret_quantile_0.5_20`
- `pair_feature_ret_1_regime_x_feature_ret_quantile_0.8_20`
- `pair_feature_ret_1_regime_x_feature_ret_skew_20`
- `pair_feature_ret_1_regime_x_feature_sin_time`
- `pair_feature_ret_1_regime_x_feature_spread_proxy`
- `pair_feature_ret_1_regime_x_feature_spread_proxy_regime`
- `pair_feature_ret_1_regime_x_feature_spread_proxy_zscore`
- `pair_feature_ret_1_regime_x_feature_true_range`
- `pair_feature_ret_1_regime_x_feature_true_range_zscore`
- `pair_feature_ret_1_regime_x_feature_vwap_deviation`
- `pair_feature_ret_1_regime_x_htf_daily_return_1`
- `pair_feature_ret_1_regime_x_htf_daily_trend_slope_10`
- `pair_feature_ret_1_regime_x_htf_daily_vol_5`
- `pair_feature_ret_1_regime_x_htf_distance_to_daily_high`
- `pair_feature_ret_1_regime_x_htf_distance_to_daily_low`
- `pair_feature_ret_1_regime_x_htf_hourly_trend_alignment`
- `pair_feature_ret_1_regime_x_htf_volatility_ratio`
- `pair_feature_ret_1_x_feature_ret_10`
- `pair_feature_ret_1_x_feature_ret_10_regime`
- `pair_feature_ret_1_x_feature_ret_10_zscore`
- `pair_feature_ret_1_x_feature_ret_1_regime`
- `pair_feature_ret_1_x_feature_ret_1_zscore`
- `pair_feature_ret_1_x_feature_ret_20`
- `pair_feature_ret_1_x_feature_ret_20_zscore`
- `pair_feature_ret_1_x_feature_ret_5`
- `pair_feature_ret_1_x_feature_ret_5_regime`
- `pair_feature_ret_1_x_feature_ret_5_zscore`
- `pair_feature_ret_1_x_feature_ret_acceleration`
- `pair_feature_ret_1_x_feature_ret_kurt_20`
- `pair_feature_ret_1_x_feature_ret_quantile_0.2_20`
- `pair_feature_ret_1_x_feature_ret_quantile_0.5_20`
- `pair_feature_ret_1_x_feature_ret_quantile_0.8_20`
- `pair_feature_ret_1_x_feature_ret_skew_20`
- `pair_feature_ret_1_x_feature_sin_time`
- `pair_feature_ret_1_x_feature_spread_proxy`
- `pair_feature_ret_1_x_feature_spread_proxy_regime`
- `pair_feature_ret_1_x_feature_spread_proxy_zscore`
- `pair_feature_ret_1_x_feature_true_range`
- `pair_feature_ret_1_x_feature_true_range_zscore`
- `pair_feature_ret_1_x_feature_vwap_deviation`
- `pair_feature_ret_1_x_htf_daily_return_1`
- `pair_feature_ret_1_x_htf_daily_trend_slope_10`
- `pair_feature_ret_1_x_htf_daily_vol_5`
- `pair_feature_ret_1_x_htf_distance_to_daily_high`
- `pair_feature_ret_1_x_htf_distance_to_daily_low`
- `pair_feature_ret_1_x_htf_hourly_trend_alignment`
- `pair_feature_ret_1_x_htf_volatility_ratio`
- `pair_feature_ret_1_zscore_x_feature_ret_20`
- `pair_feature_ret_1_zscore_x_feature_ret_20_zscore`
- `pair_feature_ret_1_zscore_x_feature_ret_5`
- `pair_feature_ret_1_zscore_x_feature_ret_5_regime`
- `pair_feature_ret_1_zscore_x_feature_ret_5_zscore`
- `pair_feature_ret_1_zscore_x_feature_ret_acceleration`
- `pair_feature_ret_1_zscore_x_feature_ret_kurt_20`
- `pair_feature_ret_1_zscore_x_feature_ret_quantile_0.2_20`
- `pair_feature_ret_1_zscore_x_feature_ret_quantile_0.5_20`
- `pair_feature_ret_1_zscore_x_feature_ret_quantile_0.8_20`
- `pair_feature_ret_1_zscore_x_feature_ret_skew_20`
- `pair_feature_ret_1_zscore_x_feature_sin_time`
- `pair_feature_ret_1_zscore_x_feature_spread_proxy`
- `pair_feature_ret_1_zscore_x_feature_spread_proxy_regime`
- `pair_feature_ret_1_zscore_x_feature_spread_proxy_zscore`
- `pair_feature_ret_1_zscore_x_feature_true_range`
- `pair_feature_ret_1_zscore_x_feature_true_range_zscore`
- `pair_feature_ret_1_zscore_x_feature_vwap_deviation`
- `pair_feature_ret_1_zscore_x_htf_daily_return_1`
- `pair_feature_ret_1_zscore_x_htf_daily_trend_slope_10`
- `pair_feature_ret_1_zscore_x_htf_daily_vol_5`
- `pair_feature_ret_1_zscore_x_htf_distance_to_daily_high`
- `pair_feature_ret_1_zscore_x_htf_distance_to_daily_low`
- `pair_feature_ret_1_zscore_x_htf_hourly_trend_alignment`
- `pair_feature_ret_1_zscore_x_htf_volatility_ratio`
- `pair_feature_ret_20_x_feature_ret_20_zscore`
- `pair_feature_ret_20_x_feature_ret_5`
- `pair_feature_ret_20_x_feature_ret_5_regime`
- `pair_feature_ret_20_x_feature_ret_5_zscore`
- `pair_feature_ret_20_x_feature_ret_acceleration`
- `pair_feature_ret_20_x_feature_ret_kurt_20`
- `pair_feature_ret_20_x_feature_ret_quantile_0.2_20`
- `pair_feature_ret_20_x_feature_ret_quantile_0.5_20`
- `pair_feature_ret_20_x_feature_ret_quantile_0.8_20`
- `pair_feature_ret_20_x_feature_ret_skew_20`
- `pair_feature_ret_20_x_feature_sin_time`
- `pair_feature_ret_20_x_feature_spread_proxy`
- `pair_feature_ret_20_x_feature_spread_proxy_regime`
- `pair_feature_ret_20_x_feature_spread_proxy_zscore`
- `pair_feature_ret_20_x_feature_true_range`
- `pair_feature_ret_20_x_feature_true_range_zscore`
- `pair_feature_ret_20_x_feature_vwap_deviation`
- `pair_feature_ret_20_x_htf_daily_return_1`
- `pair_feature_ret_20_x_htf_daily_trend_slope_10`
- `pair_feature_ret_20_x_htf_daily_vol_5`
- `pair_feature_ret_20_x_htf_distance_to_daily_high`
- `pair_feature_ret_20_x_htf_distance_to_daily_low`
- `pair_feature_ret_20_x_htf_hourly_trend_alignment`
- `pair_feature_ret_20_x_htf_volatility_ratio`
- `pair_feature_ret_20_zscore_x_feature_ret_5`
- `pair_feature_ret_20_zscore_x_feature_ret_5_regime`
- `pair_feature_ret_20_zscore_x_feature_ret_5_zscore`
- `pair_feature_ret_20_zscore_x_feature_ret_acceleration`
- `pair_feature_ret_20_zscore_x_feature_ret_kurt_20`
- `pair_feature_ret_20_zscore_x_feature_ret_quantile_0.2_20`
- `pair_feature_ret_20_zscore_x_feature_ret_quantile_0.5_20`
- `pair_feature_ret_20_zscore_x_feature_ret_quantile_0.8_20`
- `pair_feature_ret_20_zscore_x_feature_ret_skew_20`
- `pair_feature_ret_20_zscore_x_feature_sin_time`
- `pair_feature_ret_20_zscore_x_feature_spread_proxy`
- `pair_feature_ret_20_zscore_x_feature_spread_proxy_regime`
- `pair_feature_ret_20_zscore_x_feature_spread_proxy_zscore`
- `pair_feature_ret_20_zscore_x_feature_true_range`
- `pair_feature_ret_20_zscore_x_feature_true_range_zscore`
- `pair_feature_ret_20_zscore_x_feature_vwap_deviation`
- `pair_feature_ret_20_zscore_x_htf_daily_return_1`
- `pair_feature_ret_20_zscore_x_htf_daily_trend_slope_10`
- `pair_feature_ret_20_zscore_x_htf_daily_vol_5`
- `pair_feature_ret_20_zscore_x_htf_distance_to_daily_high`
- `pair_feature_ret_20_zscore_x_htf_distance_to_daily_low`
- `pair_feature_ret_20_zscore_x_htf_hourly_trend_alignment`
- `pair_feature_ret_20_zscore_x_htf_volatility_ratio`
- `pair_feature_ret_5_regime_x_feature_ret_5_zscore`
- `pair_feature_ret_5_regime_x_feature_ret_acceleration`
- `pair_feature_ret_5_regime_x_feature_ret_kurt_20`
- `pair_feature_ret_5_regime_x_feature_ret_quantile_0.2_20`
- `pair_feature_ret_5_regime_x_feature_ret_quantile_0.5_20`
- `pair_feature_ret_5_regime_x_feature_ret_quantile_0.8_20`
- `pair_feature_ret_5_regime_x_feature_ret_skew_20`
- `pair_feature_ret_5_regime_x_feature_sin_time`
- `pair_feature_ret_5_regime_x_feature_spread_proxy`
- `pair_feature_ret_5_regime_x_feature_spread_proxy_regime`
- `pair_feature_ret_5_regime_x_feature_spread_proxy_zscore`
- `pair_feature_ret_5_regime_x_feature_true_range`
- `pair_feature_ret_5_regime_x_feature_true_range_zscore`
- `pair_feature_ret_5_regime_x_feature_vwap_deviation`
- `pair_feature_ret_5_regime_x_htf_daily_return_1`
- `pair_feature_ret_5_regime_x_htf_daily_trend_slope_10`
- `pair_feature_ret_5_regime_x_htf_daily_vol_5`
- `pair_feature_ret_5_regime_x_htf_distance_to_daily_high`
- `pair_feature_ret_5_regime_x_htf_distance_to_daily_low`
- `pair_feature_ret_5_regime_x_htf_hourly_trend_alignment`
- `pair_feature_ret_5_regime_x_htf_volatility_ratio`
- `pair_feature_ret_5_x_feature_ret_5_regime`
- `pair_feature_ret_5_x_feature_ret_5_zscore`
- `pair_feature_ret_5_x_feature_ret_acceleration`
- `pair_feature_ret_5_x_feature_ret_kurt_20`
- `pair_feature_ret_5_x_feature_ret_quantile_0.2_20`
- `pair_feature_ret_5_x_feature_ret_quantile_0.5_20`
- `pair_feature_ret_5_x_feature_ret_quantile_0.8_20`
- `pair_feature_ret_5_x_feature_ret_skew_20`
- `pair_feature_ret_5_x_feature_sin_time`
- `pair_feature_ret_5_x_feature_spread_proxy`
- `pair_feature_ret_5_x_feature_spread_proxy_regime`
- `pair_feature_ret_5_x_feature_spread_proxy_zscore`
- `pair_feature_ret_5_x_feature_true_range`
- `pair_feature_ret_5_x_feature_true_range_zscore`
- `pair_feature_ret_5_x_feature_vwap_deviation`
- `pair_feature_ret_5_x_htf_daily_return_1`
- `pair_feature_ret_5_x_htf_daily_trend_slope_10`
- `pair_feature_ret_5_x_htf_daily_vol_5`
- `pair_feature_ret_5_x_htf_distance_to_daily_high`
- `pair_feature_ret_5_x_htf_distance_to_daily_low`
- `pair_feature_ret_5_x_htf_hourly_trend_alignment`
- `pair_feature_ret_5_zscore_x_feature_ret_acceleration`
- `pair_feature_ret_5_zscore_x_feature_ret_kurt_20`
- `pair_feature_ret_5_zscore_x_feature_ret_quantile_0.2_20`
- `pair_feature_ret_5_zscore_x_feature_ret_quantile_0.5_20`
- `pair_feature_ret_5_zscore_x_feature_ret_quantile_0.8_20`
- `pair_feature_ret_5_zscore_x_feature_ret_skew_20`
- `pair_feature_ret_5_zscore_x_feature_sin_time`

</details>

---

## Remediation Recommendations

Based on the audit findings, the following actions are recommended:

### 1. HTF Feature Leakage (htf_* and cross_* involving HTF)

The `htf_*` features are constructed from daily and 1-hour bar data joined via `join_asof(strategy='backward')`. While `strategy='backward'` is correct in principle, if the daily bar timestamps are set to the bar's **close time** (e.g., 16:00 ET) rather than the bar's **open time** (e.g., 09:30 ET), then during the trading day the "current" daily bar's data leaks its eventual close price into the 5-minute rows.

**Fix:** Ensure daily bar timestamps represent the bar's **start** time (or the prior day's close timestamp). The `join_asof` should match each 5-min row to the **previous** completed daily bar, not the current incomplete one.

### 2. Rolling Window Computations

Features using `rolling_std`, `rolling_mean`, `rolling_quantile`, `rolling_skew`, or `rolling_kurt` include the current observation (t) in the window, which is standard practice. However, if the window center is misconfigured or the computation uses future data (e.g., `shift(-1)` before rolling), leakage occurs.

**Fix:** Audit all `rolling_*` calls. If any feature is intended to be strictly lagged (not including t=0), apply `.shift(1)` before the rolling computation.

### 3. Z-Score Features

Z-scores computed with rolling mean/std that include the current observation can create subtle forward bias because the denominator (std) is influenced by the current return. This is particularly problematic if the z-score is then used to predict the next return.

**Fix:** Compute rolling mean and std on `shift(1)`-lagged values to exclude the current observation, then apply the z-score formula to the current value using those lagged statistics.

### 4. Regime Features

The regime variable is derived from smoothed volatility which itself depends on realized returns. If the volatility computation window includes the current observation's return (which contributes to the regime classification), then the regime implicitly contains information about the current bar's outcome.

**Fix:** Lag the volatility estimate by at least 1 bar before computing the regime flag: `regime_t = f(vol_{t-1})`.

### 5. VWAP Deviation

VWAP for an incomplete bar uses the bar's current close/volume, meaning the feature contains the bar's own price action. While this is standard for execution algos, it creates look-ahead bias in a predictive context.

**Fix:** Use the previous bar's completed VWAP, or compute VWAP from t-1 to t-window.

---

*Report generated by `feature_leakage_audit.py` using data from `artifacts\full_feature_matrix_c695d166720c.parquet`.*