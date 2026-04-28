# SME_FPI v2 Methodology

## Research Design

This project builds a borrower-side SME Financing Pain Index for European countries and compares it with a common market-side stress benchmark, the ECB New CISS.

The central idea is that financial-market stress and borrower-side SME financing pain are related but not identical. CISS can fall after market stress eases, while SMEs may still report difficulty obtaining finance, higher loan costs, or weaker bank willingness to lend.

## Baseline Index

The baseline index uses six ECB SAFE borrower-side variables:

1. Access to finance is the most pressing problem.
2. Bank loan application rejected.
3. Bank loan offer refused because cost was too high.
4. Bank loan only granted for a limited amount.
5. Banks' willingness to provide credit deteriorated.
6. Interest rates increased.

Each variable is standardized:

```text
z_j(c,t) = (x_j(c,t) - mean(x_j)) / sd(x_j)
```

The raw baseline index averages available standardized components:

```text
SME_FPI_equal_raw(c,t) = mean of available z_j(c,t)
```

The index is calculated only when at least four of six components are available. The final baseline index is standardized again:

```text
SME_FPI_equal_z(c,t) = z(SME_FPI_equal_raw(c,t))
```

## Alternative Index Versions

### Fixed-Baseline SME_FPI

The fixed-baseline version standardizes components relative to 2015-S1 to 2019-S2:

```text
z_baseline_j(c,t) = (x_j(c,t) - mean_2015_2019(x_j)) / sd_2015_2019(x_j)
```

This avoids using the entire future sample as the reference point when interpreting later stress periods.

### PCA-Weighted SME_FPI

The PCA-weighted version uses absolute PC1 loadings:

```text
w_j = abs(PC1_loading_j) / sum(abs(PC1_loading_j))
```

The weighted index averages available standardized components using the available weights. This checks whether the equal-weight result depends on arbitrary equal weighting.

### Reliability-Weighted SME_FPI

The reliability-weighted version uses each component's leave-one-out correlation with the other components, multiplied by data coverage:

```text
reliability_j = corr(z_j, equal_weight_index_without_j)
coverage_j = share of non-missing observations
w_j = max(reliability_j, 0) * coverage_j
```

Weights are normalized to sum to one.

## PCA and Clustering

PCA is used to check whether the six components share a common financing-pain dimension. PC1 is oriented so that higher PC1 means more financing pain.

KMeans clustering is used descriptively to identify country-period financing regimes. The four-cluster solution is selected because it gives interpretable regimes:

- low SME financing pain,
- interest-rate pressure,
- loan-cost pressure,
- broad SME financing pain.

Elbow and silhouette diagnostics are included to make the cluster-count choice transparent. Silhouette scores alone do not determine the final model because a statistically compact two-cluster solution would be less informative for the project story.

## CISS Gap

The relative gap is:

```text
Relative_SME_CISS_Gap(c,t) = SME_FPI(c,t) - CISS_z(t)
```

This should be interpreted carefully. CISS is a common euro-area market-stress benchmark, not a country-specific SME credit condition variable. The gap indicates whether borrower-side SME financing pain is high relative to that common benchmark.

## Big Data Cube Layer

The 231,231-row SAFE Q0B cube is used as a Big Data diagnostic layer. It does not build the core SME_FPI. Instead, it helps explain broader problem pressure by country, period, firm size, sector, firm age, problem category, and response level.

Three measures are used:

```text
severity_score_1_10 = weighted mean of answer levels 1 to 10
top_box_share_8_10 = share of valid responses rated 8, 9, or 10
high_pressure_share_7_10 = share of valid responses rated 7, 8, 9, or 10
```

The top-box and high-pressure shares are included because they are safer for ordinal survey data than relying only on the exact distance between response levels.

## External and Forward Validation

The validation panel compares SME_FPI versions with future targets:

- future SME_FPI,
- future SAFE Q0B access-to-finance severity,
- future SAFE Q0B severe-response shares,
- future macro stress context,
- future CISS.

The macro context uses World Bank annual GDP growth, unemployment, inflation, and private-credit variables. These are aligned to the matching half-year period by year.

Validation results report:

- Pearson correlation,
- Spearman rank correlation,
- within-country Pearson correlation.

The within-country correlation is important because raw panel correlations can be inflated by permanent differences between countries.

## Theoretical Limits

- The analysis is descriptive and correlational, not causal.
- SAFE survey responses measure perceived and reported financing conditions.
- Annual macro variables are lower-frequency than SAFE half-year observations.
- PCA weights maximize explained variance, not economic importance.
- Reliability weights are internal validation, not external truth.
- Bank Lending Survey credit-standard variables remain a recommended next extension.

## Current Project Status

The project now satisfies the pre-dashboard methodological roadmap:

- narrative refactor,
- alternative index versions,
- PCA correlation circle,
- improved cluster plot,
- cluster-count diagnostics,
- big-cube top-box and high-pressure robustness,
- macro context panel,
- external and forward validation panel,
- data dictionary and methodology documentation.

The remaining major product step is the Streamlit dashboard.
