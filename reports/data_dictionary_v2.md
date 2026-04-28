# SME_FPI v2 Data Dictionary

## Project Question

Can a borrower-side SME Financing Pain Index reveal SME credit stress in Europe that is not fully captured by common market-stress indicators such as the ECB CISS?

## Core Datasets

| File | Rows | Role |
|---|---:|---|
| `data/processed/sme_fpi_panel_v2.csv` | 386 | Main country-half-year panel for index construction, PCA, clustering, CISS comparison, and sensitivity checks. |
| `data/processed/safe_component_long.csv` | 11,300 | Long-form ECB SAFE component data used to build the six borrower-side index variables. |
| `data/raw/safe_q0b_pressingness_big_cube.csv` | 231,231 | Large ECB SAFE Q0B cube used for Big Data visualization and additional validation targets. |
| `data/processed/safe_problem_severity_cube.csv` | 20,570 | Problem severity cube with weighted severity, top-box share, and high-pressure share. |
| `data/processed/macro_context_panel.csv` | 386 | World Bank annual macro indicators aligned to the country-half-year panel. |
| `data/processed/external_validation_panel.csv` | 386 | Main panel plus future validation targets from SAFE Q0B, CISS, and macro context. |

## SME_FPI Components

All six components are coded so that higher values mean more financing pain.

| Variable | Meaning | Source |
|---|---|---|
| `access_finance_main_problem` | SMEs saying access to finance is their most pressing problem. | ECB SAFE |
| `bank_loan_rejected` | SMEs whose bank loan application was rejected. | ECB SAFE |
| `bank_loan_cost_too_high` | SMEs refusing a bank loan because the cost was too high. | ECB SAFE |
| `bank_loan_limited_amount` | SMEs receiving only a limited bank loan amount. | ECB SAFE |
| `bank_willingness_deteriorated` | SMEs saying banks' willingness to lend deteriorated. | ECB SAFE |
| `interest_rates_increased` | SMEs reporting that interest rates increased. | ECB SAFE |

## Index Versions

| Variable | Definition | Use |
|---|---|---|
| `SME_FPI_equal_z` | Equal-weight average of available standardized components, standardized again. | Main baseline index. |
| `SME_FPI_fixed_baseline_z` | Index standardized relative to 2015-S1 to 2019-S2. | Checks whether results depend on full-sample standardization. |
| `SME_FPI_pca_weighted_z` | Component weights based on absolute PC1 loadings. | Data-driven weighting sensitivity check. |
| `SME_FPI_reliability_weighted_z` | Component weights based on leave-one-out reliability and coverage. | Internal-consistency weighting sensitivity check. |
| `SME_FPI_pca_z` | Standardized first principal component score. | PCA validation reference. |

## Relative Gap Variables

| Variable | Formula | Interpretation |
|---|---|---|
| `Hidden_SME_Stress` | `SME_FPI_equal_z - CISS_z` | Legacy project label for the relative SME-CISS gap. |
| `Relative_Gap_equal` | `SME_FPI_equal_z - CISS_z` | Borrower-side SME pain relative to common euro-area market stress. |
| `Relative_Gap_fixed_baseline` | `SME_FPI_fixed_baseline_z - CISS_z` | Gap using fixed-baseline standardization. |
| `Relative_Gap_pca_weighted` | `SME_FPI_pca_weighted_z - CISS_z` | Gap using PCA-weighted SME_FPI. |
| `Relative_Gap_reliability_weighted` | `SME_FPI_reliability_weighted_z - CISS_z` | Gap using reliability-weighted SME_FPI. |

Important interpretation: CISS is a common euro-area market-stress benchmark. The gap is not proof of country-specific local financial-market stress.

## Big Data Cube Measures

| Variable | Definition | Caveat |
|---|---|---|
| `severity_score_1_10` | Weighted mean of SAFE Q0B answer levels 1 to 10. | Treats ordinal responses pragmatically as a descriptive scale. |
| `top_box_share_8_10` | Share of valid responses rated 8, 9, or 10. | More robust to ordinal-scale concerns. |
| `high_pressure_share_7_10` | Share of valid responses rated 7, 8, 9, or 10. | Broader severe-pressure measure. |

## Macro Context Variables

World Bank annual indicators are aligned to the matching year of each half-year panel observation.

| Variable | Meaning | Stress Direction |
|---|---|---|
| `gdp_growth_pct` | GDP growth, annual %. | Lower is worse. |
| `unemployment_pct` | Unemployment rate, annual %. | Higher is worse. |
| `inflation_cpi_pct` | CPI inflation, annual %. | Higher is worse. |
| `domestic_credit_private_pct_gdp_change` | Annual change in private credit as % of GDP. | Lower change is worse. |
| `macro_stress_context_z` | Mean of available standardized macro stress components. | Higher means worse macro-financial context. |

## Validation Outputs

| File | Purpose |
|---|---|
| `weighting_comparison.csv` | Equal, PCA, and reliability weights for the six components. |
| `index_sensitivity_summary.csv` | Correlation and rank stability across index versions. |
| `cluster_validation_diagnostics.csv` | Elbow and silhouette diagnostics for KMeans cluster count. |
| `validation_results.csv` | Raw, Spearman, and within-country correlations between index versions and future validation targets. |
| `variable_expansion_candidates.csv` | Implemented and future candidate variables by conceptual layer. |

## Main Limitations

- The index is descriptive, not causal.
- SAFE is survey-based and measures reported conditions.
- Full-sample standardization is useful for retrospective comparison but not ideal for real-time monitoring.
- Fixed-baseline standardization addresses that issue but changes the scale.
- World Bank macro data are annual, so the macro validation layer is broad context rather than high-frequency evidence.
- Bank Lending Survey variables are still recommended for the next lender-side validation extension.
