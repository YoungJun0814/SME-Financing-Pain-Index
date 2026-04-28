# SME Financing Pain Index

A borrower-side SME financing stress index for Europe using ECB SAFE, ECB CISS, PCA, clustering, big-data survey visualization, and external validation checks.

## Research Question

Can a borrower-side SME Financing Pain Index reveal European SME credit stress that is not fully captured by common market-side stress indicators such as the ECB New CISS?

## What This Project Does

- Builds a transparent SME Financing Pain Index from six ECB SAFE borrower-side variables.
- Compares borrower-side SME financing pain with the ECB New CISS market-stress benchmark.
- Tests alternative index designs: equal-weight, fixed-baseline, PCA-weighted, and reliability-weighted.
- Uses PCA, PCA correlation circles, KMeans clustering, elbow diagnostics, and silhouette diagnostics.
- Uses a 231,231-row ECB SAFE Q0B survey cube for Big Data visualization and robustness checks.
- Adds macro context and forward validation using World Bank indicators and future SAFE Q0B outcomes.
- Produces an executed Jupyter notebook with code, outputs, and chart-by-chart design rationale.

## Repository Structure

| Path | Purpose |
|---|---|
| `scripts/` | Reproducible Python pipeline. |
| `notebooks/` | Executed portfolio notebook for Big Data Visualization and Analysis. |
| `data/raw/` | Downloaded raw ECB/World Bank inputs, except the very large Q0B raw cache. |
| `data/processed/` | Cleaned panels, index versions, validation outputs, and diagnostics. |
| `figures/` | Static and interactive visualization outputs. |
| `reports/` | Methodology notes, data dictionary, R Markdown report, SQL notes, and review documents. |

## Main Outputs

| Output | Description |
|---|---|
| `notebooks/BigData_SME_FPI_Portfolio.ipynb` | Main executed notebook with code, outputs, explanations, and visual design rationale. |
| `data/processed/sme_fpi_panel_v2.csv` | Main country-half-year panel with SME_FPI versions, PCA, clusters, CISS, and relative gaps. |
| `data/processed/weighting_comparison.csv` | Equal, PCA, and reliability weights for the six SME_FPI components. |
| `data/processed/index_sensitivity_summary.csv` | Robustness comparison across alternative index versions. |
| `data/processed/cluster_validation_diagnostics.csv` | Elbow and silhouette diagnostics for KMeans cluster count. |
| `data/processed/safe_problem_severity_cube.csv` | Big-cube severity, top-box, and high-pressure measures. |
| `data/processed/external_validation_panel.csv` | SME_FPI, CISS, macro context, and future validation targets. |
| `data/processed/validation_results.csv` | Pearson, Spearman, and within-country validation correlations. |
| `reports/data_dictionary_v2.md` | Data dictionary for the v2 project. |
| `reports/SME_FPI_v2_methodology.md` | Methodology document. |
| `figures/notebook_generated/00_signature_sme_fpi_story.png` | One-page signature visual summary. |

## Data Sources

- ECB Survey on the Access to Finance of Enterprises (SAFE).
- ECB New Composite Indicator of Systemic Stress (New CISS).
- World Bank macro indicators:
  - GDP growth,
  - unemployment,
  - CPI inflation,
  - domestic credit to the private sector.

## Important Note About Large Data

The raw SAFE Q0B cube is about 131 MB and is intentionally excluded from GitHub:

```text
data/raw/safe_q0b_pressingness_big_cube.csv
```

It can be regenerated with:

```powershell
python scripts/06_build_big_cube.py
```

Processed outputs derived from the cube are included where practical.

## Reproducible Pipeline

Run from the repository root:

```powershell
python scripts/01_download_data.py
python scripts/02_build_panel.py
python scripts/06_build_big_cube.py
python scripts/09_build_external_validation.py
python scripts/03_generate_figures.py
python scripts/07_generate_polished_figures.py
python scripts/08_generate_signature_visual.py
python scripts/05_create_sqlite_demo.py
python scripts/04_create_bigdata_notebook.py
```

To execute the notebook:

```powershell
python -m jupyter nbconvert --to notebook --execute notebooks/BigData_SME_FPI_Portfolio.ipynb --inplace --ExecutePreprocessor.timeout=1200
```

## Methodological Caveats

- The index is descriptive and correlational, not causal.
- SAFE is survey-based and measures reported borrower-side conditions.
- CISS is a common euro-area benchmark, not a country-specific SME credit variable.
- The relative SME-CISS gap should not be interpreted as proof of local financial-market stress.
- Q0B severity uses ordinal survey answers, so top-box and high-pressure shares are added as robustness checks.
- World Bank macro variables are annual and should be read as broad context rather than high-frequency validation.

## Project Status

The project is complete through the pre-dashboard stage:

- core index,
- alternative weighting,
- PCA and clustering,
- cluster validation,
- big-data survey diagnostics,
- macro context,
- forward validation,
- executed notebook,
- methodology documentation.

The next major extension is a Streamlit dashboard.
