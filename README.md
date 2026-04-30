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
- Adds an early-warning forecasting layer using SAFE micro diagnostics, World Bank macro context, ECB BLS lender-side signals, ECB MIR loan-rate/volume data, and Eurostat business registration/bankruptcy signals.
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
| `dashboard/` | Interactive Dash dashboard for storytelling, hover diagnostics, and exploratory analysis. |

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
| `data/processed/forecasting_feature_panel.csv` | Expanded forecast panel with core SME-FPI, macro, micro, BLS, MIR, Eurostat business-demography predictors, and lagged features. |
| `data/processed/forecasting_model_evaluation.csv` | Rolling-origin H+1 forecast evaluation comparing Elastic Net, Ridge, Random Forest, Gradient Boosting, and five simple baselines. |
| `data/processed/forecast_decision_board.csv` | Latest decision-board risk tier, forecast agreement, confidence, and driver summary by country. |
| `data/processed/dashboard_source_catalog.csv` | Dashboard-facing inventory of datasets, roles, and index-vs-forecast usage. |
| `data/processed/validation_results.csv` | Pearson, Spearman, and within-country validation correlations. |
| `reports/data_dictionary_v2.md` | Data dictionary for the v2 project. |
| `reports/SME_FPI_v2_methodology.md` | Methodology document. |
| `figures/notebook_generated/00_signature_sme_fpi_story.png` | One-page signature visual summary. |
| `dashboard/app.py` | Interactive SME Financing Pain Observatory built with Dash and Plotly. |

## Data Sources

- ECB Survey on the Access to Finance of Enterprises (SAFE).
- ECB New Composite Indicator of Systemic Stress (New CISS).
- ECB Bank Lending Survey (BLS):
  - SME credit standards,
  - SME loan demand,
  - SME loan terms and conditions,
  - rejected enterprise loan applications.
- ECB MFI Interest Rate Statistics (MIR):
  - small corporate loan rates,
  - large corporate loan rates,
  - small corporate loan volumes.
- Eurostat short-term business statistics:
  - business bankruptcy declarations index,
  - business registrations index.
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
python scripts/10_download_forecasting_data.py
python scripts/11_build_forecasting_layer.py
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

## Interactive Dashboard

The dashboard is designed as an interactive companion to the notebook. It explains the project, defines SME_FPI, lets the viewer hover over each chart for numeric and interpretive labels, and adds chart-level rationale for data visualization assessment.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run from the repository root:

```powershell
python dashboard/app.py
```

Then open:

```text
http://127.0.0.1:8050
```

Main dashboard tabs:

- Decision Board: current watchlist, risk tier, confidence, model agreement, and primary country drivers.
- Overview: research question, plain-English SME_FPI definition, first-read data/method map, KPI cards, monitor thresholds, dataset shape, and storytelling logic.
- Context guides: short explanations of SAFE, CISS, the relative SME-CISS gap, and the four index versions for non-finance readers.
- Explorer: SME_FPI vs CISS time series, weighting robustness, animated stress motion, country-period heatmap, and component heatmap.
- Hidden Stress: choropleth map, relative SME-CISS gap ranking, and diagnostic bubble chart.
- Regimes: PCA cluster scatter, PCA correlation circle, and an exploratory 3D PCA regime view.
- Big Data Cube: local controls for SAFE problem, firm size, and sector, plus trend, bar, and heatmap diagnostics from the 231,231-row cube.
- Forecast Lab: expanded machine-learning early-warning model suite using core SME-FPI, macro, micro, BLS, MIR, and Eurostat predictors, with Elastic Net, Ridge, Random Forest, Gradient Boosting, and strong simple-baseline checks.
- Country Diagnosis: evidence-bundle cards linking risk tiers to current score, hidden gap, H+1 forecast, model agreement, and drivers.
- Methodology: data source inventory, validation chart, design safeguards, and main limitations.

## Methodological Caveats

- The index is descriptive and correlational, not causal.
- SAFE is survey-based and measures reported borrower-side conditions.
- CISS is a common euro-area benchmark, not a country-specific SME credit variable.
- The relative SME-CISS gap should not be interpreted as proof of local financial-market stress.
- Q0B severity uses ordinal survey answers, so top-box and high-pressure shares are added as robustness checks.
- World Bank macro variables are annual and should be read as broad context rather than high-frequency validation.
- BLS, MIR, and Eurostat variables are used as forecasting predictors, not as SME-FPI Core components, to keep the index borrower-side and interpretable.
- The forecast layer is an early-warning experiment on a small country-period panel, not a production credit-risk model.

## Project Status

The project now includes:

- core index,
- alternative weighting,
- PCA and clustering,
- cluster validation,
- big-data survey diagnostics,
- macro context,
- forward validation,
- BLS/MIR/Eurostat forecasting data,
- expanded early-warning feature panel,
- executed notebook,
- methodology documentation.
- interactive Dash dashboard.
