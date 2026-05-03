# SME Financing Pain Index

A borrower-side SME financing stress index for Europe, built from ECB SAFE survey signals and compared with the ECB New CISS market-stress benchmark. The project includes a reproducible notebook, processed datasets, validation outputs, forecasting diagnostics, and an interactive Dash dashboard.

## Quick Start

Run these commands from the repository root:

```powershell
python -m pip install -r requirements.txt
python dashboard/app.py
```

Then open:

```text
http://127.0.0.1:8050
```

If port `8050` is already busy, use the helper runner:

```powershell
python dashboard/run_8051.py
```

Then open:

```text
http://127.0.0.1:8051
```

To open the portfolio notebook:

```powershell
python -m jupyter notebook notebooks/BigData_SME_FPI_Portfolio.ipynb
```

To re-execute the notebook in place:

```powershell
python -m jupyter nbconvert --to notebook --execute notebooks/BigData_SME_FPI_Portfolio.ipynb --inplace --ExecutePreprocessor.timeout=1200
```

## What To Read First

For a first-time reader, use this order:

1. `dashboard/app.py`: run the dashboard and start on **Start Here**.
2. **Current Board**: see the latest monitoring tiers and country drivers.
3. **Country Diagnosis**: inspect why a country was flagged.
4. **Forecast Check**: read the H+1 early-warning model as a validation layer, not as a deterministic forecast.
5. **Data & Method** and **Data Preview**: audit data sources, rows, limitations, and reproducibility.
6. `notebooks/BigData_SME_FPI_Portfolio.ipynb`: inspect formulas, figure rationale, and the new dashboard companion evidence section.

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
- Provides a dashboard-first reading path with latest monitoring tiers, country diagnosis cards, forecast audit views, and data preview tools.

## Repository Structure

| Path | Purpose |
|---|---|
| `dashboard/` | Interactive Dash dashboard for storytelling, hover diagnostics, current monitoring, and exploratory analysis. |
| `notebooks/` | Executed portfolio notebook for Big Data Visualization and Analysis. |
| `data/processed/` | Cleaned panels, index versions, validation outputs, forecast outputs, and dashboard-ready diagnostics. |
| `data/raw/` | Downloaded raw ECB/World Bank inputs, except the very large Q0B raw cache. |
| `figures/` | Static and interactive visualization outputs generated from the notebook and scripts. |
| `reports/` | Methodology notes, data dictionary, profiling report, and UI/UX review documents. |
| `scripts/` | Reproducible Python pipeline. |
| `tests/` | Smoke tests for dashboard imports, source files, and key generated views. |

## Main Outputs

| Output | Description |
|---|---|
| `dashboard/app.py` | Interactive SME Financing Pain Observatory built with Dash and Plotly. |
| `notebooks/BigData_SME_FPI_Portfolio.ipynb` | Main executed notebook with code, outputs, explanations, chart rationale, and dashboard companion evidence. |
| `data/processed/sme_fpi_panel_v2.csv` | Main country-half-year panel with SME_FPI versions, PCA, clusters, CISS, and relative gaps. |
| `data/processed/forecast_decision_board.csv` | Latest decision-board risk tier, signal type, model agreement quality, and driver summary by country. |
| `data/processed/forecasting_feature_panel.csv` | Expanded forecast panel with core SME-FPI, macro, micro, BLS, MIR, Eurostat predictors, and lagged features. |
| `data/processed/forecasting_model_evaluation.csv` | Rolling-origin H+1 forecast evaluation comparing ML models, simple baselines, and ARIMA/ARIMAX benchmarks. |
| `data/processed/dashboard_source_catalog.csv` | Dashboard-facing inventory of datasets, roles, row counts, and index-vs-forecast usage. |
| `data/processed/safe_problem_severity_cube.csv` | Big-cube severity, top-box, and high-pressure measures. |
| `data/processed/validation_results.csv` | Pearson, Spearman, and within-country validation correlations. |
| `reports/data_dictionary_v2.md` | Data dictionary for the v2 project. |
| `reports/SME_FPI_v2_methodology.md` | Methodology document. |

## Dashboard Tabs

- **Start Here:** project claim, five-minute reading path, glossary, and presentation logic.
- **Current Board:** latest monitoring tier, signal type, confidence, model agreement, and country drivers.
- **Defense & Findings:** direct answers to data sufficiency, readability, visualization fit, and forecast defensibility questions.
- **SME-FPI Index:** plain-English index definition, formula logic, KPI cards, thresholds, and storytelling structure.
- **Context Guides:** short explanations of SAFE, CISS, the relative SME-CISS gap, and index versions.
- **Index Explorer:** SME_FPI vs CISS time series, weighting robustness, animated stress motion, heatmaps, and component diagnostics.
- **Borrower-Market Gap:** choropleth map, relative gap ranking, and diagnostic bubble chart.
- **Regime Appendix:** PCA cluster scatter, PCA correlation circle, and exploratory 3D PCA regime view.
- **Micro Cube:** SAFE problem, firm-size, and sector diagnostics from the 231,231-row survey cube.
- **Forecast Check:** H+1 early-warning models with rolling-origin MAE/RMSE loss, benchmarks, model rank stability, and country error views.
- **Country Diagnosis:** evidence cards linking monitoring tiers to current score, relative gap, forecast direction, agreement, and drivers.
- **Data Preview:** quick preview of raw, processed, validation, and forecast files.
- **Data & Method:** source catalog, validation chart, design safeguards, and main limitations.

## Data Sources

- ECB Survey on the Access to Finance of Enterprises (SAFE).
- ECB New Composite Indicator of Systemic Stress (New CISS).
- ECB Bank Lending Survey (BLS): SME credit standards, SME loan demand, SME loan terms and conditions, and rejected enterprise loan applications.
- ECB MFI Interest Rate Statistics (MIR): small corporate loan rates, large corporate loan rates, and small corporate loan volumes.
- Eurostat short-term business statistics: business bankruptcy declarations index and business registrations index.
- World Bank macro indicators: GDP growth, unemployment, CPI inflation, and domestic credit to the private sector.

## Large Data Note

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

## Methodological Caveats

- The index is descriptive and correlational, not causal.
- SAFE is survey-based and measures reported borrower-side conditions.
- CISS is a common euro-area benchmark, not a country-specific SME credit variable.
- The relative SME-CISS gap should not be interpreted as proof of local financial-market stress.
- Q0B severity uses ordinal survey answers, so top-box and high-pressure shares are added as robustness checks.
- World Bank macro variables are annual and should be read as broad context rather than high-frequency validation.
- BLS, MIR, and Eurostat variables are used as forecasting predictors, not as SME-FPI Core components.
- The forecast layer is an early-warning experiment on a small country-period panel, not a production credit-risk model.

## Troubleshooting

- If a module import fails, run `python -m pip install -r requirements.txt` again in the same environment that runs the dashboard.
- If the dashboard opens but styles look stale, hard-refresh the browser tab.
- If port `8050` is busy, run `python dashboard/run_8051.py` and open `http://127.0.0.1:8051`.
- If the big raw SAFE cube is missing, regenerate it with `python scripts/06_build_big_cube.py`; the dashboard normally uses processed files under `data/processed/`.
