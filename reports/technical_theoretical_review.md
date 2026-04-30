# Technical and Theoretical Review

## Review date

2026-04-25

## Implementation status

The first implementation of the SME Financing Pain Index is complete:

- data downloaded from ECB Data API,
- SAFE borrower-side components extracted,
- ECB New CISS downloaded as benchmark,
- country-period panel built,
- equal-weight index computed,
- PCA version computed,
- k-means financing regimes computed,
- six static visualizations generated,
- one interactive map generated,
- one general-audience redesign figure generated,
- polished figure set regenerated with consistent typography, colors, labels, and 2015-2025 storytelling window,
- SQLite database and SQL outputs generated,
- larger SAFE Q0B survey cube added for the Big Data Visualization angle,
- Big Data Visualization notebook created and corrected for top-to-bottom execution,
- ydata_profiling diagnostic cell added to the notebook,
- R Markdown report drafted and corrected for `base_dir` setup order.

## Data review

### Passed checks

| Check | Result |
|---|---|
| SAFE API access works | Pass |
| CISS API access works | Pass |
| Main panel has observations | Pass |
| All six components are available for the main SME all-sector panel | Pass |
| Country-period panel is complete enough for PCA | Pass |
| CISS is merged by half-year, matching SAFE frequency | Pass |
| SQLite database export works | Pass |

### Data dimensions

The processed main panel contains:

```text
12 countries
33 half-year periods
386 country-period rows
6 borrower-side components
```

The raw selected SAFE component data contain firm-size breakdowns and 11,300 rows.

The added SAFE Q0B pressingness cube contains:

```text
231,231 raw rows
about 137 MB raw CSV
14 regions
6 firm-size categories
5 sector categories
6 firm-age categories
7 problem items
11 answer levels
21 half-year periods
```

This larger cube is the stronger basis for calling the project a Big Data Visualization project. The compact 386-row panel should be described as the analytical index table, not as the full dataset.

### Important limitation

The selected SAFE index components do not provide sector breakdowns in the downloaded series, despite the SAFE data structure including a sector dimension. The implementation originally attempted a firm-size by sector chart, but this would have been misleading because only the all-sector code `A` was available for these indicators. The visualization was corrected to a firm-size by component heatmap.

The larger Q0B cube does provide both firm-size and sector breakdowns, but not always as a full crossed firm-size by sector matrix. The implementation therefore presents firm-size and sector as honest marginal breakdowns.

## Theoretical review

### Is the index conceptually valid?

Yes, with a clear limitation.

The SME_FPI is valid as a **borrower-side survey-based financing pain index**. It should not be described as a full financial conditions index because it does not include all market, banking, and macro-financial variables.

Correct phrasing:

```text
borrower-side SME financing pain index
```

Avoid:

```text
complete European financial conditions index
```

### Does it duplicate ECB CISS?

No.

The CISS is market-side/systemic. SME_FPI is borrower-side/survey-based. CISS is used only as a benchmark.

### Are all component directions correct?

Yes.

All selected components are coded so higher values mean more financing pain:

| Component | Direction |
|---|---|
| Access to finance as main problem | higher = more pain |
| Bank loan rejected | higher = more pain |
| Bank loan cost too high | higher = more pain |
| Bank loan limited amount | higher = more pain |
| Banks' willingness deteriorated | higher = more pain |
| Interest rates increased | higher = more pain |

### Is PCA theoretically consistent?

Yes.

PC1 explains about half of the standardized component variance in the current run. The PC1 loadings are all positive after orientation, so PC1 can reasonably be interpreted as a general SME financing pain factor.

### Is the hidden stress gap valid?

Mostly yes, but it must be interpreted carefully.

The gap is:

```text
Hidden_SME_Stress = z(SME_FPI) - z(CISS)
```

This compares relative positions, not absolute units. A positive gap means SME financing pain is high relative to its own history while market stress is lower relative to its own history.

CISS is a common euro-area market-side benchmark, not a country-specific SME credit variable. The gap should therefore be described as a relative borrower-side gap against a shared benchmark.

It should not be interpreted as:

```text
SMEs are exactly X units more stressed than markets.
```

## Technical review

### Issue 1: R is not available on PATH

`Rscript` was not found in the current shell environment. Therefore, the R Markdown report was drafted but not knitted.

Impact:

- The `.Rmd` file exists and can be opened in RStudio.
- The `base_dir` setup-order bug has been corrected.
- The Python pipeline has been executed and verified.
- Knitting must be done once R/RStudio is available.

### Issue 2: KMeans warnings on Windows

Scikit-learn produced a non-fatal Windows/MKL warning about core detection. The clustering completed successfully.

Impact:

- No output failure.
- If needed, set `OMP_NUM_THREADS=2` before running the script.

### Issue 3: Sector dimension not available for selected variables

The initial plan assumed sector breakdown would be available. It was not available for the exact selected SAFE components.

Fix:

- Replaced sector-size heatmap with firm-size-component heatmap.
- Documented this limitation.

### Issue 4: Notebook execution order

The notebook originally displayed the signature figure before defining `BASE` and importing display helpers.

Fix:

- Moved setup, imports, robust project-root detection, and data loading to the first code cell.
- Regenerated the notebook.
- Added a ydata profiling cell that writes `reports/sme_fpi_ydata_profile.html`.

### Issue 5: Dynamic component denominator

The report originally described the index as a literal six-component mean in every row. The implementation averages available standardized components and requires at least four components.

Fix:

- Updated the report and notebook to define the available component set `I(c,t)`.
- Kept the equal-weight interpretation, with explicit disclosure of the minimum component rule.

### Issue 6: Ordinal severity score

The SAFE Q0B severity score is based on 1-10 ordinal response levels.

Fix:

- Documented the severity score as a pragmatic descriptive weighted mean, not an exact cardinal measure.

## Course-fit review

| Requirement | Status |
|---|---|
| One research question | Pass |
| 4-6 visualizations | Pass |
| Written design rationale | Pass in Rmd and notebook |
| Big data / multidimensional data | Pass, strengthened by the 231,231-row SAFE Q0B survey cube |
| R Markdown report | Drafted and setup-order corrected, not knitted |
| dplyr summaries | Included in Rmd |
| SQL/database component | Pass |
| MDA technique | PCA and clustering implemented |
| General audience redesign | Pass |

## Recommended next fixes

1. Knit the R Markdown report once R is available.
2. Optional: add Eurostat bankruptcy data as an external validation outcome, but do not mix it into the first version of the index without a clear theory.

## Bottom line

The implementation is theoretically coherent and technically functional. SQL and the general-audience redesign are now implemented. The main remaining packaging gap is that the R Markdown report could not be knitted because `Rscript` is not available on PATH in the current shell environment.

## 2026-04-30 Dashboard and Forecasting Upgrade Review

### Additional implementation status

- Forecasting layer now includes BLS, MIR, Eurostat, macro, micro SAFE-cube, CISS, and lagged SME-FPI predictors.
- The core SME-FPI index remains built only from the six borrower-side ECB SAFE financing-pain components.
- Rolling-origin validation now produces all-origin model predictions, historical decision-board tiers, country-level forecast errors, and risk-tier validation.
- The dashboard exposes the method/data distinction through the Overview, Forecast Lab, Decision Board, Country Diagnosis, and Methodology/source catalog views.

### Technical checks passed

| Check | Result |
|---|---|
| Python syntax check for dashboard and forecasting scripts | Pass |
| Forecasting layer rebuild | Pass |
| Nine-model suite present in model evaluation | Pass |
| Future/target leakage blocked from forecast feature columns | Pass |
| Eurostat predictors documented as external forecasting inputs | Pass |
| Historical decision-board outputs generated | Pass |
| New Dash figure builders render from processed data | Pass |
| `unittest` smoke suite | Pass |

### Forecasting interpretation

The recent best model is Ridge, with a small positive MAE improvement versus the strongest simple baseline. This is theoretically plausible because the panel is small, country-period observations are correlated, and regularized linear models are less likely to overfit than high-capacity nonlinear models.

The dashboard should therefore describe the forecast as an early-warning experiment, not as a production credit-risk model. The risk tier is a triage label that combines current borrower-side pressure, hidden SME-CISS gap, H+1 model signal, and model agreement. It is not a causal or policy-action rule.

### Theoretical caveats to keep visible

- Eurostat, BLS, MIR, macro, and micro SAFE-cube variables are predictors or validation context, not components of the SME-FPI Core.
- CISS remains a shared euro-area market-stress benchmark, not a country-specific SME credit series.
- Full-sample z-scores are useful for retrospective analysis; fixed-baseline and robustness views should be used when presenting a real-time monitoring interpretation.
- Historical Alert/Watch tiers can indicate high stress even when the next period mean-reverts, so tier validation should be read as stress-state validation plus direction-check evidence, not as guaranteed acceleration.
