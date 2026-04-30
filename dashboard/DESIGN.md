# SME-FPI Dashboard Design Rules

This dashboard follows a Stitch-style workflow: define the reader intent first, convert it into a clear screen hierarchy, then iterate by checking the actual rendered interface.

## Audience

The reader is a data visualization professor who can analyze charts, but may not know SME finance, ECB SAFE, or CISS. The UI should therefore explain finance context without teaching basic visualization.

## First-Read Contract

1. State the analytical question in the hero.
2. Give the minimum finance vocabulary before the user reaches dense charts.
3. Show the data/method map early: which datasets build SME-FPI Core and which datasets only support forecasting or validation.
4. Keep default filters simple: all countries, equal-weight index, full period.
5. Use progressive disclosure for detailed controls and methodology notes.
6. Separate the story into decision board, overview, exploration, forecast, country diagnosis, hidden stress, regimes, big-data diagnostics, and methodology.

## Visual System

- Surface: quiet finance-monitor dashboard, not a marketing landing page.
- Radius: 8px maximum for cards and controls.
- Primary ink: #13202b.
- Primary dashboard blue: #244c68.
- Analytical blue: #2f6f9f.
- Warning/accent gold: #d19a2e.
- Stress red: #b23a35.
- Background: light blue-gray working surface with white analytical panels.
- Hero: dark blue grid surface with the actual signature figure as evidence, not decoration.

## Interaction Rules

- Country selection is collapsed by default because all countries are selected by default.
- Crowded country views should switch to summaries, bands, rankings, or latest-period snapshots instead of plotting every trace.
- Hover text must include numeric value, interpretation, coverage, and methodological caveat when relevant.
- 3D views are exploratory. Static 2D figures remain the main evidence.

## Chart Story Roles

- Time series: show whether SME-FPI and CISS move differently through time.
- Robustness lines: show whether index weighting choices change the story.
- Heatmaps: show country-period and component structure.
- Choropleth plus tiles: keep geographic context while reducing country-area bias.
- Gap ranking and bubble chart: expose borrower-side stress that is high relative to CISS.
- PCA scatter and correlation circle: support multidimensional analysis.
- Big Data Cube charts: explain which firm groups and problem categories sit behind the index.
- Decision Board: answer who needs attention, why, whether stress is hidden from CISS, and whether the H+1 model suite expects pressure to rise.
- Forecast Lab: compare Elastic Net, Ridge, Random Forest, and Gradient Boosting against naive, country mean, country AR(1), momentum, and pooled lag OLS baselines using the wider SAFE + macro + BLS + MIR + Eurostat predictor stack.
- Country Diagnosis: show country-level evidence bundles with current score, gap, forecast, model agreement, confidence, and drivers.
- Validation chart: check whether today's index aligns with future stress-related outcomes.
- Source catalog: clarify data provenance and prevent index-vs-predictor confusion.

## Quality Checks

- No title should overlap with legends, controls, or plot marks.
- The first viewport should communicate the project before any filtering.
- A non-finance reader should understand SAFE, SME-FPI, CISS, and the SME-CISS gap from visible dashboard text.
- A first-time viewer should see that SME-FPI Core is borrower-side SAFE, while macro/micro/BLS/MIR/Eurostat data are forecast predictors.
- Every chart should have a clear job in the story; remove charts that only add novelty.
