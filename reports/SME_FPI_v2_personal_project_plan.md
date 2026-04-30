# SME_FPI v2 Personal Project Plan

## Project Direction

This project started as an integrated final project for three courses, but the next version should be treated as a personal research and portfolio project.

The current version is a strong baseline:

- It builds a borrower-side SME Financing Pain Index from ECB SAFE survey variables.
- It compares borrower-side stress with the ECB CISS market-stress benchmark.
- It uses a larger SAFE Q0B survey cube for Big Data Visualization.
- It includes PCA, clustering, correlation analysis, and several explanatory visualizations.

The next goal is to move from a course-ready prototype to a more robust composite-indicator project with stronger methodology, richer validation, and a dashboard.

## Target Submission Context

The intended audience is **Anna Kovbasiuk**, who teaches all three relevant courses:

1. Data Visualization.
2. Multidimensional Data Analysis.
3. Big Data Visualization and Analysis.

Because the same professor evaluates all three courses, the project should not feel like three disconnected assignments. It should feel like one coherent analytical product with three layers:

- **Data Visualization layer:** strong chart choices, deliberate visual hierarchy, clean color logic, clear captions, and avoidance of chart crimes.
- **Multidimensional Data Analysis layer:** PCA, loadings interpretation, clustering, regime interpretation, correlation structure, and methodological limitations.
- **Big Data Visualization and Analysis layer:** meaningful use of the 231,231-row SAFE Q0B high-dimensional survey cube, not only the smaller 386-row index panel.

The strongest positioning is:

```text
This project builds a borrower-side SME financing stress index,
validates it with multidimensional methods,
and uses a larger SAFE survey cube to explain where the stress appears across countries, time, and firm groups.
```

## Current Quality Assessment

Current status:

```text
Strong course-ready prototype; good enough for submission, but can be improved into a more polished personal portfolio project.
```

Main strengths:

- The research question is clear and practical.
- The topic is niche enough to feel original, but still grounded in official data.
- SAFE and CISS are conceptually different, which creates a real analytical tension.
- The notebook already includes code, outputs, comments, chart rationales, and a Big Data layer.
- PCA, clustering, correlation analysis, and big-cube diagnostics give the project enough methodological breadth for three courses.

Main risk:

```text
The biggest risk is not technical execution.
The biggest risk is narrative focus and methodological defensibility.
```

In other words, the professor is more likely to question:

- Why these exact charts?
- Why these exact variables?
- Why this weighting method?
- What does the CISS gap really mean?
- Does the Big Data cube contribute insight, or is it only added for size?
- Are PCA and clustering interpreted carefully?

The project should therefore prioritize storytelling discipline and methodological caveats.

## Core Research Question

Can a borrower-side SME Financing Pain Index reveal credit stress in European SMEs that is not fully captured by market-side stress indicators such as the ECB CISS?

## Storytelling Structure

The project should follow a clear analytical story:

1. **Problem**
   - Market stress indicators such as CISS measure financial-market stress.
   - They may not fully capture the financing pain experienced by small and medium-sized enterprises.

2. **Research Question**
   - Can borrower-side survey data reveal hidden or persistent SME financing stress?

3. **Hypotheses**
   - H1: SAFE borrower-side financing variables contain one common financing-pain dimension.
   - H2: SME_FPI does not always move together with CISS.
   - H3: SME financing pain differs by country, period, firm size, and economic context.
   - H4: A weighted or externally validated version of SME_FPI may improve the baseline equal-weight index.

4. **Model Building**
   - Build a transparent baseline index from SAFE borrower-side variables.
   - Build improved versions using alternative weighting methods.
   - Optionally build an augmented stress index that includes lender-side and macro context variables.

5. **Model Validation**
   - Use PCA loadings to test whether the components share a common dimension.
   - Add a PCA correlation circle to show how variables relate to PC1 and PC2.
   - Use correlations to check whether the index is distinct from CISS.
   - Use clustering to identify financing regimes and improve the cluster plot with centroids, clearer labels, and explained-variance axis labels.
   - Use external variables to test whether the index is associated with future stress outcomes.

6. **Visualization and Insights**
   - Use a small core portfolio of strong visualizations.
   - Keep additional charts as diagnostics or dashboard views.
   - Avoid adding chart types only for variety.

7. **Limitations**
   - Clearly state that the index is descriptive, not causal.
   - CISS is a common euro-area benchmark, not a country-specific SME credit variable.
   - Survey-based indicators contain perception and measurement limitations.

## Recommended Final Storyline

The final notebook, report, and dashboard should follow this storyline explicitly:

```text
Problem -> Hypotheses -> Model -> Validation -> Visual Insights -> Limitations
```

### Problem

Financial-market stress indicators such as CISS can show whether markets are under pressure, but they may not fully show what small and medium-sized firms experience when they try to obtain finance.

The project should open with this contrast:

```text
Market-side stress can fall while borrower-side SME financing pain remains visible.
```

### Hypotheses

Use these hypotheses in the notebook or report introduction:

- **H1: Common factor hypothesis**
  - SAFE borrower-side financing variables share one broad financing-pain dimension.

- **H2: Benchmark-gap hypothesis**
  - SME_FPI does not always move together with CISS, because CISS measures market stress while SAFE measures borrower-side financing pain.

- **H3: Heterogeneity hypothesis**
  - SME financing pain differs by country, time period, and firm characteristics.

- **H4: Methodology improvement hypothesis**
  - A weighted or externally validated version of SME_FPI may improve the baseline equal-weight index.

### Model

Keep the model explanation simple:

```text
Six borrower-side SAFE variables
-> direction alignment
-> z-score standardization
-> available-component average
-> standardized SME_FPI
-> comparison with CISS
```

### Validation

The validation section should answer:

- Do the six variables behave like a coherent index?
- Does PCA support one broad financing-pain dimension?
- Does clustering reveal interpretable financing regimes?
- Is SME_FPI distinct from CISS?
- Are results stable across weighting and missing-data choices?

### Visual Insights

The visual insight section should avoid showing every available chart as if all are equally important.

Use the six core visuals as the main story and move diagnostic visuals into appendix or dashboard views.

### Limitations

Limitations should be stated before the professor has to infer them:

- The index is descriptive, not causal.
- SAFE is survey-based.
- CISS is a common euro-area benchmark, not country-specific SME stress.
- Equal weighting is transparent but simplistic.
- Full-sample standardization is not ideal for real-time monitoring.
- The Q0B severity score is a pragmatic treatment of ordinal response levels.

## Current Data Assets

### Main Analytical Panel

File:

```text
SME_FPI/data/processed/sme_fpi_panel.csv
```

Shape:

```text
386 rows x 28 columns
```

Role:

- Main country-period panel.
- Used to calculate SME_FPI.
- Used for CISS comparison, PCA, clustering, and hidden-gap analysis.

### SAFE Selected Component Long Data

File:

```text
SME_FPI/data/processed/safe_component_long.csv
```

Shape:

```text
11,300 rows x 37 columns
```

Role:

- Long-form SAFE data for selected financing-pain components.
- Useful for checking coverage, firm-size summaries, and component-level behavior.

### SAFE Q0B Big Cube

Raw file:

```text
SME_FPI/data/raw/safe_q0b_pressingness_big_cube.csv
```

Shape:

```text
231,231 rows x 33 columns
```

Processed slim file:

```text
SME_FPI/data/processed/safe_q0b_big_cube_slim.csv
```

Role:

- Big Data Visualization layer.
- Contains country, period, firm size, sector, firm age, problem type, answer level, and weighted response value.
- Used to build severity diagnostics.

### SAFE Problem Severity Cube

File:

```text
SME_FPI/data/processed/safe_problem_severity_cube.csv
```

Shape:

```text
20,570 rows x 11 columns
```

Role:

- Weighted descriptive severity score by country, period, firm group, and problem type.
- Used for Big Data visualizations.

## Current Baseline Index

The current SME_FPI uses six borrower-side SAFE variables:

1. `access_finance_main_problem`
2. `bank_loan_rejected`
3. `bank_loan_cost_too_high`
4. `bank_loan_limited_amount`
5. `bank_willingness_deteriorated`
6. `interest_rates_increased`

All variables are coded so that higher values mean greater financing pain.

### Baseline Formula

Each component is standardized:

```text
z_j(c,t) = (x_j(c,t) - mean(x_j)) / sd(x_j)
```

The raw index is the average of available standardized components:

```text
SME_FPI_raw(c,t) = mean of available z_j(c,t)
```

The index is calculated only when at least four components are available.

The raw index is standardized again:

```text
SME_FPI(c,t) = z(SME_FPI_raw(c,t))
```

The relative gap against CISS is:

```text
Hidden_SME_Stress(c,t) = SME_FPI(c,t) - CISS_z(t)
```

## Current Baseline Strengths

- Transparent and easy to explain.
- All variables come from the same borrower-side survey source.
- All component directions are aligned.
- Z-score standardization makes different variables comparable.
- PCA supports the interpretation of a common financing-pain dimension.
- CISS is kept outside the index and used only as a benchmark.

## Current Baseline Limitations

### Equal-Weight Assumption

The current index assumes all six variables have the same importance.

This is transparent, but potentially unrealistic.

### Available-Component Averaging

The denominator changes when a component is missing.

This is defensible, but it means the index is not perfectly balanced across all country-period observations.

### Full-Sample Standardization

The current z-scores use the full sample mean and standard deviation.

For a personal project, this should be improved because full-sample standardization uses future information when interpreting earlier periods.

### Survey Data Limitation

SAFE measures reported financing conditions, not direct transaction-level credit outcomes.

This is valuable, but it should be described as borrower-side perception and experience.

### CISS Benchmark Limitation

CISS is a common euro-area financial-market stress benchmark.

It is not country-specific SME credit stress.

Therefore, the hidden stress gap should be interpreted as a relative benchmark gap, not direct evidence of local financial-market stress.

### Likert Severity Limitation

SAFE Q0B severity scores treat 1-to-10 survey answers as a descriptive scale.

This is useful for visualization, but it should not be presented as exact cardinal measurement.

## Technical and Theoretical Review

This section records the main risks that should be addressed before turning the project into a polished personal portfolio project.

### Priority 1: Story Focus

Issue:

- The notebook contains many strong charts, but too many visuals can weaken the main story.

Why it matters:

- A data visualization professor will likely reward intentional selection more than chart quantity.

Fix:

- Present six core visuals as the main portfolio.
- Move all other charts into diagnostics, appendix, or dashboard tabs.
- Make every core chart answer a specific part of the research question.

Recommended core visuals:

1. SME_FPI vs CISS line chart.
2. Country-period heatmap.
3. Latest SME-CISS gap bar chart.
4. PCA cluster scatterplot.
5. PCA correlation circle or PCA loadings.
6. Big Data cube problem severity heatmap.

### Priority 2: PCA and Missing Values

Issue:

- PCA and KMeans currently use `fillna(0.0)` after z-score standardization.

Why it matters:

- Because z-scores are centered at zero, this is equivalent to mean imputation in standardized space.
- This is defensible, but it must be explained.

Fix:

- State explicitly:

```text
Missing standardized component values are imputed as 0, which represents the sample mean after z-score standardization.
```

- Add a sensitivity check comparing:
  - available-component index,
  - balanced-sample index,
  - mean-imputed PCA/KMeans version.

### Priority 3: Cluster Count Justification

Issue:

- KMeans currently uses four clusters.

Why it matters:

- Four regimes are interpretable, but the choice should not look arbitrary in a Multidimensional Data Analysis project.

Fix:

- Add one of the following:
  - elbow plot,
  - silhouette score,
  - short interpretability rationale for four clusters.

Recommended framing:

```text
A four-cluster solution is used because it produces interpretable financing regimes:
low pain, interest-rate pressure, loan-cost pressure, and broad financing pain.
```

If possible, support this with silhouette or elbow diagnostics.

### Priority 4: Hidden Stress Terminology

Issue:

- `Hidden_SME_Stress` is a useful narrative label, but it can sound stronger than the data support.

Why it matters:

- CISS is not a country-specific SME credit condition measure.

Fix:

- In formal methodology sections, use:

```text
Relative SME-CISS Gap
```

- In storytelling sections, `hidden SME stress` can be used as a softer interpretive phrase, but always define it as:

```text
SME_FPI_z - common CISS_z
```

Correct interpretation:

```text
The gap indicates when borrower-side SME financing pain is high relative to the common euro-area market-stress benchmark.
```

Avoid:

```text
This proves local hidden financial-market stress.
```

### Priority 5: Full-Sample Standardization

Issue:

- Current z-scores use the full sample mean and standard deviation.

Why it matters:

- This is acceptable for retrospective analysis, but not ideal for real-time monitoring because it uses future information.

Fix:

- Keep full-sample z-scores as the baseline for the course version.
- Add a fixed-baseline version for the personal project:

```text
2015-S1 to 2019-S2 baseline
```

- Compare full-sample and fixed-baseline indexes.

### Priority 6: Equal Weighting

Issue:

- Equal weighting assumes each component contributes equally to financing pain.

Why it matters:

- Transparent, but potentially simplistic.

Fix:

- Keep equal weight as the baseline.
- Add PCA-weighted and reliability-weighted alternatives.
- Compare whether country rankings and time patterns change materially.

### Priority 7: Big Data Cube Interpretation

Issue:

- The big cube is large and high-dimensional, but the core index is still built from the smaller 386-row panel.

Why it matters:

- The Big Data layer must contribute real insight, not only satisfy a size requirement.

Fix:

- Explain clearly:

```text
The 386-row panel builds the main SME_FPI.
The 231,231-row SAFE Q0B cube is used to diagnose the broader problem landscape by country, period, firm size, sector, age, problem type, and response level.
```

- Add top-box share or high-pressure share to complement the mean severity score.

## Variable Expansion Plan

The next version should separate variables into three layers.

## Layer A: Core Borrower-Side SAFE Variables

These variables can be considered for the improved borrower-side index because they directly describe firm-level financing conditions.

Potential additions:

- Availability of bank loans deteriorated.
- Availability of credit lines deteriorated.
- Availability of trade credit deteriorated.
- Collateral requirements increased.
- Other financing costs increased.
- Need for bank loans increased.
- Profits deteriorated.
- Turnover deteriorated.

Decision rule:

- Include only variables that directly represent SME financing pain.
- Avoid adding variables that measure general business performance unless they are used as context or validation.
- Keep the core index conceptually narrow.

## Layer B: Lender-Side Validation Variables

These should usually not be included directly in the borrower-side index.

They should be used to validate whether SME_FPI moves with lender-side tightening.

Potential sources:

- ECB Bank Lending Survey.

Potential variables:

- Credit standards for loans to firms.
- Loan demand.
- Terms and conditions.
- Margins on loans.
- Banks' risk perception.

Use:

- Compare SME_FPI with bank-reported credit tightening.
- Test whether borrower-side pain rises when banks tighten lending standards.
- Use as validation or as a separate lender-side subindex.

## Layer C: Market and Macro Context Variables

These should be used for external validation and economic interpretation.

Potential variables:

- NFC lending rates.
- Loan growth to non-financial corporations.
- GDP growth.
- Unemployment rate.
- HICP inflation.
- Business bankruptcy or insolvency indicators.
- CISS and CISS subcomponents.

Use:

- Test whether SME_FPI is associated with worsening macro-financial conditions.
- Build an augmented index only if the research question expands beyond borrower-side financing pain.

## Proposed Model Versions

## Model 1: Baseline Equal-Weight SME_FPI

Keep the current index as the baseline.

Purpose:

- Transparent benchmark.
- Easy to explain.
- Useful for comparison against improved models.

Formula:

```text
SME_FPI_equal(c,t) = average of available standardized SAFE borrower-side components
```

## Model 2: PCA-Weighted SME_FPI

Use PC1 loadings to weight components.

Formula:

```text
w_j = abs(PC1_loading_j) / sum(abs(PC1_loading_j))

SME_FPI_PCA(c,t) = sum_j w_j z_j(c,t) / sum_j w_j
```

Handling missing values:

```text
SME_FPI_PCA(c,t) = sum over available components / sum of available weights
```

Strength:

- Data-driven.
- Gives more weight to variables that contribute more to the common stress dimension.

Limitation:

- PCA weights maximize variance, not necessarily economic importance.

## Model 3: Reliability-Weighted SME_FPI

Weight variables by their consistency with the overall index.

Candidate diagnostics:

- Correlation of each component with the equal-weight index.
- Leave-one-out correlation.
- Cronbach's alpha contribution.
- Missing-data coverage.

Example weight logic:

```text
reliability_j = corr(z_j, SME_FPI_equal_without_j)
coverage_j = share of non-missing observations
w_j = positive_part(reliability_j) * coverage_j
w_j = w_j / sum(w_j)
```

Strength:

- Penalizes noisy or weakly related components.
- Penalizes variables with poor coverage.

Limitation:

- Still internal validation only.
- May overweight variables that are redundant.

## Model 4: Externally Validated SME_FPI

Choose weights based on external predictive or validation performance.

Possible target variables:

- Future bankruptcy increase.
- Future NFC loan growth decline.
- Future GDP slowdown.
- Future unemployment increase.
- Future BLS credit-standard tightening.

Example target:

```text
Target_Stress(c,t+1)
= average of standardized future bankruptcy increase,
  loan growth decline,
  GDP growth decline,
  and BLS tightening
```

Weight selection:

```text
Choose weights that maximize out-of-sample correlation
between SME_FPI(c,t) and Target_Stress(c,t+1)
```

Strength:

- Stronger personal-project methodology.
- Shows whether the index is useful beyond visual storytelling.

Limitation:

- Requires careful time alignment.
- Risk of overfitting if the sample is small.

## Model 5: Augmented SME Credit Stress Index

This should be separate from the pure borrower-side SME_FPI.

Suggested structure:

```text
Augmented_SME_Credit_Stress(c,t)
= lambda_B * Borrower_Pain(c,t)
 + lambda_L * Lender_Tightening(c,t)
 + lambda_M * Macro_Fragility(c,t)
```

Initial expert weights:

```text
lambda_B = 0.60
lambda_L = 0.25
lambda_M = 0.15
```

Interpretation:

- Borrower_Pain: SAFE borrower-side variables.
- Lender_Tightening: BLS or bank-side credit indicators.
- Macro_Fragility: GDP, unemployment, inflation, bankruptcies, or loan growth.

Important:

- This should not replace SME_FPI.
- It should be presented as a broader stress dashboard index.

## Standardization Improvements

## Option A: Fixed Baseline Standardization

Use a stable pre-shock baseline period, for example:

```text
2015-S1 to 2019-S2
```

Formula:

```text
z_j(c,t) = (x_j(c,t) - mean_baseline_j) / sd_baseline_j
```

Strength:

- Easy to explain.
- Avoids using future information for earlier periods.
- COVID and rate-shock periods become deviations from a normal baseline.

## Option B: Expanding Standardization

Use only information available before or at time `t`.

Formula:

```text
z_j(c,t) = (x_j(c,t) - mean_j(before t)) / sd_j(before t)
```

Strength:

- More realistic for real-time monitoring.

Limitation:

- More complex.
- Early-period values may be unstable.

## Recommended Choice

For this project, use:

```text
Fixed baseline standardization: 2015-S1 to 2019-S2
```

Keep full-sample z-scores as a comparison in the appendix.

## Missing Data Improvement Plan

Create three versions:

1. **Available-component version**
   - Current approach.
   - Uses available standardized components.

2. **Balanced-component version**
   - Uses only observations with all required components.
   - Cleaner but smaller sample.

3. **Imputed version**
   - Uses simple interpolation or model-based imputation.
   - Should be treated as sensitivity analysis, not the main result.

Compare:

- Correlation among versions.
- Ranking stability by country.
- Time-series similarity.
- Impact on PCA and clustering.

## Big Data Cube Improvement Plan

The current severity score is:

```text
severity_score = weighted average of answer levels 1 to 10
```

Keep this, but add two alternative measures:

### Top-Box Share

Share of responses rated 8 to 10:

```text
top_box_share = share(L8 + L9 + L10)
```

Interpretation:

- Measures severe pressure.
- Avoids relying only on the exact interval interpretation of 1-to-10 answers.

### High-Pressure Share

Share of responses rated 7 to 10:

```text
high_pressure_share = share(L7 + L8 + L9 + L10)
```

Interpretation:

- More inclusive measure of high concern.

Use:

- Compare severity mean, top-box share, and high-pressure share.
- If all three show similar patterns, the big-cube insight is more robust.

## Visualization Strategy

The core story should not use every chart.

For Anna Kovbasiuk's courses, the visual strategy should emphasize reasoning:

- Use a chart only when it answers a specific analytical question.
- Explain why the chart type fits the data structure.
- Use color to encode meaning, not decoration.
- Keep exact rankings in bar charts rather than maps.
- Use maps for spatial overview, not precise comparison.
- Use MDA visuals to explain structure, not to impress with complexity.

Use six core figures:

1. SME_FPI vs CISS line chart.
2. Country-period heatmap.
3. Latest SME-CISS gap bar chart.
4. PCA cluster scatterplot.
5. PCA loadings chart or PCA correlation circle.
6. Big Data cube severity heatmap.

Optional diagnostic figures:

- PCA correlation circle, if the loadings bar chart is kept as the core PCA figure.
- Correlation heatmap.
- Parallel coordinates plot.
- Bubble chart.
- Choropleth map.
- Firm-size component heatmap.
- General-audience redesign figure.
- Big cube trend chart.

## Final Narrative Visual Order

Recommended final order for the main story:

1. **Line chart: SME_FPI vs CISS**
   - Purpose: problem statement.
   - Message: borrower-side pain and market-side stress do not always move together.

2. **Country-period heatmap**
   - Purpose: temporal and geographic pattern.
   - Message: SME financing pain is uneven across countries and time.

3. **Latest SME-CISS gap bar chart**
   - Purpose: precise ranking.
   - Message: some countries have higher borrower-side pain relative to the common CISS benchmark.

4. **PCA correlation circle or PCA loadings**
   - Purpose: model validation.
   - Message: the six variables form a coherent financing-pain dimension.

5. **Improved PCA cluster plot**
   - Purpose: multidimensional regime discovery.
   - Message: country-periods fall into interpretable financing regimes.

6. **Big Data cube severity heatmap**
   - Purpose: Big Data diagnostic insight.
   - Message: the broader SAFE cube reveals which firm groups and problem categories carry the strongest pressure.

Figures that should usually be moved to appendix or dashboard:

- Bubble chart.
- Choropleth map.
- Correlation heatmap.
- Parallel coordinates plot.
- Boxplot by regime.
- General-audience redesign.
- Access-to-finance marginal breakdowns.
- Big cube trend chart.

## Visualization Quality Improvements

### Line Chart

Add cross-country uncertainty or dispersion band.

Recommended label:

```text
Interquartile range across countries
```

Avoid calling it a confidence interval unless a valid sampling-error model is available.

### Heatmaps

Keep diverging color scales centered at zero for standardized variables.

Use sequential color scales only for severity scores.

### PCA Correlation Circle

Add a PCA correlation circle as a methodology and validation figure.

Purpose:

- Show how the six financing-pain variables relate to PC1 and PC2.
- Check whether variables point in a consistent direction.
- Make the interpretation of PC1 and PC2 more transparent than a loadings table alone.

Expected interpretation:

- Access-to-finance, bank willingness, loan rejection, and limited loan amount should mostly align with PC1.
- Cost-too-high and interest-rate-increased may contribute more to PC2.
- PC1 can be interpreted as broad SME financing pain.
- PC2 can be interpreted as a cost or interest-rate pressure dimension, if the loading pattern supports it.

Use:

- Keep either the PCA loadings bar chart or the PCA correlation circle in the core six-figure narrative.
- Put the other one in the validation appendix or dashboard methodology page.

Design notes:

- Draw the unit circle.
- Use arrows from the origin to each variable coordinate.
- Label PC1 and PC2 with explained variance percentages.
- Avoid overcrowding labels; use short variable names.

### PCA Cluster Plot

Keep the PCA cluster scatterplot as a core visual, but improve it rather than adding a redundant second cluster chart.

Recommended improvements:

- Add cluster centroids.
- Label PC1 and PC2 with explained variance percentages.
- Use shorter, cleaner regime names.
- Label only the most extreme observations to avoid clutter.
- Consider confidence ellipses or convex hulls only if they improve readability.
- Keep the boxplot by regime as a supporting validation chart.

### Bubble Chart

Clarify that CISS is common by half-year, so vertical bands are expected.

### Choropleth Map

Use as overview only.

Pair with bar chart for exact ranking.

### Treemap

Do not use unless a true part-to-whole variable is created.

Current SAFE severity variables are not clean parts of a single total.

## Dashboard Plan

Recommended framework:

```text
Dash + Plotly
```

Reason:

- Open source.
- Python-native.
- Works well with pandas and Plotly.
- More natural for Plotly hover interactions, animated frames, and 3D figures.
- Suitable for a personal portfolio dashboard.

## Dashboard Tabs

### Tab 1: Overview

Content:

- Research question.
- Key metrics.
- SME_FPI plain-English definition.
- Dataset shape.
- Monitoring-style threshold notes.
- Storytelling logic.

### Tab 2: Explorer

Controls:

- Country selector.
- Period range selector.
- Index version selector.

Charts:

- SME_FPI vs CISS time series.
- Weighting robustness comparison.
- Animated stress motion over time.
- Country-period heatmap.
- Component heatmap.

### Tab 3: Hidden Stress

Charts:

- Choropleth map.
- Relative SME-CISS gap ranking.
- Bubble diagnostic.

### Tab 4: Regimes

Charts:

- PCA correlation circle.
- PCA cluster scatterplot.
- Exploratory 3D PCA regime plot.

### Tab 5: Big Data Cube

Controls:

- Problem type.
- Firm size.
- Sector.
- Country.
- Period.

Charts:

- Severity heatmap.
- Top-box share heatmap.
- Trend chart.

### Tab 6: Methodology

Content:

- Variables.
- Equations.
- Weighting methods.
- Missing data handling.
- Limitations.

## Dashboard Implementation Update

Implemented in:

```text
dashboard/app.py
dashboard/assets/style.css
```

Current dashboard design decisions:

- Use one global control area for countries, index version, and period range.
- Move SAFE problem, firm size, and sector controls into the Big Data Cube tab because those filters only affect the survey-cube diagnostics.
- Keep all countries selected by default, but switch crowded charts to summary mode where needed.
- Use tabs instead of one long report page: Overview, Explorer, Hidden Stress, Regimes, Big Data Cube, and Methodology.
- Add an animated stress-motion chart to show country movement through time.
- Add a 3D PCA regime explorer as an exploratory multidimensional view, with a clear caveat that it is not the primary evidence.
- Add figure notes under charts so a non-specialist viewer can understand what each chart shows, why the chart type was chosen, and the key insight.
- Add context guides for SAFE, CISS, the relative SME-CISS gap, and the four index versions so a data-visualization evaluator can read the finance context without prior domain knowledge.

## Three-Course Alignment Checklist

### Data Visualization

The final version should demonstrate:

- deliberate chart choice,
- clean visual hierarchy,
- appropriate color scales,
- readable labels and annotations,
- clear figure captions,
- avoidance of unnecessary chart types,
- explanation of chart limitations.

High-value evidence:

- before/after redesign figure,
- core six-visual narrative,
- chart-type rationale under each figure,
- map used only as overview,
- bar chart used for exact ranking.

### Multidimensional Data Analysis

The final version should demonstrate:

- PCA as dimensionality reduction,
- explained variance interpretation,
- PCA loadings or correlation circle,
- clustering of country-period observations,
- cluster profile interpretation,
- cluster validation or interpretability rationale,
- sensitivity checks across index versions.

High-value evidence:

- PCA correlation circle,
- improved PCA cluster plot with centroids,
- boxplot by regime,
- parallel coordinates profile plot,
- elbow or silhouette diagnostic.

### Big Data Visualization and Analysis

The final version should demonstrate:

- use of the 231,231-row SAFE Q0B high-dimensional survey cube,
- explanation of the difference between the main panel and the big cube,
- visualization of problem severity by firm size, country, period, and problem category,
- honest treatment of ordinal survey severity scores,
- optional top-box or high-pressure share as robustness checks.

High-value evidence:

- big cube summary table,
- severity heatmap,
- top-box share heatmap,
- big cube trend chart,
- clear note that the big cube is diagnostic rather than the core index-building table.

## Implementation Roadmap

## Phase 0: Narrative Refactor

Tasks:

- Add an explicit analytical story section to the notebook:
  - problem,
  - hypotheses,
  - model,
  - validation,
  - insights,
  - limitations.
- Mark six figures as the main story visuals.
- Move extra figures into diagnostics or appendix wording.
- Rename or explain `Hidden_SME_Stress` as `Relative SME-CISS Gap`.
- Add a short "How this project satisfies the three courses" note.

Outputs:

- Updated notebook introduction.
- Clear core-vs-diagnostic visual structure.
- Improved figure captions and interpretation cells.

## Phase 1: Methodology Refactor

Tasks:

- Keep current equal-weight index as baseline.
- Add fixed-baseline z-score option.
- Add PCA-weighted index.
- Add reliability-weighted index.
- Add PCA correlation circle output.
- Improve PCA cluster plot with centroids, explained-variance labels, and clearer regime names.
- Add elbow or silhouette diagnostics for the four-cluster solution.
- Add explicit note that PCA/KMeans missing values are mean-imputed in standardized space.
- Add sensitivity checks.

Outputs:

- `sme_fpi_panel_v2.csv`
- `weighting_comparison.csv`
- `index_sensitivity_summary.csv`
- `pca_correlation_circle.png`
- `pca_cluster_regimes_v2.png`

## Phase 2: Variable Expansion

Tasks:

- Identify additional SAFE borrower-side variables.
- Add lender-side validation variables if available.
- Add macro context variables.
- Align all data to country-half-year panel.

Outputs:

- `external_validation_panel.csv`
- `macro_context_panel.csv`
- `data_dictionary_v2.md`

## Phase 3: External Validation

Tasks:

- Compare SME_FPI versions against external targets.
- Test correlations and lead-lag relationships.
- Evaluate whether v2 improves over baseline.

Outputs:

- `validation_results.csv`
- `validation_figures/`
- `SME_FPI_v2_methodology.md`

## Phase 4: Dashboard

Tasks:

- Build Dash + Plotly dashboard.
- Add tab navigation.
- Add global controls and local Big Data Cube controls.
- Add hover diagnostics, animated stress motion, and 3D PCA explorer.
- Add chart explanations and methodology caveats.

Outputs:

- `dashboard/app.py`
- `dashboard/assets/style.css`

## Phase 5: Final Portfolio Polish

Tasks:

- Reduce final narrative to six core visuals.
- Move diagnostics to appendix or dashboard.
- Improve figure captions.
- Add limitations and methodology notes.

Outputs:

- Final notebook.
- Dashboard.
- Technical methodology document.
- Project README.

## Recommended Next Step

The next concrete step should be:

```text
1. Refactor the notebook narrative around Problem -> Hypotheses -> Model -> Validation -> Insights -> Limitations.
2. Add PCA correlation circle and improve the PCA cluster plot.
3. Implement SME_FPI_v2 with alternative weighting methods:
   - equal weight
   - PCA weight
   - reliability weight
   - fixed-baseline z-score
4. Add big-cube robustness measures:
   - severity mean
   - top-box share
   - high-pressure share
```

This is the best next step because it improves the current story and methodology without depending on new external data collection.

The dashboard has now been implemented as a Dash + Plotly observatory. The next improvement should focus on polishing the notebook/report narrative and deciding whether additional external validation variables are worth adding.

## Final Judgement

Current project quality:

```text
Strong enough for course submission and promising as a personal portfolio project.
```

To satisfy Anna Kovbasiuk across all three courses, the final version should avoid looking like a collection of many charts. It should look like a deliberate analytical product:

```text
one research question,
one index,
one validation framework,
one visual story,
one big-data diagnostic layer.
```

The highest-impact improvements are:

1. Make the storytelling structure explicit.
2. Use six core visuals and move extras to diagnostics.
3. Add PCA correlation circle.
4. Improve PCA cluster plot.
5. Add index sensitivity checks.
6. Add top-box or high-pressure share for the big cube.
7. Treat CISS gap language carefully.

If these improvements are implemented, the project should feel substantially stronger than a standard class assignment.
