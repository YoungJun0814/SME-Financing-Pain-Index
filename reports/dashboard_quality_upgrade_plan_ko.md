# SME-FPI Dashboard Quality Upgrade Review and Execution Plan

작성일: 2026-04-30

## 1. Executive Summary

현재 SME Financing Pain Observatory는 이미 단순 수업용 시각화보다 한 단계 위에 있다. 핵심 강점은 명확하다.

- SME-FPI Core를 ECB SAFE borrower-side 변수로만 유지한다.
- CISS와의 차이를 통해 market stress와 borrower pain의 불일치를 설명한다.
- Forecast Lab이 Elastic Net, Ridge, Random Forest, Gradient Boosting을 naive뿐 아니라 country mean, country AR(1), momentum, pooled lag OLS와 비교한다.
- Decision Board와 Country Diagnosis가 추가되어 "그래서 어떤 나라를 봐야 하는가?"에 답하기 시작했다.
- Macro, micro, BLS, MIR, Eurostat 데이터가 core index가 아니라 forecasting/validation layer로 분리되어 있다.

그러나 더 높은 수준으로 가려면 목표를 분명히 바꿔야 한다.

> 현재 대시보드는 "좋은 연구형 대시보드"다. 다음 단계는 "정책/금융 분석가가 바로 쓸 수 있는 monitoring product"처럼 보이게 만드는 것이다.

가장 중요한 개선 방향은 더 많은 차트가 아니라 다음 네 가지다.

1. Decision Board를 진짜 첫 화면으로 완성한다.
2. Forecasting을 더 정직하고 강한 validation story로 만든다.
3. Country Diagnosis를 "왜 위험한가?"에 답하는 증거 묶음으로 강화한다.
4. UI/UX를 설명 중심에서 판단 중심으로 조정한다.

## 2. Current State Snapshot

현재 산출물 기준:

- Main panel: 386 country-half-year observations.
- Forecast feature panel: 386 rows, 188 columns.
- Forecast evaluation: 216 rolling-origin model rows.
- Model suite: 9 models.
- Recent best ML model: Ridge.
- Recent Ridge MAE: 0.326.
- Strongest recent simple baseline: Pooled lag OLS, MAE 0.340.
- ML edge versus strongest baseline: 0.014.
- Latest Decision Board: 12 countries.
- Latest tiers: 6 Monitor, 6 Normal, 0 Watch, 0 Alert.

Interpretation:

- Ridge winning is theoretically plausible because the panel is small and correlated.
- The ML edge over pooled lag OLS is positive but small. This must be framed carefully.
- The lack of Watch/Alert in 2025-S1 is not a failure. It means the current endpoint is not a broad crisis moment under the chosen rules. But the dashboard should let viewers inspect historical alert periods so the tiering system feels meaningful.

## 3. Highest-Level Evaluation

### Overall Grade

- Research question: 9/10
- Data architecture: 8.5/10
- Index construction: 8/10
- Forecasting honesty: 8/10
- Forecasting rigor: 7/10
- Decision usefulness: 7/10
- UI/UX polish: 7/10
- Theoretical caution: 8/10
- Portfolio/presentation impact: 8/10

Overall: strong 8/10. With the upgrades below, it can look like a 9/10 project.

### Main Diagnosis

The dashboard has moved from descriptive analytics toward early-warning monitoring, but it still needs a sharper "analysis to action" chain.

The target chain should be:

```text
Data source -> Index construction -> Hidden gap -> Risk tier -> Country driver -> Forecast agreement -> Validation evidence -> Caveat
```

Right now the chain exists, but it is spread across tabs. The next upgrade should make the chain visible without requiring the reader to discover it.

## 4. Product and Storytelling Improvements

### 4.1 First Screen Should Answer Four Questions

The first screen should immediately answer:

1. Who needs attention?
2. Why is that country flagged?
3. Is the stress hidden from CISS?
4. Do forecast models agree that pressure may rise?

Current status:

- Decision Board answers this partly.
- It still feels like an added tab rather than the command center of the product.

Recommended improvement:

- Move Decision Board to default tab permanently.
- Add a compact "Current read" panel at the top:

```text
Latest period: 2025-S1
Current status: no Alert/Watch countries, 6 Monitor countries
Main hidden-stress monitors: Greece, Finland, Ireland
Main rising-model monitors: France, Belgium, Germany
Best recent ML model: Ridge
ML edge vs strongest baseline: modest positive
```

Acceptance criteria:

- A first-time viewer can understand the current system status in under 20 seconds.
- The dashboard does not require scrolling to know whether the latest period is calm or concerning.
- The wording avoids panic when there are no Watch/Alert countries.

### 4.2 Add "Latest Interpretation" Narrative

The dashboard should generate a short text summary from the data.

Example:

```text
In the latest period, no country reaches Watch or Alert. Greece, Finland, and Ireland remain Monitor cases because borrower-side pain is high relative to CISS. France, Belgium, and Germany are Monitor cases mainly because the model suite expects some upward pressure from low current levels. This should be read as a monitoring signal, not a crisis call.
```

Why it matters:

- This makes the project feel analyst-led rather than chart-led.
- It protects against overinterpreting model points.
- It gives the user a script for presentations.

Implementation:

- Add function `decision_narrative(board)` in `dashboard/app.py`.
- Return a paragraph using counts, top hidden-gap countries, top rising forecast countries, best model, and baseline edge.
- Render it above Decision Board figures.

## 5. Theoretical and Research Design Improvements

### 5.1 Clarify the Index Identity Even More

Current strength:

- SME-FPI Core remains borrower-side SAFE only.
- Macro/BLS/MIR/Eurostat are predictors or validation context.

Remaining issue:

- Some hero and pipeline text still says "BLS/MIR forecast layer" or "BLS + MIR", while the actual layer now includes Eurostat. This creates small but visible inconsistency.

Action:

- Replace all dashboard-facing "BLS/MIR forecast layer" with "BLS/MIR/Eurostat forecast layer".
- Replace pipeline step "BLS + MIR" with "BLS + MIR + Eurostat".
- Add one sentence in Overview:

```text
Eurostat business-demography data are broad real-economy context, not SME-specific credit outcomes.
```

Acceptance criteria:

- No visible text implies Eurostat is part of SME-FPI Core.
- No visible text omits Eurostat from the forecasting layer.

### 5.2 Add a "What This Does Not Claim" Panel

High-quality dashboards are explicit about limits.

Add a small panel in Methodology or Decision Board:

```text
This dashboard does not claim:
- causal impact of financing conditions,
- firm-level default prediction,
- country-specific market stress measurement,
- production-grade credit risk scoring,
- policy action without local context.
```

Why:

- It makes the project sound more professional.
- It prevents the forecasting tab from being judged as overclaiming.

### 5.3 Real-Time Standardization Caveat

Issue:

- Full-sample z-scores are useful retrospectively but not ideal for real-time monitoring.

Current mitigation:

- Fixed-baseline version exists.

Upgrade:

- In Decision Board, show a small "real-time caution" badge if equal-weight full-sample index is selected.
- Add a toggle or text comparison: current Decision Board under equal-weight vs fixed-baseline.

Implementation idea:

- Add a secondary chart or mini cards showing whether top Monitor/Watch countries change under fixed-baseline SME-FPI.

Acceptance criteria:

- Reader sees that full-sample standardization is not hidden.
- Fixed-baseline robustness is not buried in Explorer only.

## 6. Forecasting Improvements

### 6.1 Show the Small ML Edge Honestly

Current result:

- Ridge MAE 0.326.
- Pooled lag OLS MAE 0.340.
- Improvement is 0.014.

This is positive but small.

Recommendation:

- Forecast Lab should say:

```text
Best recent ML model improves modestly over the strongest simple baseline.
```

Do not say:

```text
ML strongly outperforms all baselines.
```

Implementation:

- Add "Effect size" card:
  - "ML edge vs strongest baseline: 0.014 MAE"
  - Label as "modest", "meaningful", or "weak" based on thresholds.

Suggested thresholds:

```text
edge < 0.01: weak
0.01 <= edge < 0.05: modest
edge >= 0.05: meaningful
```

### 6.2 Add Forecast Stability Over Time

Current issue:

- Forecast Lab shows recent average model comparison but not whether the best model changes over time.

Why it matters:

- A top evaluator will ask whether Ridge is consistently good or only recently good.

Add chart:

- Rolling-origin MAE line by model family.
- Or heatmap: origin period x model, colored by rank.

Recommended chart:

```text
Model rank heatmap
Rows: model
Columns: rolling-origin period
Color: rank or MAE
```

Acceptance criteria:

- Viewer can see model stability and whether tree models fail only in recent windows or throughout.

### 6.3 Add Country-Level Forecast Error

Current issue:

- Model evaluation is aggregate by test window.
- Country-level reliability is not visible.

Add:

- Country average absolute forecast error.
- Country-specific bias: predicted too high or too low.
- Number of evaluation windows per country.

Why:

- Some countries may be systematically harder to forecast.
- Decision Board confidence should eventually account for country-level forecast error.

Implementation:

- Extend `forecasting_model_evaluation.csv` or add `forecasting_country_error.csv`.
- During rolling evaluation, store country-level predictions.
- Aggregate by country and model.

Acceptance criteria:

- Country Diagnosis card can show:

```text
Historical forecast reliability for this country: medium
Average H+1 error: 0.31
Bias: model tends to underpredict pressure
```

### 6.4 Calibrate Risk Tiers Using Historical Outcomes

Current issue:

- Risk tier thresholds are heuristic.
- That is okay, but should be evaluated.

Upgrade:

- Backtest the risk tiers.
- For each historical origin:
  - compute tier,
  - observe next-period SME-FPI,
  - observe whether pressure rose,
  - observe whether future access severity rose.

Metrics:

- Average next H+1 SME-FPI by tier.
- Share of countries with rising pressure by tier.
- Precision for "Watch/Alert".
- False positive and false negative counts.

Acceptance criteria:

- Methodology can say:

```text
Historical Watch/Alert periods are followed by higher next-period pressure on average.
```

Or, if not true:

```text
Tiers are descriptive triage labels rather than validated warning classifications.
```

Either result is useful because it is honest.

## 7. Data Improvements

### 7.1 Add European Commission Business Confidence

Priority: high.

Why:

- It provides expectations and business-cycle sentiment.
- It is closer to forward-looking business conditions than annual World Bank macro variables.

Use:

- Forecasting predictor.
- External validation context.

Keep out of:

- SME-FPI Core.

Suggested variables:

- business confidence indicator,
- industry confidence,
- services confidence,
- construction confidence,
- economic sentiment indicator.

Acceptance criteria:

- Source catalog includes European Commission data as implemented.
- Forecast feature panel gains monthly/quarterly confidence predictors.
- Methodology states these are expectation/context predictors only.

### 7.2 Better Eurostat Integration

Current status:

- Eurostat bankruptcies and registrations are included.

Needed improvement:

- Add coverage diagnostics because not all countries/periods have equal coverage.
- Show latest available Eurostat period relative to SAFE latest period.
- Add caveat in dashboard UI, not only docs.

Acceptance criteria:

- Dashboard source catalog shows rows and role.
- Methodology has a visible caveat:

```text
Eurostat business-demography indicators are broad economy signals, not SME-specific financing outcomes.
```

### 7.3 Add Data Coverage Score

Current issue:

- Component coverage is shown in hover, but feature coverage is not central.

Add:

- Data coverage score per country-period:
  - core component coverage,
  - macro coverage,
  - micro coverage,
  - BLS/MIR/Eurostat coverage.

Use:

- Decision Board confidence.
- Country Diagnosis.

Acceptance criteria:

- A country is not high-confidence only because models agree; it also needs adequate data coverage.

## 8. UI/UX Improvements

### 8.1 Reduce Cognitive Load on First View

Current issue:

- There is a lot of explanatory text.
- This is useful for grading but can dilute the product feel.

Recommendation:

- Keep explanations, but move some into collapsible "Method note" drawers.
- First screen should prioritize:
  - current status,
  - watchlist,
  - top drivers,
  - model confidence.

Acceptance criteria:

- Above the fold: user sees status and watchlist before long methodology copy.
- Methodology remains available but not visually dominant.

### 8.2 Update Hero and Badges

Current issue:

- Hero badge still says "BLS/MIR forecast layer".
- Pipeline step says "BLS + MIR".

Action:

- Replace with:

```text
BLS/MIR/Eurostat forecast layer
BLS + MIR + Eurostat
```

Also add:

```text
Decision Board
9-model forecast suite
```

Why:

- The first viewport should reflect the current strongest features.

### 8.3 Make Tabs More Product-Like

Current tab order is good:

1. Decision Board
2. Overview
3. Explorer
4. Forecast
5. Country Diagnosis
6. Hidden Stress
7. Regimes
8. Big Data Cube
9. Methodology

Potential issue:

- Nine tabs can feel heavy.

Alternative:

Group into three sections:

```text
Monitor: Decision Board, Country Diagnosis, Forecast
Explore: Overview, Explorer, Hidden Stress, Regimes, Big Data Cube
Audit: Methodology
```

Implementation option:

- Keep current tabs for simplicity.
- Add small section labels or reorder:

1. Decision Board
2. Country Diagnosis
3. Forecast
4. Hidden Stress
5. Explorer
6. Regimes
7. Big Data Cube
8. Overview
9. Methodology

Recommended:

- Keep Overview second for academic readability.
- But in presentation, start at Decision Board.

### 8.4 Improve Risk Tier Colors and Labels

Current:

- Alert red, Watch amber, Monitor blue, Normal gray.

Good.

Potential issue:

- If latest period has no Alert/Watch, the dashboard might look less dramatic.

Solution:

- Add "Historical alert periods" mini chart.
- Do not artificially lower thresholds to produce Alert/Watch.

Acceptance criteria:

- The dashboard remains honest even when latest status is mostly Monitor/Normal.
- Historical alerts show that the system can identify severe periods.

## 9. Visualization Improvements

### 9.1 Add Historical Risk Timeline

Chart:

- Country x period heatmap of risk tier.

Use:

- Decision Board or Country Diagnosis.

Why:

- Shows whether current Monitor status is calm relative to past crises.
- Gives risk-tier rules historical meaning.

Acceptance criteria:

- Viewer can see historical Alert/Watch clusters, likely around crisis/pandemic/rate-shock periods if rules capture them.

### 9.2 Add Driver Contribution Bar

Current:

- Driver heatmap shows relative drivers.

Upgrade:

- For selected country or top country, show top 5 drivers as horizontal bars.

Challenge:

- Model explainability is not yet SHAP-based.

Recommended safe version:

- Do not claim model causal contribution.
- Use "diagnostic driver score" from standardized current variables and gap/forecast metrics.

Label:

```text
Diagnostic drivers, not causal model attributions
```

### 9.3 Improve Forecast Figure

Current:

- Left panel current vs H+1 forecast.
- Right panel recent model MAE.

Add:

- Indicate strongest simple baseline directly in color or annotation.
- Add "lower is better" and "baseline" grouping.
- Show rank, not only MAE, if space is tight.

Acceptance criteria:

- Viewer immediately sees Ridge is only modestly ahead of pooled lag OLS.

## 10. Technical Engineering Improvements

### 10.1 Split `dashboard/app.py`

Current issue:

- `app.py` is large and mixes:
  - constants,
  - data loading,
  - modeling,
  - figure builders,
  - layout,
  - callbacks.

Refactor target:

```text
dashboard/
  app.py
  data.py
  modeling.py
  risk.py
  figures/
    decision.py
    forecast.py
    explorer.py
    regimes.py
    big_cube.py
  layout/
    components.py
    tabs.py
```

Why:

- Easier debugging.
- Easier testing.
- Better portfolio impression.

Acceptance criteria:

- `app.py` becomes a thin entrypoint.
- No behavioral changes after refactor.
- Existing 8051 dashboard still renders all tabs.

### 10.2 Add Automated Smoke Tests

Add tests:

- data files exist,
- no target leakage in features,
- all forecast models appear,
- Decision Board has expected columns,
- all Dash figure builders return non-empty figures,
- dashboard layout contains all tabs.

Suggested file:

```text
tests/test_dashboard_smoke.py
```

Acceptance criteria:

```text
python -m pytest tests/test_dashboard_smoke.py
```

passes.

### 10.3 Add Reproducibility Script

Add:

```text
scripts/run_full_pipeline.py
```

or a PowerShell script:

```text
scripts/run_full_pipeline.ps1
```

Purpose:

- Downloads data.
- Builds panel.
- Builds validation.
- Builds forecasting layer.
- Runs smoke checks.

Acceptance criteria:

- One command rebuilds core outputs.
- README points to it.

## 11. Performance Improvements

### 11.1 Cache Expensive Model Refit

Current issue:

- Dashboard refits models dynamically for selected period.
- With current size it is acceptable, but interaction may slow as features grow.

Improvement:

- Precompute origin-period predictions for all periods and all models.
- Store:

```text
forecast_model_predictions_all_origins.csv
decision_board_all_origins.csv
```

Then dashboard reads instead of refitting.

Acceptance criteria:

- Forecast tab responds quickly.
- Model results are exactly reproducible from pipeline outputs.

### 11.2 Reduce Browser Render Cost

Potential issue:

- Many Plotly graphs across tabs can be heavy.

Improvement:

- Lazy render tab content if needed.
- Or keep current approach but measure load time.

Acceptance criteria:

- Initial load under 3 seconds locally.
- Tab switch under 1 second for most tabs.

## 12. Accessibility and Polish

### 12.1 Color Accessibility

Check:

- Red/blue diverging scale.
- Amber watch badges.
- Text contrast in hero and cards.

Action:

- Add non-color encoding:
  - tier labels,
  - symbols,
  - text annotations.

Acceptance criteria:

- Risk tier can be understood without color.

### 12.2 Mobile and Small Screens

Current CSS has responsive rules.

Needed:

- Verify 390px mobile viewport.
- Check tab wrapping.
- Check long table overflow.
- Check diagnosis cards text overflow.

Acceptance criteria:

- No button/text overlap.
- Tables scroll horizontally.
- Cards remain readable.

## 13. Portfolio and Presentation Improvements

### 13.1 Add a One-Page "How to Present This" Guide

Add file:

```text
reports/presentation_script_sme_fpi_dashboard.md
```

Content:

- 60-second overview.
- 3-minute walkthrough.
- 7-minute technical explanation.
- expected professor questions and answers.

Key message:

```text
SME-FPI helps identify countries where SME borrower-side financing pain is high, hidden from market-stress indicators, or likely to require monitoring next period.
```

### 13.2 Add "Reviewer Questions" Section

Likely questions:

1. Why not include BLS/MIR/Eurostat in the core index?
2. Why use z-scores?
3. Why equal weights?
4. Why not causal modeling?
5. Why use ML with only 386 observations?
6. What does Ridge beating pooled lag OLS by 0.014 mean?
7. Are Decision Board thresholds validated?
8. What are the main limitations?

Prepare concise answers.

## 14. Prioritized Execution Plan

### Phase 1: Narrative and UI Consistency

Estimated time: 1-2 hours.

Tasks:

1. Update hero badges:
   - "BLS/MIR forecast layer" -> "BLS/MIR/Eurostat forecast layer".
   - Add "Decision Board" and "9-model forecast suite" if space allows.
2. Update Overview pipeline:
   - "BLS + MIR" -> "BLS + MIR + Eurostat".
3. Add latest interpretation paragraph to Decision Board.
4. Add "What this does not claim" panel.
5. Re-check dashboard layout and DOM.

Acceptance criteria:

- No visible text conflicts with the current data architecture.
- First screen explains latest status.

### Phase 2: Historical Risk Tier Backtest

Estimated time: 3-5 hours.

Tasks:

1. Extend forecasting script to generate `decision_board_all_origins.csv`.
2. Compute risk tiers for every rolling-origin period.
3. Add metrics:
   - H+1 average SME-FPI by tier,
   - H+1 rise rate by tier,
   - future access severity by tier where available.
4. Add historical tier heatmap.
5. Add tier validation cards.

Acceptance criteria:

- The dashboard can defend the Decision Board thresholds empirically.
- If thresholds are weak, the dashboard says so honestly.

### Phase 3: Forecast Reliability Deepening

Estimated time: 4-6 hours.

Tasks:

1. Store country-level rolling predictions.
2. Create `forecasting_country_error.csv`.
3. Add country forecast reliability to Country Diagnosis.
4. Add model rank heatmap over rolling-origin windows.
5. Add forecast bias by country.

Acceptance criteria:

- A reviewer can see whether the model is reliable for each country.
- Forecast Lab no longer relies only on aggregate MAE.

### Phase 4: Data Expansion

Estimated time: 4-8 hours depending on API access.

Tasks:

1. Add European Commission business confidence data.
2. Align monthly data to half-year periods.
3. Add coverage diagnostics.
4. Update source catalog and data dictionary.
5. Rebuild forecasting layer and compare performance.

Acceptance criteria:

- New data improves either interpretability or forecast performance.
- If performance does not improve, keep the data only if it improves explanation.

### Phase 5: Codebase Refactor and Testing

Estimated time: 4-8 hours.

Tasks:

1. Split `dashboard/app.py` into modules.
2. Add smoke tests.
3. Add leakage tests.
4. Add pipeline rebuild script.
5. Add CI-like local check command in README.

Acceptance criteria:

- `python -m pytest` passes.
- `python dashboard/app.py` or `python dashboard/run_8051.py` still works.
- No feature leakage into forecast model.

### Phase 6: Final Polish

Estimated time: 2-4 hours.

Tasks:

1. Browser QA at desktop and mobile widths.
2. Check all tabs for text overflow.
3. Reduce excessive explanatory text above the fold.
4. Add presentation script.
5. Update README screenshots or figure references if needed.

Acceptance criteria:

- Dashboard feels like a polished monitoring product.
- Portfolio documentation can answer technical and theoretical questions.

## 15. Recommended Final Target State

The final dashboard should let the viewer say:

```text
This project builds a borrower-side SME financing pain index from SAFE survey data, compares it with the ECB CISS market-stress benchmark, and uses macro, micro, lender-side, loan-market, and business-demography predictors to flag countries that deserve monitoring. The dashboard separates the core index from forecasting predictors, validates the signal against future outcomes, compares ML against strong simple baselines, and shows country-level diagnostic drivers with uncertainty.
```

That is the standard to aim for.

## 16. Immediate Next Actions

If continuing implementation, do these first:

1. Fix remaining visible BLS/MIR-only labels in hero and Overview pipeline.
2. Add Decision Board narrative text.
3. Generate historical decision-board backtest outputs.
4. Add risk-tier history heatmap.
5. Add country-level forecast error table.
6. Add smoke tests and leakage tests.

The single most valuable next feature is:

> Historical risk-tier validation.

It would turn the Decision Board from a reasonable heuristic into something the project can defend empirically.

