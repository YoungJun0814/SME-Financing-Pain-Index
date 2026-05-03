# SME-FPI Dashboard UI/UX Final Improvement Blueprint

작성일: 2026-05-03  
대상: SME Financing Pain Observatory Dash dashboard  
목적: 현재 UI/UX 개선이 충분한지 판단하고, 남은 개선 여지를 빠짐없이 정리하여 최종 완성 기준과 실행 계획을 고정한다.

## 0. 결론

현재까지의 개선은 충분한가?

부분적으로는 충분하지만, 최종 품질로는 아직 충분하지 않다.

이미 해결된 것:

- Interactive Navigator의 구조적 실패를 command-panel 방식으로 1차 교체했다.
- Five-minute reading path의 7번째 카드가 다음 줄로 떨어지는 문제를 timeline 구조로 바꿨다.
- Reader mode, recommended action, quick jump, reading path click interaction을 추가했다.
- smoke test, compile check, Dash layout response, browser interaction check를 통과했다.

하지만 아직 남은 핵심 과제:

- 전체 dashboard가 하나의 고급 분석 제품처럼 보이려면 시각 시스템을 더 일관되게 정리해야 한다.
- Start Here, Current Board, Forecast, Method, PCA, Survey Detail이 서로 다른 목적을 가진 화면처럼 보여야 한다.
- 차트마다 "어떻게 읽어야 하는지"가 더 명확해야 한다.
- Forecast는 이론적 타당성, 한계, benchmark 해석이 더 쉽게 보여야 한다.
- 데이터셋 충분성, 비전공자 이해도, 시각화 적절성, forecasting 타당성이라는 네 가지 평가 질문이 UI 흐름 안에서 더 강하게 드러나야 한다.

최종 목표:

이 dashboard는 "데이터가 많은 과제"가 아니라, 처음 보는 사람이 5분 안에 논리를 따라가고, 교수님이 봤을 때 데이터, 시각화, forecasting, 방법론이 방어 가능하다고 느끼는 guided analytical product가 되어야 한다.

## 1. 최종 제품 원칙

### 1.1 Guided analytical product

사용자가 스스로 탐색하기 전에 dashboard가 먼저 읽는 길을 제공해야 한다.

필수 조건:

- 첫 화면에서 프로젝트 질문이 보인다.
- 첫 화면에서 최신 상태가 보인다.
- 첫 화면에서 어떤 순서로 읽어야 하는지 보인다.
- 기술 탭을 열기 전에 핵심 개념을 이해할 수 있다.

실패 조건:

- 사용자가 탭 이름만 보고 어디로 가야 할지 고민한다.
- PCA, SAFE, H+1, CISS 같은 용어가 설명 없이 먼저 나온다.
- 모든 섹션이 같은 흰 카드로 보여 중요도가 구분되지 않는다.

### 1.2 Progressive disclosure

처음부터 모든 정보를 보여주면 비전공자는 이탈한다. 정보는 단계적으로 열려야 한다.

권장 흐름:

1. Claim: 무엇을 주장하는가?
2. Board: 지금 어떤 국가가 중요해 보이는가?
3. Defense: 이 분석은 채점 질문에 답하는가?
4. Evidence: 어떤 차트와 데이터가 그 주장을 뒷받침하는가?
5. Method: 데이터와 모델이 어떻게 만들어졌는가?
6. Limits: 무엇을 결론내리면 안 되는가?

### 1.3 Evidence hierarchy

모든 데이터와 차트가 같은 위계를 가지면 안 된다.

우선순위:

1. Main claim
2. Latest monitoring evidence
3. Defense against grading questions
4. Forecast validation and uncertainty
5. Country diagnosis
6. Robustness and PCA
7. Raw data and source audit

### 1.4 Reader-first language

문장은 학술 논문식보다 dashboard reading guide 방식이어야 한다.

각 핵심 섹션은 다음 구조를 사용한다.

```text
Question
Answer
Evidence
Caution
```

예시:

```text
Question: Can current information forecast next-half-year SME financing pain?
Answer: It can support early warning, but should not be read as deterministic prediction.
Evidence: Rolling-origin validation, benchmark comparison, and country-level error.
Caution: Small samples and publication lags limit production-grade forecasting claims.
```

### 1.5 Forecast humility

Forecast는 강하게 주장하면 위험하다. 이 dashboard에서 forecasting은 prediction machine이 아니라 early-warning diagnostic이다.

반드시 지켜야 할 표현:

- "Forecast Check"
- "Early-warning diagnostic"
- "H+1 next half-year"
- "Benchmark comparison"
- "Model agreement, not certainty"
- "Country error, not universal reliability"

피해야 할 표현:

- "Prediction is accurate"
- "This country will experience stress"
- "The model proves future pain"
- "Forecast is the main result"

## 2. 현재 문제 총진단

### 2.1 Interactive Navigator

기존 문제:

- 흰 박스들이 나열되어 premium dashboard 느낌이 없었다.
- 왼쪽 설명, 가운데 route, 오른쪽 quick jump가 서로 경쟁했다.
- reader mode 선택이 기능적으로는 있었지만 시각적 설득력이 약했다.
- quick jump가 단순 세로 메뉴처럼 보였다.

현재 1차 개선:

- command-panel 구조로 변경했다.
- Reader mode에 따라 추천 경로, 설명, 지표, CTA가 바뀌도록 했다.
- compact jump bar로 전환했다.

추가 개선 여지:

- 각 reader mode의 색상과 icon 또는 marker를 더 분명히 구분할 수 있다.
- mode별 "why this path"를 더 짧고 강하게 쓸 수 있다.
- CTA 클릭 후 현재 탭이 바뀌었다는 active feedback을 줄 수 있다.
- keyboard focus style과 accessible selected state를 더 명확히 할 수 있다.

### 2.2 Five-minute Reading Path

기존 문제:

- 7개 카드를 6열 grid에 넣어 7번이 아래로 떨어졌다.
- 카드가 너무 커서 reading path라기보다 또 다른 카드 목록처럼 보였다.
- 긴 설명이 카드 내부에 들어가 가독성이 낮았다.

현재 1차 개선:

- 7-step timeline으로 바꿨다.
- 각 step이 클릭 가능하다.
- 설명은 아래 detail panel로 분리했다.

추가 개선 여지:

- 현재 선택된 step을 active state로 표시할 수 있다.
- timeline detail panel이 탭 이동 후에도 선택 상태를 유지하게 만들 수 있다.
- 모바일에서는 horizontal scroll affordance를 더 명확히 줄 수 있다.
- 각 step에 "time cost" 또는 "reader value"를 표시할 수 있다.

### 2.3 전체 시각 품질

현재 가장 큰 위험:

- 기존 CSS가 누적되어 있어 일부 컴포넌트가 여전히 같은 흰 카드 패턴을 반복한다.
- 섹션마다 목적이 다른데 시각적 차이가 충분하지 않을 수 있다.
- chart, note, guide, card, method panel의 hierarchy가 아직 더 정리될 수 있다.

추가 개선 방향:

- Start, Defense, Forecast, Method, Survey, PCA에 각각 다른 section identity를 부여한다.
- 카드 남발을 줄이고 section band, caption strip, compact chips를 더 활용한다.
- "중요한 것"은 어두운 surface 또는 strong tint, "보조 정보"는 light surface로 분리한다.
- 모든 카드가 동일한 border radius, shadow, padding 체계를 따르도록 정리한다.

### 2.4 탭 구조

현재 위험:

- 탭이 많기 때문에 처음 보는 사람에게 부담이 크다.
- 기술적 탭이 evidence layer라는 점이 충분히 명확하지 않을 수 있다.
- "PCA Analysis", "Firm Survey Detail", "Borrower-Market Gap"의 역할이 처음부터 분리되어 보이지 않을 수 있다.

최종 방향:

- 제출/발표 흐름과 탐색 흐름을 분리한다.
- Start Here와 Current Board를 사실상 homepage와 current product view로 사용한다.
- PCA, Survey Detail, Gap, Trend Explorer는 proof 또는 drill-down layer로 표시한다.

## 3. 최종 정보 구조

### 3.1 최상단 구조

권장 최종 구조:

```text
Project Claim Bar
Reader Mode Navigator
Five-minute Reading Path
Main Tabs
Evidence Sections
Method and Source Audit
```

각 영역의 역할:

| 영역 | 목적 | 사용자가 얻어야 하는 답 |
|---|---|---|
| Project Claim Bar | 전체 주장 요약 | 이 프로젝트가 무엇을 하려는가? |
| Reader Mode Navigator | 읽는 방식 선택 | 나는 어떤 관점으로 읽어야 하는가? |
| Five-minute Reading Path | 제출용 읽기 순서 | 어떤 순서로 읽으면 되는가? |
| Main Tabs | 분석 영역 진입 | 상세 증거는 어디에 있는가? |
| Evidence Sections | 차트와 진단 | 실제 데이터는 무엇을 말하는가? |
| Method and Source Audit | 방어와 재현성 | 이 분석을 믿을 수 있는가? |

### 3.2 최종 탭 순서

권장 탭 순서:

1. Start Here
2. Current Board
3. Defense & Findings
4. Index Basics
5. Forecast & Validation
6. Country Diagnosis
7. Data & Method
8. PCA Analysis
9. Borrower-Market Gap
10. Trend Explorer
11. Firm Survey Detail

탭별 역할:

| 탭 | 역할 | 초보자용 질문 |
|---|---|---|
| Start Here | 프로젝트 요약 | 이 프로젝트는 무엇을 주장하는가? |
| Current Board | 최신 monitoring | 지금 어떤 국가가 중요해 보이는가? |
| Defense & Findings | 채점 질문 방어 | 이 분석은 충분히 타당한가? |
| Index Basics | 지표 이해 | SME-FPI는 무엇을 측정하는가? |
| Forecast & Validation | forecasting 검토 | H+1 forecast는 방어 가능한가? |
| Country Diagnosis | 국가별 설명 | 왜 이 국가가 flagged 되었는가? |
| Data & Method | 방법론 감사 | 어떤 데이터가 어떤 역할을 하는가? |
| PCA Analysis | 구조적 robustness | PCA가 지표 구조를 어떻게 보여주는가? |
| Borrower-Market Gap | 핵심 차이 분석 | borrower pain과 market stress가 언제 갈라지는가? |
| Trend Explorer | 상세 탐색 | 국가별 시간 흐름은 어떤가? |
| Firm Survey Detail | firm-level slice | firm size, sector, survey slices는 무엇을 말하는가? |

## 4. 화면별 최종 개선 계획

### 4.1 Start Here

목표:

처음 보는 사람이 이 페이지 하나만 보고도 프로젝트 논리를 이해해야 한다.

필수 구성:

- Project claim
- Why SME-FPI is borrower-side
- Why CISS is market-side
- Headline findings
- Four grading questions preview
- Five-minute reading path
- Glossary
- Boundary and limitations

개선 체크리스트:

- `SME-FPI`가 처음 등장하는 곳에 plain-English definition을 둔다.
- `CISS`가 market-stress benchmark임을 반복해서 명확히 한다.
- `Relative Gap`은 "SME-FPI minus CISS"라고 반드시 설명한다.
- `H+1`은 "next half-year"라고 설명한다.
- "This is not causal policy evaluation" 문구를 유지한다.

### 4.2 Interactive Navigator

목표:

사용자의 목적에 따라 dashboard가 읽는 길을 제안해야 한다.

Reader modes:

| Mode | 대상 사용자 | 추천 경로 | CTA |
|---|---|---|---|
| First read | 처음 보는 독자 | Start Here -> Current Board -> Defense | Open Start Here |
| Defense | 교수님 질문 대비 | Defense -> Data & Method -> Forecast | Open Defense |
| Forecast | forecast 검토자 | Forecast -> Current Board -> Country Diagnosis | Open Forecast |
| Method | 방법론 감사자 | Index Basics -> PCA Analysis -> Survey Detail | Open Method |

추가 개선:

- active mode에 따라 panel accent color 변경
- mode별 recommended path에 active step marker 추가
- CTA 버튼에 현재 mode의 목적을 반영한 label 사용
- quick jump는 compact bar로 유지
- hover state는 부드럽게, layout shift는 없어야 한다.

완성 기준:

- 한눈에 command center처럼 보여야 한다.
- 흰 박스 나열처럼 보이면 실패다.
- mode 변경 이유가 문장과 path에서 동시에 드러나야 한다.

### 4.3 Five-minute Reading Path

목표:

제출물이 live presentation 없이도 읽히도록 순서를 제공한다.

최종 구조:

```text
01 Start -> 02 Board -> 03 Defense -> 04 Index -> 05 Forecast -> 06 Diagnosis -> 07 Method
```

구현 기준:

- Desktop에서 7개가 한 줄 timeline으로 유지된다.
- 7번째 step이 혼자 다음 줄로 떨어지지 않는다.
- 좁은 화면에서는 horizontal scroll 또는 compact wrapping을 사용한다.
- 각 step 클릭 시 해당 탭으로 이동한다.
- detail panel은 현재 step의 "what"과 "why"를 보여준다.

추가 개선:

- active step styling
- progress line animation
- scroll shadow on mobile
- step마다 "Reader outcome" 한 줄 추가

### 4.4 Current Board

목표:

최신 monitoring 결과를 비전공자도 바로 이해해야 한다.

필수 정보:

- Latest period
- Alert/Watch/Monitor countries
- Why flagged
- Confidence or agreement quality
- Current SME-FPI
- Borrower-market gap
- H+1 forecast direction

개선 방향:

- country cards에 "Why flagged?" expansion 추가
- `Level`, `Gap`, `Forecast`, `Agreement` 네 축을 mini chips로 표시
- 위험 tier 색상은 과장하지 않고 monitoring tone으로 사용
- current board table은 보조로 두고 card-first로 구성

완성 기준:

- "누가 중요한가"와 "왜 중요한가"가 30초 안에 보인다.

### 4.5 Defense & Findings

목표:

교수님이 요구한 네 질문에 직접 답한다.

네 질문:

1. 올바른 목표를 위해 충분한 dataset을 활용했는가?
2. 비전공자가 봐도 한눈에 이해하기 쉬운가?
3. 올바른 시각화 자료를 활용했는가?
4. forecasting 대상과 방법론은 적절한가?

각 질문 카드 구조:

```text
Question
Short answer
Evidence
Boundary
Where to inspect
```

개선 방향:

- 각 질문에 score 또는 verdict chip 추가
- evidence link를 실제 탭과 연결
- "Boundary"는 숨기지 말고 명시한다.

완성 기준:

- 이 탭만 읽어도 평가 질문에 답할 수 있어야 한다.

### 4.6 Forecast & Validation

목표:

Forecast가 이론적으로 방어 가능한 early-warning check라는 점을 보여준다.

반드시 설명할 것:

- Target: H+1 SME-FPI
- Why target is appropriate: borrower-side future pressure
- Method: rolling-origin validation
- Baselines: naive, country mean, AR(1), ARIMA/ARIMAX, momentum, pooled lag OLS
- ML models: Elastic Net, Ridge, Random Forest, Gradient Boosting
- Metrics: MAE, RMSE, benchmark dominance
- Uncertainty: model range and agreement
- Limits: small sample, publication lag, non-causal claim

UI 개선:

- Forecast top에 "Forecast is not the main claim" caution strip
- `Point forecast`, `Uncertainty range`, `Benchmark comparison` segmented toggle
- country error chart 아래 "where to trust cautiously" note
- model leaderboard는 table보다 heatmap 또는 compact rank panel 우선

완성 기준:

- 사용자가 forecast를 deterministic prediction으로 오해하지 않아야 한다.
- benchmark 대비 성능을 먼저 보고, 그 다음 projection을 보게 해야 한다.

### 4.7 Data & Method

목표:

데이터 역할과 방법론 경계를 명확히 한다.

데이터 role taxonomy:

| Role | 설명 | 예시 |
|---|---|---|
| Core index | SME-FPI를 직접 만드는 데이터 | six borrower-side SAFE components |
| Diagnostic detail | 설명과 drill-down용 survey slices | firm size, sector, problem severity |
| Market benchmark | 비교 대상 | ECB CISS |
| Forecast-only predictors | forecasting에만 사용 | BLS, MIR, Eurostat, macro |
| Validation targets | 미래 정렬 확인 | H+1, H+2 future outcomes |
| Raw source | audit trail | raw SAFE, CISS, external files |

개선 방향:

- Data catalog를 role별 filter로 탐색하게 한다.
- Core index와 forecast-only predictors를 시각적으로 분리한다.
- "not part of SME-FPI Core" 문구를 더 잘 보이게 한다.

완성 기준:

- 사용자가 외부 macro/BLS/MIR/Eurostat 데이터가 core index에 들어갔다고 오해하지 않아야 한다.

### 4.8 PCA Analysis

목표:

PCA를 어려운 통계 탭이 아니라 robustness와 structure check로 이해하게 한다.

표현 기준:

- "PCA Analysis"는 적절하다.
- "expert: regime" 같은 표현은 제거한다.
- "SAFE cube"는 설명 없이 쓰면 안 된다.

필수 설명:

- PCA가 무엇을 확인하는지
- PC1 loading이 index weighting robustness에 어떤 의미인지
- cluster/regime은 descriptive grouping이지 causal class가 아님

UI 개선:

- PCA 탭 상단에 "What PCA answers" panel 추가
- loading chart에는 "which components move together" explanation 추가
- scatter plot에는 "distance means similarity, not causality" caution 추가

### 4.9 Firm Survey Detail

목표:

Firm survey slices가 core formula가 아니라 explanation layer임을 보여준다.

개선 방향:

- "SAFE cube" 대신 "Grouped SAFE survey-response detail" 같은 plain name 사용
- firm size, sector, problem severity를 filter chips로 표시
- detail table보다 "who feels pressure?" view를 먼저 보여준다.

완성 기준:

- 독자가 "이 상세 survey data가 SME-FPI core를 바꾸는가?"라고 오해하지 않아야 한다.

## 5. 시각 디자인 시스템

### 5.1 Color system

권장 palette:

| Purpose | Color family | Use |
|---|---|---|
| Core claim | Navy | hero, command center, reading detail |
| Current monitoring | Blue | board, levels, current status |
| Defense | Indigo | grading questions |
| Forecast caution | Amber | forecast, uncertainty, caveat |
| Method/Data | Teal | data roles, audit trail |
| Neutral background | Blue-gray | page base and section separation |

주의:

- 한 화면이 파란색 계열만으로 보이면 안 된다.
- 흰 카드만 반복되면 안 된다.
- 경고색 amber는 과장된 alarm처럼 보이면 안 된다.

### 5.2 Surface system

Surface 종류:

| Surface | 사용처 |
|---|---|
| Dark command surface | navigator, selected reading detail |
| Tinted panel | section intro, defense, forecast, method |
| Raised card | repeated country cards, dataset cards |
| Compact chip | metric, role, status, model label |
| Caption strip | chart reading notes |
| Audit table | source catalog and method details |

금지:

- 카드 안에 또 큰 카드를 넣는 구조
- 모든 section heading을 같은 흰 카드로 만드는 구조
- 긴 문장을 좁은 카드 안에 넣는 구조

### 5.3 Typography

기준:

- Hero/command title: 1.4rem to 1.8rem
- Section heading: 1.25rem to 1.55rem
- Card title: 0.92rem to 1.1rem
- Body: 0.88rem to 0.98rem
- Metric: 1.1rem to 1.5rem

주의:

- viewport width로 font-size를 스케일하지 않는다.
- compact panel 안에 hero-scale title을 넣지 않는다.
- letter spacing은 0을 유지한다.

### 5.4 Spacing

기준:

- Page section gap: 18px to 28px
- Panel padding: 18px to 26px
- Compact card padding: 12px to 16px
- Chip padding: 8px to 12px
- Card gap: 10px to 16px

완성 기준:

- 정보 밀도는 높지만 답답하지 않아야 한다.
- 빈 공간이 너무 커서 dashboard가 허술해 보이면 안 된다.

## 6. Visualization 개선 원칙

### 6.1 Chart selection by question

| 질문 | 적절한 시각화 |
|---|---|
| 시간이 지나며 어떻게 변했는가? | line chart |
| 국가별 현재 수준이 어떻게 다른가? | sorted bar, card board |
| 두 지표가 얼마나 다른가? | diverging bar, slope/dumbbell |
| forecast와 현재값이 어떻게 다른가? | dumbbell with uncertainty range |
| 모델 성능이 어떤가? | loss chart, rank heatmap |
| component가 어떻게 묶이는가? | PCA loading chart |
| 데이터 역할이 무엇인가? | role table, source cards |

### 6.2 Chart guide rule

모든 핵심 차트 아래에는 다음을 붙인다.

```text
What this shows
How to read it
What not to conclude
```

예시:

```text
What this shows: Current SME-FPI and H+1 forecast for selected countries.
How to read it: A wider range means model disagreement is higher.
What not to conclude: The forecast range is not a probability interval or a crisis guarantee.
```

### 6.3 Hover and tooltip

Tooltip에는 숫자만이 아니라 해석 단위가 들어가야 한다.

좋은 tooltip:

```text
Country: Italy
Current SME-FPI: 1.24
H+1 forecast: 1.47
Model agreement: High
Interpretation: pressure is elevated and expected to rise
```

나쁜 tooltip:

```text
x: 1.24
y: IT
model: ridge
```

## 7. Interaction 개선 원칙

### 7.1 필수 인터랙션

- Reader mode selector
- Recommended tab CTA
- Quick jump bar
- Five-minute timeline click
- Country "Why flagged?" expansion
- Forecast view toggle
- Data role filter
- Chart explanation accordion
- Glossary reveal for technical terms

### 7.2 Interaction quality rules

- 클릭 가능한 것은 클릭 가능해 보여야 한다.
- hover는 layout shift를 만들면 안 된다.
- active state가 명확해야 한다.
- 버튼 텍스트가 overflow되면 안 된다.
- mobile에서도 tap target이 충분해야 한다.

### 7.3 State feedback

개선 여지:

- Reader mode 선택 후 selected route를 강조한다.
- Timeline step 클릭 후 active node를 유지한다.
- Quick jump 후 해당 tab의 active 상태와 reading context가 자연스럽게 연결된다.

## 8. Accessibility and Responsive QA

### 8.1 Viewports

반드시 검토할 화면:

- 1636px desktop: 현재 사용자가 주로 보는 화면
- 1440px laptop
- 1280px compact desktop
- 1100px tablet landscape
- 768px tablet
- 390px mobile

### 8.2 Layout checks

확인 항목:

- 7번째 reading path item이 orphan으로 떨어지지 않는다.
- Navigator route cards가 좁게 찢어지지 않는다.
- Quick jump 버튼 텍스트가 잘리지 않는다.
- Tabs가 겹치지 않는다.
- Chart caption이 chart와 겹치지 않는다.
- Section heading이 과도하게 커지지 않는다.

### 8.3 Accessibility checks

확인 항목:

- 텍스트 contrast 충분
- 버튼 focus state 명확
- RadioItems selected state 명확
- 색상만으로 의미를 전달하지 않음
- `Alert`, `Watch`, `Monitor` 같은 label이 텍스트로도 표시됨
- mobile tap target 최소 40px 이상

## 9. Forecast 이론 방어 기준

Forecasting 대상은 적절한가?

적절하다. 단, 다음 조건이 지켜져야 한다.

- target은 H+1 SME-FPI로 정의한다.
- H+1은 next half-year borrower-side financing pain이다.
- forecasting은 production prediction이 아니라 monitoring diagnostic이다.
- rolling-origin validation을 사용한다.
- future target leakage를 막는다.
- simple baselines와 비교한다.
- model agreement와 country error를 함께 보여준다.
- 작은 성능 개선은 겸손하게 해석한다.

방법론 방어 기준:

| 항목 | 기준 |
|---|---|
| Target | H+1 SME-FPI is aligned with early-warning monitoring |
| Validation | Rolling-origin, out-of-sample |
| Baselines | Naive, mean, AR(1), ARIMA/ARIMAX, momentum |
| ML | Compact models only, not oversized neural network |
| Metrics | MAE, RMSE, benchmark edge |
| Display | Forecast plus uncertainty or model range |
| Caution | Not deterministic, not causal, not firm-level credit risk |

## 10. Data sufficiency defense

질문:

올바른 목표를 달성하기 위해 충분한 데이터셋을 활용했는가?

답변 방향:

충분하다. 단, 목표가 causal policy evaluation이나 firm-level default prediction이 아니라 borrower-side monitoring and early-warning diagnosis라는 조건에서 충분하다.

보여줘야 할 근거:

- SME-FPI core panel
- six borrower-side SAFE components
- CISS market-stress benchmark
- grouped SAFE survey detail
- macro and external forecast-only predictors
- rolling-origin forecast outputs
- risk-tier validation
- source catalog and raw audit trail

반드시 명시할 한계:

- sample size is limited by semiannual country panel structure
- exact publication lag for every external source may not be fully modeled
- no firm-level default outcome
- no causal treatment design

## 11. 비전공자 이해도 방어 기준

질문:

비전공자가 봐도 한눈에 이해하기 쉽게 만들었는가?

답변 방향:

최종 dashboard는 reader mode, five-minute path, glossary, chart guides, and defense cards를 통해 비전공자용 reading layer를 제공해야 한다.

필수 장치:

- Start Here
- Interactive Navigator
- Five-minute Reading Path
- Plain-English Glossary
- Question/Answer/Evidence/Caution blocks
- Chart guide captions
- No-overclaim panels

실패 기준:

- 사용자가 첫 화면에서 PCA나 forecasting부터 마주한다.
- SAFE, CISS, H+1이 설명 없이 나온다.
- 기술 용어가 tab label과 chart title에만 존재한다.

## 12. 시각화 적절성 방어 기준

질문:

올바른 시각화 자료를 활용했는가?

답변 방향:

각 시각화는 특정 질문에 대응해야 한다. 단순히 chart가 많다는 것은 방어가 아니다.

검토 기준:

- Monitoring question: board/card/table
- Time trend: line chart
- Gap: diverging bar or gap chart
- Country diagnosis: cards plus driver breakdown
- Forecast: forecast range, loss chart, benchmark comparison
- PCA: loading and structure plot
- Data role: source catalog and role table

개선 여지:

- 각 chart 아래에 reading guide를 추가한다.
- forecast chart는 prediction value보다 uncertainty and benchmark를 먼저 보이게 한다.
- PCA chart는 interpretation panel과 함께 배치한다.

## 13. 구현 우선순위

### Phase 1: Critical visual polish

목표:

현재 1차 redesign이 premium하게 보이도록 시각 품질을 다듬는다.

작업:

- Navigator active state 강화
- Timeline active state 추가
- Section heading consistency 정리
- 흰 카드 반복 영역 추가 제거
- responsive QA

완료 기준:

- 사용자가 상단을 봤을 때 "임시 구현" 느낌이 없어야 한다.

### Phase 2: Tab-level UX refinement

목표:

각 탭이 목적과 visual identity를 가진 화면처럼 보이게 한다.

작업:

- Current Board card hierarchy 개선
- Defense answer cards 강화
- Forecast top explanation 재정리
- Data & Method role taxonomy 강화
- PCA explanation panel 추가
- Survey Detail naming and explanation 개선

완료 기준:

- 탭마다 "이 탭이 왜 필요한지"가 바로 보인다.

### Phase 3: Chart explanation layer

목표:

차트를 읽는 방법을 dashboard가 직접 가르친다.

작업:

- 핵심 chart마다 What this shows 추가
- How to read it 추가
- What not to conclude 추가
- Tooltip text 개선

완료 기준:

- 비전공자도 chart를 보고 최소한의 해석을 할 수 있다.

### Phase 4: Forecast defense upgrade

목표:

Forecasting 방법론이 이론적으로 방어 가능하다는 점을 UI에서 보여준다.

작업:

- H+1 target explanation panel
- rolling-origin validation diagram
- benchmark dominance summary
- model agreement and uncertainty toggle
- country error caution

완료 기준:

- Forecast가 과장되지 않고, 타당한 early-warning check로 읽힌다.

### Phase 5: Final QA

검증:

```text
python -m py_compile dashboard\app.py tests\test_dashboard_smoke.py
python -m unittest tests.test_dashboard_smoke -v
Dash layout HTTP 200
Browser console error check
Navigator mode click
Primary CTA click
Quick jump click
Reading path click
Responsive viewport check
```

## 14. 최종 Acceptance Criteria

이 dashboard는 아래 기준을 모두 통과해야 최종 완성으로 본다.

### UX

- 처음 보는 사용자가 첫 화면에서 프로젝트 목적을 이해한다.
- 사용자가 어떤 순서로 읽어야 하는지 바로 안다.
- 탭이 많아도 cognitive overload가 줄어든다.
- technical tabs는 evidence layer로 보인다.

### UI

- Interactive Navigator가 premium command center처럼 보인다.
- Five-minute reading path가 한 줄 timeline으로 유지된다.
- 7번째 step이 혼자 다음 줄로 떨어지지 않는다.
- 흰 카드 반복감이 크게 줄어든다.
- 섹션별 visual identity가 명확하다.

### Data and Method

- Core index data와 forecast-only predictors가 구분된다.
- Source catalog가 audit trail 역할을 한다.
- 한계가 숨겨지지 않는다.

### Forecast

- H+1 target이 명확하다.
- rolling-origin validation이 보인다.
- benchmark comparison이 보인다.
- uncertainty and model agreement가 보인다.
- deterministic prediction으로 오해되지 않는다.

### Visualization

- 각 chart는 명확한 질문에 답한다.
- 핵심 chart에는 reading guide가 있다.
- 시각화가 과도하게 기술적이지 않다.

### Technical QA

- Python compile 통과
- unit smoke test 통과
- Dash layout 200
- browser console error 없음
- key interactions 정상 작동
- responsive layout에서 text overlap 없음

## 15. 최종 판단

현재 1차 개선은 반드시 필요한 구조적 문제를 해결했다. 하지만 "완벽한 최종 UI/UX"라고 부르기에는 아직 추가 개선 여지가 있다.

앞으로의 핵심은 더 많은 요소를 추가하는 것이 아니다. 이미 정보량은 충분하다. 필요한 것은 다음 세 가지다.

1. 정보 위계를 더 강하게 만드는 것
2. 각 chart와 tab이 무엇을 증명하는지 더 명확히 하는 것
3. Forecast와 method를 과장 없이, 그러나 방어 가능하게 보여주는 것

최종 제품의 방향은 다음 한 문장으로 고정한다.

SME-FPI dashboard should read like a guided analytical product: first explain the claim, then show the current monitoring evidence, then defend data and forecasting choices, and finally expose the full audit trail.

## 16. 독자 모델 재정의: Professor-first, Expert-proof

이 dashboard의 최종 독자는 사실상 두 층으로 나뉜다.

1. Primary reader: 비전공자인 교수님
2. Secondary quality bar: 이 분야의 world-class expert

따라서 dashboard는 일반 대중용으로 지나치게 단순화되어도 안 되고, 전문가용 technical dashboard처럼 처음부터 복잡해도 안 된다. 최종 설계 원칙은 다음과 같다.

```text
Professor-first:
The professor should understand the project logic, chart meaning, and grading answers without knowing SME finance beforehand.

Expert-proof:
A top expert should see enough data lineage, validation logic, benchmark comparison, limitations, and research value to take the dashboard seriously.
```

### 16.1 교수님 중심 설계 원칙

교수님은 이 분야 전공자가 아니지만, 설명이 충분하면 시각화 자료는 빠르게 이해할 수 있는 독자다. 따라서 dashboard는 차트를 숨기거나 지나치게 쉽게 만들 필요는 없다. 대신 차트 앞뒤의 framing을 잘 설계해야 한다.

교수님에게 필요한 것은 다음 네 가지다.

| 필요 | Dashboard가 제공해야 하는 것 |
|---|---|
| 이 프로젝트가 무엇을 주장하는가 | Start Here와 Project Claim |
| 어떤 순서로 읽으면 되는가 | Interactive Navigator와 Five-minute Reading Path |
| 차트를 어떻게 해석하면 되는가 | What this shows / How to read / What not to conclude |
| 채점 질문에 어떻게 답하는가 | Defense & Findings의 네 질문 구조 |

교수님용 UX 기준:

- 첫 5분 안에 전체 논리가 읽혀야 한다.
- 전문 용어는 처음 등장할 때 짧게 정의되어야 한다.
- 차트는 숨기지 말고, 해석 가이드를 붙여야 한다.
- Forecast와 PCA는 "어려운 분석"이 아니라 "왜 필요한 검증인지"가 먼저 설명되어야 한다.
- 모든 탭은 "이 탭을 왜 보는지"가 상단에서 설명되어야 한다.

교수님이 최종적으로 말할 수 있어야 하는 문장:

```text
이 프로젝트는 SME들이 직접 보고한 financing pain을 지표화했고,
그 지표가 시장 스트레스 지표와 다르게 움직일 수 있음을 보여주며,
데이터와 forecast 검증을 통해 monitoring signal로서의 가치를 확인하려 했다.
```

### 16.2 교수님용 화면 설계 수정안

교수님은 한 명의 평가자이므로, dashboard는 "여러 사람이 각자 탐색하는 툴"보다 "한 명이 읽는 self-guided analytical report"처럼 동작해야 한다.

필수 개선:

- Start Here를 사실상 report abstract로 만든다.
- Five-minute Reading Path는 제출물을 읽는 공식 순서로 표시한다.
- Defense & Findings는 교수님의 네 질문에 직접 답하는 구역으로 유지한다.
- 각 chart 아래에는 교수님이 읽을 수 있는 짧은 해설을 둔다.
- 기술 탭은 "나중에 검증하려면 여기로 들어가라"는 evidence layer로 보여준다.

교수님용 탭 framing:

| 탭 | 교수님에게 전달할 메시지 |
|---|---|
| Start Here | 이 프로젝트의 주장과 용어를 먼저 읽으면 된다 |
| Current Board | 최신 데이터가 어떤 국가를 주목하라고 말하는지 보여준다 |
| Defense & Findings | 채점 기준 네 가지에 직접 답한다 |
| Index Basics | SME-FPI가 무엇을 측정하는지 설명한다 |
| Forecast & Validation | forecast는 예언이 아니라 검증용 early-warning check다 |
| Country Diagnosis | 특정 국가가 왜 flagged 되었는지 풀어준다 |
| Data & Method | 데이터와 방법론을 감사할 수 있게 한다 |
| PCA Analysis | index 구조가 robust한지 확인하는 보조 검증이다 |
| Survey Detail | firm group별 세부 survey evidence를 보여준다 |

### 16.3 World-class expert 기준

전문가는 "보기 좋다"만으로 납득하지 않는다. 전문가는 다음을 본다.

- 데이터 source가 명확한가?
- core index와 forecast-only predictors가 분리되어 있는가?
- target과 validation design이 이론적으로 타당한가?
- benchmark가 충분히 강한가?
- 결과가 overclaimed되지 않았는가?
- reproducibility와 audit trail이 있는가?
- 이 dashboard가 실무적 또는 연구적으로 다음 질문을 만들 수 있는가?

전문가가 납득할 최소 조건:

| 기준 | Dashboard에서 보여야 하는 증거 |
|---|---|
| Construct validity | SME-FPI가 borrower-side financing pain을 측정한다는 명확한 component logic |
| Data lineage | raw source -> processed panel -> index -> validation output 흐름 |
| Separation of roles | core index, diagnostic detail, benchmark, forecast-only predictor의 구분 |
| Forecast validity | H+1 target, rolling-origin validation, benchmark comparison |
| Robustness | equal-weight, PCA-weighted, reliability-weighted comparison |
| Interpretability | country drivers, gap, model agreement, forecast uncertainty |
| Limits | causal claim, default prediction, policy prescription을 하지 않는다는 명시 |

### 16.4 Expert-proof dashboard로 끌어올리는 개선안

전문가가 봤을 때 연구적/실무적 가치를 더 느끼게 하려면 다음 레이어가 필요하다.

#### A. Data lineage map

현재 Data & Method에 source catalog가 있지만, 전문가에게는 한눈에 보이는 pipeline map이 더 강하다.

권장 구조:

```text
Raw SAFE survey components
-> standardized borrower-side component panel
-> SME-FPI index variants
-> borrower-market gap vs CISS
-> validation targets and forecast feature panel
-> monitoring board and country diagnosis
```

목표:

- 어떤 데이터가 core index에 들어가는지 보인다.
- 어떤 데이터가 forecast-only인지 보인다.
- 어떤 데이터가 validation용인지 보인다.

#### B. Claim-to-evidence matrix

전문가는 각 claim이 어떤 evidence로 지지되는지 본다.

추가할 표:

| Claim | Evidence | Dashboard location | Limit |
|---|---|---|---|
| SME borrower pain can diverge from market stress | SME-FPI minus CISS gap | Borrower-Market Gap | not causal |
| The index is not one arbitrary weighting choice | equal, PCA, reliability comparisons | Index Basics / PCA | small panel |
| Forecast is useful as early warning | rolling-origin benchmark comparison | Forecast & Validation | modest gains |
| Country flags are explainable | drivers, gap, forecast direction, agreement | Country Diagnosis | monitoring only |

#### C. Forecast audit card

Forecast tab 상단에 전문가용 audit card를 추가할 가치가 있다.

필수 항목:

- Target: H+1 SME-FPI
- Horizon: one half-year ahead
- Validation: rolling-origin out-of-sample
- Leakage control: future targets excluded from features
- Benchmarks: naive, country mean, AR(1), ARIMA/ARIMAX, momentum
- Models: compact ML models
- Metrics: MAE, RMSE
- Interpretation: early-warning diagnostic

#### D. Research extension panel

World-class expert가 "이걸 어떻게 더 발전시킬 수 있나"를 볼 수 있어야 한다.

추가할 연구 확장 질문:

- Can SME-FPI predict later credit tightening better than market stress alone?
- Which borrower-pressure components lead or lag CISS?
- Are positive borrower-market gaps concentrated in particular country groups?
- Do firm-size and sector slices explain country-level monitoring flags?
- How would results change with stricter publication-lag alignment?
- Can this framework support a policy dashboard without making causal claims?

#### E. Practical value panel

실무적 가치를 보여주는 짧은 섹션도 필요하다.

실무 적용 가능성:

- early warning for SME financing stress
- country monitoring prioritization
- borrower-side complement to financial-market stress indicators
- dashboard for analyst briefings
- diagnostic support for policy discussion, not policy automation

### 16.5 이중 설명층 구조

최종 dashboard는 같은 내용을 두 수준으로 설명해야 한다.

| Layer | 대상 | 문체 | 위치 |
|---|---|---|---|
| Professor read | 비전공 교수님 | 짧고 직관적 | 탭 상단, chart caption |
| Expert audit | 전문가 | 방법론, 검증, 한계 | Data & Method, Forecast, PCA, expandable details |

모든 핵심 차트는 다음 구조를 갖는 것이 이상적이다.

```text
Professor read:
This chart shows whether SMEs report more financing pain than market stress alone would suggest.

Expert audit:
The comparison uses standardized SME-FPI and CISS values by country-half-year. It supports diagnostic divergence, not causal attribution.
```

### 16.6 최종 UI/UX 계획 수정 방향

기존 계획은 "비전공자도 이해하기 쉽게"에 초점이 있었다. 이제는 더 정확히 다음처럼 수정해야 한다.

```text
Design for one non-domain professor first,
but expose enough methodological depth for a world-class expert to audit and extend the work.
```

이에 따라 우선순위도 바뀐다.

1. Professor reading path 완성
2. Chart explanation layer 강화
3. Defense & Findings를 교수님 질문 중심으로 더 직접화
4. Forecast audit card 추가
5. Data lineage map 추가
6. Claim-to-evidence matrix 추가
7. Research/practical value panel 추가
8. Expert-level caveats와 extension questions 정리

### 16.7 최종 성공 기준

교수님 기준 성공:

- 교수님이 dashboard를 처음 열고 5분 안에 프로젝트 목적을 이해한다.
- 교수님이 각 차트의 설명을 읽고 시각화의 핵심 의미를 파악한다.
- 교수님이 네 평가 질문에 대한 답을 dashboard 안에서 찾을 수 있다.
- 교수님이 forecast를 예언이 아니라 검증/monitoring 장치로 이해한다.

전문가 기준 성공:

- 전문가가 data lineage와 variable role separation을 확인할 수 있다.
- 전문가가 forecasting validation design을 audit할 수 있다.
- 전문가가 robustness checks와 limitations를 볼 수 있다.
- 전문가가 이 dashboard를 실무 monitoring 또는 후속 연구의 출발점으로 볼 수 있다.

두 기준을 동시에 만족하는 최종 문장:

```text
For the professor, the dashboard should read as a clear self-guided project report.
For the expert, it should stand as an auditable monitoring framework with transparent data roles, defensible validation, and clear research extensions.
```
