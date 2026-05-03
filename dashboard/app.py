import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, dcc, html, ctx
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning as StatsmodelsConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX


BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"

PANEL_PATH = DATA_DIR / "sme_fpi_panel_v2.csv"
SEVERITY_PATH = DATA_DIR / "safe_problem_severity_cube.csv"
VALIDATION_PATH = DATA_DIR / "validation_results.csv"
PCA_LOADINGS_PATH = DATA_DIR / "pca_loadings.csv"
WEIGHTS_PATH = DATA_DIR / "weighting_comparison.csv"
BIG_CUBE_PATH = DATA_DIR / "safe_q0b_big_cube_slim.csv"
FORECAST_FEATURES_PATH = DATA_DIR / "forecasting_feature_panel.csv"
FORECAST_EVALUATION_PATH = DATA_DIR / "forecasting_model_evaluation.csv"
FORECAST_SUMMARY_PATH = DATA_DIR / "forecasting_layer_summary.csv"
SOURCE_CATALOG_PATH = DATA_DIR / "dashboard_source_catalog.csv"
LATEST_FORECAST_PREDICTIONS_PATH = DATA_DIR / "latest_forecast_model_predictions.csv"
FORECAST_DECISION_BOARD_PATH = DATA_DIR / "forecast_decision_board.csv"
FORECAST_ALL_PREDICTIONS_PATH = DATA_DIR / "forecast_model_predictions_all_origins.csv"
FORECAST_DECISION_HISTORY_PATH = DATA_DIR / "decision_board_all_origins.csv"
FORECAST_COUNTRY_ERROR_PATH = DATA_DIR / "forecasting_country_error.csv"
RISK_TIER_VALIDATION_PATH = DATA_DIR / "risk_tier_validation.csv"

ISO3 = {
    "AT": "AUT",
    "BE": "BEL",
    "DE": "DEU",
    "ES": "ESP",
    "FI": "FIN",
    "FR": "FRA",
    "GR": "GRC",
    "IE": "IRL",
    "IT": "ITA",
    "NL": "NLD",
    "PT": "PRT",
    "SK": "SVK",
}

INDEX_OPTIONS = {
    "SME_FPI_equal_z": "Equal-weight SME-FPI",
    "SME_FPI_fixed_baseline_z": "Fixed-baseline SME-FPI",
    "SME_FPI_pca_weighted_z": "PCA-weighted SME-FPI",
    "SME_FPI_reliability_weighted_z": "Reliability-weighted SME-FPI",
}

PCA_3D_Z_OPTIONS = {
    "SME_FPI_equal_z": "SME-FPI level",
    "Relative_Gap_equal": "SME-FPI minus CISS gap",
    "CISS_z": "CISS market stress",
}

GAP_BY_INDEX = {
    "SME_FPI_equal_z": "Relative_Gap_equal",
    "SME_FPI_fixed_baseline_z": "Relative_Gap_fixed_baseline",
    "SME_FPI_pca_weighted_z": "Relative_Gap_pca_weighted",
    "SME_FPI_reliability_weighted_z": "Relative_Gap_reliability_weighted",
}

COMPONENTS = [
    "access_finance_main_problem",
    "bank_loan_rejected",
    "bank_loan_cost_too_high",
    "bank_loan_limited_amount",
    "bank_willingness_deteriorated",
    "interest_rates_increased",
]

Z_COMPONENTS = [f"z_{component}" for component in COMPONENTS]

COMPONENT_LABELS = {
    "access_finance_main_problem": "Access to finance is the main problem",
    "bank_loan_rejected": "Bank loan application rejected",
    "bank_loan_cost_too_high": "Loan cost too high",
    "bank_loan_limited_amount": "Loan amount limited",
    "bank_willingness_deteriorated": "Bank willingness deteriorated",
    "interest_rates_increased": "Interest rates increased",
    "z_access_finance_main_problem": "Access to finance is the main problem",
    "z_bank_loan_rejected": "Bank loan application rejected",
    "z_bank_loan_cost_too_high": "Loan cost too high",
    "z_bank_loan_limited_amount": "Loan amount limited",
    "z_bank_willingness_deteriorated": "Bank willingness deteriorated",
    "z_interest_rates_increased": "Interest rates increased",
}

PCA_SHORT_LABELS = {
    "access_finance_main_problem": "Access",
    "bank_loan_rejected": "Rejected",
    "bank_loan_cost_too_high": "Cost high",
    "bank_loan_limited_amount": "Limited",
    "bank_willingness_deteriorated": "Willingness",
    "interest_rates_increased": "Rates increased",
}

PCA_LABEL_OFFSETS = {
    "access_finance_main_problem": (66, -42),
    "bank_loan_rejected": (-44, -34),
    "bank_loan_cost_too_high": (-22, 18),
    "bank_loan_limited_amount": (42, -18),
    "bank_willingness_deteriorated": (56, 10),
    "interest_rates_increased": (-24, 8),
}

CLUSTER_ORDER = [
    "Low SME financing pain",
    "Interest Rates Increased driven stress",
    "Bank Loan Cost Too High driven stress",
    "Broad SME financing pain",
]

CLUSTER_DISPLAY_NAMES = {
    "Low SME financing pain": "Low pain",
    "Interest Rates Increased driven stress": "Rate pressure",
    "Bank Loan Cost Too High driven stress": "Loan-cost pressure",
    "Broad SME financing pain": "Broad pain",
}

STRESS_SCALE = [
    [0.00, "#2f6f9f"],
    [0.35, "#dbe7ee"],
    [0.50, "#f4f1ea"],
    [0.65, "#f2b6a0"],
    [1.00, "#a4312e"],
]

CATEGORICAL_COLORS = {
    "Low pain": "#2f6f9f",
    "Rate pressure": "#d19a2e",
    "Loan-cost pressure": "#8a63a6",
    "Broad pain": "#b23a35",
}

FORECAST_MODEL_LABELS = {
    "naive": "Naive current value",
    "country_mean": "Country historical mean",
    "country_ar1": "Country AR(1)",
    "country_arima": "Country ARIMA(1,0,0)",
    "country_arimax_ciss": "Country ARIMAX + CISS",
    "momentum": "Last-change extrapolation",
    "pooled_lag_ols": "Pooled lag OLS",
    "elastic_net": "Elastic Net",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

SIMPLE_BASELINE_MODEL_KEYS = ["naive", "country_mean", "country_ar1", "momentum", "pooled_lag_ols"]
TIME_SERIES_MODEL_KEYS = ["country_arima", "country_arimax_ciss"]
BASELINE_MODEL_KEYS = SIMPLE_BASELINE_MODEL_KEYS + TIME_SERIES_MODEL_KEYS
ML_MODEL_KEYS = ["elastic_net", "ridge", "random_forest", "gradient_boosting"]
MODEL_FAMILIES = {
    "naive": "baseline",
    "country_mean": "baseline",
    "country_ar1": "baseline",
    "momentum": "baseline",
    "pooled_lag_ols": "baseline",
    "country_arima": "time-series benchmark",
    "country_arimax_ciss": "time-series benchmark",
    "elastic_net": "machine learning",
    "ridge": "machine learning",
    "random_forest": "machine learning",
    "gradient_boosting": "machine learning",
}

RISK_TIER_COLORS = {
    "Alert": "#a4312e",
    "Watch": "#d19a2e",
    "Monitor": "#2f6f9f",
    "Normal": "#8b98a5",
}

QUICK_JUMP_TABS = {
    "jump-start": "start",
    "jump-board": "decision",
    "jump-defense": "findings",
    "jump-forecast": "forecast",
    "jump-method": "methodology",
    "jump-survey": "big-data",
}

LENS_PRIMARY_TABS = {
    "first": "start",
    "defense": "findings",
    "forecast": "forecast",
    "method": "methodology",
}

READING_PATH_STEPS = [
    {
        "number": "01",
        "title": "Start Here",
        "short": "Project claim",
        "body": "Read the thesis, findings, and glossary first. This page is the project abstract.",
        "why": "It gives the reader the claim before asking them to interpret technical charts.",
        "outcome": "The reader can state the project claim without opening a technical tab.",
        "target": "start",
    },
    {
        "number": "02",
        "title": "Current Board",
        "short": "Latest signals",
        "body": "See the latest countries that deserve attention, then read the reason and confidence level.",
        "why": "It turns the index into a plain-language monitoring question: who needs attention now?",
        "outcome": "The reader knows who is flagged and which evidence made the country visible.",
        "target": "decision",
    },
    {
        "number": "03",
        "title": "Defense & Findings",
        "short": "Grading answers",
        "body": "Answer the four grading questions before moving into technical evidence.",
        "why": "It proves the dashboard can defend data sufficiency, readability, visualization choice, and forecasting validity.",
        "outcome": "The professor can find the direct answer to each evaluation question.",
        "target": "findings",
    },
    {
        "number": "04",
        "title": "Index Basics",
        "short": "What it measures",
        "body": "Understand what SME-FPI measures and why it is borrower-side rather than market-side.",
        "why": "It prevents the reader from confusing SME survey pressure with market stress.",
        "outcome": "SME-FPI, CISS, Relative Gap, and H+1 are no longer unexplained acronyms.",
        "target": "overview",
    },
    {
        "number": "05",
        "title": "Forecast",
        "short": "H+1 validation",
        "body": "Judge whether the H+1 forecast target, method, and error views are defensible.",
        "why": "It frames forecasting as an early-warning diagnostic, not a deterministic prediction.",
        "outcome": "Forecasting is read as a benchmarked validation layer, not the main result.",
        "target": "forecast",
    },
    {
        "number": "06",
        "title": "Country Diagnosis",
        "short": "Why flagged",
        "body": "Explain why a country is flagged using drivers, gap, forecast direction, and model agreement.",
        "why": "It connects the latest board to country-level evidence and avoids black-box labels.",
        "outcome": "A country flag becomes an evidence bundle rather than a standalone label.",
        "target": "diagnosis",
    },
    {
        "number": "07",
        "title": "Data & Method",
        "short": "Audit trail",
        "body": "Use the method and data pages to verify sources, robustness checks, and limits.",
        "why": "It closes the report with source roles, model limits, and reproducibility evidence.",
        "outcome": "The reader can separate core index data, forecast-only predictors, validation outputs, and raw sources.",
        "target": "methodology",
    },
]

READING_PATH_TABS = {f"read-path-{step['target']}": step["target"] for step in READING_PATH_STEPS}
READING_PATH_LOOKUP = {step["target"]: step for step in READING_PATH_STEPS}

COMPONENT_DRIVER_LABELS = {
    "z_access_finance_main_problem": "Access to finance",
    "z_bank_loan_rejected": "Bank loan rejection",
    "z_bank_loan_cost_too_high": "Loan cost too high",
    "z_bank_loan_limited_amount": "Limited loan amount",
    "z_bank_willingness_deteriorated": "Bank willingness",
    "z_interest_rates_increased": "Interest rates increased",
}

DIAGNOSTIC_DRIVER_COLUMNS = {
    "current_score": "Current SME-FPI",
    "gap_value": "SME-FPI minus CISS",
    "best_model_delta": "Best ML delta",
    "ml_model_agreement_rising": "ML rising agreement",
    "availability_pressure_z": "Access/availability",
    "cost_pressure_z": "Cost pressure",
    "salience_pressure_z": "Finance salience",
}

PRESSURE_GROUPS = {
    "availability_pressure_z": {
        "label": "Access/availability pressure",
        "short": "Availability",
        "columns": ["z_bank_loan_rejected", "z_bank_loan_limited_amount", "z_bank_willingness_deteriorated"],
        "description": "Loan rejection, limited amount, and weaker bank willingness.",
    },
    "cost_pressure_z": {
        "label": "Cost pressure",
        "short": "Cost",
        "columns": ["z_bank_loan_cost_too_high", "z_interest_rates_increased"],
        "description": "High loan cost and rising interest-rate pressure.",
    },
    "salience_pressure_z": {
        "label": "Finance salience",
        "short": "Salience",
        "columns": ["z_access_finance_main_problem"],
        "description": "Access to finance reported as the most pressing problem.",
    },
}

PRESSURE_GROUP_LABELS = {key: value["label"] for key, value in PRESSURE_GROUPS.items()}


def load_optional_csv(path, default):
    if path.exists():
        return pd.read_csv(path)
    return default.copy() if isinstance(default, pd.DataFrame) else default


def add_pressure_groups(frame):
    out = frame.copy()
    for key, spec in PRESSURE_GROUPS.items():
        columns = [col for col in spec["columns"] if col in out.columns]
        if columns:
            out[key] = out[columns].mean(axis=1, skipna=True)
    return out


def load_data():
    panel = pd.read_csv(PANEL_PATH)
    severity = pd.read_csv(SEVERITY_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    pca_loadings = pd.read_csv(PCA_LOADINGS_PATH)
    weights = pd.read_csv(WEIGHTS_PATH)
    big_cube_rows = len(pd.read_csv(BIG_CUBE_PATH, usecols=["REF_AREA"]))
    forecast_panel = load_optional_csv(FORECAST_FEATURES_PATH, panel)
    forecast_evaluation = load_optional_csv(FORECAST_EVALUATION_PATH, pd.DataFrame())
    forecast_summary = load_optional_csv(FORECAST_SUMMARY_PATH, pd.DataFrame())
    source_catalog = load_optional_csv(SOURCE_CATALOG_PATH, pd.DataFrame())
    latest_forecast_predictions = load_optional_csv(LATEST_FORECAST_PREDICTIONS_PATH, pd.DataFrame())
    forecast_decision_board = load_optional_csv(FORECAST_DECISION_BOARD_PATH, pd.DataFrame())
    forecast_all_predictions = load_optional_csv(FORECAST_ALL_PREDICTIONS_PATH, pd.DataFrame())
    forecast_decision_history = load_optional_csv(FORECAST_DECISION_HISTORY_PATH, pd.DataFrame())
    forecast_country_error = load_optional_csv(FORECAST_COUNTRY_ERROR_PATH, pd.DataFrame())
    risk_tier_validation = load_optional_csv(RISK_TIER_VALIDATION_PATH, pd.DataFrame())

    panel = add_pressure_groups(panel)
    forecast_panel = add_pressure_groups(forecast_panel)
    panel["iso3"] = panel["REF_AREA"].map(ISO3)
    panel["coverage_label"] = panel["components_available"].astype("Int64").astype(str) + " / 6 components"
    panel = panel.sort_values(["period_sort", "country_name"]).reset_index(drop=True)
    periods = (
        panel[["TIME_PERIOD", "period_sort"]]
        .drop_duplicates()
        .sort_values("period_sort")["TIME_PERIOD"]
        .tolist()
    )
    period_to_index = {period: index for index, period in enumerate(periods)}
    panel["period_index"] = panel["TIME_PERIOD"].map(period_to_index)
    if "components_available" in forecast_panel.columns:
        forecast_panel["coverage_label"] = (
            forecast_panel["components_available"].astype("Int64").astype(str) + " / 6 components"
        )
    if "period_index" not in forecast_panel.columns:
        forecast_panel = forecast_panel.merge(
            panel[["REF_AREA", "TIME_PERIOD", "period_index"]],
            on=["REF_AREA", "TIME_PERIOD"],
            how="left",
        )
    forecast_panel = forecast_panel.sort_values(["period_sort", "country_name"]).reset_index(drop=True)
    for frame in [forecast_all_predictions, forecast_decision_history, forecast_decision_board]:
        if not frame.empty and "TIME_PERIOD" in frame.columns and "period_index" not in frame.columns:
            frame["period_index"] = frame["TIME_PERIOD"].map(period_to_index)
    severity = severity.sort_values(["TIME_PERIOD", "country_name"]).reset_index(drop=True)
    severity["period_index"] = severity["TIME_PERIOD"].map(period_to_index)

    return (
        panel,
        severity,
        validation,
        pca_loadings,
        weights,
        big_cube_rows,
        periods,
        forecast_panel,
        forecast_evaluation,
        forecast_summary,
        source_catalog,
        latest_forecast_predictions,
        forecast_decision_board,
        forecast_all_predictions,
        forecast_decision_history,
        forecast_country_error,
        risk_tier_validation,
    )


(
    PANEL,
    SEVERITY,
    VALIDATION,
    PCA_LOADINGS,
    WEIGHTS,
    BIG_CUBE_ROWS,
    PERIODS,
    FORECAST_PANEL,
    FORECAST_EVALUATION,
    FORECAST_SUMMARY,
    SOURCE_CATALOG,
    LATEST_FORECAST_PREDICTIONS,
    FORECAST_DECISION_BOARD,
    FORECAST_ALL_PREDICTIONS,
    FORECAST_DECISION_HISTORY,
    FORECAST_COUNTRY_ERROR,
    RISK_TIER_VALIDATION,
) = load_data()
COUNTRIES = PANEL[["REF_AREA", "country_name"]].drop_duplicates().sort_values("country_name")
COUNTRY_OPTIONS = [
    {"label": row.country_name, "value": row.REF_AREA} for row in COUNTRIES.itertuples()
]
ALL_COUNTRY_CODES = COUNTRIES["REF_AREA"].tolist()
DEFAULT_COUNTRIES = ALL_COUNTRY_CODES
FOCUS_COUNTRY_LIMIT = 6
PERIOD_MARKS = {
    index: period.replace("-S", " H")
    for index, period in enumerate(PERIODS)
    if index in {0, len(PERIODS) - 1} or index % 8 == 0
}
PROBLEMS = sorted(SEVERITY["problem"].dropna().unique())
FIRM_SIZE_OPTIONS = [
    {"label": row.firm_size_label, "value": row.FIRM_SIZE}
    for row in SEVERITY[["FIRM_SIZE", "firm_size_label"]].drop_duplicates().sort_values("firm_size_label").itertuples()
]
SECTOR_OPTIONS = [
    {"label": row.sector_label, "value": row.FIRM_SECTOR}
    for row in SEVERITY[["FIRM_SECTOR", "sector_label"]].drop_duplicates().sort_values("sector_label").itertuples()
]

DATASET_PREVIEWS = [
    {
        "key": "sme_fpi_panel_v2",
        "label": "SME-FPI main panel",
        "layer": "Core analytical panel",
        "path": PANEL_PATH,
        "purpose": "Country-half-year panel used for the main index, gap analysis, clusters, validation, and forecasting features.",
        "rows": len(PANEL),
    },
    {
        "key": "safe_component_long",
        "label": "SAFE component long table",
        "layer": "Core SAFE inputs",
        "path": DATA_DIR / "safe_component_long.csv",
        "purpose": "Long-form version of the six borrower-side SAFE components before they are combined into SME-FPI.",
        "rows": 11300,
    },
    {
        "key": "safe_problem_severity_cube",
        "label": "SAFE problem-severity cube",
        "layer": "Big Data diagnostic layer",
        "path": SEVERITY_PATH,
        "purpose": "Firm-size, sector, and problem-severity table used for firm-survey diagnostics and micro vulnerability features.",
        "rows": len(SEVERITY),
    },
    {
        "key": "safe_q0b_big_cube_slim",
        "label": "SAFE Q0B response cube",
        "layer": "Large survey-response cube",
        "path": BIG_CUBE_PATH,
        "purpose": "231,231 grouped survey-response cells by country, period, firm group, problem, and answer severity.",
        "rows": BIG_CUBE_ROWS,
    },
    {
        "key": "macro_context_panel",
        "label": "World Bank macro context panel",
        "layer": "Macro context",
        "path": DATA_DIR / "macro_context_panel.csv",
        "purpose": "Semiannual macro context used as forecasting predictors and validation context, not as SME-FPI Core.",
        "rows": 386,
    },
    {
        "key": "ciss_semiannual",
        "label": "ECB CISS semiannual benchmark",
        "layer": "Market-stress benchmark",
        "path": DATA_DIR / "ciss_semiannual.csv",
        "purpose": "ECB market-stress benchmark used for SME-FPI minus CISS gaps and external validation.",
        "rows": 386,
    },
    {
        "key": "forecasting_feature_panel",
        "label": "Forecasting feature panel",
        "layer": "Forecast feature stack",
        "path": FORECAST_FEATURES_PATH,
        "purpose": "Expanded model matrix combining SME-FPI, detailed SAFE survey slices, macro, BLS, MIR, Eurostat, and lagged information available at the forecast origin.",
        "rows": len(FORECAST_PANEL),
    },
    {
        "key": "forecasting_model_evaluation",
        "label": "Rolling-origin model evaluation",
        "layer": "Forecast validation",
        "path": FORECAST_EVALUATION_PATH,
        "purpose": "Backtest table comparing ML models with naive, country mean, AR(1), ARIMA/ARIMAX, momentum, and pooled lag OLS benchmarks.",
        "rows": len(FORECAST_EVALUATION),
    },
    {
        "key": "forecast_model_predictions_all_origins",
        "label": "All rolling-origin predictions",
        "layer": "Forecast validation",
        "path": FORECAST_ALL_PREDICTIONS_PATH,
        "purpose": "Country-model prediction table for every rolling-origin window, including ARIMA/ARIMAX benchmarks and ML forecasts.",
        "rows": len(FORECAST_ALL_PREDICTIONS),
    },
    {
        "key": "latest_forecast_model_predictions",
        "label": "Latest forecast model predictions",
        "layer": "Forecast output",
        "path": LATEST_FORECAST_PREDICTIONS_PATH,
        "purpose": "Latest country-level H+1 predictions by model, used by the current Forecast Check and Monitoring Board evidence.",
        "rows": len(LATEST_FORECAST_PREDICTIONS),
    },
    {
        "key": "decision_board_history",
        "label": "Historical Monitoring Board",
        "layer": "Risk-tier validation",
        "path": FORECAST_DECISION_HISTORY_PATH,
        "purpose": "Rolling-origin decision-board assignments used to check whether Alert, Watch, Monitor, and Normal tiers behave sensibly through time.",
        "rows": len(FORECAST_DECISION_HISTORY),
    },
    {
        "key": "risk_tier_validation",
        "label": "Risk-tier validation summary",
        "layer": "Risk-tier validation",
        "path": RISK_TIER_VALIDATION_PATH,
        "purpose": "Summary table comparing diagnostic tiers with next-period pressure movement and directional hit rates.",
        "rows": len(RISK_TIER_VALIDATION),
    },
    {
        "key": "forecasting_country_error",
        "label": "Country forecast-error summary",
        "layer": "Forecast validation",
        "path": FORECAST_COUNTRY_ERROR_PATH,
        "purpose": "Country-by-model error table used to show where forecasts are reliable or difficult.",
        "rows": len(FORECAST_COUNTRY_ERROR),
    },
    {
        "key": "raw_safe_access_finance_main_problem",
        "label": "Raw SAFE access-finance series",
        "layer": "Raw SAFE input",
        "path": RAW_DIR / "safe_access_finance_main_problem.csv",
        "purpose": "One of the six ECB SAFE borrower-side inputs used to construct the core index.",
        "rows": None,
    },
    {
        "key": "raw_safe_q0b_cube",
        "label": "Raw SAFE Q0B cube",
        "layer": "Raw SAFE response-detail source",
        "path": RAW_DIR / "safe_q0b_pressingness_big_cube.csv",
        "purpose": "Raw large response cube behind the processed Big Data diagnostic layer.",
        "rows": None,
    },
    {
        "key": "raw_worldbank_macro",
        "label": "Raw World Bank macro data",
        "layer": "Raw macro input",
        "path": RAW_DIR / "worldbank_macro_annual.csv",
        "purpose": "Annual macro indicators later aligned to the semiannual dashboard panel.",
        "rows": None,
    },
    {
        "key": "raw_ecb_ciss_daily",
        "label": "Raw ECB CISS daily data",
        "layer": "Raw market benchmark",
        "path": RAW_DIR / "ecb_ciss_new_daily.csv",
        "purpose": "Daily ECB CISS observations aggregated to the semiannual market-stress benchmark.",
        "rows": None,
    },
    {
        "key": "raw_bls_credit_standards",
        "label": "Raw ECB BLS credit standards",
        "layer": "Raw lender-side input",
        "path": RAW_DIR / "forecast_bls_credit_standards_sme.csv",
        "purpose": "Bank Lending Survey signal used only in the forecasting layer, not in SME-FPI Core.",
        "rows": None,
    },
    {
        "key": "raw_mir_small_loan_rate",
        "label": "Raw ECB MIR small-loan rate",
        "layer": "Raw loan-market input",
        "path": RAW_DIR / "forecast_mir_small_loan_rate.csv",
        "purpose": "Small-loan interest-rate series used only as a forecast predictor.",
        "rows": None,
    },
    {
        "key": "raw_eurostat_bankruptcies",
        "label": "Raw Eurostat bankruptcies index",
        "layer": "Raw business-demography input",
        "path": RAW_DIR / "forecast_eurostat_bankruptcies_index.csv",
        "purpose": "Business-demography stress context used only as a forecast predictor.",
        "rows": None,
    },
    {
        "key": "raw_eurostat_registrations",
        "label": "Raw Eurostat registrations index",
        "layer": "Raw business-demography input",
        "path": RAW_DIR / "forecast_eurostat_registrations_index.csv",
        "purpose": "Business-formation context used only as a forecast predictor.",
        "rows": None,
    },
]
DATASET_PREVIEW_LOOKUP = {item["key"]: item for item in DATASET_PREVIEWS}
DATASET_ROW_OPTIONS = [{"label": f"First {n} rows", "value": n} for n in [5, 10, 20, 50]]


def period_filter(df, period_range):
    start, end = period_range
    return df[(df["period_index"] >= start) & (df["period_index"] <= end)].copy()


def short_period(period):
    return str(period).replace("-S", " H")


def period_tick_values(periods, max_ticks=6):
    values = list(dict.fromkeys([period for period in periods if pd.notna(period)]))
    if len(values) <= max_ticks:
        return values
    step = max(1, int(np.ceil((len(values) - 1) / (max_ticks - 1))))
    selected = values[::step]
    if values[-1] not in selected:
        selected.append(values[-1])
    return selected


def apply_period_ticks(fig, periods, max_ticks=6, angle=0):
    values = period_tick_values(periods, max_ticks=max_ticks)
    fig.update_xaxes(
        tickmode="array",
        tickvals=values,
        ticktext=[short_period(period) for period in values],
        tickangle=angle,
    )
    return fig


def selected_or_default(values):
    if values:
        return values
    return []


def many_countries_selected(countries):
    return len(selected_or_default(countries)) > FOCUS_COUNTRY_LIMIT


def classify_index(value):
    if pd.isna(value):
        return "Missing"
    if value >= 1.0:
        return "High borrower-side financing pain"
    if value >= 0.35:
        return "Above-average financing pain"
    if value <= -1.0:
        return "Low financing pain"
    if value <= -0.35:
        return "Below-average financing pain"
    return "Near the sample average"


def classify_gap(value):
    if pd.isna(value):
        return "Missing"
    if value >= 1.0:
        return "SME pain is much higher than the CISS benchmark"
    if value >= 0.35:
        return "SME pain is higher than the CISS benchmark"
    if value <= -1.0:
        return "Market stress is much higher than SME pain"
    if value <= -0.35:
        return "Market stress is higher than SME pain"
    return "SME pain and CISS are broadly aligned"


def add_interpretation(df, metric_col):
    gap_col = GAP_BY_INDEX[metric_col]
    out = df.copy()
    out["metric_value"] = out[metric_col]
    out["metric_label"] = out["metric_value"].apply(classify_index)
    out["gap_value"] = out[gap_col]
    out["gap_label"] = out["gap_value"].apply(classify_gap)
    return out


def format_number(value, digits=2):
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def latest_period_df(df):
    latest_idx = df["period_index"].max()
    return df[df["period_index"] == latest_idx].copy()


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 15, "color": "#58616a"},
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin={"l": 30, "r": 30, "t": 30, "b": 30},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def polish(fig, title, height=430, y_title=None, x_title=None, showlegend=True):
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 16}},
        height=height,
        margin={"l": 54, "r": 28, "t": 126, "b": 70},
        font={"family": "Inter, Segoe UI, Arial, sans-serif", "size": 12, "color": "#1f2933"},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.42)",
        hoverlabel={
            "bgcolor": "rgba(255,255,255,0.96)",
            "bordercolor": "rgba(184,200,213,0.86)",
            "font_size": 12,
            "font_family": "Inter, Segoe UI, Arial, sans-serif",
        },
        legend={"orientation": "h", "yanchor": "top", "y": -0.16, "xanchor": "left", "x": 0},
        legend_title_text="",
        showlegend=showlegend,
    )
    fig.update_xaxes(
        title=x_title,
        showgrid=True,
        gridcolor="rgba(183, 199, 212, 0.34)",
        zerolinecolor="rgba(139, 152, 165, 0.72)",
        linecolor="rgba(184, 200, 213, 0.70)",
    )
    fig.update_yaxes(
        title=y_title,
        showgrid=True,
        gridcolor="rgba(183, 199, 212, 0.34)",
        zerolinecolor="rgba(139, 152, 165, 0.72)",
        linecolor="rgba(184, 200, 213, 0.70)",
    )
    return fig


def chart_title(headline, subtitle):
    return (
        "<span style='display:inline-block;padding:3px 7px;border-radius:999px;"
        "background:rgba(47,111,159,0.12);color:#244c68;font-size:10px;"
        "font-weight:900;letter-spacing:0.08em'>"
        "ANALYTIC VIEW</span><br>"
        f"<span style='font-size:18px;color:#13202b;font-weight:850'>{headline}</span><br>"
        f"<span style='font-size:12px;color:#53616d'>{subtitle}</span>"
    )


def compact_text(text, limit=118):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "..."


def compact_bullet(label, text, limit=118):
    return html.Li([html.Strong(label), html.Span(compact_text(text, limit))])


def visual_bullets(items, className="visual-bullet-row"):
    return html.Ul(className=className, children=[html.Li(item) for item in items])


def compact_copy_bullets(items, className="compact-copy-list", limit=138):
    children = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            children.append(compact_bullet(item[0], item[1], limit))
        else:
            children.append(html.Li(compact_text(item, limit)))
    return html.Ul(className=className, children=children)


def compact_summary_panel(kicker, title, bullets, detail=None, className="", accent="blue"):
    panel_class = f"compact-summary-panel compact-summary-{accent} {className}".strip()
    children = [
        html.Span(kicker, className="panel-kicker"),
        html.H3(title),
        visual_bullets(bullets),
    ]
    if detail:
        children.append(
            html.Details(
                className="compact-summary-detail",
                children=[
                    html.Summary("Details"),
                    compact_copy_bullets(detail, className="compact-copy-list compact-copy-list-detail", limit=150),
                ],
            )
        )
    return html.Div(className=panel_class, children=children)


def explanation(title, what, why, insight):
    return html.Div(
        className="explanation",
        children=[
            html.H4(title),
            html.Ul(
                className="compact-guide-list",
                children=[
                    compact_bullet("Shows", what),
                    compact_bullet("Read", why),
                    compact_bullet("Takeaway", insight),
                ],
            ),
            html.Details(
                className="compact-note",
                children=[
                    html.Summary("Caution"),
                    html.P("Do not read this chart alone as causal proof, a crisis guarantee, or a production forecast."),
                ],
            ),
        ],
    )


def metric_card(label, value, note):
    return html.Div(
        className="metric-card",
        children=[
            html.Span(label, className="metric-label"),
            html.Strong(value, className="metric-value"),
            html.Span(note, className="metric-note"),
        ],
    )


def concept_card(title, tag, body):
    return html.Div(
        className="concept-card",
        children=[
            html.Span(tag, className="concept-tag"),
            html.H3(title),
            html.P(compact_text(body, 118)),
        ],
    )


def tab_guide(question, takeaway, caveat):
    return html.Div(
        className="tab-guide",
        children=[
            html.Div(className="tab-guide-card tab-guide-question", children=[html.Span("Question"), html.Strong(question)]),
            html.Div(className="tab-guide-card tab-guide-takeaway", children=[html.Span("Takeaway"), html.Strong(compact_text(takeaway, 132))]),
            html.Div(className="tab-guide-card tab-guide-caveat", children=[html.Span("Caution"), html.Strong(compact_text(caveat, 132))]),
        ],
    )


def reader_checkpoint(question, answer, interpretation, caution):
    items = [
        ("01", "Question", question, "question"),
        ("02", "Answer", answer, "answer"),
        ("03", "Read", interpretation, "read"),
        ("04", "Caution", caution, "caution"),
    ]
    return html.Div(
        className="rc-grid",
        children=[
            html.Div(
                className=f"rc-card rc-card-{tone}",
                children=[
                    html.Div(
                        className="rc-card-head",
                        children=[
                            html.Span(number, className="rc-number"),
                            html.Span(label, className="rc-label"),
                        ],
                    ),
                    html.P(compact_text(text, 168), className="rc-body"),
                ],
            )
            for number, label, text, tone in items
        ],
    )


def evidence_card(number, title, tag, body):
    return html.Div(
        className="evidence-card",
        children=[
            html.Span(number, className="evidence-number"),
            html.Div(children=[html.Span(tag, className="concept-tag"), html.H3(title), html.P(compact_text(body, 126))]),
        ],
    )


def pipeline_step(number, title, tag, body):
    return html.Div(
        className="pipeline-step",
        children=[
            html.Span(number, className="pipeline-number"),
            html.Div(
                children=[
                    html.Span(tag, className="pipeline-tag"),
                    html.H3(title),
                    html.P(compact_text(body, 118)),
                ]
            ),
        ],
    )


def source_status_class(status):
    status = str(status).lower()
    if "implemented" in status:
        return "source-status source-status--live"
    if "planned" in status:
        return "source-status source-status--planned"
    return "source-status source-status--ready"


def source_role_options():
    if SOURCE_CATALOG.empty or "role" not in SOURCE_CATALOG.columns:
        return [{"label": "All roles", "value": "all"}]
    roles = sorted(SOURCE_CATALOG["role"].dropna().astype(str).unique())
    return [{"label": "All roles", "value": "all"}] + [{"label": role, "value": role} for role in roles]


def source_catalog_table(limit=None, role_filter=None):
    if SOURCE_CATALOG.empty:
        return html.Div(
            className="source-table-empty",
            children="Source catalog is not available. Run scripts/11_build_forecasting_layer.py to generate it.",
        )
    frame = SOURCE_CATALOG.copy()
    if role_filter and role_filter != "all" and "role" in frame.columns:
        frame = frame[frame["role"].astype(str).eq(str(role_filter))].copy()
    if limit:
        frame = frame.head(limit)
    if frame.empty:
        return html.Div(
            className="source-table-empty",
            children="No source rows match this role filter.",
        )
    rows = []
    for row in frame.itertuples():
        rows.append(
            html.Tr(
                children=[
                    html.Td(str(row.layer)),
                    html.Td(str(row.dataset)),
                    html.Td(html.Span(str(row.status), className=source_status_class(row.status))),
                    html.Td(f"{int(row.rows):,}" if pd.notna(row.rows) else "n/a"),
                    html.Td(str(row.dashboard_use)),
                ]
            )
        )
    return html.Div(
        className="source-table-wrap",
        children=[
            html.Table(
                className="source-table",
                children=[
                    html.Thead(
                        html.Tr(
                            children=[
                                html.Th("Layer"),
                                html.Th("Dataset"),
                                html.Th("Status"),
                                html.Th("Rows"),
                                html.Th("Dashboard use"),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
            )
        ],
    )


def forecast_stability_stats():
    if FORECAST_EVALUATION.empty or "model_key" not in FORECAST_EVALUATION.columns:
        return {}
    evaluation = FORECAST_EVALUATION.copy()
    if "model_family" not in evaluation.columns:
        evaluation["model_family"] = evaluation["model_key"].map(MODEL_FAMILIES).fillna("model")
    required = {"origin_period_sort", "origin_period", "model_key", "model_label", "model_family", "mae"}
    if not required.issubset(evaluation.columns):
        return {}

    best_by_origin = evaluation.loc[evaluation.groupby("origin_period_sort")["mae"].idxmin()].copy()
    best_model_counts = best_by_origin["model_label"].value_counts()
    best_family_counts = best_by_origin["model_family"].value_counts()

    ml = (
        evaluation[evaluation["model_key"].isin(ML_MODEL_KEYS)]
        .groupby(["origin_period_sort", "origin_period"], as_index=False)["mae"]
        .min()
        .rename(columns={"mae": "best_ml_mae"})
    )
    baseline = (
        evaluation[evaluation["model_key"].isin(BASELINE_MODEL_KEYS)]
        .groupby(["origin_period_sort", "origin_period"], as_index=False)["mae"]
        .min()
        .rename(columns={"mae": "best_baseline_mae"})
    )
    comparison = ml.merge(baseline, on=["origin_period_sort", "origin_period"], how="inner")
    if comparison.empty:
        return {}
    comparison["edge"] = comparison["best_baseline_mae"] - comparison["best_ml_mae"]
    top_model = best_model_counts.index[0] if not best_model_counts.empty else "n/a"
    top_model_wins = int(best_model_counts.iloc[0]) if not best_model_counts.empty else 0
    ml_family_wins = int(best_family_counts.get("machine learning", 0))
    return {
        "n_origins": int(len(comparison)),
        "ml_wins": int((comparison["edge"] > 0).sum()),
        "ml_win_share": float((comparison["edge"] > 0).mean()),
        "median_edge": float(comparison["edge"].median()),
        "mean_edge": float(comparison["edge"].mean()),
        "top_model": str(top_model),
        "top_model_wins": top_model_wins,
        "ml_family_wins": ml_family_wins,
    }


def edge_strength_label(edge):
    if pd.isna(edge):
        return "unknown"
    if edge < 0.01:
        return "weak"
    if edge < 0.05:
        return "modest"
    return "meaningful"


def safe_int(value, default=0):
    if pd.isna(value):
        return default
    return int(value)


def defense_evidence_item(text):
    return html.Li(text)


def defense_card(number, question, verdict, answer, evidence, caveat, tone="solid", inspect=None):
    return html.Div(
        className=f"defense-card defense-card-{tone}",
        children=[
            html.Div(
                className="defense-card-head",
                children=[
                    html.Span(number, className="defense-number"),
                    html.Span(verdict, className=f"defense-verdict defense-verdict-{tone}"),
                ],
            ),
            html.H3(question),
            html.P(compact_text(answer, 150), className="defense-answer"),
            html.Ul(
                className="defense-evidence-list",
                children=[defense_evidence_item(compact_text(item, 112)) for item in evidence],
            ),
            html.Div(
                className="defense-inspect",
                children=[
                    html.Strong("Where to inspect"),
                    html.Span(compact_text(inspect or "Use the recommended reading path and linked evidence tabs.", 104)),
                ],
            ),
            html.Div(
                className="defense-caveat",
                children=[html.Strong("Boundary"), html.Span(compact_text(caveat, 118))],
            ),
        ],
    )


def defense_question_panel():
    summary = FORECAST_SUMMARY.iloc[0] if not FORECAST_SUMMARY.empty else pd.Series(dtype=object)
    source_rows = len(SOURCE_CATALOG) if not SOURCE_CATALOG.empty else len(DATASET_PREVIEWS)
    implemented_sources = (
        int(SOURCE_CATALOG["status"].astype(str).str.lower().eq("implemented").sum())
        if not SOURCE_CATALOG.empty and "status" in SOURCE_CATALOG.columns
        else source_rows
    )
    data_roles = (
        int(SOURCE_CATALOG["role"].dropna().astype(str).nunique())
        if not SOURCE_CATALOG.empty and "role" in SOURCE_CATALOG.columns
        else 0
    )
    countries = PANEL["REF_AREA"].nunique()
    periods = PANEL["TIME_PERIOD"].nunique()
    forecast_columns = safe_int(summary.get("columns_forecasting_feature_panel", len(FORECAST_PANEL.columns)))
    macro_micro = safe_int(summary.get("macro_micro_predictor_columns", 0))
    optional_external = safe_int(summary.get("optional_external_predictor_columns", 0))
    rolling_origins = safe_int(summary.get("rolling_origin_count", 0))
    ml_wins = safe_int(summary.get("ml_beats_strongest_baseline_count", 0))
    ml_win_share = summary.get("ml_beats_strongest_baseline_share", np.nan)
    best_model = summary.get("best_recent_model_label", "best recent model")
    strongest_baseline = summary.get("strongest_recent_baseline_label", "strongest benchmark")
    ml_edge = summary.get("best_recent_ml_improvement_vs_strongest_baseline", np.nan)

    return html.Div(
        className="defense-panel",
        children=[
            html.Div(
                className="defense-panel-copy",
                children=[
                    html.Span("Defense questions", className="panel-kicker"),
                    html.H3("Four grading questions"),
                    html.Ul(
                        className="visual-bullet-row",
                        children=[
                            html.Li("Data sufficiency"),
                            html.Li("Non-specialist readability"),
                            html.Li("Visualization fit"),
                            html.Li("Forecast defensibility"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="defense-grid",
                children=[
                    defense_card(
                        "01",
                        "Did we use enough datasets for the objective?",
                        "Yes, for a diagnostic monitoring objective",
                        (
                            f"The core evidence uses {countries} countries across {periods} half-year periods, "
                            f"a {len(PANEL):,}-row SME-FPI panel, the {BIG_CUBE_ROWS:,}-row SAFE response cube, "
                            f"and {implemented_sources} implemented source layers covering {data_roles} roles."
                        ),
                        [
                            "SME-FPI Core stays borrower-side, so the main target is not diluted by unrelated macro variables.",
                            f"The forecast feature panel has {forecast_columns:,} columns, including {macro_micro} macro/micro and {optional_external} optional external predictor columns.",
                            "CISS, World Bank, BLS, MIR, Eurostat, validation, and forecast outputs are separated by role in Data & Method.",
                        ],
                        "Sufficient for descriptive diagnosis and H+1 early-warning checks; not sufficient for causal policy evaluation or firm-level default scoring.",
                        "solid",
                        "Data & Method: role taxonomy, source catalog, data lineage map, and raw/processed preview.",
                    ),
                    defense_card(
                        "02",
                        "Can a non-specialist understand it at a glance?",
                        "Mostly yes, by starting from the board",
                        (
                            "The first screen now answers who needs attention, why the country is flagged, "
                            "whether CISS misses the borrower-side signal, and how confident the model agreement is."
                        ),
                        [
                            "Current Board is the default tab and uses Alert, Watch, Monitor, Normal labels instead of raw model output.",
                            "Every major tab starts with a plain-English question, takeaway, and caveat.",
                            "Glossary, reading path, and figure notes explain SAFE, CISS, SME-FPI, H+1, MAE, and model agreement in context.",
                        ],
                        "Expert tabs remain available, but they are secondary drill-down layers rather than the first reading path.",
                        "solid",
                        "Start Here: glossary and five-minute path; Current Board: country cards and chart captions.",
                    ),
                    defense_card(
                        "03",
                        "Are the visualizations appropriate?",
                        "Yes, chart types match the analytical tasks",
                        (
                            "The redesign uses a monitoring board for triage, ranked bars for comparison, heatmaps for country-period matrices, "
                            "scatter/dumbbell views for diagnosis and uncertainty, and tables for auditability."
                        ),
                        [
                            "Heatmaps are used only when both country and period or component dimensions matter.",
                            "Forecast charts show loss, benchmark dominance, rank stability, and country error instead of only a single projection.",
                            "Choropleth/map-style views are contextual; the main decisions are carried by readable board, rank, and validation views.",
                        ],
                        "The dashboard avoids decorative chart variety; every figure needs to answer a specific reader question.",
                        "solid",
                        "Current Board, Borrower-Market Gap, Trend Explorer, PCA Analysis, and Firm Survey Detail chart notes.",
                    ),
                    defense_card(
                        "04",
                        "Is the forecasting target and method defensible?",
                        "Yes, with modest claims",
                        (
                            "The target is next-half-year SME-FPI because the project is about early warning for borrower-side SME financing pain, "
                            "not predicting market stress or firm defaults."
                        ),
                        [
                            f"The model suite is tested over {rolling_origins} rolling origins; ML beats the strongest benchmark {ml_wins} times ({format_number(ml_win_share * 100 if pd.notna(ml_win_share) else np.nan, 0)}%).",
                            f"The latest best model is {best_model}, but it is compared against {strongest_baseline}; the latest ML edge is {format_number(ml_edge, 3)} ({edge_strength_label(ml_edge)}).",
                            "Naive, country mean, AR(1), ARIMA/ARIMAX, momentum, and pooled lag OLS keep the ML results theoretically honest for a short semiannual panel.",
                        ],
                        "The forecast supports monitoring and prioritization; it should not be presented as deterministic prediction or production credit risk.",
                        "watch",
                        "Forecast & Validation: audit card, rolling-origin loss, benchmark dominance, rank stability, and country error.",
                    ),
                ],
            ),
        ],
    )


def forecast_defense_panel():
    summary = FORECAST_SUMMARY.iloc[0] if not FORECAST_SUMMARY.empty else pd.Series(dtype=object)
    best_model = summary.get("best_recent_model_label", "best recent model")
    best_mae = summary.get("best_recent_model_mae", np.nan)
    strongest_baseline = summary.get("strongest_recent_baseline_label", "strongest benchmark")
    baseline_mae = summary.get("strongest_recent_baseline_mae", np.nan)
    rolling_origins = safe_int(summary.get("rolling_origin_count", 0))
    rows = safe_int(summary.get("rows_forecasting_feature_panel", len(FORECAST_PANEL)))

    checks = [
        (
            "Target",
            "Next half-year SME-FPI",
            "The target matches the project question: can current borrower-side and context signals warn about later SME financing pain?",
        ),
        (
            "Method",
            "Rolling-origin comparison",
            f"{rolling_origins} pseudo-real-time origins compare {best_model} with naive, AR(1), ARIMA/ARIMAX, momentum, pooled OLS, and regularized/tree ML.",
        ),
        (
            "Visualization",
            "Loss, benchmark, stability",
            f"The reader sees MAE/RMSE, country error, benchmark dominance, and rank stability; latest MAE is {format_number(best_mae)} vs {format_number(baseline_mae)} for {strongest_baseline}.",
        ),
        (
            "Sample discipline",
            f"{rows:,} country-period rows",
            "The dashboard favors compact regularized models and simple time-series benchmarks because each country has a short semiannual history.",
        ),
    ]
    return html.Div(
        className="forecast-defense-panel",
        children=[
            html.Div(
                className="forecast-defense-copy",
                children=[
                    html.Span("Forecast defense", className="panel-kicker"),
                    html.H3("Why the forecast is only an early-warning check"),
                    html.Ul(
                        className="visual-bullet-row",
                        children=[
                            html.Li("H+1 target"),
                            html.Li("Rolling-origin validation"),
                            html.Li("Strong baselines"),
                            html.Li("Visible uncertainty"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="forecast-defense-grid",
                children=[
                    html.Div(
                        className="forecast-defense-card",
                        children=[html.Span(tag), html.Strong(title), html.P(compact_text(body, 120))],
                    )
                    for tag, title, body in checks
                ],
            ),
        ],
    )


def forecast_audit_card():
    summary = FORECAST_SUMMARY.iloc[0] if not FORECAST_SUMMARY.empty else pd.Series(dtype=object)
    best_model = summary.get("best_recent_model_label", "best recent ML model")
    best_mae = summary.get("best_recent_model_mae", np.nan)
    strongest_baseline = summary.get("strongest_recent_baseline_label", "strongest benchmark")
    baseline_mae = summary.get("strongest_recent_baseline_mae", np.nan)
    checks = [
        ("Target", "H+1 SME-FPI", "Next-half-year borrower-side financing pain."),
        ("Horizon", "one half-year ahead", "The target period is the next semiannual SAFE observation."),
        ("Validation", "rolling-origin out-of-sample", "Each origin mimics information available before the target period."),
        ("Leakage control", "future targets excluded", "Future and target columns are blocked from feature selection."),
        ("Benchmarks", "naive, mean, AR(1), ARIMA/ARIMAX, momentum, pooled OLS", "ML has to beat simple and compact time-series alternatives."),
        ("Models", "Elastic Net, Ridge, Random Forest, Gradient Boosting", "Compact models fit the short country-half-year panel better than oversized neural networks."),
        ("Metrics", "MAE and RMSE", f"Recent best: {best_model} MAE {format_number(best_mae)} vs {strongest_baseline} {format_number(baseline_mae)}."),
        ("Interpretation", "early-warning diagnostic", "Model agreement is consistency evidence, not certainty."),
    ]
    return html.Div(
        className="forecast-audit-panel",
        children=[
            html.Div(
                className="forecast-audit-copy",
                children=[
                    html.Span("Forecast audit card", className="panel-kicker"),
                    html.H3("H+1 validation layer"),
                    html.Ul(
                        className="visual-bullet-row",
                        children=[html.Li("not the main claim"), html.Li("benchmarked"), html.Li("leakage controlled"), html.Li("not deterministic")],
                    ),
                ],
            ),
            html.Div(
                className="forecast-audit-grid",
                children=[
                    html.Div(
                        className="forecast-audit-item",
                        children=[html.Span(label), html.Strong(value), html.P(compact_text(note, 118))],
                    )
                    for label, value, note in checks
                ],
            ),
        ],
    )


def forecast_view_selector():
    return html.Div(
        className="forecast-view-control",
        children=[
            html.Div(
                children=[
                    html.Span("Forecast view", className="panel-kicker"),
                    html.H3("Read benchmark evidence first"),
                    html.Ul(
                        className="visual-bullet-row",
                        children=[html.Li("benchmark"), html.Li("uncertainty"), html.Li("point forecast")],
                    ),
                ],
            ),
            dcc.RadioItems(
                id="forecast-view-select",
                value="benchmark",
                options=[
                    {"label": "Benchmark comparison", "value": "benchmark"},
                    {"label": "Uncertainty range", "value": "uncertainty"},
                    {"label": "Point forecast", "value": "point"},
                ],
                className="forecast-view-segmented",
                inputClassName="forecast-view-input",
                labelClassName="forecast-view-label",
            ),
        ],
    )


def forecast_view_styles(selected):
    selected = selected or "benchmark"
    hidden = {"display": "none"}
    visible = {"display": "block"}
    return (
        visible if selected == "point" else hidden,
        visible if selected == "uncertainty" else hidden,
        visible if selected == "benchmark" else hidden,
    )


def data_role_taxonomy_panel():
    roles = [
        ("Core index", "SME-FPI formula inputs", "Six borrower-side SAFE components. These are the only ingredients that directly build the core index."),
        ("Diagnostic detail", "explanation layer", "Firm-size, sector, and problem-severity survey slices explain who feels pressure."),
        ("Market benchmark", "comparison object", "ECB CISS is used to compare borrower-side pain with financial-market stress."),
        ("Forecast-only predictors", "H+1 model features", "BLS, MIR, Eurostat, macro, and detailed survey features support forecasting only."),
        ("Validation targets", "future checks", "H+1 and H+2 future outcomes test whether current signals align with later stress."),
        ("Raw source", "audit trail", "Raw SAFE, CISS, BLS, MIR, Eurostat, macro, and generated files keep the work inspectable."),
    ]
    return html.Div(
        className="role-taxonomy-panel",
        children=[
            html.Div(
                className="role-taxonomy-copy",
                children=[
                    html.Span("Data role taxonomy", className="panel-kicker"),
                    html.H3("Core index vs forecast-only predictors"),
                    html.Ul(
                        className="visual-bullet-row",
                        children=[html.Li("6 SAFE core inputs"), html.Li("CISS benchmark"), html.Li("external predictors"), html.Li("validation outputs")],
                    ),
                ],
            ),
            html.Div(
                className="role-taxonomy-grid",
                children=[concept_card(title, tag, body) for title, tag, body in roles],
            ),
        ],
    )


def data_lineage_map():
    steps = [
        ("01", "Raw SAFE survey components", "source", "Borrower-side survey answers and grouped response detail."),
        ("02", "Standardized component panel", "core", "Country-half-year z-scores, coverage checks, and pressure families."),
        ("03", "SME-FPI index variants", "index", "Equal, fixed-baseline, PCA-weighted, and reliability-weighted versions."),
        ("04", "SME-FPI minus CISS gap", "benchmark", "Borrower-side pressure compared with market-stress context."),
        ("05", "Forecast feature panel", "validation", "Forecast-only predictors and H+1/H+2 targets stay outside the core formula."),
        ("06", "Monitoring board", "product", "Country flags, diagnosis cards, validation charts, and audit trail."),
    ]
    return html.Div(
        className="data-lineage-panel",
        children=[
            html.Div(
                className="data-lineage-copy",
                children=[
                    html.Span("Data lineage map", className="panel-kicker"),
                    html.H3("Raw source -> processed panel -> index -> validation output"),
                    html.Ul(
                        className="visual-bullet-row",
                        children=[html.Li("source"), html.Li("panel"), html.Li("index"), html.Li("board")],
                    ),
                ],
            ),
            html.Div(
                className="data-lineage-track",
                children=[
                    html.Div(
                        className="data-lineage-node",
                        children=[
                            html.Span(number),
                            html.Strong(title),
                            html.Small(tag),
                            html.P(compact_text(body, 112)),
                        ],
                    )
                    for number, title, tag, body in steps
                ],
            ),
        ],
    )


def claim_evidence_matrix():
    rows = [
        (
            "SME borrower pain can diverge from market stress",
            "SME-FPI minus CISS gap",
            "Borrower-Market Gap / Current Board",
            "diagnostic, not causal",
        ),
        (
            "The index is not one arbitrary weighting choice",
            "equal, fixed-baseline, PCA-weighted, and reliability-weighted checks",
            "Index Basics / PCA Analysis",
            "small semiannual panel",
        ),
        (
            "Forecasting is useful only as early warning",
            "rolling-origin benchmark comparison and country error",
            "Forecast & Validation",
            "modest gains, no deterministic prediction",
        ),
        (
            "Country flags are explainable",
            "drivers, relative gap, forecast direction, and model agreement",
            "Current Board / Country Diagnosis",
            "monitoring tier, not policy verdict",
        ),
    ]
    return html.Div(
        className="claim-evidence-panel",
        children=[
            html.Div(
                className="source-catalog-copy",
                children=[
                    html.Span("Claim-to-evidence matrix", className="panel-kicker"),
                    html.H3("Claim -> evidence -> location -> limit"),
                ],
            ),
            html.Div(
                className="claim-evidence-card-grid",
                children=[
                    html.Div(
                        className="claim-evidence-card",
                        children=[
                            html.H4(claim),
                            compact_copy_bullets(
                                [
                                    ("Evidence", evidence),
                                    ("Where", location),
                                    ("Limit", limit),
                                ],
                                className="compact-guide-list claim-evidence-card-list",
                                limit=118,
                            ),
                        ],
                    )
                    for claim, evidence, location, limit in rows
                ],
            ),
        ],
    )


def research_value_panel():
    research_questions = [
        "Can SME-FPI predict later credit tightening better than market stress alone?",
        "Which borrower-pressure components lead or lag CISS?",
        "Are positive borrower-market gaps concentrated in particular country groups?",
        "Do firm-size and sector slices explain country-level monitoring flags?",
        "How would results change with stricter publication-lag alignment?",
        "Can this framework support a policy dashboard without making causal claims?",
    ]
    practical_values = [
        "early warning for SME financing stress",
        "country monitoring prioritization",
        "borrower-side complement to financial-market stress indicators",
        "dashboard for analyst briefings",
        "diagnostic support for policy discussion, not policy automation",
    ]
    return html.Div(
        className="research-value-panel",
        children=[
            html.Div(
                children=[
                    html.Span("Research extension", className="panel-kicker"),
                    html.H3("What a world-class expert can do next"),
                    html.Ul(children=[html.Li(item) for item in research_questions]),
                ],
            ),
            html.Div(
                children=[
                    html.Span("Practical value", className="panel-kicker"),
                    html.H3("Why the dashboard has monitoring value"),
                    html.Ul(children=[html.Li(item) for item in practical_values]),
                ],
            ),
        ],
    )


def pca_answer_panel():
    return html.Div(
        className="pca-answer-panel",
        children=[
            html.Div(
                children=[
                    html.Span("What PCA answers", className="panel-kicker"),
                    html.H3("Structure and robustness check"),
                    html.Ul(
                        className="visual-bullet-row",
                        children=[html.Li("component co-movement"), html.Li("weighting robustness"), html.Li("similarity only"), html.Li("not causal")],
                    ),
                ],
            ),
            html.Div(
                className="pca-answer-grid",
                children=[
                    concept_card("PC1 loadings", "weighting check", "Components loading together support the idea that equal-weight SME-FPI is not arbitrary."),
                    concept_card("Component circle", "co-movement", "Arrows pointing in similar directions mean components tend to move together."),
                    concept_card("Scatter distance", "similarity only", "Nearby points are similar in component structure; distance does not imply causality."),
                    concept_card("Descriptive groups", "not classes", "Groups summarize observed patterns; they are not supervised predictions or causal regimes."),
                ],
            ),
        ],
    )


def firm_survey_pressure_panel():
    return html.Div(
        className="survey-pressure-panel",
        children=[
            html.Div(
                children=[
                    html.Span("Who feels pressure?", className="panel-kicker"),
                    html.H3("Grouped SAFE survey-response detail"),
                    html.Ul(
                        className="visual-bullet-row",
                        children=[html.Li("explains flags"), html.Li("firm groups"), html.Li("problem severity"), html.Li("not core formula")],
                    ),
                ],
            ),
            html.Div(
                className="survey-chip-row",
                children=[
                    html.Span("firm size"),
                    html.Span("sector"),
                    html.Span("problem severity"),
                    html.Span("top-box share"),
                    html.Span("high-pressure share"),
                ],
            ),
        ],
    )


def lens_route_step(number, title, body):
    return html.Div(
        className="lens-route-step",
        children=[
            html.Span(number),
            html.Div(children=[html.Strong(title), html.P(compact_text(body, 118))]),
        ],
    )


def lens_signal_card(label, value, note):
    return html.Div(
        className="lens-signal-card",
        children=[
            html.Span(label),
            html.Strong(value),
            html.Small(note),
        ],
    )


def reader_lens_panel(mode):
    mode = mode or "first"
    summary = FORECAST_SUMMARY.iloc[0] if not FORECAST_SUMMARY.empty else pd.Series(dtype=object)
    latest = latest_period_df(PANEL)
    latest_period = latest["TIME_PERIOD"].iloc[0] if not latest.empty else "latest period"
    monitor_count = 0
    if not FORECAST_DECISION_BOARD.empty and "risk_tier" in FORECAST_DECISION_BOARD.columns:
        monitor_count = int(FORECAST_DECISION_BOARD["risk_tier"].isin(["Alert", "Watch", "Monitor"]).sum())
    rolling_origins = safe_int(summary.get("rolling_origin_count", 0))
    ml_wins = safe_int(summary.get("ml_beats_strongest_baseline_count", 0))
    best_model = str(summary.get("best_recent_model_label", "Ridge"))
    best_mae = summary.get("best_recent_model_mae", np.nan)

    lens_data = {
        "first": {
            "kicker": "First-time reader",
            "title": "Understand the project in five minutes.",
            "body": "Start with the claim, read the latest monitoring board, then use the defense cards before opening technical tabs.",
            "cta": "Open Start Here",
            "route": [
                ("01", "Start Here", "Project claim"),
                ("02", "Current Board", f"{latest_period} signals"),
                ("03", "Defense", "Grading answers"),
            ],
            "signals": [
                ("Latest period", str(latest_period), "current dashboard endpoint"),
                ("Monitoring reads", str(monitor_count), "Alert, Watch, or Monitor countries"),
                ("Core panel", f"{len(PANEL):,}", "country-half-year rows"),
            ],
        },
        "defense": {
            "kicker": "Defense mode",
            "title": "Prepare the professor-facing argument first.",
            "body": "Lead with data sufficiency, reader clarity, visualization choice, and forecast validity before the detailed evidence.",
            "cta": "Open Defense",
            "route": [
                ("01", "Defense", "Four answers"),
                ("02", "Data & Method", "Source roles"),
                ("03", "Forecast", "H+1 proof"),
            ],
            "signals": [
                ("Data layers", f"{len(SOURCE_CATALOG) if not SOURCE_CATALOG.empty else len(DATASET_PREVIEWS)}", "catalogued evidence roles"),
                ("SAFE detail", f"{BIG_CUBE_ROWS:,}", "grouped survey-response rows"),
                ("Forecast origins", str(rolling_origins), "rolling-origin checks"),
            ],
        },
        "forecast": {
            "kicker": "Forecast lens",
            "title": "Read forecasting as an early-warning diagnostic.",
            "body": "The forecast is defensible only if H+1 SME-FPI predictions beat simple persistence and compact time-series baselines.",
            "cta": "Open Forecast",
            "route": [
                ("01", "Forecast", "Target and loss"),
                ("02", "Current Board", "Tier impact"),
                ("03", "Diagnosis", "Why flagged"),
            ],
            "signals": [
                ("Best model", best_model, f"recent MAE {format_number(best_mae)}"),
                ("ML wins", f"{ml_wins}/{rolling_origins}", "vs strongest benchmark"),
                ("Target", "H+1 SME-FPI", "next half-year borrower pain"),
            ],
        },
        "method": {
            "kicker": "Method lens",
            "title": "Audit what belongs to the index and what only validates it.",
            "body": "Keep the six borrower-side SAFE inputs separate from forecast-only predictors, robustness checks, and raw data sources.",
            "cta": "Open Method",
            "route": [
                ("01", "Index Basics", "Six inputs"),
                ("02", "PCA Analysis", "Structure"),
                ("03", "Survey Detail", "Firm slices"),
            ],
            "signals": [
                ("Components", "6", "borrower-side SAFE inputs"),
                ("Countries", str(PANEL["REF_AREA"].nunique()), "European panel members"),
                ("Periods", str(PANEL["TIME_PERIOD"].nunique()), "half-year observations"),
            ],
        },
    }
    data = lens_data.get(mode, lens_data["first"])
    return html.Div(
        className=f"navigator-mode-panel navigator-mode-{mode}",
        children=[
            html.Div(
                className="mode-copy",
                children=[
                    html.Span(data["kicker"], className="panel-kicker"),
                    html.H2(data["title"]),
                    visual_bullets([data["body"], "recommended path below", "jump bar stays compact"]),
                ],
            ),
            html.Div(
                className="mode-route",
                children=[
                    html.Span("Recommended path", className="mode-panel-label"),
                    html.Div(
                        className="mode-route-track",
                        children=[lens_route_step(number, title, body) for number, title, body in data["route"]],
                    ),
                ],
            ),
            html.Div(
                className="mode-stats",
                children=[
                    html.Span("Evidence snapshot", className="mode-panel-label"),
                    html.Div(
                        className="lens-signal-grid",
                        children=[lens_signal_card(label, value, note) for label, value, note in data["signals"]],
                    ),
                ],
            ),
            html.Div(
                className="mode-cta-hint",
                children=[html.Strong(data["cta"]), html.Span("or use the compact jump bar below")],
            ),
        ],
    )


def quick_jump_button(button_id, label, note):
    return html.Button(
        id=button_id,
        n_clicks=0,
        className="quick-jump-button",
        children=[
            html.Strong(label),
            html.Span(note),
        ],
    )


def target_tab_from_button(button_id, reader_mode=None):
    if button_id == "lens-primary-action":
        return LENS_PRIMARY_TABS.get(reader_mode or "first", "start")
    if button_id in READING_PATH_TABS:
        return READING_PATH_TABS[button_id]
    return QUICK_JUMP_TABS.get(button_id, "start")


def interaction_hub():
    return html.Section(
        className="section interaction-hub interaction-hub-v2",
        children=[
            html.Div(
                className="interaction-shell",
                children=[
                    html.Div(
                        className="navigator-header",
                        children=[
                            html.Div(
                                className="navigator-title-block",
                                children=[
                                    html.Span("Interactive navigator", className="panel-kicker"),
                                    html.H2("Choose the reader mode before opening the evidence tabs."),
                                    visual_bullets(["choose intent", "follow route", "jump to evidence"]),
                                ],
                            ),
                            html.Div(
                                className="lens-control-card",
                                children=[
                                    html.Span("Reader mode", className="mode-panel-label"),
                                    dcc.RadioItems(
                                        id="reader-lens",
                                        className="lens-segmented",
                                        value="first",
                                        options=[
                                            {"label": "First read", "value": "first"},
                                            {"label": "Defense", "value": "defense"},
                                            {"label": "Forecast", "value": "forecast"},
                                            {"label": "Method", "value": "method"},
                                        ],
                                        inputClassName="lens-input",
                                        labelClassName="lens-label",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(id="lens-panel", className="lens-panel", children=reader_lens_panel("first")),
                    html.Div(
                        className="navigator-action-row",
                        children=[
                            html.Button(
                                id="lens-primary-action",
                                n_clicks=0,
                                className="navigator-primary-action",
                                children=[html.Strong("Open recommended tab"), html.Span("changes with reader mode")],
                            ),
                            html.Div(
                                className="quick-jump-grid",
                                children=[
                                    quick_jump_button("jump-start", "Start", "claim"),
                                    quick_jump_button("jump-board", "Board", "signals"),
                                    quick_jump_button("jump-defense", "Defense", "answers"),
                                    quick_jump_button("jump-forecast", "Forecast", "H+1"),
                                    quick_jump_button("jump-method", "Method", "audit"),
                                    quick_jump_button("jump-survey", "Survey", "slices"),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )


def forecasting_summary_cards():
    if FORECAST_SUMMARY.empty:
        return [
            metric_card("Forecast feature panel", "not built", "run scripts/11_build_forecasting_layer.py"),
            metric_card("External predictors", "pending", "BLS/MIR/Eurostat optional files not summarized yet"),
            metric_card("Backtest", "pending", "rolling-origin evaluation not available"),
        ]
    row = FORECAST_SUMMARY.iloc[0]
    best_model = row.get("best_recent_model_label", "n/a")
    best_mae = row.get("best_recent_model_mae", np.nan)
    elastic_mae = row.get("latest_backtest_elastic_net_mae", np.nan)
    arimax_mae = row.get("latest_backtest_country_arimax_ciss_mae", np.nan)
    naive_mae = row.get("latest_backtest_naive_mae", np.nan)
    strong_baseline = row.get("strongest_recent_baseline_label", "n/a")
    strong_baseline_mae = row.get("strongest_recent_baseline_mae", np.nan)
    improvement = row.get("best_recent_ml_improvement_vs_strongest_baseline", np.nan)
    stability = forecast_stability_stats()
    return [
        metric_card(
            "Forecast feature panel",
            f"{int(row.get('columns_forecasting_feature_panel', 0)):,} cols",
            f"{int(row.get('rows_forecasting_feature_panel', 0)):,} country x half-year rows",
        ),
        metric_card(
            "Macro + micro predictors",
            f"{int(row.get('macro_micro_predictor_columns', 0)):,}",
            "SAFE survey-detail plus macro context columns",
        ),
        metric_card(
            "BLS/MIR/Eurostat predictors",
            f"{int(row.get('optional_external_predictor_columns', 0)):,}",
            "lender-side, loan-rate, and business-demography columns",
        ),
        metric_card(
            "Best recent ML model",
            str(best_model),
            f"MAE {format_number(best_mae)} vs naive {format_number(naive_mae)}",
        ),
        metric_card(
            "Strongest benchmark",
            str(strong_baseline),
            f"includes simple and time-series; recent ML edge {format_number(improvement, 3)} ({edge_strength_label(improvement)})",
        ),
        metric_card(
            "ML win rate",
            f"{stability.get('ml_wins', 0)} / {stability.get('n_origins', 0)}",
            "origins where best ML beats the strongest benchmark",
        ),
        metric_card(
            "Median ML edge",
            format_number(stability.get("median_edge", np.nan), 3),
            f"mean edge {format_number(stability.get('mean_edge', np.nan), 3)} MAE",
        ),
        metric_card(
            "Best-model stability",
            str(stability.get("top_model", "n/a")),
            f"{stability.get('top_model_wins', 0)} origin wins; ML family {stability.get('ml_family_wins', 0)} wins",
        ),
        metric_card(
            "ARIMAX+CISS check",
            f"MAE {format_number(arimax_mae)}",
            f"regularized linear check: Elastic Net {format_number(elastic_mae)}",
        ),
    ]


def latest_findings_cards():
    summary = FORECAST_SUMMARY.iloc[0] if not FORECAST_SUMMARY.empty else pd.Series(dtype=object)
    tier = RISK_TIER_VALIDATION.copy()
    alert_watch = 0
    monitor_count = 0
    hidden_text = "none"
    rising_text = "none"
    if not FORECAST_DECISION_BOARD.empty and "risk_tier" in FORECAST_DECISION_BOARD.columns:
        alert_watch = int(FORECAST_DECISION_BOARD["risk_tier"].isin(["Alert", "Watch"]).sum())
        monitor_count = int(FORECAST_DECISION_BOARD["risk_tier"].eq("Monitor").sum())
        board = FORECAST_DECISION_BOARD.copy()
        if "signal_type" not in board.columns:
            board["signal_type"] = [monitor_signal_type(row) for row in board.itertuples()]
        hidden_text = list_country_names(board[board["signal_type"].isin(["Hidden-gap monitor", "Mixed monitor", "Visible or hidden high pressure"])])
        rising_text = list_country_names(board[board["signal_type"].isin(["Rising-forecast monitor", "Mixed monitor", "Forward-looking warning"])])
    best_model = summary.get("best_recent_model_label", "Ridge / regularized model")
    best_mae = summary.get("best_recent_model_mae", np.nan)
    strong_baseline = summary.get("strongest_recent_baseline_label", "pooled lag OLS")
    baseline_mae = summary.get("strongest_recent_baseline_mae", np.nan)
    ml_edge = summary.get("best_recent_ml_improvement_vs_strongest_baseline", np.nan)
    stability = forecast_stability_stats()
    if not tier.empty and "risk_tier" in tier.columns:
        tier_order = {"Alert": 0, "Watch": 1, "Monitor": 2, "Normal": 3}
        tier = tier.assign(_order=tier["risk_tier"].map(tier_order)).sort_values("_order")
        tier_sentence = "; ".join(
            [
                f"{row.risk_tier}: next score {format_number(row.mean_actual_next_score)}"
                for row in tier.itertuples()
                if pd.notna(getattr(row, "mean_actual_next_score", np.nan))
            ][:3]
        )
    else:
        tier_sentence = "historical tier validation available after rebuilding the forecast layer"

    return html.Div(
        className="findings-grid",
        children=[
            evidence_card(
                "01",
                "Borrower-side pressure is not the same object as market stress",
                "core claim",
                "SME-FPI is built from SAFE borrower responses, while CISS measures financial-market stress. The gap is therefore interpretable as a diagnostic difference, not a duplicate benchmark.",
            ),
            evidence_card(
                "02",
                "The current Monitoring Board is a monitoring screen, not a policy verdict",
                "latest status",
                f"The latest watchlist has {alert_watch} Alert/Watch and {monitor_count} Monitor countries. Hidden-gap monitors: {hidden_text}. Rising-forecast monitors: {rising_text}.",
            ),
            evidence_card(
                "03",
                "Forecast gains are visible but small",
                "forecast validation",
                f"The best recent ML model is {best_model} with MAE {format_number(best_mae)}. The strongest benchmark is {strong_baseline} with MAE {format_number(baseline_mae)}, so the recent ML edge is {format_number(ml_edge, 3)} ({edge_strength_label(ml_edge)}). Across rolling origins, ML beats the strongest benchmark {stability.get('ml_wins', 0)} of {stability.get('n_origins', 0)} times.",
            ),
            evidence_card(
                "04",
                "Time-series models are useful benchmarks, not the headline engine",
                "ARIMA check",
                "Country ARIMA and ARIMAX+CISS are included as compact rolling-origin benchmarks. They are kept secondary because each country has only about 33 half-year observations.",
            ),
            evidence_card(
                "05",
                "Historical tiers add discipline to the story",
                "backtest",
                f"The historical replay checks whether higher diagnostic tiers line up with later stress levels. Recent summary: {tier_sentence}.",
            ),
            evidence_card(
                "06",
                "The honest boundary is diagnostic, not causal",
                "limitation",
                "The dashboard can flag borrower-side pressure, relative gaps, and forecast disagreement. It should not claim that any single macro, lender-side, loan-rate, or business-demography variable causes SME financing pain.",
            ),
        ],
    )


def professor_start_cards():
    latest = latest_period_df(PANEL)
    latest_period = latest["TIME_PERIOD"].iloc[0] if not latest.empty else "latest period"
    highest = latest.sort_values("SME_FPI_equal_z", ascending=False).iloc[0] if not latest.empty else None
    gap = latest.sort_values("Relative_Gap_equal", ascending=False).iloc[0] if not latest.empty else None
    board = FORECAST_DECISION_BOARD.copy()
    monitor_count = int(board["risk_tier"].isin(["Alert", "Watch", "Monitor"]).sum()) if "risk_tier" in board.columns else 0
    hidden_text = "none"
    rising_text = "none"
    if not board.empty:
        if "signal_type" not in board.columns:
            board["signal_type"] = [monitor_signal_type(row) for row in board.itertuples()]
        attention_board = board[board["risk_tier"].isin(["Alert", "Watch", "Monitor"])] if "risk_tier" in board.columns else board
        hidden_text = list_country_names(attention_board[attention_board["signal_type"].astype(str).str.contains("gap|hidden", case=False, na=False)])
        rising_text = list_country_names(attention_board[attention_board["signal_type"].astype(str).str.contains("rising|forecast", case=False, na=False)])
    summary = FORECAST_SUMMARY.iloc[0] if not FORECAST_SUMMARY.empty else pd.Series(dtype=object)
    best_model = summary.get("best_recent_model_label", "best recent model")
    ml_edge = summary.get("best_recent_ml_improvement_vs_strongest_baseline", np.nan)
    stability = forecast_stability_stats()

    return html.Div(
        className="start-summary-grid",
        children=[
            metric_card(
                "Latest reading",
                str(latest_period),
                f"highest SME-FPI: {highest['country_name']} ({format_number(highest['SME_FPI_equal_z'])})" if highest is not None else "panel endpoint",
            ),
            metric_card(
                "Largest borrower-market gap",
                str(gap["country_name"]) if gap is not None else "n/a",
                f"SME-FPI minus CISS = {format_number(gap['Relative_Gap_equal'])}" if gap is not None else "gap unavailable",
            ),
            metric_card(
                "Monitoring list",
                f"{monitor_count} countries",
                f"gap monitors: {hidden_text}; rising forecast: {rising_text}",
            ),
            metric_card(
                "Forecast check",
                str(best_model),
                f"recent ML edge vs strongest benchmark: {format_number(ml_edge, 3)}; ML wins {stability.get('ml_wins', 0)}/{stability.get('n_origins', 0)} origins",
            ),
        ],
    )


def board_counts(board):
    if board.empty or "risk_tier" not in board.columns:
        return {"Alert": 0, "Watch": 0, "Monitor": 0, "Normal": 0}
    counts = board["risk_tier"].value_counts()
    return {tier: int(counts.get(tier, 0)) for tier in ["Alert", "Watch", "Monitor", "Normal"]}


def board_period_label(board, period_range=None):
    if not board.empty and "TIME_PERIOD" in board.columns and board["TIME_PERIOD"].notna().any():
        return str(board["TIME_PERIOD"].dropna().iloc[0])
    if period_range:
        end_idx = period_range[1]
        if 0 <= end_idx < len(PERIODS):
            return str(PERIODS[end_idx])
    return "latest period"


def board_attention_rows(board, limit=5):
    if board.empty:
        return []
    attention = board[board["risk_tier"].isin(["Alert", "Watch", "Monitor"])].copy()
    if attention.empty:
        attention = board.head(limit).copy()
    return attention.head(limit)


def board_country_names(board, limit=3):
    return list_country_names(board, limit=limit) if not board.empty else "none"


def board_status_chips(board):
    counts = board_counts(board)
    hidden = board[
        board["signal_type"].isin(["Hidden-gap monitor", "Mixed monitor", "Visible or hidden high pressure"])
    ] if not board.empty and "signal_type" in board.columns else pd.DataFrame()
    rising = board[
        board["signal_type"].isin(["Rising-forecast monitor", "Mixed monitor", "Forward-looking warning"])
    ] if not board.empty and "signal_type" in board.columns else pd.DataFrame()
    return [
        html.Span(f"{counts['Alert']} Alert", className="status-chip status-chip-alert"),
        html.Span(f"{counts['Watch']} Watch", className="status-chip status-chip-watch"),
        html.Span(f"{counts['Monitor']} Monitor", className="status-chip status-chip-monitor"),
        html.Span(f"Hidden gap: {board_country_names(hidden)}", className="status-chip"),
        html.Span(f"Rising forecast: {board_country_names(rising)}", className="status-chip"),
    ]


def hero_monitor_snapshot():
    board = build_decision_board([0, len(PERIODS) - 1], DEFAULT_COUNTRIES, "SME_FPI_equal_z")
    if board.empty:
        return html.Div(
            className="hero-status-panel",
            children=[
                html.Span("Current read", className="panel-kicker"),
                html.H2("Monitoring board not available"),
                visual_bullets(["forecast layer needed", "watchlist unavailable"]),
            ],
        )
    counts = board_counts(board)
    period = board_period_label(board)
    top = board.iloc[0]
    attention = board_attention_rows(board, limit=3)
    return html.Div(
        className="hero-status-panel",
        children=[
            html.Span("Current read", className="panel-kicker"),
            html.H2(f"{period}: {counts['Alert']} Alert, {counts['Watch']} Watch, {counts['Monitor']} Monitor"),
            visual_bullets(["watchlist first", "drivers next", "forecast agreement", "validation"]),
            html.Div(className="hero-status-chips", children=board_status_chips(board)[:3]),
            html.Div(
                className="hero-watchlist",
                children=[
                    html.Div(
                        children=[
                            html.Strong(str(row.country_name)),
                            html.Span(f"{row.risk_tier} | score {format_number(row.risk_score, 1)}"),
                        ]
                    )
                    for row in attention.itertuples()
                ],
            ),
            html.Small(
                f"Top signal: {top['country_name']} because {str(top.get('primary_drivers', 'drivers are available in the board')).lower()}."
            ),
        ],
    )


def command_center_panel(board, metric_col, period_range):
    if board.empty:
        return html.Div(
            className="command-center-panel",
            children=[
                html.Div(
                    className="command-copy",
                    children=[
                        html.Span("Current board", className="panel-kicker"),
                        html.H2("No monitoring read for this selection"),
                        visual_bullets(["select a later period", "or rebuild the forecast layer"]),
                    ],
                )
            ],
        )
    counts = board_counts(board)
    period = board_period_label(board, period_range)
    attention = board_attention_rows(board, limit=6)
    hidden = board[
        board["signal_type"].isin(["Hidden-gap monitor", "Mixed monitor", "Visible or hidden high pressure"])
    ] if "signal_type" in board.columns else pd.DataFrame()
    rising = board[
        board["signal_type"].isin(["Rising-forecast monitor", "Mixed monitor", "Forward-looking warning"])
    ] if "signal_type" in board.columns else pd.DataFrame()
    high_agreement = int((board["agreement_quality"] == "High").sum()) if "agreement_quality" in board.columns else 0
    top = board.iloc[0]
    selected_label = INDEX_OPTIONS.get(metric_col, "selected index")
    status_sentence = (
        f"{period} has {counts['Alert']} Alert, {counts['Watch']} Watch, and {counts['Monitor']} Monitor countries "
        f"under the {selected_label}. This is a triage read, not a crisis probability."
    )
    return html.Div(
        className="command-center-panel",
        children=[
            html.Div(
                className="command-copy",
                children=[
                    html.Span("Current board", className="panel-kicker"),
                    html.H2("Who needs attention, why, and how confident is the signal?"),
                    compact_copy_bullets([("Status", status_sentence)], className="compact-guide-list command-status-list", limit=150),
                    html.Div(className="command-status-chips", children=board_status_chips(board)),
                ],
            ),
            html.Div(
                className="command-kpi-grid",
                children=[
                    metric_card("Current status", f"{counts['Alert'] + counts['Watch'] + counts['Monitor']}", f"{counts['Alert']} alert, {counts['Watch']} watch, {counts['Monitor']} monitor"),
                    metric_card("Top country", str(top["country_name"]), f"{top['risk_tier']} score {format_number(top['risk_score'], 1)}"),
                    metric_card("Hidden-gap reads", str(len(hidden)), board_country_names(hidden)),
                    metric_card("Rising forecast reads", str(len(rising)), board_country_names(rising)),
                    metric_card("High agreement", str(high_agreement), "model agreement, not certainty"),
                ],
            ),
            html.Div(
                className="watchlist-panel",
                children=[
                    html.Div(
                        className="watchlist-heading",
                        children=[
                            html.Span("Watchlist", className="panel-kicker"),
                            html.H3("Read these countries first"),
                        ],
                    ),
                    html.Div(
                        className="watchlist-rows",
                        children=[
                            html.Details(
                                className=f"watchlist-row watchlist-row-{str(row.risk_tier).lower()}",
                                children=[
                                    html.Summary(
                                        children=[
                                            html.Div(
                                                className="watchlist-country",
                                                children=[
                                                    html.Strong(str(row.country_name)),
                                                    html.Span(f"{row.risk_tier} | {row.signal_type}"),
                                                ],
                                            ),
                                            html.Div(
                                                className="watchlist-chip-grid",
                                                children=[
                                                    html.Span(f"Level {format_number(row.current_score)}"),
                                                    html.Span(f"Gap {format_number(row.gap_value)}"),
                                                    html.Span(f"H+1 {format_number(row.best_model_forecast)}"),
                                                    html.Span(f"Agreement {getattr(row, 'agreement_quality', 'n/a')}"),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="watchlist-reason",
                                        children=[
                                            html.Strong("Why flagged?"),
                                            compact_copy_bullets(
                                                [("Drivers", str(row.primary_drivers))],
                                                className="compact-guide-list watchlist-reason-list",
                                                limit=150,
                                            ),
                                            html.Small(str(getattr(row, "recommended_read", "Inspect the linked evidence tabs."))),
                                        ],
                                    ),
                                ],
                            )
                            for row in attention.itertuples()
                        ],
                    ),
                ],
            ),
        ],
    )


def reading_path_node_class(target, active_target):
    suffix = " reading-path-node--active" if target == active_target else ""
    return f"reading-path-node{suffix}"


def reading_path_node_classes(active_target):
    return [reading_path_node_class(step["target"], active_target) for step in READING_PATH_STEPS]


def professor_reading_path(active_target="start"):
    return html.Div(
        className="reading-path-timeline",
        children=[
            html.Div(
                className="reading-path-track",
                children=[
                    html.Button(
                        id=f"read-path-{step['target']}",
                        n_clicks=0,
                        className=reading_path_node_class(step["target"], active_target),
                        children=[
                            html.Span(step["number"], className="reading-node-number"),
                            html.Strong(step["title"]),
                            html.Small(step["short"]),
                        ],
                    )
                    for step in READING_PATH_STEPS
                ],
            ),
            html.Div(id="reading-path-detail", className="reading-path-detail", children=reading_path_detail("start")),
        ],
    )


def reading_path_detail(target):
    step = READING_PATH_LOOKUP.get(target, READING_PATH_LOOKUP["start"])
    return [
        html.Div(
            className="reading-detail-title",
            children=[
                html.Span(f"{step['number']} / {step['short']}", className="panel-kicker"),
                html.H3(step["title"]),
            ],
        ),
        html.Div(
            className="reading-detail-copy",
            children=[
                compact_copy_bullets(
                    [
                        ("Read", step["body"]),
                        ("Why", step["why"]),
                        ("Outcome", step["outcome"]),
                    ],
                    className="compact-guide-list reading-detail-bullets",
                    limit=150,
                ),
            ],
        ),
    ]


def professor_glossary_panel():
    terms = [
        ("SAFE", "ECB firm survey", "The source that asks firms directly about financing conditions."),
        ("CISS", "market-stress benchmark", "A financial-market stress indicator, not an SME borrower survey."),
        ("SME-FPI", "our borrower-side index", "Higher values mean SMEs report above-average financing pain."),
        ("Relative gap", "SME-FPI minus CISS", "Positive values mean borrower pain is high relative to market stress."),
        ("H+1", "next half-year", "The forecast target period: one semiannual step ahead."),
        ("MAE", "average forecast error", "Lower is better; one unit is one standardized-index point."),
    ]
    return html.Div(
        className="glossary-grid",
        children=[concept_card(title, tag, body) for title, tag, body in terms],
    )


def professor_submission_panel():
    return html.Div(
        className="submission-panel",
        children=[
            html.Div(
                children=[
                    html.Span("Submission reader guide", className="panel-kicker"),
                    html.H3("The dashboard should be read as a self-contained analytical report"),
                    visual_bullets(
                        [
                            "no SME-finance background assumed",
                            "borrower-side index first",
                            "CISS benchmark comparison",
                            "monitoring value checked",
                        ]
                    ),
                ],
            ),
            html.Div(
                className="submission-rule-list",
                children=[
                    html.Div(children=[html.Strong("Main claim"), html.Span("SME borrower pressure can diverge from market stress.")]),
                    html.Div(children=[html.Strong("Evidence"), html.Span("levels, gaps, forecast errors, tier validation")]),
                    html.Div(children=[html.Strong("Boundary"), html.Span("monitoring, not causal policy or crisis prediction")]),
                ],
            ),
        ],
    )


def make_baseline_dominance_figure():
    if FORECAST_EVALUATION.empty or "model_key" not in FORECAST_EVALUATION.columns:
        return empty_figure("Forecast evaluation is not available.")
    data = FORECAST_EVALUATION.copy()
    data["model_family"] = data.get("model_family", data["model_key"].map(MODEL_FAMILIES)).fillna("model")
    strongest_benchmark = (
        data[data["model_key"].isin(BASELINE_MODEL_KEYS)]
        .groupby("origin_period_sort")["mae"]
        .min()
        .rename("strongest_benchmark_mae")
    )
    best_ml = (
        data[data["model_key"].isin(ML_MODEL_KEYS)]
        .sort_values("mae")
        .groupby("origin_period_sort", as_index=False)
        .first()[["origin_period_sort", "origin_period", "model_label", "mae"]]
        .rename(columns={"mae": "best_ml_mae", "model_label": "best_ml_model"})
    )
    plot = best_ml.merge(strongest_benchmark, on="origin_period_sort", how="left")
    plot["ml_edge"] = plot["strongest_benchmark_mae"] - plot["best_ml_mae"]
    plot = plot.sort_values("origin_period_sort")
    win_count = int((plot["ml_edge"] > 0).sum())
    origin_count = int(len(plot))
    median_edge = float(plot["ml_edge"].median()) if origin_count else np.nan
    colors = np.where(plot["ml_edge"] >= 0, "#2f6f9f", "#d19a2e")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot["origin_period"],
            y=plot["ml_edge"],
            marker={"color": colors},
            customdata=np.stack([plot["best_ml_model"], plot["best_ml_mae"], plot["strongest_benchmark_mae"]], axis=-1),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Best ML model: %{customdata[0]}<br>"
                "Best ML MAE: %{customdata[1]:.3f}<br>"
                "Strongest benchmark MAE: %{customdata[2]:.3f}<br>"
                "ML edge: %{y:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#8b98a5")
    fig.update_xaxes(title_text="Forecast origin", tickangle=-45)
    fig.update_yaxes(title_text="MAE improvement vs strongest benchmark")
    return polish(
        fig,
        chart_title(
            "ML usually helps, but the edge is not always large",
            f"Best ML beats the strongest benchmark in {win_count}/{origin_count} origins; median edge {format_number(median_edge, 3)} MAE",
        ),
        height=470,
    )


def make_forecast_loss_figure(countries=None):
    if FORECAST_EVALUATION.empty or "model_key" not in FORECAST_EVALUATION.columns:
        return empty_figure("Forecast error evaluation is not available.")
    evaluation = FORECAST_EVALUATION.copy()
    required = {"origin_period_sort", "origin_period", "model_key", "model_label", "mae", "rmse"}
    if not required.issubset(evaluation.columns):
        return empty_figure("Forecast evaluation file is missing MAE/RMSE fields.")
    evaluation["model_family"] = evaluation.get("model_family", evaluation["model_key"].map(MODEL_FAMILIES)).fillna("model")

    best_ml = (
        evaluation[evaluation["model_key"].isin(ML_MODEL_KEYS)]
        .sort_values(["origin_period_sort", "mae"])
        .groupby("origin_period_sort", as_index=False)
        .first()[["origin_period_sort", "origin_period", "model_label", "mae", "rmse"]]
        .rename(columns={"model_label": "best_ml_model", "mae": "best_ml_mae", "rmse": "best_ml_rmse"})
    )
    strongest_benchmark = (
        evaluation[evaluation["model_key"].isin(BASELINE_MODEL_KEYS)]
        .sort_values(["origin_period_sort", "mae"])
        .groupby("origin_period_sort", as_index=False)
        .first()[["origin_period_sort", "model_label", "mae", "rmse"]]
        .rename(
            columns={
                "model_label": "strongest_benchmark_model",
                "mae": "strongest_benchmark_mae",
                "rmse": "strongest_benchmark_rmse",
            }
        )
    )
    trend = best_ml.merge(strongest_benchmark, on="origin_period_sort", how="inner").sort_values("origin_period_sort")
    if trend.empty:
        return empty_figure("No comparable ML and benchmark forecast errors are available.")
    trend["ml_edge"] = trend["strongest_benchmark_mae"] - trend["best_ml_mae"]

    country_error = pd.DataFrame()
    best_key = best_forecast_model_key()
    if not FORECAST_COUNTRY_ERROR.empty and "model_key" in FORECAST_COUNTRY_ERROR.columns:
        country_error = FORECAST_COUNTRY_ERROR[FORECAST_COUNTRY_ERROR["model_key"] == best_key].copy()
        if country_error.empty and "model_family" in FORECAST_COUNTRY_ERROR.columns:
            country_error = (
                FORECAST_COUNTRY_ERROR[FORECAST_COUNTRY_ERROR["model_family"] == "machine learning"]
                .sort_values(["REF_AREA", "mean_abs_error"])
                .groupby(["REF_AREA", "country_name"], as_index=False)
                .first()
            )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Rolling-origin forecast error", "Country error, current best model"),
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.16,
    )
    fig.add_trace(
        go.Scatter(
            x=trend["origin_period"],
            y=trend["best_ml_mae"],
            mode="lines+markers",
            name="Best ML MAE",
            line={"color": "#2f6f9f", "width": 3},
            marker={"size": 7},
            customdata=np.stack(
                [
                    trend["best_ml_model"],
                    trend["best_ml_rmse"],
                    trend["strongest_benchmark_model"],
                    trend["strongest_benchmark_mae"],
                    trend["ml_edge"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Best ML model: %{customdata[0]}<br>"
                "ML MAE: %{y:.3f}<br>"
                "ML RMSE: %{customdata[1]:.3f}<br>"
                "Strongest benchmark: %{customdata[2]}<br>"
                "Benchmark MAE: %{customdata[3]:.3f}<br>"
                "ML edge: %{customdata[4]:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trend["origin_period"],
            y=trend["strongest_benchmark_mae"],
            mode="lines+markers",
            name="Strongest benchmark MAE",
            line={"color": "#8b98a5", "width": 2, "dash": "dash"},
            marker={"size": 6},
            customdata=np.stack(
                [trend["strongest_benchmark_model"], trend["strongest_benchmark_rmse"], trend["best_ml_model"]],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Benchmark: %{customdata[0]}<br>"
                "Benchmark MAE: %{y:.3f}<br>"
                "Benchmark RMSE: %{customdata[1]:.3f}<br>"
                "Best ML model: %{customdata[2]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    if not country_error.empty:
        selected = set(selected_or_default(countries))
        country_error = country_error.sort_values("mean_abs_error", ascending=True)
        colors = np.where(country_error["REF_AREA"].isin(selected), "#2f6f9f", "#aeb8c2")
        fig.add_trace(
            go.Bar(
                x=country_error["mean_abs_error"],
                y=country_error["country_name"],
                orientation="h",
                name="Country MAE",
                marker={"color": colors, "line": {"color": "#ffffff", "width": 0.8}},
                customdata=np.stack(
                    [
                        country_error.get("model_label", pd.Series(["Best model"] * len(country_error))),
                        country_error.get("rmse", pd.Series([np.nan] * len(country_error))),
                        country_error.get("bias", pd.Series([np.nan] * len(country_error))),
                        country_error.get("direction_hit_share", pd.Series([np.nan] * len(country_error))) * 100,
                        country_error.get("n_origins", pd.Series([np.nan] * len(country_error))),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Model: %{customdata[0]}<br>"
                    "Mean abs. error: %{x:.3f}<br>"
                    "RMSE: %{customdata[1]:.3f}<br>"
                    "Bias: %{customdata[2]:.3f}<br>"
                    "Directional hit: %{customdata[3]:.0f}%<br>"
                    "Rolling origins: %{customdata[4]:.0f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    else:
        fig.add_annotation(
            text="Country-level forecast errors are unavailable.",
            x=0.5,
            y=0.5,
            xref="x2 domain",
            yref="y2 domain",
            showarrow=False,
            font={"color": "#53616d"},
        )

    fig.update_xaxes(title_text="Forecast origin", tickangle=-45, row=1, col=1)
    fig.update_yaxes(title_text="MAE, lower is better", row=1, col=1)
    fig.update_xaxes(title_text="Mean absolute error", row=1, col=2)
    fig.update_yaxes(title_text=None, row=1, col=2)
    fig = polish(
        fig,
        chart_title(
            "Forecast loss is reported as out-of-sample error",
            "MAE is the main loss metric because SME-FPI is standardized; RMSE is kept in hover text to reveal larger misses",
        ),
        height=560,
    )
    fig.update_layout(margin={"l": 74, "r": 36, "t": 132, "b": 104}, legend={"orientation": "h", "y": 1.04})
    return fig


def forecast_feature_list():
    groups = [
        ("Current pain", "SME-FPI variants, component z-scores, PCA coordinates, CISS gap"),
        ("Macro climate", "GDP slowdown, unemployment, inflation, credit contraction, macro stress context"),
        ("Micro vulnerability", "SAFE Q0B access severity by firm size, sector, and broad problem pressure"),
        ("Lender-side signal", "ECB BLS SME credit standards, loan demand, terms, and rejection pressure"),
        ("Loan market signal", "ECB MIR small-loan rates, large-loan benchmark, spreads, and volumes"),
        ("Business outcome signal", "Eurostat quarterly registrations and bankruptcy indexes as external real-economy predictors"),
        ("Time-series benchmark", "Country ARIMA(1,0,0) and ARIMAX+CISS test whether a compact country-level time-series model is competitive"),
        ("Model suite", "Elastic Net, Ridge, Random Forest, and Gradient Boosting compared against naive, country mean, country AR(1), ARIMA/ARIMAX, momentum, and pooled OLS baselines"),
    ]
    return html.Div(
        className="forecast-feature-grid",
        children=[concept_card(title, "predictor layer", body) for title, body in groups],
    )


def read_step(number, title, body):
    return html.Div(
        className="read-step",
        children=[
            html.Span(number),
            html.Div(children=[html.Strong(title), html.P(compact_text(body, 112))]),
        ],
    )


def term_item(title, tag, body):
    return html.Div(
        className="term-item",
        children=[
            html.Span(tag),
            html.Strong(title),
            html.P(compact_text(body, 108)),
        ],
    )


def variable_item(symbol, title, body):
    return html.Div(
        className="variable-item",
        children=[
            dcc.Markdown(symbol, mathjax=True, className="variable-symbol"),
            html.Div(children=[html.Strong(title), html.P(compact_text(body, 112))]),
        ],
    )


def journey_card(number, title, body):
    return html.Div(
        className="journey-card",
        children=[
            html.Span(number, className="journey-number"),
            html.H3(title),
            html.P(compact_text(body, 112)),
        ],
    )


def methodology_pipeline():
    steps = [
        ("01", "Raw sources", "SAFE, CISS, World Bank, BLS, MIR, Eurostat"),
        ("02", "Cleaned panels", "Country-half-year harmonization and source-specific aggregation"),
        ("03", "SME-FPI core", "Borrower-side SAFE components, z-scores, coverage rules, robustness weights"),
        ("04", "Diagnostic layers", "Relative SME-CISS gap, PCA clusters, SAFE survey-detail slices, country cards"),
        ("05", "Forecast layer", "Rolling-origin baselines, ARIMA/ARIMAX, ML models, model agreement"),
        ("06", "Validation", "Future outcomes, historical monitoring tiers, country forecast errors, limitations"),
    ]
    nodes = []
    for number, title, body in steps:
        nodes.append(
            html.Div(
                className="pipeline-flow-node",
                children=[
                    html.Span(number),
                    html.Strong(title),
                    html.P(compact_text(body, 112)),
                ],
            )
        )
    return html.Div(className="pipeline-flow", children=nodes)


def control(label, element):
    return html.Div(className="control", children=[html.Span(label), element])


def loading_graph(graph_id, className="chart", figure=None):
    graph_kwargs = {
        "id": graph_id,
        "className": className,
        "config": {"displayModeBar": False, "responsive": True},
    }
    if figure is not None:
        graph_kwargs["figure"] = figure
    return dcc.Loading(
        className="graph-loading",
        type="circle",
        color="#2f6f9f",
        children=dcc.Graph(**graph_kwargs),
    )


def relative_display_path(path):
    try:
        return str(path.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def dataset_source_label(item):
    text = f"{item.get('label', '')} {item.get('path', '')}".lower()
    if "sme-fpi" in text or "sme_fpi" in text:
        return "ECB SAFE-derived processed panel"
    if "safe" in text:
        return "ECB SAFE survey"
    if "ciss" in text:
        return "ECB CISS market-stress benchmark"
    if "worldbank" in text or "macro" in text:
        return "World Bank macro context"
    if "bls" in text:
        return "ECB Bank Lending Survey"
    if "mir" in text:
        return "ECB MFI Interest Rate Statistics"
    if "eurostat" in text:
        return "Eurostat business demography"
    if "forecast" in text or "decision" in text or "validation" in text:
        return "Generated project output"
    return "Generated analytical layer"


def dataset_frequency_label(item):
    identity = f"{item.get('key', '')} {item.get('label', '')} {item.get('layer', '')} {item.get('path', '')}".lower()
    if "daily" in identity:
        return "Daily source, semiannual dashboard aggregation"
    if "quarter" in identity or "eurostat" in identity or "bls" in identity:
        return "Quarterly source, semiannual dashboard aggregation"
    if "monthly" in identity or "mir" in identity:
        return "Monthly source, semiannual dashboard aggregation"
    if any(token in identity for token in ["forecast", "decision", "tier", "error", "evaluation"]):
        return "Rolling-origin country-half-year output"
    return "Country-half-year analytical panel"


def dataset_used_in_label(item):
    text = f"{item.get('key', '')} {item.get('label', '')} {item.get('purpose', '')}".lower()
    tabs = []
    if any(token in text for token in ["main panel", "sme-fpi", "component", "ciss"]):
        tabs.extend(["Index Basics", "Trend Explorer", "Borrower-Market Gap"])
    if any(token in text for token in ["forecast", "model", "decision", "tier", "error"]):
        tabs.extend(["Monitoring Board", "Forecast Check", "Country Diagnosis"])
    if any(token in text for token in ["cube", "q0b", "severity"]):
        tabs.append("Firm Survey Detail")
    if any(token in text for token in ["validation", "catalog", "source"]):
        tabs.append("Data & Method")
    if not tabs:
        tabs.append("Data Preview")
    return ", ".join(dict.fromkeys(tabs))


def metadata_item(label, value):
    return html.Div(
        className="dataset-metadata-item",
        children=[html.Span(label), html.Strong(value)],
    )


def dataset_preview_options():
    return [
        {
            "label": f"{item['label']} - {item['layer']}",
            "value": item["key"],
        }
        for item in DATASET_PREVIEWS
        if item["path"].exists()
    ]


def render_dataset_preview(dataset_key, row_count):
    item = DATASET_PREVIEW_LOOKUP.get(dataset_key) or DATASET_PREVIEWS[0]
    path = item["path"]
    safe_row_count = int(row_count or 10)
    if safe_row_count not in {5, 10, 20, 50}:
        safe_row_count = 10

    if not path.exists():
        return html.Div(
            className="source-table-empty",
            children=f"Dataset file is missing: {relative_display_path(path)}",
        )

    try:
        preview = pd.read_csv(path, nrows=safe_row_count)
    except Exception as exc:
        return html.Div(
            className="source-table-empty",
            children=f"Could not preview {relative_display_path(path)}: {exc}",
        )

    preview = preview.replace({np.nan: ""})
    columns = [str(col) for col in preview.columns]
    rows = [
        html.Tr([html.Td(str(value)) for value in row])
        for row in preview.itertuples(index=False, name=None)
    ]
    row_label = f"{int(item['rows']):,}" if item.get("rows") is not None and pd.notna(item.get("rows")) else "see file"
    key_columns = ", ".join(columns[:6]) + (" ..." if len(columns) > 6 else "")

    return html.Div(
        className="dataset-preview-panel",
        children=[
            html.Div(
                className="dataset-preview-summary",
                children=[
                    metric_card("Selected dataset", item["label"], item["layer"]),
                    metric_card("File path", relative_display_path(path), "local project file"),
                    metric_card("Rows in layer", row_label, f"showing first {safe_row_count} rows"),
                    metric_card("Columns", f"{len(columns):,}", "all preview columns are shown"),
                ],
            ),
            html.Div(
                className="dataset-purpose-panel",
                children=[
                    html.Span("How this dataset is used", className="panel-kicker"),
                    compact_copy_bullets([("Purpose", item["purpose"])], className="compact-guide-list dataset-purpose-list", limit=150),
                    html.Div(
                        className="dataset-metadata-grid",
                        children=[
                            metadata_item("Source", dataset_source_label(item)),
                            metadata_item("Frequency", dataset_frequency_label(item)),
                            metadata_item("Used in tabs", dataset_used_in_label(item)),
                            metadata_item("Key columns", key_columns),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="source-table-wrap dataset-preview-table-wrap",
                children=[
                    html.Table(
                        className="source-table dataset-preview-table",
                        children=[
                            html.Thead(html.Tr([html.Th(col) for col in columns])),
                            html.Tbody(rows),
                        ],
                    )
                ],
            ),
        ],
    )


def make_map(df, metric_col):
    if df.empty:
        return empty_figure("No country data are available for the selected period range.")
    latest = latest_period_df(add_interpretation(df, metric_col))
    latest_period = latest["TIME_PERIOD"].iloc[0]
    tile_positions = {
        "FI": (5, 0),
        "IE": (0, 2),
        "NL": (2, 1),
        "BE": (2, 2),
        "FR": (2, 3),
        "PT": (1, 4),
        "ES": (2, 4),
        "DE": (3, 1),
        "AT": (4, 2),
        "IT": (4, 3),
        "SK": (5, 2),
        "GR": (5, 4),
    }
    tile_df = latest.copy()
    tile_df["tile_x"] = tile_df["REF_AREA"].map(lambda code: tile_positions.get(code, (np.nan, np.nan))[0])
    tile_df["tile_y"] = tile_df["REF_AREA"].map(lambda code: tile_positions.get(code, (np.nan, np.nan))[1])
    tile_df = tile_df.dropna(subset=["tile_x", "tile_y"]).copy()
    scale_limit = max(1.0, float(latest["metric_value"].abs().max()))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "choropleth"}, {"type": "xy"}]],
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.04,
        subplot_titles=("Muted geographic context", "Equal-size stress tiles"),
    )
    fig.add_trace(
        go.Choropleth(
            locations=latest["iso3"],
            z=latest["metric_value"],
            text=latest["country_name"],
            customdata=np.stack(
                [
                    latest["TIME_PERIOD"],
                    latest["metric_value"],
                    latest["metric_label"],
                    latest["CISS_z"],
                    latest["gap_value"],
                    latest["gap_label"],
                    latest["coverage_label"],
                ],
                axis=-1,
            ),
            colorscale=STRESS_SCALE,
            zmid=0,
            zmin=-scale_limit,
            zmax=scale_limit,
            marker_line_color="white",
            marker_line_width=0.7,
            colorbar={"title": "z-score"},
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Period: %{customdata[0]}<br>"
                "SME-FPI: %{customdata[1]:.2f}<br>"
                "%{customdata[2]}<br>"
                "CISS z: %{customdata[3]:.2f}<br>"
                "Relative gap: %{customdata[4]:.2f}<br>"
                "%{customdata[5]}<br>"
                "Coverage: %{customdata[6]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tile_df["tile_x"],
            y=tile_df["tile_y"],
            mode="markers+text",
            text=tile_df["REF_AREA"],
            textposition="middle center",
            marker={
                "symbol": "square",
                "size": 58,
                "color": tile_df["metric_value"],
                "colorscale": STRESS_SCALE,
                "cmin": -scale_limit,
                "cmax": scale_limit,
                "line": {"color": "white", "width": 1.4},
                "showscale": False,
            },
            textfont={"color": "#13202b", "size": 13, "family": "Inter, Segoe UI, Arial, sans-serif"},
            customdata=np.stack(
                [
                    tile_df["country_name"],
                    tile_df["TIME_PERIOD"],
                    tile_df["metric_value"],
                    tile_df["metric_label"],
                    tile_df["CISS_z"],
                    tile_df["gap_value"],
                    tile_df["gap_label"],
                    tile_df["coverage_label"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Period: %{customdata[1]}<br>"
                "SME-FPI: %{customdata[2]:.2f}<br>"
                "%{customdata[3]}<br>"
                "CISS z: %{customdata[4]:.2f}<br>"
                "Relative gap: %{customdata[5]:.2f}<br>"
                "%{customdata[6]}<br>"
                "Coverage: %{customdata[7]}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_geos(
        fitbounds="locations",
        visible=True,
        bgcolor="rgba(0,0,0,0)",
        showland=True,
        landcolor="rgba(148, 163, 184, 0.16)",
        showcountries=True,
        countrycolor="rgba(255,255,255,0.85)",
        showcoastlines=False,
        showframe=False,
    )
    fig.update_xaxes(visible=False, range=[-0.7, 5.7], row=1, col=2)
    fig.update_yaxes(visible=False, range=[4.7, -0.7], row=1, col=2, scaleanchor="x2", scaleratio=1)
    fig = polish(
        fig,
        chart_title(
            "Where is borrower-side SME stress visible?",
            f"{INDEX_OPTIONS[metric_col]}, {latest_period}; map for geography, tiles for fair comparison",
        ),
        height=470,
        showlegend=False,
    )
    fig.update_layout(margin={"l": 28, "r": 28, "t": 132, "b": 76})
    return fig


def make_time_series(df, countries, metric_col):
    countries = selected_or_default(countries)
    country_df = add_interpretation(df[df["REF_AREA"].isin(countries)], metric_col)
    if country_df.empty:
        return empty_figure("Select at least one country with available data.")
    country_df = country_df.sort_values(["period_index", "country_name"])

    if len(countries) > FOCUS_COUNTRY_LIMIT:
        grouped = (
            country_df.groupby(["TIME_PERIOD", "period_index"], as_index=False)
            .agg(
                median_score=("metric_value", "median"),
                q25=("metric_value", lambda values: values.quantile(0.25)),
                q75=("metric_value", lambda values: values.quantile(0.75)),
                ciss_z=("CISS_z", "mean"),
            )
            .sort_values("period_index")
        )
        latest = latest_period_df(country_df)
        highlight_codes = latest.nlargest(3, "metric_value")["REF_AREA"].tolist()
        highlight_df = country_df[country_df["REF_AREA"].isin(highlight_codes)]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=grouped["TIME_PERIOD"],
                y=grouped["q75"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
                name="75th percentile",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["TIME_PERIOD"],
                y=grouped["q25"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(47, 111, 159, 0.18)",
                line={"width": 0},
                name="Middle 50% cross-country band, not CI",
                hovertemplate="Middle 50% cross-country band<br>Period: %{x}<br>This is dispersion, not a statistical confidence interval.<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["TIME_PERIOD"],
                y=grouped["median_score"],
                mode="lines+markers",
                name="Selected-country median SME-FPI",
                line={"color": "#2f6f9f", "width": 3},
                hovertemplate="Median SME-FPI<br>Period: %{x}<br>Median z-score: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grouped["TIME_PERIOD"],
                y=grouped["ciss_z"],
                mode="lines",
                name="CISS z benchmark",
                line={"color": "#1f2933", "dash": "dash", "width": 2.5},
                hovertemplate="CISS z benchmark<br>Period: %{x}<br>CISS z: %{y:.2f}<extra></extra>",
            )
        )
        for color, (country_name, country_slice) in zip(
            px.colors.qualitative.Safe,
            highlight_df.groupby("country_name", sort=False),
        ):
            fig.add_trace(
                go.Scatter(
                    x=country_slice["TIME_PERIOD"],
                    y=country_slice["metric_value"],
                    mode="lines",
                    name=f"Latest high: {country_name}",
                    line={"color": color, "width": 2.1},
                    opacity=0.86,
                    hovertemplate=(
                        f"<b>{country_name}</b><br>"
                        "Period: %{x}<br>"
                        "SME-FPI: %{y:.2f}<extra></extra>"
                    ),
                )
            )
        fig.add_hline(y=0, line_dash="dot", line_color="#8b98a5")
        fig = polish(
            fig,
            chart_title(
                "Market stress can ease while SME pain remains uneven",
                "Median SME-FPI, middle-50% country band, latest high-stress countries, and CISS benchmark",
            ),
            height=470,
            y_title="Standardized score",
            x_title="Half-year period",
        )
        fig.update_layout(
            legend={"orientation": "h", "yanchor": "top", "y": -0.2, "xanchor": "left", "x": 0},
            margin={"l": 54, "r": 28, "t": 132, "b": 118},
        )
        return apply_period_ticks(fig, grouped["TIME_PERIOD"].tolist(), max_ticks=6, angle=0)

    fig = px.line(
        country_df,
        x="TIME_PERIOD",
        y="metric_value",
        color="country_name",
        markers=True,
        custom_data=["metric_label", "gap_value", "gap_label", "coverage_label", "CISS_z"],
        category_orders={"TIME_PERIOD": PERIODS},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Period: %{x}<br>"
            "SME-FPI: %{y:.2f}<br>"
            "%{customdata[0]}<br>"
            "CISS z: %{customdata[4]:.2f}<br>"
            "Relative gap: %{customdata[1]:.2f}<br>"
            "%{customdata[2]}<br>"
            "Coverage: %{customdata[3]}<extra></extra>"
        )
    )
    ciss = (
        country_df[["TIME_PERIOD", "period_index", "CISS_z"]]
        .drop_duplicates()
        .sort_values("period_index")
    )
    fig.add_trace(
        go.Scatter(
            x=ciss["TIME_PERIOD"],
            y=ciss["CISS_z"],
            mode="lines",
            name="CISS z benchmark",
            line={"color": "#1f2933", "dash": "dash", "width": 2.5},
            hovertemplate="CISS z benchmark<br>Period: %{x}<br>CISS z: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#8b98a5")
    fig = polish(
        fig,
        chart_title(
            "Do selected countries move with market stress?",
            "SME-FPI country paths compared with the common CISS benchmark",
        ),
        height=470,
        y_title="Standardized score",
        x_title="Half-year period",
    )
    return apply_period_ticks(fig, country_df["TIME_PERIOD"].drop_duplicates().tolist(), max_ticks=6, angle=0)


def make_heatmap(df, metric_col):
    if df.empty:
        return empty_figure("No data are available for the selected period range.")
    plot_df = add_interpretation(df, metric_col)
    pivot = plot_df.pivot_table(
        index="country_name",
        columns="TIME_PERIOD",
        values="metric_value",
        aggfunc="mean",
    )
    coverage = plot_df.pivot_table(
        index="country_name",
        columns="TIME_PERIOD",
        values="components_available",
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=[period for period in PERIODS if period in pivot.columns])
    coverage = coverage.reindex(index=pivot.index, columns=pivot.columns)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            customdata=np.dstack([coverage.reindex(index=pivot.index).values]),
            colorscale=STRESS_SCALE,
            zmid=0,
            colorbar={"title": "z-score"},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Period: %{x}<br>"
                "SME-FPI: %{z:.2f}<br>"
                "Components available: %{customdata[0]:.0f} / 6<extra></extra>"
            ),
        )
    )
    fig = polish(
        fig,
        chart_title(
            "Where does SME financing pain persist?",
            "Country-period matrix of standardized borrower-side stress",
        ),
        height=480,
        y_title="Country",
        x_title="Half-year period",
        showlegend=False,
    )
    return apply_period_ticks(fig, pivot.columns.tolist(), max_ticks=6, angle=0)


def make_component_heatmap(df, countries):
    countries = selected_or_default(countries)
    subset = df[df["REF_AREA"].isin(countries)].copy()
    if subset.empty:
        return empty_figure("No component data are available for the selected countries.")
    melted = subset.melt(
        id_vars=["country_name"],
        value_vars=Z_COMPONENTS,
        var_name="component",
        value_name="z_value",
    )
    melted["component_label"] = melted["component"].map(COMPONENT_LABELS)
    grouped = (
        melted.groupby(["country_name", "component_label"], as_index=False)["z_value"]
        .mean()
    )
    pivot = grouped.pivot(index="country_name", columns="component_label", values="z_value")
    ordered_components = [COMPONENT_LABELS[z_component] for z_component in Z_COMPONENTS]
    pivot = pivot.reindex(columns=ordered_components)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=STRESS_SCALE,
            zmid=0,
            colorbar={"title": "mean z"},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x}<br>"
                "Average standardized component: %{z:.2f}<extra></extra>"
            ),
        )
    )
    return polish(
        fig,
        chart_title(
            "Which financing frictions drive each country?",
            "Average standardized contribution of the six SME-FPI components",
        ),
        height=430,
        y_title="Country",
        x_title="SME-FPI component",
        showlegend=False,
    )


def make_gap_ranking(df, metric_col):
    if df.empty:
        return empty_figure("No gap data are available for the selected period range.")
    gap_col = GAP_BY_INDEX[metric_col]
    latest = latest_period_df(add_interpretation(df, metric_col))
    latest = latest.sort_values(gap_col, ascending=True)
    fig = px.bar(
        latest,
        x=gap_col,
        y="country_name",
        orientation="h",
        color=gap_col,
        color_continuous_scale=STRESS_SCALE,
        color_continuous_midpoint=0,
        custom_data=["TIME_PERIOD", "SME_FPI_equal_z", "CISS_z", "gap_label", "coverage_label"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Period: %{customdata[0]}<br>"
            "Relative gap: %{x:.2f}<br>"
            "Equal-weight SME-FPI: %{customdata[1]:.2f}<br>"
            "CISS z: %{customdata[2]:.2f}<br>"
            "%{customdata[3]}<br>"
            "Coverage: %{customdata[4]}<extra></extra>"
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#6b7580")
    fig.update_coloraxes(showscale=False)
    return polish(
        fig,
        chart_title(
            "Hidden borrower stress: who sits above the CISS benchmark?",
            "Latest-period SME-FPI minus common market-stress z-score",
        ),
        height=440,
        y_title=None,
        x_title="SME-FPI minus CISS z-score",
        showlegend=False,
    )


def make_bubble_gap(df, countries, metric_col):
    countries = selected_or_default(countries)
    subset = add_interpretation(df[df["REF_AREA"].isin(countries)], metric_col)
    if subset.empty:
        return empty_figure("No selected country-period observations are available.")
    title = chart_title(
        "Stress disagreement diagnostic",
        "Borrower-side SME pain plotted against market-side CISS",
    )
    if len(countries) > FOCUS_COUNTRY_LIMIT:
        subset = latest_period_df(subset)
        title = chart_title(
            "Latest stress disagreement diagnostic",
            "Equal-size latest-period comparison avoids overplotting when many countries are selected",
        )
    fig = px.scatter(
        subset,
        x="CISS_z",
        y="metric_value",
        size="components_available",
        color="country_name",
        animation_frame=None,
        custom_data=["TIME_PERIOD", "gap_value", "gap_label", "coverage_label", "metric_label"],
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(
        marker={"sizemin": 7, "line": {"width": 0.7, "color": "white"}},
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Period: %{customdata[0]}<br>"
            "CISS z: %{x:.2f}<br>"
            "SME-FPI: %{y:.2f}<br>"
            "%{customdata[4]}<br>"
            "Relative gap: %{customdata[1]:.2f}<br>"
            "%{customdata[2]}<br>"
            "Coverage: %{customdata[3]}<extra></extra>"
        ),
    )
    x_vals = subset["CISS_z"].dropna()
    y_vals = subset["metric_value"].dropna()
    if not x_vals.empty and not y_vals.empty:
        low = min(x_vals.min(), y_vals.min()) - 0.2
        high = max(x_vals.max(), y_vals.max()) + 0.2
        fig.add_trace(
            go.Scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                name="SME-FPI = CISS",
                line={"color": "#6b7580", "dash": "dot"},
                hoverinfo="skip",
            )
        )
    fig = polish(
        fig,
        title,
        height=450,
        y_title="Selected SME-FPI z-score",
        x_title="CISS z-score",
    )
    fig.update_layout(
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "xanchor": "left", "x": 0},
        margin={"l": 54, "r": 28, "t": 132, "b": 142},
    )
    return fig


def make_animated_stress_motion(df, countries, metric_col):
    countries = selected_or_default(countries)
    motion = add_interpretation(df[df["REF_AREA"].isin(countries)], metric_col)
    motion = motion.dropna(subset=["CISS_z", "metric_value", "TIME_PERIOD"]).copy()
    if motion.empty:
        return empty_figure("Select at least one country to animate its stress trajectory.")

    motion["cluster_display"] = motion["cluster_name"].map(CLUSTER_DISPLAY_NAMES).fillna(motion["cluster_name"])
    motion["bubble_size"] = motion["gap_value"].abs().fillna(0) + 0.35
    motion["country_label"] = motion["REF_AREA"]
    frame_order = (
        motion[["TIME_PERIOD", "period_index"]]
        .drop_duplicates()
        .sort_values("period_index")["TIME_PERIOD"]
        .tolist()
    )
    x_values = motion["CISS_z"].dropna()
    y_values = motion["metric_value"].dropna()
    low = min(x_values.min(), y_values.min()) - 0.45
    high = max(x_values.max(), y_values.max()) + 0.45

    fig = px.scatter(
        motion,
        x="CISS_z",
        y="metric_value",
        animation_frame="TIME_PERIOD",
        animation_group="REF_AREA",
        color="cluster_display",
        size="bubble_size",
        text="country_label",
        category_orders={
            "TIME_PERIOD": frame_order,
            "cluster_display": [CLUSTER_DISPLAY_NAMES[name] for name in CLUSTER_ORDER],
        },
        color_discrete_map=CATEGORICAL_COLORS,
        range_x=[low, high],
        range_y=[low, high],
        size_max=24,
        custom_data=[
            "country_name",
            "TIME_PERIOD",
            "metric_value",
            "CISS_z",
            "gap_value",
            "gap_label",
            "coverage_label",
            "cluster_name",
        ],
    )
    fig.update_traces(
        textposition="middle center",
        textfont={"size": 10, "color": "white"},
        marker={"opacity": 0.84, "line": {"width": 1.1, "color": "white"}},
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Period: %{customdata[1]}<br>"
            "SME-FPI: %{customdata[2]:.2f}<br>"
            "CISS z: %{customdata[3]:.2f}<br>"
            "Relative gap: %{customdata[4]:.2f}<br>"
            "%{customdata[5]}<br>"
            "Coverage: %{customdata[6]}<br>"
            "Cluster: %{customdata[7]}<extra></extra>"
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="SME-FPI = CISS",
            line={"color": "#1f2933", "dash": "dot", "width": 1.8},
            hoverinfo="skip",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#b8c2cc")
    fig.add_vline(x=0, line_dash="dot", line_color="#b8c2cc")
    fig = polish(
        fig,
        chart_title(
            "Stress trajectories: countries moving through time",
            "Animated path between market-side CISS and borrower-side SME-FPI",
        ),
        height=540,
        y_title="SME-FPI z-score",
        x_title="CISS z-score",
    )
    fig.update_layout(
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 54, "r": 28, "t": 132, "b": 118},
        transition={"duration": 250},
    )
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 250
        fig.layout.updatemenus[0].buttons[0].args[1]["fromcurrent"] = True
        fig.layout.updatemenus[0].x = 0.0
        fig.layout.updatemenus[0].y = -0.13
    for slider in fig.layout.sliders:
        slider.x = 0.13
        slider.y = -0.13
        slider.len = 0.84
        slider.currentvalue = {"prefix": "Period: ", "font": {"size": 12}}
        slider.pad = {"t": 36, "b": 8}
    return fig


def make_robustness(df, countries):
    countries = selected_or_default(countries)
    subset = df[df["REF_AREA"].isin(countries)].copy()
    if subset.empty:
        return empty_figure("No index-version data are available for the selected countries.")
    version_cols = list(INDEX_OPTIONS.keys())
    grouped = subset.groupby(["TIME_PERIOD", "period_index"], as_index=False)[version_cols].mean()
    melted = grouped.melt(
        id_vars=["TIME_PERIOD", "period_index"],
        value_vars=version_cols,
        var_name="index_version",
        value_name="score",
    )
    melted["index_label"] = melted["index_version"].map(INDEX_OPTIONS)
    melted = melted.sort_values("period_index")
    fig = px.line(
        melted,
        x="TIME_PERIOD",
        y="score",
        color="index_label",
        markers=True,
        category_orders={"TIME_PERIOD": PERIODS},
        color_discrete_sequence=["#1f77b4", "#d19a2e", "#8a63a6", "#2a9d8f"],
    )
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Period: %{x}<br>Average score: %{y:.2f}<extra></extra>"
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#8b98a5")
    fig = polish(
        fig,
        chart_title(
            "Does the story survive alternative weights?",
            "Equal-weight, fixed-baseline, PCA-weighted, and reliability-weighted SME-FPI versions",
        ),
        height=430,
        y_title="Average standardized index score",
        x_title="Half-year period",
    )
    fig.update_layout(
        legend={"orientation": "h", "yanchor": "top", "y": -0.2, "xanchor": "left", "x": 0},
        margin={"l": 54, "r": 28, "t": 132, "b": 108},
    )
    return apply_period_ticks(fig, grouped["TIME_PERIOD"].tolist(), max_ticks=6, angle=0)


def next_half_year_label(period):
    year, half = str(period).split("-S")
    if half == "1":
        return f"{year}-S2"
    return f"{int(year) + 1}-S1"


def make_forecast_dataset(metric_col):
    gap_col = GAP_BY_INDEX.get(metric_col, "Relative_Gap_equal")
    base = FORECAST_PANEL if metric_col in FORECAST_PANEL.columns else PANEL
    df = base.sort_values(["REF_AREA", "period_index"]).copy()
    df["metric_current"] = df[metric_col]
    target_col = f"target_next_{metric_col}"
    if target_col in df.columns:
        df["target_next"] = df[target_col]
    else:
        df["target_next"] = df.groupby("REF_AREA")[metric_col].shift(-1)
    derived = {}
    for source_col, feature_name in [
        (metric_col, "metric"),
        ("CISS_z", "ciss"),
        (gap_col, "gap"),
    ]:
        if source_col in df.columns:
            derived[f"{feature_name}_lag1"] = df.groupby("REF_AREA")[source_col].shift(1)
            derived[f"{feature_name}_lag2"] = df.groupby("REF_AREA")[source_col].shift(2)
    if derived:
        df = pd.concat([df, pd.DataFrame(derived)], axis=1)

    excluded = {
        "target_next",
        "period_sort",
        "metric_current",
        metric_col,
    }
    blocked_prefixes = ("future_", "target_next_")
    feature_cols = ["metric_current"]
    for col in df.columns:
        if col in excluded or col.startswith(blocked_prefixes):
            continue
        if col in {"REF_AREA", "country_name", "TIME_PERIOD", "coverage_label", "cluster_name", "iso3"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    feature_cols = list(dict.fromkeys(feature_cols))
    country_features = pd.get_dummies(df["REF_AREA"], prefix="country")
    features = pd.concat([df[feature_cols], country_features], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    return df, features


def make_forecast_estimator(model_key):
    if model_key == "elastic_net":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            ElasticNetCV(
                alphas=np.logspace(-3, 1, 24),
                l1_ratio=[0.15, 0.5, 0.85],
                cv=3,
                max_iter=20000,
                random_state=42,
            ),
        )
    if model_key == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 3, 25)),
        )
    if model_key == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=1,
            ),
        )
    if model_key == "gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(
                n_estimators=160,
                learning_rate=0.045,
                max_depth=2,
                min_samples_leaf=5,
                subsample=0.82,
                random_state=42,
            ),
        )
    raise ValueError(f"Unknown forecast model: {model_key}")


def simple_forecast_feature_frame(meta, features):
    simple_cols = [
        col
        for col in [
            "metric_current",
            "metric_lag1",
            "metric_lag2",
            "ciss_lag1",
            "ciss_lag2",
            "gap_lag1",
            "gap_lag2",
            "CISS_z",
            "Relative_Gap_equal",
            "components_available",
        ]
        if col in features.columns
    ]
    country_cols = [col for col in features.columns if col.startswith("country_")]
    return features[simple_cols + country_cols].replace([np.inf, -np.inf], np.nan)


def country_time_series_prediction(model_key, history, row, fallback_value):
    history = history.dropna(subset=["target_next"]).sort_values("period_index").copy()
    if len(history) < 8 or history["target_next"].nunique() < 2:
        return fallback_value

    endog = pd.to_numeric(history["target_next"], errors="coerce")
    if endog.isna().all():
        return fallback_value
    endog = endog.interpolate(limit_direction="both").ffill().bfill()
    if endog.isna().any() or endog.nunique() < 2:
        return fallback_value

    exog = None
    future_exog = None
    if model_key == "country_arimax_ciss":
        if "CISS_z" not in history.columns or pd.isna(row.get("CISS_z", np.nan)):
            return fallback_value
        exog_series = pd.to_numeric(history["CISS_z"], errors="coerce")
        if exog_series.notna().sum() < 6:
            return fallback_value
        exog_series = exog_series.interpolate(limit_direction="both").ffill().bfill()
        if exog_series.isna().any():
            return fallback_value
        exog = exog_series.to_frame("CISS_z")
        future_exog = pd.DataFrame({"CISS_z": [float(row["CISS_z"])]})

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StatsmodelsConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FutureWarning)
            model = SARIMAX(
                endog,
                exog=exog,
                order=(1, 0, 0),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=80)
            forecast = fit.forecast(steps=1, exog=future_exog)
    except Exception:
        return fallback_value

    value = float(forecast.iloc[0] if hasattr(forecast, "iloc") else forecast[0])
    return value if np.isfinite(value) else fallback_value


def time_series_predictions(model_key, meta, train_mask, test_mask, y_train):
    test_index = meta.index[test_mask]
    fallback = meta.loc[test_index, "metric_current"].copy().fillna(float(y_train.mean()))
    values = []
    for idx, row in meta.loc[test_index].iterrows():
        country = row["REF_AREA"]
        history_mask = train_mask & meta["REF_AREA"].eq(country)
        history_cols = ["period_index", "target_next"]
        if "CISS_z" in meta.columns:
            history_cols.append("CISS_z")
        history = meta.loc[history_mask, history_cols].copy()
        values.append(country_time_series_prediction(model_key, history, row, float(fallback.loc[idx])))
    return pd.Series(values, index=test_index)


def baseline_predictions(model_key, meta, train_mask, test_mask, y_train, simple_features):
    test_index = meta.index[test_mask]
    fallback = meta.loc[test_index, "metric_current"].copy().fillna(float(y_train.mean()))

    if model_key == "naive":
        pred = fallback
    elif model_key == "country_mean":
        means = y_train.groupby(meta.loc[train_mask, "REF_AREA"]).mean()
        pred = meta.loc[test_index, "REF_AREA"].map(means).fillna(float(y_train.mean()))
    elif model_key == "momentum":
        if "metric_lag1" in meta.columns:
            pred = meta.loc[test_index, "metric_current"] + (
                meta.loc[test_index, "metric_current"] - meta.loc[test_index, "metric_lag1"]
            )
            pred = pred.fillna(fallback)
        else:
            pred = fallback
    elif model_key == "country_ar1":
        train = meta.loc[train_mask, ["REF_AREA", "metric_current", "target_next"]].dropna()
        global_fit = train[["metric_current", "target_next"]].dropna()
        if len(global_fit) >= 4 and global_fit["metric_current"].nunique() > 1:
            global_slope, global_intercept = np.polyfit(global_fit["metric_current"], global_fit["target_next"], 1)
        else:
            global_slope, global_intercept = 1.0, 0.0
        country_models = {}
        for country, subset in train.groupby("REF_AREA"):
            if len(subset) >= 4 and subset["metric_current"].nunique() > 1:
                slope, intercept = np.polyfit(subset["metric_current"], subset["target_next"], 1)
                country_models[country] = (float(slope), float(intercept))
        values = []
        for idx, row in meta.loc[test_index, ["REF_AREA", "metric_current"]].iterrows():
            current = row["metric_current"]
            if pd.isna(current):
                values.append(fallback.loc[idx])
                continue
            slope, intercept = country_models.get(row["REF_AREA"], (global_slope, global_intercept))
            values.append(intercept + slope * current)
        pred = pd.Series(values, index=test_index)
    elif model_key == "pooled_lag_ols":
        X_train = simple_features.loc[train_mask]
        X_test = simple_features.loc[test_mask]
        usable_cols = X_train.columns[X_train.notna().any(axis=0)]
        if len(usable_cols) == 0:
            pred = fallback
        else:
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
            model.fit(X_train[usable_cols], y_train)
            pred = pd.Series(model.predict(X_test[usable_cols]), index=test_index)
    elif model_key in TIME_SERIES_MODEL_KEYS:
        pred = time_series_predictions(model_key, meta, train_mask, test_mask, y_train)
    else:
        raise ValueError(f"Unknown baseline model: {model_key}")

    pred = pd.Series(pred, index=test_index).replace([np.inf, -np.inf], np.nan).fillna(fallback)
    return pred.to_numpy(dtype=float)


def model_predictions_for_origin(end_idx, metric_col, countries=None):
    if metric_col == "SME_FPI_equal_z" and not FORECAST_ALL_PREDICTIONS.empty and end_idx < len(PERIODS):
        origin_period = PERIODS[end_idx]
        precomputed = FORECAST_ALL_PREDICTIONS.copy()
        period_col = "forecast_origin_period" if "forecast_origin_period" in precomputed.columns else "TIME_PERIOD"
        if period_col in precomputed.columns:
            precomputed = precomputed[precomputed[period_col].astype(str).eq(str(origin_period))].copy()
            if countries:
                precomputed = precomputed[precomputed["REF_AREA"].isin(countries)]
            if not precomputed.empty:
                return precomputed

    meta, features = make_forecast_dataset(metric_col)
    train_mask = (meta["period_index"] < end_idx) & meta["target_next"].notna()
    test_mask = meta["period_index"] == end_idx
    if countries:
        test_mask = test_mask & meta["REF_AREA"].isin(countries)
    if train_mask.sum() < 60 or test_mask.sum() == 0:
        return pd.DataFrame()

    latest = meta.loc[test_mask, ["REF_AREA", "country_name", "TIME_PERIOD", "metric_current", "components_available", "coverage_label"]].copy()
    origin_period = str(latest["TIME_PERIOD"].iloc[0])
    target_period = PERIODS[end_idx + 1] if end_idx + 1 < len(PERIODS) else next_half_year_label(origin_period)
    y_train = meta.loc[train_mask, "target_next"]
    simple_features = simple_forecast_feature_frame(meta, features)
    rows = []

    for model_key in BASELINE_MODEL_KEYS:
        pred = baseline_predictions(model_key, meta, train_mask, test_mask, y_train, simple_features)
        out = latest.copy()
        out["forecast_origin_period"] = origin_period
        out["forecast_target_period"] = target_period
        out["model_key"] = model_key
        out["model_label"] = FORECAST_MODEL_LABELS[model_key]
        out["model_family"] = MODEL_FAMILIES[model_key]
        out["predicted_score"] = pred
        out["predicted_delta"] = out["predicted_score"] - out["metric_current"]
        out["residual_scale"] = np.nan
        rows.append(out)

    X_train_full = features.loc[train_mask].copy()
    X_test_full = features.loc[test_mask].copy()
    usable_cols = X_train_full.columns[X_train_full.notna().any(axis=0)]
    X_train_full = X_train_full[usable_cols]
    X_test_full = X_test_full[usable_cols]
    for model_key in ML_MODEL_KEYS:
        model = make_forecast_estimator(model_key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train_full, y_train)
        pred = model.predict(X_test_full)
        train_pred = model.predict(X_train_full)
        residual_scale = float(np.nanstd(y_train.to_numpy(dtype=float) - train_pred))
        if not np.isfinite(residual_scale) or residual_scale < 0.12:
            residual_scale = 0.12
        out = latest.copy()
        out["forecast_origin_period"] = origin_period
        out["forecast_target_period"] = target_period
        out["model_key"] = model_key
        out["model_label"] = FORECAST_MODEL_LABELS[model_key]
        out["model_family"] = MODEL_FAMILIES[model_key]
        out["predicted_score"] = pred
        out["predicted_delta"] = out["predicted_score"] - out["metric_current"]
        out["residual_scale"] = residual_scale
        rows.append(out)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def recent_model_scoreboard():
    if FORECAST_EVALUATION.empty or "model_key" not in FORECAST_EVALUATION.columns:
        return pd.DataFrame()
    recent_periods = (
        FORECAST_EVALUATION["origin_period_sort"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tail(5)
        .tolist()
    )
    if not recent_periods:
        return pd.DataFrame()
    recent = FORECAST_EVALUATION[FORECAST_EVALUATION["origin_period_sort"].isin(recent_periods)].copy()
    if "model_family" not in recent.columns:
        recent["model_family"] = recent["model_key"].map(MODEL_FAMILIES).fillna("model")
    scores = recent.groupby(["model_key", "model_label", "model_family"], as_index=False).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), n_windows=("origin_period", "nunique")
    )
    if scores.empty:
        return scores
    baseline = scores.loc[scores["model_key"] == "naive", "mae"]
    baseline_mae = float(baseline.iloc[0]) if not baseline.empty else np.nan
    strong_baseline = scores[scores["model_key"].isin(BASELINE_MODEL_KEYS)]["mae"]
    strong_baseline_mae = float(strong_baseline.min()) if not strong_baseline.empty else np.nan
    scores["mae_improvement_vs_naive"] = baseline_mae - scores["mae"]
    scores["mae_improvement_vs_strongest_baseline"] = strong_baseline_mae - scores["mae"]
    scores["display_label"] = scores["model_label"]
    scores = scores.sort_values("mae", ascending=True).reset_index(drop=True)
    scores["rank"] = np.arange(1, len(scores) + 1)
    return scores


def best_forecast_model_key():
    if not FORECAST_SUMMARY.empty:
        key = FORECAST_SUMMARY.iloc[0].get("best_recent_model_key", None)
        if isinstance(key, str) and key in ML_MODEL_KEYS:
            return key
    scores = recent_model_scoreboard()
    scores = scores[scores["model_key"].isin(ML_MODEL_KEYS)] if not scores.empty else scores
    if not scores.empty:
        return str(scores.iloc[0]["model_key"])
    return "random_forest"


def fit_forecast_model(meta, features, train_mask, model_key):
    X_train = features.loc[train_mask].copy()
    y_train = meta.loc[train_mask, "target_next"].copy()
    valid = y_train.notna()
    X_train = X_train.loc[valid]
    y_train = y_train.loc[valid]
    if len(y_train) < 60:
        return None, None, None
    usable_cols = X_train.columns[X_train.notna().any(axis=0)]
    X_train = X_train[usable_cols]
    model = make_forecast_estimator(model_key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    residual_scale = float(np.nanstd(y_train.to_numpy(dtype=float) - train_pred))
    if not np.isfinite(residual_scale) or residual_scale < 0.12:
        residual_scale = 0.12
    return model, residual_scale, usable_cols


def predict_with_interval(model, residual_scale, X, columns):
    X = X.reindex(columns=columns)
    point = model.predict(X)
    lower = point - residual_scale
    upper = point + residual_scale
    return point, lower, upper


def make_forecast_figure(period_range, countries, metric_col):
    countries = selected_or_default(countries)
    end_idx = period_range[1]
    meta, features = make_forecast_dataset(metric_col)
    train_mask = (meta["period_index"] < end_idx) & meta["target_next"].notna()
    model_key = best_forecast_model_key()
    model_label = FORECAST_MODEL_LABELS.get(model_key, model_key)
    model, residual_scale, model_columns = fit_forecast_model(meta, features, train_mask, model_key)
    if model is None:
        return empty_figure("Not enough historical country-period observations for a stable ML forecast.")

    latest = meta[(meta["period_index"] == end_idx) & meta["REF_AREA"].isin(countries)].copy()
    if latest.empty:
        return empty_figure("No country observations are available at the selected forecast origin.")
    latest_features = features.loc[latest.index]
    point, lower, upper = predict_with_interval(model, residual_scale, latest_features, model_columns)
    latest["forecast"] = point
    latest["forecast_lower"] = lower
    latest["forecast_upper"] = upper
    latest["direction"] = np.where(latest["forecast"] >= latest[metric_col], "Rising pressure", "Easing pressure")
    latest = latest.sort_values("forecast", ascending=True)

    has_realized_target = end_idx + 1 < len(PERIODS)
    next_period = PERIODS[end_idx + 1] if has_realized_target else next_half_year_label(latest["TIME_PERIOD"].iloc[0])
    target_status = "historical target available" if has_realized_target else "target actual not yet available"

    scores = recent_model_scoreboard()
    mae_label = "Backtest unavailable"
    if not scores.empty:
        selected = scores[scores["model_key"] == model_key]
        naive = scores[scores["model_key"] == "naive"]
        if not selected.empty and not naive.empty:
            mae_label = (
                f"Recent MAE: {model_label} {selected['mae'].iloc[0]:.2f} "
                f"vs naive {naive['mae'].iloc[0]:.2f}"
            )

    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "xy"}]],
        row_heights=[0.62, 0.38],
        vertical_spacing=0.18,
        subplot_titles=("Next-period country forecast", "Recent rolling-origin model comparison"),
    )
    fig.add_trace(
        go.Scatter(
            x=latest[metric_col],
            y=latest["country_name"],
            mode="markers",
            name="Current SME-FPI",
            marker={"color": "#2f6f9f", "size": 9},
            customdata=np.stack([latest["TIME_PERIOD"], latest["coverage_label"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Forecast origin: %{customdata[0]}<br>"
                "Current score: %{x:.2f}<br>"
                "Coverage: %{customdata[1]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=latest["forecast"],
            y=latest["country_name"],
            mode="markers",
            name=f"{model_label} early-warning score for {next_period}",
            marker={"color": "#b23a35", "size": 11, "symbol": "diamond"},
            error_x={
                "type": "data",
                "array": latest["forecast_upper"] - latest["forecast"],
                "arrayminus": latest["forecast"] - latest["forecast_lower"],
                "color": "rgba(178, 58, 53, 0.42)",
                "thickness": 1.6,
            },
            customdata=np.stack(
                [
                    latest["forecast_lower"],
                    latest["forecast_upper"],
                    latest["direction"],
                    latest["components_available"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"Model: {model_label}<br>"
                "Forecast score: %{x:.2f}<br>"
                "Residual interval: %{customdata[0]:.2f} to %{customdata[1]:.2f}<br>"
                "%{customdata[2]}<br>"
                "Components available: %{customdata[3]:.0f} / 6<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    for row in latest.itertuples():
        fig.add_shape(
            type="line",
            x0=getattr(row, metric_col),
            x1=row.forecast,
            y0=row.country_name,
            y1=row.country_name,
            line={"color": "rgba(99, 114, 130, 0.32)", "width": 1},
            row=1,
            col=1,
        )

    if scores.empty:
        fig.add_annotation(
            text="Model leaderboard unavailable. Rebuild forecasting_model_evaluation.csv.",
            x=0.5,
            y=0.5,
            xref="x2 domain",
            yref="y2 domain",
            showarrow=False,
            font={"color": "#53616d"},
        )
    else:
        scores = scores.sort_values("mae", ascending=True)
        colors = [
            "#b23a35" if row.model_key == model_key else "#8b98a5" if row.model_family == "baseline" else "#2f6f9f"
            for row in scores.itertuples()
        ]
        fig.add_trace(
            go.Bar(
                x=scores["mae"],
                y=scores["model_label"],
                orientation="h",
                marker={"color": colors, "line": {"color": "#ffffff", "width": 0.8}},
                name="Recent MAE",
                customdata=np.stack(
                    [
                        scores["rmse"],
                        scores["mae_improvement_vs_naive"],
                        scores["mae_improvement_vs_strongest_baseline"],
                        scores["n_windows"],
                        scores["model_family"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Recent MAE: %{x:.3f}<br>"
                    "Recent RMSE: %{customdata[0]:.3f}<br>"
                    "MAE improvement vs naive: %{customdata[1]:.3f}<br>"
                    "MAE improvement vs strongest baseline: %{customdata[2]:.3f}<br>"
                    "Backtest windows: %{customdata[3]:.0f}<br>"
                    "Family: %{customdata[4]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(autorange="reversed", row=2, col=1)

    fig.add_vline(x=0, line_dash="dot", line_color="#8b98a5", row=1, col=1)
    fig.update_xaxes(title_text="Standardized score", row=1, col=1)
    fig.update_yaxes(title_text=None, row=1, col=1)
    fig.update_xaxes(title_text="Mean absolute error, lower is better", row=2, col=1)
    fig.update_yaxes(title_text=None, row=2, col=1)
    fig = polish(
        fig,
        chart_title(
            "Forecasts are diagnostic early-warning scores, not deterministic outcomes",
            f"{latest['TIME_PERIOD'].iloc[0]} to {next_period}; {target_status}; expanded predictors; {mae_label}",
        ),
        height=760,
    )
    fig.update_layout(margin={"l": 116, "r": 34, "t": 138, "b": 94})
    return fig


def risk_tier(score):
    if score >= 5:
        return "Alert"
    if score >= 3:
        return "Watch"
    if score >= 1.5:
        return "Monitor"
    return "Normal"


def confidence_label(model_agreement, forecast_range, components_available):
    agreement_strength = max(model_agreement, 1 - model_agreement) if pd.notna(model_agreement) else 0
    if components_available >= 5 and agreement_strength >= 0.75 and forecast_range <= 0.55:
        return "High"
    if components_available >= 4 and agreement_strength >= 0.60 and forecast_range <= 0.85:
        return "Medium"
    return "Low"


def monitor_signal_type(row):
    tier = str(getattr(row, "risk_tier", "Normal"))
    current = getattr(row, "current_score", np.nan)
    gap = getattr(row, "gap_value", np.nan)
    delta = getattr(row, "best_model_delta", np.nan)
    agreement = getattr(row, "ml_model_agreement_rising", np.nan)
    hidden_signal = pd.notna(gap) and gap >= 0.35
    visible_level = pd.notna(current) and current >= 0.35
    rising_signal = pd.notna(delta) and delta >= 0.10
    rising_agreement = pd.notna(agreement) and agreement >= 0.75 and pd.notna(delta) and delta >= 0

    if tier in {"Alert", "Watch"}:
        if hidden_signal and (rising_signal or rising_agreement):
            return "High pressure + rising forecast"
        if hidden_signal or visible_level:
            return "Visible or hidden high pressure"
        if rising_signal or rising_agreement:
            return "Forward-looking warning"
        return "Composite warning"
    if tier == "Monitor":
        if hidden_signal and (rising_signal or rising_agreement):
            return "Mixed monitor"
        if hidden_signal or visible_level:
            return "Hidden-gap monitor"
        if rising_signal or rising_agreement:
            return "Rising-forecast monitor"
        return "Single-signal monitor"
    if hidden_signal:
        return "Below-threshold hidden gap"
    if rising_signal or rising_agreement:
        return "Below-threshold rising signal"
    return "Normal"


def list_country_names(frame, limit=3):
    if frame.empty:
        return "none"
    names = frame["country_name"].astype(str).head(limit).tolist()
    suffix = "" if len(frame) <= limit else f" +{len(frame) - limit}"
    return ", ".join(names) + suffix


def decision_driver_text(row, context):
    drivers = []
    for col, label in COMPONENT_DRIVER_LABELS.items():
        if col in row.index and pd.notna(row[col]):
            drivers.append((float(row[col]), label))
    for col, label in PRESSURE_GROUP_LABELS.items():
        if col in row.index and pd.notna(row[col]):
            drivers.append((float(row[col]), label))
    context_labels = {
        "micro_access_finance_severity": "SAFE access severity",
        "micro_problem_pressure_mean": "Broad SAFE problem pressure",
        "bls_credit_standards_sme": "BLS credit standards",
        "bls_terms_conditions_sme": "BLS terms pressure",
        "mir_small_loan_rate": "MIR small-loan rate",
        "mir_rate_spread_small_large": "Small-loan spread",
        "eurostat_bankruptcies_index": "Eurostat bankruptcy index",
        "eurostat_bankruptcy_registration_gap": "Bankruptcy-registration gap",
    }
    for col, label in context_labels.items():
        if col in row.index and col in context.columns and pd.notna(row[col]):
            values = pd.to_numeric(context[col], errors="coerce")
            std = values.std(ddof=0)
            if pd.notna(std) and std > 0:
                score = (float(row[col]) - float(values.mean())) / float(std)
                if score > 0:
                    drivers.append((score, label))
    if pd.notna(row.get("gap_value", np.nan)) and row["gap_value"] >= 0.35:
        drivers.append((float(row["gap_value"]) + 0.5, "Relative SME-CISS gap"))
    if pd.notna(row.get("best_model_delta", np.nan)) and row["best_model_delta"] >= 0.10:
        drivers.append((float(row["best_model_delta"]) + 0.5, "ML forecast rising"))
    if not drivers:
        return "No dominant warning driver; keep monitoring coverage and next SAFE release."
    return "; ".join([label for _, label in sorted(drivers, reverse=True)[:3]])


def build_decision_board(period_range, countries, metric_col):
    end_idx = period_range[1]
    predictions = model_predictions_for_origin(end_idx, metric_col)
    if predictions.empty:
        return pd.DataFrame()
    gap_col = GAP_BY_INDEX.get(metric_col, "Relative_Gap_equal")
    source = FORECAST_PANEL if metric_col in FORECAST_PANEL.columns else PANEL
    latest = source[source["period_index"] == end_idx].copy()
    if latest.empty:
        return pd.DataFrame()
    latest["current_score"] = latest[metric_col]
    latest["gap_value"] = latest[gap_col] if gap_col in latest.columns else np.nan

    ml_predictions = predictions[predictions["model_key"].isin(ML_MODEL_KEYS)].copy()
    ml_summary = (
        ml_predictions.groupby(["REF_AREA", "country_name"], as_index=False)
        .agg(
            ml_median_forecast=("predicted_score", "median"),
            ml_min_forecast=("predicted_score", "min"),
            ml_max_forecast=("predicted_score", "max"),
            ml_model_agreement_rising=("predicted_delta", lambda values: float(np.mean(values > 0))),
            ml_model_count=("model_key", "nunique"),
        )
    )
    best_key = best_forecast_model_key()
    best_cols = [
        "REF_AREA",
        "predicted_score",
        "predicted_delta",
        "model_label",
        "forecast_target_period",
        "actual_next_score",
        "actual_delta",
        "forecast_abs_error",
    ]
    best_cols = [col for col in best_cols if col in predictions.columns]
    best = predictions[predictions["model_key"] == best_key][best_cols].rename(
        columns={
            "predicted_score": "best_model_forecast",
            "predicted_delta": "best_model_delta",
            "model_label": "best_model_label",
        }
    )
    board = latest.merge(ml_summary, on=["REF_AREA", "country_name"], how="left").merge(best, on="REF_AREA", how="left")
    board["forecast_range"] = board["ml_max_forecast"] - board["ml_min_forecast"]
    board["forecast_direction"] = np.where(board["best_model_delta"] >= 0, "Rising", "Easing")
    scores = []
    for row in board.itertuples():
        score = 0.0
        current = getattr(row, "current_score", np.nan)
        gap = getattr(row, "gap_value", np.nan)
        forecast = getattr(row, "best_model_forecast", np.nan)
        delta = getattr(row, "best_model_delta", np.nan)
        agreement = getattr(row, "ml_model_agreement_rising", np.nan)
        if pd.notna(current):
            score += 2 if current >= 1.0 else 1.5 if current >= 0.75 else 1 if current >= 0.35 else 0.5 if current >= 0.20 else 0
        if pd.notna(gap):
            score += 2 if gap >= 1.0 else 1.5 if gap >= 0.75 else 1 if gap >= 0.35 else 0
        if pd.notna(forecast) or pd.notna(delta):
            score += 2 if (forecast >= 1.0 or delta >= 0.30) else 1 if (forecast >= 0.35 or delta >= 0.10) else 0
        if pd.notna(agreement):
            score += 1 if agreement >= 0.75 else 0.5 if agreement >= 0.50 else 0
        scores.append(score)
    board["risk_score"] = scores
    board["risk_tier"] = board["risk_score"].map(risk_tier)
    board["confidence"] = [
        confidence_label(row.ml_model_agreement_rising, row.forecast_range, row.components_available)
        for row in board.itertuples()
    ]
    board["agreement_quality"] = board["confidence"]
    board["signal_type"] = [monitor_signal_type(row) for row in board.itertuples()]
    board["primary_drivers"] = board.apply(lambda row: decision_driver_text(row, board), axis=1)
    board["recommended_read"] = np.select(
        [
            board["risk_tier"].eq("Alert"),
            board["risk_tier"].eq("Watch"),
            board["risk_tier"].eq("Monitor"),
        ],
        [
            "Immediate watchlist: inspect drivers and compare with local SME context.",
            "Watch closely: pressure is visible or forecast to rise.",
            "Monitor: at least one warning signal is present; check its signal type.",
        ],
        default="Normal: no major warning signal under the current rules.",
    )
    tier_order = {"Alert": 0, "Watch": 1, "Monitor": 2, "Normal": 3}
    board["_tier_order"] = board["risk_tier"].map(tier_order)
    board = board.sort_values(["_tier_order", "risk_score", "current_score"], ascending=[True, False, False])
    if countries:
        board = board[board["REF_AREA"].isin(countries)]
    return board.drop(columns=["_tier_order"])


def decision_summary_cards(board):
    if board.empty:
        return [
            metric_card("Monitoring board", "not available", "select a later period with enough training data"),
            metric_card("Watchlist", "n/a", "forecast model unavailable"),
            metric_card("Model agreement", "n/a", "no country diagnostics"),
            metric_card("Main risk", "n/a", "no current signal"),
        ]
    alert_count = int((board["risk_tier"] == "Alert").sum())
    watch_count = int((board["risk_tier"] == "Watch").sum())
    monitor_count = int((board["risk_tier"] == "Monitor").sum())
    top = board.iloc[0]
    rising = board[board["forecast_direction"] == "Rising"]
    high_conf = int((board["agreement_quality"] == "High").sum())
    hidden_monitors = board[board["signal_type"].isin(["Hidden-gap monitor", "Mixed monitor", "Visible or hidden high pressure"])]
    forward_monitors = board[board["signal_type"].isin(["Rising-forecast monitor", "Mixed monitor", "Forward-looking warning"])]
    return [
        metric_card("Current read", f"{alert_count + watch_count + monitor_count}", f"{alert_count} alert, {watch_count} watch, {monitor_count} monitor"),
        metric_card("Hidden-gap monitors", f"{len(hidden_monitors)}", list_country_names(hidden_monitors)),
        metric_card("Rising-forecast monitors", f"{len(forward_monitors)}", list_country_names(forward_monitors)),
        metric_card("Top signal", str(top["country_name"]), f"{top['risk_tier']} score {format_number(top['risk_score'], 1)}"),
        metric_card("High agreement reads", f"{high_conf}", "model agreement quality, not forecast certainty"),
    ]


def decision_narrative(board):
    if board.empty:
        return html.Div(
            className="decision-narrative",
            children=[
                html.Span("Latest read", className="panel-kicker"),
                visual_bullets(["no board for this period", "move the period slider", "or rebuild forecast layer"]),
            ],
        )
    counts = board["risk_tier"].value_counts()
    alert_count = int(counts.get("Alert", 0))
    watch_count = int(counts.get("Watch", 0))
    monitor_count = int(counts.get("Monitor", 0))
    rising_count = int((board["forecast_direction"] == "Rising").sum())
    high_conf = int((board["agreement_quality"] == "High").sum())
    top_names = ", ".join(board.head(3)["country_name"].astype(str).tolist())
    hidden_monitors = board[board["signal_type"].isin(["Hidden-gap monitor", "Mixed monitor", "Visible or hidden high pressure"])]
    forward_monitors = board[board["signal_type"].isin(["Rising-forecast monitor", "Mixed monitor", "Forward-looking warning"])]
    model_label = str(board["best_model_label"].dropna().iloc[0]) if board["best_model_label"].notna().any() else "best recent ML model"
    origin = str(board["TIME_PERIOD"].iloc[0])
    target = (
        str(board["forecast_target_period"].dropna().iloc[0])
        if "forecast_target_period" in board.columns and board["forecast_target_period"].notna().any()
        else "the next half-year"
    )
    actual_note = "The target-period actual score is available for this historical origin."
    if "actual_next_score" not in board.columns or not board["actual_next_score"].notna().any():
        actual_note = f"The {target} actual SME-FPI is not available yet, so this latest read remains an ex-ante monitoring signal."
    return html.Div(
        className="decision-narrative",
        children=[
            html.Span("Latest read", className="panel-kicker"),
            compact_copy_bullets(
                [
                    ("Status", f"{origin}: {alert_count} Alert, {watch_count} Watch, {monitor_count} Monitor."),
                    ("Forecast", f"{model_label} projects {rising_count} countries with rising H+1 pressure toward {target}; {high_conf} reads have high model agreement."),
                    ("Hidden gap", f"Hidden-gap monitors: {list_country_names(hidden_monitors)}. Rising-forecast monitors: {list_country_names(forward_monitors)}."),
                    ("Next read", f"Start with {top_names}; explain level, SME-CISS gap, and forecast momentum with the driver tabs."),
                    ("Actual", actual_note),
                ],
                className="compact-guide-list decision-narrative-list",
                limit=158,
            ),
        ],
    )


def standardization_caveat(metric_col, period_range):
    end_idx = period_range[1]
    period_label = PERIODS[end_idx] if 0 <= end_idx < len(PERIODS) else "selected period"
    latest = PANEL[PANEL["period_index"] == end_idx].copy()
    fixed_names = "not available"
    if not latest.empty and "Relative_Gap_fixed_baseline" in latest.columns:
        fixed_names = list_country_names(latest.sort_values("Relative_Gap_fixed_baseline", ascending=False), 3)

    if metric_col == "SME_FPI_equal_z":
        title = "Retrospective scale is selected"
        body = (
            "Equal-weight SME-FPI uses full-sample z-score standardization, which is strong for comparing the whole "
            "2010s-2020s history but less ideal as a real-time monitoring scale. The fixed-baseline version checks the "
            "same story against the 2015-S1 to 2019-S2 pre-pandemic reference period."
        )
    elif metric_col == "SME_FPI_fixed_baseline_z":
        title = "Fixed-baseline scale is selected"
        body = (
            "This view compares later stress with the 2015-S1 to 2019-S2 reference period. It is easier to defend as a "
            "real-time monitoring scale, but its numbers are not directly the same as the full-sample equal-weight z-score."
        )
    else:
        title = "Robustness scale is selected"
        body = (
            "This view uses an alternative weighting design. It is useful for checking whether the story survives a different "
            "index construction, but the Monitoring Board should still be read as a diagnostic screen rather than a policy rule."
        )

    return html.Div(
        className="standardization-caveat",
        children=[
            html.Div(
                children=[
                    html.Span("Scale caveat", className="panel-kicker"),
                    html.H3(title),
                    compact_copy_bullets([("Scale", body)], className="compact-guide-list scale-caveat-list", limit=150),
                ]
            ),
            html.Div(
                className="scale-compare-list",
                children=[
                    html.Strong(f"{period_label} fixed-baseline gap leaders"),
                    html.Span(fixed_names),
                    html.Small("Cross-check for full-sample z-score hindsight."),
                ],
            ),
        ],
    )


def risk_rule_guide():
    rules = [
        ("Level", "current SME-FPI", "0.20, 0.35, 0.75, 1.00"),
        ("Gap", "SME-FPI minus CISS", "0.35, 0.75, 1.00"),
        ("Forward", "H+1 level or delta", "forecast >= 0.35 or delta >= 0.10"),
        ("Agreement", "ML models rising", "50% or 75% agreement"),
    ]
    tiers = [
        ("Alert", "risk score >= 5", "already high or multi-signal stress"),
        ("Watch", "risk score >= 3", "visible pressure or stronger forecast warning"),
        ("Monitor", "risk score >= 1.5", "one warning signal deserves attention"),
        ("Normal", "risk score < 1.5", "no major warning under current rules"),
    ]
    return html.Div(
        className="risk-rule-guide",
        children=[
            html.Div(
                children=[
                    html.Span("Tier score", className="panel-kicker"),
                    html.H3("The tier is a monitoring label, not a probability"),
                    visual_bullets(["stress-level warning", "direction checked separately", "not a probability"]),
                ],
            ),
            html.Div(
                className="risk-rule-grid",
                children=[
                    html.Div(
                        children=[
                            html.Strong(name),
                            html.Span(signal),
                            html.Small(thresholds),
                        ]
                    )
                    for name, signal, thresholds in rules
                ],
            ),
            html.Div(
                className="tier-threshold-grid",
                children=[
                    html.Div(
                        className=f"tier-threshold tier-threshold-{tier.lower()}",
                        children=[html.Strong(tier), html.Span(rule), html.Small(read)]
                    )
                    for tier, rule, read in tiers
                ],
            ),
        ],
    )


def no_overclaim_panel():
    claims = [
        "causal impact of market stress, macro variables, or bank lending conditions",
        "firm-level default prediction or production credit-risk scoring",
        "country-specific market stress measurement from CISS",
        "automatic policy action without local institutional context",
        "fully real-time forecasting with every external-source publication lag modeled",
    ]
    return html.Div(
        className="no-claim-panel",
        children=[
            html.Div(
                children=[
                    html.Span("Claim boundary", className="panel-kicker"),
                    html.H3("What this dashboard does not claim"),
                    visual_bullets(["monitoring and diagnosis only", "stronger claims deliberately excluded"]),
                ]
            ),
            compact_copy_bullets(claims, className="compact-copy-list no-claim-list", limit=120),
        ],
    )


def component_family_panel():
    return html.Div(
        className="component-family-panel",
        children=[
            html.Div(
                children=[
                    html.Span("Finance translation", className="panel-kicker"),
                    html.H3("The six SAFE inputs are easier to read as three pressure families"),
                    visual_bullets(["formula unchanged", "access pressure", "loan-cost pressure", "finance-salience pressure"]),
                ]
            ),
            html.Div(
                className="component-family-grid",
                children=[
                    concept_card(spec["label"], spec["short"], spec["description"])
                    for spec in PRESSURE_GROUPS.values()
                ],
            ),
        ],
    )


def make_risk_history_figure(period_range, countries):
    if FORECAST_DECISION_HISTORY.empty:
        return empty_figure("Historical decision-board backtest is not available. Rebuild the forecasting layer.")
    countries = selected_or_default(countries)
    history = FORECAST_DECISION_HISTORY.copy()
    if "period_index" not in history.columns:
        period_lookup = {period: index for index, period in enumerate(PERIODS)}
        history["period_index"] = history["TIME_PERIOD"].map(period_lookup)
    history = history[
        (history["period_index"] >= period_range[0])
        & (history["period_index"] <= period_range[1])
        & (history["REF_AREA"].isin(countries))
    ].copy()
    if history.empty:
        return empty_figure("No historical tier observations are available for this selection.")

    tier_score = {"Normal": 0, "Monitor": 1, "Watch": 2, "Alert": 3}
    tier_sort = {"Alert": 0, "Watch": 1, "Monitor": 2, "Normal": 3}
    history["tier_score"] = history["risk_tier"].map(tier_score)
    latest = history.sort_values("period_index").groupby("country_name", as_index=False).tail(1)
    country_order = (
        latest.assign(_tier_order=latest["risk_tier"].map(tier_sort))
        .sort_values(["_tier_order", "risk_score", "country_name"], ascending=[True, False, True])["country_name"]
        .tolist()
    )
    periods = [period for period in PERIODS[period_range[0] : period_range[1] + 1] if period in set(history["TIME_PERIOD"])]
    matrix = (
        history.pivot_table(index="country_name", columns="TIME_PERIOD", values="tier_score", aggfunc="max")
        .reindex(index=country_order, columns=periods)
    )
    text = (
        history.pivot_table(index="country_name", columns="TIME_PERIOD", values="risk_tier", aggfunc="first")
        .reindex(index=country_order, columns=periods)
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=matrix.columns,
            y=matrix.index,
            text=text.fillna("").to_numpy(),
            zmin=0,
            zmax=3,
            colorscale=[
                [0.00, "#8b98a5"],
                [0.24, "#8b98a5"],
                [0.25, "#2f6f9f"],
                [0.49, "#2f6f9f"],
                [0.50, "#d19a2e"],
                [0.74, "#d19a2e"],
                [0.75, "#a4312e"],
                [1.00, "#a4312e"],
            ],
            colorbar={"title": "Tier", "tickvals": [0, 1, 2, 3], "ticktext": ["Normal", "Monitor", "Watch", "Alert"]},
            hovertemplate="<b>%{y}</b><br>%{x}<br>Tier: %{text}<extra></extra>",
        )
    )
    fig = polish(
        fig,
        chart_title(
            "Decision tiers through time",
            "Rolling-origin Alert/Watch/Monitor/Normal assignments",
        ),
        height=470,
        x_title="Forecast origin",
        y_title="Country",
        showlegend=False,
    )
    fig.update_layout(margin={"l": 118, "r": 34, "t": 132, "b": 92})
    return fig


def make_tier_validation_figure():
    if RISK_TIER_VALIDATION.empty:
        return empty_figure("Risk-tier validation is not available. Rebuild the forecasting layer.")
    validation = RISK_TIER_VALIDATION.copy()
    order = ["Alert", "Watch", "Monitor", "Normal"]
    validation["risk_tier"] = pd.Categorical(validation["risk_tier"], categories=order, ordered=True)
    validation = validation.sort_values("risk_tier")
    colors = [RISK_TIER_COLORS.get(str(tier), "#8b98a5") for tier in validation["risk_tier"].astype(str)]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=validation["risk_tier"].astype(str),
            y=validation["pressure_rose_share"] * 100,
            marker={"color": colors, "line": {"color": "#ffffff", "width": 1}},
            name="Pressure rose",
            customdata=np.column_stack(
                [
                    validation["n_country_origins"],
                    validation["mean_current_score"],
                    validation["mean_actual_next_score"],
                    validation["mean_actual_delta"],
                    validation["mean_forecast_abs_error"],
                    validation["direction_hit_share"] * 100,
                ]
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Pressure rose next period: %{y:.0f}%<br>"
                "Country-origin reads: %{customdata[0]:.0f}<br>"
                "Mean current score: %{customdata[1]:.2f}<br>"
                "Mean next score: %{customdata[2]:.2f}<br>"
                "Mean actual delta: %{customdata[3]:.2f}<br>"
                "Mean forecast abs. error: %{customdata[4]:.2f}<br>"
                "Directional hit rate: %{customdata[5]:.0f}%<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=validation["risk_tier"].astype(str),
            y=validation["direction_hit_share"] * 100,
            mode="lines+markers",
            name="Directional hit",
            yaxis="y2",
            line={"color": "#244c68", "width": 2},
            marker={"size": 8, "color": "#ffffff", "line": {"color": "#244c68", "width": 2}},
            hovertemplate="<b>%{x}</b><br>Directional hit rate: %{y:.0f}%<extra></extra>",
        )
    )
    fig = polish(
        fig,
        chart_title(
            "Monitoring tiers separate stress levels, not only acceleration",
            "Alert can ease next period after high stress; read pressure-rise share together with mean next score and directional hit rate",
        ),
        height=470,
        y_title="Pressure rose next period (%)",
        x_title="Decision tier",
    )
    fig.update_layout(
        yaxis2={
            "title": "Directional hit (%)",
            "overlaying": "y",
            "side": "right",
            "range": [0, 100],
            "showgrid": False,
        },
        yaxis={"range": [0, 100]},
        margin={"l": 62, "r": 72, "t": 132, "b": 88},
    )
    return fig


def make_model_rank_heatmap():
    if FORECAST_EVALUATION.empty:
        return empty_figure("Model leaderboard history is unavailable. Rebuild forecasting_model_evaluation.csv.")
    evaluation = FORECAST_EVALUATION.copy()
    required = {"origin_period", "model_label", "mae"}
    if not required.issubset(evaluation.columns):
        return empty_figure("Model evaluation file is missing rank fields.")
    evaluation["rank"] = evaluation.groupby("origin_period_sort")["mae"].rank(method="min", ascending=True)
    model_order = (
        evaluation.groupby("model_label", as_index=False)["mae"]
        .mean()
        .sort_values("mae")["model_label"]
        .tolist()
    )
    period_order = evaluation.sort_values("origin_period_sort")["origin_period"].drop_duplicates().tolist()
    rank_matrix = (
        evaluation.pivot_table(index="model_label", columns="origin_period", values="rank", aggfunc="min")
        .reindex(index=model_order, columns=period_order)
    )
    mae_matrix = (
        evaluation.pivot_table(index="model_label", columns="origin_period", values="mae", aggfunc="mean")
        .reindex(index=model_order, columns=period_order)
    )
    stability = forecast_stability_stats()
    fig = go.Figure(
        data=go.Heatmap(
            z=rank_matrix.to_numpy(dtype=float),
            x=rank_matrix.columns,
            y=rank_matrix.index,
            text=np.round(mae_matrix.to_numpy(dtype=float), 3),
            zmin=1,
            zmax=max(1, len(model_order)),
            reversescale=True,
            colorscale="Blues",
            colorbar={"title": "Rank", "tickmode": "linear", "dtick": 1},
            hovertemplate="<b>%{y}</b><br>Origin: %{x}<br>MAE rank: %{z:.0f}<br>MAE: %{text}<extra></extra>",
        )
    )
    fig = polish(
        fig,
        chart_title(
            "Ridge is useful because it is repeatedly competitive, not because it wins once",
            f"Rolling-origin MAE rank; {stability.get('top_model', 'top model')} wins {stability.get('top_model_wins', 0)} origins and ML family wins {stability.get('ml_family_wins', 0)}",
        ),
        height=480,
        x_title="Forecast origin",
        y_title="Model",
        showlegend=False,
    )
    fig.update_layout(margin={"l": 142, "r": 34, "t": 132, "b": 92})
    return fig


def decision_table(board):
    if board.empty:
        return html.Div(className="source-table-empty", children="Monitoring board is unavailable for this period.")
    rows = []
    for row in board.head(12).itertuples():
        rows.append(
            html.Tr(
                children=[
                    html.Td(str(row.country_name)),
                    html.Td(html.Span(str(row.risk_tier), className=f"risk-badge risk-badge-{str(row.risk_tier).lower()}")),
                    html.Td(str(row.signal_type)),
                    html.Td(format_number(row.current_score)),
                    html.Td(format_number(row.gap_value)),
                    html.Td(format_number(row.best_model_forecast)),
                    html.Td(f"{format_number(row.ml_model_agreement_rising * 100, 0)}%"),
                    html.Td(str(row.agreement_quality)),
                    html.Td(str(row.primary_drivers)),
                ]
            )
        )
    return html.Div(
        className="source-table-wrap decision-table-wrap",
        children=[
            html.Table(
                className="source-table decision-table",
                children=[
                    html.Thead(
                        html.Tr(
                            children=[
                                html.Th("Country"),
                                html.Th("Tier"),
                                html.Th("Signal type"),
                                html.Th("Current"),
                                html.Th("Gap"),
                                html.Th("H+1"),
                                html.Th("ML rising"),
                                html.Th("Agreement quality"),
                                html.Th("Primary drivers"),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
            )
        ],
    )


def make_decision_scatter(board):
    if board.empty:
        return empty_figure("Monitoring board is unavailable for this period.")
    fig = px.scatter(
        board,
        x="gap_value",
        y="current_score",
        size=np.clip(board["risk_score"], 0.8, None),
        color="risk_tier",
        text="REF_AREA",
        category_orders={"risk_tier": ["Alert", "Watch", "Monitor", "Normal"]},
        color_discrete_map=RISK_TIER_COLORS,
        custom_data=[
            "country_name",
            "best_model_forecast",
            "best_model_delta",
            "ml_model_agreement_rising",
            "agreement_quality",
            "signal_type",
            "primary_drivers",
        ],
    )
    fig.update_traces(
        textposition="top center",
        marker={"line": {"color": "#ffffff", "width": 1.2}, "sizemin": 9},
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Current SME-FPI: %{y:.2f}<br>"
            "SME-FPI minus CISS: %{x:.2f}<br>"
            "Best ML H+1: %{customdata[1]:.2f}<br>"
            "Delta: %{customdata[2]:.2f}<br>"
            "ML rising agreement: %{customdata[3]:.0%}<br>"
            "Agreement quality: %{customdata[4]}<br>"
            "Signal type: %{customdata[5]}<br>"
            "Drivers: %{customdata[6]}<extra></extra>"
        ),
    )
    fig.add_hline(y=0.35, line_dash="dot", line_color="#d19a2e")
    fig.add_vline(x=0.35, line_dash="dot", line_color="#d19a2e")
    fig.add_hline(y=1.0, line_dash="dot", line_color="#a4312e")
    fig.add_vline(x=1.0, line_dash="dot", line_color="#a4312e")
    fig = polish(
        fig,
        chart_title(
            "The watchlist separates current pressure, hidden gap, and forward momentum",
            "Monitoring tier combines current SME-FPI, relative SME-CISS gap, H+1 forecast, and ML agreement",
        ),
        height=520,
        y_title="Current SME-FPI z-score",
        x_title="SME-FPI minus CISS z-score",
    )
    fig.update_layout(margin={"l": 62, "r": 28, "t": 132, "b": 92})
    return fig


def make_agreement_figure(board):
    if board.empty:
        return empty_figure("No model-agreement diagnostics are available.")
    focus = board.head(10).sort_values("best_model_forecast")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=focus["current_score"],
            y=focus["country_name"],
            mode="markers",
            name="Current",
            marker={"color": "#2f6f9f", "size": 9},
            hovertemplate="<b>%{y}</b><br>Current SME-FPI: %{x:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=focus["best_model_forecast"],
            y=focus["country_name"],
            mode="markers",
            name="Best ML H+1",
            marker={"color": "#b23a35", "size": 11, "symbol": "diamond"},
            error_x={
                "type": "data",
                "array": focus["ml_max_forecast"] - focus["best_model_forecast"],
                "arrayminus": focus["best_model_forecast"] - focus["ml_min_forecast"],
                "color": "rgba(178,58,53,0.36)",
                "thickness": 1.6,
            },
            customdata=np.stack([focus["forecast_direction"], focus["ml_model_agreement_rising"], focus["agreement_quality"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Best ML H+1: %{x:.2f}<br>"
                "%{customdata[0]} pressure<br>"
                "ML rising agreement: %{customdata[1]:.0%}<br>"
                "Agreement quality: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    for row in focus.itertuples():
        fig.add_shape(
            type="line",
            x0=row.current_score,
            x1=row.best_model_forecast,
            y0=row.country_name,
            y1=row.country_name,
            line={"color": "rgba(99, 114, 130, 0.32)", "width": 1},
        )
    fig.add_vline(x=0, line_dash="dot", line_color="#8b98a5")
    fig = polish(
        fig,
        chart_title(
            "Forecast agreement is shown as a range, not a guarantee",
            "Current score, best ML forecast, and ML min-max range for the highest-risk selected countries",
        ),
        height=500,
        x_title="Standardized SME-FPI score",
    )
    fig.update_layout(margin={"l": 112, "r": 28, "t": 132, "b": 92})
    return fig


def make_driver_heatmap(board):
    if board.empty:
        return empty_figure("No country-driver diagnostics are available.")
    focus = board.head(8).copy()
    driver_cols = [col for col in COMPONENT_DRIVER_LABELS if col in focus.columns]
    value_cols = list(DIAGNOSTIC_DRIVER_COLUMNS.keys()) + driver_cols
    labels = {**DIAGNOSTIC_DRIVER_COLUMNS, **COMPONENT_DRIVER_LABELS}
    matrix = focus[["country_name", *value_cols]].set_index("country_name")
    for col in matrix.columns:
        values = pd.to_numeric(matrix[col], errors="coerce")
        std = values.std(ddof=0)
        matrix[col] = 0 if pd.isna(std) or std == 0 else (values - values.mean()) / std
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=[labels.get(col, col) for col in matrix.columns],
            y=matrix.index,
            colorscale=STRESS_SCALE,
            zmid=0,
            colorbar={"title": "Relative driver"},
            hovertemplate="<b>%{y}</b><br>Driver: %{x}<br>Relative score: %{z:.2f}<extra></extra>",
        )
    )
    fig = polish(
        fig,
        chart_title(
            "What explains the watchlist?",
            "Relative driver heatmap for top selected countries; values are standardized within the displayed group",
        ),
        height=470,
        y_title="Country",
        x_title="Driver",
        showlegend=False,
    )
    fig.update_layout(margin={"l": 110, "r": 34, "t": 132, "b": 138})
    fig.update_xaxes(tickangle=30)
    return fig


def diagnosis_cards(board):
    if board.empty:
        return html.Div(className="source-table-empty", children="No country diagnosis is available for this period.")
    country_error = pd.DataFrame()
    if not FORECAST_COUNTRY_ERROR.empty:
        best_key = best_forecast_model_key()
        country_error = FORECAST_COUNTRY_ERROR[FORECAST_COUNTRY_ERROR["model_key"] == best_key].copy()
        if country_error.empty:
            country_error = (
                FORECAST_COUNTRY_ERROR[FORECAST_COUNTRY_ERROR["model_family"] == "machine learning"]
                .groupby(["REF_AREA", "country_name"], as_index=False)
                .agg(
                    mean_abs_error=("mean_abs_error", "mean"),
                    direction_hit_share=("direction_hit_share", "mean"),
                    n_origins=("n_origins", "max"),
                )
            )
    error_lookup = country_error.set_index("REF_AREA").to_dict("index") if not country_error.empty else {}
    cards = []
    for row in board.head(6).itertuples():
        error_info = error_lookup.get(str(row.REF_AREA), {})
        hist_error = error_info.get("mean_abs_error", np.nan)
        hit_share = error_info.get("direction_hit_share", np.nan)
        cards.append(
            html.Div(
                className=f"diagnosis-card diagnosis-card-{str(row.risk_tier).lower()}",
                children=[
                    html.Div(
                        className="diagnosis-card-head",
                        children=[
                            html.Div(children=[html.Span(str(row.REF_AREA), className="panel-kicker"), html.H3(str(row.country_name))]),
                            html.Span(str(row.risk_tier), className=f"risk-badge risk-badge-{str(row.risk_tier).lower()}"),
                        ],
                    ),
                    html.Div(
                        className="diagnosis-metrics",
                        children=[
                            html.Div(children=[html.Span("Current"), html.Strong(format_number(row.current_score))]),
                            html.Div(children=[html.Span("Gap"), html.Strong(format_number(row.gap_value))]),
                            html.Div(children=[html.Span("H+1"), html.Strong(format_number(row.best_model_forecast))]),
                            html.Div(children=[html.Span("ML rising"), html.Strong(f"{format_number(row.ml_model_agreement_rising * 100, 0)}%")]),
                            html.Div(
                                children=[
                                    html.Span("Hist error"),
                                    html.Strong(format_number(hist_error) if pd.notna(hist_error) else "n/a"),
                                ]
                            ),
                        ],
                    ),
                    html.Small(f"Signal type: {row.signal_type}. Agreement quality: {row.agreement_quality}."),
                    compact_copy_bullets(
                        [("Drivers", str(row.primary_drivers))],
                        className="compact-guide-list diagnosis-driver-list",
                        limit=150,
                    ),
                    html.Small(
                        f"{row.recommended_read} Historical direction hit: "
                        f"{format_number(hit_share * 100, 0)}%."
                        if pd.notna(hit_share)
                        else str(row.recommended_read)
                    ),
                ],
            )
        )
    return html.Div(className="diagnosis-grid", children=cards)


def make_pca_scatter(df, countries):
    countries = selected_or_default(countries)
    subset = add_interpretation(df.dropna(subset=["PC1", "PC2", "cluster_name"]), "SME_FPI_equal_z")
    if subset.empty:
        return empty_figure("No PCA coordinates are available.")
    subset["selected_country"] = np.where(subset["REF_AREA"].isin(countries), "Selected", "Other")
    subset["cluster_display"] = subset["cluster_name"].map(CLUSTER_DISPLAY_NAMES).fillna(subset["cluster_name"])
    fig = px.scatter(
        subset,
        x="PC1",
        y="PC2",
        color="cluster_display",
        opacity=0.72,
        category_orders={"cluster_display": [CLUSTER_DISPLAY_NAMES[name] for name in CLUSTER_ORDER]},
        color_discrete_map=CATEGORICAL_COLORS,
        custom_data=[
            "country_name",
            "TIME_PERIOD",
            "cluster_name",
            "metric_value",
            "gap_value",
            "coverage_label",
        ],
    )
    fig.update_traces(
        marker={"size": 8, "line": {"width": 0.4, "color": "white"}},
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Period: %{customdata[1]}<br>"
            "Cluster: %{customdata[2]}<br>"
            "PC1: %{x:.2f}<br>"
            "PC2: %{y:.2f}<br>"
            "Equal-weight SME-FPI: %{customdata[3]:.2f}<br>"
            "Relative gap: %{customdata[4]:.2f}<br>"
            "Coverage: %{customdata[5]}<extra></extra>"
        ),
    )
    selected = subset[subset["REF_AREA"].isin(countries)] if len(countries) <= FOCUS_COUNTRY_LIMIT else subset.iloc[0:0]
    if not selected.empty:
        fig.add_trace(
            go.Scatter(
                x=selected["PC1"],
                y=selected["PC2"],
                mode="markers",
                name="Selected countries",
                marker={
                    "size": 13,
                    "color": "rgba(0,0,0,0)",
                    "line": {"width": 2.2, "color": "#111827"},
                },
                customdata=np.stack(
                    [
                        selected["country_name"],
                        selected["TIME_PERIOD"],
                        selected["cluster_name"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Period: %{customdata[1]}<br>"
                    "Cluster: %{customdata[2]}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#b8c2cc")
    fig.add_vline(x=0, line_dash="dot", line_color="#b8c2cc")
    fig = polish(
        fig,
        chart_title(
            "Financing-pain regimes in PCA space",
            "Country-period observations grouped by multivariate stress patterns",
        ),
        height=520,
        y_title="PC2",
        x_title="PC1: common financing-pain dimension",
    )
    fig.update_layout(
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0},
        margin={"l": 54, "r": 28, "t": 132, "b": 104},
    )
    return fig


def make_pca_3d_regime(df, countries, z_metric):
    countries = selected_or_default(countries)
    z_metric = z_metric if z_metric in PCA_3D_Z_OPTIONS else "SME_FPI_equal_z"
    required = ["PC1", "PC2", z_metric, "cluster_name"]
    subset = add_interpretation(df.dropna(subset=required), "SME_FPI_equal_z")
    if subset.empty:
        return empty_figure("No PCA coordinates are available for the selected range.")

    subset["cluster_display"] = subset["cluster_name"].map(CLUSTER_DISPLAY_NAMES).fillna(subset["cluster_name"])
    subset["selected_label"] = np.where(subset["REF_AREA"].isin(countries), "selected", "not selected")
    subset["marker_size"] = np.where(subset["REF_AREA"].isin(countries), 4.8, 3.2)

    fig = px.scatter_3d(
        subset,
        x="PC1",
        y="PC2",
        z=z_metric,
        color="cluster_display",
        size="marker_size",
        size_max=7,
        opacity=0.74,
        category_orders={"cluster_display": [CLUSTER_DISPLAY_NAMES[name] for name in CLUSTER_ORDER]},
        color_discrete_map=CATEGORICAL_COLORS,
        custom_data=[
            "country_name",
            "TIME_PERIOD",
            "cluster_name",
            "metric_value",
            "gap_value",
            "CISS_z",
            "coverage_label",
            "selected_label",
        ],
    )
    fig.update_traces(
        marker={"line": {"width": 0.15, "color": "white"}},
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Period: %{customdata[1]}<br>"
            "Cluster: %{customdata[2]}<br>"
            "Selected group: %{customdata[7]}<br>"
            "PC1: %{x:.2f}<br>"
            "PC2: %{y:.2f}<br>"
            f"{PCA_3D_Z_OPTIONS[z_metric]}: %{{z:.2f}}<br>"
            "Equal-weight SME-FPI: %{customdata[3]:.2f}<br>"
            "Relative gap: %{customdata[4]:.2f}<br>"
            "CISS: %{customdata[5]:.2f}<br>"
            "Coverage: %{customdata[6]}<extra></extra>"
        ),
    )
    fig.update_layout(
        height=620,
        margin={"l": 0, "r": 0, "t": 44, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        legend_title_text="",
        scene={
            "xaxis_title": "PC1",
            "yaxis_title": "PC2",
            "zaxis_title": PCA_3D_Z_OPTIONS[z_metric],
            "camera": {"eye": {"x": 1.55, "y": 1.65, "z": 1.05}},
        },
    )
    fig.update_layout(showlegend=True)
    return fig


def make_pca_circle():
    loadings = PCA_LOADINGS.copy()
    loadings["component_label"] = loadings["component"].map(COMPONENT_LABELS)
    loadings["short_label"] = loadings["component"].map(PCA_SHORT_LABELS)
    theta = np.linspace(0, 2 * np.pi, 200)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            name="Unit circle",
            line={"color": "#b8c2cc", "width": 1.5},
            hoverinfo="skip",
        )
    )
    for row in loadings.itertuples():
        fig.add_trace(
            go.Scatter(
                x=[0, row.PC1_loading],
                y=[0, row.PC2_loading],
                mode="lines+markers",
                name=row.component_label,
                line={"width": 2},
                marker={"size": 7},
                hovertemplate=(
                    f"<b>{row.component_label}</b><br>"
                    f"PC1 loading: {row.PC1_loading:.2f}<br>"
                    f"PC2 loading: {row.PC2_loading:.2f}<extra></extra>"
                ),
            )
        )
        xshift, yshift = PCA_LABEL_OFFSETS.get(row.component, (12, 10))
        fig.add_annotation(
            x=row.PC1_loading,
            y=row.PC2_loading,
            text=row.short_label,
            showarrow=False,
            xshift=xshift,
            yshift=yshift,
            font={"size": 11, "color": "#1f2933"},
            bgcolor="rgba(255,255,255,0.72)",
            borderpad=2,
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#d4dce3")
    fig.add_vline(x=0, line_dash="dot", line_color="#d4dce3")
    fig.update_xaxes(range=[-1.05, 1.05], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-1.05, 1.05])
    return polish(
        fig,
        chart_title(
            "What defines the PCA dimensions?",
            "Component loadings reveal broad financing pain versus cost/rate pressure",
        ),
        height=560,
        y_title="PC2 loading",
        x_title="PC1 loading",
        showlegend=False,
    )


def make_validation_chart():
    validation = VALIDATION.copy()
    target_labels = {
        "future_SME_FPI_equal_z_1h": "SME-FPI H+1",
        "future_access_finance_severity_1_10_1h": "Access severity H+1",
        "future_access_finance_top_box_8_10_1h": "Access top-box H+1",
        "future_access_finance_high_pressure_7_10_1h": "Access high pressure H+1",
        "future_macro_stress_context_z_1h": "Macro context H+1",
        "future_CISS_z_1h": "CISS H+1",
        "future_SME_FPI_equal_z_2h": "SME-FPI H+2",
        "future_access_finance_severity_1_10_2h": "Access severity H+2",
        "future_macro_stress_context_z_2h": "Macro context H+2",
    }
    validation["target_label"] = validation["target"].map(target_labels).fillna(
        validation["target"].str.replace("_", " ", regex=False)
    )
    validation["version_label"] = validation["index_version"].map(INDEX_OPTIONS).fillna(validation["index_version"])
    validation = validation[validation["target"].str.contains("future", na=False)]
    fig = px.bar(
        validation,
        x="within_country_pearson_corr",
        y="target_label",
        color="version_label",
        orientation="h",
        barmode="group",
        custom_data=["n", "pearson_corr", "spearman_corr"],
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Target: %{y}<br>"
            "Within-country correlation: %{x:.2f}<br>"
            "Panel n: %{customdata[0]}<br>"
            "Pearson: %{customdata[1]:.2f}<br>"
            "Spearman: %{customdata[2]:.2f}<extra></extra>"
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#8b98a5")
    fig = polish(
        fig,
        chart_title(
            "Does SME-FPI anticipate future stress signals?",
            "Within-country correlations with future survey, macro, and CISS outcomes",
        ),
        height=620,
        y_title="Future validation target",
        x_title="Within-country Pearson correlation",
    )
    fig.update_layout(
        legend={"orientation": "h", "yanchor": "top", "y": -0.14, "xanchor": "left", "x": 0},
        margin={"l": 132, "r": 28, "t": 132, "b": 112},
    )
    return fig


def validation_summary_cards():
    if VALIDATION.empty:
        return [metric_card("Validation", "n/a", "validation_results.csv is missing")]
    validation = VALIDATION.copy()
    default = validation[validation["index_version"] == "SME_FPI_equal_z"].copy()
    external = default[~default["target"].str.contains("future_SME_FPI", na=False)].copy()
    best_external = external.sort_values("within_country_pearson_corr", ascending=False).iloc[0]
    persistence = default[default["target"] == "future_SME_FPI_equal_z_1h"].iloc[0]
    macro = default[default["target"] == "future_macro_stress_context_z_1h"].iloc[0]
    ciss = default[default["target"] == "future_CISS_z_1h"].iloc[0]
    return [
        metric_card("H+1 persistence", format_number(persistence["within_country_pearson_corr"]), f"within-country n={int(persistence['n'])}"),
        metric_card("Best external check", format_number(best_external["within_country_pearson_corr"]), str(best_external["target"]).replace("_", " ")),
        metric_card("Macro context H+1", format_number(macro["within_country_pearson_corr"]), "broad annual macro alignment"),
        metric_card("CISS H+1", format_number(ciss["within_country_pearson_corr"]), "market-stress comparison"),
    ]


def filter_severity(period_range, countries, problem=None, firm_size=None, sector=None):
    countries = selected_or_default(countries)
    df = period_filter(SEVERITY, period_range)
    df = df[df["REF_AREA"].isin(countries)]
    if problem:
        df = df[df["problem"] == problem]
    if firm_size:
        df = df[df["FIRM_SIZE"] == firm_size]
    if sector:
        df = df[df["FIRM_SECTOR"] == sector]
    return df.copy()


def make_big_cube_trend(period_range, countries, problem, firm_size, sector):
    df = filter_severity(period_range, countries, problem, firm_size, sector)
    if df.empty:
        return empty_figure("No SAFE Q0B severity data match the selected filters.")
    grouped = (
        df.groupby(["TIME_PERIOD", "period_index"], as_index=False)[
            ["severity_score_1_10", "top_box_share_8_10", "high_pressure_share_7_10"]
        ]
        .mean()
        .sort_values("period_index")
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=grouped["TIME_PERIOD"],
            y=grouped["top_box_share_8_10"],
            mode="lines+markers",
            name="Top-box share, 8-10",
            line={"color": "#b23a35", "width": 2.4},
            hovertemplate="Top-box share, 8-10<br>Period: %{x}<br>Share: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["TIME_PERIOD"],
            y=grouped["high_pressure_share_7_10"],
            mode="lines+markers",
            name="High-pressure share, 7-10",
            line={"color": "#d19a2e", "width": 2.4},
            hovertemplate="High-pressure share, 7-10<br>Period: %{x}<br>Share: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["TIME_PERIOD"],
            y=grouped["severity_score_1_10"],
            mode="lines+markers",
            name="Severity score, 1-10",
            line={"color": "#2f6f9f", "width": 2.4, "dash": "dot"},
            hovertemplate="Severity score, 1-10<br>Period: %{x}<br>Score: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig = polish(
        fig,
        chart_title(
            f"How severe is '{problem}' over time?",
            "Severity score plus severe-response shares from the 231,231-row SAFE survey-detail table",
        ),
        height=440,
        y_title="Severe response share (%)",
        x_title="Half-year period",
    )
    fig.update_yaxes(title_text="Severity score, 1-10", secondary_y=True, showgrid=False)
    return apply_period_ticks(fig, grouped["TIME_PERIOD"].tolist(), max_ticks=6, angle=0)


def make_problem_bar(period_range, countries, firm_size, sector):
    df = filter_severity(period_range, countries, problem=None, firm_size=firm_size, sector=sector)
    df = df[df["problem"] != "Other"]
    if df.empty:
        return empty_figure("No problem-severity data match the selected filters.")
    grouped = (
        df.groupby("problem", as_index=False)[["severity_score_1_10", "top_box_share_8_10"]]
        .mean()
        .sort_values("severity_score_1_10", ascending=True)
    )
    fig = px.bar(
        grouped,
        x="severity_score_1_10",
        y="problem",
        orientation="h",
        color="top_box_share_8_10",
        color_continuous_scale=STRESS_SCALE,
        custom_data=["top_box_share_8_10"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Average severity: %{x:.2f}<br>"
            "Top-box share 8-10: %{customdata[0]:.2f}%<extra></extra>"
        )
    )
    fig.update_coloraxes(colorbar_title="Top-box %")
    return polish(
        fig,
        chart_title(
            "Which business problems feel most severe?",
            "Average SAFE severity score, excluding the non-specific Other category",
        ),
        height=420,
        y_title=None,
        x_title="Average severity score, 1-10",
        showlegend=False,
    )


def make_size_problem_heatmap(period_range, countries, sector):
    df = filter_severity(period_range, countries, problem=None, firm_size=None, sector=sector)
    df = df[df["problem"] != "Other"]
    if df.empty:
        return empty_figure("No firm-size severity data match the selected filters.")
    grouped = (
        df.groupby(["firm_size_label", "problem"], as_index=False)["high_pressure_share_7_10"]
        .mean()
    )
    pivot = grouped.pivot(index="firm_size_label", columns="problem", values="high_pressure_share_7_10")
    preferred_order = ["All", "SME", "Micro", "Small", "Medium", "Large"]
    pivot = pivot.reindex(index=[item for item in preferred_order if item in pivot.index])
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=STRESS_SCALE,
            colorbar={"title": "High-pressure %"},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Problem: %{x}<br>"
                "High-pressure share 7-10: %{z:.2f}%<extra></extra>"
            ),
        )
    )
    return polish(
        fig,
        chart_title(
            "Which firm sizes report high-pressure problems?",
            "High-pressure response share by firm size and named problem category",
        ),
        height=420,
        y_title="Firm size",
        x_title="Problem category",
        showlegend=False,
    )


def kpi_cards(df):
    latest = latest_period_df(df)
    highest = latest.sort_values("SME_FPI_equal_z", ascending=False).iloc[0]
    gap = latest.sort_values("Relative_Gap_equal", ascending=False).iloc[0]
    return [
        metric_card("Main panel", f"{len(PANEL):,} rows", "country x half-year observations"),
        metric_card("Firm survey detail", f"{BIG_CUBE_ROWS:,} rows", "large grouped SAFE response table"),
        metric_card("Latest period", latest["TIME_PERIOD"].iloc[0], "current dashboard endpoint"),
        metric_card(
            "Highest latest SME-FPI",
            highest["country_name"],
            f"equal-weight z = {format_number(highest['SME_FPI_equal_z'])}",
        ),
        metric_card(
            "Largest latest SME-CISS gap",
            gap["country_name"],
            f"gap = {format_number(gap['Relative_Gap_equal'])}",
        ),
    ]


app = Dash(
    __name__,
    title="SME Financing Pain Observatory",
    assets_folder=str(ASSETS_DIR),
    assets_url_path="/assets",
)
server = app.server

def tab(label, value, children):
    visual_order = {
        "start": 1,
        "decision": 2,
        "findings": 3,
        "overview": 4,
        "forecast": 5,
        "diagnosis": 6,
        "methodology": 7,
        "regimes": 8,
        "hidden-stress": 9,
        "explorer": 10,
        "big-data": 11,
        "data-preview": 12,
    }
    return dcc.Tab(
        label=label,
        value=value,
        className=f"main-tab main-tab-{value}",
        selected_className=f"main-tab main-tab-{value} main-tab--selected",
        style={"order": visual_order.get(value, 99)},
        children=html.Div(className=f"tab-content tab-content-{value}", children=children),
    )


app.layout = html.Div(
    className="app-shell",
    children=[
        html.Header(
            className="hero",
            children=[
                    html.Div(
                        className="hero-copy",
                        children=[
                            html.P("SME credit-stress monitoring dashboard", className="eyebrow"),
                            html.H1("SME Financing Pain Observatory"),
                            visual_bullets(
                                [
                                    "borrower-side SME pain",
                                    "CISS benchmark gap",
                                    "current watchlist first",
                                    "drivers, forecasts, provenance",
                                ],
                                className="visual-bullet-row hero-summary-bullets",
                            ),
                        html.Div(
                            className="hero-insight-grid",
                            children=[
                                html.Div(
                                    className="hero-insight",
                                    children=[
                                        html.Span("Question"),
                                        html.Strong("Can market stress miss SME borrower pain?"),
                                    ],
                                ),
                                html.Div(
                                    className="hero-insight",
                                    children=[
                                        html.Span("Signal"),
                                        html.Strong("SAFE survey pressure minus CISS benchmark"),
                                    ],
                                ),
                                html.Div(
                                    className="hero-insight",
                                    children=[
                                        html.Span("Reader"),
                                        html.Strong("Finance context explained inside each layer"),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="hero-badges",
                            children=[
                                html.Span("386-row panel"),
                                html.Span("231k firm-survey detail"),
                                html.Span("Decision Board"),
                                html.Span("Country diagnosis"),
                                html.Span("CISS benchmark"),
                                html.Span("9-model forecast suite"),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="hero-stack",
                    children=[
                        hero_monitor_snapshot(),
                        html.Div(
                            className="hero-method",
                            children=[
                                html.H2("Core idea"),
                                visual_bullets(["CISS = market stress", "SME-FPI = borrower pressure", "gap = diagnostic mismatch"]),
                            ],
                        ),
                        html.Div(
                            className="hero-preview",
                            children=[
                                html.Img(
                                    src="/assets/signature_sme_fpi_story.png",
                                    alt="Signature overview of the SME Financing Pain Index project",
                                ),
                                html.Div(
                                    className="preview-caption",
                                    children=[
                                        html.Strong("Signal map"),
                                        html.Span("From market stress to borrower-side SME pressure"),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Main(
            children=[
                html.Section(
                    className="section controls-section",
                    children=[
                        html.Details(
                            className="control-deck",
                            children=[
                                html.Summary(
                                    children=[
                                        html.Strong("Dashboard controls"),
                                        html.Span("Open to change countries, index version, or period range"),
                                    ],
                                ),
                                html.Div(
                                    className="control-deck-body",
                                    children=[
                                        html.Div(
                                            className="section-heading compact-heading controls-inner-heading",
                                            children=[
                                                html.H2("Global Controls"),
                                                visual_bullets(["countries", "index version", "period range", "hover for exact values"]),
                                            ],
                                        ),
                                        html.Div(
                                            className="global-control-grid",
                                            children=[
                                                control(
                                                    "Countries",
                                                    html.Details(
                                                        className="country-drawer",
                                                        children=[
                                                            html.Summary("Country selection, all selected by default"),
                                                            html.Div(
                                                                className="country-control",
                                                                children=[
                                                                    html.Div(
                                                                        className="country-actions",
                                                                        children=[
                                                                            html.Button("Select all", id="select-all-countries", n_clicks=0),
                                                                            html.Button("Clear all", id="clear-all-countries", n_clicks=0),
                                                                            html.Span(
                                                                                "Crowded views automatically switch to summary mode so the story stays readable.",
                                                                                className="control-hint",
                                                                            ),
                                                                        ],
                                                                    ),
                                                                    dcc.Checklist(
                                                                        id="country-select",
                                                                        options=COUNTRY_OPTIONS,
                                                                        value=DEFAULT_COUNTRIES,
                                                                        className="country-checklist",
                                                                        inputStyle={"marginRight": "6px"},
                                                                    ),
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                ),
                                                control(
                                                    "Index version",
                                                    dcc.Dropdown(
                                                        id="index-select",
                                                        options=[
                                                            {"label": label, "value": value}
                                                            for value, label in INDEX_OPTIONS.items()
                                                        ],
                                                        value="SME_FPI_equal_z",
                                                        clearable=False,
                                                    ),
                                                ),
                                                control(
                                                    "Period range",
                                                    dcc.RangeSlider(
                                                        id="period-range",
                                                        min=0,
                                                        max=len(PERIODS) - 1,
                                                        value=[0, len(PERIODS) - 1],
                                                        step=1,
                                                        marks=PERIOD_MARKS,
                                                        allowCross=False,
                                                    ),
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="quick-context-strip",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Strong("First read"),
                                                        html.Span("default index + all countries"),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Strong("Then focus"),
                                                        html.Span("select a few countries for exact comparisons"),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Strong("Interpretation rule"),
                                                        html.Span("positive gap = borrower pain above CISS"),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Details(
                                            className="guide-panel compact-details",
                                            children=[
                                                html.Summary("Index version guide"),
                                                html.Div(
                                                    className="guide-panel-copy",
                                                    children=[
                                                        compact_copy_bullets(
                                                            [
                                                                ("Same question", "Only the index standardization or weighting changes."),
                                                                ("Default", "Equal-weight SME-FPI is the clearest first-read baseline."),
                                                                ("Robustness", "Alternative versions test whether the story survives another scale."),
                                                            ],
                                                            className="compact-guide-list",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="index-guide-grid",
                                                    children=[
                                                        concept_card(
                                                            "Equal-weight SME-FPI",
                                                            "default",
                                                            "Simple average of the six standardized SME financing-pain components. Best for the main story because it is transparent.",
                                                        ),
                                                        concept_card(
                                                            "Fixed-baseline SME-FPI",
                                                            "time anchor",
                                                            "Uses 2015-S1 to 2019-S2 as the reference period, so later stress is compared with a pre-pandemic baseline instead of the full sample.",
                                                        ),
                                                        concept_card(
                                                            "PCA-weighted SME-FPI",
                                                            "data-driven",
                                                            "Weights components by their PC1 loadings. Useful for checking whether the result depends on equal weights.",
                                                        ),
                                                        concept_card(
                                                            "Reliability-weighted SME-FPI",
                                                            "robustness",
                                                            "Weights components by how consistently they move with the other components and by data coverage.",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        research_value_panel(),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                interaction_hub(),
                dcc.Tabs(
                    id="main-tabs",
                    value="start",
                    parent_className="tabs-shell",
                    className="main-tabs",
                    children=[
                        tab(
                            "Start Here",
                            "start",
                            [
                                html.Section(
                                    className="section start-section",
                                    children=[
                                        html.Div(
                                            className="section-heading start-heading",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.H2("Start Here: Self-Guided Project Summary"),
                                                        visual_bullets(["project question", "finance terms", "headline results", "claim limits"]),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        professor_submission_panel(),
                                        tab_guide(
                                            "What is this final project trying to show?",
                                            "It builds SME-FPI, a borrower-side financing-pain index from firm survey answers, then checks whether it reveals pressure that a market-stress benchmark can miss.",
                                            "The dashboard is a diagnostic monitoring product. It does not prove causality, predict firm defaults, or prescribe policy action.",
                                        ),
                                        reader_checkpoint(
                                            "What should the professor remember after five minutes?",
                                            "SME-FPI measures what firms report as borrowers; CISS measures financial-market stress. The useful signal is when those two objects diverge.",
                                            "A positive borrower-market gap means SMEs report financing pain above what the market-stress benchmark alone would suggest.",
                                            "Do not read a positive gap as proof of a hidden financial crisis. It is evidence for closer inspection.",
                                        ),
                                        professor_start_cards(),
                                        html.Div(
                                            className="start-subsection",
                                            children=[
                                                html.Span("Headline findings", className="panel-kicker"),
                                                html.H3("What the dashboard concludes"),
                                                latest_findings_cards(),
                                            ],
                                        ),
                                        claim_evidence_matrix(),
                                        html.Div(
                                            className="start-subsection reading-path-subsection",
                                            children=[
                                                html.Span("Five-minute reading path", className="panel-kicker"),
                                                html.H3("Read these tabs in this order"),
                                                visual_bullets(["interactive tabs", "recommended order", "works without live presentation"]),
                                                professor_reading_path(),
                                            ],
                                        ),
                                        html.Div(
                                            className="start-subsection",
                                            children=[
                                                html.Span("Plain-English glossary", className="panel-kicker"),
                                                html.H3("Terms the reader needs before looking at the charts"),
                                                professor_glossary_panel(),
                                            ],
                                        ),
                                        no_overclaim_panel(),
                                    ],
                                ),
                            ],
                        ),
                        tab(
                            "Current Board",
                            "decision",
                            [
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(
                                            className="section-heading",
                                            children=[
                                                html.H2("Current Monitoring Board"),
                                                visual_bullets(["who needs attention", "why flagged", "hidden from CISS?", "H+1 rising?"]),
                                            ],
                                        ),
                                        html.Div(id="command-panel-container"),
                                        defense_question_panel(),
                                        tab_guide(
                                            "Which countries deserve attention first?",
                                            "The board combines current SME-FPI, SME-FPI minus CISS, H+1 forecast direction, and model agreement into a diagnostic tier.",
                                            "Alert, Watch, Monitor, and Normal are monitoring labels, not policy decisions or causal estimates.",
                                        ),
                                        reader_checkpoint(
                                            "What does a monitoring tier mean?",
                                            "It is a triage label that organizes evidence from current SME-FPI, the borrower-market gap, H+1 forecast direction, and model agreement.",
                                            "Use it to decide which countries deserve closer reading, then inspect the driver text and country evidence.",
                                            "Do not treat Alert, Watch, or Monitor as probabilities of crisis or as automatic policy recommendations.",
                                        ),
                                        html.Div(id="standardization-caveat-container"),
                                        html.Div(id="decision-summary-cards", className="metric-grid decision-metric-grid"),
                                        html.Div(id="decision-narrative-container"),
                                        html.Div(
                                            className="decision-logic-panel",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Span("Risk rule", className="panel-kicker"),
                                                        html.H3("Current pain + relative gap + H+1 forecast + model agreement"),
                                                        visual_bullets(["diagnostic tiers", "level + SME-CISS gap", "best H+1 forecast", "model agreement, not certainty"]),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="risk-legend",
                                                    children=[
                                                        html.Span("Alert", className="risk-badge risk-badge-alert"),
                                                        html.Span("Watch", className="risk-badge risk-badge-watch"),
                                                        html.Span("Monitor", className="risk-badge risk-badge-monitor"),
                                                        html.Span("Normal", className="risk-badge risk-badge-normal"),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        risk_rule_guide(),
                                        html.Div(
                                            className="two-col",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        loading_graph("decision-scatter-figure", className="chart chart-glow"),
                                                        explanation(
                                                            "Figure note",
                                                            "The scatter places each country by current borrower-side pain and the relative SME-CISS gap; marker size reflects the diagnostic risk score.",
                                                            "This turns the dashboard from descriptive charts into a triage surface for monitoring.",
                                                            "Countries in the upper-right require the most careful reading because borrower pain is high and market stress may not fully show it.",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        loading_graph("agreement-figure", className="chart"),
                                                        explanation(
                                                            "Figure note",
                                                            "The dumbbell compares current SME-FPI with the best recent ML H+1 forecast, while the error bar shows disagreement across ML models.",
                                                            "This keeps uncertainty visible instead of presenting the forecast as a single deterministic value.",
                                                            "A tight range with high directional agreement is more credible than a lone point estimate.",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="two-col decision-validation-grid",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        loading_graph("risk-history-figure", className="chart"),
                                                        explanation(
                                                            "Validation note",
                                                            "This heatmap replays the Monitoring Board through historical forecast origins.",
                                                            "It shows whether countries repeatedly appeared as Alert, Watch, or Monitor instead of only reporting the latest status.",
                                                            "Persistent warm cells are more meaningful than one isolated warning cell.",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        loading_graph("tier-validation-figure", className="chart"),
                                                        explanation(
                                                            "Validation note",
                                                            "The tier validation chart compares diagnostic tiers with realized next-period movement.",
                                                            "This avoids treating the monitoring tier as a black box: a good monitoring tier should separate high-pressure states while making forecast uncertainty visible.",
                                                            "If a high tier often eases next period, interpret it as a stress-level warning rather than a guaranteed acceleration signal.",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="decision-table-panel",
                                            children=[
                                                html.Div(
                                                    className="inline-heading",
                                                    children=[
                                                        html.Div(
                                                            children=[
                                                                html.Span("Watchlist table", className="panel-kicker"),
                                                                html.H3("Country-level diagnosis"),
                                                            ]
                                                        ),
                                                        visual_bullets(["tier sorted", "risk score sorted", "driver column points to next tab"]),
                                                    ],
                                                ),
                                                html.Div(id="decision-table-container"),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        tab(
                            "Defense & Findings",
                            "findings",
                            [
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(
                                            className="section-heading",
                                            children=[
                                                html.H2("Defense & Findings"),
                                                visual_bullets(["evaluation answers", "supported claims", "explicit limits"]),
                                            ],
                                        ),
                                        defense_question_panel(),
                                        tab_guide(
                                            "What did we learn from the dashboard?",
                                            "SME-FPI is most useful as a borrower-side monitoring signal that can diverge from market stress and can be checked against forecasts and historical tiers.",
                                            "The evidence is descriptive and diagnostic. It should not be read as a causal policy evaluation.",
                                        ),
                                        latest_findings_cards(),
                                        claim_evidence_matrix(),
                                        research_value_panel(),
                                        html.Div(
                                            className="claim-boundary-panel",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Span("Claim boundary", className="panel-kicker"),
                                                        html.H3("Strong project claim"),
                                                        visual_bullets(["borrower signal differs from CISS", "organized by country", "gap, forecast, validation views"]),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Span("Do not overclaim", className="panel-kicker"),
                                                        html.H3("What the dashboard does not prove"),
                                                        visual_bullets(["no CISS causality", "no deterministic forecast", "no automatic policy response"]),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        tab(
                            "Index Basics",
                            "overview",
                            [
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(
                                            className="section-heading",
                                            children=[
                                                html.H2("SME-FPI Index"),
                                                visual_bullets(["borrower-side index", "market-side benchmark", "early-warning check", "external predictors stay separate"]),
                                            ],
                                        ),
                                        tab_guide(
                                            "What is the project about in one sentence?",
                                            "It builds a borrower-side SME financing pain signal and compares it with market stress, then uses validation and forecasting to judge whether the signal is useful.",
                                            "The core index stays borrower-side; macro, lender-side, loan-rate, and business-demography data are predictors or checks, not index ingredients.",
                                        ),
                                        reader_checkpoint(
                                            "What exactly is being measured?",
                                            "SME-FPI measures financing pain reported by firms, using six SAFE survey variables about access, rejection, cost, loan amount, bank willingness, and interest rates.",
                                            "This makes the index a borrower-side measure, which is why it can differ from market stress indicators.",
                                            "External macro, BLS, MIR, and Eurostat variables are not part of the core index; they support forecasting and validation.",
                                        ),
                                        html.Div(
                                            className="professor-brief",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Span("Reader lens", className="panel-kicker"),
                                                        html.H3("For a data-visualization reader who is new to SME finance"),
                                                        visual_bullets(["plain-English finance objects", "one argument per visual layer", "monitoring with caveats"]),
                                                    ],
                                                ),
                                                html.Ul(
                                                    children=[
                                                        html.Li("SME-FPI is the borrower-side signal built from ECB SAFE survey answers."),
                                                        html.Li("CISS is the market-side benchmark, useful as a comparison but not an SME survey."),
                                                        html.Li("Forecasts are early-warning diagnostics, not causal claims or policy decisions."),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        component_family_panel(),
                                        html.Div(
                                            className="overview-brief",
                                            children=[
                                                html.Div(
                                                    className="overview-thesis",
                                                    children=[
                                                        html.Span("One-sentence thesis", className="panel-kicker"),
                                                        html.H3("SME-FPI asks whether small firms feel financing pressure that market stress does not fully show."),
                                                        visual_bullets(["CISS = market-side", "SME-FPI = borrower-side", "SAFE answers build the index"]),
                                                        html.Div(
                                                            className="signal-legend",
                                                            children=[
                                                                html.Span("Market-side: CISS"),
                                                                html.Span("Borrower-side: SME-FPI"),
                                                                html.Span("Diagnostic gap: SME-FPI - CISS"),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="overview-reader-map",
                                                    children=[
                                                        html.Span("How to read the dashboard", className="panel-kicker"),
                                                        read_step("1", "Start with the summary", "Start Here gives the professor the full project logic before the technical tabs."),
                                                        read_step("2", "Read the current board", "Current Board turns the latest data into a plain-language monitoring list."),
                                                        read_step("3", "Check the defense", "Defense & Findings answers the grading questions before technical detail."),
                                                        read_step("4", "Then drill down", "Forecast, Country Diagnosis, Borrower-Market Gap, and Data & Method provide the evidence."),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="onboarding-panel",
                                            children=[
                                                html.Div(
                                                    className="onboarding-copy",
                                                    children=[
                                                        html.Span("First-read data map", className="panel-kicker"),
                                                        html.H3("The index stays borrower-side; the forecast uses a wider evidence stack."),
                                                        visual_bullets(["SAFE builds core index", "macro and lending data predict", "future outcomes validate"]),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="pipeline-grid",
                                                    children=[
                                                        pipeline_step(
                                                            "01",
                                                            "Core index",
                                                            "ECB SAFE",
                                                            "Six borrower-side financing-pain variables define SME-FPI.",
                                                        ),
                                                        pipeline_step(
                                                            "02",
                                                            "Context",
                                                            "macro + market",
                                                            "World Bank macro indicators and ECB CISS explain the environment.",
                                                        ),
                                                        pipeline_step(
                                                            "03",
                                                            "Vulnerability",
                                                            "firm survey detail",
                                                            "Firm size, sector, and problem-severity slices explain who is exposed.",
                                                        ),
                                                        pipeline_step(
                                                            "04",
                                                            "Early warning",
                                                            "BLS + MIR + Eurostat",
                                                            "Bank lending survey, loan-rate, and business-demography data feed the forecast layer.",
                                                        ),
                                                        pipeline_step(
                                                            "05",
                                                            "Validation",
                                                            "future outcomes",
                                                            "Future SME-FPI, access severity, macro stress, and CISS check signal quality.",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(id="kpi-row", className="metric-grid"),
                                        html.Div(
                                            className="storyline-panel",
                                            children=[
                                                html.Div(
                                                    className="storyline-head",
                                                    children=[
                                                        html.Span("Analytical storyline", className="panel-kicker"),
                                                        html.H3("From problem to practical diagnostic"),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="journey-board",
                                                    children=[
                                                        journey_card(
                                                            "01",
                                                            "Problem",
                                                            "Market indicators may miss SME borrower pressure.",
                                                        ),
                                                        journey_card(
                                                            "02",
                                                            "Hypothesis",
                                                            "SAFE survey answers contain a separate signal.",
                                                        ),
                                                        journey_card(
                                                            "03",
                                                            "Model",
                                                            "Six borrower-side components form SME-FPI.",
                                                        ),
                                                        journey_card(
                                                            "04",
                                                            "Validation & Forecast",
                                                            "Weights, PCA, clusters, future outcomes, and H+1 forecasting test robustness.",
                                                        ),
                                                        journey_card(
                                                            "05",
                                                            "Insight",
                                                            "Large positive gaps flag hidden SME financing stress.",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="overview-context-strip",
                                            children=[
                                                term_item(
                                                    "SAFE",
                                                    "data source",
                                                    "ECB firm survey used to measure reported financing conditions.",
                                                ),
                                                term_item(
                                                    "SME-FPI",
                                                    "this index",
                                                    "Higher values mean SMEs report more financing pain than average.",
                                                ),
                                                term_item(
                                                    "CISS",
                                                    "benchmark",
                                                    "ECB market-stress thermometer; useful comparison, not an SME survey.",
                                                ),
                                                term_item(
                                                    "SME-CISS gap",
                                                    "diagnostic",
                                                    "Positive gap means SME pain is high relative to market stress.",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="equation-panel",
                                            children=[
                                                html.Div(
                                                    className="equation-copy",
                                                    children=[
                                                        html.Span("Model definition", className="panel-kicker"),
                                                        html.H3("SME-FPI in plain English"),
                                                        visual_bullets(["standardize components", "average available inputs", "standardize final index", "at least four components"]),
                                                        dcc.Markdown(
                                                            r"""
$$
\mathrm{SME\text{-}FPI}_{c,t}
=
z\!\left(
\frac{1}{|A_{c,t}|}
\sum_{j \in A_{c,t}} z(x_{j,c,t})
\right),
\qquad |A_{c,t}| \ge 4
$$
""",
                                                            mathjax=True,
                                                            className="formula-card",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="variable-panel",
                                                    children=[
                                                        html.H3("What the symbols mean"),
                                                        variable_item("$c$", "Country", "The country being measured, such as Greece, Ireland, or Germany."),
                                                        variable_item("$t$", "Half-year period", "The time point, for example 2025-S1."),
                                                        variable_item("$A_{c,t}$", "Available components", "The set of SME financing-pain variables available for country c at time t."),
                                                        variable_item("$j$", "Component index", "One variable inside that available set, such as rejection, high cost, or limited amount."),
                                                        variable_item("$x_{j,c,t}$", "Raw component value", "The observed value of component j for country c at time t before standardization."),
                                                        variable_item("$z(\\cdot)$", "Z-score standardization", "Transforms values so 0 is average and positive values mean above-average financing pain."),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="story-grid story-grid-spaced overview-footnotes",
                                            children=[
                                                html.Div(
                                                    className="method-block",
                                                    children=[
                                                        html.H3("Dataset shape"),
                                                        compact_copy_bullets(
                                                            [
                                                                ("Core panel", "386 country-half-year observations."),
                                                                ("Survey detail", "231,231 grouped SAFE response cells by country, period, firm size, sector, problem, and response level."),
                                                            ],
                                                            className="compact-guide-list method-block-list",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="method-block",
                                                    children=[
                                                        html.H3("Design stance"),
                                                        compact_copy_bullets(
                                                            [
                                                                ("Separated views", "Index, trends, gaps, PCA, survey detail, and method each keep one job."),
                                                                ("Workflow", "See status, drill into a pattern, then inspect the method."),
                                                            ],
                                                            className="compact-guide-list method-block-list",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                )
                            ],
                        ),
                        tab(
                            "Trend Explorer",
                            "explorer",
                            [
                                html.Section(
                                    className="section section-compact",
                                    children=[
                                        tab_guide(
                                            "Do SME-FPI and CISS move together over time?",
                                            "The Trend Explorer separates temporal movement, robustness across weighting choices, and component-level country-period structure.",
                                            "The shaded distribution bands are descriptive interquartile ranges, not statistical confidence intervals.",
                                        ),
                                    ],
                                ),
                                html.Section(
                                    className="section two-col",
                                    children=[
                                        html.Div(
                                            children=[
                                                loading_graph("time-series-figure", className="chart"),
                                                explanation(
                                                    "Figure note",
                                                    "The line chart compares selected countries' SME-FPI with the common CISS benchmark. When many countries are selected, it switches to a median line, middle-50% band, and latest high-stress highlights.",
                                                    "This avoids a spaghetti chart while still showing temporal change, cross-country spread, and the benchmark comparison. The shaded band is an interquartile dispersion band, not a statistical confidence interval.",
                                                    "A formal confidence interval is not added here because the panel is a small country-period analytical panel rather than a random sample with a clear sampling-error model.",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            children=[
                                                loading_graph("robustness-figure", className="chart"),
                                                explanation(
                                                    "Figure note",
                                                    "This chart compares equal-weight, fixed-baseline, PCA-weighted, and reliability-weighted versions.",
                                                    "A multi-line design makes sensitivity transparent without claiming a false confidence interval.",
                                                    "If the lines move together, the main story is less dependent on one arbitrary weighting choice.",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(
                                            className="section-heading",
                                            children=[
                                                html.H2("Animated Stress Motion"),
                                                visual_bullets(["country-half-year time-lapse", "CISS x-axis", "SME-FPI y-axis", "movement between regimes"]),
                                            ],
                                        ),
                                        loading_graph("animated-stress-figure", className="chart"),
                                        explanation(
                                            "Figure note",
                                            "Each frame is one half-year period. Countries above the diagonal have borrower-side SME financing pain above the CISS benchmark.",
                                            "Animation is useful because the analytical question is dynamic: countries do not only differ in level, they move between stress regimes over time.",
                                            "Use this as an exploratory motion view; the static line chart and heatmap remain better for precise comparison.",
                                        ),
                                    ],
                                ),
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(className="section-heading", children=[html.H2("Country-Period Structure")]),
                                        loading_graph("heatmap-figure", className="chart"),
                                        explanation(
                                            "Figure note",
                                            "The heatmap displays SME-FPI values across countries and half-year periods.",
                                            "Heatmaps are efficient for matrix-like comparisons where both rows and columns matter.",
                                            "Long red bands suggest persistent borrower-side pressure, while isolated red cells suggest shorter stress episodes.",
                                        ),
                                        loading_graph("component-heatmap-figure", className="chart"),
                                        explanation(
                                            "Figure note",
                                            "This component heatmap shows which standardized variables drive the selected countries' index levels.",
                                            "A heatmap keeps the six-component structure visible instead of hiding the index as a black box.",
                                            "It helps distinguish interest-rate pressure from bank rejection, limited amount, or willingness-to-lend pressure.",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        tab(
                            "Forecast & Validation",
                            "forecast",
                            [
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(
                                            className="section-heading forecast-heading",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.H2("Forecast Check"),
                                                        visual_bullets(["H+1 early-warning test", "regularized + tree ML", "external predictors", "strong baseline check"]),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="forecast-badges",
                                                    children=[
                                                        html.Span("Elastic Net"),
                                                        html.Span("Ridge"),
                                                        html.Span("Random Forest"),
                                                        html.Span("Gradient Boosting"),
                                                        html.Span("ARIMA/ARIMAX"),
                                                        html.Span("H+1 forecast"),
                                                        html.Span("strong baseline check"),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        tab_guide(
                                            "Can current information forecast next half-year SME-FPI?",
                                            "Ridge is the current best recent ML model, but its edge over the strongest benchmark is small; the cards and charts keep that comparison visible.",
                                            "This is a pseudo-real-time rolling-origin design: future values are blocked, but release-specific publication lags are not fully modeled. ARIMA/ARIMAX are benchmarks; GARCH and seasonal SARIMA are not headline models because the data are semiannual and short by country.",
                                        ),
                                        reader_checkpoint(
                                            "What question should the forecast answer?",
                                            "Whether current information gives any early-warning value for next half-year SME-FPI beyond simple persistence and compact time-series baselines.",
                                            "The strongest evidence is not the model name; it is the out-of-sample MAE/RMSE and whether ML repeatedly beats the strongest benchmark.",
                                            "Because recent ML gains are modest, the forecast should support the diagnosis rather than replace the descriptive evidence.",
                                        ),
                                        forecast_audit_card(),
                                        forecast_defense_panel(),
                                        html.Div(className="metric-grid forecast-metric-grid", children=forecasting_summary_cards()),
                                        html.Div(
                                            className="forecast-stack-panel",
                                            children=[
                                                html.Div(
                                                    className="forecast-stack-copy",
                                                    children=[
                                                        html.Span("Predictor stack", className="panel-kicker"),
                                                        html.H3("What the model is allowed to know at the forecast origin"),
                                                        visual_bullets(["current + lagged info only", "future columns blocked", "naive / AR(1) / ARIMA baselines", "publication-lag caveat visible"]),
                                                    ],
                                                ),
                                                forecast_feature_list(),
                                            ],
                                        ),
                                        html.Div(
                                            className="forecast-caveat",
                                            children=[
                                                html.Strong("Why this model suite?"),
                                                visual_bullets(["386 observations", "compact linear models", "tree-based nonlinear check", "no production credit-risk claim"]),
                                            ],
                                        ),
                                        forecast_view_selector(),
                                        html.Div(
                                            id="forecast-view-benchmark",
                                            className="forecast-view-panel forecast-view-panel-benchmark",
                                            children=[
                                                loading_graph("baseline-dominance-figure", figure=make_baseline_dominance_figure(), className="chart"),
                                                explanation(
                                                    "Benchmark note",
                                                    "This bar chart asks whether the best ML model beats the strongest non-ML benchmark in each rolling-origin window.",
                                                    "It prevents the dashboard from over-selling ML when persistence, pooled OLS, or ARIMA-style benchmarks are competitive.",
                                                    "Positive bars are useful evidence for ML; negative or small bars mean the forecast should be read mainly as a diagnostic consistency check.",
                                                ),
                                                loading_graph("model-rank-figure", className="chart"),
                                                explanation(
                                                    "Model stability note",
                                                    "The rank heatmap checks whether a model performs consistently across rolling-origin windows.",
                                                    "This matters because the best recent model should not be trusted only because it won one period by chance.",
                                                    "Stable regularized models are usually preferable in this small country-period panel unless a nonlinear model repeatedly beats the simple baselines.",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="forecast-view-uncertainty",
                                            className="forecast-view-panel forecast-view-panel-uncertainty",
                                            style={"display": "none"},
                                            children=[
                                                loading_graph("forecast-loss-figure", className="chart"),
                                                explanation(
                                                    "Loss note",
                                                    "This is the forecast-error view: it reports rolling-origin out-of-sample MAE and RMSE, not training loss.",
                                                    "MAE is the clearest main loss metric here because SME-FPI is standardized; one MAE point means one standardized-index point of average miss.",
                                                    "The country bars show where the best current model has historically been less reliable, so the forecast can be trusted more cautiously for those countries.",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            id="forecast-view-point",
                                            className="forecast-view-panel forecast-view-panel-point",
                                            style={"display": "none"},
                                            children=[
                                                loading_graph("forecast-figure", className="chart chart-glow"),
                                                explanation(
                                                    "Figure note",
                                                    "The top panel forecasts next half-year SME-FPI from the selected period using the best recent ML model. The bottom panel compares Elastic Net, Ridge, Random Forest, Gradient Boosting, simple baselines, and ARIMA/ARIMAX benchmarks.",
                                                    "A country-level dumbbell plus model leaderboard is clearer than a dense projection because the data are a small country-period panel.",
                                                    "Use the forecast to flag countries for attention, then use the historical and component charts to explain why.",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        tab(
                            "Country Diagnosis",
                            "diagnosis",
                            [
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(
                                            className="section-heading",
                                            children=[
                                                html.H2("Country Diagnosis"),
                                                visual_bullets(["strongest current signals", "tier + current level", "gap + forecast direction", "agreement + drivers"]),
                                            ],
                                        ),
                                        tab_guide(
                                            "Why is a country flagged?",
                                            "Country cards bundle the visible evidence: current level, relative SME-CISS gap, forecast direction, model agreement, and driver variables.",
                                            "A country can be important even without an Alert tier if the relative gap or forecast momentum is moving in the wrong direction.",
                                        ),
                                        html.Div(id="diagnosis-card-board"),
                                        html.Div(
                                            className="two-col",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        loading_graph("driver-heatmap-figure", className="chart"),
                                                        explanation(
                                                            "Figure note",
                                                            "The heatmap standardizes drivers within the displayed countries, so color shows which factors are relatively high in that watchlist group.",
                                                            "This turns the index from a single score into a country explanation layer.",
                                                            "Use it to separate borrower-side survey pressure from relative gap and forecast momentum.",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="diagnosis-method-panel",
                                                    children=[
                                                        html.Span("How to use this view", className="panel-kicker"),
                                                        html.H3("Read country cards as evidence bundles"),
                                                        visual_bullets(["not one metric", "level + gap + forecast + agreement", "drivers point to evidence tabs"]),
                                                        html.Div(
                                                            className="diagnosis-rule-list",
                                                            children=[
                                                                html.Div(children=[html.Strong("Level"), html.Span("current borrower-side SME-FPI")]),
                                                                html.Div(children=[html.Strong("Hidden"), html.Span("SME-FPI minus CISS gap")]),
                                                                html.Div(children=[html.Strong("Forward"), html.Span("best recent ML H+1 forecast")]),
                                                                html.Div(children=[html.Strong("Agreement"), html.Span("share of ML models pointing upward")]),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        tab(
                            "Borrower-Market Gap",
                            "hidden-stress",
                            [
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(
                                            className="section-heading",
                                            children=[
                                                html.H2("Geographic Snapshot"),
                                                visual_bullets(["latest selected period", "country comparison", "equal-size tiles"]),
                                            ],
                                        ),
                                        tab_guide(
                                            "Where is borrower-side stress above the market-stress benchmark?",
                                            "Positive SME-CISS gaps show countries where borrower-reported financing pain is above the common market-stress benchmark.",
                                            "CISS is euro-area market stress, not a country-specific SME credit survey, so the gap is a diagnostic comparison rather than a measurement error.",
                                        ),
                                        reader_checkpoint(
                                            "How should I read the gap?",
                                            "A positive gap means SME-FPI is high relative to CISS; the borrower survey is showing more pressure than the market-stress benchmark.",
                                            "This is the central diagnostic comparison in the project because it links the borrower-side and market-side measurement objects.",
                                            "The gap does not prove that CISS is wrong or that a hidden crisis exists; it flags a mismatch worth explaining.",
                                        ),
                                        loading_graph("map-figure", className="chart"),
                                        explanation(
                                            "Figure note",
                                            "The figure combines a muted geographic map with an equal-size country tile view for the latest period inside the chosen range.",
                                            "The choropleth gives geographic context, but the tile panel avoids the area-bias problem where large countries look more important simply because they occupy more screen space.",
                                            "Use the map for location and the equal-size tiles for fair country-to-country comparison. Very pale countries are geographic context, not part of the selected analytical panel.",
                                        ),
                                    ],
                                ),
                                html.Section(
                                    className="section two-col",
                                    children=[
                                        html.Div(
                                            children=[
                                                loading_graph("gap-ranking-figure", className="chart"),
                                                explanation(
                                                    "Figure note",
                                                    "The ranking shows where SME-FPI is high relative to the common euro-area CISS benchmark.",
                                                    "A sorted bar chart is the clearest design for comparing countries on one gap metric.",
                                                    "Positive values indicate borrower-side SME pain above the market-stress benchmark.",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            children=[
                                                loading_graph("bubble-gap-figure", className="chart"),
                                                explanation(
                                                    "Figure note",
                                                    "The bubble chart places market stress on the x-axis and borrower-side SME pain on the y-axis. With many countries selected, it uses the latest period only.",
                                                    "This keeps the diagnostic readable and makes the diagonal disagreement pattern easy to inspect.",
                                                    "Points above the diagonal are countries where SME financing pain exceeds the CISS benchmark.",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        tab(
                            "PCA Analysis",
                            "regimes",
                            [
                                html.Section(
                                    className="section section-compact",
                                    children=[
                                        pca_answer_panel(),
                                        tab_guide(
                                            "Does PCA support the structure of the SME-FPI index?",
                                            "PCA checks whether borrower-side SAFE components move together and whether the main index story is robust to data-driven weighting.",
                                            "PCA groups are descriptive patterns, not causal classes or supervised predictions.",
                                        ),
                                    ],
                                ),
                                html.Section(
                                    className="section two-col",
                                    children=[
                                        html.Div(
                                            children=[
                                                loading_graph("pca-scatter-figure", className="chart"),
                                                explanation(
                                                    "Figure note",
                                                    "The PCA scatter groups country-period observations into descriptive financing-pain patterns.",
                                                    "A scatterplot is suitable because PCA produces two continuous dimensions for multivariate structure.",
                                                    "Distance means similarity in component structure, not causality or an expert-labeled class.",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            children=[
                                                loading_graph("pca-circle-figure", figure=make_pca_circle(), className="chart"),
                                                explanation(
                                                    "Figure note",
                                                    "The correlation circle shows how each SME-FPI component loads on PC1 and PC2.",
                                                    "This is a standard multidimensional analysis diagnostic for checking whether variables align with a common dimension.",
                                                    "Components pointing in a similar direction support the idea of a shared borrower-side financing-pain signal.",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(
                                            className="section-heading inline-heading",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.H2("3D PCA Analysis Explorer"),
                                                        visual_bullets(["exploratory rotation", "PC1 + PC2 + stress metric", "2D views remain primary"]),
                                                    ]
                                                ),
                                                html.Div(
                                                    className="inline-control",
                                                    children=[
                                                        html.Span("Z-axis"),
                                                        dcc.Dropdown(
                                                            id="pca-3d-z-select",
                                                            options=[
                                                                {"label": label, "value": value}
                                                                for value, label in PCA_3D_Z_OPTIONS.items()
                                                            ],
                                                            value="SME_FPI_equal_z",
                                                            clearable=False,
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        loading_graph("pca-3d-figure", className="chart"),
                                        explanation(
                                            "Figure note",
                                            "The 3D plot combines PC1, PC2, and one selected stress metric. Color still represents the descriptive component group.",
                                            "This helps show whether clusters are also separated by stress level or by the SME-CISS gap.",
                                            "It should be treated as an exploratory view because 3D perspective can distort exact visual distance.",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        tab(
                            "Firm Survey Detail",
                            "big-data",
                            [
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(
                                            className="section-heading",
                                            children=[
                                                html.H2("Firm Survey Detail"),
                                                visual_bullets(["386-row index panel", "231,231-row survey detail", "firm size", "sector and problem slices"]),
                                            ],
                                        ),
                                        tab_guide(
                                            "Which firm groups and business problems sit behind the index story?",
                                            "The firm-survey detail table is a diagnostic drill-down layer for firm size, sector, problem type, and response severity.",
                                            "This detailed table is not the core SME-FPI formula; it explains patterns after the borrower-side index has flagged them.",
                                        ),
                                        firm_survey_pressure_panel(),
                                        html.Details(
                                            className="guide-panel compact-details cube-guide",
                                            children=[
                                                html.Summary("Firm survey detail guide"),
                                                html.Div(
                                                    className="guide-panel-copy",
                                                    children=[
                                                        html.H3("What does the firm survey detail table mean here?"),
                                                        compact_copy_bullets(
                                                            [
                                                                ("Meaning", "Grouped SAFE detail combines country, period, firm size, sector, problem category, and response level."),
                                                                ("Formula boundary", "It does not build the core SME-FPI; it explains where pressure appears."),
                                                                ("Coverage", "Smaller countries such as Slovakia may only appear in broader SME or all-sector slices."),
                                                                ("Reader use", "Use it to ask which firm size, sector, or problem sits behind a country flag."),
                                                            ],
                                                            className="compact-guide-list cube-guide-list",
                                                            limit=150,
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="concept-grid",
                                                    children=[
                                                        concept_card(
                                                            "Rows",
                                                            "231,231",
                                                            "Many grouped SAFE survey-response cells, not 231,231 individual firms shown one by one.",
                                                        ),
                                                        concept_card(
                                                            "Dimensions",
                                                            "survey slices",
                                                            "Country, period, firm size, sector, firm age, problem type, and answer severity level.",
                                                        ),
                                                        concept_card(
                                                            "Measures",
                                                            "severity",
                                                            "Severity score 1-10, top-box share 8-10, and high-pressure share 7-10.",
                                                        ),
                                                        concept_card(
                                                            "Purpose",
                                                            "diagnostic",
                                                            "Use it to explain which firm groups and business problems sit behind the index story.",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Details(
                                            className="local-control-panel local-filter-drawer",
                                            children=[
                                                html.Summary("Firm survey filters"),
                                                html.Div(
                                                    className="local-control-grid",
                                                    children=[
                                                        control(
                                                            "Business problem",
                                                            dcc.Dropdown(
                                                                id="problem-select",
                                                                options=[{"label": problem, "value": problem} for problem in PROBLEMS],
                                                                value="Access to finance",
                                                                clearable=False,
                                                            ),
                                                        ),
                                                        control(
                                                            "Firm size",
                                                            dcc.Dropdown(
                                                                id="firm-size-select",
                                                                options=FIRM_SIZE_OPTIONS,
                                                                value="SME",
                                                                clearable=False,
                                                            ),
                                                        ),
                                                        control(
                                                            "Sector",
                                                            dcc.Dropdown(
                                                                id="sector-select",
                                                                options=SECTOR_OPTIONS,
                                                                value="A",
                                                                clearable=False,
                                                            ),
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="two-col",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        loading_graph("big-cube-trend-figure", className="chart"),
                                                        explanation(
                                                            "Figure note",
                                                            "The trend follows severity and severe-response shares for the selected business problem.",
                                                            "A line chart shows whether problem pressure rises, falls, or persists through time; separate y-axes avoid mixing a 1-10 score with percentage shares.",
                                                            "Top-box and high-pressure shares are included because they are safer for ordinal survey answers than relying only on the exact mean score.",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        loading_graph("problem-bar-figure", className="chart"),
                                                        explanation(
                                                            "Figure note",
                                                            "The bar chart compares named problem categories in the large SAFE survey-detail table; the non-specific Other category is excluded.",
                                                            "Sorting supports fast comparison and reduces unnecessary visual search.",
                                                            "It shows whether access to finance is the dominant pressure or part of a broader business-stress mix.",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        loading_graph("size-problem-heatmap-figure", className="chart"),
                                        explanation(
                                            "Figure note",
                                            "The heatmap compares high-pressure shares across firm sizes and named problem categories, excluding the non-specific Other category.",
                                            "This design is compact for a two-way categorical comparison.",
                                            "It helps identify whether micro, small, medium, or large firms experience the most severe pressure for each problem.",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        tab(
                            "Data Preview",
                            "data-preview",
                            [
                                html.Section(
                                    className="section data-preview-section",
                                    children=[
                                        html.Div(
                                            className="section-heading",
                                            children=[
                                                html.H2("Data Preview"),
                                                visual_bullets(["actual files", "raw + processed layers", "structure before charts"]),
                                            ],
                                        ),
                                        tab_guide(
                                            "What do the actual data files look like?",
                                            "The tab provides a quick audit window into raw inputs, processed panels, forecast outputs, and validation files.",
                                            "Previewed rows are for transparency and structure checking; they are not a substitute for the charts or validation results.",
                                        ),
                                        html.Div(
                                            className="source-catalog-panel data-preview-intro",
                                            children=[
                                                html.Div(
                                                    className="source-catalog-copy",
                                                    children=[
                                                        html.Span("Dataset transparency", className="panel-kicker"),
                                                        html.H3("Show the first rows, then explain what each file does"),
                                                        visual_bullets(["raw files = audit trail", "processed files = dashboard layers", "forecast outputs = validation evidence"]),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="concept-grid data-preview-principles",
                                                    children=[
                                                        concept_card(
                                                            "Raw inputs",
                                                            "source layer",
                                                            "SAFE, CISS, World Bank, BLS, MIR, and Eurostat files before dashboard-specific modeling.",
                                                        ),
                                                        concept_card(
                                                            "Processed panels",
                                                            "analysis layer",
                                                            "Country-period tables used to build SME-FPI, gaps, PCA, clusters, forecasts, and diagnostics.",
                                                        ),
                                                        concept_card(
                                                            "Validation outputs",
                                                            "evidence layer",
                                                            "Model evaluation, historical monitoring tiers, and forecast-error tables used to defend the story.",
                                                        ),
                                                        concept_card(
                                                            "Preview limit",
                                                            "readability",
                                                            "Rows are analytical cells or file records, not always individual firms. The table intentionally shows only the top rows so structure is inspectable.",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="local-control-panel data-preview-control-panel",
                                            children=[
                                                html.Div(
                                                    className="inline-heading",
                                                    children=[
                                                        html.Div(
                                                            children=[
                                                                html.Span("Preview controls", className="panel-kicker"),
                                                                html.H3("Choose a dataset and how many top rows to show"),
                                                            ]
                                                        ),
                                                        visual_bullets(["start with SME-FPI panel", "compare SAFE/CISS inputs", "inspect forecast validation"]),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="local-control-grid data-preview-controls",
                                                    children=[
                                                        control(
                                                            "Dataset",
                                                            dcc.Dropdown(
                                                                id="dataset-select",
                                                                options=dataset_preview_options(),
                                                                value="sme_fpi_panel_v2",
                                                                clearable=False,
                                                            ),
                                                        ),
                                                        control(
                                                            "Top rows",
                                                            dcc.Dropdown(
                                                                id="dataset-row-count",
                                                                options=DATASET_ROW_OPTIONS,
                                                                value=10,
                                                                clearable=False,
                                                            ),
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        dcc.Loading(
                                            className="dataset-loading",
                                            type="circle",
                                            color="#2f6f9f",
                                            children=html.Div(id="dataset-preview-container"),
                                        ),
                                        explanation(
                                            "Reading note",
                                            "The preview is not a separate analysis result; it is a quick audit window into the files already used by the charts.",
                                            "Processed tables are easier to read because they have dashboard-ready country, period, model, and metric columns.",
                                            "Raw files can look wider or more cryptic because they preserve source-specific naming and response coding.",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        tab(
                            "Data & Method",
                            "methodology",
                            [
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(className="section-heading", children=[html.H2("Data, Validation, and Method")]),
                                        tab_guide(
                                            "How is the evidence produced and checked?",
                                            "The method keeps the SME-FPI core separate from external predictors, then validates the signal with future outcomes, forecasts, and historical tier replay.",
                                            "The methodology supports a monitoring product claim, not a causal finance model or a policy prescription.",
                                        ),
                                        reader_checkpoint(
                                            "What makes the project credible?",
                                            "The index ingredients, data roles, pipeline, model baselines, forecast loss, and limitations are all visible instead of being hidden behind the final dashboard.",
                                            "This lets the reader separate the core borrower-side index from forecast-only predictors and robustness checks.",
                                            "The project remains descriptive and diagnostic; it does not contain an identification strategy for causal inference.",
                                        ),
                                        data_role_taxonomy_panel(),
                                        defense_question_panel(),
                                        data_lineage_map(),
                                        claim_evidence_matrix(),
                                        html.Div(
                                            className="method-pipeline-panel",
                                            children=[
                                                html.Div(
                                                    className="source-catalog-copy",
                                                    children=[
                                                        html.Span("Reproducible pipeline", className="panel-kicker"),
                                                        html.H3("From source files to dashboard evidence"),
                                                        visual_bullets(["harmonize raw sources", "build borrower-side core", "external data predict/check", "validation bounds claims"]),
                                                    ],
                                                ),
                                                methodology_pipeline(),
                                            ],
                                        ),
                                        no_overclaim_panel(),
                                        html.Div(
                                            className="source-catalog-panel",
                                            children=[
                                                html.Div(
                                                    className="source-catalog-copy",
                                                    children=[
                                                        html.Span("Data and method inventory", className="panel-kicker"),
                                                        html.H3("What data is used, and where it enters the story"),
                                                        visual_bullets(["core index inputs", "forecast-only predictors", "validation targets", "raw audit trail"]),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="source-role-filter",
                                                    children=[
                                                        html.Span("Filter by role"),
                                                        dcc.Dropdown(
                                                            id="source-role-filter",
                                                            options=source_role_options(),
                                                            value="all",
                                                            clearable=False,
                                                        ),
                                                    ],
                                                ),
                                                html.Div(id="source-catalog-table-container", children=source_catalog_table(role_filter="all")),
                                            ],
                                        ),
                                        html.Div(
                                            className="guide-panel validation-guide",
                                            children=[
                                                html.Div(
                                                    className="guide-panel-copy",
                                                    children=[
                                                        html.H3("How to read the validation y-axis"),
                                                        compact_copy_bullets(
                                                            [
                                                                ("Rows", "Future targets used to test whether today's SME-FPI aligns with later stress."),
                                                                ("H+1", "One half-year ahead."),
                                                                ("H+2", "Two half-years ahead."),
                                                            ],
                                                            className="compact-guide-list validation-axis-list",
                                                        ),
                                                    ],
                                                ),
                                                html.Details(
                                                    className="validation-target-drawer",
                                                    children=[
                                                        html.Summary("Validation target glossary"),
                                                        html.Div(
                                                            className="concept-grid validation-concepts",
                                                            children=[
                                                                concept_card(
                                                                    "SME-FPI H+1 / H+2",
                                                                    "future index",
                                                                    "The same borrower-side financing-pain index one or two half-years later.",
                                                                ),
                                                                concept_card(
                                                                    "Access severity H+1 / H+2",
                                                                    "SAFE survey detail",
                                                                    "Future 1-10 severity score for access to finance as a business problem.",
                                                                ),
                                                                concept_card(
                                                                    "Access top-box H+1",
                                                                    "8-10 share",
                                                                    "Future share of responses in the most severe 8-10 range for access to finance.",
                                                                ),
                                                                concept_card(
                                                                    "Access high pressure H+1",
                                                                    "7-10 share",
                                                                    "Future share of responses in the high-pressure 7-10 range for access to finance.",
                                                                ),
                                                                concept_card(
                                                                    "Macro context H+1 / H+2",
                                                                    "World Bank",
                                                                    "Future macro stress context built from GDP growth, unemployment, inflation, and private credit indicators.",
                                                                ),
                                                                concept_card(
                                                                    "CISS H+1",
                                                                    "market stress",
                                                                    "Future ECB market-stress benchmark. This checks whether SME-FPI aligns with later market stress.",
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(className="metric-grid validation-metric-grid", children=validation_summary_cards()),
                                        loading_graph("validation-figure", figure=make_validation_chart(), className="chart"),
                                        explanation(
                                            "Figure note",
                                            "The validation chart compares each index version with later stress-related outcomes shown on the y-axis.",
                                            "Grouped bars let the viewer compare index versions for the same future target without overplotting.",
                                            "Read the x-axis as correlation strength. Higher positive values mean the index is more aligned with that future target within countries.",
                                        ),
                                        html.Div(
                                            className="methodology-grid",
                                            children=[
                                                html.Div(
                                                    className="method-block",
                                                    children=[
                                                        html.H3("Design safeguards"),
                                                        html.Ul(
                                                            children=[
                                                                html.Li("All components are coded so higher values mean more financing pain."),
                                                                html.Li("The index is calculated only when at least four of six components are available."),
                                                                html.Li("Component coverage is shown in hover labels to avoid hiding missingness."),
                                                                html.Li("Alternative weighting versions are shown as a robustness range, not as a statistical confidence interval."),
                                                                html.Li("Forecast models use rolling origins; future target columns are blocked from training features."),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="method-block",
                                                    children=[
                                                        html.H3("Main limitations"),
                                                        html.Ul(
                                                            children=[
                                                                html.Li("The index is descriptive and diagnostic, not causal."),
                                                                html.Li("SAFE survey answers measure reported conditions and partly ordinal responses."),
                                                                html.Li("CISS is a common euro-area market-stress benchmark, not a country-specific SME credit variable."),
                                                                html.Li("Full-sample z-scores are best for retrospective comparison; fixed-baseline SME-FPI is the real-time robustness check."),
                                                                html.Li("PCA weights maximize explained variance, not economic importance."),
                                                                html.Li("Model agreement quality means model consistency, not certainty that the H+1 outcome will occur."),
                                                                html.Li("Rolling-origin forecasts block future values, but exact publication lags for every external source are not fully modeled."),
                                                                html.Li("GARCH is excluded because the dashboard predicts semiannual survey-index levels, not high-frequency volatility."),
                                                                html.Li("Seasonal SARIMA is not a headline model because each country has too few half-year observations to defend seasonal structure."),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("lens-panel", "children"),
    Input("reader-lens", "value"),
)
def update_reader_lens(mode):
    return reader_lens_panel(mode)


@app.callback(
    Output("main-tabs", "value"),
    Input("jump-start", "n_clicks"),
    Input("jump-board", "n_clicks"),
    Input("jump-defense", "n_clicks"),
    Input("jump-forecast", "n_clicks"),
    Input("jump-method", "n_clicks"),
    Input("jump-survey", "n_clicks"),
    Input("lens-primary-action", "n_clicks"),
    Input("read-path-start", "n_clicks"),
    Input("read-path-decision", "n_clicks"),
    Input("read-path-findings", "n_clicks"),
    Input("read-path-overview", "n_clicks"),
    Input("read-path-forecast", "n_clicks"),
    Input("read-path-diagnosis", "n_clicks"),
    Input("read-path-methodology", "n_clicks"),
    State("reader-lens", "value"),
    prevent_initial_call=True,
)
def jump_to_tab(*args):
    reader_mode = args[-1]
    return target_tab_from_button(ctx.triggered_id, reader_mode)


@app.callback(
    Output("reading-path-detail", "children"),
    Output("read-path-start", "className"),
    Output("read-path-decision", "className"),
    Output("read-path-findings", "className"),
    Output("read-path-overview", "className"),
    Output("read-path-forecast", "className"),
    Output("read-path-diagnosis", "className"),
    Output("read-path-methodology", "className"),
    Input("read-path-start", "n_clicks"),
    Input("read-path-decision", "n_clicks"),
    Input("read-path-findings", "n_clicks"),
    Input("read-path-overview", "n_clicks"),
    Input("read-path-forecast", "n_clicks"),
    Input("read-path-diagnosis", "n_clicks"),
    Input("read-path-methodology", "n_clicks"),
    prevent_initial_call=True,
)
def update_reading_path_detail(*_clicks):
    target = READING_PATH_TABS.get(ctx.triggered_id, "start")
    return [reading_path_detail(target), *reading_path_node_classes(target)]


@app.callback(
    Output("forecast-view-point", "style"),
    Output("forecast-view-uncertainty", "style"),
    Output("forecast-view-benchmark", "style"),
    Input("forecast-view-select", "value"),
)
def update_forecast_view(selected):
    return forecast_view_styles(selected)


@app.callback(
    Output("source-catalog-table-container", "children"),
    Input("source-role-filter", "value"),
)
def update_source_catalog_table(role_filter):
    return source_catalog_table(role_filter=role_filter)


@app.callback(
    Output("country-select", "value"),
    Input("select-all-countries", "n_clicks"),
    Input("clear-all-countries", "n_clicks"),
    prevent_initial_call=True,
)
def set_country_selection(_select_all_clicks, _clear_all_clicks):
    if ctx.triggered_id == "select-all-countries":
        return ALL_COUNTRY_CODES
    if ctx.triggered_id == "clear-all-countries":
        return []
    return DEFAULT_COUNTRIES


@app.callback(
    Output("dataset-preview-container", "children"),
    Input("dataset-select", "value"),
    Input("dataset-row-count", "value"),
)
def update_dataset_preview(dataset_key, row_count):
    return render_dataset_preview(dataset_key, row_count)


@app.callback(
    Output("kpi-row", "children"),
    Output("decision-summary-cards", "children"),
    Output("decision-narrative-container", "children"),
    Output("standardization-caveat-container", "children"),
    Output("command-panel-container", "children"),
    Output("decision-table-container", "children"),
    Output("decision-scatter-figure", "figure"),
    Output("agreement-figure", "figure"),
    Output("risk-history-figure", "figure"),
    Output("tier-validation-figure", "figure"),
    Output("driver-heatmap-figure", "figure"),
    Output("diagnosis-card-board", "children"),
    Output("map-figure", "figure"),
    Output("time-series-figure", "figure"),
    Output("robustness-figure", "figure"),
    Output("forecast-figure", "figure"),
    Output("forecast-loss-figure", "figure"),
    Output("model-rank-figure", "figure"),
    Output("animated-stress-figure", "figure"),
    Output("heatmap-figure", "figure"),
    Output("component-heatmap-figure", "figure"),
    Output("gap-ranking-figure", "figure"),
    Output("bubble-gap-figure", "figure"),
    Output("pca-scatter-figure", "figure"),
    Output("pca-3d-figure", "figure"),
    Output("big-cube-trend-figure", "figure"),
    Output("problem-bar-figure", "figure"),
    Output("size-problem-heatmap-figure", "figure"),
    Input("country-select", "value"),
    Input("index-select", "value"),
    Input("period-range", "value"),
    Input("pca-3d-z-select", "value"),
    Input("problem-select", "value"),
    Input("firm-size-select", "value"),
    Input("sector-select", "value"),
)
def update_dashboard(countries, metric_col, period_range, pca_3d_z, problem, firm_size, sector):
    panel_range = period_filter(PANEL, period_range)
    decision_board = build_decision_board(period_range, countries, metric_col)
    return (
        kpi_cards(panel_range),
        decision_summary_cards(decision_board),
        decision_narrative(decision_board),
        standardization_caveat(metric_col, period_range),
        command_center_panel(decision_board, metric_col, period_range),
        decision_table(decision_board),
        make_decision_scatter(decision_board),
        make_agreement_figure(decision_board),
        make_risk_history_figure(period_range, countries),
        make_tier_validation_figure(),
        make_driver_heatmap(decision_board),
        diagnosis_cards(decision_board),
        make_map(panel_range, metric_col),
        make_time_series(panel_range, countries, metric_col),
        make_robustness(panel_range, countries),
        make_forecast_figure(period_range, countries, metric_col),
        make_forecast_loss_figure(countries),
        make_model_rank_heatmap(),
        make_animated_stress_motion(panel_range, countries, metric_col),
        make_heatmap(panel_range, metric_col),
        make_component_heatmap(panel_range, countries),
        make_gap_ranking(panel_range, metric_col),
        make_bubble_gap(panel_range, countries, metric_col),
        make_pca_scatter(panel_range, countries),
        make_pca_3d_regime(panel_range, countries, pca_3d_z),
        make_big_cube_trend(period_range, countries, problem, firm_size, sector),
        make_problem_bar(period_range, countries, firm_size, sector),
        make_size_problem_heatmap(period_range, countries, sector),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
