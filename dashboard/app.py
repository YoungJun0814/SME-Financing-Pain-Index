import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, dcc, html, ctx
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

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
    "momentum": "Last-change extrapolation",
    "pooled_lag_ols": "Pooled lag OLS",
    "elastic_net": "Elastic Net",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

BASELINE_MODEL_KEYS = ["naive", "country_mean", "country_ar1", "momentum", "pooled_lag_ols"]
ML_MODEL_KEYS = ["elastic_net", "ridge", "random_forest", "gradient_boosting"]
MODEL_FAMILIES = {
    "naive": "baseline",
    "country_mean": "baseline",
    "country_ar1": "baseline",
    "momentum": "baseline",
    "pooled_lag_ols": "baseline",
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
}


def load_optional_csv(path, default):
    if path.exists():
        return pd.read_csv(path)
    return default.copy() if isinstance(default, pd.DataFrame) else default


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
        plot_bgcolor="rgba(255,255,255,0.72)",
        hoverlabel={"bgcolor": "white", "font_size": 12, "font_family": "Inter, Segoe UI, Arial, sans-serif"},
        legend={"orientation": "h", "yanchor": "top", "y": -0.16, "xanchor": "left", "x": 0},
        legend_title_text="",
        showlegend=showlegend,
    )
    fig.update_xaxes(
        title=x_title,
        showgrid=True,
        gridcolor="#edf1f4",
        zerolinecolor="#b8c2cc",
        linecolor="#d4dce3",
    )
    fig.update_yaxes(
        title=y_title,
        showgrid=True,
        gridcolor="#edf1f4",
        zerolinecolor="#b8c2cc",
        linecolor="#d4dce3",
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


def explanation(title, what, why, insight):
    return html.Div(
        className="explanation",
        children=[
            html.H4(title),
            html.P([html.Strong("What it shows: "), what]),
            html.P([html.Strong("Why this chart: "), why]),
            html.P([html.Strong("Key insight: "), insight]),
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
            html.P(body),
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
                    html.P(body),
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


def source_catalog_table(limit=None):
    if SOURCE_CATALOG.empty:
        return html.Div(
            className="source-table-empty",
            children="Source catalog is not available. Run scripts/11_build_forecasting_layer.py to generate it.",
        )
    frame = SOURCE_CATALOG.copy()
    if limit:
        frame = frame.head(limit)
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
    naive_mae = row.get("latest_backtest_naive_mae", np.nan)
    strong_baseline = row.get("strongest_recent_baseline_label", "n/a")
    strong_baseline_mae = row.get("strongest_recent_baseline_mae", np.nan)
    improvement = row.get("best_recent_ml_improvement_vs_strongest_baseline", np.nan)
    return [
        metric_card(
            "Forecast feature panel",
            f"{int(row.get('columns_forecasting_feature_panel', 0)):,} cols",
            f"{int(row.get('rows_forecasting_feature_panel', 0)):,} country x half-year rows",
        ),
        metric_card(
            "Macro + micro predictors",
            f"{int(row.get('macro_micro_predictor_columns', 0)):,}",
            "SAFE cube plus macro context columns",
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
            "Strongest simple baseline",
            str(strong_baseline),
            f"MAE {format_number(strong_baseline_mae)}; ML edge {format_number(improvement, 3)}",
        ),
        metric_card(
            "Elastic Net check",
            f"MAE {format_number(elastic_mae)}",
            "regularized linear benchmark",
        ),
    ]


def forecast_feature_list():
    groups = [
        ("Current pain", "SME-FPI variants, component z-scores, PCA coordinates, CISS gap"),
        ("Macro climate", "GDP slowdown, unemployment, inflation, credit contraction, macro stress context"),
        ("Micro vulnerability", "SAFE Q0B access severity by firm size, sector, and broad problem pressure"),
        ("Lender-side signal", "ECB BLS SME credit standards, loan demand, terms, and rejection pressure"),
        ("Loan market signal", "ECB MIR small-loan rates, large-loan benchmark, spreads, and volumes"),
        ("Business outcome signal", "Eurostat quarterly registrations and bankruptcy indexes as external real-economy predictors"),
        ("Model suite", "Elastic Net, Ridge, Random Forest, and Gradient Boosting compared against naive, country mean, country AR(1), momentum, and pooled OLS baselines"),
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
            html.Div(children=[html.Strong(title), html.P(body)]),
        ],
    )


def term_item(title, tag, body):
    return html.Div(
        className="term-item",
        children=[
            html.Span(tag),
            html.Strong(title),
            html.P(body),
        ],
    )


def variable_item(symbol, title, body):
    return html.Div(
        className="variable-item",
        children=[
            dcc.Markdown(symbol, mathjax=True, className="variable-symbol"),
            html.Div(children=[html.Strong(title), html.P(body)]),
        ],
    )


def journey_card(number, title, body):
    return html.Div(
        className="journey-card",
        children=[
            html.Span(number, className="journey-number"),
            html.H3(title),
            html.P(body),
        ],
    )


def control(label, element):
    return html.Div(className="control", children=[html.Span(label), element])


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
    else:
        raise ValueError(f"Unknown baseline model: {model_key}")

    pred = pd.Series(pred, index=test_index).replace([np.inf, -np.inf], np.nan).fillna(fallback)
    return pred.to_numpy(dtype=float)


def model_predictions_for_origin(end_idx, metric_col, countries=None):
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
        if isinstance(key, str) and key in FORECAST_MODEL_LABELS and key != "naive":
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

    next_period = PERIODS[end_idx + 1] if end_idx + 1 < len(PERIODS) else next_half_year_label(latest["TIME_PERIOD"].iloc[0])

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
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.12,
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
            row=1,
            col=2,
        )
        fig.update_yaxes(autorange="reversed", row=1, col=2)

    fig.add_vline(x=0, line_dash="dot", line_color="#8b98a5", row=1, col=1)
    fig.update_xaxes(title_text="Standardized score", row=1, col=1)
    fig.update_yaxes(title_text=None, row=1, col=1)
    fig.update_xaxes(title_text="Mean absolute error, lower is better", row=1, col=2)
    fig.update_yaxes(title_text=None, row=1, col=2)
    fig = polish(
        fig,
        chart_title(
            "Which countries may face rising SME financing pressure next?",
            f"Expanded SAFE + macro + BLS/MIR/Eurostat model suite from {latest['TIME_PERIOD'].iloc[0]} to {next_period}; baselines included; {mae_label}",
        ),
        height=560,
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


def decision_driver_text(row, context):
    drivers = []
    for col, label in COMPONENT_DRIVER_LABELS.items():
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
        drivers.append((float(row["gap_value"]) + 0.5, "Hidden SME-CISS gap"))
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
    best = predictions[predictions["model_key"] == best_key][
        ["REF_AREA", "predicted_score", "predicted_delta", "model_label", "forecast_target_period"]
    ].rename(
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
    board["primary_drivers"] = board.apply(lambda row: decision_driver_text(row, board), axis=1)
    board["recommended_read"] = np.select(
        [
            board["risk_tier"].eq("Alert"),
            board["risk_tier"].eq("Watch"),
            board["risk_tier"].eq("Monitor"),
        ],
        [
            "Immediate watchlist: inspect local drivers.",
            "Watch closely: pressure is visible or forecast to rise.",
            "Monitor: at least one warning signal is present.",
        ],
        default="Normal: no major warning signal.",
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
            metric_card("Decision board", "not available", "select a later period with enough training data"),
            metric_card("Watchlist", "n/a", "forecast model unavailable"),
            metric_card("Model confidence", "n/a", "no country diagnostics"),
            metric_card("Main risk", "n/a", "no current signal"),
        ]
    alert_count = int((board["risk_tier"] == "Alert").sum())
    watch_count = int((board["risk_tier"] == "Watch").sum())
    monitor_count = int((board["risk_tier"] == "Monitor").sum())
    top = board.iloc[0]
    rising = board[board["forecast_direction"] == "Rising"]
    high_conf = int((board["confidence"] == "High").sum())
    return [
        metric_card("Decision watchlist", f"{alert_count + watch_count + monitor_count}", f"{alert_count} alert, {watch_count} watch, {monitor_count} monitor"),
        metric_card("Top current signal", str(top["country_name"]), f"{top['risk_tier']} score {format_number(top['risk_score'], 1)}"),
        metric_card("Forecast rising", f"{len(rising)} countries", "best recent ML model delta is positive"),
        metric_card("High-confidence reads", f"{high_conf}", "tight ML range and strong agreement"),
    ]


def decision_narrative(board):
    if board.empty:
        return html.Div(
            className="decision-narrative",
            children=[
                html.Span("Latest read", className="panel-kicker"),
                html.P("No decision-board read is available for this period. Move the period slider later or rebuild the forecasting layer."),
            ],
        )
    counts = board["risk_tier"].value_counts()
    alert_count = int(counts.get("Alert", 0))
    watch_count = int(counts.get("Watch", 0))
    monitor_count = int(counts.get("Monitor", 0))
    rising_count = int((board["forecast_direction"] == "Rising").sum())
    high_conf = int((board["confidence"] == "High").sum())
    top_names = ", ".join(board.head(3)["country_name"].astype(str).tolist())
    model_label = str(board["best_model_label"].dropna().iloc[0]) if board["best_model_label"].notna().any() else "best recent ML model"
    origin = str(board["TIME_PERIOD"].iloc[0])
    target = (
        str(board["forecast_target_period"].dropna().iloc[0])
        if "forecast_target_period" in board.columns and board["forecast_target_period"].notna().any()
        else "the next half-year"
    )
    return html.Div(
        className="decision-narrative",
        children=[
            html.Span("Latest read", className="panel-kicker"),
            html.P(
                f"As of {origin}, the selected view has {alert_count} Alert, {watch_count} Watch, "
                f"and {monitor_count} Monitor countries. {model_label} projects {rising_count} countries "
                f"with rising H+1 pressure toward {target}; {high_conf} country reads have high model-agreement confidence. "
                f"Start with {top_names}, then use the driver heatmap and country cards to explain whether the signal comes from "
                "current borrower pain, a hidden SME-CISS gap, or forward model momentum."
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
            "Did the decision rule behave consistently through time?",
            "Historical Alert/Watch/Monitor/Normal assignments from rolling-origin forecast backtests",
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
                    validation["mean_actual_delta"],
                    validation["mean_forecast_abs_error"],
                    validation["direction_hit_share"] * 100,
                ]
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Pressure rose next period: %{y:.0f}%<br>"
                "Country-origin reads: %{customdata[0]:.0f}<br>"
                "Mean actual delta: %{customdata[1]:.2f}<br>"
                "Mean forecast abs. error: %{customdata[2]:.2f}<br>"
                "Directional hit rate: %{customdata[3]:.0f}%<extra></extra>"
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
            "How should the risk tiers be trusted?",
            "Historical validation by tier; bars show next-period pressure-rise share, line shows directional hit rate",
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
            "Which forecasting models are stable, not just lucky?",
            "Rolling-origin MAE rank by model; darker cells indicate stronger relative performance",
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
        return html.Div(className="source-table-empty", children="Decision board is unavailable for this period.")
    rows = []
    for row in board.head(12).itertuples():
        rows.append(
            html.Tr(
                children=[
                    html.Td(str(row.country_name)),
                    html.Td(html.Span(str(row.risk_tier), className=f"risk-badge risk-badge-{str(row.risk_tier).lower()}")),
                    html.Td(format_number(row.current_score)),
                    html.Td(format_number(row.gap_value)),
                    html.Td(format_number(row.best_model_forecast)),
                    html.Td(f"{format_number(row.ml_model_agreement_rising * 100, 0)}%"),
                    html.Td(str(row.confidence)),
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
                                html.Th("Current"),
                                html.Th("Gap"),
                                html.Th("H+1"),
                                html.Th("ML rising"),
                                html.Th("Confidence"),
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
        return empty_figure("Decision board is unavailable for this period.")
    fig = px.scatter(
        board,
        x="gap_value",
        y="current_score",
        size=np.clip(board["risk_score"], 0.8, None),
        color="risk_tier",
        text="REF_AREA",
        category_orders={"risk_tier": ["Alert", "Watch", "Monitor", "Normal"]},
        color_discrete_map=RISK_TIER_COLORS,
        custom_data=["country_name", "best_model_forecast", "best_model_delta", "ml_model_agreement_rising", "confidence", "primary_drivers"],
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
            "Confidence: %{customdata[4]}<br>"
            "Drivers: %{customdata[5]}<extra></extra>"
        ),
    )
    fig.add_hline(y=0.35, line_dash="dot", line_color="#d19a2e")
    fig.add_vline(x=0.35, line_dash="dot", line_color="#d19a2e")
    fig.add_hline(y=1.0, line_dash="dot", line_color="#a4312e")
    fig.add_vline(x=1.0, line_dash="dot", line_color="#a4312e")
    fig = polish(
        fig,
        chart_title(
            "Who needs attention, and why?",
            "Risk tier combines current SME-FPI, hidden SME-CISS gap, H+1 forecast, and ML agreement",
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
            customdata=np.stack([focus["forecast_direction"], focus["ml_model_agreement_rising"], focus["confidence"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Best ML H+1: %{x:.2f}<br>"
                "%{customdata[0]} pressure<br>"
                "ML rising agreement: %{customdata[1]:.0%}<br>"
                "Confidence: %{customdata[2]}<extra></extra>"
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
            "Does the model suite agree on direction?",
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
                    html.P(str(row.primary_drivers)),
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
            "Severity score plus severe-response shares from the 231,231-row SAFE cube",
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
        metric_card("SAFE Q0B cube", f"{BIG_CUBE_ROWS:,} rows", "large survey-response cube"),
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


app = Dash(__name__, title="SME Financing Pain Observatory")
server = app.server

def tab(label, value, children):
    return dcc.Tab(
        label=label,
        value=value,
        className="main-tab",
        selected_className="main-tab main-tab--selected",
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
                        html.P("Interactive data visualization project", className="eyebrow"),
                        html.H1("SME Financing Pain Observatory"),
                        html.P(
                            "A borrower-side index for detecting when European SMEs report financing pain "
                            "that is not fully visible in common market-stress benchmarks such as the ECB CISS."
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
                                html.Span("231k SAFE cube"),
                                html.Span("PCA + clusters"),
                                html.Span("CISS benchmark"),
                                html.Span("Macro + micro predictors"),
                                html.Span("BLS/MIR/Eurostat forecast layer"),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="hero-stack",
                    children=[
                        html.Div(
                            className="hero-method",
                            children=[
                                html.H2("Core idea"),
                                html.P(
                                    "CISS is the ECB Composite Indicator of Systemic Stress, a thermometer for stress "
                                    "inside financial markets. SME-FPI is different: it measures pressure reported by "
                                    "small and medium-sized firms as borrowers."
                                ),
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
                                                html.P(
                                                    "These controls update the dashboard story. Hover chart elements to see values, "
                                                    "interpretation labels, coverage, and methodology notes."
                                                ),
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
                                                        html.Span("Use the default equal-weight index and all countries to see the overall pattern."),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Strong("Then focus"),
                                                        html.Span("Clear countries and select a few countries when exact line-by-line comparison matters."),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        html.Strong("Interpretation rule"),
                                                        html.Span("Positive SME-CISS gaps mean borrower-side pain is high relative to market stress."),
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
                                                        html.P(
                                                            "The dropdown does not change the research question. It changes how the same "
                                                            "borrower-side financing-pain signal is standardized or weighted. The default "
                                                            "equal-weight version is the easiest baseline for a first reading."
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
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tabs(
                    id="main-tabs",
                    value="decision",
                    parent_className="tabs-shell",
                    className="main-tabs",
                    children=[
                        tab(
                            "Decision Board",
                            "decision",
                            [
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(
                                            className="section-heading",
                                            children=[
                                                html.H2("Decision Board"),
                                                html.P(
                                                    "A monitoring-style first screen that answers who needs attention, why the signal appears, "
                                                    "whether the stress is hidden from CISS, and whether the H+1 model suite expects pressure to rise."
                                                ),
                                            ],
                                        ),
                                        html.Div(id="decision-summary-cards", className="metric-grid decision-metric-grid"),
                                        html.Div(id="decision-narrative-container"),
                                        html.Div(
                                            className="decision-logic-panel",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Span("Risk rule", className="panel-kicker"),
                                                        html.H3("Current pain + hidden gap + H+1 forecast + model agreement"),
                                                        html.P(
                                                            "Alert, Watch, Monitor, and Normal are diagnostic tiers, not policy decisions. "
                                                            "They combine standardized current SME-FPI, SME-FPI minus CISS, best recent ML forecast, "
                                                            "and agreement across Elastic Net, Ridge, Random Forest, and Gradient Boosting."
                                                        ),
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
                                        html.Div(
                                            className="two-col",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        dcc.Graph(id="decision-scatter-figure", className="chart chart-glow"),
                                                        explanation(
                                                            "Figure note",
                                                            "The scatter places each country by current borrower-side pain and hidden SME-CISS gap; marker size reflects the diagnostic risk score.",
                                                            "This turns the dashboard from descriptive charts into a triage surface for monitoring.",
                                                            "Countries in the upper-right require the most careful reading because borrower pain is high and market stress may not fully show it.",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        dcc.Graph(id="agreement-figure", className="chart"),
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
                                                        dcc.Graph(id="risk-history-figure", className="chart"),
                                                        explanation(
                                                            "Validation note",
                                                            "This heatmap replays the Decision Board through historical forecast origins.",
                                                            "It shows whether countries repeatedly appeared as Alert, Watch, or Monitor instead of only reporting the latest status.",
                                                            "Persistent warm cells are more meaningful than one isolated warning cell.",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        dcc.Graph(id="tier-validation-figure", className="chart"),
                                                        explanation(
                                                            "Validation note",
                                                            "The tier validation chart compares diagnostic tiers with realized next-period movement.",
                                                            "This avoids treating the risk tier as a black box: a good monitoring tier should separate high-pressure states while making forecast uncertainty visible.",
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
                                                        html.P("Sorted by tier, risk score, and current SME-FPI. Use the driver column to decide which diagnostic tab to inspect next."),
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
                            "Overview",
                            "overview",
                            [
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(
                                            className="section-heading",
                                            children=[
                                                html.H2("Project Overview"),
                                                html.P(
                                                    "Research question: can a borrower-side SME Financing Pain Index reveal "
                                                    "European SME credit stress that a market-side stress index does not fully capture, "
                                                    "and can external macro, micro, lender-side, and loan-market data turn it into an early-warning signal?"
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="overview-brief",
                                            children=[
                                                html.Div(
                                                    className="overview-thesis",
                                                    children=[
                                                        html.Span("One-sentence thesis", className="panel-kicker"),
                                                        html.H3("SME-FPI asks whether small firms feel financing pressure that market stress does not fully show."),
                                                        html.P(
                                                            "The project compares two signals: CISS as a market-side stress thermometer, "
                                                            "and SME-FPI as a borrower-side pressure measure built from ECB SAFE survey answers."
                                                        ),
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
                                                        read_step("1", "Start with decisions", "Decision Board shows the current watchlist, risk tier, model agreement, and drivers."),
                                                        read_step("2", "Compare over time", "Explorer shows whether SME-FPI and CISS move together or diverge."),
                                                        read_step("3", "Forecast carefully", "Forecast Lab checks ML models against stronger simple baselines before flagging H+1 pressure."),
                                                        read_step("4", "Explain the country", "Country Diagnosis and Hidden Stress connect risk tiers to drivers and SME-CISS gaps."),
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
                                                        html.P(
                                                            "This separation keeps SME-FPI interpretable. SAFE borrower answers build the core index, "
                                                            "while macro context, firm-group vulnerability, bank lending conditions, loan rates, and "
                                                            "future outcomes are used for forecasting and validation."
                                                        ),
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
                                                            "SAFE cube",
                                                            "Firm size, sector, and problem-severity slices show who is exposed.",
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
                                                        html.P(
                                                            "For each country and half-year, available borrower-side SAFE components are standardized, "
                                                            "averaged, and standardized again. The index is calculated only when at least four "
                                                            "components are available."
                                                        ),
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
                                                        html.P(
                                                            "The main model uses a country-by-half-year panel with 386 observations. "
                                                            "The Big Data layer uses a 231,231-row SAFE Q0B cube by country, period, "
                                                            "firm size, sector, problem category, and response level."
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="method-block",
                                                    children=[
                                                        html.H3("Design stance"),
                                                        html.P(
                                                            "The dashboard keeps overview, exploration, hidden stress, regimes, big-data "
                                                            "diagnostics, and methodology separated. This follows a monitoring-style workflow: "
                                                            "see the status, drill into a pattern, then inspect the method."
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
                            "Explorer",
                            "explorer",
                            [
                                html.Section(
                                    className="section two-col",
                                    children=[
                                        html.Div(
                                            children=[
                                                dcc.Graph(id="time-series-figure", className="chart"),
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
                                                dcc.Graph(id="robustness-figure", className="chart"),
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
                                                html.P(
                                                    "This animation plays the country-half-year panel like a time-lapse. Each country "
                                                    "moves through the space between market-side CISS and borrower-side SME-FPI."
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(id="animated-stress-figure", className="chart"),
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
                                        dcc.Graph(id="heatmap-figure", className="chart"),
                                        explanation(
                                            "Figure note",
                                            "The heatmap displays SME-FPI values across countries and half-year periods.",
                                            "Heatmaps are efficient for matrix-like comparisons where both rows and columns matter.",
                                            "Long red bands suggest persistent borrower-side pressure, while isolated red cells suggest shorter stress episodes.",
                                        ),
                                        dcc.Graph(id="component-heatmap-figure", className="chart"),
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
                            "Forecast",
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
                                                        html.H2("Forecast Lab"),
                                                        html.P(
                                                            "A one-half-year-ahead early-warning experiment that compares Elastic Net, Ridge, "
                                                            "Random Forest, and Gradient Boosting using SME-FPI plus macro, micro, lender-side, "
                                                            "and loan-market predictors, then checks them against stronger simple baselines."
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="forecast-badges",
                                                    children=[
                                                        html.Span("Elastic Net"),
                                                        html.Span("Ridge"),
                                                        html.Span("Random Forest"),
                                                        html.Span("Gradient Boosting"),
                                                        html.Span("H+1 forecast"),
                                                        html.Span("strong baseline check"),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        html.Div(className="metric-grid forecast-metric-grid", children=forecasting_summary_cards()),
                                        html.Div(
                                            className="forecast-stack-panel",
                                            children=[
                                                html.Div(
                                                    className="forecast-stack-copy",
                                                    children=[
                                                        html.Span("Predictor stack", className="panel-kicker"),
                                                        html.H3("What the model is allowed to know at the forecast origin"),
                                                        html.P(
                                                            "The forecast uses current and lagged information available at the selected period. "
                                                            "Future outcome columns are blocked from training, and the right panel compares "
                                                            "each model against naive, country mean, country AR(1), momentum, and pooled lag OLS baselines."
                                                        ),
                                                    ],
                                                ),
                                                forecast_feature_list(),
                                            ],
                                        ),
                                        html.Div(
                                            className="forecast-caveat",
                                            children=[
                                                html.Strong("Why this model suite?"),
                                                html.Span(
                                                    "The panel has 386 country-period observations, so the dashboard uses compact regularized "
                                                    "linear models plus tree-based nonlinear models. This gives a stronger ML story without "
                                                    "pretending the sample can support a large neural network."
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(id="forecast-figure", className="chart chart-glow"),
                                        explanation(
                                            "Figure note",
                                            "The left panel forecasts next half-year SME-FPI from the selected period using the best recent ML model. The right panel compares Elastic Net, Ridge, Random Forest, Gradient Boosting, and five simple baselines.",
                                            "A country-level dumbbell plus model leaderboard is clearer than a dense projection because the data are a small country-period panel.",
                                            "Use the forecast to flag countries for attention, then use the historical and component charts to explain why.",
                                        ),
                                        dcc.Graph(id="model-rank-figure", className="chart"),
                                        explanation(
                                            "Model stability note",
                                            "The rank heatmap checks whether a model performs consistently across rolling-origin windows.",
                                            "This matters because the best recent model should not be trusted only because it won one period by chance.",
                                            "Stable regularized models are usually preferable in this small country-period panel unless a nonlinear model repeatedly beats the simple baselines.",
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
                                                html.P(
                                                    "A drill-down view for the countries with the strongest current diagnostic signal. "
                                                    "Each card links the risk tier to current SME-FPI, hidden gap, forecast direction, model agreement, and top drivers."
                                                ),
                                            ],
                                        ),
                                        html.Div(id="diagnosis-card-board"),
                                        html.Div(
                                            className="two-col",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        dcc.Graph(id="driver-heatmap-figure", className="chart"),
                                                        explanation(
                                                            "Figure note",
                                                            "The heatmap standardizes drivers within the displayed countries, so color shows which factors are relatively high in that watchlist group.",
                                                            "This turns the index from a single score into a country explanation layer.",
                                                            "Use it to separate borrower-side survey pressure from hidden gap and forecast momentum.",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="diagnosis-method-panel",
                                                    children=[
                                                        html.Span("How to use this view", className="panel-kicker"),
                                                        html.H3("Read country cards as evidence bundles"),
                                                        html.P(
                                                            "A country is not flagged because of one metric alone. The tier combines level, gap, forecast direction, and model agreement. "
                                                            "The driver text names the most important visible reasons, then the Explorer, Hidden Stress, Forecast, and Big Data Cube tabs provide the detailed evidence."
                                                        ),
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
                            "Hidden Stress",
                            "hidden-stress",
                            [
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(
                                            className="section-heading",
                                            children=[
                                                html.H2("Geographic Snapshot"),
                                                html.P(
                                                    "A country-level view of the latest available period inside the selected range."
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(id="map-figure", className="chart"),
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
                                                dcc.Graph(id="gap-ranking-figure", className="chart"),
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
                                                dcc.Graph(id="bubble-gap-figure", className="chart"),
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
                            "Regimes",
                            "regimes",
                            [
                                html.Section(
                                    className="section two-col",
                                    children=[
                                        html.Div(
                                            children=[
                                                dcc.Graph(id="pca-scatter-figure", className="chart"),
                                                explanation(
                                                    "Figure note",
                                                    "The PCA scatter groups country-period observations into financing-pain regimes.",
                                                    "A scatterplot is suitable because PCA produces two continuous dimensions for multivariate structure.",
                                                    "Clusters are descriptive regimes, not causal classes; they help explain recurring stress patterns.",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            children=[
                                                dcc.Graph(id="pca-circle-figure", figure=make_pca_circle(), className="chart"),
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
                                                        html.H2("3D PCA Regime Explorer"),
                                                        html.P(
                                                            "A rotating exploratory view for multidimensional data analysis. It is useful "
                                                            "for inspection, but the 2D PCA scatter and validation chart remain the primary evidence."
                                                        ),
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
                                        dcc.Graph(id="pca-3d-figure", className="chart"),
                                        explanation(
                                            "Figure note",
                                            "The 3D plot combines PC1, PC2, and one selected stress metric. Color still represents the descriptive cluster regime.",
                                            "This helps show whether clusters are also separated by stress level or by the SME-CISS gap.",
                                            "It should be treated as an exploratory view because 3D perspective can distort exact visual distance.",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        tab(
                            "Big Data Cube",
                            "big-data",
                            [
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(
                                            className="section-heading",
                                            children=[
                                                html.H2("Big Data SAFE Cube Diagnostics"),
                                                html.P(
                                                    "The 386-row panel builds the index. The 231,231-row SAFE Q0B cube is used as a larger diagnostic layer by country, period, firm size, sector, problem, and response level."
                                                ),
                                            ],
                                        ),
                                        html.Details(
                                            className="guide-panel compact-details cube-guide",
                                            children=[
                                                html.Summary("Big Data Cube guide"),
                                                html.Div(
                                                    className="guide-panel-copy",
                                                    children=[
                                                        html.H3("What does 'Big Data Cube' mean here?"),
                                                        html.P(
                                                            "A cube is a multi-dimensional survey table. Each slice combines country, half-year period, firm size, sector, problem category, and response level. It is not used to build the core SME-FPI; it explains where the broader pressure appears."
                                                        ),
                                                        html.P(
                                                            "Coverage caveat: Slovakia enters the main SME-FPI panel in 2014-S1. In the Big Data Cube, smaller countries such as Slovakia are mainly available at the SME / all-sector slice, so very detailed firm-size or sector filters may exclude them."
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
                                                            "cube axes",
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
                                                html.Summary("SAFE cube filters"),
                                                html.Div(
                                                    className="local-control-grid",
                                                    children=[
                                                        control(
                                                            "SAFE problem",
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
                                                        dcc.Graph(id="big-cube-trend-figure", className="chart"),
                                                        explanation(
                                                            "Figure note",
                                                            "The trend follows severity and severe-response shares for the selected SAFE problem.",
                                                            "A line chart shows whether problem pressure rises, falls, or persists through time; separate y-axes avoid mixing a 1-10 score with percentage shares.",
                                                            "Top-box and high-pressure shares are included because they are safer for ordinal survey answers than relying only on the exact mean score.",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    children=[
                                                        dcc.Graph(id="problem-bar-figure", className="chart"),
                                                        explanation(
                                                            "Figure note",
                                                            "The bar chart compares named problem categories in the large SAFE cube; the non-specific Other category is excluded.",
                                                            "Sorting supports fast comparison and reduces unnecessary visual search.",
                                                            "It shows whether access to finance is the dominant pressure or part of a broader business-stress mix.",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        dcc.Graph(id="size-problem-heatmap-figure", className="chart"),
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
                            "Methodology",
                            "methodology",
                            [
                                html.Section(
                                    className="section",
                                    children=[
                                        html.Div(className="section-heading", children=[html.H2("Validation and Methodology")]),
                                        html.Div(
                                            className="source-catalog-panel",
                                            children=[
                                                html.Div(
                                                    className="source-catalog-copy",
                                                    children=[
                                                        html.Span("Data and method inventory", className="panel-kicker"),
                                                        html.H3("What data is used, and where it enters the story"),
                                                        html.P(
                                                            "The table separates index construction from forecasting and validation. "
                                                            "This is intentional: SME-FPI Core remains a borrower-side SAFE index, while "
                                                            "macro, micro, lender-side, and loan-market variables are predictors or checks."
                                                        ),
                                                    ],
                                                ),
                                                source_catalog_table(),
                                            ],
                                        ),
                                        html.Div(
                                            className="guide-panel validation-guide",
                                            children=[
                                                html.Div(
                                                    className="guide-panel-copy",
                                                    children=[
                                                        html.H3("How to read the validation y-axis"),
                                                        html.P(
                                                            "Each row is a future target used to check whether today's SME-FPI is aligned with later stress. H+1 means one half-year ahead; H+2 means two half-years ahead."
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
                                                                    "SAFE cube",
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
                                        dcc.Graph(id="validation-figure", figure=make_validation_chart(), className="chart"),
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
                                                                html.Li("PCA weights maximize explained variance, not economic importance."),
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
    Output("kpi-row", "children"),
    Output("decision-summary-cards", "children"),
    Output("decision-narrative-container", "children"),
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
