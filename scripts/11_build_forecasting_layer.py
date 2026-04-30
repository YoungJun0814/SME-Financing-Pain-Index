from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


COMPONENTS = [
    "access_finance_main_problem",
    "bank_loan_rejected",
    "bank_loan_cost_too_high",
    "bank_loan_limited_amount",
    "bank_willingness_deteriorated",
    "interest_rates_increased",
]
Z_COMPONENTS = [f"z_{component}" for component in COMPONENTS]

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
MODEL_SUITE_KEYS = BASELINE_MODEL_KEYS + ML_MODEL_KEYS

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

COMPONENT_DRIVER_LABELS = {
    "z_access_finance_main_problem": "Access to finance is the main problem",
    "z_bank_loan_rejected": "Bank loan rejection pressure",
    "z_bank_loan_cost_too_high": "Loan cost too high",
    "z_bank_loan_limited_amount": "Limited loan amount",
    "z_bank_willingness_deteriorated": "Bank willingness deteriorated",
    "z_interest_rates_increased": "Interest rates increased",
}

ECB_OPTIONAL_SERIES = [
    {
        "file": "forecast_bls_credit_standards_sme.csv",
        "variable": "bls_credit_standards_sme",
        "source": "ECB Bank Lending Survey",
        "role": "lender-side predictor",
        "aggregation": "mean",
    },
    {
        "file": "forecast_bls_credit_standards_sme_fwd.csv",
        "variable": "bls_credit_standards_sme_fwd",
        "source": "ECB Bank Lending Survey",
        "role": "lender-side forward expectation predictor",
        "aggregation": "mean",
    },
    {
        "file": "forecast_bls_terms_conditions_sme.csv",
        "variable": "bls_terms_conditions_sme",
        "source": "ECB Bank Lending Survey",
        "role": "lender-side predictor",
        "aggregation": "mean",
    },
    {
        "file": "forecast_bls_loan_demand_sme.csv",
        "variable": "bls_loan_demand_sme",
        "source": "ECB Bank Lending Survey",
        "role": "credit demand predictor",
        "aggregation": "mean",
    },
    {
        "file": "forecast_bls_rejection_share_enterprise.csv",
        "variable": "bls_rejection_share_enterprise",
        "source": "ECB Bank Lending Survey",
        "role": "lender-side outcome predictor",
        "aggregation": "mean",
    },
    {
        "file": "forecast_mir_small_loan_rate.csv",
        "variable": "mir_small_loan_rate",
        "source": "ECB MFI Interest Rate Statistics",
        "role": "loan-cost predictor",
        "aggregation": "mean",
    },
    {
        "file": "forecast_mir_large_loan_rate.csv",
        "variable": "mir_large_loan_rate",
        "source": "ECB MFI Interest Rate Statistics",
        "role": "large-loan cost benchmark",
        "aggregation": "mean",
    },
    {
        "file": "forecast_mir_small_loan_volume.csv",
        "variable": "mir_small_loan_volume",
        "source": "ECB MFI Interest Rate Statistics",
        "role": "small-loan volume predictor",
        "aggregation": "sum",
    },
    {
        "file": "forecast_eurostat_bankruptcies_index.csv",
        "variable": "eurostat_bankruptcies_index",
        "source": "Eurostat short-term business statistics",
        "role": "real-economy business-stress predictor",
        "aggregation": "mean",
    },
    {
        "file": "forecast_eurostat_registrations_index.csv",
        "variable": "eurostat_registrations_index",
        "source": "Eurostat short-term business statistics",
        "role": "business-formation predictor",
        "aggregation": "mean",
    },
]


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series * np.nan
    return (series - series.mean()) / std


def quarter_to_half_year(period: str) -> str | None:
    if not isinstance(period, str) or "-Q" not in period:
        return None
    year, quarter = period.split("-Q")
    half = "S1" if int(quarter) <= 2 else "S2"
    return f"{year}-{half}"


def month_to_half_year(period: str) -> str | None:
    if not isinstance(period, str) or len(period) < 7:
        return None
    year = period[:4]
    month = int(period[5:7])
    half = "S1" if month <= 6 else "S2"
    return f"{year}-{half}"


def read_ecb_optional_series(spec: dict[str, str]) -> pd.DataFrame:
    path = RAW_DIR / spec["file"]
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    if raw.empty or "OBS_VALUE" not in raw.columns:
        return pd.DataFrame()
    frame = raw[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
    frame["OBS_VALUE"] = pd.to_numeric(frame["OBS_VALUE"], errors="coerce")
    if spec["file"].startswith(("forecast_bls", "forecast_eurostat")):
        frame["TIME_PERIOD"] = frame["TIME_PERIOD"].map(quarter_to_half_year)
    else:
        frame["TIME_PERIOD"] = frame["TIME_PERIOD"].map(month_to_half_year)
    frame = frame.dropna(subset=["REF_AREA", "TIME_PERIOD", "OBS_VALUE"])
    aggfunc = "sum" if spec["aggregation"] == "sum" else "mean"
    out = (
        frame.groupby(["REF_AREA", "TIME_PERIOD"], as_index=False)["OBS_VALUE"]
        .agg(aggfunc)
        .rename(columns={"OBS_VALUE": spec["variable"]})
    )
    return out


def build_optional_external_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    features = panel[["REF_AREA", "TIME_PERIOD"]].copy()
    catalog_records: list[dict[str, object]] = []
    for spec in ECB_OPTIONAL_SERIES:
        data = read_ecb_optional_series(spec)
        status = "implemented" if not data.empty else "ready, raw file missing"
        non_missing = int(data[spec["variable"]].notna().sum()) if not data.empty else 0
        if not data.empty:
            features = features.merge(data, on=["REF_AREA", "TIME_PERIOD"], how="left")
        catalog_records.append(
            {
                "layer": spec["source"],
                "dataset": spec["file"],
                "status": status,
                "role": spec["role"],
                "rows": non_missing,
                "dashboard_use": "Forecasting predictor only; not part of SME-FPI Core",
            }
        )
    if {"mir_small_loan_rate", "mir_large_loan_rate"}.issubset(features.columns):
        features["mir_rate_spread_small_large"] = (
            features["mir_small_loan_rate"] - features["mir_large_loan_rate"]
        )
    if "mir_small_loan_volume" in features.columns:
        features = features.sort_values(["REF_AREA", "TIME_PERIOD"])
        features["mir_small_loan_volume_yoy_pct"] = features.groupby("REF_AREA")[
            "mir_small_loan_volume"
        ].pct_change(2, fill_method=None) * 100
    if {"eurostat_bankruptcies_index", "eurostat_registrations_index"}.issubset(features.columns):
        features["eurostat_bankruptcy_registration_gap"] = (
            features["eurostat_bankruptcies_index"] - features["eurostat_registrations_index"]
        )
    return features, catalog_records


def build_micro_features(panel: pd.DataFrame) -> pd.DataFrame:
    severity = pd.read_csv(PROCESSED_DIR / "safe_problem_severity_cube.csv")
    base = panel[["REF_AREA", "TIME_PERIOD"]].copy()

    access_all = (
        severity[
            (severity["problem"] == "Access to finance")
            & (severity["FIRM_SIZE"] == "SME")
            & (severity["FIRM_SECTOR"] == "A")
            & (severity["FIRM_AGE"] == 0)
        ][
            [
                "REF_AREA",
                "TIME_PERIOD",
                "severity_score_1_10",
                "top_box_share_8_10",
                "high_pressure_share_7_10",
            ]
        ]
        .rename(
            columns={
                "severity_score_1_10": "micro_access_finance_severity",
                "top_box_share_8_10": "micro_access_finance_top_box",
                "high_pressure_share_7_10": "micro_access_finance_high_pressure",
            }
        )
    )
    base = base.merge(access_all, on=["REF_AREA", "TIME_PERIOD"], how="left")

    firm_size = severity[
        (severity["problem"] == "Access to finance")
        & (severity["FIRM_SECTOR"] == "A")
        & (severity["FIRM_AGE"] == 0)
        & (severity["FIRM_SIZE"].isin(["MIC", "SML", "MED", "LAR"]))
    ].copy()
    if not firm_size.empty:
        pivot = firm_size.pivot_table(
            index=["REF_AREA", "TIME_PERIOD"],
            columns="FIRM_SIZE",
            values="severity_score_1_10",
            aggfunc="mean",
        ).reset_index()
        pivot = pivot.rename(
            columns={
                "MIC": "micro_access_severity_micro_firms",
                "SML": "micro_access_severity_small_firms",
                "MED": "micro_access_severity_medium_firms",
                "LAR": "micro_access_severity_large_firms",
            }
        )
        base = base.merge(pivot, on=["REF_AREA", "TIME_PERIOD"], how="left")

    sector = severity[
        (severity["problem"] == "Access to finance")
        & (severity["FIRM_SIZE"] == "SME")
        & (severity["FIRM_AGE"] == 0)
        & (severity["FIRM_SECTOR"].isin(["C", "I", "S", "T"]))
    ].copy()
    if not sector.empty:
        pivot = sector.pivot_table(
            index=["REF_AREA", "TIME_PERIOD"],
            columns="FIRM_SECTOR",
            values="severity_score_1_10",
            aggfunc="mean",
        ).reset_index()
        pivot = pivot.rename(
            columns={
                "C": "micro_access_severity_construction",
                "I": "micro_access_severity_industry",
                "S": "micro_access_severity_services",
                "T": "micro_access_severity_trade",
            }
        )
        base = base.merge(pivot, on=["REF_AREA", "TIME_PERIOD"], how="left")

    broad = (
        severity[
            (severity["FIRM_SIZE"] == "SME")
            & (severity["FIRM_SECTOR"] == "A")
            & (severity["FIRM_AGE"] == 0)
            & (severity["problem"] != "Other")
        ]
        .groupby(["REF_AREA", "TIME_PERIOD"], as_index=False)
        .agg(
            micro_problem_pressure_mean=("severity_score_1_10", "mean"),
            micro_problem_pressure_max=("severity_score_1_10", "max"),
            micro_problem_high_pressure_mean=("high_pressure_share_7_10", "mean"),
        )
    )
    base = base.merge(broad, on=["REF_AREA", "TIME_PERIOD"], how="left")

    if {
        "micro_access_severity_micro_firms",
        "micro_access_severity_medium_firms",
    }.issubset(base.columns):
        base["micro_medium_access_gap"] = (
            base["micro_access_severity_micro_firms"]
            - base["micro_access_severity_medium_firms"]
        )
    return base


def add_lags(frame: pd.DataFrame, columns: list[str], group_col: str = "REF_AREA") -> pd.DataFrame:
    out = frame.sort_values([group_col, "period_sort"]).copy()
    lagged = {}
    for col in columns:
        lagged[f"{col}_lag1"] = out.groupby(group_col)[col].shift(1)
        lagged[f"{col}_lag2"] = out.groupby(group_col)[col].shift(2)
    if lagged:
        out = pd.concat([out, pd.DataFrame(lagged, index=out.index)], axis=1).copy()
    return out


def feature_columns(frame: pd.DataFrame, target_col: str) -> list[str]:
    excluded = {
        "REF_AREA",
        "country_name",
        "TIME_PERIOD",
        "period_sort",
        "cluster_name",
        "target_next",
        target_col,
    }
    blocked_prefixes = ("future_", "target_")
    cols = []
    for col in frame.columns:
        if col in excluded or col.startswith(blocked_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def make_forecast_estimator(model_key: str):
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


def simple_baseline_feature_columns(frame: pd.DataFrame, target_col: str) -> list[str]:
    candidates = [
        target_col,
        f"{target_col}_lag1",
        f"{target_col}_lag2",
        "CISS_z",
        "CISS_z_lag1",
        "Relative_Gap_equal",
        "Relative_Gap_equal_lag1",
        "components_available",
    ]
    return [col for col in candidates if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col])]


def baseline_predictions(
    model_key: str,
    work: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    target_col: str,
    y_train: pd.Series,
    simple_features: pd.DataFrame,
) -> np.ndarray:
    test_index = work.index[test_mask]
    fallback = work.loc[test_index, target_col].copy()
    fallback = fallback.fillna(float(y_train.mean()))

    if model_key == "naive":
        pred = fallback
    elif model_key == "country_mean":
        means = y_train.groupby(work.loc[train_mask, "REF_AREA"]).mean()
        pred = work.loc[test_index, "REF_AREA"].map(means).fillna(float(y_train.mean()))
    elif model_key == "momentum":
        lag_col = f"{target_col}_lag1"
        if lag_col in work.columns:
            pred = work.loc[test_index, target_col] + (
                work.loc[test_index, target_col] - work.loc[test_index, lag_col]
            )
            pred = pred.fillna(fallback)
        else:
            pred = fallback
    elif model_key == "country_ar1":
        train = work.loc[train_mask, ["REF_AREA", target_col]].copy()
        train["target_next"] = y_train
        train = train.dropna(subset=[target_col, "target_next"])
        global_fit = train[[target_col, "target_next"]].dropna()
        if len(global_fit) >= 4 and global_fit[target_col].nunique() > 1:
            global_slope, global_intercept = np.polyfit(global_fit[target_col], global_fit["target_next"], 1)
        else:
            global_slope, global_intercept = 1.0, 0.0
        country_models: dict[str, tuple[float, float]] = {}
        for country, subset in train.groupby("REF_AREA"):
            if len(subset) >= 4 and subset[target_col].nunique() > 1:
                slope, intercept = np.polyfit(subset[target_col], subset["target_next"], 1)
                country_models[country] = (float(slope), float(intercept))
        values = []
        for idx, row in work.loc[test_index, ["REF_AREA", target_col]].iterrows():
            current = row[target_col]
            if pd.isna(current):
                values.append(fallback.loc[idx])
                continue
            slope, intercept = country_models.get(row["REF_AREA"], (global_slope, global_intercept))
            values.append(intercept + slope * current)
        pred = pd.Series(values, index=test_index)
    elif model_key == "pooled_lag_ols":
        X_train = simple_features.loc[train_mask].copy()
        X_test = simple_features.loc[test_mask].copy()
        usable_cols = X_train.columns[X_train.notna().any(axis=0)]
        if len(usable_cols) == 0:
            pred = fallback
        else:
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
            model.fit(X_train[usable_cols], y_train)
            pred = pd.Series(model.predict(X_test[usable_cols]), index=test_index)
    else:
        raise ValueError(f"Unknown baseline model: {model_key}")

    pred = pd.Series(pred, index=test_index).replace([np.inf, -np.inf], np.nan)
    pred = pred.fillna(fallback)
    return pred.to_numpy(dtype=float)


def annotate_forecast_evaluation(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    out = metrics.copy()
    out["model_family"] = out["model_key"].map(MODEL_FAMILIES).fillna("model")
    naive_mae = (
        out[out["model_key"] == "naive"]
        .set_index("origin_period_sort")["mae"]
        .to_dict()
    )
    strongest_baseline = (
        out[out["model_key"].isin(BASELINE_MODEL_KEYS)]
        .groupby("origin_period_sort")["mae"]
        .min()
        .to_dict()
    )
    out["mae_improvement_vs_naive"] = out["origin_period_sort"].map(naive_mae) - out["mae"]
    out["mae_improvement_vs_strongest_baseline"] = (
        out["origin_period_sort"].map(strongest_baseline) - out["mae"]
    )
    out["beats_naive"] = out["mae_improvement_vs_naive"] > 0
    out["beats_strongest_baseline"] = out["mae_improvement_vs_strongest_baseline"] > 0
    return out


def evaluate_forecast_models(frame: pd.DataFrame, target_col: str) -> pd.DataFrame:
    work = frame.sort_values(["REF_AREA", "period_sort"]).copy()
    work["target_next"] = work.groupby("REF_AREA")[target_col].shift(-1)
    base_cols = feature_columns(work, target_col)
    country_dummies = pd.get_dummies(work["REF_AREA"], prefix="country")
    X_all = pd.concat([work[base_cols], country_dummies], axis=1).replace([np.inf, -np.inf], np.nan)
    simple_cols = simple_baseline_feature_columns(work, target_col)
    simple_features = pd.concat([work[simple_cols], country_dummies], axis=1).replace([np.inf, -np.inf], np.nan)
    y_all = work["target_next"]
    rows = []
    periods = sorted(work["period_sort"].unique())
    for test_period in periods[8:-1]:
        train_mask = (work["period_sort"] < test_period) & y_all.notna()
        test_mask = (work["period_sort"] == test_period) & y_all.notna()
        if train_mask.sum() < 60 or test_mask.sum() < 4:
            continue
        X_train = X_all.loc[train_mask]
        X_test = X_all.loc[test_mask]
        y_train = y_all.loc[train_mask]
        y_test = y_all.loc[test_mask]
        usable_cols = X_train.columns[X_train.notna().any(axis=0)]
        X_train = X_train[usable_cols]
        X_test = X_test[usable_cols]
        origin = work.loc[test_mask, "TIME_PERIOD"].iloc[0]
        for model_key in BASELINE_MODEL_KEYS:
            pred = baseline_predictions(
                model_key,
                work,
                train_mask,
                test_mask,
                target_col,
                y_train,
                simple_features,
            )
            features_used = 1 if model_key != "pooled_lag_ols" else len(simple_features.columns)
            rows.append(
                forecast_metric_record(
                    test_period,
                    origin,
                    model_key,
                    y_test,
                    pred,
                    int(test_mask.sum()),
                    features_used,
                )
            )
        for model_key in ML_MODEL_KEYS:
            model = make_forecast_estimator(model_key)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rows.append(
                forecast_metric_record(
                    test_period,
                    origin,
                    model_key,
                    y_test,
                    pred,
                    int(test_mask.sum()),
                    len(usable_cols),
                )
            )
    return annotate_forecast_evaluation(pd.DataFrame(rows))


def forecast_metric_record(
    origin_period_sort: int,
    origin_period: str,
    model_key: str,
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    n_test: int,
    features_used: int,
) -> dict[str, object]:
    return {
        "origin_period_sort": origin_period_sort,
        "origin_period": origin_period,
        "model_key": model_key,
        "model_label": FORECAST_MODEL_LABELS[model_key],
        "n_test": n_test,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "features_used": features_used,
    }


def next_half_year_label(period: str) -> str:
    year, half = str(period).split("-S")
    if half == "1":
        return f"{year}-S2"
    return f"{int(year) + 1}-S1"


def forecast_predictions_for_origin(
    frame: pd.DataFrame,
    target_col: str,
    origin_period_sort: int,
) -> pd.DataFrame:
    work = frame.sort_values(["REF_AREA", "period_sort"]).copy()
    work["target_next"] = work.groupby("REF_AREA")[target_col].shift(-1)
    base_cols = feature_columns(work, target_col)
    country_dummies = pd.get_dummies(work["REF_AREA"], prefix="country")
    X_all = pd.concat([work[base_cols], country_dummies], axis=1).replace([np.inf, -np.inf], np.nan)
    simple_cols = simple_baseline_feature_columns(work, target_col)
    simple_features = pd.concat([work[simple_cols], country_dummies], axis=1).replace([np.inf, -np.inf], np.nan)
    y_all = work["target_next"]
    train_mask = (work["period_sort"] < origin_period_sort) & y_all.notna()
    test_mask = work["period_sort"] == origin_period_sort
    if train_mask.sum() < 60 or test_mask.sum() == 0:
        return pd.DataFrame()

    latest = work.loc[
        test_mask,
        ["REF_AREA", "country_name", "TIME_PERIOD", "period_sort", target_col, "target_next"],
    ].copy()
    latest = latest.rename(
        columns={
            "period_sort": "origin_period_sort",
            "target_next": "actual_next_score",
        }
    )
    origin_period = str(latest["TIME_PERIOD"].iloc[0])
    target_period = next_half_year_label(origin_period)
    y_train = y_all.loc[train_mask]
    rows: list[pd.DataFrame] = []

    for model_key in BASELINE_MODEL_KEYS:
        pred = baseline_predictions(
            model_key,
            work,
            train_mask,
            test_mask,
            target_col,
            y_train,
            simple_features,
        )
        out = latest.copy()
        out["forecast_origin_period"] = origin_period
        out["forecast_target_period"] = target_period
        out["model_key"] = model_key
        out["model_label"] = FORECAST_MODEL_LABELS[model_key]
        out["model_family"] = MODEL_FAMILIES[model_key]
        out["predicted_score"] = pred
        out["predicted_delta"] = out["predicted_score"] - out[target_col]
        out["residual_scale"] = np.nan
        rows.append(out)

    X_train = X_all.loc[train_mask]
    X_test = X_all.loc[test_mask]
    usable_cols = X_train.columns[X_train.notna().any(axis=0)]
    X_train = X_train[usable_cols]
    X_test = X_test[usable_cols]
    for model_key in ML_MODEL_KEYS:
        model = make_forecast_estimator(model_key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train, y_train)
        pred = model.predict(X_test)
        train_pred = model.predict(X_train)
        residual_scale = float(np.nanstd(y_train.to_numpy(dtype=float) - train_pred))
        out = latest.copy()
        out["forecast_origin_period"] = origin_period
        out["forecast_target_period"] = target_period
        out["model_key"] = model_key
        out["model_label"] = FORECAST_MODEL_LABELS[model_key]
        out["model_family"] = MODEL_FAMILIES[model_key]
        out["predicted_score"] = pred
        out["predicted_delta"] = out["predicted_score"] - out[target_col]
        out["residual_scale"] = residual_scale
        rows.append(out)

    predictions = pd.concat(rows, ignore_index=True)
    predictions = predictions.rename(columns={target_col: "current_score"})
    predictions["actual_delta"] = predictions["actual_next_score"] - predictions["current_score"]
    predictions["forecast_error"] = predictions["predicted_score"] - predictions["actual_next_score"]
    predictions["forecast_abs_error"] = predictions["forecast_error"].abs()
    return predictions


def risk_tier(score: float) -> str:
    if score >= 5:
        return "Alert"
    if score >= 3:
        return "Watch"
    if score >= 1.5:
        return "Monitor"
    return "Normal"


def confidence_label(model_agreement: float, forecast_range: float, components_available: float) -> str:
    agreement_strength = max(model_agreement, 1 - model_agreement) if pd.notna(model_agreement) else 0
    if components_available >= 5 and agreement_strength >= 0.75 and forecast_range <= 0.55:
        return "High"
    if components_available >= 4 and agreement_strength >= 0.60 and forecast_range <= 0.85:
        return "Medium"
    return "Low"


def driver_text(row: pd.Series, latest_context: pd.DataFrame) -> str:
    drivers: list[tuple[float, str]] = []
    for col, label in COMPONENT_DRIVER_LABELS.items():
        if col in row.index and pd.notna(row[col]):
            drivers.append((float(row[col]), label))
    context_labels = {
        "micro_access_finance_severity": "SAFE cube access-finance severity",
        "micro_problem_pressure_mean": "Broad SAFE problem pressure",
        "bls_credit_standards_sme": "BLS credit standards tightening",
        "bls_terms_conditions_sme": "BLS terms and conditions pressure",
        "mir_small_loan_rate": "MIR small-loan rate pressure",
        "mir_rate_spread_small_large": "Small-loan spread pressure",
        "eurostat_bankruptcies_index": "Eurostat bankruptcy index pressure",
        "eurostat_bankruptcy_registration_gap": "Eurostat bankruptcy-registration gap",
    }
    for col, label in context_labels.items():
        if col in row.index and col in latest_context.columns and pd.notna(row[col]):
            series = pd.to_numeric(latest_context[col], errors="coerce")
            std = series.std(ddof=0)
            if pd.notna(std) and std > 0:
                score = (float(row[col]) - float(series.mean())) / float(std)
                if score > 0:
                    drivers.append((score, label))
    if pd.notna(row.get("gap_value", np.nan)) and row["gap_value"] >= 0.35:
        drivers.append((float(row["gap_value"]) + 0.5, "SME-FPI is high relative to CISS"))
    if pd.notna(row.get("best_model_delta", np.nan)) and row["best_model_delta"] >= 0.10:
        drivers.append((float(row["best_model_delta"]) + 0.5, "Best recent ML model expects rising H+1 pressure"))
    if not drivers:
        return "No dominant warning driver; monitor data coverage and next SAFE release."
    ordered = [label for _, label in sorted(drivers, reverse=True)[:3]]
    return "; ".join(ordered)


def build_decision_board(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    best_ml_key: str,
    target_col: str,
    origin_period_sort: int | None = None,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    if origin_period_sort is None:
        if "origin_period_sort" in predictions.columns and predictions["origin_period_sort"].notna().any():
            origin_period_sort = int(predictions["origin_period_sort"].dropna().iloc[0])
        else:
            origin_period_sort = int(frame["period_sort"].max())
    latest = frame[frame["period_sort"] == origin_period_sort].copy()
    latest["origin_period_sort"] = int(origin_period_sort)
    latest = latest.rename(columns={target_col: "current_score", "Relative_Gap_equal": "gap_value"})

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
    best_columns = [
        "REF_AREA",
        "predicted_score",
        "predicted_delta",
        "model_label",
        "forecast_target_period",
        "actual_next_score",
        "actual_delta",
        "forecast_abs_error",
    ]
    best_columns = [col for col in best_columns if col in predictions.columns]
    best = predictions[predictions["model_key"] == best_ml_key][best_columns].rename(
        columns={
            "predicted_score": "best_model_forecast",
            "predicted_delta": "best_model_delta",
            "model_label": "best_model_label",
        }
    )
    board = latest.merge(ml_summary, on=["REF_AREA", "country_name"], how="left").merge(best, on="REF_AREA", how="left")
    board["forecast_range"] = board["ml_max_forecast"] - board["ml_min_forecast"]
    board["forecast_direction"] = np.where(board["best_model_delta"] >= 0, "Rising", "Easing")
    if {"actual_next_score", "actual_delta"}.issubset(board.columns):
        board["realized_direction"] = np.where(board["actual_delta"] >= 0, "Rising", "Easing")
        board.loc[board["actual_delta"].isna(), "realized_direction"] = np.nan
        board["forecast_hit_direction"] = np.where(
            board["actual_delta"].notna(),
            np.sign(board["best_model_delta"]) == np.sign(board["actual_delta"]),
            np.nan,
        )

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
    board["primary_drivers"] = board.apply(lambda row: driver_text(row, board), axis=1)
    board["recommended_read"] = np.select(
        [
            board["risk_tier"].eq("Alert"),
            board["risk_tier"].eq("Watch"),
            board["risk_tier"].eq("Monitor"),
        ],
        [
            "Immediate watchlist: inspect drivers and compare with local SME policy context.",
            "Watch closely: pressure is visible or forecast to rise.",
            "Monitor: at least one warning signal is present.",
        ],
        default="Normal: no major warning signal in the current diagnostic rules.",
    )
    ordered_cols = [
        "REF_AREA",
        "country_name",
        "TIME_PERIOD",
        "origin_period_sort",
        "forecast_target_period",
        "risk_tier",
        "risk_score",
        "confidence",
        "current_score",
        "gap_value",
        "best_model_label",
        "best_model_forecast",
        "best_model_delta",
        "actual_next_score",
        "actual_delta",
        "forecast_abs_error",
        "forecast_hit_direction",
        "realized_direction",
        "forecast_direction",
        "ml_median_forecast",
        "ml_min_forecast",
        "ml_max_forecast",
        "forecast_range",
        "ml_model_agreement_rising",
        "ml_model_count",
        "components_available",
        "primary_drivers",
        "recommended_read",
    ]
    board = board[[col for col in ordered_cols if col in board.columns]]
    tier_order = {"Alert": 0, "Watch": 1, "Monitor": 2, "Normal": 3}
    board["_tier_order"] = board["risk_tier"].map(tier_order)
    board = board.sort_values(["_tier_order", "risk_score", "current_score"], ascending=[True, False, False])
    return board.drop(columns=["_tier_order"])


def build_all_forecast_predictions(frame: pd.DataFrame, target_col: str) -> pd.DataFrame:
    periods = sorted(frame["period_sort"].dropna().unique())
    prediction_frames = []
    for origin_period_sort in periods[8:]:
        predictions = forecast_predictions_for_origin(frame, target_col, int(origin_period_sort))
        if not predictions.empty:
            prediction_frames.append(predictions)
    if not prediction_frames:
        return pd.DataFrame()
    return pd.concat(prediction_frames, ignore_index=True)


def build_all_decision_boards(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    best_ml_key: str,
    target_col: str,
) -> pd.DataFrame:
    if predictions.empty or "origin_period_sort" not in predictions.columns:
        return pd.DataFrame()
    boards = []
    for origin_period_sort, subset in predictions.groupby("origin_period_sort"):
        board = build_decision_board(
            frame,
            subset.copy(),
            best_ml_key,
            target_col,
            origin_period_sort=int(origin_period_sort),
        )
        if not board.empty:
            boards.append(board)
    if not boards:
        return pd.DataFrame()
    return pd.concat(boards, ignore_index=True)


def build_forecast_country_error(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "actual_next_score" not in predictions.columns:
        return pd.DataFrame()
    valid = predictions[predictions["actual_next_score"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["direction_hit"] = np.sign(valid["predicted_delta"]) == np.sign(valid["actual_delta"])
    grouped = (
        valid.groupby(["REF_AREA", "country_name", "model_key", "model_label", "model_family"], as_index=False)
        .agg(
            n_origins=("origin_period_sort", "nunique"),
            mean_abs_error=("forecast_abs_error", "mean"),
            rmse=("forecast_error", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            bias=("forecast_error", "mean"),
            direction_hit_share=("direction_hit", "mean"),
            mean_predicted_delta=("predicted_delta", "mean"),
            mean_actual_delta=("actual_delta", "mean"),
        )
        .sort_values(["model_family", "mean_abs_error", "country_name"])
    )
    return grouped


def build_risk_tier_validation(decision_history: pd.DataFrame) -> pd.DataFrame:
    if decision_history.empty or "actual_next_score" not in decision_history.columns:
        return pd.DataFrame()
    valid = decision_history[decision_history["actual_next_score"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    grouped = (
        valid.groupby("risk_tier", as_index=False)
        .agg(
            n_country_origins=("REF_AREA", "count"),
            n_countries=("REF_AREA", "nunique"),
            n_origins=("origin_period_sort", "nunique"),
            mean_risk_score=("risk_score", "mean"),
            mean_current_score=("current_score", "mean"),
            mean_actual_next_score=("actual_next_score", "mean"),
            mean_actual_delta=("actual_delta", "mean"),
            pressure_rose_share=("actual_delta", lambda values: float(np.mean(values > 0))),
            direction_hit_share=("forecast_hit_direction", "mean"),
            mean_forecast_abs_error=("forecast_abs_error", "mean"),
        )
    )
    tier_order = {"Alert": 0, "Watch": 1, "Monitor": 2, "Normal": 3}
    grouped["_tier_order"] = grouped["risk_tier"].map(tier_order)
    return grouped.sort_values("_tier_order").drop(columns="_tier_order")


def write_source_catalog(
    panel: pd.DataFrame,
    external_validation: pd.DataFrame,
    optional_records: list[dict[str, object]],
) -> None:
    raw_safe_rows = len(pd.read_csv(PROCESSED_DIR / "safe_component_long.csv"))
    big_cube_rows = len(pd.read_csv(PROCESSED_DIR / "safe_q0b_big_cube_slim.csv", usecols=["REF_AREA"]))
    macro_rows = len(pd.read_csv(PROCESSED_DIR / "macro_context_panel.csv"))
    records = [
        {
            "layer": "SME-FPI Core",
            "dataset": "ECB SAFE component series",
            "status": "implemented",
            "role": "borrower-side index construction",
            "rows": raw_safe_rows,
            "dashboard_use": "Builds the six-component SME-FPI Core",
        },
        {
            "layer": "Micro diagnostic layer",
            "dataset": "ECB SAFE Q0B problem-severity cube",
            "status": "implemented",
            "role": "firm-size, sector, and problem pressure predictors",
            "rows": big_cube_rows,
            "dashboard_use": "Forecasting features, Big Data diagnostics, validation targets",
        },
        {
            "layer": "Macro context",
            "dataset": "World Bank annual indicators",
            "status": "implemented",
            "role": "macro pressure predictors and validation context",
            "rows": macro_rows,
            "dashboard_use": "Forecasting predictor only; not part of SME-FPI Core",
        },
        {
            "layer": "Market benchmark",
            "dataset": "ECB New CISS",
            "status": "implemented",
            "role": "financial-market stress benchmark",
            "rows": len(panel),
            "dashboard_use": "Gap analysis, validation, and forecasting predictor",
        },
        *optional_records,
        {
            "layer": "Business-cycle expectations",
            "dataset": "European Commission Business and Consumer Surveys",
            "status": "planned next",
            "role": "monthly confidence and expectations predictors",
            "rows": 0,
            "dashboard_use": "Forecasting predictor once pulled",
        },
    ]
    pd.DataFrame(records).to_csv(PROCESSED_DIR / "dashboard_source_catalog.csv", index=False)


def build_forecasting_layer() -> None:
    panel = pd.read_csv(PROCESSED_DIR / "sme_fpi_panel_v2.csv")
    external_validation = pd.read_csv(PROCESSED_DIR / "external_validation_panel.csv")
    micro = build_micro_features(panel)
    optional, optional_records = build_optional_external_features(panel)

    macro_cols = [
        "REF_AREA",
        "TIME_PERIOD",
        "gdp_growth_pct",
        "unemployment_pct",
        "inflation_cpi_pct",
        "domestic_credit_private_pct_gdp_change",
        "macro_gdp_slowdown_z",
        "macro_unemployment_z",
        "macro_inflation_z",
        "macro_credit_contraction_z",
        "macro_stress_context_z",
    ]
    macro_cols = [col for col in macro_cols if col in external_validation.columns]
    frame = panel.merge(
        external_validation[macro_cols],
        on=["REF_AREA", "TIME_PERIOD"],
        how="left",
    )
    frame = frame.merge(micro, on=["REF_AREA", "TIME_PERIOD"], how="left")
    frame = frame.merge(optional, on=["REF_AREA", "TIME_PERIOD"], how="left")

    predictor_cols = [
        "SME_FPI_equal_z",
        "SME_FPI_fixed_baseline_z",
        "SME_FPI_pca_weighted_z",
        "SME_FPI_reliability_weighted_z",
        "CISS_z",
        "Relative_Gap_equal",
        "Relative_Gap_fixed_baseline",
        "Relative_Gap_pca_weighted",
        "Relative_Gap_reliability_weighted",
        "PC1",
        "PC2",
        "components_available",
        *Z_COMPONENTS,
    ]
    extra_numeric = [
        col
        for col in frame.columns
        if (
            col.startswith("macro_")
            or col.startswith("micro_")
            or col.startswith("bls_")
            or col.startswith("mir_")
            or col.startswith("eurostat_")
        )
    ]
    lag_cols = [col for col in predictor_cols + extra_numeric if col in frame.columns]
    frame = add_lags(frame, lag_cols)
    frame["macro_micro_predictor_count"] = frame[
        [col for col in extra_numeric if col in frame.columns]
    ].notna().sum(axis=1)

    target_columns = {}
    for col in [
        "SME_FPI_equal_z",
        "micro_access_finance_severity",
        "macro_stress_context_z",
        "CISS_z",
    ]:
        if col in frame.columns:
            target_columns[f"target_next_{col}"] = frame.groupby("REF_AREA")[col].shift(-1)
    if target_columns:
        frame = pd.concat([frame, pd.DataFrame(target_columns)], axis=1).copy()

    frame.to_csv(PROCESSED_DIR / "forecasting_feature_panel.csv", index=False)
    evaluation = evaluate_forecast_models(frame, "SME_FPI_equal_z")
    evaluation.to_csv(PROCESSED_DIR / "forecasting_model_evaluation.csv", index=False)
    write_source_catalog(panel, external_validation, optional_records)

    recent_eval = (
        evaluation.sort_values("origin_period_sort")
        .groupby("model_key", as_index=False)
        .tail(5)
    )
    recent_scores = (
        recent_eval.groupby(["model_key", "model_label"], as_index=False)
        .agg(recent_mae=("mae", "mean"), recent_rmse=("rmse", "mean"))
        .sort_values("recent_mae")
    )
    ml_recent = recent_scores[recent_scores["model_key"] != "naive"].copy()
    ml_recent = ml_recent[ml_recent["model_key"].isin(ML_MODEL_KEYS)].copy()
    best_row = ml_recent.iloc[0] if not ml_recent.empty else recent_scores.iloc[0]
    score_lookup = recent_scores.set_index("model_key")
    strongest_baseline = recent_scores[recent_scores["model_key"].isin(BASELINE_MODEL_KEYS)].sort_values("recent_mae").iloc[0]

    latest_origin = int(frame["period_sort"].max())
    all_predictions = build_all_forecast_predictions(frame, "SME_FPI_equal_z")
    all_predictions.to_csv(PROCESSED_DIR / "forecast_model_predictions_all_origins.csv", index=False)
    latest_predictions = all_predictions[all_predictions["origin_period_sort"] == latest_origin].copy()
    if latest_predictions.empty:
        latest_predictions = forecast_predictions_for_origin(frame, "SME_FPI_equal_z", latest_origin)
    latest_predictions.to_csv(PROCESSED_DIR / "latest_forecast_model_predictions.csv", index=False)
    decision_history = build_all_decision_boards(
        frame,
        all_predictions,
        str(best_row["model_key"]),
        "SME_FPI_equal_z",
    )
    decision_history.to_csv(PROCESSED_DIR / "decision_board_all_origins.csv", index=False)
    if not decision_history.empty:
        decision_board = decision_history[decision_history["origin_period_sort"] == latest_origin].copy()
    else:
        decision_board = build_decision_board(
            frame,
            latest_predictions,
            str(best_row["model_key"]),
            "SME_FPI_equal_z",
            origin_period_sort=latest_origin,
        )
    decision_board.to_csv(PROCESSED_DIR / "forecast_decision_board.csv", index=False)
    country_error = build_forecast_country_error(all_predictions)
    country_error.to_csv(PROCESSED_DIR / "forecasting_country_error.csv", index=False)
    tier_validation = build_risk_tier_validation(decision_history)
    tier_validation.to_csv(PROCESSED_DIR / "risk_tier_validation.csv", index=False)

    summary = {
        "rows_forecasting_feature_panel": len(frame),
        "columns_forecasting_feature_panel": frame.shape[1],
        "macro_micro_predictor_columns": len(extra_numeric),
        "optional_external_predictor_columns": len(
            [col for col in frame.columns if col.startswith(("bls_", "mir_", "eurostat_"))]
        ),
        "backtest_rows": len(evaluation),
        "forecast_prediction_rows_all_origins": len(all_predictions),
        "decision_history_rows": len(decision_history),
        "country_error_rows": len(country_error),
        "risk_tier_validation_rows": len(tier_validation),
        "best_recent_model_key": best_row["model_key"],
        "best_recent_model_label": best_row["model_label"],
        "best_recent_model_mae": best_row["recent_mae"],
        "strongest_recent_baseline_key": strongest_baseline["model_key"],
        "strongest_recent_baseline_label": strongest_baseline["model_label"],
        "strongest_recent_baseline_mae": strongest_baseline["recent_mae"],
        "best_recent_ml_improvement_vs_strongest_baseline": strongest_baseline["recent_mae"] - best_row["recent_mae"],
        "latest_decision_board_rows": len(decision_board),
        "latest_decision_alert_count": int((decision_board["risk_tier"] == "Alert").sum()) if not decision_board.empty else 0,
        "latest_decision_watch_count": int((decision_board["risk_tier"] == "Watch").sum()) if not decision_board.empty else 0,
        "latest_decision_monitor_count": int((decision_board["risk_tier"] == "Monitor").sum()) if not decision_board.empty else 0,
        "latest_backtest_naive_mae": score_lookup.loc["naive", "recent_mae"] if "naive" in score_lookup.index else np.nan,
        "latest_backtest_country_mean_mae": score_lookup.loc["country_mean", "recent_mae"] if "country_mean" in score_lookup.index else np.nan,
        "latest_backtest_country_ar1_mae": score_lookup.loc["country_ar1", "recent_mae"] if "country_ar1" in score_lookup.index else np.nan,
        "latest_backtest_momentum_mae": score_lookup.loc["momentum", "recent_mae"] if "momentum" in score_lookup.index else np.nan,
        "latest_backtest_pooled_lag_ols_mae": score_lookup.loc["pooled_lag_ols", "recent_mae"] if "pooled_lag_ols" in score_lookup.index else np.nan,
        "latest_backtest_elastic_net_mae": score_lookup.loc["elastic_net", "recent_mae"] if "elastic_net" in score_lookup.index else np.nan,
        "latest_backtest_ridge_mae": score_lookup.loc["ridge", "recent_mae"] if "ridge" in score_lookup.index else np.nan,
        "latest_backtest_rf_mae": score_lookup.loc["random_forest", "recent_mae"] if "random_forest" in score_lookup.index else np.nan,
        "latest_backtest_gradient_boosting_mae": (
            score_lookup.loc["gradient_boosting", "recent_mae"]
            if "gradient_boosting" in score_lookup.index
            else np.nan
        ),
    }
    pd.DataFrame([summary]).to_csv(PROCESSED_DIR / "forecasting_layer_summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


def main() -> None:
    build_forecasting_layer()
    print("Forecasting layer build complete.")


if __name__ == "__main__":
    main()
