from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

COUNTRY_TO_ISO3 = {
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
ISO3_TO_COUNTRY = {value: key for key, value in COUNTRY_TO_ISO3.items()}

WB_INDICATORS = {
    "gdp_growth_pct": {
        "indicator": "NY.GDP.MKTP.KD.ZG",
        "label": "GDP growth, annual %",
        "stress_direction": "lower_is_worse",
    },
    "unemployment_pct": {
        "indicator": "SL.UEM.TOTL.ZS",
        "label": "Unemployment rate, annual %",
        "stress_direction": "higher_is_worse",
    },
    "inflation_cpi_pct": {
        "indicator": "FP.CPI.TOTL.ZG",
        "label": "Inflation, consumer prices, annual %",
        "stress_direction": "higher_is_worse",
    },
    "domestic_credit_private_pct_gdp": {
        "indicator": "FS.AST.PRVT.GD.ZS",
        "label": "Domestic credit to private sector, % of GDP",
        "stress_direction": "lower_change_is_worse",
    },
}


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series * np.nan
    return (series - series.mean()) / std


def period_sort(period: str) -> int:
    year, half = period.split("-S")
    return int(year) * 10 + int(half)


def fetch_world_bank_indicator(
    indicator: str, start_year: int, end_year: int
) -> pd.DataFrame:
    countries = ";".join(COUNTRY_TO_ISO3.values())
    url = (
        f"https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
        f"?format=json&per_page=20000&date={start_year}:{end_year}"
    )
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"Unexpected World Bank response for {indicator}")
    records = []
    for row in payload[1]:
        iso3 = row.get("countryiso3code")
        if iso3 not in ISO3_TO_COUNTRY:
            continue
        records.append(
            {
                "REF_AREA": ISO3_TO_COUNTRY[iso3],
                "country_name_wb": row.get("country", {}).get("value"),
                "year": int(row["date"]),
                "value": row.get("value"),
            }
        )
    return pd.DataFrame(records)


def build_macro_annual(panel: pd.DataFrame) -> pd.DataFrame:
    raw_path = RAW_DIR / "worldbank_macro_annual.csv"
    start_year = int(panel["TIME_PERIOD"].str[:4].min())
    end_year = int(panel["TIME_PERIOD"].str[:4].max())

    if raw_path.exists():
        raw = pd.read_csv(raw_path)
    else:
        frames = []
        for variable, meta in WB_INDICATORS.items():
            df = fetch_world_bank_indicator(meta["indicator"], start_year, end_year)
            df["variable"] = variable
            df["indicator"] = meta["indicator"]
            df["label"] = meta["label"]
            frames.append(df)
        raw = pd.concat(frames, ignore_index=True)
        raw.to_csv(raw_path, index=False)

    macro = (
        raw.pivot_table(
            index=["REF_AREA", "country_name_wb", "year"],
            columns="variable",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["REF_AREA", "year"])
    )
    macro.columns.name = None
    macro["domestic_credit_private_pct_gdp_change"] = macro.groupby("REF_AREA")[
        "domestic_credit_private_pct_gdp"
    ].diff()
    return macro


def expand_macro_to_half_year(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    years = panel[["REF_AREA", "country_name", "TIME_PERIOD", "period_sort"]].copy()
    years["year"] = years["TIME_PERIOD"].str[:4].astype(int)
    out = years.merge(macro, on=["REF_AREA", "year"], how="left")

    out["macro_gdp_slowdown_z"] = zscore(-out["gdp_growth_pct"])
    out["macro_unemployment_z"] = zscore(out["unemployment_pct"])
    out["macro_inflation_z"] = zscore(out["inflation_cpi_pct"])
    out["macro_credit_contraction_z"] = zscore(
        -out["domestic_credit_private_pct_gdp_change"]
    )
    stress_cols = [
        "macro_gdp_slowdown_z",
        "macro_unemployment_z",
        "macro_inflation_z",
        "macro_credit_contraction_z",
    ]
    out["macro_stress_context_z"] = out[stress_cols].mean(axis=1, skipna=True)
    out.loc[out[stress_cols].notna().sum(axis=1) < 2, "macro_stress_context_z"] = np.nan
    out.to_csv(PROCESSED_DIR / "macro_context_panel.csv", index=False)
    return out


def build_access_validation_target() -> pd.DataFrame:
    access = pd.read_csv(PROCESSED_DIR / "access_finance_severity_cube.csv")
    target = (
        access[
            (access["FIRM_SIZE"] == "SME")
            & (access["FIRM_SECTOR"] == "A")
            & (access["FIRM_AGE"] == 0)
            & (~access["REF_AREA"].isin(["U2", "T2"]))
        ]
        .groupby(["REF_AREA", "country_name", "TIME_PERIOD"], as_index=False)
        .agg(
            access_finance_severity_1_10=("severity_score_1_10", "mean"),
            access_finance_top_box_8_10=("top_box_share_8_10", "mean"),
            access_finance_high_pressure_7_10=("high_pressure_share_7_10", "mean"),
        )
    )
    target["period_sort"] = target["TIME_PERIOD"].map(period_sort)
    return target


def add_leads(
    frame: pd.DataFrame, columns: list[str], group_col: str = "REF_AREA"
) -> pd.DataFrame:
    out = frame.sort_values([group_col, "period_sort"]).copy()
    for col in columns:
        out[f"future_{col}_1h"] = out.groupby(group_col)[col].shift(-1)
        out[f"future_{col}_2h"] = out.groupby(group_col)[col].shift(-2)
    return out


def correlation_records(
    frame: pd.DataFrame, index_cols: list[str], target_cols: list[str]
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index_col in index_cols:
        for target_col in target_cols:
            valid = frame[[index_col, target_col]].dropna()
            if len(valid) < 20:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = valid[index_col].corr(valid[target_col], method="pearson")
                spearman = valid[index_col].corr(valid[target_col], method="spearman")
            within = frame[["REF_AREA", index_col, target_col]].dropna().copy()
            if len(within) < 20:
                within_pearson = np.nan
            else:
                within[index_col] = within[index_col] - within.groupby("REF_AREA")[
                    index_col
                ].transform("mean")
                within[target_col] = within[target_col] - within.groupby("REF_AREA")[
                    target_col
                ].transform("mean")
                within_pearson = within[index_col].corr(
                    within[target_col], method="pearson"
                )
            records.append(
                {
                    "index_version": index_col,
                    "target": target_col,
                    "n": len(valid),
                    "pearson_corr": pearson,
                    "spearman_corr": spearman,
                    "within_country_pearson_corr": within_pearson,
                }
            )
    return records


def write_variable_expansion_candidates() -> None:
    records = [
        {
            "layer": "Core borrower-side SAFE",
            "variable": "Current six SME_FPI components",
            "status": "implemented",
            "use": "Baseline and alternative SME_FPI construction",
        },
        {
            "layer": "Core borrower-side SAFE",
            "variable": "Availability/collateral/other cost variables",
            "status": "candidate",
            "use": "Future index expansion if exact SAFE series are added",
        },
        {
            "layer": "Big SAFE Q0B cube",
            "variable": "Access-to-finance severity and severe-response shares",
            "status": "implemented",
            "use": "Validation target and Big Data diagnostic layer",
        },
        {
            "layer": "Market benchmark",
            "variable": "ECB New CISS",
            "status": "implemented",
            "use": "Common euro-area market-stress benchmark",
        },
        {
            "layer": "Macro context",
            "variable": "GDP growth, unemployment, inflation, private credit",
            "status": "implemented",
            "use": "External macro context and future stress validation",
        },
        {
            "layer": "Lender-side validation",
            "variable": "ECB Bank Lending Survey credit standards",
            "status": "not implemented",
            "use": "Recommended next validation extension",
        },
    ]
    pd.DataFrame(records).to_csv(
        PROCESSED_DIR / "variable_expansion_candidates.csv", index=False
    )


def build_external_validation() -> None:
    panel = pd.read_csv(PROCESSED_DIR / "sme_fpi_panel_v2.csv")
    macro_annual = build_macro_annual(panel)
    macro_panel = expand_macro_to_half_year(panel, macro_annual)
    access_target = build_access_validation_target()

    validation = panel.merge(
        macro_panel.drop(columns=["country_name"], errors="ignore"),
        on=["REF_AREA", "TIME_PERIOD", "period_sort"],
        how="left",
    )
    validation = validation.merge(
        access_target.drop(columns=["country_name"], errors="ignore"),
        on=["REF_AREA", "TIME_PERIOD", "period_sort"],
        how="left",
    )

    lead_columns = [
        "SME_FPI_equal_z",
        "SME_FPI_pca_weighted_z",
        "SME_FPI_reliability_weighted_z",
        "CISS_z",
        "macro_stress_context_z",
        "access_finance_severity_1_10",
        "access_finance_top_box_8_10",
        "access_finance_high_pressure_7_10",
    ]
    validation = add_leads(validation, lead_columns)
    validation.to_csv(PROCESSED_DIR / "external_validation_panel.csv", index=False)

    index_cols = [
        "SME_FPI_equal_z",
        "SME_FPI_pca_weighted_z",
        "SME_FPI_reliability_weighted_z",
        "SME_FPI_fixed_baseline_z",
    ]
    target_cols = [
        "future_SME_FPI_equal_z_1h",
        "future_access_finance_severity_1_10_1h",
        "future_access_finance_top_box_8_10_1h",
        "future_access_finance_high_pressure_7_10_1h",
        "future_macro_stress_context_z_1h",
        "future_CISS_z_1h",
        "future_SME_FPI_equal_z_2h",
        "future_access_finance_severity_1_10_2h",
        "future_macro_stress_context_z_2h",
    ]
    results = pd.DataFrame(correlation_records(validation, index_cols, target_cols))
    results.to_csv(PROCESSED_DIR / "validation_results.csv", index=False)

    summary = {
        "rows_external_validation_panel": len(validation),
        "columns_external_validation_panel": validation.shape[1],
        "rows_macro_context_panel": len(macro_panel),
        "macro_rows_with_stress_context": int(
            macro_panel["macro_stress_context_z"].notna().sum()
        ),
        "rows_with_access_target": int(
            validation["access_finance_severity_1_10"].notna().sum()
        ),
        "validation_result_rows": len(results),
    }
    pd.DataFrame([summary]).to_csv(
        PROCESSED_DIR / "external_validation_summary.csv", index=False
    )
    write_variable_expansion_candidates()
    print(pd.DataFrame([summary]).to_string(index=False))


def main() -> None:
    build_external_validation()
    print("External validation build complete.")


if __name__ == "__main__":
    main()
