from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

AGGREGATE_AREAS = {"U2", "T2"}
MIN_COMPONENTS_REQUIRED = 4
FIXED_BASELINE_START = "2015-S1"
FIXED_BASELINE_END = "2019-S2"

COUNTRY_NAMES = {
    "AT": "Austria",
    "BE": "Belgium",
    "DE": "Germany",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "GR": "Greece",
    "IE": "Ireland",
    "IT": "Italy",
    "NL": "Netherlands",
    "PT": "Portugal",
    "SK": "Slovakia",
}


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series * np.nan
    return (series - series.mean()) / std


def fixed_baseline_zscore(series: pd.Series, baseline_mask: pd.Series) -> pd.Series:
    baseline = series[baseline_mask]
    std = baseline.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series * np.nan
    return (series - baseline.mean()) / std


def weighted_available_average(
    frame: pd.DataFrame, columns: list[str], weights: pd.Series
) -> pd.Series:
    values = frame[columns]
    available = values.notna()
    numerator = values.fillna(0.0).mul(weights, axis=1).sum(axis=1)
    denominator = available.mul(weights, axis=1).sum(axis=1)
    result = numerator / denominator.replace(0.0, np.nan)
    result[available.sum(axis=1) < MIN_COMPONENTS_REQUIRED] = np.nan
    return result


def period_to_sort_key(period: str) -> int:
    year, half = period.split("-S")
    return int(year) * 10 + int(half)


def date_to_half_year(date: pd.Timestamp) -> str:
    half = "S1" if date.month <= 6 else "S2"
    return f"{date.year}-{half}"


def load_safe_components() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(RAW_DIR / "safe_component_metadata.csv")
    frames = []
    for row in metadata.itertuples(index=False):
        path = RAW_DIR / f"safe_{row.component}.csv"
        df = pd.read_csv(path)
        df["component"] = row.component
        df["component_label"] = row.label
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)
    long_df["OBS_VALUE"] = pd.to_numeric(long_df["OBS_VALUE"], errors="coerce")
    long_df["country_name"] = long_df["REF_AREA"].map(COUNTRY_NAMES)
    long_df["period_sort"] = long_df["TIME_PERIOD"].map(period_to_sort_key)
    return long_df, metadata


def build_main_panel(long_df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    main = long_df[
        (~long_df["REF_AREA"].isin(AGGREGATE_AREAS))
        & (long_df["FIRM_SIZE"] == "SME")
        & (long_df["FIRM_SECTOR"] == "A")
    ].copy()

    panel = (
        main.pivot_table(
            index=["REF_AREA", "country_name", "TIME_PERIOD", "period_sort"],
            columns="component",
            values="OBS_VALUE",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["REF_AREA", "period_sort"])
    )
    panel.columns.name = None

    component_cols = metadata["component"].tolist()
    for col in component_cols:
        panel[f"z_{col}"] = zscore(panel[col])

    z_cols = [f"z_{col}" for col in component_cols]
    panel["components_available"] = panel[component_cols].notna().sum(axis=1)
    panel["SME_FPI_equal_raw"] = panel[z_cols].mean(axis=1, skipna=True)
    panel.loc[
        panel["components_available"] < MIN_COMPONENTS_REQUIRED, "SME_FPI_equal_raw"
    ] = np.nan
    panel["SME_FPI_equal_z"] = zscore(panel["SME_FPI_equal_raw"])

    baseline_mask = panel["period_sort"].between(
        period_to_sort_key(FIXED_BASELINE_START),
        period_to_sort_key(FIXED_BASELINE_END),
    )
    for col in component_cols:
        panel[f"fbz_{col}"] = fixed_baseline_zscore(panel[col], baseline_mask)
    fbz_cols = [f"fbz_{col}" for col in component_cols]
    panel["SME_FPI_fixed_baseline_raw"] = panel[fbz_cols].mean(axis=1, skipna=True)
    panel.loc[
        panel["components_available"] < MIN_COMPONENTS_REQUIRED,
        "SME_FPI_fixed_baseline_raw",
    ] = np.nan
    panel["SME_FPI_fixed_baseline_z"] = fixed_baseline_zscore(
        panel["SME_FPI_fixed_baseline_raw"], baseline_mask
    )

    pca_ready = panel[panel["components_available"] >= MIN_COMPONENTS_REQUIRED].copy()
    x = pca_ready[z_cols].fillna(0.0).to_numpy()
    pca = PCA(n_components=2, random_state=0)
    scores = pca.fit_transform(x)

    loadings = pd.DataFrame(
        {
            "component": component_cols,
            "PC1_loading": pca.components_[0],
            "PC2_loading": pca.components_[1],
        }
    )

    # Orient PC1 so higher values mean more SME financing pain.
    corr = np.corrcoef(scores[:, 0], pca_ready["SME_FPI_equal_z"])[0, 1]
    sign = -1 if corr < 0 else 1
    scores[:, 0] *= sign
    loadings["PC1_loading"] *= sign

    pca_component_weights = loadings["PC1_loading"].abs()
    pca_component_weights = pca_component_weights / pca_component_weights.sum()
    pca_weights = pd.Series(
        pca_component_weights.to_numpy(),
        index=z_cols,
        dtype=float,
    )
    panel["SME_FPI_pca_weighted_raw"] = weighted_available_average(
        panel, z_cols, pca_weights
    )
    panel["SME_FPI_pca_weighted_z"] = zscore(panel["SME_FPI_pca_weighted_raw"])

    reliability_records = []
    reliability_scores = []
    for component, z_col in zip(component_cols, z_cols):
        other_cols = [col for col in z_cols if col != z_col]
        leave_one_out = panel[other_cols].mean(axis=1, skipna=True)
        valid = panel[[z_col]].join(leave_one_out.rename("leave_one_out")).dropna()
        corr = valid[z_col].corr(valid["leave_one_out"]) if len(valid) > 2 else np.nan
        coverage = panel[component].notna().mean()
        score = max(corr, 0.0) * coverage if not np.isnan(corr) else 0.0
        reliability_scores.append(score)
        reliability_records.append(
            {
                "component": component,
                "reliability_corr": corr,
                "coverage": coverage,
                "reliability_score": score,
            }
        )
    reliability_scores = np.array(reliability_scores, dtype=float)
    if reliability_scores.sum() == 0:
        reliability_scores = np.ones(len(component_cols), dtype=float)
    reliability_component_weights = reliability_scores / reliability_scores.sum()
    reliability_weights = pd.Series(
        reliability_component_weights,
        index=z_cols,
        dtype=float,
    )
    panel["SME_FPI_reliability_weighted_raw"] = weighted_available_average(
        panel, z_cols, reliability_weights
    )
    panel["SME_FPI_reliability_weighted_z"] = zscore(
        panel["SME_FPI_reliability_weighted_raw"]
    )

    pca_ready["PC1"] = scores[:, 0]
    pca_ready["PC2"] = scores[:, 1]
    pca_ready["SME_FPI_pca_z"] = zscore(pca_ready["PC1"])

    cluster_input = pca_ready[z_cols].fillna(0.0).to_numpy()
    cluster_validation = []
    for k in range(2, 9):
        candidate = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = candidate.fit_predict(cluster_input)
        cluster_validation.append(
            {
                "k": k,
                "inertia": candidate.inertia_,
                "silhouette_score": silhouette_score(cluster_input, labels),
            }
        )
    pd.DataFrame(cluster_validation).to_csv(
        PROCESSED_DIR / "cluster_validation_diagnostics.csv", index=False
    )

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    pca_ready["cluster"] = kmeans.fit_predict(cluster_input)

    panel = panel.merge(
        pca_ready[["REF_AREA", "TIME_PERIOD", "PC1", "PC2", "SME_FPI_pca_z", "cluster"]],
        on=["REF_AREA", "TIME_PERIOD"],
        how="left",
    )

    profiles = (
        pca_ready.groupby("cluster")[z_cols + ["SME_FPI_equal_z"]]
        .mean()
        .reset_index()
    )
    cluster_names = assign_cluster_names(profiles, z_cols)
    profiles["cluster_name"] = profiles["cluster"].map(cluster_names)
    panel["cluster_name"] = panel["cluster"].map(cluster_names)

    explained = pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )

    loadings.to_csv(PROCESSED_DIR / "pca_loadings.csv", index=False)
    explained.to_csv(PROCESSED_DIR / "pca_explained_variance.csv", index=False)
    profiles.to_csv(PROCESSED_DIR / "sme_fpi_cluster_profiles.csv", index=False)
    write_index_sensitivity_outputs(
        panel,
        metadata,
        loadings,
        pca_component_weights,
        reliability_records,
        reliability_component_weights,
    )

    return panel


def write_index_sensitivity_outputs(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    loadings: pd.DataFrame,
    pca_component_weights: pd.Series,
    reliability_records: list[dict[str, float]],
    reliability_component_weights: np.ndarray,
) -> None:
    component_cols = metadata["component"].tolist()
    weights = pd.DataFrame(
        {
            "component": component_cols,
            "component_label": metadata["label"].tolist(),
            "equal_weight": np.repeat(1 / len(component_cols), len(component_cols)),
            "pca_weight": pca_component_weights.to_numpy(),
            "reliability_weight": reliability_component_weights,
        }
    )
    weights = weights.merge(loadings, on="component", how="left")
    weights = weights.merge(pd.DataFrame(reliability_records), on="component", how="left")
    weights.to_csv(PROCESSED_DIR / "weighting_comparison.csv", index=False)

    variant_cols = [
        "SME_FPI_equal_z",
        "SME_FPI_fixed_baseline_z",
        "SME_FPI_pca_weighted_z",
        "SME_FPI_reliability_weighted_z",
        "SME_FPI_pca_z",
    ]
    corr_matrix = panel[variant_cols].corr()
    corr_matrix.to_csv(PROCESSED_DIR / "index_variant_correlation_matrix.csv")

    latest = (
        panel.dropna(subset=["SME_FPI_equal_z"])
        .sort_values("period_sort")
        .groupby("REF_AREA", as_index=False)
        .tail(1)
    )
    records = []
    equal = "SME_FPI_equal_z"
    for col in variant_cols:
        if col == equal:
            records.append(
                {
                    "index_version": col,
                    "non_missing_rows": int(panel[col].notna().sum()),
                    "correlation_with_equal": 1.0,
                    "mean_abs_difference_from_equal": 0.0,
                    "latest_country_rank_spearman_vs_equal": 1.0,
                }
            )
            continue
        valid = panel[[equal, col]].dropna()
        latest_valid = latest[[equal, col]].dropna()
        records.append(
            {
                "index_version": col,
                "non_missing_rows": int(panel[col].notna().sum()),
                "correlation_with_equal": valid[equal].corr(valid[col]),
                "mean_abs_difference_from_equal": (
                    valid[equal] - valid[col]
                ).abs().mean(),
                "latest_country_rank_spearman_vs_equal": latest_valid[equal]
                .rank(ascending=False)
                .corr(latest_valid[col].rank(ascending=False), method="spearman"),
            }
        )
    pd.DataFrame(records).to_csv(
        PROCESSED_DIR / "index_sensitivity_summary.csv", index=False
    )


def assign_cluster_names(profiles: pd.DataFrame, z_cols: list[str]) -> dict[int, str]:
    names: dict[int, str] = {}
    low_cluster = profiles.sort_values("SME_FPI_equal_z").iloc[0]["cluster"]
    high_cluster = profiles.sort_values("SME_FPI_equal_z").iloc[-1]["cluster"]
    for row in profiles.itertuples(index=False):
        cluster = int(row.cluster)
        if cluster == int(low_cluster):
            names[cluster] = "Low SME financing pain"
        elif cluster == int(high_cluster):
            names[cluster] = "Broad SME financing pain"
        else:
            values = {col: getattr(row, col) for col in z_cols}
            top = max(values, key=values.get)
            label = top.removeprefix("z_").replace("_", " ")
            names[cluster] = f"{label.title()} driven stress"
    return names


def build_ciss_half_year() -> pd.DataFrame:
    ciss = pd.read_csv(RAW_DIR / "ecb_ciss_new_daily.csv")
    ciss = ciss[["TIME_PERIOD", "OBS_VALUE"]].copy()
    ciss["date"] = pd.to_datetime(ciss["TIME_PERIOD"], errors="coerce")
    ciss["CISS"] = pd.to_numeric(ciss["OBS_VALUE"], errors="coerce")
    ciss = ciss.dropna(subset=["date", "CISS"])
    ciss["TIME_PERIOD"] = ciss["date"].map(date_to_half_year)
    half = (
        ciss.groupby("TIME_PERIOD", as_index=False)
        .agg(CISS_mean=("CISS", "mean"), CISS_max=("CISS", "max"))
        .sort_values("TIME_PERIOD")
    )
    half["period_sort"] = half["TIME_PERIOD"].map(period_to_sort_key)
    half["CISS_z"] = zscore(half["CISS_mean"])
    return half


def build_auxiliary_outputs(long_df: pd.DataFrame, metadata: pd.DataFrame) -> None:
    long_df.to_csv(PROCESSED_DIR / "safe_component_long.csv", index=False)

    coverage = (
        long_df.groupby(["component", "REF_AREA", "FIRM_SIZE", "FIRM_SECTOR"])
        .agg(
            observations=("OBS_VALUE", "count"),
            first_period=("TIME_PERIOD", "min"),
            last_period=("TIME_PERIOD", "max"),
        )
        .reset_index()
    )
    coverage.to_csv(PROCESSED_DIR / "coverage_summary.csv", index=False)

    sector_size = (
        long_df[
            (~long_df["REF_AREA"].isin(AGGREGATE_AREAS))
            & (long_df["FIRM_SIZE"].isin(["MIC", "SML", "MED", "SME", "LAR"]))
            & (long_df["FIRM_SECTOR"].isin(["A", "C", "I", "S", "T"]))
        ]
        .groupby(["component", "FIRM_SIZE", "FIRM_SECTOR"], as_index=False)
        .agg(mean_value=("OBS_VALUE", "mean"), observations=("OBS_VALUE", "count"))
    )
    sector_size.to_csv(PROCESSED_DIR / "sector_size_component_summary.csv", index=False)


def main() -> None:
    long_df, metadata = load_safe_components()
    build_auxiliary_outputs(long_df, metadata)

    panel = build_main_panel(long_df, metadata)
    ciss = build_ciss_half_year()

    panel = panel.merge(
        ciss[["TIME_PERIOD", "CISS_mean", "CISS_max", "CISS_z"]],
        on="TIME_PERIOD",
        how="left",
    )
    panel["Hidden_SME_Stress"] = panel["SME_FPI_equal_z"] - panel["CISS_z"]
    panel["Relative_Gap_equal"] = panel["SME_FPI_equal_z"] - panel["CISS_z"]
    panel["Relative_Gap_fixed_baseline"] = (
        panel["SME_FPI_fixed_baseline_z"] - panel["CISS_z"]
    )
    panel["Relative_Gap_pca_weighted"] = (
        panel["SME_FPI_pca_weighted_z"] - panel["CISS_z"]
    )
    panel["Relative_Gap_reliability_weighted"] = (
        panel["SME_FPI_reliability_weighted_z"] - panel["CISS_z"]
    )
    panel.to_csv(PROCESSED_DIR / "sme_fpi_panel.csv", index=False)
    panel.to_csv(PROCESSED_DIR / "sme_fpi_panel_v2.csv", index=False)
    ciss.to_csv(PROCESSED_DIR / "ciss_semiannual.csv", index=False)

    review = {
        "rows_safe_long": len(long_df),
        "rows_main_panel": len(panel),
        "countries_main_panel": panel["REF_AREA"].nunique(),
        "periods_main_panel": panel["TIME_PERIOD"].nunique(),
        "components": len(metadata),
        "rows_with_index": int(panel["SME_FPI_equal_z"].notna().sum()),
        "rows_with_pca": int(panel["SME_FPI_pca_z"].notna().sum()),
        "rows_with_fixed_baseline": int(
            panel["SME_FPI_fixed_baseline_z"].notna().sum()
        ),
        "rows_with_pca_weighted": int(panel["SME_FPI_pca_weighted_z"].notna().sum()),
        "rows_with_reliability_weighted": int(
            panel["SME_FPI_reliability_weighted_z"].notna().sum()
        ),
    }
    pd.DataFrame([review]).to_csv(PROCESSED_DIR / "build_review_summary.csv", index=False)
    print(pd.DataFrame([review]).to_string(index=False))
    print("Panel build complete.")


if __name__ == "__main__":
    main()
