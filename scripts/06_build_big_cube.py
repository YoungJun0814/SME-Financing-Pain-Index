from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "figures"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RAW_Q0B = RAW_DIR / "safe_q0b_pressingness_big_cube.csv"

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
    "U2": "Euro area",
    "T2": "EU aggregate",
}

PROBLEM_LABELS = {
    "P1": "Finding customers",
    "P2": "Competition",
    "P3": "Access to finance",
    "P4": "Production/labour costs",
    "P5": "Skilled staff",
    "P6": "Regulation",
    "N8": "Other",
}

SIZE_LABELS = {
    "MIC": "Micro",
    "SML": "Small",
    "MED": "Medium",
    "SME": "SME",
    "LAR": "Large",
    "ALL": "All",
}

SECTOR_LABELS = {
    "A": "All",
    "C": "Construction",
    "I": "Industry",
    "S": "Services",
    "T": "Trade",
}


def download_q0b_if_needed() -> None:
    if RAW_Q0B.exists():
        print(f"using existing {RAW_Q0B}")
        return
    parts = ["H", "", "", "", "", "", "", "Q0B", "", "", "AL", "WP"]
    key = ".".join(parts)
    url = (
        "https://data-api.ecb.europa.eu/service/data/SAFE/"
        f"{key}?format=csvdata&startPeriod=2015-S1"
    )
    response = requests.get(url, timeout=240)
    response.raise_for_status()
    RAW_Q0B.write_bytes(response.content)
    print(f"saved {RAW_Q0B} ({len(response.content):,} bytes)")


def severity_measures(group: pd.DataFrame) -> pd.Series:
    # Treat L1-L10 answers as a pragmatic descriptive scale, not exact cardinal units.
    weights = group["OBS_VALUE"].sum()
    if weights == 0 or np.isnan(weights):
        return pd.Series(
            {
                "severity_score_1_10": np.nan,
                "top_box_share_8_10": np.nan,
                "high_pressure_share_7_10": np.nan,
            }
        )
    return pd.Series(
        {
            "severity_score_1_10": float(
                (group["severity_level"] * group["OBS_VALUE"]).sum() / weights
            ),
            "top_box_share_8_10": float(
                group.loc[group["severity_level"] >= 8, "OBS_VALUE"].sum()
                / weights
                * 100
            ),
            "high_pressure_share_7_10": float(
                group.loc[group["severity_level"] >= 7, "OBS_VALUE"].sum()
                / weights
                * 100
            ),
        }
    )


def build_cube() -> None:
    raw_column_count = len(pd.read_csv(RAW_Q0B, nrows=0).columns)
    usecols = [
        "KEY",
        "REF_AREA",
        "FIRM_SIZE",
        "FIRM_SECTOR",
        "FIRM_TURNOVER",
        "FIRM_AGE",
        "FIRM_OWNERSHIP",
        "SAFE_ITEM",
        "SAFE_ANSWER",
        "TIME_PERIOD",
        "OBS_VALUE",
    ]
    df = pd.read_csv(RAW_Q0B, usecols=usecols)
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df["country_name"] = df["REF_AREA"].map(COUNTRY_NAMES)
    df["problem"] = df["SAFE_ITEM"].map(PROBLEM_LABELS)
    df["firm_size_label"] = df["FIRM_SIZE"].map(SIZE_LABELS)
    df["sector_label"] = df["FIRM_SECTOR"].map(SECTOR_LABELS)
    df["severity_level"] = (
        df["SAFE_ANSWER"].str.extract(r"^L(\d+)$")[0].astype(float)
    )

    slim = df[
        [
            "REF_AREA",
            "country_name",
            "FIRM_SIZE",
            "firm_size_label",
            "FIRM_SECTOR",
            "sector_label",
            "FIRM_AGE",
            "SAFE_ITEM",
            "problem",
            "SAFE_ANSWER",
            "severity_level",
            "TIME_PERIOD",
            "OBS_VALUE",
        ]
    ].copy()
    slim.to_csv(PROCESSED_DIR / "safe_q0b_big_cube_slim.csv", index=False)

    likert = slim.dropna(subset=["severity_level", "OBS_VALUE"]).copy()
    group_cols = [
        "REF_AREA",
        "country_name",
        "TIME_PERIOD",
        "FIRM_SIZE",
        "firm_size_label",
        "FIRM_SECTOR",
        "sector_label",
        "FIRM_AGE",
        "SAFE_ITEM",
        "problem",
    ]
    severity = (
        likert.groupby(group_cols)
        .apply(severity_measures, include_groups=False)
        .reset_index()
    )
    severity.to_csv(PROCESSED_DIR / "safe_problem_severity_cube.csv", index=False)

    access = severity[severity["SAFE_ITEM"] == "P3"].copy()
    access.to_csv(PROCESSED_DIR / "access_finance_severity_cube.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "raw_rows": len(df),
                "raw_columns": raw_column_count,
                "loaded_columns": len(df.columns),
                "raw_file_mb": RAW_Q0B.stat().st_size / 1_000_000,
                "countries": df["REF_AREA"].nunique(),
                "firm_sizes": df["FIRM_SIZE"].nunique(),
                "sectors": df["FIRM_SECTOR"].nunique(),
                "firm_ages": df["FIRM_AGE"].nunique(),
                "problem_items": df["SAFE_ITEM"].nunique(),
                "answer_levels": df["SAFE_ANSWER"].nunique(),
                "periods": df["TIME_PERIOD"].nunique(),
                "severity_rows": len(severity),
                "access_severity_rows": len(access),
            }
        ]
    )
    summary.to_csv(PROCESSED_DIR / "big_cube_review_summary.csv", index=False)
    print(summary.to_string(index=False))

    make_figures(severity, access)


def savefig(name: str) -> None:
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def make_figures(severity: pd.DataFrame, access: pd.DataFrame) -> None:
    latest_period = severity["TIME_PERIOD"].max()
    latest = severity[
        (severity["TIME_PERIOD"] == latest_period)
        & (severity["FIRM_SIZE"].isin(["MIC", "SML", "MED", "LAR"]))
        & (severity["FIRM_SECTOR"] == "A")
        & (severity["FIRM_AGE"] == 0)
        & (~severity["REF_AREA"].isin(["U2", "T2"]))
    ].copy()
    pivot = latest.pivot_table(
        index="problem",
        columns="firm_size_label",
        values="severity_score_1_10",
        aggfunc="mean",
    )
    pivot = pivot.reindex(
        [
            "Finding customers",
            "Competition",
            "Access to finance",
            "Production/labour costs",
            "Skilled staff",
            "Regulation",
            "Other",
        ]
    )[["Micro", "Small", "Medium", "Large"]]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=1, vmax=10, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9)
    ax.set_title(
        f"SAFE big cube: problem severity by firm size ({latest_period})",
        fontsize=13,
        weight="bold",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Weighted severity score, 1-10")
    savefig("09_big_cube_problem_severity_by_size.png")

    metric_specs = [
        ("top_box_share_8_10", "Top-box share, 8-10"),
        ("high_pressure_share_7_10", "High-pressure share, 7-10"),
    ]
    pivots = []
    for metric, _ in metric_specs:
        metric_pivot = latest.pivot_table(
            index="problem",
            columns="firm_size_label",
            values=metric,
            aggfunc="mean",
        )
        metric_pivot = metric_pivot.reindex(
            [
                "Finding customers",
                "Competition",
                "Access to finance",
                "Production/labour costs",
                "Skilled staff",
                "Regulation",
                "Other",
            ]
        )[["Micro", "Small", "Medium", "Large"]]
        pivots.append(metric_pivot)
    vmax = max(10, float(np.nanmax([pivot.values for pivot in pivots])))
    vmax = np.ceil(vmax / 10) * 10

    fig = plt.figure(figsize=(14.4, 6.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.035])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    cax = fig.add_subplot(grid[0, 2])
    for ax, pivot_data, (_, title) in zip(axes, pivots, metric_specs):
        im = ax.imshow(pivot_data.values, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_xticklabels(pivot_data.columns)
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_yticklabels(pivot_data.index)
        for i in range(pivot_data.shape[0]):
            for j in range(pivot_data.shape[1]):
                val = pivot_data.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=9)
        ax.set_title(title)
    axes[1].set_yticklabels([])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Share of valid 1-10 responses")
    fig.suptitle(
        f"SAFE big cube robustness: severe-response shares ({latest_period})",
        fontsize=13,
        weight="bold",
    )
    path = FIG_DIR / "09b_big_cube_topbox_highpressure.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")

    # SAFE publishes firm-size and sector breakdowns for this question, but not
    # every cross-product of both dimensions. Present them as two honest
    # marginal breakdowns instead of fabricating a size x sector matrix.
    by_size = access[
        (access["FIRM_SIZE"].isin(["MIC", "SML", "MED", "LAR"]))
        & (access["FIRM_SECTOR"] == "A")
        & (access["FIRM_AGE"] == 0)
        & (~access["REF_AREA"].isin(["U2", "T2"]))
    ].copy()
    by_sector = access[
        (access["FIRM_SIZE"] == "ALL")
        & (access["FIRM_SECTOR"].isin(["C", "I", "S", "T"]))
        & (access["FIRM_AGE"] == 0)
        & (~access["REF_AREA"].isin(["U2", "T2"]))
    ].copy()
    size_avg = (
        by_size.groupby("firm_size_label", as_index=False)["severity_score_1_10"]
        .mean()
        .set_index("firm_size_label")
        .reindex(["Micro", "Small", "Medium", "Large"])
        .reset_index()
    )
    sector_avg = (
        by_sector.groupby("sector_label", as_index=False)["severity_score_1_10"]
        .mean()
        .set_index("sector_label")
        .reindex(["Construction", "Industry", "Services", "Trade"])
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    axes[0].bar(size_avg["firm_size_label"], size_avg["severity_score_1_10"], color="#1b7f79")
    axes[0].set_title("By firm size")
    axes[0].set_ylabel("Access-to-finance severity, 1-10")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(sector_avg["sector_label"], sector_avg["severity_score_1_10"], color="#e0a526")
    axes[1].set_title("By sector")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.set_ylim(1, 10)
        for patch in ax.patches:
            height = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + 0.08,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle(
        "Access-to-finance severity: honest marginal breakdowns",
        fontsize=13,
        weight="bold",
    )
    savefig("10_access_finance_severity_marginal_breakdowns.png")

    trend = (
        access[
            (access["FIRM_SIZE"] == "SME")
            & (access["FIRM_SECTOR"] == "A")
            & (access["FIRM_AGE"] == 0)
            & (~access["REF_AREA"].isin(["U2", "T2"]))
        ]
        .groupby(["TIME_PERIOD", "country_name"], as_index=False)[
            "severity_score_1_10"
        ]
        .mean()
    )
    period_order = sorted(trend["TIME_PERIOD"].unique(), key=lambda x: int(x[:4]) * 10 + int(x[-1]))
    x_lookup = {p: i for i, p in enumerate(period_order)}
    trend["x"] = trend["TIME_PERIOD"].map(x_lookup)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for _, sub in trend.groupby("country_name"):
        ax.plot(sub["x"], sub["severity_score_1_10"], color="#9ca3af", alpha=0.55, linewidth=1)
    avg = trend.groupby("x", as_index=False)["severity_score_1_10"].mean()
    ax.plot(avg["x"], avg["severity_score_1_10"], color="#1b7f79", linewidth=2.6, label="Country average")
    ax.set_xticks(np.arange(len(period_order))[::3])
    ax.set_xticklabels(period_order[::3], rotation=45, ha="right")
    ax.set_ylabel("Access-to-finance severity, 1-10")
    ax.set_title("Big cube trend: access-to-finance problem severity", fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    savefig("11_access_finance_severity_big_cube_trend.png")


def main() -> None:
    download_q0b_if_needed()
    build_cube()
    print("Big cube build complete.")


if __name__ == "__main__":
    main()
