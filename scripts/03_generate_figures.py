from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COMPONENT_LABELS = {
    "access_finance_main_problem": "Access to finance\nmain problem",
    "bank_loan_rejected": "Loan\nrejected",
    "bank_loan_cost_too_high": "Cost too\nhigh",
    "bank_loan_limited_amount": "Limited\namount",
    "bank_willingness_deteriorated": "Bank willingness\ndeteriorated",
    "interest_rates_increased": "Interest rates\nincreased",
}

SECTOR_LABELS = {
    "A": "All",
    "C": "Construction",
    "I": "Industry",
    "S": "Services",
    "T": "Trade",
}

SIZE_LABELS = {
    "MIC": "Micro",
    "SML": "Small",
    "MED": "Medium",
    "SME": "SME",
    "LAR": "Large",
}


def savefig(name: str) -> None:
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def sparse_xticks(ax, labels, step=4):
    locs = np.arange(len(labels))
    ax.set_xticks(locs[::step])
    ax.set_xticklabels([labels[i] for i in locs[::step]], rotation=45, ha="right")


def figure_timeseries(panel: pd.DataFrame) -> None:
    avg = (
        panel.groupby(["TIME_PERIOD", "period_sort"], as_index=False)
        .agg(
            SME_FPI_equal_z=("SME_FPI_equal_z", "mean"),
            CISS_z=("CISS_z", "mean"),
        )
        .sort_values("period_sort")
    )
    x = np.arange(len(avg))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, avg["SME_FPI_equal_z"], color="#1b7f79", linewidth=2.4, label="SME Financing Pain Index")
    ax.plot(x, avg["CISS_z"], color="#9b2d20", linewidth=2.0, label="ECB New CISS benchmark")
    ax.axhline(0, color="#555555", linewidth=0.8, alpha=0.7)

    for period, label in [("2020-S1", "COVID shock"), ("2022-S1", "Inflation tightening")]:
        idx = avg.index[avg["TIME_PERIOD"] == period]
        if len(idx):
            xpos = avg.index.get_loc(idx[0])
            ax.axvspan(xpos - 0.5, xpos + 1.5, color="#d9d9d9", alpha=0.35)
            ax.text(xpos + 0.1, ax.get_ylim()[1] * 0.88, label, fontsize=9, color="#333333")

    sparse_xticks(ax, avg["TIME_PERIOD"].tolist(), step=4)
    ax.set_title("Borrower-side SME financing pain vs market-side systemic stress", fontsize=14, weight="bold")
    ax.set_ylabel("Standardized index value")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    savefig("01_sme_fpi_vs_ciss_timeseries.png")


def figure_country_heatmap(panel: pd.DataFrame) -> None:
    order = (
        panel.groupby("country_name")["SME_FPI_equal_z"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    pivot = (
        panel.pivot_table(index="country_name", columns="TIME_PERIOD", values="SME_FPI_equal_z")
        .reindex(order)
    )
    cols = sorted(pivot.columns, key=lambda x: int(x[:4]) * 10 + int(x[-1]))
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r", vmin=-2.2, vmax=2.2)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    sparse_xticks(ax, cols, step=4)
    ax.set_title("SME Financing Pain Index by country and half-year", fontsize=14, weight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("SME_FPI, standardized")
    savefig("02_country_period_heatmap.png")


def figure_hidden_gap(panel: pd.DataFrame) -> None:
    latest = (
        panel[panel["Hidden_SME_Stress"].notna()]
        .sort_values("period_sort")
        .groupby("REF_AREA", as_index=False)
        .tail(1)
        .sort_values("Hidden_SME_Stress", ascending=True)
    )
    colors = ["#1b7f79" if v < 0 else "#9b2d20" for v in latest["Hidden_SME_Stress"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(latest["country_name"], latest["Hidden_SME_Stress"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    period = latest["TIME_PERIOD"].iloc[0] if len(latest) else "latest"
    ax.set_title(f"Hidden SME stress gap by country ({period})", fontsize=14, weight="bold")
    ax.set_xlabel("SME_FPI_z minus CISS_z")
    ax.grid(axis="x", alpha=0.25)
    savefig("03_hidden_sme_stress_gap_latest.png")


def figure_pca_loadings() -> None:
    loadings = pd.read_csv(PROCESSED_DIR / "pca_loadings.csv")
    loadings["label"] = loadings["component"].map(COMPONENT_LABELS)
    loadings = loadings.sort_values("PC1_loading")
    colors = ["#9b2d20" if v > 0 else "#1b7f79" for v in loadings["PC1_loading"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(loadings["label"], loadings["PC1_loading"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title("PCA loadings: what defines borrower-side financing pain?", fontsize=14, weight="bold")
    ax.set_xlabel("PC1 loading, oriented so higher means more pain")
    ax.grid(axis="x", alpha=0.25)
    savefig("04_pca_loadings.png")


def figure_correlation_heatmap(panel: pd.DataFrame) -> None:
    cols = [
        "access_finance_main_problem",
        "bank_loan_rejected",
        "bank_loan_cost_too_high",
        "bank_loan_limited_amount",
        "bank_willingness_deteriorated",
        "interest_rates_increased",
        "SME_FPI_equal_z",
        "CISS_z",
        "Hidden_SME_Stress",
    ]
    labels = {
        "access_finance_main_problem": "Access finance\nmain problem",
        "bank_loan_rejected": "Loan\nrejected",
        "bank_loan_cost_too_high": "Cost too\nhigh",
        "bank_loan_limited_amount": "Limited\namount",
        "bank_willingness_deteriorated": "Bank willingness\ndeteriorated",
        "interest_rates_increased": "Interest rates\nincreased",
        "SME_FPI_equal_z": "SME_FPI",
        "CISS_z": "CISS",
        "Hidden_SME_Stress": "Hidden\nstress gap",
    }
    corr = panel[cols].corr(numeric_only=True)
    corr = corr.rename(index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(10.5, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation coefficient"},
        ax=ax,
    )
    ax.set_title(
        "Correlation heatmap for numeric financing-stress variables",
        fontsize=14,
        weight="bold",
    )
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    savefig("04b_numeric_correlation_heatmap.png")


def figure_cluster_scatter(panel: pd.DataFrame) -> None:
    p = panel.dropna(subset=["PC1", "PC2", "cluster_name"]).copy()
    names = sorted(p["cluster_name"].unique())
    palette = ["#1b7f79", "#e0a526", "#7d5ba6", "#9b2d20", "#3d5a80"]
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(names)}

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for name, sub in p.groupby("cluster_name"):
        ax.scatter(sub["PC1"], sub["PC2"], s=28, alpha=0.75, label=name, color=color_map[name])

    extreme = p.sort_values("SME_FPI_equal_z", ascending=False).head(6)
    for row in extreme.itertuples(index=False):
        ax.text(row.PC1, row.PC2, f"{row.REF_AREA} {row.TIME_PERIOD}", fontsize=8)

    ax.axhline(0, color="#555555", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="#555555", linewidth=0.6, alpha=0.5)
    ax.set_title("Country-period financing regimes from SME stress components", fontsize=14, weight="bold")
    ax.set_xlabel("PC1: borrower-side financing pain")
    ax.set_ylabel("PC2: secondary stress pattern")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    savefig("05_pca_cluster_regimes.png")


def figure_firm_size_component_heatmap() -> None:
    summary = pd.read_csv(PROCESSED_DIR / "sector_size_component_summary.csv")
    subset = summary[summary["FIRM_SECTOR"] == "A"].copy()
    pivot = subset.pivot_table(index="FIRM_SIZE", columns="component", values="mean_value")
    ordered_components = list(COMPONENT_LABELS.keys())
    pivot = pivot.reindex(["MIC", "SML", "MED", "SME", "LAR"])[ordered_components]

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([COMPONENT_LABELS[c] for c in pivot.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([SIZE_LABELS[i] for i in pivot.index])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9)

    ax.set_title("Financing-pain components by firm size", fontsize=13, weight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Weighted percentage of responses")
    savefig("06_firm_size_component_heatmap.png")


def figure_plotly_map(panel: pd.DataFrame) -> None:
    try:
        import plotly.express as px
    except Exception as exc:
        print(f"plotly unavailable, skipping map: {exc}")
        return

    latest = (
        panel[panel["SME_FPI_equal_z"].notna()]
        .sort_values("period_sort")
        .groupby("REF_AREA", as_index=False)
        .tail(1)
    )
    fig = px.choropleth(
        latest,
        locations="REF_AREA",
        locationmode="ISO-3",
        color="SME_FPI_equal_z",
        hover_name="country_name",
        hover_data=["TIME_PERIOD", "Hidden_SME_Stress"],
        color_continuous_scale="RdYlBu_r",
        title="Latest SME Financing Pain Index by country",
    )
    # Plotly's ISO-3 mode expects three-letter codes. Provide a small fallback mapping.
    iso3 = {
        "AT": "AUT", "BE": "BEL", "DE": "DEU", "ES": "ESP", "FI": "FIN", "FR": "FRA",
        "GR": "GRC", "IE": "IRL", "IT": "ITA", "NL": "NLD", "PT": "PRT", "SK": "SVK",
    }
    latest = latest.assign(ISO3=latest["REF_AREA"].map(iso3))
    fig = px.choropleth(
        latest,
        locations="ISO3",
        color="SME_FPI_equal_z",
        hover_name="country_name",
        hover_data=["TIME_PERIOD", "Hidden_SME_Stress"],
        color_continuous_scale="RdYlBu_r",
        title="Latest SME Financing Pain Index by country",
        scope="europe",
    )
    fig.write_html(FIG_DIR / "07_latest_sme_fpi_map.html")
    print(f"saved {FIG_DIR / '07_latest_sme_fpi_map.html'}")


def figure_general_audience_redesign(panel: pd.DataFrame) -> None:
    p = panel.dropna(subset=["SME_FPI_equal_z"]).copy()
    periods = (
        p[["TIME_PERIOD", "period_sort"]]
        .drop_duplicates()
        .sort_values("period_sort")["TIME_PERIOD"]
        .tolist()
    )
    x_lookup = {period: i for i, period in enumerate(periods)}
    p["x"] = p["TIME_PERIOD"].map(x_lookup)

    avg = p.groupby(["TIME_PERIOD", "x"], as_index=False)["SME_FPI_equal_z"].mean()
    latest_period = periods[-1]
    latest = p[p["TIME_PERIOD"] == latest_period].sort_values("SME_FPI_equal_z", ascending=False)
    highlight = latest.head(3)["REF_AREA"].tolist()
    highlighted = p[p["REF_AREA"].isin(highlight)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)

    ax = axes[0]
    for _, sub in p.groupby("REF_AREA"):
        ax.plot(sub["x"], sub["SME_FPI_equal_z"], color="#6b7280", alpha=0.55, linewidth=1.0)
    ax.set_title("Before: all countries compete for attention", fontsize=12, weight="bold")
    ax.set_ylabel("SME_FPI, standardized")
    ax.grid(axis="y", alpha=0.25)
    sparse_xticks(ax, periods, step=5)

    ax = axes[1]
    ax.plot(avg["x"], avg["SME_FPI_equal_z"], color="#111827", linewidth=2.4, label="European sample average")
    colors = ["#9b2d20", "#e0a526", "#1b7f79"]
    for color, (code, sub) in zip(colors, highlighted.groupby("REF_AREA")):
        name = sub["country_name"].iloc[0]
        ax.plot(sub["x"], sub["SME_FPI_equal_z"], color=color, linewidth=2.2)
        last = sub.sort_values("x").iloc[-1]
        ax.text(last["x"] + 0.15, last["SME_FPI_equal_z"], name, color=color, fontsize=9, va="center")
    for period, label in [("2020-S1", "COVID"), ("2022-S1", "rate shock")]:
        if period in x_lookup:
            xpos = x_lookup[period]
            ax.axvspan(xpos - 0.5, xpos + 1.5, color="#d9d9d9", alpha=0.35)
            ax.text(xpos, ax.get_ylim()[1] * 0.85, label, fontsize=9, color="#333333")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("After: guide the reader to the story", fontsize=12, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    sparse_xticks(ax, periods, step=5)
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle("General-audience redesign: from spaghetti chart to guided comparison", fontsize=14, weight="bold")
    savefig("08_general_audience_redesign.png")


def main() -> None:
    panel = pd.read_csv(PROCESSED_DIR / "sme_fpi_panel.csv")
    figure_timeseries(panel)
    figure_country_heatmap(panel)
    figure_hidden_gap(panel)
    figure_pca_loadings()
    figure_correlation_heatmap(panel)
    figure_cluster_scatter(panel)
    figure_firm_size_component_heatmap()
    figure_plotly_map(panel)
    figure_general_audience_redesign(panel)
    print("Figures complete.")


if __name__ == "__main__":
    main()
