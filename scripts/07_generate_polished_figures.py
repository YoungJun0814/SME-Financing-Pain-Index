from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

START_PERIOD = "2015-S1"

INK = "#1f2933"
MUTED = "#64748b"
GRID = "#e5e7eb"
TEAL = "#147c78"
RED = "#a33a2a"
GOLD = "#d99a1e"
BLUE = "#3b6ea8"
PURPLE = "#7663a6"
SLATE = "#475569"
LIGHT_BG = "#fbfaf7"

COMPONENT_LABELS = {
    "access_finance_main_problem": "Access to finance\nmain problem",
    "bank_loan_rejected": "Loan\nrejected",
    "bank_loan_cost_too_high": "Cost too\nhigh",
    "bank_loan_limited_amount": "Limited\namount",
    "bank_willingness_deteriorated": "Bank willingness\ndeteriorated",
    "interest_rates_increased": "Interest rates\nincreased",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d1d5db",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": False,
            "savefig.facecolor": "white",
        }
    )


def period_sort(period: str) -> int:
    return int(period[:4]) * 10 + int(period[-1])


def filter_periods(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["period_sort"] >= period_sort(START_PERIOD)].copy()


def x_positions(periods: list[str]) -> dict[str, int]:
    ordered = sorted(periods, key=period_sort)
    return {period: i for i, period in enumerate(ordered)}


def add_period_ticks(ax, periods: list[str], step: int = 2) -> None:
    ordered = sorted(periods, key=period_sort)
    locs = np.arange(len(ordered))
    labels = [p.replace("-S", " H") for p in ordered]
    ax.set_xticks(locs[::step])
    ax.set_xticklabels([labels[i] for i in locs[::step]], rotation=0, ha="center")


def add_source_note(fig, text: str = "Source: ECB SAFE and ECB CISS; calculations by author.") -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8, color=MUTED)


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.grid(axis="y", color=GRID, linewidth=0.8)


def savefig(name: str) -> None:
    path = FIG_DIR / name
    plt.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close()
    print(f"saved {path}")


def fig_timeseries(panel: pd.DataFrame) -> None:
    p = filter_periods(panel)
    avg = (
        p.groupby(["TIME_PERIOD", "period_sort"], as_index=False)
        .agg(SME_FPI_equal_z=("SME_FPI_equal_z", "mean"), CISS_z=("CISS_z", "mean"))
        .sort_values("period_sort")
    )
    periods = avg["TIME_PERIOD"].tolist()
    x = np.arange(len(periods))

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.plot(x, avg["SME_FPI_equal_z"], color=TEAL, linewidth=2.8, label="SME Financing Pain Index")
    ax.plot(x, avg["CISS_z"], color=RED, linewidth=2.4, label="ECB New CISS")
    ax.fill_between(x, avg["SME_FPI_equal_z"], avg["CISS_z"], color="#94a3b8", alpha=0.12)

    for period, label in [("2020-S1", "COVID shock"), ("2022-S1", "Rate tightening")]:
        if period in periods:
            xpos = periods.index(period)
            ax.axvspan(xpos - 0.45, xpos + 1.45, color="#f1f5f9", zorder=0)
            ax.text(xpos + 0.05, 1.55, label, fontsize=9, color=SLATE)

    ax.axhline(0, color="#94a3b8", linewidth=1)
    add_period_ticks(ax, periods, step=2)
    ax.set_ylabel("Standardized value")
    ax.set_title("Borrower-side SME stress does not always move with market stress")
    ax.legend(frameon=False, loc="upper right")
    clean_axes(ax)
    add_source_note(fig)
    savefig("01_sme_fpi_vs_ciss_timeseries.png")


def fig_country_heatmap(panel: pd.DataFrame) -> None:
    p = filter_periods(panel)
    order = (
        p.groupby("country_name")["SME_FPI_equal_z"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    pivot = p.pivot_table(index="country_name", columns="TIME_PERIOD", values="SME_FPI_equal_z").reindex(order)
    cols = sorted(pivot.columns, key=period_sort)
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(12, 6.1))
    sns.heatmap(
        pivot,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0,
        vmin=-1.8,
        vmax=1.8,
        linewidths=0.25,
        linecolor="#f8fafc",
        cbar_kws={"label": "SME_FPI, standardized", "shrink": 0.85},
        ax=ax,
    )
    ax.set_title("Where and when SME financing pain is concentrated")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([c.replace("-S", " H") for c in cols], rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=9)
    add_source_note(fig)
    savefig("02_country_period_heatmap.png")


def fig_hidden_gap(panel: pd.DataFrame) -> None:
    p = filter_periods(panel)
    latest = (
        p[p["Hidden_SME_Stress"].notna()]
        .sort_values("period_sort")
        .groupby("REF_AREA", as_index=False)
        .tail(1)
        .sort_values("Hidden_SME_Stress")
    )
    colors = [TEAL if v < 0 else RED for v in latest["Hidden_SME_Stress"]]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    bars = ax.barh(latest["country_name"], latest["Hidden_SME_Stress"], color=colors, height=0.64)
    ax.axvline(0, color="#94a3b8", linewidth=1)
    for bar, value in zip(bars, latest["Hidden_SME_Stress"]):
        offset = 0.03 if value >= 0 else -0.03
        ha = "left" if value >= 0 else "right"
        ax.text(value + offset, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", ha=ha, fontsize=8, color=INK)
    period = latest["TIME_PERIOD"].iloc[0]
    ax.set_title(f"Relative SME-CISS gap in {period.replace('-S', ' H')}")
    ax.set_xlabel("SME_FPI_z - common CISS_z")
    clean_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    add_source_note(fig)
    savefig("03_hidden_sme_stress_gap_latest.png")


def fig_pca_loadings() -> None:
    loadings = pd.read_csv(PROCESSED_DIR / "pca_loadings.csv")
    loadings["label"] = loadings["component"].map(COMPONENT_LABELS)
    loadings = loadings.sort_values("PC1_loading")
    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    bars = ax.barh(loadings["label"], loadings["PC1_loading"], color=TEAL, height=0.62)
    for bar, value in zip(bars, loadings["PC1_loading"]):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=8, color=INK)
    ax.set_title("The index is mainly driven by access barriers and bank willingness")
    ax.set_xlabel("PC1 loading")
    ax.set_xlim(0, max(loadings["PC1_loading"]) + 0.08)
    clean_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    add_source_note(fig, "Source: ECB SAFE; PCA calculated by author.")
    savefig("04_pca_loadings.png")


def fig_correlation(panel: pd.DataFrame) -> None:
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
        "access_finance_main_problem": "Access finance",
        "bank_loan_rejected": "Loan rejected",
        "bank_loan_cost_too_high": "Cost too high",
        "bank_loan_limited_amount": "Limited amount",
        "bank_willingness_deteriorated": "Bank willingness",
        "interest_rates_increased": "Interest rates",
        "SME_FPI_equal_z": "SME_FPI",
        "CISS_z": "CISS",
        "Hidden_SME_Stress": "Hidden gap",
    }
    p = filter_periods(panel)
    corr = p[cols].corr(numeric_only=True).rename(index=labels, columns=labels)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(9, 7.2))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Numeric variables move together, but CISS remains distinct")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    add_source_note(fig)
    savefig("04b_numeric_correlation_heatmap.png")


def fig_clusters(panel: pd.DataFrame) -> None:
    p = filter_periods(panel).dropna(subset=["PC1", "PC2", "cluster_name"]).copy()
    names = [
        "Low SME financing pain",
        "Interest Rates Increased driven stress",
        "Bank Loan Cost Too High driven stress",
        "Broad SME financing pain",
    ]
    palette = {
        "Low SME financing pain": "#4f9f7f",
        "Interest Rates Increased driven stress": GOLD,
        "Bank Loan Cost Too High driven stress": PURPLE,
        "Broad SME financing pain": RED,
    }
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    for name in names:
        sub = p[p["cluster_name"] == name]
        if len(sub) == 0:
            continue
        ax.scatter(sub["PC1"], sub["PC2"], s=32, alpha=0.74, label=name, color=palette[name], edgecolor="white", linewidth=0.35)

    extremes = p.sort_values("SME_FPI_equal_z", ascending=False).head(5)
    for row in extremes.itertuples(index=False):
        ax.text(row.PC1 + 0.05, row.PC2 + 0.05, f"{row.REF_AREA} {row.TIME_PERIOD}", fontsize=8, color=INK)

    ax.axhline(0, color="#cbd5e1", linewidth=0.9)
    ax.axvline(0, color="#cbd5e1", linewidth=0.9)
    ax.set_title("Country-periods fall into distinct financing regimes")
    ax.set_xlabel("PC1: general SME financing pain")
    ax.set_ylabel("PC2: secondary stress pattern")
    clean_axes(ax)
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=8)
    add_source_note(fig, "Source: ECB SAFE; PCA and k-means calculated by author.")
    savefig("05_pca_cluster_regimes.png")


def fig_size_component() -> None:
    summary = pd.read_csv(PROCESSED_DIR / "sector_size_component_summary.csv")
    subset = summary[summary["FIRM_SECTOR"] == "A"].copy()
    pivot = subset.pivot_table(index="FIRM_SIZE", columns="component", values="mean_value")
    pivot = pivot.reindex(["MIC", "SML", "MED", "SME", "LAR"])[list(COMPONENT_LABELS.keys())]
    pivot.index = ["Micro", "Small", "Medium", "SME", "Large"]
    pivot.columns = [COMPONENT_LABELS[c].replace("\n", " ") for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(11.2, 5.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrBr",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Weighted percentage of responses", "shrink": 0.78},
        ax=ax,
    )
    ax.set_title("Financing-pain components vary strongly by firm size")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    add_source_note(fig, "Source: ECB SAFE; calculations by author.")
    savefig("06_firm_size_component_heatmap.png")


def fig_redesign(panel: pd.DataFrame) -> None:
    p = filter_periods(panel.dropna(subset=["SME_FPI_equal_z"]).copy())
    periods = sorted(p["TIME_PERIOD"].unique(), key=period_sort)
    lookup = {period: i for i, period in enumerate(periods)}
    p["x"] = p["TIME_PERIOD"].map(lookup)
    avg = p.groupby(["TIME_PERIOD", "x"], as_index=False)["SME_FPI_equal_z"].mean()
    latest_period = periods[-1]
    latest = p[p["TIME_PERIOD"] == latest_period].sort_values("SME_FPI_equal_z", ascending=False)
    highlight = latest.head(3)["REF_AREA"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True, gridspec_kw={"wspace": 0.08})

    ax = axes[0]
    for _, sub in p.groupby("REF_AREA"):
        ax.plot(sub["x"], sub["SME_FPI_equal_z"], color="#94a3b8", alpha=0.55, linewidth=1)
    ax.set_title("Before: too many equal-weight lines")
    ax.set_ylabel("SME_FPI, standardized")
    ax.set_xticks(np.arange(len(periods))[::3])
    ax.set_xticklabels([periods[i].replace("-S", " H") for i in np.arange(len(periods))[::3]], rotation=45, ha="right")
    clean_axes(ax)

    ax = axes[1]
    ax.plot(avg["x"], avg["SME_FPI_equal_z"], color=INK, linewidth=2.4, label="Sample average")
    colors = [GOLD, TEAL, RED]
    label_offsets = [0.10, 0.00, -0.10]
    for color, code, yoff in zip(colors, highlight, label_offsets):
        sub = p[p["REF_AREA"] == code].sort_values("x")
        name = sub["country_name"].iloc[0]
        ax.plot(sub["x"], sub["SME_FPI_equal_z"], color=color, linewidth=2.5)
        last = sub.iloc[-1]
        ax.text(last["x"] + 0.25, last["SME_FPI_equal_z"] + yoff, name, color=color, fontsize=9, va="center")
    for period, label in [("2020-S1", "COVID"), ("2022-S1", "rates")]:
        if period in lookup:
            xpos = lookup[period]
            ax.axvspan(xpos - 0.45, xpos + 1.45, color="#f1f5f9", zorder=0)
            ax.text(xpos + 0.05, ax.get_ylim()[1] * 0.84, label, fontsize=8, color=SLATE)
    ax.axhline(0, color="#94a3b8", linewidth=1)
    ax.set_xlim(-0.3, len(periods) + 2.2)
    ax.set_title("After: highlight the comparison that matters")
    ax.set_xticks(np.arange(len(periods))[::3])
    ax.set_xticklabels([periods[i].replace("-S", " H") for i in np.arange(len(periods))[::3]], rotation=45, ha="right")
    clean_axes(ax)
    ax.legend(frameon=False, loc="upper left")

    fig.suptitle("Redesigning a spaghetti chart for a general audience", fontsize=15, weight="bold", color=INK, y=1.02)
    add_source_note(fig)
    savefig("08_general_audience_redesign.png")


def fig_big_cube_problem() -> None:
    severity = pd.read_csv(PROCESSED_DIR / "safe_problem_severity_cube.csv")
    latest_period = severity["TIME_PERIOD"].max()
    latest = severity[
        (severity["TIME_PERIOD"] == latest_period)
        & (severity["FIRM_SIZE"].isin(["MIC", "SML", "MED", "LAR"]))
        & (severity["FIRM_SECTOR"] == "A")
        & (severity["FIRM_AGE"] == 0)
        & (~severity["REF_AREA"].isin(["U2", "T2"]))
    ].copy()
    pivot = latest.pivot_table(index="problem", columns="firm_size_label", values="severity_score_1_10", aggfunc="mean")
    pivot = pivot.reindex(
        ["Finding customers", "Competition", "Access to finance", "Production/labour costs", "Skilled staff", "Regulation", "Other"]
    )[["Micro", "Small", "Medium", "Large"]]

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        vmin=1,
        vmax=10,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Weighted severity score, 1-10", "shrink": 0.82},
        ax=ax,
    )
    ax.set_title(f"SMEs rate labour costs and skills as more pressing than finance")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    add_source_note(fig, "Source: ECB SAFE Q0B pressingness cube; calculations by author.")
    savefig("09_big_cube_problem_severity_by_size.png")


def fig_marginal_breakdowns() -> None:
    access = pd.read_csv(PROCESSED_DIR / "access_finance_severity_cube.csv")
    by_size = access[
        (access["FIRM_SIZE"].isin(["MIC", "SML", "MED", "LAR"]))
        & (access["FIRM_SECTOR"] == "A")
        & (access["FIRM_AGE"] == 0)
        & (~access["REF_AREA"].isin(["U2", "T2"]))
    ]
    by_sector = access[
        (access["FIRM_SIZE"] == "ALL")
        & (access["FIRM_SECTOR"].isin(["C", "I", "S", "T"]))
        & (access["FIRM_AGE"] == 0)
        & (~access["REF_AREA"].isin(["U2", "T2"]))
    ]
    size_avg = by_size.groupby("firm_size_label", as_index=False)["severity_score_1_10"].mean()
    size_avg["firm_size_label"] = pd.Categorical(size_avg["firm_size_label"], ["Micro", "Small", "Medium", "Large"], ordered=True)
    size_avg = size_avg.sort_values("firm_size_label")
    sector_avg = by_sector.groupby("sector_label", as_index=False)["severity_score_1_10"].mean()
    sector_avg["sector_label"] = pd.Categorical(sector_avg["sector_label"], ["Construction", "Industry", "Services", "Trade"], ordered=True)
    sector_avg = sector_avg.sort_values("sector_label")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for ax, data, xcol, color, title in [
        (axes[0], size_avg, "firm_size_label", TEAL, "By firm size"),
        (axes[1], sector_avg, "sector_label", GOLD, "By sector"),
    ]:
        bars = ax.bar(data[xcol].astype(str), data["severity_score_1_10"], color=color, width=0.62)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.07, f"{height:.1f}", ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, 6)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d1d5db")
        ax.spines["bottom"].set_color("#d1d5db")
        ax.tick_params(axis="x", rotation=15)
    axes[0].set_ylabel("Access-to-finance severity, 1-10")
    fig.suptitle("Access-to-finance concern is moderate across firm groups", fontsize=14, weight="bold", y=1.02)
    add_source_note(fig, "Source: ECB SAFE Q0B pressingness cube; calculations by author.")
    savefig("10_access_finance_severity_marginal_breakdowns.png")


def fig_big_cube_trend() -> None:
    access = pd.read_csv(PROCESSED_DIR / "access_finance_severity_cube.csv")
    trend = (
        access[
            (access["FIRM_SIZE"] == "SME")
            & (access["FIRM_SECTOR"] == "A")
            & (access["FIRM_AGE"] == 0)
            & (~access["REF_AREA"].isin(["U2", "T2"]))
        ]
        .groupby(["TIME_PERIOD", "country_name"], as_index=False)["severity_score_1_10"]
        .mean()
    )
    periods = sorted(trend["TIME_PERIOD"].unique(), key=period_sort)
    lookup = {p: i for i, p in enumerate(periods)}
    trend["x"] = trend["TIME_PERIOD"].map(lookup)
    avg = trend.groupby("x", as_index=False)["severity_score_1_10"].mean()

    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    for _, sub in trend.groupby("country_name"):
        ax.plot(sub["x"], sub["severity_score_1_10"], color="#cbd5e1", alpha=0.75, linewidth=1)
    ax.plot(avg["x"], avg["severity_score_1_10"], color=TEAL, linewidth=2.8, label="Country average")
    ax.set_xticks(np.arange(len(periods))[::2])
    ax.set_xticklabels([periods[i].replace("-S", " H") for i in np.arange(len(periods))[::2]], rotation=0)
    ax.set_ylabel("Severity score, 1-10")
    ax.set_title("Access-to-finance severity has eased since 2015")
    ax.set_ylim(2.4, 8.1)
    clean_axes(ax)
    ax.legend(frameon=False)
    add_source_note(fig, "Source: ECB SAFE Q0B pressingness cube; calculations by author.")
    savefig("11_access_finance_severity_big_cube_trend.png")


def main() -> None:
    setup_style()
    panel = pd.read_csv(PROCESSED_DIR / "sme_fpi_panel.csv")
    fig_timeseries(panel)
    fig_country_heatmap(panel)
    fig_hidden_gap(panel)
    fig_pca_loadings()
    fig_correlation(panel)
    fig_clusters(panel)
    fig_size_component()
    fig_redesign(panel)
    fig_big_cube_problem()
    fig_marginal_breakdowns()
    fig_big_cube_trend()
    print("Polished figures complete.")


if __name__ == "__main__":
    main()
