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

INK = "#172033"
MUTED = "#64748b"
GRID = "#e5e7eb"
TEAL = "#147c78"
RED = "#a33a2a"
GOLD = "#d99a1e"


def period_sort(period: str) -> int:
    return int(period[:4]) * 10 + int(period[-1])


def style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#d1d5db",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "legend.fontsize": 8.5,
            "savefig.facecolor": "white",
        }
    )


def clean_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)


def main() -> None:
    style()
    panel = pd.read_csv(PROCESSED_DIR / "sme_fpi_panel.csv")
    big_review = pd.read_csv(PROCESSED_DIR / "big_cube_review_summary.csv").iloc[0]
    panel = panel[panel["period_sort"] >= period_sort("2015-S1")].copy()

    periods = sorted(panel["TIME_PERIOD"].unique(), key=period_sort)
    x_lookup = {period: i for i, period in enumerate(periods)}

    avg = (
        panel.groupby(["TIME_PERIOD", "period_sort"], as_index=False)
        .agg(SME_FPI=("SME_FPI_equal_z", "mean"), CISS=("CISS_z", "mean"))
        .sort_values("period_sort")
    )
    avg["x"] = avg["TIME_PERIOD"].map(x_lookup)

    country_order = (
        panel.groupby("country_name")["SME_FPI_equal_z"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    heat = (
        panel.pivot_table(index="country_name", columns="TIME_PERIOD", values="SME_FPI_equal_z")
        .reindex(country_order)[periods]
    )

    latest = (
        panel.dropna(subset=["Hidden_SME_Stress"])
        .sort_values("period_sort")
        .groupby("REF_AREA", as_index=False)
        .tail(1)
        .sort_values("Hidden_SME_Stress")
    )

    fig = plt.figure(figsize=(15.2, 9.6))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[0.46, 1.02, 1.22],
        width_ratios=[1.15, 1.15, 0.92],
        hspace=0.46,
        wspace=0.38,
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0,
        0.95,
        "Hidden SME Financing Stress in Europe",
        fontsize=21,
        weight="bold",
        color=INK,
        va="top",
    )
    title_ax.text(
        0,
        0.43,
        "Borrower-side survey index vs common euro-area CISS benchmark.\n"
        "The gap shows where SME financing pain stays high relative to market-side stress.",
        fontsize=10.7,
        color=MUTED,
        va="top",
    )
    title_ax.text(
        0.995,
        0.86,
        f"SAFE Q0B cube\n{int(big_review['raw_rows']):,} rows | {big_review['raw_file_mb']:.0f} MB\n"
        f"{int(big_review['countries'])} regions | {int(big_review['periods'])} half-years",
        fontsize=9.0,
        color=INK,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#dbe3ea"),
    )

    ax_time = fig.add_subplot(gs[1, :2])
    x = avg["x"].to_numpy()
    ax_time.plot(x, avg["SME_FPI"], color=TEAL, linewidth=3.0, label="Borrower-side SME_FPI")
    ax_time.plot(x, avg["CISS"], color=RED, linewidth=2.6, label="Market-side ECB CISS")
    ax_time.fill_between(x, avg["SME_FPI"], avg["CISS"], color="#94a3b8", alpha=0.13)
    for period, label in [("2020-S1", "COVID"), ("2022-S1", "rate shock")]:
        if period in x_lookup:
            xpos = x_lookup[period]
            ax_time.axvspan(xpos - 0.45, xpos + 1.45, color="#eef2f7", zorder=0)
            ax_time.text(xpos + 0.05, ax_time.get_ylim()[1] * 0.82, label, fontsize=9, color=MUTED)
    ax_time.axhline(0, color="#9aa7b8", linewidth=1)
    tick_locs = np.arange(len(periods))[::2]
    ax_time.set_xticks(tick_locs)
    ax_time.set_xticklabels([periods[i].replace("-S", " H") for i in tick_locs])
    ax_time.set_ylabel("Standardized value")
    ax_time.set_title("Market stress spikes, but borrower-side stress is more persistent")
    ax_time.legend(frameon=False, loc="upper left")
    clean_axes(ax_time)

    ax_gap = fig.add_subplot(gs[1:, 2])
    colors = [TEAL if v < 0 else RED for v in latest["Hidden_SME_Stress"]]
    bars = ax_gap.barh(latest["country_name"], latest["Hidden_SME_Stress"], color=colors, height=0.62)
    ax_gap.axvline(0, color="#9aa7b8", linewidth=1)
    for bar, val in zip(bars, latest["Hidden_SME_Stress"]):
        offset = 0.025 if val >= 0 else -0.035
        ha = "left" if val >= 0 else "right"
        ax_gap.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", ha=ha, fontsize=8)
    latest_period = latest["TIME_PERIOD"].iloc[0].replace("-S", " H")
    ax_gap.set_title(f"Relative gap vs common CISS benchmark\nlatest period: {latest_period}")
    ax_gap.set_xlabel("SME_FPI_z - common CISS_z")
    ax_gap.set_xlim(-0.42, 1.22)
    ax_gap.tick_params(axis="y", labelsize=9, pad=8)
    clean_axes(ax_gap, grid_axis="x")
    ax_gap.grid(axis="y", visible=False)

    ax_heat = fig.add_subplot(gs[2, :2])
    sns.heatmap(
        heat,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0,
        vmin=-1.8,
        vmax=1.8,
        linewidths=0.25,
        linecolor="#f8fafc",
        cbar_kws={"label": "SME_FPI, standardized", "shrink": 0.82, "pad": 0.02},
        ax=ax_heat,
    )
    ax_heat.set_title("Country-period heatmap reveals persistent cross-country differences")
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.set_xticklabels([p.replace("-S", " H") for p in periods], rotation=45, ha="right", fontsize=7.5)
    ax_heat.tick_params(axis="y", labelsize=8.5)

    fig.text(
        0.01,
        0.015,
        "Source: ECB SAFE and ECB CISS. SME_FPI is an equal-weight standardized borrower-side index; CISS is a common euro-area benchmark.",
        fontsize=8.5,
        color=MUTED,
    )

    out = FIG_DIR / "00_signature_sme_fpi_story.png"
    plt.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.18)
    plt.close()
    print(f"saved {out}")

    # Extra copy without leading underscore for easier opening in some file pickers.
    contact = FIG_DIR / "_contact_sheet_polished.png"
    if contact.exists():
        from PIL import Image

        im = Image.open(contact).convert("RGB")
        im.thumbnail((1000, 2000))
        easy = FIG_DIR / "contact_sheet_polished_preview.jpg"
        im.save(easy, quality=92)
        print(f"saved {easy}")


if __name__ == "__main__":
    main()
