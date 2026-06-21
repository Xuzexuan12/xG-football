
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGES = [
    ROOT / "SR_Submission",
]
BLUE = "#2F7EBB"
ORANGE = "#B85C38"
GREEN = "#5FA37F"
GREY = "#666666"
LIGHT_GREY = "#D8D8D8"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "axes.linewidth": 0.7,
            "legend.fontsize": 6.6,
            "legend.frameon": False,
        }
    )


def reliability_curve(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    d = df.dropna(subset=["is_goal", "xg_all_shot"]).copy()
    d["bin"] = pd.qcut(d["xg_all_shot"].rank(method="first"), q=n_bins, labels=False) + 1
    return (
        d.groupby("bin", as_index=False)
        .agg(
            mean_predicted_xg=("xg_all_shot", "mean"),
            observed_goal_rate=("is_goal", "mean"),
            n=("is_goal", "size"),
        )
        .sort_values("bin")
    )


def load_strata(package_dir: Path) -> pd.DataFrame:
    path = package_dir / "results" / "calibration_strata_diagnostics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def load_shots(package_dir: Path) -> pd.DataFrame:
    path = package_dir / "results" / "all_competition_shots_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.16, linewidth=0.55)
    ax.set_axisbelow(True)


def panel_label(ax, label: str):
    ax.text(-0.13, 1.13, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top")


def short_competition(label: str) -> str:
    return label.replace("European Championship", "Euro").replace("World Cup", "World Cup")


def plot_package(package_dir: Path) -> None:
    set_style()
    shots = load_shots(package_dir)
    strata = load_strata(package_dir)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.75), gridspec_kw={"hspace": 0.58, "wspace": 0.42})
    axes = axes.ravel()

    rel = reliability_curve(shots)
    ax = axes[0]
    clean_axis(ax)
    panel_label(ax, "a")
    ax.plot([0.0, 0.8], [0.0, 0.8], color=GREY, linestyle="--", linewidth=0.9)
    ax.plot(
        rel["mean_predicted_xg"],
        rel["observed_goal_rate"],
        marker="o",
        color=BLUE,
        linewidth=1.6,
        markersize=4,
    )
    ax.set_xlim(0.0, 0.8)
    ax.set_ylim(0.0, 0.8)
    ax.set_xlabel("Mean predicted xG")
    ax.set_ylabel("Observed goal rate")
    ax.set_title("Pooled reliability")

    ax = axes[1]
    clean_axis(ax)
    panel_label(ax, "b")
    xg_bins = strata[strata["stratum_type"] == "predicted_xg_decile"].copy()
    xg_bins["order"] = xg_bins["stratum"].str.extract(r"Q(\d+)").astype(int)
    xg_bins = xg_bins.sort_values("order")
    ax.axhline(0.0, color=GREY, linewidth=0.8)
    ax.bar(
        np.arange(len(xg_bins)),
        xg_bins["calibration_gap_observed_minus_predicted"],
        color="#8FB9D9",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(np.arange(len(xg_bins)))
    ax.set_xticklabels(xg_bins["stratum"], fontsize=8)
    ax.set_xlabel("Predicted xG decile")
    ax.set_ylabel("Observed minus predicted")
    ax.set_title("Local gap by xG decile")

    ax = axes[2]
    clean_axis(ax)
    panel_label(ax, "c")
    comp = strata[strata["stratum_type"] == "competition"].copy()
    comp = comp.sort_values("calibration_gap_observed_minus_predicted")
    y = np.arange(len(comp))
    ax.axvline(0.0, color=GREY, linewidth=0.8)
    ax.barh(
        y,
        comp["calibration_gap_observed_minus_predicted"],
        color=GREEN,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([short_competition(v) for v in comp["stratum"]], fontsize=8)
    ax.set_xlabel("Observed minus predicted")
    ax.set_title("Competition-level gap")

    ax = axes[3]
    clean_axis(ax)
    panel_label(ax, "d")
    shot = strata[strata["stratum_type"] == "shot_type"].copy()
    order = ["Regular shots", "Direct free kicks", "Penalties"]
    shot["order"] = shot["stratum"].map({name: i for i, name in enumerate(order)}).fillna(99)
    shot = shot.sort_values("order")
    labels = ["Regular", "Direct FK", "Penalty"][: len(shot)]
    y = np.arange(len(shot))
    ax.hlines(
        y,
        shot["observed_goal_rate"],
        shot["mean_predicted_xg"],
        color=LIGHT_GREY,
        linewidth=1.6,
        zorder=1,
    )
    ax.scatter(shot["observed_goal_rate"], y, color=BLUE, s=22, label="Observed", zorder=3)
    ax.scatter(shot["mean_predicted_xg"], y, color=ORANGE, s=22, marker="D", label="Predicted", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Goal rate or mean xG")
    ax.set_xlim(0.0, 0.86)
    ax.set_title("Routed module check")
    ax.legend(loc="lower right", handletextpad=0.4)
    penalty = shot[shot["stratum"] == "Penalties"]
    if not penalty.empty:
        row = penalty.iloc[0]
        ax.annotate(
            "fixed 0.79 xG",
            xy=(row["mean_predicted_xg"], float(row["order"])),
            xytext=(-42, -18),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=6.4,
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": GREY},
        )

    fig.suptitle("Stratified calibration audit for the routed all-shot xG workflow", x=0.01, ha="left", fontsize=9.5)
    fig.subplots_adjust(top=0.87, bottom=0.12, left=0.10, right=0.98, hspace=0.65, wspace=0.42)
    fig_dir = package_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "tiff"):
        fig.savefig(fig_dir / f"fig2_calibration_audit.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", action="append", type=Path, help="SR package directory to update")
    args = parser.parse_args()
    packages = args.package if args.package else DEFAULT_PACKAGES
    for package_dir in packages:
        plot_package(package_dir.resolve())


if __name__ == "__main__":
    main()
