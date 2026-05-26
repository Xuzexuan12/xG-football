import csv
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from revision_multicomp_analysis import OUT_DIR, ROOT, clip01, oof_predictions, usable_features
from revision_plot_style import PALETTE, clean_axis, save_submission_figure, set_pub_style


SOURCE_TEX = ROOT / "sage_latex_template_4" / "Sage_LaTeX_Guidelines.tex"
SOURCE_BIB = ROOT / "sage_latex_template_4" / "ref.bib"
SOURCE_FIG_DIR = ROOT / "sage_latex_template_4" / "figure"
JQAS_TEMPLATE_DIR = ROOT.parent / "JQAS" / "JQAS_JQAS_LaTeX-Template-for-Authors"
PACKAGE_DIR = ROOT / "sage_latex_template_4" / "jqas_submission"
FIG_DIR = PACKAGE_DIR / "figures"
FRONTIERS_PACKAGE_DIR = ROOT / "frontiers_submission"
FRONTIERS_FIG_DIR = FRONTIERS_PACKAGE_DIR / "figures"


METHOD_COLORS = {
    "baseline": "#5B677A",
    "chosen": "#2F7EBB",
    "best": "#B85C38",
    "full": "#2D8A62",
    "hgb": "#8B5AA0",
    "none": "#6E6E6E",
    "sigmoid": "#2F7EBB",
    "isotonic": "#B85C38",
    "direct transfer": "#6E6E6E",
    "intercept recalibration": "#2D8A62",
    "sigmoid recalibration": "#2F7EBB",
    "isotonic recalibration": "#B85C38",
}


def panel_label(ax, label):
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def label_for_model(label):
    s = str(label).lower()
    if "baseline" in s:
        return METHOD_COLORS["baseline"]
    if "chosen" in s or "k*" in s:
        return METHOD_COLORS["chosen"]
    if "best" in s:
        return METHOD_COLORS["best"]
    if "hgb" in s:
        return METHOD_COLORS["hgb"]
    return METHOD_COLORS["full"]


def save_jqas(fig, stem):
    save_submission_figure(fig, FIG_DIR / stem, dpi=600, preview=True)
    for suffix in (".pdf", ".tiff", ".png"):
        src = FIG_DIR / f"{stem}{suffix}"
        if src.exists():
            FRONTIERS_FIG_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, FRONTIERS_FIG_DIR / src.name)


def aggregate_with_band(df, group_cols, value_col):
    out = df.groupby(group_cols, as_index=False).agg(
        mean=(value_col, "mean"),
        sd=(value_col, "std"),
        n=(value_col, "size"),
        train_goals=("train_goals", "mean") if "train_goals" in df.columns else (value_col, "size"),
        calibration_goals=("calibration_goals", "mean") if "calibration_goals" in df.columns else (value_col, "size"),
    )
    out["se95"] = 1.96 * out["sd"].fillna(0) / np.sqrt(out["n"].clip(lower=1))
    return out


def selected_k(perf):
    best = perf.loc[perf["LogLoss"].idxmin()]
    threshold = float(best["LogLoss"] + best["LogLoss_SE"])
    k_star = int(perf.loc[perf["LogLoss"] <= threshold, "k"].min())
    return k_star, int(best["k"])


def plot_pipeline_data():
    df = pd.read_csv(OUT_DIR / "dataset_summary.csv")
    data_rows = df[df["competition"] != "Total"].copy()
    total = data_rows[["matches", "analytic_shots", "analytic_goals"]].sum()
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = GridSpec(2, 1, height_ratios=[1.05, 1.0], hspace=0.42, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax0.set_axis_off()
    panel_label(ax0, "a")

    steps = [
        ("Raw events", "Wyscout event logs\n7 competitions"),
        ("Shot labels", "Native goal tags\nno post-shot features"),
        ("Features", "25 transparent\nshot features"),
        ("Evaluation", "Grouped CV,\ncalibration, transfer"),
        ("xG evidence", "Proper scoring rules\nand uncertainty"),
    ]
    x0s = np.linspace(0.02, 0.82, len(steps))
    width = 0.15
    for i, ((title, body), x0) in enumerate(zip(steps, x0s)):
        box = FancyBboxPatch(
            (x0, 0.32),
            width,
            0.42,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=0.8,
            edgecolor="#333333",
            facecolor="#F4F6F8",
            transform=ax0.transAxes,
        )
        ax0.add_patch(box)
        ax0.text(x0 + width / 2, 0.62, title, ha="center", va="center", fontsize=8.2, fontweight="bold", transform=ax0.transAxes)
        ax0.text(x0 + width / 2, 0.45, body, ha="center", va="center", fontsize=7.5, transform=ax0.transAxes)
        if i < len(steps) - 1:
            ax0.add_patch(
                FancyArrowPatch(
                    (x0 + width + 0.006, 0.53),
                    (x0s[i + 1] - 0.006, 0.53),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=0.8,
                    color="#555555",
                    transform=ax0.transAxes,
                )
            )
    ax0.text(
        0.5,
        0.08,
        f"Analytical sample: {int(total['matches']):,} matches, {int(total['analytic_shots']):,} shots, {int(total['analytic_goals']):,} goals",
        ha="center",
        fontsize=8,
        transform=ax0.transAxes,
    )

    clean_axis(ax1, grid=False)
    panel_label(ax1, "b")
    order = ["England", "France", "Germany", "Italy", "Spain", "European Championship", "World Cup"]
    g = data_rows.set_index("competition").loc[order].reset_index()
    labels = ["England", "France", "Germany", "Italy", "Spain", "Euro", "World Cup"]
    x = np.arange(len(g))
    ax1.bar(x, g["analytic_shots"], color="#8FB9D9", edgecolor="white", linewidth=0.6, label="Shots")
    ax1.set_ylabel("Analytical shots")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1b = ax1.twinx()
    ax1b.plot(x, g["goal_rate"] * 100, color="#B85C38", marker="o", lw=1.4, label="Goal rate")
    ax1b.set_ylabel("Goal rate (%)", color="#B85C38")
    ax1b.tick_params(axis="y", colors="#B85C38")
    ax1b.spines["top"].set_visible(False)
    ax1b.spines["right"].set_linewidth(0.75)
    lines = [ax1.patches[0], ax1b.lines[0]]
    ax1.legend(lines, ["Shots", "Goal rate"], loc="upper right")
    save_jqas(fig, "fig1_pipeline_data")


def plot_validation_benchmark():
    val = pd.read_csv(OUT_DIR / "validation_design_sensitivity.csv")
    metrics = pd.read_csv(OUT_DIR / "model_metrics.csv")
    pooled = metrics[metrics["setting"] == "Pooled grouped CV"].copy()
    boot = pd.read_csv(OUT_DIR / "pooled_bootstrap_logloss_diff.csv")
    order = ["Random shot-level CV", "Match-grouped CV", "Shooting-team blocked CV", "Leave-one-competition-out"]
    val = val.set_index("validation_design").loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), gridspec_kw={"width_ratios": [1.2, 1.05, 0.95]})
    ax = axes[0]
    clean_axis(ax)
    panel_label(ax, "a")
    x = np.arange(len(val))
    yerr = np.vstack([val["LogLoss"] - val["LogLoss_ci_low"], val["LogLoss_ci_high"] - val["LogLoss"]])
    ax.bar(x, val["LogLoss"], yerr=yerr, capsize=2.5, color="#88B6D9", edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Random\nshot", "Match\ngrouped", "Team\nblocked", "Leave-one\ncompetition"], rotation=0)
    ax.set_ylabel("LogLoss")
    ax.set_ylim(val["LogLoss"].min() - 0.018, val["LogLoss"].max() + 0.018)
    axb = ax.twinx()
    axb.plot(x, val["ECE"], color="#B85C38", marker="D", lw=1.2, label="ECE")
    axb.set_ylabel("ECE", color="#B85C38")
    axb.tick_params(axis="y", colors="#B85C38")
    axb.spines["top"].set_visible(False)

    ax = axes[1]
    clean_axis(ax)
    panel_label(ax, "b")
    model_order = [
        "Full logistic none",
        "Full logistic sigmoid",
        "Full logistic isotonic",
        "HGB sigmoid fixed comparator",
    ]
    p = pooled.set_index("model").loc[model_order].reset_index()
    x = np.arange(len(p))
    colors = [METHOD_COLORS["none"], METHOD_COLORS["sigmoid"], METHOD_COLORS["isotonic"], METHOD_COLORS["hgb"]]
    ax.bar(x, p["LogLoss"], color=colors, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Uncal.", "Sigmoid", "Isotonic", "HGB"], rotation=30, ha="right")
    ax.set_ylabel("Pooled LogLoss")
    ax.set_ylim(p["LogLoss"].min() - 0.0025, p["LogLoss"].max() + 0.0025)

    ax = axes[2]
    clean_axis(ax)
    panel_label(ax, "c")
    b = boot[boot["comparison"].str.contains("HGB")].iloc[0]
    ax.errorbar([0], [b["mean_diff"]], yerr=[[b["mean_diff"] - b["ci_low"]], [b["ci_high"] - b["mean_diff"]]], fmt="o", color=METHOD_COLORS["hgb"], capsize=4)
    ax.axhline(0, color="#555555", lw=0.8, ls="--")
    ax.set_xlim(-0.7, 0.7)
    ax.set_xticks([0])
    ax.set_xticklabels(["HGB -\nlogistic"])
    ax.set_ylabel(r"$\Delta$ LogLoss")
    ax.set_ylim(-0.0032, 0.0002)
    ax.text(
        0.05,
        0.90,
        f"mean {b['mean_diff']:.4f}\n95% CI [{b['ci_low']:.4f}, {b['ci_high']:.4f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
    )
    save_jqas(fig, "fig2_validation_benchmark")


def plot_worldcup_stress():
    all_shots = pd.read_csv(OUT_DIR / "all_competition_shots_features.csv")
    world = all_shots[all_shots["competition"] == "World_Cup"].copy()
    cols = usable_features(all_shots)
    perf = pd.read_csv(OUT_DIR / "worldcup_perf_curve.csv")
    k_star, k_best = selected_k(perf)

    pred_dict = {}
    y_common = None
    for label, model_cols in [
        ("Baseline(1)", cols[:1]),
        (f"Chosen(k*={k_star})", cols[:k_star]),
        (f"Best(k={k_best})", cols[:k_best]),
    ]:
        _, y, p = oof_predictions(world, model_cols, "logistic", "isotonic")
        pred_dict[label] = clip01(p)
        y_common = y

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.48)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    ax = axes[0]
    clean_axis(ax)
    panel_label(ax, "a")
    x = perf["k"].values
    ax.plot(x, perf["R2"], color=METHOD_COLORS["chosen"], marker="o", ms=3.0, lw=1.2, label=r"Nagelkerke $R^2$")
    ax.set_xlabel("Number of features")
    ax.set_ylabel(r"$R^2$", color=METHOD_COLORS["chosen"])
    ax.tick_params(axis="y", colors=METHOD_COLORS["chosen"])
    axb = ax.twinx()
    axb.plot(x, perf["LogLoss"], color=METHOD_COLORS["best"], marker="D", ms=2.8, lw=1.2, label="LogLoss")
    axb.fill_between(x, perf["LogLoss"] - perf["LogLoss_SE"], perf["LogLoss"] + perf["LogLoss_SE"], color=METHOD_COLORS["best"], alpha=0.16, linewidth=0)
    axb.set_ylabel("LogLoss", color=METHOD_COLORS["best"])
    axb.tick_params(axis="y", colors=METHOD_COLORS["best"])
    axb.spines["top"].set_visible(False)
    ax.axvline(k_star, color="#555555", ls="--", lw=0.8)
    ax.axvline(k_best, color="#555555", ls=":", lw=0.8)

    ax = axes[1]
    clean_axis(ax)
    panel_label(ax, "b")
    d = np.r_[0.0, np.diff(perf["LogLoss"].values)]
    colors = ["#8F8F8F"] + ["#2D8A62" if v < 0 else "#B85C38" for v in d[1:]]
    ax.bar(x, d, color=colors, edgecolor="white", linewidth=0.4, width=0.72)
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Number of features")
    ax.set_ylabel(r"$\Delta$ LogLoss")

    ax = axes[2]
    clean_axis(ax)
    panel_label(ax, "c")
    wc_metrics = pd.read_csv(OUT_DIR / "model_metrics.csv")
    wc = wc_metrics[(wc_metrics["setting"] == "World Cup grouped CV") & wc_metrics["model"].isin(["Baseline(1)", f"Chosen(k*={k_star})", f"World Cup-selected Best(k={k_best})"])]
    labels = ["Baseline", "Chosen", "Best"]
    ax.bar(np.arange(3), wc["LogLoss"], color=[METHOD_COLORS["baseline"], METHOD_COLORS["chosen"], METHOD_COLORS["best"]], edgecolor="white", linewidth=0.6)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("World Cup LogLoss")
    ax.set_ylim(wc["LogLoss"].min() - 0.018, wc["LogLoss"].max() + 0.018)

    ax = axes[3]
    clean_axis(ax)
    panel_label(ax, "d")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#777777")
    for label, p in pred_dict.items():
        fpr, tpr, _ = roc_curve(y_common, p)
        ax.plot(fpr, tpr, lw=1.25, color=label_for_model(label), label=f"{label} ({roc_auc_score(y_common, p):.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=7.4)

    ax = axes[4]
    clean_axis(ax)
    panel_label(ax, "e")
    ax.axhline(float(np.mean(y_common)), 0, 1, ls="--", lw=0.8, color="#777777")
    for label, p in pred_dict.items():
        prec, rec, _ = precision_recall_curve(y_common, p)
        ax.plot(rec, prec, lw=1.25, color=label_for_model(label), label=f"{label} ({average_precision_score(y_common, p):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=7.4)

    ax = axes[5]
    clean_axis(ax)
    panel_label(ax, "f")
    ax.plot([0, 0.75], [0, 0.75], ls="--", lw=0.8, color="#777777")
    q = np.linspace(0, 1, 11)
    for label, p in pred_dict.items():
        bins = np.quantile(p, q)
        bins[0], bins[-1] = -np.inf, np.inf
        xs, ys = [], []
        for i in range(10):
            m = (p >= bins[i]) & (p < bins[i + 1])
            if m.any():
                xs.append(float(p[m].mean()))
                ys.append(float(np.mean(y_common[m])))
        ax.plot(xs, ys, marker="o", ms=3.0, lw=1.25, color=label_for_model(label), label=label)
    ax.set_xlabel("Mean predicted xG")
    ax.set_ylabel("Observed goal rate")
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 0.75)
    return world, cols, y_common, pred_dict[f"Best(k={k_best})"], save_jqas(fig, "fig3_worldcup_stress")


def plot_calibration_deployment():
    sample = pd.read_csv(OUT_DIR / "calibration_sample_size_stability.csv")
    target = pd.read_csv(OUT_DIR / "target_domain_recalibration.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), gridspec_kw={"hspace": 0.55, "wspace": 0.42})
    calibrations = ["none", "sigmoid", "isotonic"]
    labels = {"none": "Uncalibrated", "sigmoid": "Sigmoid", "isotonic": "Isotonic"}
    for ax, metric, label in zip(axes[0], ["LogLoss", "ECE"], ["a", "b"]):
        clean_axis(ax)
        panel_label(ax, label)
        agg = aggregate_with_band(sample, ["train_fraction", "calibration"], metric)
        for calibration in calibrations:
            g = agg[agg["calibration"] == calibration].sort_values("train_fraction")
            ax.plot(g["train_goals"], g["mean"], color=METHOD_COLORS[calibration], marker="o", lw=1.25, label=labels[calibration])
            ax.fill_between(g["train_goals"], g["mean"] - g["se95"], g["mean"] + g["se95"], color=METHOD_COLORS[calibration], alpha=0.14, linewidth=0)
        ax.set_xlabel("Mean training goals")
        ax.set_ylabel(metric)
    axes[0, 0].legend(loc="best", fontsize=8.0)

    strategies = ["direct transfer", "intercept recalibration", "sigmoid recalibration", "isotonic recalibration"]
    strategy_labels = {
        "direct transfer": "Direct",
        "intercept recalibration": "Intercept",
        "sigmoid recalibration": "Sigmoid",
        "isotonic recalibration": "Isotonic",
    }
    for ax, metric, label in zip(axes[1], ["LogLoss", "ECE"], ["c", "d"]):
        clean_axis(ax)
        panel_label(ax, label)
        agg = aggregate_with_band(target, ["calibration_fraction", "strategy"], metric)
        for strategy in strategies:
            g = agg[agg["strategy"] == strategy].sort_values("calibration_fraction")
            ax.plot(g["calibration_goals"], g["mean"], color=METHOD_COLORS[strategy], marker="o", lw=1.15, label=strategy_labels[strategy])
            ax.fill_between(g["calibration_goals"], g["mean"] - g["se95"], g["mean"] + g["se95"], color=METHOD_COLORS[strategy], alpha=0.12, linewidth=0)
        ax.set_xlabel("Mean target calibration goals")
        ax.set_ylabel(metric)
    axes[1, 0].legend(loc="best", fontsize=8.0)
    save_jqas(fig, "fig4_calibration_deployment")


def plot_feature_interpretation(world, cols, best_pred):
    family = pd.read_csv(OUT_DIR / "feature_family_ablation.csv")
    order = [
        "Geometry only",
        "Geometry + Match",
        "Geometry + Previous",
        "Geometry + Execution",
        "Geometry + Shot context",
        "Full",
        "Full minus Previous",
        "Full minus Execution",
        "Full minus Match",
        "Full minus Shot context",
    ]
    fig = plt.figure(figsize=(7.2, 5.15))
    gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.45)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for ax, setting, label in zip(axes[:2], ["Pooled match-grouped CV", "World Cup match-grouped CV"], ["a", "b"]):
        clean_axis(ax)
        panel_label(ax, label)
        g = family[family["setting"] == setting].set_index("feature_family_model").loc[order].reset_index()
        improvement = -g["delta_logloss_vs_geometry"].values
        y = np.arange(len(g))
        colors = ["#7F7F7F" if name == "Geometry only" else "#6FAE8F" for name in g["feature_family_model"]]
        ax.barh(y, improvement, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="#555555", lw=0.75)
        ax.set_yticks(y)
        short_labels = [
            name.replace("Geometry", "Geom.")
            .replace("Previous", "Prev.")
            .replace("Execution", "Exec.")
            .replace("Shot context", "Shot ctx")
            .replace("Match", "Match")
            .replace("Full minus", "Full -")
            for name in g["feature_family_model"]
        ]
        ax.set_yticklabels(short_labels, fontsize=8.0)
        ax.invert_yaxis()
        ax.set_xlabel("LogLoss improvement vs geometry")
        ax.set_title(setting.replace(" match-grouped CV", ""), fontsize=8.5)

    partial_cols = ["feat_dist_m", "feat_angle_rad"]
    names = {"feat_dist_m": "Distance to goal (m)", "feat_angle_rad": "Shooting angle (rad)"}
    for ax, col, label in zip(axes[2:], partial_cols, ["c", "d"]):
        clean_axis(ax)
        panel_label(ax, label)
        x = world[col].astype(float).values
        q = np.quantile(x[~np.isnan(x)], np.linspace(0, 1, 11))
        q[0] -= 1e-9
        q[-1] += 1e-9
        xs, mu, se = [], [], []
        for i in range(10):
            m = (x >= q[i]) & (x < q[i + 1])
            if m.any():
                vals = best_pred[m]
                xs.append(float(np.mean(x[m])))
                mu.append(float(np.mean(vals)))
                se.append(float(np.std(vals, ddof=1)) / np.sqrt(max(1, int(m.sum()))))
        xs, mu, se = np.array(xs), np.array(mu), np.array(se)
        ax.plot(xs, mu, color=METHOD_COLORS["best"], marker="o", lw=1.3)
        ax.fill_between(xs, mu - 1.96 * se, mu + 1.96 * se, color=METHOD_COLORS["best"], alpha=0.15, linewidth=0)
        ax.set_xlabel(names[col])
        ax.set_ylabel("Predicted xG")
    save_jqas(fig, "fig6_feature_interpretation")


def draw_half_pitch(ax):
    ax.set_xlim(50, 101.5)
    ax.set_ylim(-1.5, 101.5)
    ax.set_aspect("equal")
    ax.axis("off")
    line_color = "#2A2A2A"
    lw = 0.8
    ax.add_patch(Rectangle((50, 0), 50, 100, fill=False, ec=line_color, lw=lw))
    ax.plot([100, 100], [44.6, 55.4], color=line_color, lw=1.2)
    ax.add_patch(Rectangle((83.5, 21.1), 16.5, 57.8, fill=False, ec=line_color, lw=lw))
    ax.add_patch(Rectangle((94.5, 36.8), 5.5, 26.4, fill=False, ec=line_color, lw=lw))
    ax.scatter([89], [50], s=5, color=line_color, zorder=5)
    ax.add_patch(Arc((89, 50), 18.3, 18.3, theta1=128, theta2=232, ec=line_color, lw=lw))


def _bayes_wide(summary: pd.DataFrame, parameter: str) -> pd.DataFrame:
    out = summary[summary["parameter"] == parameter].copy()
    return out.rename(
        columns={
            "mean": f"{parameter}_mean",
            "hdi_2_5": f"{parameter}_low",
            "hdi_97_5": f"{parameter}_high",
        }
    )


def plot_bayesian_xg_visual():
    all_shots = pd.read_csv(OUT_DIR / "all_competition_shots_features.csv")
    summary_path = OUT_DIR / "bayesian_competition_calibration_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run Code/bayesian_xg_analysis.py before regenerating figures."
        )
    bayes = pd.read_csv(summary_path)
    dataset = pd.read_csv(OUT_DIR / "dataset_summary.csv")
    comp_order = [c for c in dataset.loc[dataset["competition"] != "Total", "competition"].tolist()]

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = GridSpec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.0], hspace=0.32, wspace=0.28, figure=fig)
    ax_surface = fig.add_subplot(gs[0, 0])
    ax_shots = fig.add_subplot(gs[0, 1])
    ax_intercept = fig.add_subplot(gs[1, 0])
    ax_slope = fig.add_subplot(gs[1, 1])

    panel_label(ax_surface, "a")
    draw_half_pitch(ax_surface)
    d = all_shots.dropna(subset=["ax", "ay", "xg_all_shot"]).copy()
    d = d[(d["ax"] >= 50) & (d["ax"] <= 100) & (d["ay"] >= 0) & (d["ay"] <= 100)]
    hb = ax_surface.hexbin(
        d["ax"],
        d["ay"],
        C=d["xg_all_shot"],
        reduce_C_function=np.mean,
        gridsize=31,
        extent=(50, 100, 0, 100),
        mincnt=8,
        cmap="YlGnBu",
        linewidths=0,
        alpha=0.95,
    )
    cb = fig.colorbar(hb, ax=ax_surface, fraction=0.046, pad=0.02)
    cb.set_label("Mean xG")
    ax_surface.set_title("Shot-location xG surface")

    panel_label(ax_shots, "b")
    draw_half_pitch(ax_shots)
    wc = all_shots[all_shots["competition"] == "World_Cup"].dropna(subset=["ax", "ay", "xg_all_shot"]).copy()
    miss = wc[wc["is_goal"].astype(int) == 0]
    goal = wc[wc["is_goal"].astype(int) == 1]
    ax_shots.scatter(
        miss["ax"],
        miss["ay"],
        s=6 + 55 * miss["xg_all_shot"],
        c="#8A97A6",
        alpha=0.35,
        linewidths=0,
        label="No goal",
    )
    ax_shots.scatter(
        goal["ax"],
        goal["ay"],
        s=14 + 70 * goal["xg_all_shot"],
        c="#B85C38",
        alpha=0.78,
        edgecolors="white",
        linewidths=0.25,
        label="Goal",
    )
    ax_shots.set_title("World Cup shots sized by xG")
    ax_shots.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=7)

    alpha = _bayes_wide(bayes, "alpha").set_index("competition").reindex(comp_order).dropna().reset_index()
    beta = _bayes_wide(bayes, "beta").set_index("competition").reindex(comp_order).dropna().reset_index()
    labels = [c.replace("European Championship", "Euro").replace("World Cup", "World Cup") for c in alpha["competition"]]
    y = np.arange(len(alpha))

    clean_axis(ax_intercept)
    panel_label(ax_intercept, "c")
    ax_intercept.errorbar(
        alpha["alpha_mean"],
        y,
        xerr=np.vstack([alpha["alpha_mean"] - alpha["alpha_low"], alpha["alpha_high"] - alpha["alpha_mean"]]),
        fmt="o",
        color="#2F7EBB",
        ecolor="#8FB9D9",
        elinewidth=1.2,
        capsize=2.5,
        ms=3.5,
    )
    ax_intercept.axvline(0, color="#555555", lw=0.8, ls="--")
    ax_intercept.set_yticks(y)
    ax_intercept.set_yticklabels(labels)
    ax_intercept.invert_yaxis()
    ax_intercept.set_xlabel("Posterior calibration intercept")
    ax_intercept.set_title("Competition-level offset")

    clean_axis(ax_slope)
    panel_label(ax_slope, "d")
    ax_slope.errorbar(
        beta["beta_mean"],
        y,
        xerr=np.vstack([beta["beta_mean"] - beta["beta_low"], beta["beta_high"] - beta["beta_mean"]]),
        fmt="o",
        color="#2D8A62",
        ecolor="#A8D1BF",
        elinewidth=1.2,
        capsize=2.5,
        ms=3.5,
    )
    ax_slope.axvline(1, color="#555555", lw=0.8, ls="--")
    ax_slope.set_yticks(y)
    ax_slope.set_yticklabels([])
    ax_slope.invert_yaxis()
    ax_slope.set_xlabel("Posterior calibration slope")
    ax_slope.set_title("Competition-level spread")

    save_jqas(fig, "fig5_bayesian_xg_visual")


def plot_supplementary_diagnostics():
    files = [
        ("fig_corr_heatmap_kbest.png", "Feature correlation"),
        ("fig_xg_surface_geom.png", "Geometry xG surface"),
        ("fig_player_goals_vs_xg_dumbbell.png", "Player goals vs xG"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.0, 0.8, 1.0]})
    for ax, (name, title), label in zip(axes, files, ["a", "b", "c"]):
        ax.set_axis_off()
        panel_label(ax, label)
        img = plt.imread(SOURCE_FIG_DIR / name)
        ax.imshow(img)
        ax.set_title(title, fontsize=8.5)
    save_jqas(fig, "figS1_supplementary_diagnostics")


def write_manifest():
    rows = [
        ["Figure 1", "fig1_pipeline_data", "main", "Pipeline and analytical sample", "Code/frontiers_figure_package.py", "pdf;tiff;png"],
        ["Figure 2", "fig2_validation_benchmark", "main", "Validation design and pooled benchmark", "Code/frontiers_figure_package.py", "pdf;tiff;png"],
        ["Figure 3", "fig3_worldcup_stress", "main", "World Cup small-sample stress test", "Code/frontiers_figure_package.py", "pdf;tiff;png"],
        ["Figure 4", "fig4_calibration_deployment", "main", "Calibration sample size and deployment recalibration", "Code/frontiers_figure_package.py", "pdf;tiff;png"],
        ["Figure 5", "fig5_bayesian_xg_visual", "main", "xG surface and Bayesian competition calibration", "Code/frontiers_figure_package.py", "pdf;tiff;png"],
        ["Figure 6", "fig6_feature_interpretation", "main", "Feature-family and partial-effect interpretation", "Code/frontiers_figure_package.py", "pdf;tiff;png"],
        ["Supplementary Figure S1", "figS1_supplementary_diagnostics", "supplementary", "Correlation, xG surface and player residual diagnostics", "Code/frontiers_figure_package.py", "pdf;tiff;png"],
    ]
    path = PACKAGE_DIR / "figure_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["figure_id", "file_stem", "placement", "purpose", "source_script", "outputs"])
        writer.writerows(rows)
    frontiers_path = FRONTIERS_PACKAGE_DIR / "figure_manifest.csv"
    if frontiers_path.parent.exists():
        with frontiers_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["figure_id", "file_stem", "placement", "purpose", "source_script", "outputs"])
            writer.writerows(rows)


def extract_one(pattern, text, label):
    m = re.search(pattern, text, flags=re.S)
    if not m:
        raise ValueError(f"Could not extract {label}")
    return m.group(1).strip()


def figure_block(filename, caption, label, width=r"\linewidth"):
    return (
        "\\begin{figure}[!ht]\n"
        "    \\centering\n"
        f"    \\includegraphics[width={width}]{{{filename}.pdf}}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        "\\end{figure}\n\n"
    )


def insert_after(text, marker, insertion):
    if marker not in text:
        raise ValueError(f"Marker not found: {marker[:80]}")
    return text.replace(marker, marker + "\n\n" + insertion, 1)


def build_manuscript():
    src = SOURCE_TEX.read_text(encoding="utf-8")
    title = extract_one(r"\\title\{(.*?)\}", src, "title")
    abstract = extract_one(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, "abstract")
    keywords = extract_one(r"\\keywords\{(.*?)\}", src, "keywords").replace(";", ",")
    body = src[src.index(r"\section{Introduction}") : src.index("%%Harvard")]

    body = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}\s*", "", body, flags=re.S)
    body = body.replace(r"\begin{table*}", r"\begin{table}")
    body = body.replace(r"\end{table*}", r"\end{table}")
    body = body.replace(r"\resizebox{\textwidth}{!}", r"\resizebox{\linewidth}{!}")
    body = body.replace(r"Figures~\ref{fig:perf_vs_features} and \ref{fig:delta_logloss}", r"Figure~\ref{fig:worldcup_stress}")
    body = body.replace(r"Figure~\ref{fig:calibration_sample_size}", r"Figure~\ref{fig:calibration_deployment}")
    body = body.replace(r"Figure~\ref{fig:feature_family}", r"Figure~\ref{fig:feature_interpretation}")
    body = body.replace(r"Figure~\ref{fig:target_recalibration}", r"Figure~\ref{fig:calibration_deployment}")
    body = body.replace(r"Figure~\ref{fig:partials}", r"Figure~\ref{fig:feature_interpretation}")
    body = re.sub(
        r"\nAs an additional qualitative check, Figure~\\ref\{fig:xg_surface\}.*?\n\\subsection\{Feature dependence and collinearity structure\}",
        lambda _: "\n\\subsection{Feature dependence and collinearity structure}",
        body,
        flags=re.S,
    )
    body = re.sub(
        r"\n\\subsection\{Feature dependence and collinearity structure\}.*?\n\\section\{Discussion\}",
        lambda _: "\n\\section{Discussion}",
        body,
        flags=re.S,
    )

    body = insert_after(
        body,
        r"Figure \ref{fig:pipeline} summarizes the end-to-end pipeline from raw event ingestion to the final shot-level evaluation dataset.",
        figure_block(
            "fig1_pipeline_data",
            "End-to-end xG evaluation workflow and analytical sample. Panel a summarizes data ingestion, feature construction, calibration-aware evaluation and evidence reporting; panel b shows shots and goal rates by competition.",
            "fig:pipeline",
        ),
    )
    body = body.replace(
        "\\end{table}\n\n\\subsection{Main pooled performance and nonlinear benchmark}",
        "\\end{table}\n\n"
        + figure_block(
            "fig2_validation_benchmark",
            "Validation-design sensitivity and pooled benchmark results. LogLoss remains similar across within-competition validation designs, whereas leave-one-competition-out validation mainly increases calibration error; the nonlinear benchmark gives a small pooled LogLoss improvement over the interpretable logistic model.",
            "fig:validation_design",
        )
        + "\\subsection{Main pooled performance and nonlinear benchmark}",
        1,
    )
    body = insert_after(
        body,
        "These World Cup-selected models are not used to define the main pooled model.",
        figure_block(
            "fig3_worldcup_stress",
            "World Cup small-sample stress test. Panels combine the feature path, stepwise LogLoss changes, representative model scores, ROC/PR curves and reliability diagnostics for the descriptive World Cup model sequence.",
            "fig:worldcup_stress",
        ),
    )
    body = insert_after(
        body,
        "The practical implication is not that target-domain calibration is unnecessary in general, but that small calibration sets can add variance and should be benchmarked against direct transfer and simple intercept adjustment.",
        figure_block(
            "fig4_calibration_deployment",
            "Calibration and deployment diagnostics. Flexible isotonic calibration is unstable at small sample sizes, and target-domain recalibration should be benchmarked against direct transfer when few target goals are available.",
            "fig:calibration_deployment",
        ),
    )
    body = insert_after(
        body,
        "In contrast, the previous-action speed proxy exhibits a weaker and less stable association with wider uncertainty bands, consistent with its smaller incremental contribution in the stepwise LogLoss analysis.",
        figure_block(
            "fig6_feature_interpretation",
            "Feature contribution and interpretation. Feature-family ablations show that geometry carries the dominant signal while execution and previous-event context add measurable improvement; binned partial effects recover the expected distance and angle patterns.",
            "fig:feature_interpretation",
        ),
    )

    preamble = rf"""\documentclass[USenglish]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[big,online]{{dgruyter}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage[numbers,square,sort&compress]{{natbib}}
\usepackage{{booktabs}}
\usepackage{{amsmath,amssymb}}
\graphicspath{{{{figures/}}{{logos/}}}}

\begin{{document}}

\articletype{{Research Article}}
\received{{Month DD, YYYY}}
\revised{{Month DD, YYYY}}
\accepted{{Month DD, YYYY}}
\journalname{{Journal of Quantitative Analysis in Sports}}
\journalyear{{2026}}
\journalvolume{{XX}}
\journalissue{{X}}
\startpage{{1}}
\aop
\DOI{{10.1515/jqas-2026-XXXX}}

\title{{{title}}}
\runningtitle{{Calibration-aware xG evaluation}}

\author*[1]{{Zexuan Xu}}
\author[1]{{Rahimi Bin Che Jusoh}}
\runningauthor{{Z. Xu and R. B. C. Jusoh}}
\affil[1]{{\protect\raggedright City University Malaysia, 6100 Petaling Jaya, Selangor, Malaysia, e-mail: xzxxx9731@gmail.com}}

\abstract{{{abstract}}}

\keywords{{{keywords}}}

\maketitle

"""
    tex = preamble + body + "\n\\bibliographystyle{abbrvnat}\n\\bibliography{ref}\n\n\\end{document}\n"
    (PACKAGE_DIR / "journal-article-xg-jqas.tex").write_text(tex, encoding="utf-8")

    supp = r"""\documentclass[USenglish]{article}
\usepackage[utf8]{inputenc}
\usepackage[big,online]{dgruyter}
\usepackage{lmodern}
\usepackage{microtype}
\graphicspath{{figures/}{logos/}}

\begin{document}
\section*{Supplementary diagnostic figures}
\begin{figure}[!ht]
    \centering
    \includegraphics[width=\linewidth]{figS1_supplementary_diagnostics.pdf}
    \caption{Supplementary diagnostics. Panel a shows collinearity in the World Cup-selected feature set, panel b shows the geometry-only xG surface, and panel c shows descriptive player-level goals-versus-xG residuals.}
    \label{fig:supp_diagnostics}
\end{figure}
\end{document}
"""
    (PACKAGE_DIR / "supplementary-figures.tex").write_text(supp, encoding="utf-8")


def copy_template_support_files():
    for name in ["dgruyter.sty", "dgruyter.ist", "dgruyter.xdy"]:
        src = JQAS_TEMPLATE_DIR / name
        if src.exists():
            shutil.copyfile(src, PACKAGE_DIR / name)
    logos_src = JQAS_TEMPLATE_DIR / "logos"
    if logos_src.exists():
        shutil.copytree(logos_src, PACKAGE_DIR / "logos", dirs_exist_ok=True)
    if SOURCE_BIB.exists():
        shutil.copyfile(SOURCE_BIB, PACKAGE_DIR / "ref.bib")


def main():
    set_pub_style(dpi=600)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FRONTIERS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    plot_pipeline_data()
    plot_validation_benchmark()
    world, cols, _, best_pred, _ = plot_worldcup_stress()
    plot_calibration_deployment()
    plot_bayesian_xg_visual()
    plot_feature_interpretation(world, cols, best_pred)
    plot_supplementary_diagnostics()
    write_manifest()
    copy_template_support_files()
    build_manuscript()
    print(f"Wrote JQAS package to {PACKAGE_DIR}")


if __name__ == "__main__":
    main()
