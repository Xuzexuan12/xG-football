import csv
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from multicomp_analysis import OUT_DIR, ROOT, clip01, oof_predictions, usable_features
from plot_style import PALETTE, clean_axis, save_submission_figure, set_pub_style


SR_PACKAGE_DIR = ROOT / "SR_Submission"
FIG_DIR = SR_PACKAGE_DIR / "figures"
SOURCE_FIG_DIR = FIG_DIR


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


def save_sr_figure(fig, stem):
    save_submission_figure(fig, FIG_DIR / stem, dpi=600, preview=True)


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
    save_sr_figure(fig, "fig1_pipeline_data")


def plot_validation_benchmark():
    set_pub_style(dpi=300)
    val = pd.read_csv(OUT_DIR / "validation_design_sensitivity.csv")
    metrics = pd.read_csv(OUT_DIR / "model_metrics.csv")
    pooled = metrics[metrics["setting"] == "Pooled grouped CV"].copy()
    boot = pd.read_csv(OUT_DIR / "pooled_bootstrap_logloss_diff.csv")
    order = ["Random shot-level CV", "Match-grouped CV", "Shooting-team blocked CV", "Leave-one-competition-out"]
    val = val.set_index("validation_design").loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [1.35, 1.0, 1.15], "wspace": 0.52})
    ax = axes[0]
    clean_axis(ax)
    panel_label(ax, "a")
    labels = ["Random shot", "Match grouped", "Team blocked", "Leave-one\ncompetition"]
    y = np.arange(len(val))[::-1]
    xerr = np.vstack([val["LogLoss"] - val["LogLoss_ci_low"], val["LogLoss_ci_high"] - val["LogLoss"]])
    ax.errorbar(
        val["LogLoss"],
        y,
        xerr=xerr,
        fmt="o",
        color=METHOD_COLORS["sigmoid"],
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2.4,
        markersize=4.0,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("LogLoss")
    ax.set_title("Validation design")
    ax.set_xlim(0.236, 0.302)
    ax.grid(True, axis="x", alpha=0.16)
    ax.grid(False, axis="y")
    ece_x = 0.298
    ax.text(ece_x, y.max() + 0.46, "ECE", ha="center", va="bottom", fontsize=6.6, color=METHOD_COLORS["isotonic"])
    for yi, ece in zip(y, val["ECE"]):
        ax.text(ece_x, yi, f"{ece:.3f}", ha="center", va="center", fontsize=6.3, color=METHOD_COLORS["isotonic"])
    ax.text(0.236, -0.70, "Error bars: split-level 95% interval", ha="left", va="top", fontsize=5.8, color="#555555")

    ax = axes[1]
    clean_axis(ax)
    panel_label(ax, "b")
    model_order = [
        "Full logistic uncalibrated",
        "Full logistic sigmoid",
        "Full logistic isotonic",
        "HGB sigmoid fixed comparator",
    ]
    p = pooled.set_index("model").loc[model_order].reset_index()
    p["short"] = ["Uncalibrated", "Sigmoid", "Isotonic", "HGB"]
    p = p.sort_values("LogLoss", ascending=True)
    y = np.arange(len(p))
    colors = [METHOD_COLORS["none"], METHOD_COLORS["sigmoid"], METHOD_COLORS["isotonic"], METHOD_COLORS["hgb"]]
    color_map = dict(zip(["Uncalibrated", "Sigmoid", "Isotonic", "HGB"], colors))
    ax.barh(y, p["LogLoss"], color=[color_map[v] for v in p["short"]], edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(p["short"])
    ax.set_xlabel("Pooled LogLoss")
    ax.set_title("Calibration choice")
    ax.set_xlim(p["LogLoss"].min() - 0.001, p["LogLoss"].max() + 0.001)
    for yi, val_logloss in zip(y, p["LogLoss"]):
        ax.text(val_logloss + 0.00012, yi, f"{val_logloss:.4f}", va="center", fontsize=5.8)

    ax = axes[2]
    clean_axis(ax)
    panel_label(ax, "c")
    comparisons = boot.copy()
    comparisons["label"] = comparisons["comparison"].map(
        {
            "HGB sigmoid fixed comparator minus Full logistic sigmoid": "HGB - logistic",
            "Full logistic isotonic minus Full logistic sigmoid": "Isotonic - sigmoid",
        }
    )
    comparisons = comparisons.dropna(subset=["label"]).iloc[::-1].reset_index(drop=True)
    y = np.arange(len(comparisons))
    xerr = np.vstack(
        [
            comparisons["mean_diff"] - comparisons["ci_low"],
            comparisons["ci_high"] - comparisons["mean_diff"],
        ]
    )
    colors = [METHOD_COLORS["hgb"] if "HGB" in label else METHOD_COLORS["isotonic"] for label in comparisons["label"]]
    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.errorbar(
        comparisons["mean_diff"],
        y,
        xerr=xerr,
        fmt="o",
        color="#222222",
        ecolor="#222222",
        capsize=3,
        elinewidth=0.9,
        markersize=0,
    )
    ax.scatter(comparisons["mean_diff"], y, s=28, color=colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(comparisons["label"])
    ax.set_xlabel(r"$\Delta$ LogLoss")
    ax.set_title("Bootstrap model contrasts")
    ax.set_xlim(-0.0034, 0.0050)
    ax.set_ylim(-0.45, len(comparisons) - 0.55)
    for yi, row in zip(y, comparisons.to_dict("records")):
        ax.text(
            row["mean_diff"],
            yi + 0.18,
            f"{row['mean_diff']:.4f}",
            ha="center",
            va="bottom",
            fontsize=5.8,
            color="#333333",
        )
    save_sr_figure(fig, "fig2_validation_benchmark")


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
    return world, cols, y_common, pred_dict[f"Best(k={k_best})"], save_sr_figure(fig, "fig4_worldcup_stress")


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
    save_sr_figure(fig, "fig3_calibration_deployment")


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
    save_sr_figure(fig, "fig5_feature_interpretation")


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

    save_sr_figure(fig, "fig6_bayesian_xg_visual")


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
    save_sr_figure(fig, "figS1_supplementary_diagnostics")


def write_manifest():
    rows = [
        ["Figure 1", "fig1_pipeline_data", "main", "Pipeline and analytical sample", "Code/sr_figure_package.py", "pdf;tiff;png"],
        ["Figure 2", "fig2_validation_benchmark", "main", "Validation design and pooled benchmark", "Code/sr_figure_package.py", "pdf;tiff;png"],
        ["Figure 3", "fig3_calibration_deployment", "main", "Calibration sample size and deployment recalibration", "Code/sr_figure_package.py", "pdf;tiff;png"],
        ["Figure 4", "fig4_worldcup_stress", "main", "World Cup small-sample stress test", "Code/sr_figure_package.py", "pdf;tiff;png"],
        ["Figure 5", "fig5_feature_interpretation", "main", "Feature-family and partial-effect interpretation", "Code/sr_figure_package.py", "pdf;tiff;png"],
        ["Figure 6", "fig6_bayesian_xg_visual", "main", "xG surface and Bayesian competition calibration", "Code/sr_figure_package.py", "pdf;tiff;png"],
        ["Supplementary Figure S1", "figS1_supplementary_diagnostics", "supplementary", "Correlation, xG surface and player residual diagnostics", "Code/sr_figure_package.py", "pdf;tiff;png"],
    ]
    path = SR_PACKAGE_DIR / "figure_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["figure_id", "file_stem", "placement", "purpose", "source_script", "outputs"])
        writer.writerows(rows)


def main():
    set_pub_style(dpi=600)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_pipeline_data()
    plot_validation_benchmark()
    world, cols, _, best_pred, _ = plot_worldcup_stress()
    plot_calibration_deployment()
    plot_bayesian_xg_visual()
    plot_feature_interpretation(world, cols, best_pred)
    plot_supplementary_diagnostics()
    write_manifest()
    print(f"Wrote SR figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
