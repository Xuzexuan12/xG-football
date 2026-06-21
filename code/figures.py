from pathlib import Path
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from multicomp_analysis import OUT_DIR, ROOT, clip01, oof_predictions, usable_features
from plot_style import PALETTE, clean_axis, save_figure, set_pub_style


FIG_DIR = ROOT / "SR_Submission" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    save_figure(fig, FIG_DIR / name)


def selected_k(perf):
    best = perf.loc[perf["LogLoss"].idxmin()]
    threshold = float(best["LogLoss"] + best["LogLoss_SE"])
    k_star = int(perf.loc[perf["LogLoss"] <= threshold, "k"].min())
    return k_star, int(best["k"])


def plot_perf_curve(perf, k_star, k_best):
    x = perf["k"].values
    r2 = perf["R2"].values
    ll = perf["LogLoss"].values
    se = perf["LogLoss_SE"].values

    fig, ax1 = plt.subplots(figsize=(11.2, 5.8))
    clean_axis(ax1)
    ax1.plot(x, r2, color=PALETTE["blue"], lw=2.3, marker="o", ms=5, label="Nagelkerke $R^2$")
    ax1.set_xlabel("Number of features")
    ax1.set_ylabel("$R^2$", color=PALETTE["blue"])
    ax1.tick_params(axis="y", colors=PALETTE["blue"])

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(x, ll, color=PALETTE["vermillion"], lw=2.3, marker="D", ms=5, label="LogLoss")
    ax2.fill_between(x, ll - se, ll + se, color=PALETTE["vermillion"], alpha=0.18, linewidth=0)
    ax2.set_ylabel("LogLoss", color=PALETTE["vermillion"])
    ax2.tick_params(axis="y", colors=PALETTE["vermillion"])

    ax1.axvline(k_star, color=PALETTE["grey"], ls="--", lw=1.4)
    ax1.text(k_star + 0.2, ax1.get_ylim()[0] + 0.04 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]), f"$k^*$={k_star}")
    ax1.axvline(k_best, color=PALETTE["grey"], ls=":", lw=1.6)
    ax1.text(k_best + 0.2, ax1.get_ylim()[0] + 0.11 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]), f"$k_{{best}}$={k_best}")

    lines = ax1.get_lines()[:1] + ax2.get_lines()[:1]
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper center", ncol=2)
    save(fig, "fig_perf_vs_features_pub.png")

    d = np.r_[0.0, np.diff(ll)]
    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    clean_axis(ax)
    colors = [PALETTE["grey"]] + [PALETTE["blue"] if v < 0 else PALETTE["vermillion"] for v in d[1:]]
    ax.bar(x, d, color=colors, width=0.72, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=PALETTE["grey"], lw=1.0)
    ax.set_xlabel("Number of features")
    ax.set_ylabel(r"$\Delta$ LogLoss")
    save(fig, "fig_delta_logloss_pub.png")


def plot_roc_pr_reliability(y, pred_dict):
    line_colors = [PALETTE["grey"], PALETTE["blue"], PALETTE["vermillion"]]

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    clean_axis(ax)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.1, color=PALETTE["grey"])
    for (label, p), color in zip(pred_dict.items(), line_colors):
        p = clip01(p)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2.2, color=color, label=f"{label} (AUC={roc_auc_score(y, p):.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    save(fig, "roc_pub.png")

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    clean_axis(ax)
    ax.hlines(float(np.mean(y)), 0, 1, ls="--", lw=1.1, color=PALETTE["grey"])
    for (label, p), color in zip(pred_dict.items(), line_colors):
        p = clip01(p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, lw=2.2, color=color, label=f"{label} (AP={average_precision_score(y, p):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    save(fig, "pr_pub.png")

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    clean_axis(ax)
    ax.plot([0, 0.8], [0, 0.8], ls="--", lw=1.1, color=PALETTE["grey"])
    q = np.linspace(0, 1, 11)
    for (label, p), color in zip(pred_dict.items(), line_colors):
        p = clip01(p)
        bins = np.quantile(p, q)
        bins[0], bins[-1] = -np.inf, np.inf
        xs, ys, ns = [], [], []
        for i in range(10):
            m = (p >= bins[i]) & (p < bins[i + 1])
            if m.any():
                xs.append(float(p[m].mean()))
                ys.append(float(np.mean(y[m])))
                ns.append(int(m.sum()))
        ax.plot(xs, ys, marker="o", lw=2.0, color=color, label=label)
        for x0, y0, n in zip(xs[::3], ys[::3], ns[::3]):
            ax.text(x0, y0 + 0.018, str(n), color=color, fontsize=7, ha="center")
    ax.set_xlabel("Mean predicted xG")
    ax.set_ylabel("Observed goal rate")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.legend(loc="upper left")
    save(fig, "fig_reliability_pub.png")


def plot_corr(world, cols_best):
    corr = world[cols_best].astype(float).corr()
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols_best)))
    ax.set_yticks(range(len(cols_best)))
    labels = [c.replace("feat_", "").replace("_", " ") for c in cols_best]
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "fig_corr_heatmap_kbest.png")


def plot_validation_design():
    df = pd.read_csv(OUT_DIR / "validation_design_sensitivity.csv")
    order = ["Random shot-level CV", "Match-grouped CV", "Shooting-team blocked CV", "Leave-one-competition-out"]
    df = df.set_index("validation_design").loc[order].reset_index()
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    clean_axis(ax)
    ax.bar(x, df["LogLoss"], color=PALETTE["blue"], width=0.62, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("LogLoss")
    ax.set_xticks(x)
    ax.set_xticklabels(["Random shot", "Match grouped", "Shooting-team", "LOGO"], rotation=0)
    ax.set_ylim(max(0, df["LogLoss"].min() - 0.02), df["LogLoss"].max() + 0.02)
    for i, val in enumerate(df["LogLoss"]):
        ax.text(i, val + 0.003, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax2 = ax.twinx()
    ax2.plot(x, df["ECE"], color=PALETTE["vermillion"], marker="D", lw=2.0, label="ECE")
    ax2.set_ylabel("ECE", color=PALETTE["vermillion"])
    ax2.tick_params(axis="y", colors=PALETTE["vermillion"])
    ax2.spines["top"].set_visible(False)
    ax2.legend(loc="upper left")
    save(fig, "fig_validation_design_sensitivity.png")


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


def plot_calibration_sample_size():
    df = pd.read_csv(OUT_DIR / "calibration_sample_size_stability.csv")
    colors = {"none": PALETTE["grey"], "sigmoid": PALETTE["blue"], "isotonic": PALETTE["vermillion"]}
    labels = {"none": "Uncalibrated", "sigmoid": "Sigmoid", "isotonic": "Isotonic"}
    metrics = [("LogLoss", "LogLoss"), ("Brier", "Brier"), ("ECE", "ECE")]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        clean_axis(ax)
        agg = aggregate_with_band(df, ["train_fraction", "calibration"], metric)
        for calibration in ["none", "sigmoid", "isotonic"]:
            g = agg[agg["calibration"] == calibration].sort_values("train_fraction")
            x = g["train_goals"].values
            y = g["mean"].values
            e = g["se95"].values
            ax.plot(x, y, color=colors[calibration], marker="o", lw=2.0, label=labels[calibration])
            ax.fill_between(x, y - e, y + e, color=colors[calibration], alpha=0.15, linewidth=0)
        ax.set_xlabel("Mean training goals")
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="best")
    save(fig, "fig_calibration_sample_size.png")


def plot_feature_family():
    df = pd.read_csv(OUT_DIR / "feature_family_ablation.csv")
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
    settings = ["Pooled match-grouped CV", "World Cup match-grouped CV"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8), sharey=True)
    for ax, setting in zip(axes, settings):
        clean_axis(ax)
        g = df[df["setting"] == setting].set_index("feature_family_model").loc[order].reset_index()
        improvement = -g["delta_logloss_vs_geometry"].values
        y = np.arange(len(g))
        colors = [PALETTE["grey"] if name == "Geometry only" else PALETTE["green"] for name in g["feature_family_model"]]
        ax.barh(y, improvement, color=colors, edgecolor="white", linewidth=0.8)
        ax.axvline(0, color=PALETTE["grey"], lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(g["feature_family_model"])
        ax.invert_yaxis()
        ax.set_xlabel("LogLoss improvement vs geometry only")
        ax.set_title(setting.replace(" match-grouped CV", ""))
    save(fig, "fig_feature_family_ablation.png")


def plot_target_recalibration():
    df = pd.read_csv(OUT_DIR / "target_domain_recalibration.csv")
    strategy_order = ["direct transfer", "intercept recalibration", "sigmoid recalibration", "isotonic recalibration"]
    colors = {
        "direct transfer": PALETTE["grey"],
        "intercept recalibration": PALETTE["green"],
        "sigmoid recalibration": PALETTE["blue"],
        "isotonic recalibration": PALETTE["vermillion"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), sharex=True)
    for ax, metric in zip(axes, ["LogLoss", "ECE"]):
        clean_axis(ax)
        agg = aggregate_with_band(df, ["calibration_fraction", "strategy"], metric)
        for strategy in strategy_order:
            g = agg[agg["strategy"] == strategy].sort_values("calibration_fraction")
            x = g["calibration_goals"].values
            y = g["mean"].values
            e = g["se95"].values
            ax.plot(x, y, color=colors[strategy], marker="o", lw=2.0, label=strategy.title())
            ax.fill_between(x, y - e, y + e, color=colors[strategy], alpha=0.14, linewidth=0)
        ax.set_xlabel("Mean target calibration goals")
        ax.set_ylabel(metric)
    axes[0].legend(loc="best", fontsize=8)
    save(fig, "fig_target_recalibration.png")


def main():
    set_pub_style()
    all_shots = pd.read_csv(OUT_DIR / "all_competition_shots_features.csv")
    world = all_shots[all_shots["competition"] == "World_Cup"].copy()
    cols = usable_features(all_shots)
    perf = pd.read_csv(OUT_DIR / "worldcup_perf_curve.csv")
    k_star, k_best = selected_k(perf)
    plot_perf_curve(perf, k_star, k_best)

    pred_dict = {}
    eval_df = None
    y_common = None
    for label, model_cols in [
        ("Baseline(1)", cols[:1]),
        (f"Chosen(k*={k_star})", cols[:k_star]),
        (f"Best(k={k_best})", cols[:k_best]),
    ]:
        d, y, p = oof_predictions(world, model_cols, "logistic", "isotonic")
        pred_dict[label] = p
        eval_df = d
        y_common = y
    plot_roc_pr_reliability(y_common, pred_dict)
    plot_corr(eval_df, cols[:k_best])

    plot_validation_design()
    plot_calibration_sample_size()
    plot_feature_family()
    plot_target_recalibration()


if __name__ == "__main__":
    main()
