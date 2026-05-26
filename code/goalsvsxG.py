
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

player_csv_path = "./Final_results/player_xg_summary.csv"
outdir = os.path.dirname(player_csv_path)
PUB = {
    "blue":  "#1f77b4",
    "orange":"#ff7f0e",
    "grey":  "#7f7f7f",
    "black": "#111111",
}

def set_pub_style(base_fontsize=11, font_family="DejaVu Sans", dpi=240):
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": font_family,
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 2,
        "axes.labelsize": base_fontsize + 1,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#222222",
        "axes.grid": False,
        "legend.frameon": False,
    })

def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

def _grid(ax):
    ax.grid(True, alpha=0.18, linewidth=0.8)

def _savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def decode_unicode_escapes(s: str) -> str:
    if not isinstance(s, str):
        return s
    if "\\u" in s or "\\U" in s:
        try:
            return s.encode("utf-8").decode("unicode_escape")
        except Exception:
            return s
    return s

def clean_player_names(df: pd.DataFrame, col="PLAYER") -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].astype(str).apply(decode_unicode_escapes)
    return df


def load_player_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Basic column check
    required = ["PLAYER", "APPS", "MINS", "GOALS", "XG", "GOALS VS XG", "SHOTS", "SOT", "CONV %", "XG PER SHOT"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"player_xg_summary.csv missing columns: {miss}")

    # Numeric conversion
    num_cols = ["APPS", "MINS", "GOALS", "XG", "GOALS VS XG", "SHOTS", "SOT", "CONV %", "XG PER SHOT"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = clean_player_names(df, col="PLAYER")
    return df

def plot_player_goals_xg_dumbbell(
    df: pd.DataFrame,
    outdir: str,
    min_shots: int = 5,
    top_n: int = 25,
    sort_mode: str = "extremes",  # "extremes" (by abs gap) OR "xg" (by xg)
    save_name: str = "fig_player_goals_vs_xg_dumbbell.png"
):
    d = df.copy()
    d = d[d["SHOTS"] >= min_shots].dropna(subset=["GOALS", "XG", "GOALS VS XG"]).copy()
    if len(d) == 0:
        raise ValueError("No players left after filtering. Reduce min_shots.")

    if sort_mode == "extremes":
        d["_abs_gap_"] = np.abs(d["GOALS VS XG"].values)
        d = d.sort_values("_abs_gap_", ascending=False).head(top_n).copy()
        d = d.sort_values("GOALS VS XG", ascending=True)  # under -> over
        d.drop(columns=["_abs_gap_"], inplace=True)
        title = f"Goals vs xG (largest |Goals−xG|, min shots ≥ {min_shots})"
    else:
        d = d.sort_values("XG", ascending=False).head(top_n).copy()
        d = d.sort_values("XG", ascending=True)
        title = f"Goals vs xG (top xG, min shots ≥ {min_shots})"

    y = np.arange(1, len(d) + 1)
    goals = d["GOALS"].values.astype(float)
    xg = d["XG"].values.astype(float)
    gap = d["GOALS VS XG"].values.astype(float)

    xmax = float(max(goals.max(), xg.max(), 1.0))
    pad = 0.14 * xmax

    fig, ax = plt.subplots(figsize=(10.4, 0.38 * len(d) + 2.6))
    _despine(ax)
    _grid(ax)

    # line: connect min->max between goals and xg
    xmin_line = np.minimum(goals, xg)
    xmax_line = np.maximum(goals, xg)
    ax.hlines(y=y, xmin=xmin_line, xmax=xmax_line, color=PUB["grey"], alpha=0.35, linewidth=3)

    # points
    ax.scatter(goals, y, s=120, color=PUB["blue"], edgecolor="white", linewidth=0.6,
               label="Goals", zorder=3)
    ax.scatter(xg, y, s=135, color=PUB["orange"], marker="X", edgecolor="white", linewidth=0.6,
               label="xG", zorder=3)

    # annotate diff at right
    for yi, g, e, diff in zip(y, goals, xg, gap):
        x_pos = max(g, e) + 0.04 * xmax
        c = PUB["blue"] if diff >= 0 else PUB["orange"]
        ax.text(x_pos, yi, f"{diff:+.2f}", va="center", ha="left", color=c, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(d["PLAYER"].astype(str).values)
    ax.set_xlabel("Goals and xG (sum)")
    ax.set_xlim(0, xmax + pad)
    # ax.set_title(title, weight="bold")
    ax.legend(loc="lower right")

    path = os.path.join(outdir, save_name)
    return _savefig(fig, path)


def plot_player_goals_minus_xg_rank(
    df: pd.DataFrame,
    outdir: str,
    top_n: int = 15,
    min_shots: int = 5,
    save_name: str = "fig_player_goals_minus_xg_rank.png"
):
    d = df.copy()
    d = d[d["SHOTS"] >= min_shots].dropna(subset=["GOALS VS XG"]).copy()
    if len(d) == 0:
        raise ValueError("No players left after filtering. Reduce min_shots.")

    d = d.sort_values("GOALS VS XG", ascending=False)
    top = d.head(top_n)
    bot = d.tail(top_n)
    plot_df = pd.concat([top, bot], axis=0).copy()
    plot_df = plot_df.sort_values("GOALS VS XG", ascending=True)  # bottom -> top

    y = np.arange(len(plot_df))
    vals = plot_df["GOALS VS XG"].values.astype(float)
    colors = [PUB["blue"] if v >= 0 else PUB["orange"] for v in vals]

    fig, ax = plt.subplots(figsize=(9.6, 0.42 * len(plot_df) + 2.2))
    _despine(ax)
    _grid(ax)

    ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color=PUB["grey"], lw=1.2)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["PLAYER"].astype(str).values)
    ax.set_xlabel("Goals − xG")
    ax.set_title(f"Over/Under-performance by player (min shots ≥ {min_shots})", weight="bold")

    for yi, v in zip(y, vals):
        ax.text(v + (0.03 if v >= 0 else -0.03),
                yi,
                f"{v:+.2f}",
                va="center",
                ha="left" if v >= 0 else "right",
                color=PUB["black"])

    path = os.path.join(outdir, save_name)
    return _savefig(fig, path)

def plot_player_goals_minus_xg_distribution(
    df: pd.DataFrame,
    outdir: str,
    min_shots: int = 5,
    bins: int = 20,
    save_name: str = "fig_player_goals_minus_xg_dist.png"
):
    d = df.copy()
    d = d[d["SHOTS"] >= min_shots].dropna(subset=["GOALS VS XG"]).copy()
    if len(d) == 0:
        raise ValueError("No players left after filtering. Reduce min_shots.")

    v = d["GOALS VS XG"].values.astype(float)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    _despine(ax)
    _grid(ax)

    ax.hist(v, bins=bins, density=True, alpha=0.35, color=PUB["blue"],
            edgecolor="white", linewidth=0.6)
    ax.axvline(0, color=PUB["grey"], lw=1.2, ls="--")

    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 1e-6
    xs = np.linspace(v.min(), v.max(), 300)
    pdf = (1.0 / (sd * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
    ax.plot(xs, pdf, lw=2.0, color=PUB["orange"], label=f"Normal fit (μ={mu:.2f}, σ={sd:.2f})")

    ax.set_xlabel("Goals − xG")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of Goals − xG (min shots ≥ {min_shots})", weight="bold")
    ax.legend(loc="upper right")

    path = os.path.join(outdir, save_name)
    return _savefig(fig, path)

def run_all_plots():
    set_pub_style(base_fontsize=11, font_family="DejaVu Sans", dpi=240)
    df = load_player_summary(player_csv_path)

    p1 = plot_player_goals_xg_dumbbell(
        df, outdir,
        min_shots=5,
        top_n=25,
        sort_mode="extremes",
        save_name="fig_player_goals_vs_xg_dumbbell.png"
    )
    p2 = plot_player_goals_minus_xg_rank(
        df, outdir,
        top_n=15,
        min_shots=5,
        save_name="fig_player_goals_minus_xg_rank.png"
    )
    p3 = plot_player_goals_minus_xg_distribution(
        df, outdir,
        min_shots=5,
        bins=20,
        save_name="fig_player_goals_minus_xg_dist.png"
    )

    print("[DONE] Saved figures:")
    print(" -", p1)
    print(" -", p2)
    print(" -", p3)


if __name__ == "__main__":
    run_all_plots()
