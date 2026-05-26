from pathlib import Path
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex_matplotlib")

import matplotlib.pyplot as plt


PALETTE = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#E69F00",
    "grey": "#666666",
    "light_grey": "#D9D9D9",
    "black": "#111111",
}


def set_pub_style(dpi: int = 300):
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.75,
            "axes.edgecolor": PALETTE["black"],
            "axes.grid": False,
            "grid.alpha": 0.15,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 8.0,
            "lines.linewidth": 1.5,
        }
    )


def clean_axis(ax, grid: bool = True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, alpha=0.15, linewidth=0.7)


def save_figure(fig, path, formats=("png",), dpi: int = 600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    suffix = path.suffix.lower().lstrip(".")
    if suffix:
        fig.savefig(path, dpi=dpi if suffix in {"png", "jpg", "jpeg", "tif", "tiff"} else None, bbox_inches="tight")
    else:
        for fmt in formats:
            fig.savefig(
                path.with_suffix(f".{fmt}"),
                dpi=dpi if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"} else None,
                bbox_inches="tight",
            )
    plt.close(fig)


def save_submission_figure(fig, stem, dpi: int = 600, preview: bool = True):
    formats = ("pdf", "tiff", "png") if preview else ("pdf", "tiff")
    save_figure(fig, Path(stem), formats=formats, dpi=dpi)
