
import csv

import sr_figure_shared as shared


SR_PACKAGE_DIR = shared.ROOT / "SR_Submission"
SR_FIG_DIR = SR_PACKAGE_DIR / "figures"
SUPPLEMENTARY_SOURCE_FILES = (
    "fig_corr_heatmap_kbest.png",
    "fig_xg_surface_geom.png",
    "fig_player_goals_vs_xg_dumbbell.png",
)


def write_sr_manifest(include_supplementary: bool) -> None:
    rows = [
        ["Figure 1", "fig1_pipeline_data", "main", "Pipeline and analytical sample", "Code/sr_figure_package.py", "pdf;tiff;png"],
        ["Figure 2", "fig2_validation_benchmark", "main", "Validation design and pooled benchmark", "Code/sr_figure_package.py", "pdf;tiff;png"],
        ["Figure 3", "fig3_calibration_deployment", "main", "Calibration sample size and deployment recalibration", "Code/sr_figure_package.py", "pdf;tiff;png"],
        ["Figure 4", "fig4_worldcup_stress", "main", "World Cup small-sample stress test", "Code/sr_figure_package.py", "pdf;tiff;png"],
        ["Figure 5", "fig5_feature_interpretation", "main", "Feature-family and partial-effect interpretation", "Code/sr_figure_package.py", "pdf;tiff;png"],
        ["Figure 6", "fig6_bayesian_xg_visual", "main", "xG surface and Bayesian competition calibration", "Code/sr_figure_package.py", "pdf;tiff;png"],
    ]
    if include_supplementary:
        rows.append(
            [
                "Supplementary Figure S1",
                "figS1_supplementary_diagnostics",
                "supplementary",
                "Correlation, xG surface and player residual diagnostics",
                "Code/sr_figure_package.py",
                "pdf;tiff;png",
            ]
        )
    path = SR_PACKAGE_DIR / "figure_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["figure_id", "file_stem", "placement", "purpose", "source_script", "outputs"])
        writer.writerows(rows)


def main() -> None:
    shared.set_pub_style(dpi=600)
    shared.FIG_DIR = SR_FIG_DIR
    SR_FIG_DIR.mkdir(parents=True, exist_ok=True)

    shared.plot_pipeline_data()
    shared.plot_validation_benchmark()
    world, cols, _, best_pred, _ = shared.plot_worldcup_stress()
    shared.plot_calibration_deployment()
    shared.plot_bayesian_xg_visual()
    shared.plot_feature_interpretation(world, cols, best_pred)
    include_supplementary = all((shared.SOURCE_FIG_DIR / name).exists() for name in SUPPLEMENTARY_SOURCE_FILES)
    if include_supplementary:
        shared.plot_supplementary_diagnostics()
    write_sr_manifest(include_supplementary)
    print(f"Wrote SR figures to {SR_FIG_DIR}")


if __name__ == "__main__":
    main()
