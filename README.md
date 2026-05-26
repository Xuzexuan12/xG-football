# Reproducible xG Analysis for Elite Football Competitions

This repository contains the code, derived tables, and manuscript-supporting figures for the expected-goals (xG) analysis used in the accompanying paper.

## Repository Contents

- `Code/`: Python analysis and figure-generation scripts.
- `frontiers_submission/results/`: derived CSV outputs used for manuscript tables, robustness checks, calibration analyses, and supplementary results.
- `figures/`: publication figures exported from the analysis pipeline.
- `data/`: local data workspace. Could download at Statmbob.

## Main Reproduction Commands

Create a Python environment and install dependencies:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the main multi-competition analysis:

```sh
python Code/revision_multicomp_analysis.py
```

Run the Bayesian calibration and finishing-residual analyses:

```sh
python Code/bayesian_xg_analysis.py --draws 1000 --tune 1000 --chains 4 --target-accept 0.92
```

Regenerate manuscript figures:

```sh
python Code/frontiers_figure_package.py
```

## Data Availability

The analysis expects Wyscout-style event and match JSON files under:

```text
data/events/
data/matches/
```

See `RAW_DATA_INVENTORY.md` for the exact raw files, record counts, fields, and the distinction between raw inputs and derived outputs.

Before public release, verify whether the raw event and match files can be redistributed under their source license. If redistribution is not permitted, publish only:

- data-source instructions;
- processed, non-sensitive derived tables needed to verify the reported results;
- checksums or filenames for the raw files expected by the scripts;
- generated result CSVs in `frontiers_submission/results/`.


## Reproducibility Notes

The current workflow writes outputs to both `sage_latex_template_4/revision_outputs/` and `frontiers_submission/results/`. For the public GitHub version, use `frontiers_submission/results/` as the canonical result-output directory and treat old template folders as local working material.
