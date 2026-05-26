"""Bayesian calibration and finishing-residual analyses for the xG study."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=/private/tmp/codex_pytensor")
os.environ.setdefault("AESARA_FLAGS", "base_compiledir=/private/tmp/codex_aesara,blas__ldflags=")

import arviz as az
import numpy as np
import pandas as pd

from revision_multicomp_analysis import (
    COMPETITIONS,
    FRONTIERS_OUT_DIR,
    OUT_DIR,
    fit_model,
    logit,
    metrics,
    predict_model,
    usable_features,
)


SEED = 42


@dataclass(frozen=True)
class PriorSpec:
    name: str
    mu_alpha_sigma: float
    sigma_alpha_sigma: float
    mu_beta_sigma: float
    sigma_beta_sigma: float


PRIOR_SPECS = {
    "baseline": PriorSpec("baseline", 1.0, 0.75, 0.5, 0.35),
    "wider": PriorSpec("wider", 1.5, 1.0, 0.75, 0.5),
    "conservative": PriorSpec("conservative", 0.5, 0.35, 0.25, 0.2),
}


def _require_pymc():
    try:
        import pymc as pm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyMC is required for Bayesian analysis. Install it with "
            "`python3 -m pip install pymc` and rerun this script."
        ) from exc
    return pm


def _mirror_to_frontiers(path: Path) -> None:
    FRONTIERS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = FRONTIERS_OUT_DIR / path.name
    if path.resolve() != target.resolve():
        target.write_bytes(path.read_bytes())


def _write_csv(df: pd.DataFrame, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    _mirror_to_frontiers(path)
    return path


def make_loco_predictions(all_shots: pd.DataFrame, cols: list[str], force: bool = False) -> pd.DataFrame:
    out_path = OUT_DIR / "bayesian_loco_predictions.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    rows = []
    for comp in COMPETITIONS:
        train = all_shots[all_shots["competition"] != comp].copy()
        test = all_shots[all_shots["competition"] == comp].copy()
        model = fit_model(train, cols, "logistic", "sigmoid_grouped")
        pred_df, y, p = predict_model(model, test, cols)
        pred_df = pred_df.reset_index(drop=True).copy()
        pred_df["held_out"] = comp.replace("_", " ")
        pred_df["loco_xg"] = p
        pred_df["loco_y"] = y
        rows.append(
            pred_df[
                [
                    "competition",
                    "held_out",
                    "id",
                    "matchId",
                    "teamId",
                    "playerId",
                    "shot_type",
                    "is_goal",
                    "loco_y",
                    "loco_xg",
                    "xg_all_shot",
                    "ax",
                    "ay",
                ]
            ]
        )
    pred = pd.concat(rows, ignore_index=True)
    _write_csv(pred, "bayesian_loco_predictions.csv")

    diag_rows = []
    for comp, grp in pred.groupby("held_out", sort=False):
        row = metrics(grp["loco_y"].astype(int).values, grp["loco_xg"].astype(float).values, include_calibration_params=True)
        row.update({"held_out": comp, "model": "LOCO logistic sigmoid grouped-inner row-level"})
        diag_rows.append(row)
    _write_csv(pd.DataFrame(diag_rows), "bayesian_loco_prediction_metrics.csv")
    return pred


def fit_competition_calibration(
    pred: pd.DataFrame,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    prior: PriorSpec = PRIOR_SPECS["baseline"],
):
    pm = _require_pymc()
    d = pred.dropna(subset=["loco_xg", "loco_y", "held_out"]).copy()
    d["loco_xg"] = np.clip(d["loco_xg"].astype(float), 1e-6, 1 - 1e-6)
    y = d["loco_y"].astype("int8").values
    z = logit(d["loco_xg"].values).astype("float64")
    comp_codes, comp_names = pd.factorize(d["held_out"], sort=True)
    comp_codes = comp_codes.astype("int32")

    coords = {"competition": list(comp_names), "shot": np.arange(len(d))}
    with pm.Model(coords=coords) as model:
        comp_idx = pm.Data("comp_idx", comp_codes, dims="shot")
        z_data = pm.Data("logit_xg", z, dims="shot")
        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=prior.mu_alpha_sigma)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=prior.sigma_alpha_sigma)
        mu_beta = pm.Normal("mu_beta", mu=1.0, sigma=prior.mu_beta_sigma)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=prior.sigma_beta_sigma)
        alpha_raw = pm.Normal("alpha_raw", mu=0.0, sigma=1.0, dims="competition")
        beta_raw = pm.Normal("beta_raw", mu=0.0, sigma=1.0, dims="competition")
        alpha = pm.Deterministic("alpha", mu_alpha + alpha_raw * sigma_alpha, dims="competition")
        beta = pm.Deterministic("beta", mu_beta + beta_raw * sigma_beta, dims="competition")
        pm.Bernoulli("goal", logit_p=alpha[comp_idx] + beta[comp_idx] * z_data, observed=y, dims="shot")
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=SEED,
            target_accept=target_accept,
            return_inferencedata=True,
        )
    return idata, list(comp_names)


def _competition_parameter_summary(idata) -> pd.DataFrame:
    summary = az.summary(idata, var_names=["alpha", "beta"], hdi_prob=0.95).reset_index()
    summary = summary.rename(
        columns={
            "index": "parameter_index",
            "hdi_2.5%": "hdi_2_5",
            "hdi_97.5%": "hdi_97_5",
            "ess_bulk": "ess_bulk",
            "ess_tail": "ess_tail",
            "r_hat": "r_hat",
        }
    )
    summary["parameter"] = summary["parameter_index"].str.extract(r"^(alpha|beta)")
    summary["competition"] = summary["parameter_index"].str.extract(r"\[(.*)\]")[0]
    summary["competition"] = summary["competition"].fillna("")
    summary = summary[
        [
            "competition",
            "parameter",
            "mean",
            "sd",
            "hdi_2_5",
            "hdi_97_5",
            "ess_bulk",
            "ess_tail",
            "r_hat",
        ]
    ]
    return summary


def summarise_competition_calibration(idata, comp_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = _competition_parameter_summary(idata)
    _write_csv(summary, "bayesian_competition_calibration_summary.csv")

    posterior = idata.posterior.stack(sample=("chain", "draw"))
    alpha = posterior["alpha"].values
    beta = posterior["beta"].values
    rows = []
    for j, comp in enumerate(comp_names):
        for sample_idx in range(alpha.shape[1]):
            rows.append(
                {
                    "sample": sample_idx,
                    "competition": comp,
                    "alpha": float(alpha[j, sample_idx]),
                    "beta": float(beta[j, sample_idx]),
                }
            )
    draws_df = pd.DataFrame(rows)
    _write_csv(draws_df, "bayesian_competition_calibration_draws.csv")

    global_summary = az.summary(idata, var_names=["mu_alpha", "sigma_alpha", "mu_beta", "sigma_beta"], hdi_prob=0.95)
    diag_rows = []
    for name, row in global_summary.reset_index().iterrows():
        diag_rows.append(
            {
                "model": "competition_calibration",
                "parameter": row["index"],
                "mean": row["mean"],
                "sd": row["sd"],
                "hdi_2_5": row["hdi_2.5%"],
                "hdi_97_5": row["hdi_97.5%"],
                "ess_bulk": row["ess_bulk"],
                "ess_tail": row["ess_tail"],
                "r_hat": row["r_hat"],
            }
        )
    comp_diag = summary.copy()
    for _, row in comp_diag.iterrows():
        diag_rows.append(
            {
                "model": "competition_calibration",
                "parameter": f"{row['parameter']}[{row['competition']}]",
                "mean": row["mean"],
                "sd": row["sd"],
                "hdi_2_5": row["hdi_2_5"],
                "hdi_97_5": row["hdi_97_5"],
                "ess_bulk": row["ess_bulk"],
                "ess_tail": row["ess_tail"],
                "r_hat": row["r_hat"],
            }
        )
    diagnostics = pd.DataFrame(diag_rows)
    if "diverging" in idata.sample_stats:
        diagnostics = pd.concat(
            [
                diagnostics,
                pd.DataFrame(
                    [
                        {
                            "model": "competition_calibration",
                            "parameter": "sampler_divergences",
                            "mean": int(idata.sample_stats["diverging"].sum().values),
                            "sd": np.nan,
                            "hdi_2_5": np.nan,
                            "hdi_97_5": np.nan,
                            "ess_bulk": np.nan,
                            "ess_tail": np.nan,
                            "r_hat": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    _write_csv(diagnostics, "bayesian_model_diagnostics.csv")
    return summary, draws_df, diagnostics


def prior_predictive_summary(pred: pd.DataFrame, prior: PriorSpec, draws: int = 1000) -> pd.DataFrame:
    d = pred.dropna(subset=["loco_xg", "loco_y", "held_out"]).copy()
    d["loco_xg"] = np.clip(d["loco_xg"].astype(float), 1e-6, 1 - 1e-6)
    d["z"] = logit(d["loco_xg"].values).astype("float64")
    rng = np.random.default_rng(SEED)
    comp_names = sorted(d["held_out"].unique())
    rows = []

    mu_alpha = rng.normal(0.0, prior.mu_alpha_sigma, size=draws)
    sigma_alpha = np.abs(rng.normal(0.0, prior.sigma_alpha_sigma, size=draws))
    mu_beta = rng.normal(1.0, prior.mu_beta_sigma, size=draws)
    sigma_beta = np.abs(rng.normal(0.0, prior.sigma_beta_sigma, size=draws))

    for comp in comp_names + ["All competitions"]:
        grp = d if comp == "All competitions" else d[d["held_out"] == comp]
        z = grp["z"].to_numpy()
        observed_rate = float(grp["loco_y"].mean())
        expected_rates = np.empty(draws)
        simulated_rates = np.empty(draws)
        for draw in range(draws):
            alpha = mu_alpha[draw] + rng.normal(0.0, 1.0) * sigma_alpha[draw]
            beta = mu_beta[draw] + rng.normal(0.0, 1.0) * sigma_beta[draw]
            q = 1.0 / (1.0 + np.exp(-(alpha + beta * z)))
            expected_rates[draw] = float(np.mean(q))
            simulated_rates[draw] = float(rng.binomial(1, q).mean())
        rows.append(
            {
                "prior": prior.name,
                "competition": comp,
                "n": int(len(grp)),
                "observed_goal_rate": observed_rate,
                "expected_goal_rate_mean": float(np.mean(expected_rates)),
                "expected_goal_rate_q2_5": float(np.quantile(expected_rates, 0.025)),
                "expected_goal_rate_q50": float(np.quantile(expected_rates, 0.5)),
                "expected_goal_rate_q97_5": float(np.quantile(expected_rates, 0.975)),
                "simulated_goal_rate_mean": float(np.mean(simulated_rates)),
                "simulated_goal_rate_q2_5": float(np.quantile(simulated_rates, 0.025)),
                "simulated_goal_rate_q50": float(np.quantile(simulated_rates, 0.5)),
                "simulated_goal_rate_q97_5": float(np.quantile(simulated_rates, 0.975)),
            }
        )
    out = pd.DataFrame(rows)
    _write_csv(out, "bayesian_prior_predictive_summary.csv")
    return out


def posterior_predictive_summary(pred: pd.DataFrame, draws_df: pd.DataFrame) -> pd.DataFrame:
    d = pred.dropna(subset=["loco_xg", "loco_y", "held_out"]).copy()
    d["loco_xg"] = np.clip(d["loco_xg"].astype(float), 1e-6, 1 - 1e-6)
    d["z"] = logit(d["loco_xg"].values).astype("float64")
    rng = np.random.default_rng(SEED)
    rows = []
    expected_by_comp = {}
    simulated_by_comp = {}

    for comp, grp in d.groupby("held_out", sort=True):
        comp_draws = draws_df[draws_df["competition"] == comp].sort_values("sample")
        z = grp["z"].to_numpy()
        y = grp["loco_y"].astype(int).to_numpy()
        expected_rates = np.empty(len(comp_draws))
        simulated_rates = np.empty(len(comp_draws))
        for j, draw in enumerate(comp_draws.itertuples(index=False)):
            q = 1.0 / (1.0 + np.exp(-(float(draw.alpha) + float(draw.beta) * z)))
            expected_rates[j] = float(np.mean(q))
            simulated_rates[j] = float(rng.binomial(1, q).mean())
        expected_by_comp[comp] = expected_rates
        simulated_by_comp[comp] = simulated_rates
        rows.append(
            {
                "competition": comp,
                "n": int(len(grp)),
                "observed_goal_rate": float(np.mean(y)),
                "expected_goal_rate_mean": float(np.mean(expected_rates)),
                "expected_goal_rate_q2_5": float(np.quantile(expected_rates, 0.025)),
                "expected_goal_rate_q50": float(np.quantile(expected_rates, 0.5)),
                "expected_goal_rate_q97_5": float(np.quantile(expected_rates, 0.975)),
                "simulated_goal_rate_mean": float(np.mean(simulated_rates)),
                "simulated_goal_rate_q2_5": float(np.quantile(simulated_rates, 0.025)),
                "simulated_goal_rate_q50": float(np.quantile(simulated_rates, 0.5)),
                "simulated_goal_rate_q97_5": float(np.quantile(simulated_rates, 0.975)),
            }
        )

    comp_sizes = d.groupby("held_out", sort=True).size()
    comp_order = list(comp_sizes.index)
    weights = comp_sizes.to_numpy(dtype=float) / float(len(d))
    expected_all = np.vstack([expected_by_comp[c] for c in comp_order])
    simulated_all = np.vstack([simulated_by_comp[c] for c in comp_order])
    expected_rates = np.average(expected_all, axis=0, weights=weights)
    simulated_rates = np.average(simulated_all, axis=0, weights=weights)
    rows.append(
        {
            "competition": "All competitions",
            "n": int(len(d)),
            "observed_goal_rate": float(d["loco_y"].mean()),
            "expected_goal_rate_mean": float(np.mean(expected_rates)),
            "expected_goal_rate_q2_5": float(np.quantile(expected_rates, 0.025)),
            "expected_goal_rate_q50": float(np.quantile(expected_rates, 0.5)),
            "expected_goal_rate_q97_5": float(np.quantile(expected_rates, 0.975)),
            "simulated_goal_rate_mean": float(np.mean(simulated_rates)),
            "simulated_goal_rate_q2_5": float(np.quantile(simulated_rates, 0.025)),
            "simulated_goal_rate_q50": float(np.quantile(simulated_rates, 0.5)),
            "simulated_goal_rate_q97_5": float(np.quantile(simulated_rates, 0.975)),
        }
    )
    out = pd.DataFrame(rows)
    _write_csv(out, "bayesian_posterior_predictive_summary.csv")
    return out


def prior_sensitivity_summary(
    pred: pd.DataFrame,
    baseline_idata,
    baseline_comp_names: list[str],
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
) -> pd.DataFrame:
    rows = []
    for prior_name, prior in PRIOR_SPECS.items():
        if prior_name == "baseline":
            idata, comp_names = baseline_idata, baseline_comp_names
        else:
            idata, comp_names = fit_competition_calibration(
                pred,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                prior=prior,
            )
        comp_summary = _competition_parameter_summary(idata)
        for _, row in comp_summary.iterrows():
            if row["competition"] in {"European Championship", "World Cup"}:
                rows.append(
                    {
                        "prior": prior_name,
                        "parameter": row["parameter"],
                        "competition": row["competition"],
                        "mean": row["mean"],
                        "sd": row["sd"],
                        "hdi_2_5": row["hdi_2_5"],
                        "hdi_97_5": row["hdi_97_5"],
                        "ess_bulk": row["ess_bulk"],
                        "r_hat": row["r_hat"],
                    }
                )
        global_summary = az.summary(
            idata, var_names=["mu_alpha", "sigma_alpha", "mu_beta", "sigma_beta"], hdi_prob=0.95
        ).reset_index()
        for _, row in global_summary.iterrows():
            rows.append(
                {
                    "prior": prior_name,
                    "parameter": row["index"],
                    "competition": "global",
                    "mean": row["mean"],
                    "sd": row["sd"],
                    "hdi_2_5": row["hdi_2.5%"],
                    "hdi_97_5": row["hdi_97.5%"],
                    "ess_bulk": row["ess_bulk"],
                    "r_hat": row["r_hat"],
                }
            )
        if "diverging" in idata.sample_stats:
            rows.append(
                {
                    "prior": prior_name,
                    "parameter": "sampler_divergences",
                    "competition": "diagnostic",
                    "mean": int(idata.sample_stats["diverging"].sum().values),
                    "sd": np.nan,
                    "hdi_2_5": np.nan,
                    "hdi_97_5": np.nan,
                    "ess_bulk": np.nan,
                    "r_hat": np.nan,
                }
            )
    out = pd.DataFrame(rows)
    _write_csv(out, "bayesian_prior_sensitivity_summary.csv")
    return out


def fit_player_residuals(all_shots: pd.DataFrame, draws: int, advi_steps: int, min_shots: int) -> pd.DataFrame:
    pm = _require_pymc()
    d = all_shots.dropna(subset=["xg_all_shot", "is_goal", "playerId"]).copy()
    d = d[(d["playerId"].astype(int) > 0)].copy()
    counts = d.groupby("playerId").size()
    keep_players = counts[counts >= min_shots].index
    d = d[d["playerId"].isin(keep_players)].copy()
    d["xg_all_shot"] = np.clip(d["xg_all_shot"].astype(float), 1e-6, 1 - 1e-6)
    y = d["is_goal"].astype("int8").values
    z = logit(d["xg_all_shot"].values).astype("float64")
    player_codes, player_ids = pd.factorize(d["playerId"].astype(int), sort=True)
    player_codes = player_codes.astype("int32")

    coords = {"player": [str(v) for v in player_ids], "shot": np.arange(len(d))}
    with pm.Model(coords=coords) as model:
        player_idx = pm.Data("player_idx", player_codes, dims="shot")
        z_data = pm.Data("logit_xg", z, dims="shot")
        sigma_player = pm.HalfNormal("sigma_player", sigma=0.75)
        theta_raw = pm.Normal("theta_raw", mu=0.0, sigma=1.0, dims="player")
        theta = pm.Deterministic("theta", theta_raw * sigma_player, dims="player")
        pm.Bernoulli("goal", logit_p=z_data + theta[player_idx], observed=y, dims="shot")
        approx = pm.fit(n=advi_steps, method="advi", random_seed=SEED, progressbar=True)
        idata = approx.sample(draws=draws, random_seed=SEED)

    summary = az.summary(idata, var_names=["theta"], hdi_prob=0.95).reset_index()
    summary["playerId"] = summary["index"].str.extract(r"\[(.*)\]")[0].astype(int)
    summary = summary.rename(columns={"hdi_2.5%": "theta_hdi_2_5", "hdi_97.5%": "theta_hdi_97_5"})
    agg = d.groupby("playerId").agg(shots=("is_goal", "size"), goals=("is_goal", "sum"), xg=("xg_all_shot", "sum")).reset_index()
    out = agg.merge(
        summary[["playerId", "mean", "sd", "theta_hdi_2_5", "theta_hdi_97_5"]],
        on="playerId",
        how="left",
    )
    out = out.rename(columns={"mean": "theta_mean", "sd": "theta_sd"})
    out["goals_minus_xg"] = out["goals"] - out["xg"]
    out["credible_interval_excludes_zero"] = (out["theta_hdi_2_5"] > 0) | (out["theta_hdi_97_5"] < 0)
    out = out.sort_values(["credible_interval_excludes_zero", "theta_mean"], ascending=[False, False])
    _write_csv(out, "bayesian_player_finishing_residuals.csv")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--player-draws", type=int, default=1000)
    parser.add_argument("--player-advi-steps", type=int, default=15000)
    parser.add_argument("--player-min-shots", type=int, default=3)
    parser.add_argument("--skip-player", action="store_true")
    parser.add_argument("--skip-prior-checks", action="store_true")
    parser.add_argument("--force-predictions", action="store_true")
    args = parser.parse_args()

    _require_pymc()
    all_path = OUT_DIR / "all_competition_shots_features.csv"
    if not all_path.exists():
        raise SystemExit(f"Missing {all_path}. Run Code/revision_multicomp_analysis.py first.")
    all_shots = pd.read_csv(all_path)
    cols = usable_features(all_shots)
    pred = make_loco_predictions(all_shots, cols, force=args.force_predictions)
    idata, comp_names = fit_competition_calibration(
        pred,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        prior=PRIOR_SPECS["baseline"],
    )
    _, draws_df, _ = summarise_competition_calibration(idata, comp_names)
    posterior_predictive_summary(pred, draws_df)
    if not args.skip_prior_checks:
        prior_predictive_summary(pred, PRIOR_SPECS["baseline"], draws=args.draws)
        prior_sensitivity_summary(
            pred,
            baseline_idata=idata,
            baseline_comp_names=comp_names,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
        )
    if not args.skip_player:
        fit_player_residuals(
            all_shots,
            draws=args.player_draws,
            advi_steps=args.player_advi_steps,
            min_shots=args.player_min_shots,
        )
    print(f"[done] Bayesian outputs written to {OUT_DIR} and mirrored to {FRONTIERS_OUT_DIR}")


if __name__ == "__main__":
    main()
