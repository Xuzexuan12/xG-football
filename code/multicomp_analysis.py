import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


ROOT = Path(__file__).resolve().parents[1]
EVENT_DIR = ROOT / "data" / "events"
MATCH_DIR = ROOT / "data" / "matches"
OUT_DIR = ROOT / "SR_Submission" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5
BOOTSTRAPS = 200
SAMPLE_SIZE_REPEATS = 20
RECALIBRATION_REPEATS = 20
PENALTY_XG = 0.79
SHOT_TYPE_REGULAR = "regular_shot"
SHOT_TYPE_DIRECT_FREE_KICK = "direct_free_kick"
SHOT_TYPE_PENALTY = "penalty"

PITCH_LEN_M = 105.0
PITCH_WID_M = 68.0
GOAL_X_NORM = 100.0
GOAL_Y_CENTER = 50.0
GOAL_HALF_WIDTH_M = 7.32 / 2.0
X_UNIT_TO_METERS = PITCH_LEN_M / 100.0
Y_UNIT_TO_METERS = PITCH_WID_M / 100.0
GOAL_HALF_WIDTH_YUNITS = GOAL_HALF_WIDTH_M / Y_UNIT_TO_METERS
PERIOD_OFFSETS = {"1H": 0, "2H": 45 * 60, "E1": 90 * 60, "E2": 105 * 60}

COMPETITIONS = [
    "England",
    "France",
    "Germany",
    "Italy",
    "Spain",
    "European_Championship",
    "World_Cup",
]


FEATURE_ORDER = [
    "feat_dist_m",
    "feat_angle_rad",
    "feat_inv_dist",
    "feat_cos_angle",
    "feat_angle_x_dist",
    "feat_x_m",
    "feat_y_m",
    "feat_lat_offset_m",
    "feat_minute",
    "feat_score_diff_before",
    "feat_prev_exists",
    "feat_prev_dt",
    "feat_prev_dist_m",
    "feat_prev_speed_mps",
    "feat_prev_lateral_m",
    "feat_prev_dx_m",
    "feat_prev_dy_m",
    "feat_tag_left_foot",
    "feat_tag_right_foot",
    "feat_tag_head",
    "feat_prev_through",
    "feat_prev_direct_fk",
    "feat_prev_indirect_fk",
    "feat_is_counter",
    "feat_is_cross",
]

FEATURE_META = [
    ("feat_dist_m", "Geometry", "Distance from shot location to goal centre", "metres"),
    ("feat_angle_rad", "Geometry", "Angle subtended by the two goalposts", "radians"),
    ("feat_inv_dist", "Geometry", "Inverse distance transform, 1/(1+distance)", "unitless"),
    ("feat_cos_angle", "Geometry", "Cosine transform of shooting angle", "unitless"),
    ("feat_angle_x_dist", "Geometry", "Angle-distance interaction", "metre-radians"),
    ("feat_x_m", "Geometry", "Longitudinal distance from goal after direction standardisation", "metres"),
    ("feat_y_m", "Geometry", "Signed lateral displacement from pitch centreline", "metres"),
    ("feat_lat_offset_m", "Geometry", "Absolute lateral offset from pitch centreline", "metres"),
    ("feat_minute", "Match context", "Match clock at the shot", "minutes"),
    ("feat_score_diff_before", "Match context", "Goal difference from shooting-team perspective before the shot", "goals"),
    ("feat_prev_exists", "Pre-shot context", "Indicator that a previous same-team event exists", "binary"),
    ("feat_prev_dt", "Pre-shot context", "Elapsed time since previous same-team event", "seconds"),
    ("feat_prev_dist_m", "Pre-shot context", "Distance from previous same-team event to shot", "metres"),
    ("feat_prev_speed_mps", "Pre-shot context", "Previous-event displacement divided by elapsed time", "m/s"),
    ("feat_prev_lateral_m", "Pre-shot context", "Absolute lateral displacement from previous event", "metres"),
    ("feat_prev_dx_m", "Pre-shot context", "Longitudinal displacement from previous event", "metres"),
    ("feat_prev_dy_m", "Pre-shot context", "Lateral displacement from previous event", "metres"),
    ("feat_tag_left_foot", "Execution", "Shot tagged as left-footed", "binary"),
    ("feat_tag_right_foot", "Execution", "Shot tagged as right-footed", "binary"),
    ("feat_tag_head", "Execution", "Shot tagged as head/body", "binary"),
    ("feat_prev_through", "Pre-shot context", "Previous same-team event tagged as a through pass", "binary"),
    ("feat_prev_direct_fk", "Set-piece context", "Previous same-team event was a direct free-kick action", "binary"),
    ("feat_prev_indirect_fk", "Set-piece context", "Previous same-team event was an indirect/set-piece action", "binary"),
    ("feat_is_counter", "Shot context", "Shot tagged as occurring in a counter-attacking phase", "binary"),
    ("feat_is_cross", "Shot context", "Shot event includes a cross-related tag", "binary"),
]

FEATURE_BLOCKS = {
    "Geometry": [
        "feat_dist_m",
        "feat_angle_rad",
        "feat_inv_dist",
        "feat_cos_angle",
        "feat_angle_x_dist",
        "feat_x_m",
        "feat_y_m",
        "feat_lat_offset_m",
    ],
    "Match context": ["feat_minute", "feat_score_diff_before"],
    "Previous-event context": [
        "feat_prev_exists",
        "feat_prev_dt",
        "feat_prev_dist_m",
        "feat_prev_speed_mps",
        "feat_prev_lateral_m",
        "feat_prev_dx_m",
        "feat_prev_dy_m",
        "feat_prev_through",
        "feat_prev_direct_fk",
        "feat_prev_indirect_fk",
    ],
    "Execution": ["feat_tag_left_foot", "feat_tag_right_foot", "feat_tag_head"],
    "Shot context": ["feat_is_counter", "feat_is_cross"],
}

PRIMARY_LOGISTIC_CALIBRATIONS = [
    ("none", "none"),
    ("sigmoid", "sigmoid_grouped"),
    ("isotonic", "isotonic_grouped"),
]
PRIMARY_HGB_CALIBRATION = "sigmoid_grouped"


def logistic_model_label(calibration_label: str) -> str:
    if calibration_label == "none":
        return "Full logistic uncalibrated"
    return f"Full logistic {calibration_label}"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tag_ids(event: dict) -> set:
    return {int(t.get("id")) for t in event.get("tags", []) if isinstance(t, dict) and t.get("id") is not None}


def event_xy(event: dict) -> Tuple[float, float]:
    pos = event.get("positions") or []
    if not pos:
        return np.nan, np.nan
    return float(pos[0].get("x", np.nan)), float(pos[0].get("y", np.nan))


def to_attacking_frame(x: float, y: float, is_attacking_right: bool) -> Tuple[float, float]:
    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan
    return (x, y) if is_attacking_right else (100.0 - x, y)


def shot_geometry_features(ax: float, ay: float) -> Dict[str, float]:
    if pd.isna(ax) or pd.isna(ay):
        return {
            "feat_dist_m": np.nan,
            "feat_angle_rad": np.nan,
            "feat_x_m": np.nan,
            "feat_y_m": np.nan,
            "feat_lat_offset_m": np.nan,
        }

    dx_units = max(0.0, GOAL_X_NORM - ax)
    dy_units = ay - GOAL_Y_CENTER
    feat_x_m = dx_units * X_UNIT_TO_METERS
    feat_y_m = dy_units * Y_UNIT_TO_METERS

    y_post_top = GOAL_Y_CENTER - GOAL_HALF_WIDTH_YUNITS
    y_post_bot = GOAL_Y_CENTER + GOAL_HALF_WIDTH_YUNITS
    angle = abs(math.atan2(ay - y_post_top, dx_units) - math.atan2(ay - y_post_bot, dx_units))

    return {
        "feat_dist_m": math.hypot(dx_units, dy_units) * X_UNIT_TO_METERS,
        "feat_angle_rad": angle,
        "feat_x_m": feat_x_m,
        "feat_y_m": feat_y_m,
        "feat_lat_offset_m": abs(feat_y_m),
    }


def records_to_frame(events: List[dict], competition: str) -> pd.DataFrame:
    rows = []
    for e in events:
        x, y = event_xy(e)
        rows.append(
            {
                "competition": competition,
                "id": int(e["id"]),
                "matchId": int(e["matchId"]),
                "teamId": int(e["teamId"]),
                "playerId": int(e.get("playerId", 0) or 0),
                "eventName": e.get("eventName"),
                "subEventName": e.get("subEventName"),
                "matchPeriod": e.get("matchPeriod"),
                "eventSec": float(e.get("eventSec", 0.0) or 0.0),
                "x_start": x,
                "y_start": y,
                "tag_ids": tag_ids(e),
            }
        )
    return pd.DataFrame(rows)


def match_date_frame(competitions: Iterable[str] = COMPETITIONS) -> pd.DataFrame:
    rows = []
    for comp in competitions:
        match_path = MATCH_DIR / f"matches_{comp}.json"
        for m in load_json(match_path):
            rows.append(
                {
                    "competition": comp,
                    "matchId": int(m["wyId"]),
                    "match_dateutc": pd.to_datetime(m.get("dateutc"), errors="coerce", utc=True),
                    "match_gameweek": int(m.get("gameweek", 0) or 0),
                    "match_roundId": int(m.get("roundId", 0) or 0),
                }
            )
    return pd.DataFrame(rows)


def infer_team_attacking_right_per_period(events: pd.DataFrame) -> Dict[Tuple[int, int, str], bool]:
    ev = events[events["matchPeriod"].isin(PERIOD_OFFSETS)].dropna(subset=["x_start"]).copy()
    shot_med = (
        ev[ev["eventName"].astype(str).str.lower() == "shot"]
        .groupby(["matchId", "teamId", "matchPeriod"])["x_start"]
        .median()
    )
    all_med = ev.groupby(["matchId", "teamId", "matchPeriod"])["x_start"].median()

    dir_map: Dict[Tuple[int, int, str], bool] = {}
    for key, xmed in shot_med.items():
        dir_map[(int(key[0]), int(key[1]), str(key[2]))] = bool(float(xmed) > 50.0)
    for key, xmed in all_med.items():
        k = (int(key[0]), int(key[1]), str(key[2]))
        dir_map.setdefault(k, bool(float(xmed) > 50.0))
    return dir_map


def add_context_features(events: pd.DataFrame) -> pd.DataFrame:
    ev = events[events["matchPeriod"].isin(PERIOD_OFFSETS)].copy()
    ev["period_offset_s"] = ev["matchPeriod"].map(PERIOD_OFFSETS).astype(float)
    ev["t_abs_s"] = ev["period_offset_s"] + pd.to_numeric(ev["eventSec"], errors="coerce").fillna(0.0)

    dir_map = infer_team_attacking_right_per_period(ev)
    ev["att_right"] = ev.apply(
        lambda r: dir_map.get((int(r["matchId"]), int(r["teamId"]), str(r["matchPeriod"])), True),
        axis=1,
    )
    norm = ev.apply(lambda r: to_attacking_frame(r["x_start"], r["y_start"], bool(r["att_right"])), axis=1)
    ev["ax"] = [v[0] for v in norm]
    ev["ay"] = [v[1] for v in norm]

    ev = ev.sort_values(["competition", "matchId", "teamId", "t_abs_s", "id"]).reset_index(drop=True)
    for c in ["t_abs_s", "ax", "ay", "tag_ids", "eventName", "subEventName"]:
        ev[f"prev_{c}"] = ev.groupby(["competition", "matchId", "teamId"])[c].shift(1)
    return ev


def add_score_diff(shots: pd.DataFrame) -> pd.DataFrame:
    s = shots.sort_values(["competition", "matchId", "t_abs_s", "id"]).copy()
    goal_times: Dict[Tuple[str, int, int], np.ndarray] = {}
    for key, grp in s[s["is_goal"] == 1].groupby(["competition", "matchId", "teamId"]):
        goal_times[(str(key[0]), int(key[1]), int(key[2]))] = np.sort(grp["t_abs_s"].astype(float).values)

    teams_by_match = s.groupby(["competition", "matchId"])["teamId"].unique().to_dict()

    def count_before(comp, mid, tid, t):
        arr = goal_times.get((str(comp), int(mid), int(tid)))
        if arr is None:
            return 0
        return int(np.searchsorted(arr, float(t), side="left"))

    def score_diff(row):
        teams = [int(t) for t in teams_by_match.get((row["competition"], row["matchId"]), [])]
        opps = [t for t in teams if t != int(row["teamId"])]
        opp = opps[0] if opps else -1
        gf = count_before(row["competition"], row["matchId"], row["teamId"], row["t_abs_s"])
        ga = count_before(row["competition"], row["matchId"], opp, row["t_abs_s"]) if opp != -1 else 0
        return gf - ga

    s["feat_score_diff_before"] = s.apply(score_diff, axis=1)
    return s


def build_shots_for_competition(competition: str) -> Tuple[pd.DataFrame, dict]:
    event_path = EVENT_DIR / f"events_{competition}.json"
    match_path = MATCH_DIR / f"matches_{competition}.json"
    events_raw = load_json(event_path)
    matches_raw = load_json(match_path)
    events = records_to_frame(events_raw, competition)
    ev = add_context_features(events)
    standard_events = events[events["matchPeriod"].isin(PERIOD_OFFSETS)].copy()
    raw_shot_mask = events["eventName"].astype(str).str.lower() == "shot"
    std_shot_mask = standard_events["eventName"].astype(str).str.lower() == "shot"
    std_free_kick_mask = standard_events["eventName"].astype(str).str.lower() == "free kick"
    standard_sub_event = standard_events["subEventName"].astype(str)
    direct_free_kick_mask = std_free_kick_mask & standard_sub_event.str.fullmatch(
        "Free kick shot", case=False
    )
    penalty_event_mask = std_free_kick_mask & standard_sub_event.str.fullmatch("Penalty", case=False)
    free_kick_shot_mask = direct_free_kick_mask | penalty_event_mask

    ev_event = ev["eventName"].astype(str).str.lower()
    ev_sub = ev["subEventName"].astype(str)
    shot_like_mask = (
        (ev_event == "shot")
        | ((ev_event == "free kick") & ev_sub.str.fullmatch("Free kick shot", case=False))
        | ((ev_event == "free kick") & ev_sub.str.fullmatch("Penalty", case=False))
    )
    shots = ev[shot_like_mask].copy()
    shots["shot_type"] = SHOT_TYPE_REGULAR
    shots.loc[
        (shots["eventName"].astype(str).str.lower() == "free kick")
        & shots["subEventName"].astype(str).str.fullmatch("Free kick shot", case=False),
        "shot_type",
    ] = SHOT_TYPE_DIRECT_FREE_KICK
    shots.loc[
        (shots["eventName"].astype(str).str.lower() == "free kick")
        & shots["subEventName"].astype(str).str.fullmatch("Penalty", case=False),
        "shot_type",
    ] = SHOT_TYPE_PENALTY
    shots["is_regular_shot"] = (shots["shot_type"] == SHOT_TYPE_REGULAR).astype(int)
    shots["is_direct_free_kick"] = (shots["shot_type"] == SHOT_TYPE_DIRECT_FREE_KICK).astype(int)
    shots["is_penalty"] = (shots["shot_type"] == SHOT_TYPE_PENALTY).astype(int)
    shots["is_goal"] = shots["tag_ids"].apply(lambda ids: int(101 in ids))
    shots["own_goal_tag"] = shots["tag_ids"].apply(lambda ids: int(102 in ids))
    shots["penalty_tag"] = shots["is_penalty"]

    geom = shots.apply(lambda r: shot_geometry_features(r["ax"], r["ay"]), axis=1)
    shots = pd.concat([shots, pd.DataFrame(list(geom), index=shots.index)], axis=1)
    shots["feat_inv_dist"] = 1.0 / (1.0 + shots["feat_dist_m"])
    shots["feat_cos_angle"] = np.cos(shots["feat_angle_rad"])
    shots["feat_angle_x_dist"] = shots["feat_angle_rad"] * shots["feat_dist_m"]
    shots["feat_minute"] = shots["t_abs_s"] / 60.0

    shots["feat_tag_left_foot"] = shots["tag_ids"].apply(lambda ids: int(401 in ids))
    shots["feat_tag_right_foot"] = shots["tag_ids"].apply(lambda ids: int(402 in ids))
    shots["feat_tag_head"] = shots["tag_ids"].apply(lambda ids: int(403 in ids))
    shots["feat_is_counter"] = shots["tag_ids"].apply(lambda ids: int(1901 in ids))
    shots["feat_is_cross"] = shots["tag_ids"].apply(lambda ids: int(801 in ids or 302 in ids))

    shots["feat_prev_exists"] = shots["prev_t_abs_s"].notna().astype(int)
    shots["feat_prev_dt"] = (shots["t_abs_s"] - shots["prev_t_abs_s"]).fillna(999.0)
    shots["feat_prev_dx_m"] = ((shots["ax"] - shots["prev_ax"]) * X_UNIT_TO_METERS).fillna(0.0)
    shots["feat_prev_dy_m"] = ((shots["ay"] - shots["prev_ay"]) * Y_UNIT_TO_METERS).fillna(0.0)
    shots["feat_prev_dist_m"] = np.hypot(shots["feat_prev_dx_m"], shots["feat_prev_dy_m"])
    shots["feat_prev_lateral_m"] = shots["feat_prev_dy_m"].abs()
    shots["feat_prev_speed_mps"] = (
        shots["feat_prev_dist_m"] / shots["feat_prev_dt"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def prev_has(ids, value):
        return int(isinstance(ids, set) and value in ids)

    shots["feat_prev_through"] = shots["prev_tag_ids"].apply(lambda ids: prev_has(ids, 901))
    shots["feat_prev_direct_fk"] = shots["prev_subEventName"].astype(str).str.contains(
        "Free kick shot|Free Kick", case=False, regex=True
    ).astype(int)
    shots["feat_prev_indirect_fk"] = shots["prev_subEventName"].astype(str).str.contains(
        "Corner|Free kick cross|Throw in", case=False, regex=True
    ).astype(int)

    shots = add_score_diff(shots)
    pre_drop_shots = len(shots)
    feature_required = shots["shot_type"] != SHOT_TYPE_PENALTY
    missing_geometry_mask = (
        feature_required
        & shots[["feat_dist_m", "feat_angle_rad", "matchId", "is_goal"]].isna().any(axis=1)
    )
    missing_geometry = int(missing_geometry_mask.sum())
    shots = shots.loc[~missing_geometry_mask].dropna(subset=["matchId", "is_goal"]).copy()

    summary = {
        "competition": competition.replace("_", " "),
        "matches": len(matches_raw),
        "events": len(events_raw),
        "standard_period_events": int(len(standard_events)),
        "free_kick_events": int(std_free_kick_mask.sum()),
        "free_kick_shot_events": int(free_kick_shot_mask.sum()),
        "direct_free_kick_shots": int(direct_free_kick_mask.sum()),
        "penalties": int(penalty_event_mask.sum()),
        "shots": int(raw_shot_mask.sum()),
        "standard_period_shot_events": int(std_shot_mask.sum()),
        "shot_events_before_geometry_drop": int(pre_drop_shots),
        "missing_position_or_geometry_drops": missing_geometry,
        "goal_tag_shots": int(shots["is_goal"].sum()),
        "own_goal_tag_shots": int(shots["own_goal_tag"].sum()),
        "penalty_tag_shots": int(shots["penalty_tag"].sum()),
        "regular_shots": int(shots["is_regular_shot"].sum()),
        "analytic_direct_free_kick_shots": int(shots["is_direct_free_kick"].sum()),
        "analytic_penalties": int(shots["is_penalty"].sum()),
        "analytic_shots": len(shots),
        "analytic_goals": int(shots["is_goal"].sum()),
        "goal_rate": float(shots["is_goal"].mean()),
    }
    return shots, summary


def clip01(p, eps=1e-15):
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def ece_quantile(y, p, n_bins=10):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    out = 0.0
    for i in range(n_bins):
        m = (p >= q[i]) & (p < q[i + 1])
        if m.any():
            out += abs(float(y[m].mean()) - float(p[m].mean())) * float(m.mean())
    return float(out)


def ece_uniform(y, p, n_bins=10):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            m = (p >= bins[i]) & (p <= bins[i + 1])
        else:
            m = (p >= bins[i]) & (p < bins[i + 1])
        if m.any():
            out += abs(float(y[m].mean()) - float(p[m].mean())) * float(m.mean())
    return float(out)


def safe_logloss_brier_ece(y, p, n_bins=10):
    y = np.asarray(y, int)
    p = clip01(p)
    out = {
        "LogLoss": np.nan,
        "Brier": np.nan,
        "ECE": np.nan,
    }
    if y.size == 0:
        return out
    out["LogLoss"] = float(log_loss(y, p, labels=[0, 1]))
    out["Brier"] = float(brier_score_loss(y, p))
    bins = max(2, min(int(n_bins), int(y.size)))
    out["ECE"] = float(ece_quantile(y, p, n_bins=bins))
    return out


def nagelkerke_r2(y, p):
    y = np.asarray(y, int)
    p = clip01(p)
    p0 = clip01([float(y.mean())])[0]
    llm = (y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    ll0 = (y * np.log(p0) + (1 - y) * np.log(1 - p0)).sum()
    cs = 1.0 - np.exp((2.0 / y.size) * (ll0 - llm))
    max_cs = 1.0 - np.exp((2.0 / y.size) * ll0)
    return float(cs / max(max_cs, 1e-15))


def calibration_intercept_slope(y, p) -> Tuple[float, float]:
    y = np.asarray(y, int)
    p = clip01(p, eps=1e-6)
    if len(np.unique(y)) < 2 or np.unique(p).size < 2:
        return np.nan, np.nan
    try:
        lr = LogisticRegression(C=1e6, max_iter=3000, solver="lbfgs")
        lr.fit(logit(p).reshape(-1, 1), y)
        return float(lr.intercept_[0]), float(lr.coef_[0][0])
    except Exception:
        return np.nan, np.nan


def metrics(y, p, include_calibration_params=False):
    y = np.asarray(y, int)
    p = clip01(p)
    base_rate = float(y.mean())
    brier = float(brier_score_loss(y, p))
    brier_null = base_rate * (1 - base_rate)
    if include_calibration_params:
        cal_intercept, cal_slope = calibration_intercept_slope(y, p)
    else:
        cal_intercept, cal_slope = np.nan, np.nan
    return {
        "n": int(y.size),
        "goals": int(y.sum()),
        "goal_rate": base_rate,
        "R2": nagelkerke_r2(y, p),
        "LogLoss": float(log_loss(y, p, labels=[0, 1])),
        "Brier": brier,
        "BSS": float(1.0 - brier / brier_null) if brier_null > 0 else np.nan,
        "ECE": ece_quantile(y, p),
        "ROC_AUC": float(roc_auc_score(y, p)),
        "PR_AUC": float(average_precision_score(y, p)),
        "MAE": float(mean_absolute_error(y, p)),
        "Calibration_Intercept": cal_intercept,
        "Calibration_Slope": cal_slope,
    }


def ece_bin_sensitivity(all_shots: pd.DataFrame, cols_full: List[str], bins=(5, 10, 15, 20)) -> pd.DataFrame:
    rows = []
    settings = [
        ("Pooled grouped CV", all_shots.copy()),
        ("World Cup grouped CV", all_shots[all_shots["competition"] == "World_Cup"].copy()),
    ]
    models = [
        ("Full logistic uncalibrated", "logistic", "none"),
        ("Full logistic sigmoid", "logistic", "sigmoid_grouped"),
        ("Full logistic isotonic", "logistic", "isotonic_grouped"),
        ("HGB sigmoid fixed comparator", "hgb", PRIMARY_HGB_CALIBRATION),
    ]
    for setting, df in settings:
        for model, model_kind, calibration in models:
            _, y, p = oof_predictions(df, cols_full, model_kind, calibration)
            base = {
                "setting": setting,
                "model": model,
                "n": int(len(y)),
                "goals": int(np.sum(y)),
                "goal_rate": float(np.mean(y)),
            }
            for n_bins in bins:
                row = dict(base)
                row.update({"n_bins": int(n_bins), "ECE": ece_quantile(y, p, n_bins=int(n_bins))})
                rows.append(row)
    return pd.DataFrame(rows)


def ece_binning_sensitivity(all_shots: pd.DataFrame, cols_full: List[str], bins=(5, 10, 15, 20)) -> pd.DataFrame:
    rows = []
    settings = [
        ("Pooled grouped CV", all_shots.copy()),
        ("World Cup grouped CV", all_shots[all_shots["competition"] == "World_Cup"].copy()),
    ]
    models = [
        ("Full logistic sigmoid", "logistic", "sigmoid_grouped"),
        ("HGB sigmoid fixed comparator", "hgb", PRIMARY_HGB_CALIBRATION),
    ]
    for setting, df in settings:
        for model, model_kind, calibration in models:
            _, y, p = oof_predictions(df, cols_full, model_kind, calibration)
            base = {
                "setting": setting,
                "model": model,
                "n": int(len(y)),
                "goals": int(np.sum(y)),
                "goal_rate": float(np.mean(y)),
            }
            for n_bins in bins:
                for binning in ["equal_frequency", "equal_width"]:
                    row = dict(base)
                    row.update(
                        {
                            "binning": binning,
                            "n_bins": int(n_bins),
                            "ECE": (
                                ece_quantile(y, p, n_bins=int(n_bins))
                                if binning == "equal_frequency"
                                else ece_uniform(y, p, n_bins=int(n_bins))
                            ),
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def _body_part_label(row: pd.Series) -> str:
    if int(row.get("feat_tag_left_foot", 0)) == 1:
        return "Left foot"
    if int(row.get("feat_tag_right_foot", 0)) == 1:
        return "Right foot"
    if int(row.get("feat_tag_head", 0)) == 1:
        return "Head/body"
    return "No body-part tag"


def calibration_strata_diagnostics(eval_df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    d = eval_df.reset_index(drop=True).copy()
    d["predicted_xg"] = np.asarray(pred, float)
    d["body_part"] = d.apply(_body_part_label, axis=1)
    d["competition_label"] = d["competition"].astype(str).str.replace("_", " ", regex=False)
    d["shot_type_label"] = d["shot_type"].astype(str).map(
        {
            SHOT_TYPE_REGULAR: "Regular shots",
            SHOT_TYPE_DIRECT_FREE_KICK: "Direct free kicks",
            SHOT_TYPE_PENALTY: "Penalties",
        }
    ).fillna(d["shot_type"].astype(str))
    d["xg_range"] = pd.cut(
        d["predicted_xg"],
        bins=[-np.inf, 0.05, 0.20, np.inf],
        labels=["Low xG (<0.05)", "Middle xG (0.05-0.20)", "High xG (>=0.20)"],
    )
    d["xg_bin"] = pd.qcut(
        d["predicted_xg"].rank(method="first"),
        q=10,
        labels=[f"Q{i}" for i in range(1, 11)],
    )

    rows = []

    xg_range_order = {
        "Low xG (<0.05)": 1,
        "Middle xG (0.05-0.20)": 2,
        "High xG (>=0.20)": 3,
    }
    shot_type_order = {
        "Regular shots": 1,
        "Direct free kicks": 2,
        "Penalties": 3,
    }

    def stratum_display_order(kind: str, group_name) -> int:
        label = str(group_name)
        if kind == "predicted_xg_decile" and label.startswith("Q"):
            return int(label[1:])
        if kind == "predicted_xg_range":
            return xg_range_order.get(label, 99)
        if kind == "shot_type":
            return shot_type_order.get(label, 99)
        return 0

    def add_rows(kind: str, col: str) -> None:
        for group_name, grp in d.groupby(col, observed=False):
            if len(grp) == 0:
                continue
            y = grp["is_goal"].astype(int).values
            p = grp["predicted_xg"].astype(float).values
            row = {
                "stratum_type": kind,
                "stratum": str(group_name),
                "n": int(len(grp)),
                "goals": int(y.sum()),
                "observed_goal_rate": float(np.mean(y)),
                "mean_predicted_xg": float(np.mean(p)),
                "calibration_gap_observed_minus_predicted": float(np.mean(y) - np.mean(p)),
                "display_order": stratum_display_order(kind, group_name),
            }
            row.update(safe_logloss_brier_ece(y, p))
            rows.append(row)

    add_rows("predicted_xg_decile", "xg_bin")
    add_rows("predicted_xg_range", "xg_range")
    add_rows("body_part", "body_part")
    add_rows("competition", "competition_label")
    add_rows("shot_type", "shot_type_label")

    out = pd.DataFrame(rows)
    order = {
        "predicted_xg_decile": 0,
        "predicted_xg_range": 1,
        "body_part": 2,
        "competition": 3,
        "shot_type": 4,
    }
    out["stratum_type_order"] = out["stratum_type"].map(order)
    out = out.sort_values(["stratum_type_order", "display_order", "stratum"]).drop(
        columns=["stratum_type_order", "display_order"]
    )
    return out


def logloss_value(y, p) -> float:
    return float(log_loss(np.asarray(y, int), clip01(p), labels=[0, 1]))


def logistic_estimator():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, solver="lbfgs")),
        ]
    )


def hgb_estimator():
    return HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=SEED,
    )


def calibrate_estimator(estimator, method: str, cv=3):
    if method == "none":
        return estimator
    return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)


def base_estimator(model_kind: str):
    return logistic_estimator() if model_kind == "logistic" else hgb_estimator()


class ConstantProbabilityModel:
    def __init__(self, p: float):
        self.p = float(np.clip(p, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def _clean_for_composite(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    d = df.dropna(subset=["is_goal", "matchId"]).copy()
    if "shot_type" not in d.columns:
        d["shot_type"] = SHOT_TYPE_REGULAR
    if "is_penalty" not in d.columns:
        d["is_penalty"] = (d["shot_type"] == SHOT_TYPE_PENALTY).astype(int)
    feature_ok = d[cols].notna().all(axis=1)
    return d.loc[(d["is_penalty"].astype(int) == 1) | feature_ok].copy()


def _fit_sklearn_model(train: pd.DataFrame, cols: List[str], model_kind="logistic", calibration="none"):
    train = train.dropna(subset=cols + ["is_goal"]).copy()
    if len(train) == 0:
        return ConstantProbabilityModel(0.05)
    ytr = train["is_goal"].astype(int).values
    if len(np.unique(ytr)) < 2:
        return ConstantProbabilityModel(float(np.mean(ytr)))
    effective_calibration = calibration
    if calibration != "none":
        counts = np.bincount(ytr, minlength=2)
        if counts.min() < 3:
            effective_calibration = "none"
    Xtr = train[cols].astype(float).values
    calibration_method = effective_calibration
    calibration_cv = 3
    if effective_calibration.endswith("_grouped"):
        calibration_method = effective_calibration.replace("_grouped", "")
        groups = train["matchId"].values
        if len(np.unique(groups)) < 3:
            calibration_method = "none"
        else:
            try:
                splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
                calibration_cv = list(splitter.split(Xtr, ytr, groups))
            except ValueError:
                calibration_method = "none"
    model = calibrate_estimator(base_estimator(model_kind), calibration_method, cv=calibration_cv)
    model.fit(Xtr, ytr)
    return model


def _predict_sklearn_model(model, test: pd.DataFrame, cols: List[str]) -> np.ndarray:
    if len(test) == 0:
        return np.array([], dtype=float)
    Xte = test[cols].astype(float).values
    return model.predict_proba(Xte)[:, 1]


def _fit_composite_model(train: pd.DataFrame, cols: List[str], model_kind="logistic", calibration="none"):
    train = _clean_for_composite(train, cols)
    regular = train[train["shot_type"] == SHOT_TYPE_REGULAR].copy()
    direct_fk = train[train["shot_type"] == SHOT_TYPE_DIRECT_FREE_KICK].copy()
    return {
        "regular": _fit_sklearn_model(regular, cols, model_kind, calibration),
        "direct_free_kick": _fit_sklearn_model(direct_fk, cols, model_kind, calibration),
        "cols": list(cols),
        "model_kind": model_kind,
        "calibration": calibration,
        "penalty_xg": PENALTY_XG,
    }


def _predict_composite_model(model, df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    d = _clean_for_composite(df, cols).copy().reset_index(drop=True)
    y = d["is_goal"].astype(int).values
    pred = np.full(len(d), np.nan, dtype=float)
    penalty_mask = d["shot_type"].values == SHOT_TYPE_PENALTY
    regular_mask = d["shot_type"].values == SHOT_TYPE_REGULAR
    direct_mask = d["shot_type"].values == SHOT_TYPE_DIRECT_FREE_KICK
    pred[penalty_mask] = float(model.get("penalty_xg", PENALTY_XG))
    pred[regular_mask] = _predict_sklearn_model(model["regular"], d.loc[regular_mask], cols)
    pred[direct_mask] = _predict_sklearn_model(model["direct_free_kick"], d.loc[direct_mask], cols)
    if np.isnan(pred).any():
        raise ValueError("Composite all-shot prediction left some rows without probabilities")
    return d, y, pred


def oof_predictions_with_details(
    df: pd.DataFrame,
    cols: List[str],
    model_kind="logistic",
    calibration="isotonic",
    cv_mode="match_grouped",
):
    d = _clean_for_composite(df, cols)
    y = d["is_goal"].astype(int).values
    if cv_mode == "random_shot":
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        split_iter = cv.split(np.zeros_like(y), y)
    elif cv_mode == "team_grouped":
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        split_iter = cv.split(np.zeros_like(y), y, d["teamId"].values)
    else:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        split_iter = cv.split(np.zeros_like(y), y, d["matchId"].values)
    pred = np.zeros_like(y, dtype=float)
    detail_rows = []
    for fold, (tr, va) in enumerate(split_iter, start=1):
        model = fit_model(d.iloc[tr], cols, model_kind, calibration)
        _, _, pv = predict_model(model, d.iloc[va], cols)
        pred[va] = pv
        row = metrics(y[va], pred[va])
        row.update({"fold": fold})
        detail_rows.append(row)
    return d, y, pred, pd.DataFrame(detail_rows)


def oof_predictions(
    df: pd.DataFrame,
    cols: List[str],
    model_kind="logistic",
    calibration="isotonic",
    cv_mode="match_grouped",
):
    d, y, pred, _ = oof_predictions_with_details(df, cols, model_kind, calibration, cv_mode)
    return d, y, pred


def fit_predict_train_test(train, test, cols, model_kind="logistic", calibration="isotonic"):
    model = fit_model(train, cols, model_kind, calibration)
    _, yte, pte = predict_model(model, test, cols)
    return yte, pte


def fit_model(train, cols, model_kind="logistic", calibration="none"):
    return _fit_composite_model(train, cols, model_kind, calibration)


def predict_model(model, df, cols):
    return _predict_composite_model(model, df, cols)


def logit(p):
    p = clip01(p, eps=1e-6)
    return np.log(p / (1.0 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def intercept_recalibrate(p_cal, y_cal, p_test):
    ybar = float(np.mean(y_cal))
    if ybar <= 0.0 or ybar >= 1.0:
        return p_test
    z_cal = logit(p_cal)

    def objective(delta):
        return float(np.mean(expit(z_cal + delta)) - ybar)

    try:
        delta = brentq(objective, -10.0, 10.0)
    except ValueError:
        delta = logit([ybar])[0] - logit([np.mean(p_cal)])[0]
    return expit(logit(p_test) + delta)


def sigmoid_recalibrate(p_cal, y_cal, p_test):
    if len(np.unique(y_cal)) < 2:
        return p_test
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(logit(p_cal).reshape(-1, 1), y_cal)
    return lr.predict_proba(logit(p_test).reshape(-1, 1))[:, 1]


def isotonic_recalibrate(p_cal, y_cal, p_test):
    if len(np.unique(y_cal)) < 2:
        return p_test
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    return iso.predict(p_test)


def recalibrate_preserve_penalties(p_cal, y_cal, type_cal, p_test, type_test, strategy: str):
    out = np.asarray(p_test, float).copy()
    cal_non_penalty = np.asarray(type_cal) != SHOT_TYPE_PENALTY
    test_non_penalty = np.asarray(type_test) != SHOT_TYPE_PENALTY
    if strategy == "direct transfer":
        return out
    if not cal_non_penalty.any() or not test_non_penalty.any():
        return out
    p_cal_np = np.asarray(p_cal, float)[cal_non_penalty]
    y_cal_np = np.asarray(y_cal, int)[cal_non_penalty]
    p_test_np = np.asarray(p_test, float)[test_non_penalty]
    if strategy == "intercept recalibration":
        out[test_non_penalty] = intercept_recalibrate(p_cal_np, y_cal_np, p_test_np)
    elif strategy == "sigmoid recalibration":
        out[test_non_penalty] = sigmoid_recalibrate(p_cal_np, y_cal_np, p_test_np)
    elif strategy == "isotonic recalibration":
        out[test_non_penalty] = isotonic_recalibrate(p_cal_np, y_cal_np, p_test_np)
    else:
        raise ValueError(f"Unknown recalibration strategy: {strategy}")
    out[np.asarray(type_test) == SHOT_TYPE_PENALTY] = PENALTY_XG
    return out


def usable_features(df: pd.DataFrame) -> List[str]:
    cols = []
    feature_df = df[df["shot_type"] != SHOT_TYPE_PENALTY].copy() if "shot_type" in df.columns else df.copy()
    for c in FEATURE_ORDER:
        if c in feature_df.columns and feature_df[c].notna().all() and feature_df[c].nunique(dropna=True) > 1:
            cols.append(c)
    return cols


def one_se_rule(perf_df: pd.DataFrame) -> Tuple[int, int]:
    idx = perf_df["LogLoss"].idxmin()
    threshold = float(perf_df.loc[idx, "LogLoss"] + perf_df.loc[idx, "LogLoss_SE"])
    eligible = perf_df[perf_df["LogLoss"] <= threshold].sort_values("k")
    return int(eligible.iloc[0]["k"]), int(perf_df.loc[idx, "k"])


def performance_curve(world: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, int, int]:
    rows = []
    d = _clean_for_composite(world, cols)
    y = d["is_goal"].astype(int).values
    groups = d["matchId"].values
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(cv.split(np.zeros_like(y), y, groups))
    for k in tqdm(range(1, len(cols) + 1), desc="World Cup feature path", leave=False):
        fold_rows = []
        pred = np.zeros_like(y, dtype=float)
        for tr, va in folds:
            model = fit_model(d.iloc[tr], cols[:k], "logistic", "isotonic_grouped")
            _, _, pv = predict_model(model, d.iloc[va], cols[:k])
            pred[va] = pv
            fold_rows.append(metrics(y[va], pv))
        fm = pd.DataFrame(fold_rows)
        row = metrics(y, pred)
        row.update(
            {
                "k": k,
                "added": cols[k - 1],
                "LogLoss_SE": float(fm["LogLoss"].std(ddof=1) / math.sqrt(len(fm))),
            }
        )
        rows.append(row)
    perf = pd.DataFrame(rows)
    k_star, k_best = one_se_rule(perf)
    return perf, k_star, k_best


def bootstrap_diff(
    df: pd.DataFrame,
    pred_cols: Dict[str, np.ndarray],
    comparisons: Iterable[Tuple[str, str]],
    n_boot=BOOTSTRAPS,
    desc="bootstrap",
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    d = df.reset_index(drop=True).copy()
    group_key = d["competition"].astype(str).values + "_" + d["matchId"].astype(str).values
    group_codes, unique_groups = pd.factorize(group_key, sort=True)
    y = d["is_goal"].astype(int).values
    rows = []
    comparisons = list(comparisons)
    bootstrap_weights = []
    for _ in range(n_boot):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        bootstrap_weights.append(np.bincount(sampled, minlength=len(unique_groups)))
    for a, b in tqdm(comparisons, desc=desc, leave=False):
        vals = []
        pa = np.asarray(pred_cols[a], float)
        pb = np.asarray(pred_cols[b], float)
        loss_a = -(y * np.log(clip01(pa)) + (1 - y) * np.log(1 - clip01(pa)))
        loss_b = -(y * np.log(clip01(pb)) + (1 - y) * np.log(1 - clip01(pb)))
        loss_diff = loss_a - loss_b
        for group_weights in bootstrap_weights:
            row_weights = group_weights[group_codes]
            vals.append(float(np.average(loss_diff, weights=row_weights)))
        vals = np.asarray(vals)
        rows.append(
            {
                "comparison": f"{a} minus {b}",
                "metric": "LogLoss",
                "mean_diff": float(np.mean(loss_diff)),
                "ci_low": float(np.quantile(vals, 0.025)),
                "ci_high": float(np.quantile(vals, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def validation_design_sensitivity_detailed(all_shots: pd.DataFrame, cols_full: List[str]) -> pd.DataFrame:
    rows = []
    for label, cv_mode in tqdm(
        [
            ("Random shot-level CV", "random_shot"),
            ("Match-grouped CV", "match_grouped"),
            ("Shooting-team blocked CV", "team_grouped"),
        ],
        desc="validation-design folds",
        leave=False,
    ):
        _, _, _, fold_metrics = oof_predictions_with_details(
            all_shots, cols_full, "logistic", "sigmoid_grouped", cv_mode=cv_mode
        )
        fold_metrics["validation_design"] = label
        fold_metrics["model"] = "Full logistic sigmoid"
        fold_metrics["k"] = len(cols_full)
        rows.append(fold_metrics)

    logo = pd.read_csv(OUT_DIR / "leave_one_competition_out.csv")
    logo = logo[logo["model"] == "logistic sigmoid"].copy()
    logo["fold"] = logo["held_out"]
    logo["validation_design"] = "Leave-one-competition-out"
    logo["model"] = "Full logistic sigmoid"
    logo["k"] = len(cols_full)
    rows.append(logo)
    return pd.concat(rows, ignore_index=True)


def validation_design_sensitivity(all_shots: pd.DataFrame, cols_full: List[str]) -> pd.DataFrame:
    rows = []
    for label, cv_mode in tqdm(
        [
            ("Random shot-level CV", "random_shot"),
            ("Match-grouped CV", "match_grouped"),
            ("Shooting-team blocked CV", "team_grouped"),
        ],
        desc="validation-design OOF",
        leave=False,
    ):
        _, y, p = oof_predictions(all_shots, cols_full, "logistic", "sigmoid_grouped", cv_mode=cv_mode)
        row = metrics(y, p, include_calibration_params=True)
        row.update({"validation_design": label, "model": "Full logistic sigmoid", "k": len(cols_full)})
        rows.append(row)

    logo = pd.read_csv(OUT_DIR / "leave_one_competition_out.csv")
    logo = logo[logo["model"] == "logistic sigmoid"].copy()
    # Weighted average by held-out shots for scalar metrics.
    row = {"validation_design": "Leave-one-competition-out", "model": "Full logistic sigmoid", "k": len(cols_full)}
    for c in ["n", "goals"]:
        row[c] = int(logo[c].sum())
    row["goal_rate"] = float(row["goals"] / row["n"])
    for c in [
        "R2",
        "LogLoss",
        "Brier",
        "BSS",
        "ECE",
        "ROC_AUC",
        "PR_AUC",
        "MAE",
        "Calibration_Intercept",
        "Calibration_Slope",
    ]:
        row[c] = float(np.average(logo[c], weights=logo["n"]))
    rows.append(row)
    out = pd.DataFrame(rows)
    detailed = validation_design_sensitivity_detailed(all_shots, cols_full)
    for design, grp in detailed.groupby("validation_design"):
        idx = out["validation_design"] == design
        for metric in ["LogLoss", "Brier", "ECE", "ROC_AUC", "PR_AUC"]:
            out.loc[idx, f"{metric}_ci_low"] = float(np.quantile(grp[metric], 0.025))
            out.loc[idx, f"{metric}_ci_high"] = float(np.quantile(grp[metric], 0.975))
    return out


def calibration_sample_size_stability(all_shots: pd.DataFrame, cols_best: List[str]) -> pd.DataFrame:
    d = all_shots.dropna(subset=cols_best + ["is_goal", "matchId"]).copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(d, d["is_goal"], d["matchId"]))
    pool = d.iloc[train_idx].copy()
    test = d.iloc[test_idx].copy()
    train_matches = np.array(sorted(pool["matchId"].unique()))
    rng = np.random.default_rng(SEED)
    rows = []
    for frac in tqdm([0.10, 0.25, 0.50, 0.75, 1.00], desc="calibration sample-size", leave=False):
        reps = 1 if frac == 1.0 else SAMPLE_SIZE_REPEATS
        for rep in range(reps):
            if frac == 1.0:
                chosen = train_matches
            else:
                n_matches = max(5, int(round(len(train_matches) * frac)))
                chosen = rng.choice(train_matches, size=n_matches, replace=False)
            train = pool[pool["matchId"].isin(chosen)].copy()
            for calibration_label, calibration in PRIMARY_LOGISTIC_CALIBRATIONS:
                model = fit_model(train, cols_best, "logistic", calibration)
                _, yte, pte = predict_model(model, test, cols_best)
                row = metrics(yte, pte)
                row.update(
                    {
                        "train_fraction": frac,
                        "repeat": rep,
                        "calibration": calibration_label,
                        "train_matches": int(train["matchId"].nunique()),
                        "train_shots": int(len(train)),
                        "train_goals": int(train["is_goal"].sum()),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def feature_family_sets(cols_best: List[str]) -> Dict[str, List[str]]:
    blocks = {k: [c for c in v if c in cols_best] for k, v in FEATURE_BLOCKS.items()}
    full = cols_best
    sets = {
        "Geometry only": blocks["Geometry"],
        "Geometry + Match": blocks["Geometry"] + blocks["Match context"],
        "Geometry + Previous": blocks["Geometry"] + blocks["Previous-event context"],
        "Geometry + Execution": blocks["Geometry"] + blocks["Execution"],
        "Geometry + Shot context": blocks["Geometry"] + blocks["Shot context"],
        "Full": full,
        "Full minus Previous": [c for c in full if c not in blocks["Previous-event context"]],
        "Full minus Execution": [c for c in full if c not in blocks["Execution"]],
        "Full minus Match": [c for c in full if c not in blocks["Match context"]],
        "Full minus Shot context": [c for c in full if c not in blocks["Shot context"]],
    }
    return {k: list(dict.fromkeys(v)) for k, v in sets.items() if len(v) > 0}


def feature_family_ablation(
    all_shots: pd.DataFrame,
    world: pd.DataFrame,
    cols_full: List[str],
    return_predictions=False,
):
    rows = []
    pred_store: Dict[Tuple[str, str], Tuple[pd.DataFrame, np.ndarray]] = {}
    for setting, df in tqdm(
        [("Pooled match-grouped CV", all_shots), ("World Cup match-grouped CV", world)],
        desc="feature-family settings",
        leave=False,
    ):
        for model_name, cols in tqdm(
            feature_family_sets(cols_full).items(), desc=f"{setting} blocks", leave=False
        ):
            d, y, p = oof_predictions(df, cols, "logistic", "sigmoid_grouped", cv_mode="match_grouped")
            row = metrics(y, p)
            row.update({"setting": setting, "feature_family_model": model_name, "k": len(cols)})
            rows.append(row)
            pred_store[(setting, model_name)] = (d, p)
    out = pd.DataFrame(rows)
    out["delta_logloss_vs_geometry"] = np.nan
    out["delta_pr_auc_vs_geometry"] = np.nan
    for setting, grp in out.groupby("setting"):
        base = grp[grp["feature_family_model"] == "Geometry only"].iloc[0]
        idx = out["setting"] == setting
        out.loc[idx, "delta_logloss_vs_geometry"] = out.loc[idx, "LogLoss"] - float(base["LogLoss"])
        out.loc[idx, "delta_pr_auc_vs_geometry"] = out.loc[idx, "PR_AUC"] - float(base["PR_AUC"])
    if return_predictions:
        return out, pred_store
    return out


def target_domain_recalibration_by_competition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (held_out, strategy), grp in df.groupby(["held_out", "strategy"]):
        row = {
            "held_out": held_out,
            "strategy": strategy,
            "repeats": int(len(grp)),
            "mean_calibration_goals": float(grp["calibration_goals"].mean()),
        }
        for metric in ["LogLoss", "Brier", "ECE", "ROC_AUC", "PR_AUC"]:
            row[f"{metric}_mean"] = float(grp[metric].mean())
            row[f"{metric}_ci_low"] = float(np.quantile(grp[metric], 0.025))
            row[f"{metric}_ci_high"] = float(np.quantile(grp[metric], 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def target_domain_recalibration(all_shots: pd.DataFrame, cols_best: List[str]) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for comp in tqdm(COMPETITIONS, desc="target-domain competitions", leave=False):
        train = all_shots[all_shots["competition"] != comp].copy()
        target = all_shots[all_shots["competition"] == comp].copy()
        target_matches = np.array(sorted(target["matchId"].unique()))
        base = fit_model(train, cols_best, "logistic", "none")
        target_pred_df, target_y, target_p = predict_model(base, target, cols_best)
        target_pred_df = target_pred_df.reset_index(drop=True)
        target_y = np.asarray(target_y, int)
        target_p = np.asarray(target_p, float)
        match_ids = target_pred_df["matchId"].values

        for frac in [0.05, 0.10, 0.20, 0.40]:
            for rep in range(RECALIBRATION_REPEATS):
                n_cal = max(2, int(round(len(target_matches) * frac)))
                n_cal = min(n_cal, len(target_matches) - 1)
                cal_matches = rng.choice(target_matches, size=n_cal, replace=False)
                cal_mask = np.isin(match_ids, cal_matches)
                test_mask = ~cal_mask
                if target_y[cal_mask].sum() == 0 or target_y[test_mask].sum() == 0:
                    continue
                p_cal = target_p[cal_mask]
                y_cal = target_y[cal_mask]
                p_test = target_p[test_mask]
                y_test = target_y[test_mask]
                type_cal = target_pred_df.loc[cal_mask, "shot_type"].values
                type_test = target_pred_df.loc[test_mask, "shot_type"].values
                for strategy in [
                    "direct transfer",
                    "intercept recalibration",
                    "sigmoid recalibration",
                    "isotonic recalibration",
                ]:
                    pred = recalibrate_preserve_penalties(p_cal, y_cal, type_cal, p_test, type_test, strategy)
                    row = metrics(y_test, pred)
                    row.update(
                        {
                            "held_out": comp.replace("_", " "),
                            "calibration_fraction": frac,
                            "repeat": rep,
                            "strategy": strategy,
                            "calibration_matches": int(n_cal),
                            "calibration_shots": int(cal_mask.sum()),
                            "calibration_goals": int(y_cal.sum()),
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def target_domain_recalibration_time_ordered(all_shots: pd.DataFrame, cols_best: List[str]) -> pd.DataFrame:
    rows = []
    match_dates = match_date_frame()
    dated = all_shots.merge(match_dates, on=["competition", "matchId"], how="left")
    if dated["match_dateutc"].isna().any():
        missing = dated.loc[dated["match_dateutc"].isna(), ["competition", "matchId"]].drop_duplicates()
        raise ValueError(f"Missing match_dateutc for {len(missing)} competition-match pairs")

    for comp in tqdm(COMPETITIONS, desc="time-ordered target-domain competitions", leave=False):
        train = dated[dated["competition"] != comp].copy()
        target = dated[dated["competition"] == comp].copy()
        target_matches = (
            target[["matchId", "match_dateutc"]]
            .drop_duplicates()
            .sort_values(["match_dateutc", "matchId"])
            ["matchId"]
            .to_numpy()
        )
        base = fit_model(train, cols_best, "logistic", "none")
        target_pred_df, target_y, target_p = predict_model(base, target, cols_best)
        target_pred_df = target_pred_df.reset_index(drop=True)
        target_y = np.asarray(target_y, int)
        target_p = np.asarray(target_p, float)
        match_ids = target_pred_df["matchId"].values

        for frac in [0.20, 0.40]:
            n_cal = max(2, int(round(len(target_matches) * frac)))
            n_cal = min(n_cal, len(target_matches) - 1)
            cal_matches = target_matches[:n_cal]
            cal_mask = np.isin(match_ids, cal_matches)
            test_mask = ~cal_mask
            if target_y[cal_mask].sum() == 0 or target_y[test_mask].sum() == 0:
                continue
            p_cal = target_p[cal_mask]
            y_cal = target_y[cal_mask]
            p_test = target_p[test_mask]
            y_test = target_y[test_mask]
            type_cal = target_pred_df.loc[cal_mask, "shot_type"].values
            type_test = target_pred_df.loc[test_mask, "shot_type"].values
            for strategy in [
                "direct transfer",
                "intercept recalibration",
                "sigmoid recalibration",
                "isotonic recalibration",
            ]:
                pred = recalibrate_preserve_penalties(p_cal, y_cal, type_cal, p_test, type_test, strategy)
                row = metrics(y_test, pred)
                row.update(
                    {
                        "held_out": comp.replace("_", " "),
                        "calibration_fraction": frac,
                        "strategy": strategy,
                        "calibration_matches": int(n_cal),
                        "test_matches": int(len(target_matches) - n_cal),
                        "calibration_shots": int(cal_mask.sum()),
                        "calibration_goals": int(y_cal.sum()),
                        "test_shots": int(test_mask.sum()),
                        "test_goals": int(y_test.sum()),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def target_domain_recalibration_time_ordered_by_competition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (held_out, frac, strategy), grp in df.groupby(["held_out", "calibration_fraction", "strategy"]):
        row = {
            "held_out": held_out,
            "calibration_fraction": float(frac),
            "strategy": strategy,
            "calibration_matches": int(grp["calibration_matches"].iloc[0]),
            "test_matches": int(grp["test_matches"].iloc[0]),
            "calibration_goals": int(grp["calibration_goals"].iloc[0]),
            "test_goals": int(grp["test_goals"].iloc[0]),
        }
        for metric in ["LogLoss", "Brier", "ECE", "ROC_AUC", "PR_AUC"]:
            row[metric] = float(grp[metric].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_inner_calibration_sensitivity(all_shots: pd.DataFrame, cols_full: List[str]) -> pd.DataFrame:
    rows = []
    settings = [
        ("Pooled grouped CV", all_shots.copy()),
        ("World Cup grouped CV", all_shots[all_shots["competition"] == "World_Cup"].copy()),
    ]
    calibrations = [
        ("default stratified inner cv=3", "sigmoid"),
        ("match-grouped inner cv=3", "sigmoid_grouped"),
    ]
    for setting, df in tqdm(settings, desc="grouped inner calibration sensitivity", leave=False):
        for label, calibration in calibrations:
            _, y, p = oof_predictions(df, cols_full, "logistic", calibration, cv_mode="match_grouped")
            row = metrics(y, p, include_calibration_params=True)
            row.update(
                {
                    "setting": setting,
                    "model": "Full logistic sigmoid",
                    "inner_calibration": label,
                    "k": len(cols_full),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _team_competition_xg_totals(df: pd.DataFrame, pred: np.ndarray) -> pd.Series:
    tmp = df[["competition", "teamId"]].copy()
    tmp["xg"] = np.asarray(pred, float)
    return tmp.groupby(["competition", "teamId"])["xg"].sum().sort_index()


def penalty_xg_sensitivity(all_shots_export: pd.DataFrame) -> pd.DataFrame:
    d = all_shots_export.reset_index(drop=True).copy()
    y = d["is_goal"].astype(int).values
    base_pred = d["xg_all_shot"].astype(float).values
    penalty_mask = d["shot_type"].values == SHOT_TYPE_PENALTY
    penalty_rate = float(d.loc[penalty_mask, "is_goal"].mean())
    variants = [
        ("fixed_0.76", 0.76),
        ("fixed_0.79", 0.79),
        ("fixed_0.80", 0.80),
        ("sample_penalty_goal_rate", penalty_rate),
    ]
    base = base_pred.copy()
    base[penalty_mask] = PENALTY_XG
    base_totals = _team_competition_xg_totals(d, base)
    base_top = set(base_totals.sort_values(ascending=False).head(20).index)
    rows = []
    for label, value in variants:
        pred = base_pred.copy()
        pred[penalty_mask] = float(value)
        totals = _team_competition_xg_totals(d, pred)
        top = set(totals.sort_values(ascending=False).head(20).index)
        row = metrics(y, pred)
        row.update(
            {
                "penalty_variant": label,
                "penalty_xg": float(value),
                "penalty_attempts": int(penalty_mask.sum()),
                "penalty_goals": int(d.loc[penalty_mask, "is_goal"].sum()),
                "total_xg": float(np.sum(pred)),
                "penalty_xg_total": float(np.sum(pred[penalty_mask])),
                "team_competition_total_xg_spearman_vs_0_79": float(totals.corr(base_totals, method="spearman")),
                "team_competition_top20_overlap_vs_0_79": float(len(top & base_top) / max(len(base_top), 1)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def direct_free_kick_submodel_metrics(all_shots_export: pd.DataFrame) -> pd.DataFrame:
    d = all_shots_export[all_shots_export["shot_type"] == SHOT_TYPE_DIRECT_FREE_KICK].copy()
    y = d["is_goal"].astype(int).values
    p = d["xg_all_shot"].astype(float).values
    row = metrics(y, p, include_calibration_params=True)
    row.update(
        {
            "shot_type": SHOT_TYPE_DIRECT_FREE_KICK,
            "mean_predicted_xg": float(np.mean(p)),
            "min_predicted_xg": float(np.min(p)),
            "max_predicted_xg": float(np.max(p)),
            "unique_predicted_xg": int(pd.Series(p).nunique()),
        }
    )
    return pd.DataFrame([row])


def regular_shot_model_metrics(eval_df: pd.DataFrame, pred_store: Dict[str, np.ndarray]) -> pd.DataFrame:
    d = eval_df.reset_index(drop=True).copy()
    regular_mask = d["shot_type"].values == SHOT_TYPE_REGULAR
    y = d.loc[regular_mask, "is_goal"].astype(int).values
    rows = []
    for model_name in [
        "Full logistic uncalibrated",
        "Full logistic sigmoid",
        "Full logistic isotonic",
        "HGB sigmoid fixed comparator",
    ]:
        p = np.asarray(pred_store[model_name], float)[regular_mask]
        row = metrics(y, p, include_calibration_params=True)
        row.update(
            {
                "setting": "Pooled regular-shot grouped CV",
                "model": model_name,
                "shot_type": SHOT_TYPE_REGULAR,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_feature_inventory(k_star: int, k_best: int):
    rows = []
    for i, (name, family, definition, units) in enumerate(FEATURE_META, start=1):
        rows.append(
            {
                "order": i,
                "feature": name,
                "family": family,
                "definition": definition,
                "units": units,
                "Baseline": "yes" if i <= 1 else "no",
                f"WorldCup_Chosen_k{k_star}": "yes" if i <= k_star else "no",
                f"WorldCup_Selected_Best_k{k_best}": "yes" if i <= k_best else "no",
                "Full_interpretable_model": "yes",
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "feature_inventory.csv", index=False)


def body_part_tag_audit(all_shots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp, df in list(all_shots.groupby("competition")) + [("Total", all_shots)]:
        body = df[["feat_tag_left_foot", "feat_tag_right_foot", "feat_tag_head"]].astype(int)
        body_sum = body.sum(axis=1)
        rows.append(
            {
                "competition": str(comp).replace("_", " "),
                "analytic_shots": int(len(df)),
                "left_foot_tags": int(body["feat_tag_left_foot"].sum()),
                "right_foot_tags": int(body["feat_tag_right_foot"].sum()),
                "head_body_tags": int(body["feat_tag_head"].sum()),
                "any_body_part_tag": int((body_sum > 0).sum()),
                "no_body_part_tag": int((body_sum == 0).sum()),
                "multiple_body_part_tags": int((body_sum > 1).sum()),
                "mutually_exclusive_rate": float((body_sum <= 1).mean()),
            }
        )
    return pd.DataFrame(rows)


def sample_construction_audit(desc: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "competition",
        "matches",
        "events",
        "standard_period_events",
        "free_kick_events",
        "free_kick_shot_events",
        "direct_free_kick_shots",
        "penalties",
        "shots",
        "standard_period_shot_events",
        "shot_events_before_geometry_drop",
        "missing_position_or_geometry_drops",
        "regular_shots",
        "analytic_direct_free_kick_shots",
        "analytic_penalties",
        "analytic_shots",
        "analytic_goals",
        "own_goal_tag_shots",
        "penalty_tag_shots",
        "goal_rate",
    ]
    return desc[cols].copy()


def main():
    shots_by_comp = []
    summaries = []
    for comp in tqdm(COMPETITIONS, desc="load competitions"):
        print(f"[load] {comp}")
        shots, summary = build_shots_for_competition(comp)
        shots_by_comp.append(shots)
        summaries.append(summary)

    all_shots = pd.concat(shots_by_comp, ignore_index=True)
    all_shots.to_csv(OUT_DIR / "all_competition_shots_features.csv", index=False)

    desc = pd.DataFrame(summaries)
    total = {
        "competition": "Total",
        "matches": int(desc["matches"].sum()),
        "events": int(desc["events"].sum()),
        "standard_period_events": int(desc["standard_period_events"].sum()),
        "free_kick_events": int(desc["free_kick_events"].sum()),
        "free_kick_shot_events": int(desc["free_kick_shot_events"].sum()),
        "direct_free_kick_shots": int(desc["direct_free_kick_shots"].sum()),
        "penalties": int(desc["penalties"].sum()),
        "shots": int(desc["shots"].sum()),
        "standard_period_shot_events": int(desc["standard_period_shot_events"].sum()),
        "shot_events_before_geometry_drop": int(desc["shot_events_before_geometry_drop"].sum()),
        "missing_position_or_geometry_drops": int(desc["missing_position_or_geometry_drops"].sum()),
        "goal_tag_shots": int(desc["goal_tag_shots"].sum()),
        "own_goal_tag_shots": int(desc["own_goal_tag_shots"].sum()),
        "penalty_tag_shots": int(desc["penalty_tag_shots"].sum()),
        "regular_shots": int(desc["regular_shots"].sum()),
        "analytic_direct_free_kick_shots": int(desc["analytic_direct_free_kick_shots"].sum()),
        "analytic_penalties": int(desc["analytic_penalties"].sum()),
        "analytic_shots": int(desc["analytic_shots"].sum()),
        "analytic_goals": int(desc["analytic_goals"].sum()),
        "goal_rate": float(desc["analytic_goals"].sum() / desc["analytic_shots"].sum()),
    }
    desc = pd.concat([desc, pd.DataFrame([total])], ignore_index=True)
    desc.to_csv(OUT_DIR / "dataset_summary.csv", index=False)
    sample_construction_audit(desc).to_csv(OUT_DIR / "sample_construction_audit.csv", index=False)
    body_part_tag_audit(all_shots).to_csv(OUT_DIR / "body_part_tag_audit.csv", index=False)

    cols = usable_features(all_shots)
    world = all_shots[all_shots["competition"] == "World_Cup"].copy()
    perf, k_star, k_best = performance_curve(world, cols)
    perf.to_csv(OUT_DIR / "worldcup_perf_curve.csv", index=False)
    write_feature_inventory(k_star, k_best)
    print(f"[selection] k_star={k_star}, k_best={k_best}, features={len(cols)}")

    cols_base = cols[:1]
    cols_star = cols[:k_star]
    cols_wc_best = cols[:k_best]
    cols_full = cols

    model_rows = []
    pred_store = {}
    wc_eval = None
    for model_name, model_cols in tqdm(
        [
        ("Baseline(1)", cols_base),
        (f"Chosen(k*={k_star})", cols_star),
        (f"World Cup-selected Best(k={k_best})", cols_wc_best),
        ],
        desc="World Cup selected models",
        leave=False,
    ):
        d, y, p = oof_predictions(world, model_cols, "logistic", "isotonic_grouped")
        row = metrics(y, p, include_calibration_params=True)
        row.update({"setting": "World Cup grouped CV", "model": model_name, "k": len(model_cols)})
        model_rows.append(row)
        pred_store[model_name] = p
        wc_eval = d

    for calibration_label, calibration in tqdm(PRIMARY_LOGISTIC_CALIBRATIONS, desc="World Cup full calibration", leave=False):
        d, y, p = oof_predictions(world, cols_full, "logistic", calibration)
        row = metrics(y, p, include_calibration_params=True)
        model_label = logistic_model_label(calibration_label)
        row.update({"setting": "World Cup grouped CV", "model": model_label, "k": len(cols_full)})
        model_rows.append(row)
        pred_store[model_label] = p

    d, y, p = oof_predictions(world, cols_full, "hgb", PRIMARY_HGB_CALIBRATION)
    row = metrics(y, p, include_calibration_params=True)
    row.update({"setting": "World Cup grouped CV", "model": "HGB sigmoid fixed comparator", "k": len(cols_full)})
    model_rows.append(row)
    pred_store["HGB sigmoid fixed comparator"] = p

    # Pooled multi-competition robustness.
    pooled_pred_store = {}
    pooled_eval = None
    for calibration_label, calibration in tqdm(PRIMARY_LOGISTIC_CALIBRATIONS, desc="Pooled full calibration", leave=False):
        d, y, p = oof_predictions(all_shots, cols_full, "logistic", calibration)
        row = metrics(y, p, include_calibration_params=True)
        model_label = logistic_model_label(calibration_label)
        row.update({"setting": "Pooled grouped CV", "model": model_label, "k": len(cols_full)})
        model_rows.append(row)
        pooled_pred_store[model_label] = p
        pooled_eval = d
    d, y, p = oof_predictions(all_shots, cols_full, "hgb", PRIMARY_HGB_CALIBRATION)
    row = metrics(y, p, include_calibration_params=True)
    row.update({"setting": "Pooled grouped CV", "model": "HGB sigmoid fixed comparator", "k": len(cols_full)})
    model_rows.append(row)
    pooled_pred_store["HGB sigmoid fixed comparator"] = p

    all_shots_export = pooled_eval.copy()
    all_shots_export["xg_all_shot"] = pooled_pred_store["Full logistic sigmoid"]
    all_shots_export.to_csv(OUT_DIR / "all_competition_shots_features.csv", index=False)
    regular_shot_model_metrics(pooled_eval, pooled_pred_store).to_csv(
        OUT_DIR / "regular_shot_model_metrics.csv", index=False
    )
    penalty_xg_sensitivity(all_shots_export).to_csv(OUT_DIR / "penalty_xg_sensitivity.csv", index=False)
    direct_free_kick_submodel_metrics(all_shots_export).to_csv(
        OUT_DIR / "direct_free_kick_submodel_metrics.csv", index=False
    )
    calibration_strata_diagnostics(
        all_shots_export, all_shots_export["xg_all_shot"].astype(float).values
    ).to_csv(OUT_DIR / "calibration_strata_diagnostics.csv", index=False)

    pd.DataFrame(model_rows).to_csv(OUT_DIR / "model_metrics.csv", index=False)
    ece_bin_sensitivity(all_shots, cols_full).to_csv(OUT_DIR / "ece_bin_sensitivity.csv", index=False)
    ece_binning_sensitivity(all_shots, cols_full).to_csv(
        OUT_DIR / "ece_binning_sensitivity.csv", index=False
    )
    grouped_inner_calibration_sensitivity(all_shots, cols_full).to_csv(
        OUT_DIR / "grouped_inner_calibration_sensitivity.csv", index=False
    )

    logo_rows = []
    for comp in tqdm(COMPETITIONS, desc="leave-one-competition-out"):
        train = all_shots[all_shots["competition"] != comp].copy()
        test = all_shots[all_shots["competition"] == comp].copy()
        for calibration_label, calibration in [("sigmoid", "sigmoid_grouped"), ("isotonic", "isotonic_grouped")]:
            yte, pte = fit_predict_train_test(train, test, cols_full, "logistic", calibration)
            row = metrics(yte, pte, include_calibration_params=True)
            row.update({"held_out": comp.replace("_", " "), "model": f"logistic {calibration_label}", "k": len(cols_full)})
            logo_rows.append(row)
        yte, pte = fit_predict_train_test(train, test, cols_full, "hgb", PRIMARY_HGB_CALIBRATION)
        row = metrics(yte, pte, include_calibration_params=True)
        row.update({"held_out": comp.replace("_", " "), "model": "HGB sigmoid fixed comparator", "k": len(cols_full)})
        logo_rows.append(row)
    pd.DataFrame(logo_rows).to_csv(OUT_DIR / "leave_one_competition_out.csv", index=False)

    print("[experiment] validation design sensitivity")
    validation_design_sensitivity(all_shots, cols_full).to_csv(
        OUT_DIR / "validation_design_sensitivity.csv", index=False
    )
    validation_design_sensitivity_detailed(all_shots, cols_full).to_csv(
        OUT_DIR / "validation_design_sensitivity_detailed.csv", index=False
    )

    boot = bootstrap_diff(
        wc_eval,
        pred_store,
        [
            (f"World Cup-selected Best(k={k_best})", "Baseline(1)"),
            (f"Chosen(k*={k_star})", "Baseline(1)"),
            ("HGB sigmoid fixed comparator", f"World Cup-selected Best(k={k_best})"),
            (f"World Cup-selected Best(k={k_best})", "Full logistic sigmoid"),
        ],
    )
    boot.to_csv(OUT_DIR / "worldcup_bootstrap_logloss_diff.csv", index=False)

    pooled_boot = bootstrap_diff(
        pooled_eval,
        pooled_pred_store,
        [
            ("HGB sigmoid fixed comparator", "Full logistic sigmoid"),
            ("Full logistic isotonic", "Full logistic sigmoid"),
        ],
    )
    pooled_boot.to_csv(OUT_DIR / "pooled_bootstrap_logloss_diff.csv", index=False)

    print("[experiment] calibration sample-size stability")
    calibration_sample_size_stability(all_shots, cols_full).to_csv(
        OUT_DIR / "calibration_sample_size_stability.csv", index=False
    )

    print("[experiment] feature-family ablation")
    feature_df, feature_preds = feature_family_ablation(all_shots, world, cols_full, return_predictions=True)
    feature_df.to_csv(OUT_DIR / "feature_family_ablation.csv", index=False)
    pooled_family_preds = {
        name: pred for (setting, name), (_, pred) in feature_preds.items() if setting == "Pooled match-grouped CV"
    }
    pooled_family_df = next(
        d for (setting, name), (d, _) in feature_preds.items() if setting == "Pooled match-grouped CV"
    )
    bootstrap_diff(
        pooled_family_df,
        pooled_family_preds,
        [
            ("Full", "Geometry only"),
            ("Geometry + Execution", "Geometry only"),
            ("Geometry + Previous", "Geometry only"),
            ("Full minus Previous", "Full"),
            ("Full minus Execution", "Full"),
            ("Full minus Match", "Full"),
            ("Full minus Shot context", "Full"),
        ],
    ).to_csv(OUT_DIR / "feature_family_bootstrap.csv", index=False)
    pooled_pred_store["Geometry only"] = pooled_family_preds["Geometry only"]
    bootstrap_diff(
        pooled_eval,
        pooled_pred_store,
        [("Full logistic sigmoid", "Geometry only")],
    ).to_csv(OUT_DIR / "pooled_full_vs_geometry_bootstrap.csv", index=False)

    print("[experiment] target-domain recalibration")
    target_recal = target_domain_recalibration(all_shots, cols_full)
    target_recal.to_csv(OUT_DIR / "target_domain_recalibration.csv", index=False)
    target_domain_recalibration_by_competition(target_recal).to_csv(
        OUT_DIR / "target_domain_recalibration_by_competition.csv", index=False
    )
    target_recal_time = target_domain_recalibration_time_ordered(all_shots, cols_full)
    target_recal_time.to_csv(OUT_DIR / "target_domain_recalibration_time_ordered.csv", index=False)
    target_domain_recalibration_time_ordered_by_competition(target_recal_time).to_csv(
        OUT_DIR / "target_domain_recalibration_time_ordered_by_competition.csv", index=False
    )

    print(f"[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
