matches_path = "./data/data/worldcup_64_match.xlsx"
playersrank_path = "./data/data/worldcup_64_playersrank.csv"
players_path = "./data/data/worldcup_64_players.csv"
teams_path = "./data/data/worldcup_64_teams.csv"
events_path = "./data/data/worldcup_64_events.csv"

outdir = "./SCI-1/Final_results"

DO_OOF_IDENTITY = False
DO_CALIBRATION = True
N_SPLITS = 5
SEED = 42
PENALTY_XG = 0.79
SHOT_TYPE_REGULAR = "regular_shot"
SHOT_TYPE_DIRECT_FREE_KICK = "direct_free_kick"
SHOT_TYPE_PENALTY = "penalty"

import os, re, ast, json, math, warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    log_loss, brier_score_loss, mean_absolute_error,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

try:
    import joblib
except Exception:
    joblib = None

warnings.filterwarnings("ignore")
os.makedirs(outdir, exist_ok=True)

TAG_RESULT = {"Goal", "Not accurate", "Blocked", "Missed ball", "Own goal", "Opportunity"}
RESULT_PREFIXES = ["Position: Goal", "Position: Out", "Position: Post"]
TAG_INTERPRETABLE = {"Left foot", "Right foot", "Head/body"}

GOAL_X_NORM = 100.0
GOAL_Y_CENTER = 50.0

PITCH_LEN_M = 105.0
PITCH_WID_M = 68.0
Y_UNIT_TO_METERS = PITCH_WID_M / 100.0
X_UNIT_TO_METERS = PITCH_LEN_M / 100.0

GOAL_HALF_WIDTH_M = 7.32 / 2.0
GOAL_HALF_WIDTH_YUNITS = GOAL_HALF_WIDTH_M / Y_UNIT_TO_METERS

PERIOD_OFFSETS = {"1H": 0, "2H": 45 * 60, "E1": 90 * 60, "E2": 105 * 60}


def parse_list(cell) -> List:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if isinstance(cell, (list, tuple, set)):
        return list(cell)
    s = str(cell)
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return []


def is_result_tag(tag: str) -> bool:
    if tag in TAG_RESULT:
        return True
    return any(str(tag).startswith(p) for p in RESULT_PREFIXES)


def tags_to_interpretable_flags(tag_list) -> Dict[str, int]:
    tags = [str(t) for t in parse_list(tag_list) if (t in TAG_INTERPRETABLE and not is_result_tag(t))]
    return {
        "feat_tag_left_foot": int("Left foot" in tags),
        "feat_tag_right_foot": int("Right foot" in tags),
        "feat_tag_head": int("Head/body" in tags),
    }


def to_attacking_frame(x, y, is_attacking_right: bool):
    if pd.isna(x) or pd.isna(y):
        return np.nan, np.nan
    return (x, y) if is_attacking_right else (100.0 - x, y)


def shot_geometry_features(ax: float, ay: float) -> Dict[str, float]:
    if any(pd.isna(v) for v in (ax, ay)):
        return {
            "feat_dist_m": np.nan, "feat_angle_rad": np.nan,
            "feat_x_m": np.nan, "feat_y_m": np.nan, "feat_lat_offset_m": np.nan
        }
    dx_units = max(0.0, GOAL_X_NORM - ax)
    dy_units = ay - GOAL_Y_CENTER

    x_m = dx_units * X_UNIT_TO_METERS
    y_m = dy_units * Y_UNIT_TO_METERS
    lat_offset_m = abs(y_m)

    y_post_top = GOAL_Y_CENTER - GOAL_HALF_WIDTH_YUNITS
    y_post_bot = GOAL_Y_CENTER + GOAL_HALF_WIDTH_YUNITS
    a_top = math.atan2(ay - y_post_top, dx_units)
    a_bot = math.atan2(ay - y_post_bot, dx_units)
    angle = abs(a_top - a_bot)

    dist_units = math.hypot(dx_units, dy_units)
    dist_m = dist_units * X_UNIT_TO_METERS

    return {
        "feat_dist_m": dist_m,
        "feat_angle_rad": angle,
        "feat_x_m": x_m,
        "feat_y_m": y_m,
        "feat_lat_offset_m": lat_offset_m
    }


def label_is_goal(tags) -> int:
    tags = [str(t) for t in (tags if isinstance(tags, (list, tuple, set)) else parse_list(tags))]
    return int(("Goal" in set(tags)) or ("Own goal" in set(tags)))


def label_is_on_target(tags) -> int:
    tags = [str(t) for t in (tags if isinstance(tags, (list, tuple, set)) else parse_list(tags))]
    pos_goal = any(t.startswith("Position: Goal") for t in tags)
    accurate = ("Accurate" in set(tags))
    return int(pos_goal or accurate)


def norm_token(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(t).lower())


def has_any(tags: List[str], candidates: List[str]) -> int:
    s = {norm_token(x) for x in tags}
    return int(any(norm_token(c) in s for c in candidates))


def add_situ_features_from_tags(shots: pd.DataFrame) -> pd.DataFrame:
    s = shots.copy()
    s["__tags__"] = s["__tags__"].apply(parse_list)
    s["feat_is_counter"] = s["__tags__"].apply(lambda L: has_any(L, ["Counter attack", "Counterattack"]))
    s["feat_is_cross"] = s["__tags__"].apply(lambda L: has_any(L, ["Cross", "Crossing"]))
    if "shot_type" in s.columns:
        s["feat_is_penalty"] = (s["shot_type"] == SHOT_TYPE_PENALTY).astype(int)
        s["feat_is_free_kick"] = (s["shot_type"] == SHOT_TYPE_DIRECT_FREE_KICK).astype(int)
    else:
        s["feat_is_penalty"] = s["__tags__"].apply(lambda L: has_any(L, ["Penalty"]))
        s["feat_is_free_kick"] = s["__tags__"].apply(
            lambda L: has_any(L, ["Free kick", "Freekick", "Direct", "Indirect"])
        )
    s["feat_is_corner"] = s["__tags__"].apply(lambda L: has_any(L, ["Corner"]))
    s["feat_sub_penalty"] = s["feat_is_penalty"]
    s["feat_sub_corner"] = s["feat_is_corner"]
    s["feat_sub_freekick"] = s["feat_is_free_kick"]
    return s


def infer_team_attacking_right_per_period(events: pd.DataFrame, matches: pd.DataFrame) -> Dict[
    Tuple[int, int, str], bool]:
    mm = matches[["wyId", "homeId", "awayId"]].rename(columns={"wyId": "matchId"})
    ev = events[["matchId", "matchPeriod", "teamId", "eventName", "x_start"]].copy().merge(mm, on="matchId", how="left")
    ev = ev[ev["matchPeriod"].isin(["1H", "2H", "E1", "E2"])].dropna(subset=["x_start"])

    shot_med = (ev[ev["eventName"].astype(str).str.lower() == "shot"]
                .groupby(["matchId", "teamId", "matchPeriod"])["x_start"].median())
    all_med = ev.groupby(["matchId", "teamId", "matchPeriod"])["x_start"].median()

    dir_map: Dict[Tuple[int, int, str], bool] = {}
    for key, xmed in shot_med.items():
        dir_map[(int(key[0]), int(key[1]), str(key[2]))] = bool(float(xmed) > 50.0)
    for key, xmed in all_med.items():
        k = (int(key[0]), int(key[1]), str(key[2]))
        if k not in dir_map:
            dir_map[k] = bool(float(xmed) > 50.0)

    period_prev = {"2H": "1H", "E1": "2H", "E2": "E1"}
    for mid, grp in ev.groupby("matchId"):
        teams = set(grp["teamId"].unique().tolist())
        for per in ["1H", "2H", "E1", "E2"]:
            for tid in teams:
                k = (int(mid), int(tid), per)
                if k in dir_map:
                    continue
                # opponent reverse
                for opp in (teams - {tid}):
                    kk = (int(mid), int(opp), per)
                    if kk in dir_map:
                        dir_map[k] = (not dir_map[kk])
                        break
                if k in dir_map:
                    continue
                # inherit from previous period
                prev = period_prev.get(per)
                if prev and (int(mid), int(tid), prev) in dir_map:
                    dir_map[k] = dir_map[(int(mid), int(tid), prev)]
                else:
                    dir_map[k] = True
    return dir_map


def build_prev_event_context(events: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    team_dir_map = infer_team_attacking_right_per_period(events, matches)
    ev = events.copy()
    ev = ev[ev["matchPeriod"].isin(["1H", "2H", "E1", "E2"])].copy()

    ev["period_offset_s"] = ev["matchPeriod"].map(PERIOD_OFFSETS).fillna(0).astype(int)
    ev["t_abs_s"] = ev["period_offset_s"] + pd.to_numeric(ev["eventSec"], errors="coerce").fillna(0.0)

    ev["att_right"] = ev.apply(
        lambda r: team_dir_map.get((int(r["matchId"]), int(r["teamId"]), str(r["matchPeriod"])), True),
        axis=1
    ).astype(bool)

    norm = ev[["x_start", "y_start", "att_right"]].apply(
        lambda r: to_attacking_frame(r["x_start"], r["y_start"], bool(r["att_right"])),
        axis=1
    )
    ev["ax"] = [t[0] for t in norm]
    ev["ay"] = [t[1] for t in norm]

    ev = ev.sort_values(["matchId", "teamId", "t_abs_s", "id"])
    ev["prev_t"] = ev.groupby(["matchId", "teamId"])["t_abs_s"].shift(1)
    ev["prev_ax"] = ev.groupby(["matchId", "teamId"])["ax"].shift(1)
    ev["prev_ay"] = ev.groupby(["matchId", "teamId"])["ay"].shift(1)

    ev["tags_list"] = ev["tag_descriptions"].apply(parse_list)
    ev["prev_tags"] = ev.groupby(["matchId", "teamId"])["tags_list"].shift(1)

    def _has(prev, token):
        if isinstance(prev, (list, tuple, set)):
            return int(token in set(prev))
        return 0

    ev["prev_through"] = ev["prev_tags"].apply(lambda t: _has(t, "Through"))
    ev["prev_direct_fk"] = ev["prev_tags"].apply(lambda t: _has(t, "Direct"))
    ev["prev_indirect_fk"] = ev["prev_tags"].apply(lambda t: _has(t, "Indirect"))

    keep = ["id", "matchId", "teamId", "matchPeriod", "prev_t", "prev_ax", "prev_ay",
            "prev_through", "prev_direct_fk", "prev_indirect_fk"]
    for c in ["id", "matchId", "teamId"]:
        ev[c] = pd.to_numeric(ev[c], errors="coerce").astype("Int64")
    return ev[keep].copy()


def build_shots(matches: pd.DataFrame, players: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    evctx = build_prev_event_context(events, matches)

    df = events.copy()
    df["eventName_low"] = df["eventName"].astype(str).str.lower()
    df["subEventName_str"] = df["subEventName"].astype(str) if "subEventName" in df.columns else ""
    df = df[df["matchPeriod"].isin(["1H", "2H", "E1", "E2"])].copy()
    df["__tags__"] = df["tag_descriptions"].apply(parse_list)

    direct_fk_mask = (df["eventName_low"] == "free kick") & df["subEventName_str"].str.fullmatch(
        "Free kick shot", case=False
    )
    penalty_mask = (df["eventName_low"] == "free kick") & df["subEventName_str"].str.fullmatch(
        "Penalty", case=False
    )
    shots = df[(df["eventName_low"] == "shot") | direct_fk_mask | penalty_mask].copy()
    shots["shot_type"] = SHOT_TYPE_REGULAR
    shots.loc[
        (shots["eventName_low"] == "free kick")
        & shots["subEventName_str"].str.fullmatch("Free kick shot", case=False),
        "shot_type"
    ] = SHOT_TYPE_DIRECT_FREE_KICK
    shots.loc[
        (shots["eventName_low"] == "free kick")
        & shots["subEventName_str"].str.fullmatch("Penalty", case=False),
        "shot_type"
    ] = SHOT_TYPE_PENALTY
    shots["is_regular_shot"] = (shots["shot_type"] == SHOT_TYPE_REGULAR).astype(int)
    shots["is_direct_free_kick"] = (shots["shot_type"] == SHOT_TYPE_DIRECT_FREE_KICK).astype(int)
    shots["is_penalty"] = (shots["shot_type"] == SHOT_TYPE_PENALTY).astype(int)

    matches_min = matches[["wyId", "homeId", "awayId"]].rename(columns={"wyId": "matchId"})
    shots = shots.merge(matches_min, on="matchId", how="left")

    shots["is_home_team"] = (shots["teamId"] == shots["homeId"]).astype(int)
    shots["opponentId"] = np.where(shots["is_home_team"] == 1, shots["awayId"], shots["homeId"])

    shots["period_offset_s"] = shots["matchPeriod"].map(PERIOD_OFFSETS).fillna(0).astype(int)
    shots["t_abs_s"] = shots["period_offset_s"] + pd.to_numeric(shots["eventSec"], errors="coerce").fillna(0.0)

    team_dir_map = infer_team_attacking_right_per_period(events, matches)
    shots["att_right"] = shots.apply(
        lambda r: team_dir_map.get((int(r["matchId"]), int(r["teamId"]), str(r["matchPeriod"])), True),
        axis=1
    ).astype(bool)

    norm = shots[["x_start", "y_start", "att_right"]].apply(
        lambda r: to_attacking_frame(r["x_start"], r["y_start"], bool(r["att_right"])),
        axis=1
    )
    shots["ax"] = [t[0] for t in norm]
    shots["ay"] = [t[1] for t in norm]

    geom = shots[["ax", "ay"]].apply(lambda r: shot_geometry_features(r["ax"], r["ay"]), axis=1)
    shots = pd.concat([shots, pd.DataFrame(list(geom))], axis=1)

    shots["feat_minute"] = shots["t_abs_s"] / 60.0
    shots["feat_cos_angle"] = np.cos(shots["feat_angle_rad"])
    shots["feat_inv_dist"] = 1.0 / (1.0 + shots["feat_dist_m"])
    shots["feat_angle_x_dist"] = shots["feat_angle_rad"] * shots["feat_dist_m"]

    tag_flags_df = pd.DataFrame(list(shots["__tags__"].apply(tags_to_interpretable_flags)))
    shots = pd.concat([shots, tag_flags_df], axis=1)

    for c in ["id", "matchId", "teamId"]:
        shots[c] = pd.to_numeric(shots[c], errors="coerce").astype("Int64")
    shots = shots.merge(evctx, on=["id", "matchId", "teamId", "matchPeriod"], how="left")

    XU2M, YU2M = X_UNIT_TO_METERS, Y_UNIT_TO_METERS
    shots["feat_prev_dt"] = (shots["t_abs_s"] - shots["prev_t"]).fillna(999.0)
    shots["feat_prev_dx_m"] = (shots["ax"] - shots["prev_ax"]) * XU2M
    shots["feat_prev_dy_m"] = (shots["ay"] - shots["prev_ay"]) * YU2M
    shots["feat_prev_dist_m"] = np.hypot(shots["feat_prev_dx_m"], shots["feat_prev_dy_m"])
    shots["feat_prev_speed_mps"] = shots["feat_prev_dist_m"] / shots["feat_prev_dt"].replace(0, np.nan)
    shots["feat_prev_speed_mps"] = shots["feat_prev_speed_mps"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    shots["feat_prev_lateral_m"] = shots["feat_prev_dy_m"].abs()

    shots["feat_prev_through"] = shots["prev_through"].fillna(0).astype(int)
    shots["feat_prev_direct_fk"] = shots["prev_direct_fk"].fillna(0).astype(int)
    shots["feat_prev_indirect_fk"] = shots["prev_indirect_fk"].fillna(0).astype(int)

    # player foot
    if "wyId" in players.columns and "foot" in players.columns:
        pmin = players[["wyId", "foot"]].rename(columns={"wyId": "playerId", "foot": "player_foot"})
        shots = shots.merge(pmin, on="playerId", how="left")
    else:
        shots["player_foot"] = np.nan

    def _weak_foot(row):
        pf = str(row.get("player_foot", "")).lower()
        if row.get("feat_tag_head", 0) == 1:
            return 0
        if row.get("feat_tag_left_foot", 0) == 1:
            return int(pf == "right")
        if row.get("feat_tag_right_foot", 0) == 1:
            return int(pf == "left")
        return 0

    shots["feat_weak_foot"] = shots.apply(_weak_foot, axis=1)

    shots["is_goal"] = shots["__tags__"].apply(label_is_goal).astype(int)
    shots["is_on_target"] = shots["__tags__"].apply(label_is_on_target).astype(int)

    # score diff BEFORE
    shots = shots.sort_values(["matchId", "teamId", "t_abs_s", "id"]).reset_index(drop=True)
    shots["goal_for"] = shots["is_goal"].astype(int)
    shots["gf_before"] = shots.groupby(["matchId", "teamId"])["goal_for"].cumsum().shift(1, fill_value=0)

    opp_goal_times: Dict[Tuple[int, int], np.ndarray] = {}
    gshots = shots[shots["is_goal"] == 1][["matchId", "teamId", "t_abs_s"]].copy()
    for (mid, tid), grp in gshots.groupby(["matchId", "teamId"]):
        opp_goal_times[(int(mid), int(tid))] = np.sort(grp["t_abs_s"].astype(float).values)

    def _count_opp_goals_before(mid, oppid, t):
        arr = opp_goal_times.get((int(mid), int(oppid)))
        if arr is None or arr.size == 0:
            return 0
        return int(np.searchsorted(arr, float(t), side="left"))

    shots["ga_before"] = shots.apply(
        lambda r: _count_opp_goals_before(r["matchId"], r["opponentId"], r["t_abs_s"]),
        axis=1
    ).astype(int)

    shots["feat_score_diff_before"] = shots["gf_before"] - shots["ga_before"]

    feature_required = shots["shot_type"] != SHOT_TYPE_PENALTY
    missing_geometry = feature_required & shots[["feat_dist_m", "feat_angle_rad"]].isna().any(axis=1)
    shots = shots.loc[~missing_geometry].dropna(subset=["matchId", "is_goal"]).copy()
    return shots


def stabilize_prev_features(shots: pd.DataFrame) -> pd.DataFrame:
    s = shots.copy()
    if "feat_prev_exists" not in s.columns and "feat_prev_dt" in s.columns:
        s["feat_prev_exists"] = (s["feat_prev_dt"] < 990.0).astype(int)
    for c in ["feat_prev_dx_m", "feat_prev_dy_m", "feat_prev_dist_m", "feat_prev_lateral_m"]:
        if c in s.columns:
            s[c] = s[c].fillna(0.0)
    for c in ["feat_prev_through", "feat_prev_direct_fk", "feat_prev_indirect_fk"]:
        if c in s.columns:
            s[c] = s[c].fillna(0).astype(int)
    return s


def feature_order_25(use_identity_oof=False) -> List[str]:
    order = [
        "feat_dist_m", "feat_angle_rad", "feat_inv_dist", "feat_cos_angle", "feat_angle_x_dist",
        "feat_x_m", "feat_y_m", "feat_lat_offset_m",
        "feat_minute", "feat_score_diff_before",
        "feat_prev_exists", "feat_prev_dt", "feat_prev_dist_m", "feat_prev_speed_mps",
        "feat_prev_lateral_m", "feat_prev_dx_m", "feat_prev_dy_m",
        "feat_tag_left_foot", "feat_tag_right_foot", "feat_tag_head", "feat_weak_foot",
        "feat_prev_through", "feat_prev_direct_fk", "feat_prev_indirect_fk",
        "feat_is_counter", "feat_is_cross",
    ]
    if use_identity_oof:
        order += ["enc_teamId_oof", "enc_playerId_oof", "enc_opponentId_oof"]
    return order


def _clip01(p, eps=1e-15):
    p = np.asarray(p, float)
    return np.clip(p, eps, 1.0 - eps)


def nagelkerke_r2(y, p, eps=1e-15):
    y = np.asarray(y, int);
    n = y.size
    p = _clip01(p, eps)
    p0 = float(y.mean())
    LLm = (y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    LL0 = (y * np.log(p0) + (1 - y) * np.log(1 - p0)).sum()
    cs = 1.0 - np.exp((2.0 / n) * (LL0 - LLm))
    max_cs = 1.0 - np.exp((2.0 / n) * LL0)
    return float(cs / max(max_cs, eps))


def expected_calibration_error(y, p, n_bins=12):
    y = np.asarray(y, int);
    p = np.asarray(p, float)
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    q[0] = -np.inf;
    q[-1] = np.inf
    ece = 0.0
    for i in range(n_bins):
        m = (p >= q[i]) & (p < q[i + 1])
        if m.sum() == 0:
            continue
        ece += abs(y[m].mean() - p[m].mean()) * m.mean()
    return float(ece)


def _make_calibrator(estimator, method="isotonic", cv=3):
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


class ConstantProbabilityModel:
    def __init__(self, p: float):
        self.p = float(np.clip(p, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def _clean_for_composite(shots: pd.DataFrame, cols: List[str], target_col="is_goal") -> pd.DataFrame:
    s = shots.dropna(subset=[target_col, "matchId"]).copy()
    if "shot_type" not in s.columns:
        s["shot_type"] = SHOT_TYPE_REGULAR
    feature_ok = s[cols].notna().all(axis=1) if len(cols) else pd.Series(True, index=s.index)
    penalty_mask = s["shot_type"].astype(str) == SHOT_TYPE_PENALTY
    return s.loc[penalty_mask | feature_ok].copy()


def _fit_sklearn_model(train: pd.DataFrame, cols: List[str], target_col="is_goal", calibrate=True):
    train = train.dropna(subset=cols + [target_col]).copy()
    if len(train) == 0:
        return ConstantProbabilityModel(0.05)

    y = train[target_col].astype(int).values
    if len(np.unique(y)) < 2:
        return ConstantProbabilityModel(float(np.mean(y)))

    effective_calibrate = calibrate
    if calibrate:
        counts = np.bincount(y, minlength=2)
        if counts.min() < 3:
            effective_calibrate = False

    X = train[cols].astype(float).values
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=3000, solver="lbfgs"))
    ])
    model = _make_calibrator(pipe, method="isotonic", cv=3) if effective_calibrate else pipe
    model.fit(X, y)
    return model


def _predict_sklearn_model(model, test: pd.DataFrame, cols: List[str]) -> np.ndarray:
    if len(test) == 0:
        return np.array([], dtype=float)
    X = test[cols].astype(float).values
    return model.predict_proba(X)[:, 1]


def fit_composite_xg_model(train: pd.DataFrame, cols: List[str], target_col="is_goal", calibrate=True):
    d = _clean_for_composite(train, cols, target_col=target_col)
    regular = d[d["shot_type"] == SHOT_TYPE_REGULAR].copy()
    direct_fk = d[d["shot_type"] == SHOT_TYPE_DIRECT_FREE_KICK].copy()
    return {
        "regular": _fit_sklearn_model(regular, cols, target_col=target_col, calibrate=calibrate),
        "direct_free_kick": _fit_sklearn_model(direct_fk, cols, target_col=target_col, calibrate=calibrate),
        "penalty_xg": PENALTY_XG,
        "cols": list(cols),
    }


def predict_composite_xg_model(model, shots: pd.DataFrame, cols: List[str], target_col="is_goal"):
    d = _clean_for_composite(shots, cols, target_col=target_col).copy().reset_index(drop=True)
    y = d[target_col].astype(int).values
    pred = np.full(len(d), np.nan, dtype=float)

    penalty_mask = d["shot_type"].astype(str).values == SHOT_TYPE_PENALTY
    regular_mask = d["shot_type"].astype(str).values == SHOT_TYPE_REGULAR
    direct_mask = d["shot_type"].astype(str).values == SHOT_TYPE_DIRECT_FREE_KICK

    pred[penalty_mask] = float(model.get("penalty_xg", PENALTY_XG))
    pred[regular_mask] = _predict_sklearn_model(model["regular"], d.loc[regular_mask], cols)
    pred[direct_mask] = _predict_sklearn_model(model["direct_free_kick"], d.loc[direct_mask], cols)

    if np.isnan(pred).any():
        raise ValueError("Composite all-shot xG prediction left some rows without probabilities.")
    return d, y, pred


def oof_predict_and_metrics(shots: pd.DataFrame, cols: List[str], target_col="is_goal",
                            calibrate=True, n_splits=5, seed=42):
    s = _clean_for_composite(shots, cols, target_col=target_col)
    y = s[target_col].astype(int).values
    g = s["matchId"].values

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros_like(y), y, g))

    oof = np.zeros_like(y, float)
    for tr, va in folds:
        model = fit_composite_xg_model(s.iloc[tr], cols, target_col=target_col, calibrate=calibrate)
        _, _, pv = predict_composite_xg_model(model, s.iloc[va], cols, target_col=target_col)
        oof[va] = pv

    base_rate = y.mean()
    brier_null = float(base_rate * (1 - base_rate))

    metrics = {
        "n_eval": int(len(y)),
        "goal_rate": float(base_rate),
        "R2": nagelkerke_r2(y, oof),
        "LogLoss": float(log_loss(y, _clip01(oof), labels=[0, 1])),
        "Brier": float(brier_score_loss(y, oof)),
        "BSS": float(1.0 - brier_score_loss(y, oof) / brier_null) if brier_null > 0 else float("nan"),
        "ECE": float(expected_calibration_error(y, oof)),
        "ROC_AUC": float(roc_auc_score(y, oof)),
        "PR_AUC": float(average_precision_score(y, oof)),
        "MAE": float(mean_absolute_error(y, oof)),
    }
    return oof, y, metrics, s.reset_index(drop=True)


def compute_perf_curve(
        shots: pd.DataFrame,
        feats_order: List[str],
        target_col='is_goal',
        calibrate=True,
        n_splits=5,
        seed=42,
        allow_identity_oof=False
) -> pd.DataFrame:
    s = stabilize_prev_features(shots)

    usable = []
    for c in feats_order:
        if (not allow_identity_oof) and str(c).startswith("enc_"):
            continue
        if c not in s.columns:
            continue
        if s[c].notna().sum() == 0:
            continue
        if s[c].nunique(dropna=True) <= 1:
            continue
        usable.append(c)

    se = _clean_for_composite(s, usable, target_col=target_col).reset_index(drop=True)

    y = se[target_col].astype(int).values
    groups = se["matchId"].values

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_indices = list(cv.split(np.zeros_like(y), y, groups))

    base_rate = y.mean()
    brier_null = float(base_rate * (1 - base_rate))

    rows = []
    selected = []
    for step, col in enumerate(usable, start=1):
        selected.append(col)

        oof = np.zeros_like(y, float)
        per_fold = []

        for tr, va in fold_indices:
            model = fit_composite_xg_model(se.iloc[tr], selected, target_col=target_col, calibrate=calibrate)
            _, _, pv = predict_composite_xg_model(model, se.iloc[va], selected, target_col=target_col)
            oof[va] = pv

            per_fold.append({
                "LogLoss": log_loss(y[va], _clip01(pv), labels=[0, 1]),
                "Brier": brier_score_loss(y[va], pv),
                "ROC_AUC": roc_auc_score(y[va], pv),
                "PR_AUC": average_precision_score(y[va], pv),
                "MAE": mean_absolute_error(y[va], pv),
                "ECE": expected_calibration_error(y[va], pv),
            })

        fm = pd.DataFrame(per_fold)
        agg = fm.mean()
        se_ = fm.std(ddof=1) / math.sqrt(len(fm))

        rows.append({
            "step": step,
            "num_params": len(selected),
            "added": col,
            "n_eval": int(len(y)),
            "R2": nagelkerke_r2(y, oof),
            "LogLoss": float(agg["LogLoss"]),
            "LogLoss_SE": float(se_["LogLoss"]),
            "Brier": float(agg["Brier"]),
            "Brier_SE": float(se_["Brier"]),
            "BSS": float(1.0 - agg["Brier"] / brier_null) if brier_null > 0 else float("nan"),
            "ECE": float(agg["ECE"]),
            "ROC_AUC": float(agg["ROC_AUC"]),
            "PR_AUC": float(agg["PR_AUC"]),
            "MAE": float(agg["MAE"]),
        })

    return pd.DataFrame(rows)


def one_se_rule_on_logloss(perf_df: pd.DataFrame):
    idx_best = perf_df["LogLoss"].idxmin()
    best_mean = float(perf_df.loc[idx_best, "LogLoss"])
    best_se = float(perf_df.loc[idx_best, "LogLoss_SE"])
    thr = best_mean + best_se

    ok = perf_df[perf_df["LogLoss"] <= thr].copy()
    k_star = int(ok["num_params"].iloc[0]) if len(ok) else int(perf_df.loc[idx_best, "num_params"])
    k_best = int(perf_df.loc[idx_best, "num_params"])
    return k_star, k_best, best_mean, best_se


def compute_3model_metrics(shots: pd.DataFrame,
                           perf_df: pd.DataFrame,
                           k_star: int,
                           k_best: int,
                           target_col="is_goal",
                           calibrate=True,
                           n_splits=5,
                           seed=42) -> pd.DataFrame:
    usable_order = perf_df.sort_values("num_params")["added"].tolist()
    cols_base = usable_order[:1]
    cols_star = usable_order[:k_star]
    cols_best = usable_order[:k_best]

    shots_common = _clean_for_composite(shots, cols_best, target_col=target_col)

    p_b, y, m_b, _ = oof_predict_and_metrics(
        shots_common, cols_base, target_col,
        calibrate=calibrate, n_splits=n_splits, seed=seed
    )
    p_s, y2, m_s, _ = oof_predict_and_metrics(
        shots_common, cols_star, target_col,
        calibrate=calibrate, n_splits=n_splits, seed=seed
    )
    p_k, y3, m_k, _ = oof_predict_and_metrics(
        shots_common, cols_best, target_col,
        calibrate=calibrate, n_splits=n_splits, seed=seed
    )

    if (not np.array_equal(y, y2)) or (not np.array_equal(y, y3)):
        raise RuntimeError("Evaluation y mismatch: ensure the same shots_common is used for all three models.")

    def _row(name, k, m):
        return {
            "Model": name,
            "k": int(k),
            "R2": float(m["R2"]),
            "LogLoss": float(m["LogLoss"]),
            "Brier": float(m["Brier"]),
            "BSS": float(m["BSS"]),
            "ECE": float(m["ECE"]),
            "ROC-AUC": float(m["ROC_AUC"]),
            "PR-AUC": float(m["PR_AUC"]),
            "MAE": float(m["MAE"]),
            "n_eval": int(m["n_eval"]),
            "goal_rate": float(m["goal_rate"]),
        }

    rows = [
        _row("Baseline(1)", 1, m_b),
        _row(f"Chosen(k*={k_star})", k_star, m_s),
        _row(f"Best(k_best={k_best})", k_best, m_k),
    ]
    df = pd.DataFrame(rows)

    df = df[["Model", "k", "R2", "LogLoss", "Brier", "BSS", "ECE", "ROC-AUC", "PR-AUC", "MAE", "n_eval", "goal_rate"]]

    return df


PUB = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "grey": "#7f7f7f",
    "black": "#111111",
}
PUB_SEQ_CMAP = "viridis"


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
        "axes.grid": False,  # global off (avoid pitch grid)
        "grid.alpha": 0.18,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "legend.frameon": False,
    })


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def _grid(ax):
    ax.grid(True, alpha=0.18, linewidth=0.8)


def _savefig(fig, savepath=None):
    fig.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def draw_attacking_third(ax, x_min=50, x_max=100, y_min=0, y_max=100, lw=1.2):
    # attacking third boundary
    ax.plot([x_min, x_max], [y_min, y_min], color=PUB["black"], lw=lw)
    ax.plot([x_min, x_max], [y_max, y_max], color=PUB["black"], lw=lw)
    ax.plot([x_min, x_min], [y_min, y_max], color=PUB["black"], lw=lw)
    ax.plot([x_max, x_max], [y_min, y_max], color=PUB["black"], lw=lw)

    # goal mouth (x=100)
    y1 = GOAL_Y_CENTER - GOAL_HALF_WIDTH_YUNITS
    y2 = GOAL_Y_CENTER + GOAL_HALF_WIDTH_YUNITS
    ax.plot([100, 100], [y1, y2], color=PUB["black"], lw=3.0)

    # penalty area + six-yard box
    pa_depth = 16.5 / PITCH_LEN_M * 100.0
    six_depth = 5.5 / PITCH_LEN_M * 100.0
    pa_width = 40.32 / PITCH_WID_M * 100.0
    six_width = 18.32 / PITCH_WID_M * 100.0

    pa_x = 100.0 - pa_depth
    six_x = 100.0 - six_depth
    pa_y1 = 50.0 - pa_width / 2.0
    pa_y2 = 50.0 + pa_width / 2.0
    six_y1 = 50.0 - six_width / 2.0
    six_y2 = 50.0 + six_width / 2.0

    ax.plot([pa_x, 100], [pa_y1, pa_y1], color=PUB["black"], lw=lw)
    ax.plot([pa_x, 100], [pa_y2, pa_y2], color=PUB["black"], lw=lw)
    ax.plot([pa_x, pa_x], [pa_y1, pa_y2], color=PUB["black"], lw=lw)

    ax.plot([six_x, 100], [six_y1, six_y1], color=PUB["black"], lw=lw)
    ax.plot([six_x, 100], [six_y2, six_y2], color=PUB["black"], lw=lw)
    ax.plot([six_x, six_x], [six_y1, six_y2], color=PUB["black"], lw=lw)

    # penalty spot (11m)
    pen_depth = 11.0 / PITCH_LEN_M * 100.0
    pen_x = 100.0 - pen_depth
    ax.scatter([pen_x], [50.0], s=18, color=PUB["black"], zorder=5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_xg_shot_map(shots_eval: pd.DataFrame, prob_col: str, savepath=None):
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    _despine(ax)
    ax.grid(False)
    draw_attacking_third(ax)

    p = np.clip(shots_eval[prob_col].astype(float).values, 1e-6, 1 - 1e-6)
    size = 36 + 860 * p

    m_goal = shots_eval["is_goal"].astype(int).values == 1
    m_nogoal = ~m_goal

    ax.scatter(shots_eval.loc[m_nogoal, "ax"], shots_eval.loc[m_nogoal, "ay"],
               s=size[m_nogoal], c=PUB["blue"], alpha=0.26, edgecolor="none",
               label="Non-goal", zorder=3)

    ax.scatter(shots_eval.loc[m_goal, "ax"], shots_eval.loc[m_goal, "ay"],
               s=size[m_goal], c=PUB["orange"], alpha=0.90,
               edgecolor="white", linewidth=0.6, label="Goal", zorder=4)

    ax.legend(loc="lower left")
    return _savefig(fig, savepath)


def fit_geom_model(shots: pd.DataFrame, cols_geom: List[str], target_col="is_goal", calibrate=True):
    s = shots.dropna(subset=cols_geom + [target_col]).copy()
    X = s[cols_geom].astype(float).values
    y = s[target_col].astype(int).values

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs"))
    ])
    model = _make_calibrator(pipe, method="isotonic", cv=5) if calibrate else pipe
    model.fit(X, y)
    return model


def make_geom_grid_predictions(model, cols_geom,
                               x_min=50, x_max=100, y_min=0, y_max=100,
                               nx=151, ny=101):
    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    XX, YY = np.meshgrid(xs, ys)

    ax_flat = XX.ravel()
    ay_flat = YY.ravel()

    geom_list = [shot_geometry_features(a, b) for a, b in zip(ax_flat, ay_flat)]
    geom_df = pd.DataFrame(geom_list)

    if "feat_inv_dist" in cols_geom:
        geom_df["feat_inv_dist"] = 1.0 / (1.0 + geom_df["feat_dist_m"])
    if "feat_cos_angle" in cols_geom:
        geom_df["feat_cos_angle"] = np.cos(geom_df["feat_angle_rad"])
    if "feat_angle_x_dist" in cols_geom:
        geom_df["feat_angle_x_dist"] = geom_df["feat_angle_rad"] * geom_df["feat_dist_m"]

    missing = [c for c in cols_geom if c not in geom_df.columns]
    if missing:
        raise ValueError(f"make_geom_grid_predictions: unsupported cols_geom: {missing}")

    Xg = geom_df[cols_geom].astype(float).values
    Pg = model.predict_proba(Xg)[:, 1]
    Z = Pg.reshape(YY.shape)
    return XX, YY, Z


def plot_xg_surface(model, cols_geom, savepath=None):
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    _despine(ax)
    ax.grid(False)
    draw_attacking_third(ax)

    XX, YY, Z = make_geom_grid_predictions(model, cols_geom)
    im = ax.imshow(Z, origin="lower",
                   extent=[XX.min(), XX.max(), YY.min(), YY.max()],
                   cmap=PUB_SEQ_CMAP, alpha=0.90, aspect="equal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.set_ylabel("xG", rotation=0, labelpad=14)
    return _savefig(fig, savepath)


def plot_figure3_elegant(perf_df, k_star=None, k_best=None, savepath=None):
    x = perf_df["num_params"].values
    r2 = perf_df["R2"].values
    ll = perf_df["LogLoss"].values
    ll_se = perf_df["LogLoss_SE"].values if "LogLoss_SE" in perf_df.columns else None

    fig, ax1 = plt.subplots(figsize=(11.2, 5.8))
    _despine(ax1)
    _grid(ax1)

    ax1.plot(x, r2, color=PUB["blue"], lw=2.4, marker="o", ms=5.2, label="Nagelkerke $R^2$")
    ax1.set_xlabel("Number of features")
    ax1.set_ylabel("$R^2$", color=PUB["blue"])
    ax1.tick_params(axis="y", colors=PUB["blue"])

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(x, ll, color=PUB["orange"], lw=2.4, marker="D", ms=5.0, label="LogLoss")
    if ll_se is not None and np.all(np.isfinite(ll_se)):
        ax2.fill_between(x, ll - ll_se, ll + ll_se, color=PUB["orange"], alpha=0.18, linewidth=0)
    ax2.set_ylabel("LogLoss", color=PUB["orange"])
    ax2.tick_params(axis="y", colors=PUB["orange"])

    if k_star is not None:
        ax1.axvline(k_star, color=PUB["grey"], ls="--", lw=1.6)
        ax1.text(k_star + 0.2, ax1.get_ylim()[0] + 0.03 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                 f"$k^*$={k_star}", color=PUB["grey"])
    if k_best is not None:
        ax1.axvline(k_best, color=PUB["grey"], ls=":", lw=1.8)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper center", ncol=2)
    return _savefig(fig, savepath)


def plot_delta_logloss_pub(perf_df, savepath=None):
    x = perf_df["num_params"].values
    d = np.r_[0.0, np.diff(perf_df["LogLoss"].values)]  # first bar=0 for stability

    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    _despine(ax)
    _grid(ax)

    colors = [PUB["grey"]] + [PUB["blue"] if v < 0 else PUB["orange"] for v in d[1:]]
    ax.bar(x, d, color=colors, width=0.72, edgecolor="white", linewidth=0.6)
    ax.axhline(0, color=PUB["grey"], lw=1.0)

    ax.set_xlabel("Number of features")
    ax.set_ylabel(r"$\Delta$ LogLoss")
    return _savefig(fig, savepath)


def _model_color_from_label(label: str):
    s = str(label).lower()
    if "baseline" in s:
        return PUB["blue"]
    if "chosen" in s or "k*" in s:
        return PUB["orange"]
    if "best" in s or "k_best" in s:
        return PUB["black"]
    return PUB["grey"]


def plot_roc_pr_multi(y, pred_dict, outdir):
    os.makedirs(outdir, exist_ok=True)
    y = np.asarray(y, int)
    base_rate = float(y.mean())

    # ROC
    fig1, ax1 = plt.subplots(figsize=(7.0, 5.8))
    _despine(ax1)
    _grid(ax1)
    ax1.plot([0, 1], [0, 1], ls="--", lw=1.2, color=PUB["grey"])

    for label, p in pred_dict.items():
        p = _clip01(p)
        fpr, tpr, _ = roc_curve(y, p)
        aucv = roc_auc_score(y, p)
        ax1.plot(fpr, tpr, lw=2.3, color=_model_color_from_label(label),
                 label=f"{label} (AUC={aucv:.3f})")

    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    # ax1.set_title("Extended xG (3-model comparison) — ROC", weight="bold")
    ax1.legend(loc="lower right")
    _savefig(fig1, os.path.join(outdir, "roc_pub.png"))

    # PR
    fig2, ax2 = plt.subplots(figsize=(7.0, 5.8))
    _despine(ax2)
    _grid(ax2)
    ax2.hlines(base_rate, 0, 1, ls="--", lw=1.2, color=PUB["grey"])

    for label, p in pred_dict.items():
        p = _clip01(p)
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax2.plot(rec, prec, lw=2.3, color=_model_color_from_label(label),
                 label=f"{label} (AP={ap:.3f})")

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    # ax2.set_title("Extended xG (3-model comparison) — Precision–Recall", weight="bold")
    ax2.legend(loc="lower left")
    _savefig(fig2, os.path.join(outdir, "pr_pub.png"))


def plot_reliability_multi(y, pred_dict, outdir, n_bins=10):
    os.makedirs(outdir, exist_ok=True)
    y = np.asarray(y, int)

    pooled = np.concatenate([_clip01(p) for p in pred_dict.values()])
    q = np.quantile(pooled, np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    _despine(ax)
    _grid(ax)

    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=PUB["grey"])

    for label, p in pred_dict.items():
        p = _clip01(p)
        xs, ys = [], []
        for i in range(n_bins):
            lo, hi = q[i], q[i + 1]
            m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
            if m.sum() == 0:
                continue
            xs.append(float(p[m].mean()))
            ys.append(float(y[m].mean()))
        ax.plot(xs, ys, marker="o", ms=5.0, lw=2.3,
                color=_model_color_from_label(label), label=label)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    # ax.set_title("Reliability diagram (multi-model)", weight="bold")
    ax.legend(loc="lower right")
    _savefig(fig, os.path.join(outdir, "fig_reliability_pub.png"))


def plot_partials_pub(df_eval: pd.DataFrame, probs: np.ndarray, cols: List[str], savepath=None, n_bins=10):
    fig, axes = plt.subplots(1, len(cols), figsize=(5.3 * len(cols), 4.6), sharey=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        _despine(ax)
        _grid(ax)

        x = df_eval[c].astype(float).values
        q = np.quantile(x[~np.isnan(x)], np.linspace(0, 1, n_bins + 1))
        q[0] -= 1e-9;
        q[-1] += 1e-9

        xs, mu, se = [], [], []
        for i in range(n_bins):
            m = (x >= q[i]) & (x < q[i + 1])
            if m.sum() == 0:
                continue
            vals = probs[m]
            xs.append(np.nanmean(x[m]))
            mu.append(np.nanmean(vals))
            s = np.nanstd(vals, ddof=1) if m.sum() > 1 else 0.0
            se.append(s / max(1e-12, np.sqrt(m.sum())))

        xs = np.array(xs);
        mu = np.array(mu);
        se = np.array(se)
        ax.plot(xs, mu, lw=2.3, marker="o", ms=5.2, color=PUB["blue"])
        ax.fill_between(xs, mu - 1.96 * se, mu + 1.96 * se, color=PUB["blue"], alpha=0.18, linewidth=0)
        ax.set_xlabel(c)

    axes[0].set_ylabel("Predicted xG")
    # fig.suptitle("Partial effects (binned mean ± 1.96SE)", weight="bold")
    return _savefig(fig, savepath)


def export_xg_artifacts(df_full: pd.DataFrame, outdir: str, prob_col="xg_all_shot",
                        save_shots_csv=True,
                        save_geom_model=False, geom_model=None, geom_cols=None,
                        save_geom_grid=False):
    os.makedirs(outdir, exist_ok=True)

    if save_shots_csv:
        keep_cols = []
        for c in [
            "id", "matchId", "teamId", "playerId", "matchPeriod", "eventSec", "t_abs_s",
            "eventName", "subEventName", "shot_type", "is_regular_shot", "is_direct_free_kick",
            "is_penalty", "ax", "ay", "is_goal", prob_col,
            "feat_dist_m", "feat_angle_rad", "feat_lat_offset_m"
        ]:
            if c in df_full.columns:
                keep_cols.append(c)
        df_out = df_full[keep_cols].copy()
        shot_csv = os.path.join(outdir, f"shots_with_{prob_col}.csv")
        df_out.to_csv(shot_csv, index=False, encoding="utf-8-sig")

        meta = {
            "prob_col": prob_col,
            "n": int(len(df_out)),
            "goal_rate": float(df_out["is_goal"].mean()) if "is_goal" in df_out.columns else None,
            "cols_saved": keep_cols,
        }
        with open(os.path.join(outdir, f"shots_with_{prob_col}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    if save_geom_model:
        if joblib is None:
            raise RuntimeError("joblib not available. Please install or remove save_geom_model=True.")
        assert geom_model is not None and geom_cols is not None
        joblib.dump({"model": geom_model, "geom_cols": list(geom_cols)},
                    os.path.join(outdir, "geom_model.joblib"))

    if save_geom_grid:
        assert geom_model is not None and geom_cols is not None
        XX, YY, Z = make_geom_grid_predictions(geom_model, geom_cols)
        np.savez_compressed(
            os.path.join(outdir, "xg_surface_geom_grid.npz"),
            XX=XX, YY=YY, Z=Z, geom_cols=np.array(list(geom_cols), dtype=object)
        )


def main():
    set_pub_style(base_fontsize=11, font_family="DejaVu Sans", dpi=240)

    print("[1] Loading data...")
    matches = pd.read_excel(matches_path)
    playersrank = pd.read_csv(playersrank_path)
    players = pd.read_csv(players_path)
    teams = pd.read_csv(teams_path)
    events = pd.read_csv(events_path)

    print("[2] Build shots...")
    shots = build_shots(matches, players, events)
    shots = stabilize_prev_features(shots)
    shots = add_situ_features_from_tags(shots)

    print("[OK] shots:", shots.shape,
          "| regular=", int(shots.get("is_regular_shot", pd.Series(dtype=int)).sum()),
          "| direct_fk=", int(shots.get("is_direct_free_kick", pd.Series(dtype=int)).sum()),
          "| penalties=", int(shots.get("is_penalty", pd.Series(dtype=int)).sum()),
          "| goal_rate=", float(shots["is_goal"].mean()),
          "| on_target_rate=", float(shots["is_on_target"].mean()) if "is_on_target" in shots.columns else float("nan"))

    shots.to_csv(os.path.join(outdir, "shots_features.csv"), index=False, encoding="utf-8-sig")

    feats_order = feature_order_25(use_identity_oof=DO_OOF_IDENTITY)

    cols_geom_min = [
        c for c in [
            "feat_dist_m", "feat_angle_rad", "feat_lat_offset_m",
            "feat_inv_dist", "feat_cos_angle", "feat_angle_x_dist"
        ]
        if c in shots.columns and shots[c].notna().sum() > 0 and shots[c].nunique(dropna=True) > 1
    ]
    if len(cols_geom_min) == 0:
        raise RuntimeError("cols_geom_min is empty. Check geometry features in shots.")

    usable_full = []
    for c in feats_order:
        if (not DO_OOF_IDENTITY) and str(c).startswith("enc_"):
            continue
        if c in shots.columns and shots[c].notna().sum() > 0 and shots[c].nunique(dropna=True) > 1:
            usable_full.append(c)

    print(f"[INFO] usable_full={len(usable_full)} | cols_geom_min={len(cols_geom_min)}")

    print("[3] Fit regular-shot geometry-only surface model...")
    geom_train = shots[shots["shot_type"] == SHOT_TYPE_REGULAR].copy() if "shot_type" in shots.columns else shots
    geom_model = fit_geom_model(geom_train, cols_geom_min, target_col="is_goal", calibrate=DO_CALIBRATION)

    print("[4] OOF eval: FULL_ALL_SHOT vs GEOM_MIN_ALL_SHOT (calibrate=False/True)")
    rows = []
    for cal in [False, True]:
        p_full_tmp, y_full_tmp, m_full_tmp, _ = oof_predict_and_metrics(
            shots, usable_full, "is_goal", calibrate=cal, n_splits=N_SPLITS, seed=SEED
        )
        m_full_tmp.update({"model": "FULL_ALL_SHOT", "calibrate": cal, "n_features": len(usable_full)})
        rows.append(m_full_tmp)

        p_min_tmp, y_min_tmp, m_min_tmp, _ = oof_predict_and_metrics(
            shots, cols_geom_min, "is_goal", calibrate=cal, n_splits=N_SPLITS, seed=SEED
        )
        m_min_tmp.update({"model": "GEOM_MIN_ALL_SHOT", "calibrate": cal, "n_features": len(cols_geom_min)})
        rows.append(m_min_tmp)

    eval_table = pd.DataFrame(rows)
    eval_table.to_csv(os.path.join(outdir, "eval_table.csv"), index=False, encoding="utf-8-sig")
    print(eval_table)

    print("[5] Compute perf curve...")
    perf_df = compute_perf_curve(
        shots=shots,
        feats_order=feats_order,
        target_col="is_goal",
        calibrate=DO_CALIBRATION,
        n_splits=N_SPLITS,
        seed=SEED,
        allow_identity_oof=DO_OOF_IDENTITY
    )
    perf_df.to_csv(os.path.join(outdir, "perf_curve.csv"), index=False, encoding="utf-8-sig")

    k_star, k_best, best_mean, best_se = one_se_rule_on_logloss(perf_df)
    print(f"[OK] 1-SE: k*={k_star} | k_best={k_best} | best={best_mean:.6f} ± {best_se:.6f}")

    plot_figure3_elegant(perf_df, k_star=k_star, k_best=k_best,
                         savepath=os.path.join(outdir, "fig_perf_vs_features_pub.png"))
    plot_delta_logloss_pub(perf_df, savepath=os.path.join(outdir, "fig_delta_logloss_pub.png"))
    metrics_3 = compute_3model_metrics(
        shots=shots,
        perf_df=perf_df,
        k_star=k_star,
        k_best=k_best,
        target_col="is_goal",
        calibrate=DO_CALIBRATION,
        n_splits=N_SPLITS,
        seed=SEED
    )

    print("[3-model OOF metrics]")
    print(metrics_3)

    metrics_3.to_csv(os.path.join(outdir, "metrics_3models.csv"), index=False, encoding="utf-8-sig")

    print("[6] OOF predict FULL (final) for all-shot xG...")
    p_full, y_full, m_full, df_full = oof_predict_and_metrics(
        shots, usable_full, "is_goal",
        calibrate=DO_CALIBRATION, n_splits=N_SPLITS, seed=SEED
    )
    df_full = df_full.copy()
    df_full["xg_all_shot"] = p_full

    export_xg_artifacts(
        df_full=df_full,
        outdir=outdir,
        prob_col="xg_all_shot",
        save_shots_csv=True,
        save_geom_model=True,  # keep
        geom_model=geom_model,
        geom_cols=cols_geom_min,
        save_geom_grid=True
    )

    print("[7] Plot xG figures...")
    plot_xg_shot_map(df_full, "xg_all_shot",
                     savepath=os.path.join(outdir, "fig_xg_shotmap_all_shot.png"))
    plot_xg_surface(geom_model, cols_geom_min,
                    savepath=os.path.join(outdir, "fig_xg_surface_geom.png"))

    print("[8] Plot 3-model comparison (ROC/PR/Reliability)...")
    usable_order = perf_df["added"].tolist()

    cols_base = usable_order[:1]
    cols_star = usable_order[:k_star]
    cols_best = usable_order[:k_best]

    shots_common = _clean_for_composite(shots, cols_best, target_col="is_goal")

    p_base, y_c, _, _ = oof_predict_and_metrics(
        shots_common, cols_base, "is_goal",
        calibrate=DO_CALIBRATION, n_splits=N_SPLITS, seed=SEED
    )
    p_star, y_c2, _, _ = oof_predict_and_metrics(
        shots_common, cols_star, "is_goal",
        calibrate=DO_CALIBRATION, n_splits=N_SPLITS, seed=SEED
    )
    p_best, y_c3, _, _ = oof_predict_and_metrics(
        shots_common, cols_best, "is_goal",
        calibrate=DO_CALIBRATION, n_splits=N_SPLITS, seed=SEED
    )
    assert np.array_equal(y_c, y_c2) and np.array_equal(y_c, y_c3), "Mismatch in evaluation sets!"

    pred_dict = {
        "Baseline(1)": p_base,
        f"Chosen(k*={k_star})": p_star,
        f"Best(k_best={k_best})": p_best
    }

    plot_roc_pr_multi(y_c, pred_dict, outdir)  # saves roc_pub.png / pr_pub.png
    plot_reliability_multi(y_c, pred_dict, outdir, n_bins=10)  # saves fig_reliability_pub.png

    print("[9] Plot partial effects (based on full all-shot xG)...")
    cand = ["feat_dist_m", "feat_angle_rad", "feat_prev_speed_mps", "feat_prev_lateral_m", "feat_score_diff_before"]
    plot_cols = [c for c in cand if c in df_full.columns and df_full[c].notna().sum() > 50][:3]
    if len(plot_cols) > 0:
        dfp = df_full.dropna(subset=plot_cols).copy()
        plot_partials_pub(dfp, probs=dfp["xg_all_shot"].values, cols=plot_cols,
                          savepath=os.path.join(outdir, "fig_partials_pub.png"))

    print("[DONE] Outputs saved to:", outdir)


if __name__ == "__main__":
    main()
