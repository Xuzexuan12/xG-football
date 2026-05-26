# -*- coding: utf-8 -*-


import os, re, ast, json, math, warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

matches_path      = "./data/worldcup_64_match.xlsx"
playersrank_path  = "./data/worldcup_64_playersrank.csv"  # not used, kept
players_path      = "./data/worldcup_64_players.csv"
teams_path        = "./data/worldcup_64_teams.csv"        # not used, kept
events_path       = "./data/worldcup_64_events.csv"

outdir            = "./SCI-1/Final_results"
os.makedirs(outdir, exist_ok=True)

DO_OOF_IDENTITY   = False

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

PERIOD_OFFSETS = {"1H": 0, "2H": 45*60, "E1": 90*60, "E2": 105*60}

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
    tags = [str(t) for t in (tags if isinstance(tags, (list,tuple,set)) else parse_list(tags))]
    return int(("Goal" in set(tags)) or ("Own goal" in set(tags)))

def norm_token(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(t).lower())

def has_any(tags: List[str], candidates: List[str]) -> int:
    s = {norm_token(x) for x in tags}
    return int(any(norm_token(c) in s for c in candidates))

def add_situ_features_from_tags(shots: pd.DataFrame) -> pd.DataFrame:
    s = shots.copy()
    s["__tags__"] = s["__tags__"].apply(parse_list)
    s["feat_is_counter"]   = s["__tags__"].apply(lambda L: has_any(L, ["Counter attack", "Counterattack"]))
    s["feat_is_cross"]     = s["__tags__"].apply(lambda L: has_any(L, ["Cross", "Crossing"]))
    s["feat_is_penalty"]   = s["__tags__"].apply(lambda L: has_any(L, ["Penalty"]))
    s["feat_is_corner"]    = s["__tags__"].apply(lambda L: has_any(L, ["Corner"]))
    s["feat_is_free_kick"] = s["__tags__"].apply(lambda L: has_any(L, ["Free kick","Freekick","Direct","Indirect"]))
    s["feat_sub_penalty"]  = s["feat_is_penalty"]
    s["feat_sub_corner"]   = s["feat_is_corner"]
    s["feat_sub_freekick"] = s["feat_is_free_kick"]
    return s

def infer_team_attacking_right_per_period(events: pd.DataFrame, matches: pd.DataFrame) -> Dict[Tuple[int,int,str], bool]:
    mm = matches[["wyId", "homeId", "awayId"]].rename(columns={"wyId":"matchId"})
    ev = events[["matchId","matchPeriod","teamId","eventName","x_start"]].copy().merge(mm, on="matchId", how="left")
    ev = ev[ev["matchPeriod"].isin(["1H","2H","E1","E2"])].dropna(subset=["x_start"])

    shot_med = (ev[ev["eventName"].astype(str).str.lower()=="shot"]
                .groupby(["matchId","teamId","matchPeriod"])["x_start"].median())
    all_med  = ev.groupby(["matchId","teamId","matchPeriod"])["x_start"].median()

    dir_map: Dict[Tuple[int,int,str], bool] = {}
    for key, xmed in shot_med.items():
        dir_map[(int(key[0]), int(key[1]), str(key[2]))] = bool(float(xmed) > 50.0)
    for key, xmed in all_med.items():
        k = (int(key[0]), int(key[1]), str(key[2]))
        if k not in dir_map:
            dir_map[k] = bool(float(xmed) > 50.0)

    period_prev = {"2H":"1H","E1":"2H","E2":"E1"}
    for mid, grp in ev.groupby("matchId"):
        teams = set(grp["teamId"].unique().tolist())
        for per in ["1H","2H","E1","E2"]:
            for tid in teams:
                k = (int(mid), int(tid), per)
                if k in dir_map:
                    continue
                # opponent opposite
                opp_found = False
                for opp in (teams - {tid}):
                    kk = (int(mid), int(opp), per)
                    if kk in dir_map:
                        dir_map[k] = (not dir_map[kk]); opp_found=True; break
                if opp_found:
                    continue
                # inherit previous period
                prev = period_prev.get(per)
                if prev and (int(mid), int(tid), prev) in dir_map:
                    dir_map[k] = dir_map[(int(mid), int(tid), prev)]
                else:
                    dir_map[k] = True
    return dir_map

def build_prev_event_context(events: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    team_dir_map = infer_team_attacking_right_per_period(events, matches)
    ev = events.copy()
    ev = ev[ev["matchPeriod"].isin(["1H","2H","E1","E2"])].copy()

    ev["period_offset_s"] = ev["matchPeriod"].map(PERIOD_OFFSETS).fillna(0).astype(int)
    ev["t_abs_s"] = ev["period_offset_s"] + pd.to_numeric(ev["eventSec"], errors="coerce").fillna(0.0)

    ev["att_right"] = ev.apply(
        lambda r: team_dir_map.get((int(r["matchId"]), int(r["teamId"]), str(r["matchPeriod"])), True),
        axis=1
    ).astype(bool)

    norm = ev[["x_start","y_start","att_right"]].apply(
        lambda r: to_attacking_frame(r["x_start"], r["y_start"], bool(r["att_right"])),
        axis=1
    )
    ev["ax"] = [t[0] for t in norm]
    ev["ay"] = [t[1] for t in norm]

    ev = ev.sort_values(["matchId","teamId","t_abs_s","id"])
    ev["prev_t"]  = ev.groupby(["matchId","teamId"])["t_abs_s"].shift(1)
    ev["prev_ax"] = ev.groupby(["matchId","teamId"])["ax"].shift(1)
    ev["prev_ay"] = ev.groupby(["matchId","teamId"])["ay"].shift(1)

    ev["tags_list"] = ev["tag_descriptions"].apply(parse_list)
    ev["prev_tags"] = ev.groupby(["matchId","teamId"])["tags_list"].shift(1)

    def _has(prev, token):
        if isinstance(prev,(list,tuple,set)):
            return int(token in set(prev))
        return 0

    ev["prev_through"]     = ev["prev_tags"].apply(lambda t: _has(t, "Through"))
    ev["prev_direct_fk"]   = ev["prev_tags"].apply(lambda t: _has(t, "Direct"))
    ev["prev_indirect_fk"] = ev["prev_tags"].apply(lambda t: _has(t, "Indirect"))

    keep = ["id","matchId","teamId","matchPeriod","prev_t","prev_ax","prev_ay",
            "prev_through","prev_direct_fk","prev_indirect_fk"]
    for c in ["id","matchId","teamId"]:
        ev[c] = pd.to_numeric(ev[c], errors="coerce").astype("Int64")
    return ev[keep].copy()

def build_shots(matches: pd.DataFrame, players: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    evctx = build_prev_event_context(events, matches)

    df = events.copy()
    df["eventName_low"] = df["eventName"].astype(str).str.lower()
    df = df[df["matchPeriod"].isin(["1H","2H","E1","E2"])].copy()
    df["__tags__"] = df["tag_descriptions"].apply(parse_list)

    shots = df[df["eventName_low"]=="shot"].copy()

    matches_min = matches[["wyId","homeId","awayId"]].rename(columns={"wyId":"matchId"})
    shots = shots.merge(matches_min, on="matchId", how="left")

    shots["is_home_team"] = (shots["teamId"] == shots["homeId"]).astype(int)
    shots["opponentId"] = np.where(shots["is_home_team"]==1, shots["awayId"], shots["homeId"])

    shots["period_offset_s"] = shots["matchPeriod"].map(PERIOD_OFFSETS).fillna(0).astype(int)
    shots["t_abs_s"] = shots["period_offset_s"] + pd.to_numeric(shots["eventSec"], errors="coerce").fillna(0.0)

    team_dir_map = infer_team_attacking_right_per_period(events, matches)
    shots["att_right"] = shots.apply(
        lambda r: team_dir_map.get((int(r["matchId"]), int(r["teamId"]), str(r["matchPeriod"])), True),
        axis=1
    ).astype(bool)

    norm = shots[["x_start","y_start","att_right"]].apply(
        lambda r: to_attacking_frame(r["x_start"], r["y_start"], bool(r["att_right"])),
        axis=1
    )
    shots["ax"] = [t[0] for t in norm]
    shots["ay"] = [t[1] for t in norm]

    geom = shots[["ax","ay"]].apply(lambda r: shot_geometry_features(r["ax"], r["ay"]), axis=1)
    shots = pd.concat([shots, pd.DataFrame(list(geom))], axis=1)

    # derived geometry terms
    shots["feat_minute"] = shots["t_abs_s"] / 60.0
    shots["feat_angle_deg"] = np.degrees(shots["feat_angle_rad"])
    shots["feat_cos_angle"] = np.cos(shots["feat_angle_rad"])
    shots["feat_inv_dist"]  = 1.0 / (1.0 + shots["feat_dist_m"])
    shots["feat_angle_x_dist"] = shots["feat_angle_rad"] * shots["feat_dist_m"]

    # interpretable tags
    tag_flags_df = pd.DataFrame(list(shots["__tags__"].apply(tags_to_interpretable_flags)))
    shots = pd.concat([shots, tag_flags_df], axis=1)

    for c in ["id","matchId","teamId"]:
        shots[c] = pd.to_numeric(shots[c], errors="coerce").astype("Int64")
    shots = shots.merge(evctx, on=["id","matchId","teamId","matchPeriod"], how="left")

    # prev-delta features
    XU2M, YU2M = X_UNIT_TO_METERS, Y_UNIT_TO_METERS
    shots["feat_prev_dt"] = (shots["t_abs_s"] - shots["prev_t"]).fillna(999.0)
    shots["feat_prev_dx_m"] = (shots["ax"] - shots["prev_ax"]) * XU2M
    shots["feat_prev_dy_m"] = (shots["ay"] - shots["prev_ay"]) * YU2M
    shots["feat_prev_dist_m"] = np.hypot(shots["feat_prev_dx_m"], shots["feat_prev_dy_m"])
    shots["feat_prev_speed_mps"] = shots["feat_prev_dist_m"] / shots["feat_prev_dt"].replace(0, np.nan)
    shots["feat_prev_speed_mps"] = shots["feat_prev_speed_mps"].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    shots["feat_prev_lateral_m"] = shots["feat_prev_dy_m"].abs()

    shots["feat_prev_through"]     = shots["prev_through"].fillna(0).astype(int)
    shots["feat_prev_direct_fk"]   = shots["prev_direct_fk"].fillna(0).astype(int)
    shots["feat_prev_indirect_fk"] = shots["prev_indirect_fk"].fillna(0).astype(int)

    # player foot + weak foot
    if "wyId" in players.columns and "foot" in players.columns:
        pmin = players[["wyId","foot"]].rename(columns={"wyId":"playerId","foot":"player_foot"})
        shots = shots.merge(pmin, on="playerId", how="left")
    else:
        shots["player_foot"] = np.nan

    def _weak_foot(row):
        pf = str(row.get("player_foot","")).lower()
        if row.get("feat_tag_head",0)==1:
            return 0
        if row.get("feat_tag_left_foot",0)==1:
            return int(pf=="right")
        if row.get("feat_tag_right_foot",0)==1:
            return int(pf=="left")
        return 0

    shots["feat_weak_foot"] = shots.apply(_weak_foot, axis=1)

    # strict label
    shots["is_goal"] = shots["__tags__"].apply(label_is_goal).astype(int)

    # score diff BEFORE
    shots = shots.sort_values(["matchId","teamId","t_abs_s","id"]).reset_index(drop=True)
    shots["goal_for"] = shots["is_goal"].astype(int)
    shots["gf_before"] = shots.groupby(["matchId","teamId"])["goal_for"].cumsum().shift(1, fill_value=0)

    opp_goal_times: Dict[Tuple[int,int], np.ndarray] = {}
    gshots = shots[shots["is_goal"]==1][["matchId","teamId","t_abs_s"]].copy()
    for (mid, tid), grp in gshots.groupby(["matchId","teamId"]):
        opp_goal_times[(int(mid), int(tid))] = np.sort(grp["t_abs_s"].astype(float).values)

    def _count_opp_goals_before(mid, oppid, t):
        arr = opp_goal_times.get((int(mid), int(oppid)))
        if arr is None or arr.size==0:
            return 0
        return int(np.searchsorted(arr, float(t), side="left"))

    shots["ga_before"] = shots.apply(
        lambda r: _count_opp_goals_before(r["matchId"], r["opponentId"], r["t_abs_s"]),
        axis=1
    ).astype(int)

    shots["feat_score_diff_before"] = shots["gf_before"] - shots["ga_before"]

    shots = shots.dropna(subset=["feat_dist_m","feat_angle_rad"]).copy()
    return shots

def stabilize_prev_features(shots: pd.DataFrame) -> pd.DataFrame:
    s = shots.copy()
    if "feat_prev_exists" not in s.columns and "feat_prev_dt" in s.columns:
        s["feat_prev_exists"] = (s["feat_prev_dt"] < 990.0).astype(int)
    for c in ["feat_prev_dx_m","feat_prev_dy_m","feat_prev_dist_m","feat_prev_lateral_m"]:
        if c in s.columns:
            s[c] = s[c].fillna(0.0)
    for c in ["feat_prev_through","feat_prev_direct_fk","feat_prev_indirect_fk"]:
        if c in s.columns:
            s[c] = s[c].fillna(0).astype(int)
    return s

def feature_order_25(use_identity_oof=False) -> List[str]:
    order = [
        "feat_dist_m","feat_angle_rad","feat_inv_dist","feat_cos_angle","feat_angle_x_dist",
        "feat_x_m","feat_y_m","feat_lat_offset_m",
        "feat_minute","feat_score_diff_before",
        "feat_prev_exists","feat_prev_dt","feat_prev_dist_m","feat_prev_speed_mps",
        "feat_prev_lateral_m","feat_prev_dx_m","feat_prev_dy_m",
        "feat_tag_left_foot","feat_tag_right_foot","feat_tag_head","feat_weak_foot",
        "feat_prev_through","feat_prev_direct_fk","feat_prev_indirect_fk",
        "feat_is_penalty","feat_is_corner","feat_is_free_kick","feat_is_counter","feat_is_cross",
    ]
    if use_identity_oof:
        order += ["enc_teamId_oof","enc_playerId_oof","enc_opponentId_oof"]
    return order

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
        "axes.grid": True,
        "grid.alpha": 0.22,
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

def _savefig(fig, savepath):
    fig.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches="tight")
    return fig

def _clean_pos(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    return x

def gamma_mom_fit(x: np.ndarray):
    """Gamma(k, theta) via method-of-moments; k=shape, theta=scale."""
    x = _clean_pos(x)
    if x.size < 5:
        return np.nan, np.nan
    m = float(x.mean())
    v = float(x.var(ddof=1))
    if m <= 0 or v <= 0:
        return np.nan, np.nan
    k = (m*m) / v
    theta = v / m
    return float(k), float(theta)

def gamma_pdf(x: np.ndarray, k: float, theta: float):
    """PDF of Gamma(k, theta). Stable using log form with lgamma."""
    x = np.asarray(x, float)
    eps = 1e-12
    x = np.maximum(x, eps)
    if not (np.isfinite(k) and np.isfinite(theta) and k > 0 and theta > 0):
        return np.full_like(x, np.nan, dtype=float)
    lg = math.lgamma(k)
    logp = (k - 1.0) * np.log(x) - (x / theta) - k * np.log(theta) - lg
    return np.exp(logp)

def beta_mom_fit(u: np.ndarray):
    """Beta(a,b) on [0,1] via method-of-moments."""
    u = np.asarray(u, float)
    u = u[np.isfinite(u)]
    u = np.clip(u, 1e-6, 1-1e-6)
    if u.size < 5:
        return np.nan, np.nan
    m = float(u.mean())
    v = float(u.var(ddof=1))
    vmax = m * (1.0 - m) - 1e-9
    if v <= 0:
        v = 1e-6
    if v >= vmax:
        v = 0.9 * vmax
    t = (m * (1.0 - m) / v) - 1.0
    a = m * t
    b = (1.0 - m) * t
    if a <= 0 or b <= 0:
        return np.nan, np.nan
    return float(a), float(b)

def beta_pdf(u: np.ndarray, a: float, b: float):
    """PDF of Beta(a,b) on [0,1] using log-gamma."""
    u = np.asarray(u, float)
    u = np.clip(u, 1e-12, 1-1e-12)
    if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        return np.full_like(u, np.nan, dtype=float)
    logB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    logp = (a - 1.0) * np.log(u) + (b - 1.0) * np.log(1.0 - u) - logB
    return np.exp(logp)

def plot_dist_angle_with_fits(shots_eval: pd.DataFrame, savepath: str,
                              bins_dist=28, bins_angle=28,
                              title="Geometry distributions with fitted densities"):
    s = shots_eval.copy()
    s["angle_deg"] = np.degrees(s["feat_angle_rad"].astype(float))

    m_goal = s["is_goal"].astype(int).values == 1

    dist_all = s["feat_dist_m"].astype(float).values
    dist_goal = s.loc[m_goal, "feat_dist_m"].astype(float).values

    ang_all = s["angle_deg"].astype(float).values
    ang_goal = s.loc[m_goal, "angle_deg"].astype(float).values

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for ax in axes:
        _despine(ax)

    # Distance (Gamma fit)
    ax = axes[0]
    da = dist_all[np.isfinite(dist_all)]
    dg = dist_goal[np.isfinite(dist_goal)]

    ax.hist(da, bins=bins_dist, density=True,
            color=PUB["blue"], alpha=0.28, edgecolor="white", linewidth=0.6, label="All shots")
    ax.hist(dg, bins=bins_dist, density=True,
            color=PUB["orange"], alpha=0.48, edgecolor="white", linewidth=0.6, label="Goals")

    # Gamma fits (MoM) for all & goals
    k_a, th_a = gamma_mom_fit(da)
    k_g, th_g = gamma_mom_fit(dg)

    xgrid = np.linspace(max(1e-6, float(np.nanmin(da))), float(np.nanmax(da)), 320)
    ax.plot(xgrid, gamma_pdf(xgrid, k_a, th_a), color=PUB["blue"], lw=2.4,
            label=f"Gamma fit (all): k={k_a:.2f}, θ={th_a:.2f}")
    ax.plot(xgrid, gamma_pdf(xgrid, k_g, th_g), color=PUB["orange"], lw=2.4, ls="--",
            label=f"Gamma fit (goal): k={k_g:.2f}, θ={th_g:.2f}")

    ax.set_xlabel("Shot distance (m)")
    ax.set_ylabel("Density")
    # ax.set_title("Distance", weight="bold")
    ax.legend(loc="upper right")

    #Angle (Beta-on-[0,max] fit)
    ax = axes[1]
    aa = ang_all[np.isfinite(ang_all)]
    ag = ang_goal[np.isfinite(ang_goal)]

    ax.hist(aa, bins=bins_angle, density=True,
            color=PUB["blue"], alpha=0.28, edgecolor="white", linewidth=0.6, label="All shots")
    ax.hist(ag, bins=bins_angle, density=True,
            color=PUB["orange"], alpha=0.48, edgecolor="white", linewidth=0.6, label="Goals")

    # Fit Beta on [0, Amax] by scaling u=angle/Amax
    Amax = float(np.nanmax(aa)) if aa.size else 1.0
    Amax = max(Amax, 1e-6)

    ua = np.clip(aa / Amax, 1e-6, 1-1e-6)
    ug = np.clip(ag / Amax, 1e-6, 1-1e-6)

    a_a, b_a = beta_mom_fit(ua)
    a_g, b_g = beta_mom_fit(ug)

    xgrid2 = np.linspace(0.0, Amax, 360)
    ugrid2 = np.clip(xgrid2 / Amax, 1e-6, 1-1e-6)

    # transform pdf_u to pdf_x by /Amax
    ax.plot(xgrid2, beta_pdf(ugrid2, a_a, b_a) / Amax, color=PUB["blue"], lw=2.4,
            label=f"Beta fit (all): α={a_a:.2f}, β={b_a:.2f}")
    ax.plot(xgrid2, beta_pdf(ugrid2, a_g, b_g) / Amax, color=PUB["orange"], lw=2.4, ls="--",
            label=f"Beta fit (goal): α={a_g:.2f}, β={b_g:.2f}")

    ax.set_xlabel("Shot angle (deg)")
    ax.set_ylabel("Density")
    # ax.set_title("Angle", weight="bold")
    ax.legend(loc="upper right")

    n_eval = int(len(s))
    goal_rate = float(s["is_goal"].mean()) if n_eval else float("nan")
    # fig.suptitle(f"{title}  (n={n_eval}, goal rate={goal_rate:.4f})", weight="bold")

    return _savefig(fig, savepath)

def main():
    set_pub_style(base_fontsize=11, font_family="DejaVu Sans", dpi=240)

    print("[1] Loading data...")
    matches     = pd.read_excel(matches_path)
    _           = pd.read_csv(playersrank_path)
    players     = pd.read_csv(players_path)
    _           = pd.read_csv(teams_path)
    events      = pd.read_csv(events_path)

    print("[2] Build shots...")
    shots = build_shots(matches, players, events)
    shots = stabilize_prev_features(shots)
    shots = add_situ_features_from_tags(shots)

    # Build usable_full list
    feats_order = feature_order_25(use_identity_oof=DO_OOF_IDENTITY)
    usable_full = []
    for c in feats_order:
        if (not DO_OOF_IDENTITY) and str(c).startswith("enc_"):
            continue
        if c in shots.columns and shots[c].notna().sum() > 0 and shots[c].nunique(dropna=True) > 1:
            usable_full.append(c)

    mask_eval = shots[usable_full].notna().all(axis=1)
    shots_eval = shots.loc[mask_eval].copy()

    outpath = os.path.join(outdir, "fig_data_dist_angle.png")
    print("[3] Plot fig_data_dist_angle (with fitted distributions)...")
    plot_dist_angle_with_fits(
        shots_eval,
        savepath=outpath,
        bins_dist=28,
        bins_angle=28
        # title="Basic geometry distributions (all shots vs goals) with fitted densities"
    )

    print("[DONE] Saved:", outpath)
    print("[INFO] n_eval =", int(len(shots_eval)), "| goal_rate =", float(shots_eval["is_goal"].mean()))

if __name__ == "__main__":
    main()
