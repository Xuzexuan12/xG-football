

import os
import argparse
import numpy as np
import pandas as pd

PERIOD_OFFSETS = {"1H": 0, "2H": 45*60, "E1": 90*60, "E2": 105*60}


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def load_player_name_map(players_path: str) -> dict:
    df = _safe_read_csv(players_path)

    # player id column
    id_col = None
    for c in ["wyId", "playerId", "id"]:
        if c in df.columns:
            id_col = c
            break

    if "shortName" in df.columns:
        df["_name_"] = df["shortName"].astype(str)
    elif "name" in df.columns:
        df["_name_"] = df["name"].astype(str)
    elif ("firstName" in df.columns) or ("lastName" in df.columns):
        fn = df["firstName"].fillna("").astype(str) if "firstName" in df.columns else ""
        ln = df["lastName"].fillna("").astype(str) if "lastName" in df.columns else ""
        df["_name_"] = (fn + " " + ln).str.strip()
    else:
        df["_name_"] = df[id_col].astype(str)

    m = dict(zip(pd.to_numeric(df[id_col], errors="coerce"), df["_name_"]))
    return m


def add_abs_time(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()
    ev = ev[ev["matchPeriod"].isin(PERIOD_OFFSETS.keys())].copy()
    ev["period_offset_s"] = ev["matchPeriod"].map(PERIOD_OFFSETS).fillna(0).astype(float)
    ev["eventSec_num"] = pd.to_numeric(ev["eventSec"], errors="coerce").fillna(0.0)
    ev["t_abs_s"] = ev["period_offset_s"] + ev["eventSec_num"]
    return ev


def estimate_apps_mins_from_events(events: pd.DataFrame,
                                  start_buffer_s: float = 5 * 60,
                                  end_buffer_s: float = 5 * 60) -> pd.DataFrame:
    ev = add_abs_time(events)

    # identify playerId column
    pid_col = None
    for c in ["playerId", "player_id", "wyIdPlayer"]:
        if c in ev.columns:
            pid_col = c
            break

    # match end times
    match_end = ev.groupby("matchId")["t_abs_s"].max().rename("match_end_s").reset_index()

    # player min/max per match
    pm = (ev.dropna(subset=[pid_col])
            .groupby(["matchId", pid_col])["t_abs_s"]
            .agg(["min", "max"])
            .reset_index()
            .rename(columns={pid_col: "playerId", "min": "min_t", "max": "max_t"}))

    pm = pm.merge(match_end, on="matchId", how="left")

    # heuristic on/off times
    pm["on_s"] = np.where(pm["min_t"] <= start_buffer_s, 0.0, pm["min_t"])
    pm["off_s"] = np.where(pm["max_t"] >= (pm["match_end_s"] - end_buffer_s), pm["match_end_s"], pm["max_t"])

    pm["mins"] = (pm["off_s"] - pm["on_s"]) / 60.0
    pm["mins"] = pm["mins"].clip(lower=0.0, upper=130.0)

    # APPS: count matches with mins>0
    apps = pm.groupby("playerId")["matchId"].nunique().rename("APPS")
    mins = pm.groupby("playerId")["mins"].sum().rename("MINS")

    out = pd.concat([apps, mins], axis=1).reset_index()
    return out


def build_player_shot_table(shots_with_xg: pd.DataFrame,
                            shots_features: pd.DataFrame,
                            prob_col: str = "xg_all_shot") -> pd.DataFrame:
    s = shots_with_xg.copy()

    # basic checks
    need = ["playerId", "matchId", "is_goal", prob_col]
    miss = [c for c in need if c not in s.columns]

    sot_col = None
    for c in ["is_on_target", "isOnTarget"]:
        if c in shots_features.columns:
            sot_col = c
            break

    if sot_col is not None:
        keys = []
        for k in ["id", "matchId", "teamId", "playerId"]:
            if k in s.columns and k in shots_features.columns:
                keys.append(k)

        if len(keys) >= 2:
            sf = shots_features[keys + [sot_col]].copy()
            sf = sf.drop_duplicates(subset=keys)
            s = s.merge(sf, on=keys, how="left")
            s.rename(columns={sot_col: "is_on_target"}, inplace=True)
        else:
            s["is_on_target"] = np.nan
    else:
        s["is_on_target"] = np.nan

    # aggregate per player
    g = s.groupby("playerId", dropna=False)

    out = pd.DataFrame({
        "SHOTS": g.size(),
        "GOALS": g["is_goal"].sum(),
        "XG": g[prob_col].sum(),
        "SOT": g["is_on_target"].sum(min_count=1),  # if all NaN -> NaN
    }).reset_index()

    # derived metrics
    out["GOALS VS XG"] = out["GOALS"] - out["XG"]
    out["CONV %"] = np.where(out["SHOTS"] > 0, 100.0 * out["GOALS"] / out["SHOTS"], 0.0)
    out["XG PER SHOT"] = np.where(out["SHOTS"] > 0, out["XG"] / out["SHOTS"], 0.0)

    return out


def main():
    outdir = "./Final_results"
    events_path = "./data/data/worldcup_64_events.csv"
    players_path = "./data/data/worldcup_64_players.csv"
    prob_col = "xg_all_shot"
    shots_xg_path = os.path.join(outdir, f"shots_with_{prob_col}.csv")
    shots_feat_path = os.path.join(outdir, "shots_features.csv")
    out_csv = os.path.join(outdir, "player_xg_summary.csv")
    topn = 20
    print("[1] Loading artifacts...")
    shots_with_xg = _safe_read_csv(shots_xg_path)
    shots_features = _safe_read_csv(shots_feat_path)
    events = _safe_read_csv(events_path)

    print("[2] Player name mapping...")
    name_map = load_player_name_map(players_path)

    print("[3] Estimate APPS & MINS from events (heuristic)...")
    apps_mins = estimate_apps_mins_from_events(events)

    print("[4] Aggregate player shot stats...")
    player_shots = build_player_shot_table(shots_with_xg, shots_features, prob_col=prob_col)

    print("[5] Merge & finalize table...")
    df = player_shots.merge(apps_mins, on="playerId", how="left")
    df["PLAYER"] = df["playerId"].map(name_map).fillna(df["playerId"].astype(str))

    df = df[[
        "PLAYER",
        "APPS",
        "MINS",
        "GOALS",
        "XG",
        "GOALS VS XG",
        "SHOTS",
        "SOT",
        "CONV %",
        "XG PER SHOT",
        "playerId"
    ]]

    df["APPS"] = df["APPS"].fillna(0).astype(int)
    df["MINS"] = df["MINS"].fillna(0.0).astype(float)
    df["GOALS"] = df["GOALS"].fillna(0).astype(int)
    df["SHOTS"] = df["SHOTS"].fillna(0).astype(int)


    df["XG"] = df["XG"].fillna(0.0).astype(float)
    df["GOALS VS XG"] = df["GOALS VS XG"].fillna(0.0).astype(float)
    df["CONV %"] = df["CONV %"].fillna(0.0).astype(float)
    df["XG PER SHOT"] = df["XG PER SHOT"].fillna(0.0).astype(float)

    # sort
    df = df.sort_values(["XG", "GOALS", "SHOTS"], ascending=[False, False, False]).reset_index(drop=True)

    # save
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # print top N
    topn = max(1, int(topn))
    show = df.head(topn).copy()

    # pretty display
    show["MINS"] = show["MINS"].map(lambda x: f"{x:.1f}")
    show["XG"] = show["XG"].map(lambda x: f"{x:.3f}")
    show["GOALS VS XG"] = show["GOALS VS XG"].map(lambda x: f"{x:.3f}")
    show["CONV %"] = show["CONV %"].map(lambda x: f"{x:.1f}")
    show["XG PER SHOT"] = show["XG PER SHOT"].map(lambda x: f"{x:.3f}")

    print("\n[Player xG summary] (Top {})".format(topn))
    print(show.drop(columns=["playerId"]).to_string(index=False))

    print("\n[Saved] {}".format(out_csv))

if __name__ == "__main__":
    main()
