# Raw Data Inventory

This file separates raw input data from derived analysis outputs in this project.

## Primary Raw Data Used by the Current Manuscript

The current multi-competition xG analysis is driven by Wyscout-style event and match JSON files under `data/events/` and `data/matches/`.

| Competition | Event file | Events | Match file | Matches |
|---|---:|---:|---:|---:|
| England | `data/events/events_England.json` | 643,150 | `data/matches/matches_England.json` | 380 |
| France | `data/events/events_France.json` | 632,807 | `data/matches/matches_France.json` | 380 |
| Germany | `data/events/events_Germany.json` | 519,407 | `data/matches/matches_Germany.json` | 306 |
| Italy | `data/events/events_Italy.json` | 647,372 | `data/matches/matches_Italy.json` | 380 |
| Spain | `data/events/events_Spain.json` | 628,659 | `data/matches/matches_Spain.json` | 380 |
| European Championship | `data/events/events_European_Championship.json` | 78,140 | `data/matches/matches_European_Championship.json` | 51 |
| World Cup | `data/events/events_World_Cup.json` | 101,759 | `data/matches/matches_World_Cup.json` | 64 |

Total primary event records: 3,251,294.  
Total primary match records: 1,941.

## Primary JSON Schema

Event records contain the fields:

```text
eventId, eventName, eventSec, id, matchId, matchPeriod, playerId,
positions, subEventId, subEventName, tags, teamId
```

These fields are used to construct:

- shot outcomes;
- shot location and attacking-frame geometry;
- body-part and contextual tags;
- previous-event context;
- match clock and score state;
- competition-level and match-level group identifiers.

Match records contain the fields:

```text
competitionId, date, dateutc, duration, gameweek, groupName, label,
referees, roundId, seasonId, status, teamsData, venue, winner, wyId
```

These fields are used to attach match metadata and recover team/match context.

## Earlier World Cup 64-Match Raw Data

The repository also contains an older World Cup 64-match data extract under `data/data/`. This is used by earlier scripts such as `Code/xG.py`, `Code/player_xg_report.py`, and older exploratory analyses. It is not the main input for the current multi-competition manuscript pipeline.

| File | Rows including header | Role |
|---|---:|---|
| `data/data/worldcup_64_events.csv` | 101,760 | event-level World Cup actions with locations, tags, players, teams, and match identifiers |
| `data/data/worldcup_64_match.xlsx` | not counted here | match-level World Cup metadata |
| `data/data/worldcup_64_players.csv` | 744 | player metadata |
| `data/data/worldcup_64_teams.csv` | 33 | team metadata |
| `data/data/worldcup_64_playersrank.csv` | 1,554 | player ranking/minutes metadata |

`worldcup_64_events.csv` has the following columns:

```text
eventId, subEventName, playerId, matchId, eventName, teamId,
matchPeriod, eventSec, subEventId, id, x_start, y_start, x_end, y_end,
tag_ids, tag_labels, tag_descriptions
```

## Derived Data, Not Raw Data

The following files and directories are generated or intermediate outputs and should not be described as raw data:

- `SR_Submission/results/*.csv`: manuscript result tables and analysis outputs.
- Duplicate or old-template result directories outside `SR_Submission/`: historical working outputs that are not part of the current public SR package.
- `data/shot_features_with_xG.csv`: derived shot-level xG table.
- `data/shot_features_open_play.csv`: derived open-play shot features.
- `data/worldcup_passes_prepared.csv`: processed pass table.
- `data/node_features.csv`, `data/edge_features.csv`, `data/global_features.csv`: derived network-analysis features.
- `data/net_outputs/*.csv`: network-analysis outputs.
- `data/*.pt`, `data/results/*.pt`, `data/*.joblib`, `data/*.npz`: trained model or binary artifacts.
- `data/*.png`: diagnostic plots and generated figures.
