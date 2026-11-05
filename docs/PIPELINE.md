# Football AI — Full Pipeline

End-to-end processing of a football video, from raw frames to a complete
analyst report. Every stage is **fully dynamic and learned per match** — there
are no fixed team colours, no fixed formations, and no per‑video constants.
Teams, pitch geometry and player roles are all discovered from the footage
itself, so the exact same command works on any match.

One command runs the whole pipeline:

```bash
python main.py --video <match>.mp4 --all --device cuda:0
```

---

## Stage 1 — Detection

A trained **YOLO** detector finds the four object classes on every frame:
**player, goalkeeper, referee, ball**. Output is a typed list of detections
(class, confidence, bounding box) per frame.

- In: video frames
- Out: per-frame detections (`<video>_detections.json`, annotated video)

## Stage 2 — Tracking

Detections are linked across time into **stable identities**:

- People (player / goalkeeper / referee) are tracked with **BoT-SORT**.
- The **ball** uses a dedicated tracker, because it is tiny and fast (its
  boxes barely overlap frame to frame, so a normal IoU tracker can't link it).

Two appearance-aware post-processes clean up the identities:

- **Stitching** — re-links a track that broke during an occlusion to the same
  person when they reappear.
- **Swap correction** — uses jersey appearance to undo ID swaps when two
  players cross, and to split a track that drifted onto a different player.

- In: detections
- Out: per-frame tracks with stable ids (`<video>_tracks_*.json`)

## Stage 3 — Role Refinement (dynamic teams)

Roles are decided **per track over many frames**, never from a single frame and
never from fixed colours:

- The two team **appearance clusters are discovered** from the players' jersey
  histograms (the teams are whatever the match actually shows).
- Each track votes its team across sampled frames; tracks far from both team
  clusters are **referee / goalkeeper** candidates, separated by their
  position and motion on the pitch.

- In: tracks + video crops
- Out: per-track roles + team (`<video>_roles.json`)

## Stage 4 — Human-in-the-loop (optional)

Exactly **two** manual touch-points, both optional, both reproducible — see
[WORKFLOW.md](WORKFLOW.md):

1. **Multi-select** — label whole teams / referees / goalkeepers by clicking
   boxes on a few frames.
2. **Correction** — fix any individual track's role/team and type a **player
   name**.

The manual decisions are stored separately and overlaid on the automatic
roles, so the automatic result is never overwritten and stays auditable.

## Stage 5 — Homography / Field Projection (our model)

The image→pitch mapping (homography) is produced **from our own trained pitch
keypoint model** — it detects the pitch reference points each frame and solves
the projection automatically, with **no manual clicking**. The per-frame
homography is then **smoothed with estimated camera motion** so it follows the
panning camera without jitter. Every player's foot point is projected to real
**pitch metres**.

- In: video + our pitch keypoint model
- Out: per-frame field positions in metres (`<video>_field_positions.json`)

## Stage 6 — Minimap & Final Video

A professional top-down minimap and the combined final video are rendered:

- Player **team colour is decided per frame from the shirt** the player is
  actually wearing, so an ID switch can never paint a player the wrong team.
- **Skeleton position overlay** — a pose model draws each player's body
  skeleton on the video instead of a plain box, for a broadcast look. The box
  label shows the **player name** (if set) next to the id.
- The minimap uses a **tactical style** (team shapes + average-position lines),
  bigger dots, and hides referees.

- Out: `<video>_minimap.mp4`, `<video>_final_with_minimap.mp4`

## Stage 7 — Speed & Distance

From the projected metres: per-player **distance covered, average and max
speed**, with moving-average smoothing and impossible-jump rejection. A small
**stat bar under each player** on the video shows live speed + distance.

- Out: `<video>_analytics.json` / `.txt`

## Stage 8 — Player Performance

Per player: **work rate, activity level, fatigue, performance status, a 0–10
rating and a natural-language insight** — all derived deterministically from
the computed metrics (no external LLM).

- Out: `<video>_player_analytics.json` / `.txt`

## Stage 9 — Heatmaps

Gaussian-smoothed position heatmaps painted over a top-down pitch (lines stay
visible), for each team and each team's best (most distance) player.

- Out: `<video>_team{0,1}_heatmap.png`, `<video>_team{0,1}_best_player_heatmap.png`

## Stage 10 — Team Tactical Analysis

Explainable team behaviour with the metric and reason behind each label:
**compactness, team shape, attacking zone, transition speed, build-up style,
pressure style**.

- Out: `<video>_team_tactical.json` / `.txt`

## Stage 11 — Ball Possession

Per-frame nearest-player-to-ball ownership in **pitch metres**, aggregated to
team possession %, a timeline, and the dominant team.

- Out: `<video>_possession.json` / `.txt`

## Stage 12 — Match Report (AI intelligence layer)

The final analyst report, combining everything above: **match summary, dominant
team, man of the match, weakest player, per-team tactical capsule, key insights
and recommendations** — fully deterministic and explainable, no LLM calls.

- Out: `<video>_final_report.json` / `.txt`

---

## Output map

| Stage | Artifact |
|---|---|
| Detection | `<video>_detections.json`, `<video>_annotated.mp4` |
| Tracking | `<video>_tracks.json` → `_tracks_stitched` → `_tracks_swapfixed` |
| Roles | `<video>_roles.json`, `<video>_roles_final.json` |
| Field projection | `<video>_field_positions.json` |
| Video | `<video>_minimap.mp4`, `<video>_final_with_minimap.mp4` |
| Speed & distance | `<video>_analytics.json` / `.txt` |
| Player performance | `<video>_player_analytics.json` / `.txt` |
| Heatmaps | `<video>_team*_heatmap.png` |
| Tactical | `<video>_team_tactical.json` / `.txt` |
| Possession | `<video>_possession.json` / `.txt` |
| Match report | `<video>_final_report.json` / `.txt` |

Every analytics artifact is emitted as both machine-readable **JSON** and
human-readable **TXT**.
