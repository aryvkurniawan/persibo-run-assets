# Football AI — Architecture

A modular, layered computer-vision system. Each stage is an **independent
package** that communicates with the next only through **typed data contracts**
(dataclasses) and JSON files on disk — never by passing raw frames around. This
makes every layer separately testable and replaceable, and lets any stage be
re-run on a previous stage's output.

> **Design principle: nothing is hard-coded.** Teams, jersey colours, pitch
> geometry and roles are all discovered from the footage per match. The same
> binary runs on any match with no per-video constants.

---

## Layered view

```
            video
              │
   ┌──────────▼───────────┐
   │ detection            │  YOLO → typed Detections
   └──────────┬───────────┘
   ┌──────────▼───────────┐
   │ tracking             │  BoT-SORT + ball tracker → Tracks (stable ids)
   │   stitch / swap-fix  │  appearance-based identity repair
   └──────────┬───────────┘
   ┌──────────▼───────────┐
   │ role_refinement      │  dynamic team clustering → RoleResult
   └──────────┬───────────┘
   ┌──────────▼───────────┐
   │ manual_correction    │  human-in-the-loop overlay (optional)
   └──────────┬───────────┘
   ┌──────────▼───────────┐
   │ calibration          │  our keypoint model → homography → field metres
   │   + minimap          │  + camera-motion smoothing, top-down render
   └──────────┬───────────┘
   ┌──────────▼───────────┐
   │ analytics            │  speed/distance, performance, heatmaps,
   │                      │  tactics, possession, match report
   └──────────────────────┘

   utils  ── config, logging, video I/O, JSON export (shared by all)
```

---

## Packages

### `detection/`
The object detector and its contracts.
- `detector.py` — wraps the trained YOLO model; the only Ultralytics
  detection path.
- `data_models.py` — `Detection` dataclass (class, confidence, bbox).
- `pipeline.py` — runs detection (+ optional tracking) over a video.

### `tracking/`
Turns detections into stable identities.
- `botsort_tracker.py` — people tracking via BoT-SORT, fed our own detections.
- `ball_tracker.py` — dedicated small-fast-object ball tracker.
- `track_stitcher.py` — re-links broken tracks (gap + predicted position +
  size, optional jersey-colour gate).
- `swap_corrector.py` — appearance-based ID-swap fix and drift split.
- `models.py` — `Track` / `FrameTracks` contracts. `track_exporter.py` — JSON.

### `role_refinement/`
Decides each track's role and team **dynamically**.
- `appearance_extractor.py` — saturation-weighted jersey HSV histograms.
- `team_clusterer.py` — numpy-only spherical k-means; the two largest clusters
  are the teams (discovered, not configured).
- `role_refiner.py` — temporal team voting + referee/goalkeeper split from
  image-space position & motion.
- `role_models.py` — `RoleResult` / `TrackAppearance` contracts.

### `manual_correction/`
The two optional human-in-the-loop tools (see [WORKFLOW.md](WORKFLOW.md)).
- `initialization.py` — **multi-select** team/role seeding.
- `correction_runner.py` / `correction_store.py` — apply & persist
  **corrections** (incl. the manual **player name**), overlaid on auto roles.
- `correction_models.py` — `Correction` / `FinalRole` contracts (carry
  `player_name`, backward compatible).
- `correction_tk_ui.py` — native desktop UI for both tools.

### `calibration/`
Image↔pitch geometry and the top-down render.
- `pitch_keypoints.py` — runs **our trained pitch keypoint model** to detect
  reference points and solve the per-frame **homography** automatically.
- `camera_motion.py` — estimates camera motion and **propagates the homography
  smoothly** across frames (anti-jitter).
- `homography.py` / `pitch_model.py` — projection maths and the canonical
  105×68 m pitch.
- `minimap.py` — top-down render, per-frame team colouring, tactical style,
  final-video compositing.

### `analytics/`
All match intelligence, each phase a small module reading field positions.
- `speed_distance.py`, `analytics_models.py` — speed & distance.
- `player_analytics.py`, `rating_engine.py` — per-player performance + rating.
- `heatmap_generator.py`, `heatmap_renderer.py` — heatmaps.
- `tactical_analysis.py` — team tactics with explainable reasons.
- `possession_engine.py` — field-space possession.
- `match_report.py` — final analyst report.
- `analytics_exporter.py` — JSON **and** TXT writers.

### `visualization/`
- `annotator.py` — box/label drawing.
- `pose_overlay.py` — **skeleton position** overlay (per-player body pose).

### `utils/`
Shared infrastructure: `config_loader.py` (YAML → typed dataclasses),
`logger.py`, `video_io.py`, `json_exporter.py`.

---

## Data contracts (the "API" between layers)

| Stage | Contract |
|---|---|
| Detection | `Detection(class, confidence, bbox)` |
| Tracking | `Track(track_id, class, bbox)` / `FrameTracks` |
| Roles | `RoleResult(track_id, refined_role, team_id, …)` |
| Corrections | `Correction(role, team_id, player_name, …)` → `FinalRole` |
| Field | `field_positions` objects `(track_id, role, team_id, field[x,y])` |
| Speed | `TrackAnalytics`, `AnalyticsResult` |
| Performance | `PlayerPerformance` |
| Tactics | `TacticalProfile` |
| Possession | `PossessionAnalysis` |
| Report | `MatchReport` |

Each contract has `to_dict()` / `from_dict()`, so every stage round-trips
through JSON and can be re-run standalone.

---

## Cross-cutting design choices

- **Dynamic, not hard-coded** — teams are clustered, the pitch homography comes
  from our model, calibration is per match. No fixed colours or formations.
- **Stable identities first** — appearance-aware stitching/swap-fix before any
  analytics, so distances and possession are credited to the right player.
- **Per-frame team colour** — the team shown is the shirt in that frame, so ID
  switches never mislabel a player.
- **Explainable analytics** — every rating/label carries the metric and reason
  it came from; **no external LLM** is used anywhere.
- **Fail-soft** — every optional stage is guarded; a failure logs a warning and
  the pipeline continues.
- **Config-driven** — all thresholds live in `config/config.yaml` → typed
  dataclasses; nothing is baked into the code paths.
