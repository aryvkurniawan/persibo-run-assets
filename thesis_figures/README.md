# Thesis Figures — Football AI

A complete, print-ready figure set for the dissertation. Every image is
rendered from the **real `input_video` match outputs**. Most stage figures use
frame 376 (exactly 22 outfield identities); the calibration figure uses frame
501 (penalty box fully visible) and the final composite uses frame 500 (best
spread, ball in play). Regenerate any time with:

```bash
python tools/make_thesis_figures.py      # figures 00-11
python tools/make_input_finals.py --pick 500   # figure 12 (fresh, current roles)
```

> Figure 12 is re-rendered from the **current** `roles_final` (not the older
> `*_final_with_minimap.mp4`), so referees/keepers are labelled by role
> (`Ref #58`, `GK #..`) and never as "Player".

All on-pitch figures use the **same colour key**:
**Team 0 = red, Team 1 = blue, Goalkeeper = amber, Referee = green, Ball = yellow.**

| # | File | Pipeline stage | What it shows |
|---|------|----------------|---------------|
| 00 | `00_detection_model_stats.png` | Detection (model) | Per-class detection counts + confidence distribution over all 750 frames (players μ≈0.82). |
| 01 | `01_detection.png` | Stage 1 — Detection | Raw YOLO boxes for the 4 classes (player / goalkeeper / referee / ball). |
| 02 | `02_tracking.png` | Stage 2 — Tracking | Stable BoT-SORT identities, one colour + id per track. |
| 03 | `03_team_role_assignment.png` | Stage 3 — Roles | Dynamic team clustering + goalkeeper/referee split + player names. |
| 04 | `04_minimap_tactical.png` | Stage 6 — Minimap | Top-down tactical minimap (team shapes, average-position lines). |
| 05 | `05_pitch_keypoints_homography.png` | Stage 5 — Calibration | Pitch keypoints (yellow) + canonical pitch projected back via the homography (green). |
| 06 | `06_skeleton_pose.png` | Stage 6 — Pose | Per-player skeleton overlay with a magnified inset. |
| 07 | `07_heatmap_team0.png` | Stage 9 — Heatmaps | Team 0 positional heatmap. |
| 08 | `08_heatmap_team1.png` | Stage 9 — Heatmaps | Team 1 positional heatmap. |
| 09 | `09_heatmap_best_player_team0.png` | Stage 9 — Heatmaps | Team 0 best (most-distance) player heatmap. |
| 10 | `10_heatmap_best_player_team1.png` | Stage 9 — Heatmaps | Team 1 best player heatmap. |
| 11 | `11_speed_distance_bar.png` | Stage 7 — Speed/Distance | Live per-player speed + distance stat bar on the video. |
| 12 | `12_final_composite.png` | Final output | Full broadcast composite: skeletons, names, minimap, possession & stat bars. |

## Suggested chapter placement
- **System Overview / Methodology** → 01, 02, 03, 05, 06 (one per stage).
- **Detection chapter** → 00 (model output analysis) + 01.
- **Calibration chapter** → 05.
- **Analytics chapters** → 04 (tactics), 07–10 (heatmaps), 11 (speed/distance).
- **Results / Demo** → 12 (the full composite).

## Note on detection *training* curves
The training loss / mAP / confusion-matrix / PR curves are produced by the
Ultralytics training run (the `runs/detect/train/` folder: `results.png`,
`confusion_matrix.png`, `PR_curve.png`). Those artifacts are **not stored in
this project** — export them from wherever the model was trained (Colab /
Roboflow / local `runs/` folder) to include them. Figure **00** above is the
model's *output* analysis on this match, computed from real detections, and can
stand in for an inference-quality figure.
