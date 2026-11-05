# Football AI — Workflow

How to run the system. Everything is automatic and dynamic per match — the
only human input is **two optional tools**: *multi-select* and *correction*.
The pitch homography is produced automatically **from our trained model**, so
there is no manual pitch clicking.

---

## 1. Fully automatic run (one command)

```bash
python main.py --video <match>.mp4 --all --device cuda:0
```

This runs the whole pipeline end to end:

```
detect → track → dynamic team/role refinement →
homography from our model (+ camera-motion smoothing) →
minimap + skeleton-position video →
speed/distance → player performance → heatmaps →
tactics → possession → final match report
```

The team colours, the two teams, and the pitch geometry are all discovered
from the footage — nothing is configured per video. You already get a complete
result and all reports from this single command.

---

## 2. The two manual tools (optional, human-in-the-loop)

Both tools only **refine** the automatic result; they never replace the
pipeline and the automatic output is always kept underneath. They are the
screens that get wired into the front-end (Flutter) app so an operator can open
them and adjust.

### a) Multi-select

Label whole teams / referees / goalkeepers fast by clicking the boxes on a few
representative frames.

```bash
python -m manual_correction initialize --video <match>.mp4
```

- Click the players you want, then assign them in one action to
  **Team 0 / Team 1 / Referee / Goalkeeper**.
- Do a few frames for good coverage, then save.

### b) Correction

Fix any single track's role/team and type a **player name** (the name then
appears on the video and in every analytics output).

```bash
python -m manual_correction review --video <match>.mp4
```

- Pick a track, set its role/team, and type the **player name**.
- Save.

> The name and the corrections are stored in the corrections file and overlaid
> on the automatic roles — so they survive, are auditable, and are read by the
> reports and the final video.

---

## 3. Re-run so your manual edits apply

After multi-select / correction, re-run **reusing the same tracks** so the
track ids (which your labels are keyed to) stay identical:

```bash
python main.py --video <match>.mp4 --all --reuse-tracks --device cuda:0
```

`--reuse-tracks` skips re-detection/re-tracking and reuses the existing
identities, so your manual teams, corrections and player names are applied,
then the minimap, skeleton video and all reports are regenerated.

> Rule of thumb: run `--all` once → use the manual tools → re-run with
> `--reuse-tracks`. Plain `--all` again would re-number the tracks and your
> manual edits would no longer match.

---

## 4. What you get

Every run writes into **one folder per match**, never into `outputs/` itself:

```bash
python main.py --video data/<match>.mp4 --output-dir outputs/<match-id>/ ...
```

```
outputs/<match-id>/
  <stem>_tracks_roster.json        detection + tracking identities
  <stem>_roles.json                automatic team / role assignment
  <stem>_roles_final.json          roles after your corrections
  <stem>_role_corrections.json     your manual team + name edits
  <stem>_homography.json           fixed pitch calibration (when present)
  manual_review/                   review manifest + per-track crops
  <stem>_detections.json
  <stem>_{analytics,player_analytics,team_tactical}.{json,txt}
  <stem>_{possession,final_report}.{json,txt}
  <stem>_field_positions.json
  <stem>_{team0,team1}_*heatmap.png
  <stem>_minimap.mp4
```

The reports exist as both machine-readable **JSON** (for the app) and
human-readable **TXT** (to read directly).

Because the folder is also where the pipeline *looks* for its inputs, re-running
that match is one command — it finds the tracks, the corrections and the
calibration in place:

```bash
python main.py --video data/<match>.mp4 --output-dir outputs/<match-id>/ \
  --reuse-tracks --auto-calibration --save-analytics --device dml
```

> `--reuse-tracks` only reuses tracks it can find **in that output dir**. Point
> it at a fresh folder and it silently falls back to full detection.

> An existing `<stem>_role_corrections.json` makes the run skip the review UI.
> Without one, a run that produces role outputs opens the Tk review window and
> waits for a human — which is why an unattended run must pass corrections.

---

## 5. HD ingest and speed (measured on the RX 580 / Xeon E5-2686 v4 box)

Numbers below were measured on this machine, not estimated. Re-measure if the
hardware changes.

### Where the time goes

| Stage | Measured |
|---|---|
| Decode 480p | 364 fps — **never the bottleneck** |
| Detect @1280 DirectML | **7.7–9 fps — the bottleneck** |
| Detect @1280 CPU | 0.5 fps (DirectML is ~17x faster) |
| Resize 1080p→1280 | 4.3 ms |

`imgsz` against source resolution (measured with ultralytics NMS):

| imgsz | ms/frame | players/frame | mean conf |
|---|---|---|---|
| 640 | 353 | 7.8 | 0.687 |
| **960** | **710** | **10.8** | 0.729 |
| 1280 | 1169 | 11.0 | 0.718 |

**960 ≈ 1280 in players found, 1.65x faster.** 640 loses ~30% of players.
For preview work on 480p, use 960. On true 1080p use 960 as well — see the
operating rules below, which supersede this table's CPU-path conclusion.

### The operating rules (decided from measurement, not preference)

**1. `imgsz` = 960. Not 1280.**

Measured through `detection/onnx_dml.py` on true 1080p60 broadcast frames
(`persibo-full.mp4`), not through ultralytics' CPU provider:

| imgsz | ms/frame | fps | players/frame | mean conf |
|---|---|---|---|---|
| **960** | **121** | **8.3** | **9.3** | 0.756 |
| 1280 | 231 | 4.3 | 9.1 | 0.786 |

1.9x faster for 0.2 players lost. The earlier table in this file was measured
on the CPU path and does not describe production; ignore its "1280 on 1080p"
conclusion.

**2. FP16 is not a speed lever here.** `quantize=16` halves the file (78 MB ->
39 MB) and is *slower*: 120 ms vs 110 ms at 1280. The RX 580 has no fast FP16
path through DirectML. Do not re-export for speed.

**3. Batching stays at 1.** Measured earlier: 7.7 -> 5.4 -> 6.2 fps for
batches of 1/4/8.

**4. Never let `weights/best.onnx` be re-exported at a different `imgsz`.**
The exported graph has a static input shape. Exporting at 960 silently
overwrote the 1280 model and halved detections until it was rebuilt. If a
different size is needed, export under a distinct filename.

**5. Calibrate at 1080p, never at 480p.** At 480p the pitch model found 10.8
keypoints per frame and the homography was rejected on 87% of frames. At 1080p
it finds 27.1 and accepts 34%. Same model, same code -- resolution is the
whole difference.

**6. Do not interpolate between calibrated frames. Hold the last one.**
Leave-one-out over 687 frames: interpolating a homography across a gap gave a
median 31.6 m error; holding the previous homography gave 9.0 m. Interpolation
won in only 20% of cases. `KeypointHomographyProvider.for_frame` already does
the right thing -- keep it.

**7. A homography is only valid where its keypoints are.** In 7 of 8 sampled
broadcast frames the keypoints sat entirely in the left two-thirds of the
pitch: zero points in the right third. Projected pitch lines landed on white
paint 2.17x better than a control *inside* the keypoint hull, and 0.45x worse
*outside* it. Treat the hull as the region of validity; a low reprojection
error does not mean the whole frame is calibrated.

### Levers that do NOT work here

- **Batching** — DirectML is slower at batch>1 (7.7 → 5.4 → 6.2 fps for 1/4/8).
- **HIPIFY / ROCm porting** — the repo contains **zero `.cu` files**; inference
  goes through ONNX Runtime, which already calls DirectML directly.

### Source resolution matters more than model size

A 480p player is ~35 px tall; the same player at 1080p is ~107 px. Upscaling
480p to 1280 adds no information. Always ingest the highest resolution the
source offers (iLeague TV publishes 1080p60), then let `imgsz` shrink it.

> **60 fps gotcha:** the 1080p stream is 60 fps. Every distance and speed
> figure divides by fps — feeding 60 fps footage without updating `fps`
> halves every speed. Set the fps from the file, never assume 30.

### Downloading a match

`yt-dlp` needs a real ffmpeg binary for `--download-sections`; putting it on
PATH is not enough, pass it explicitly:

```bash
python -m yt_dlp --js-runtimes node --ffmpeg-location "<path>/ffmpeg.exe" \
  -f 299 --download-sections "*72:00-72:30" --force-keyframes-at-cuts \
  -o "data/<match>.%(ext)s" "<youtube-url>"
```

`-f 299` = 1080p60 (3.98 GiB for a full match); `-f 399` = 1080p60 AV1
(3.40 GiB, smaller). A 30 s 1080p60 slice is ~20 MB and downloads in ~70 s.

> Broadcast clock ≠ video timestamp. In the 23 Dec 2025 match the scoreboard
> read 47:27 at video time 72:00, so a `CLIP_START` guessed from the URL is
> wrong. Read the scoreboard in a sampled frame to anchor the offset.

## 6. Front-end (Flutter) integration

The app reads/writes plain JSON, so the two manual tools map cleanly onto app
screens:

- **Read** `<match>_roles_final.json` and the analytics JSONs to show players,
  teams, names, ratings and tactics.
- **Write** the corrections file (role / team / **player name**) from the app's
  multi-select and correction screens.
- Re-run with `--reuse-tracks` to apply and regenerate everything.
