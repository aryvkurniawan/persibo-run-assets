"""Render full final-output STILL FRAMES for the Clasico clip — no minimap.

Reproduces exactly what ``calibration.minimap.render_final_video`` draws on the
broadcast frame (per-frame team colours, skeletons + names, live speed/distance
stat bars and the running possession bar) but **omits the minimap PiP**, for a
handful of hand-picked frames. Output PNGs go to ``thesis_figures/clasico/``.

Run from the football_ai directory:

    python tools/make_clasico_finals.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import calibration.minimap as mm
from analytics import build_stat_lookup, compute_analytics, draw_player_stat_bars
from analytics.possession import draw_possession_bar
from role_refinement.role_refiner import load_frame_tracks
from utils.config_loader import load_config
from visualization.pose_overlay import PoseEstimator, draw_pose

OUTPUTS = ROOT / "outputs"
OUT = ROOT / "thesis_figures" / "clasico"
VIDEO = ROOT / "clasico_clip.mp4"
STEM = "clasico_clip"
FRAMES_TO_RENDER = [490, 980, 1485]   # high player-count frames, spread out
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def cumulative_possession(timeline):
    """frame -> (pct0, pct1) running possession from the per-frame owners."""
    out, c0, c1 = {}, 0, 0
    for entry in timeline:
        t = entry.get("team")
        if t == 0:
            c0 += 1
        elif t == 1:
            c1 += 1
        tot = c0 + c1
        out[entry["frame"]] = (50.0, 50.0) if tot == 0 else (
            100.0 * c0 / tot, 100.0 * c1 / tot)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(str(ROOT / "config" / "config.yaml"))
    ball_cls = config.tracking.ball_class_name

    frames, _size, meta = load_frame_tracks(str(OUTPUTS / f"{STEM}_tracks_swapfixed.json"))
    roles_path = OUTPUTS / f"{STEM}_roles_final.json"
    roles_map = mm.load_roles_map(roles_path)
    names_map = mm.load_player_names(roles_path)
    fps = float(meta.get("video_fps", 25.0))

    # speed/distance stat-bar lookup
    positions = json.loads(
        (OUTPUTS / f"{STEM}_field_positions.json").read_text(encoding="utf-8"))["frames"]
    ac = config.analytics
    result = compute_analytics(
        positions, roles_map, fps, smoothing_window=ac.smoothing_window,
        max_speed_kmh=ac.max_speed_kmh, min_samples=ac.min_samples,
        ball_class_name=ball_cls, exclude_roles=ac.exclude_roles,
        min_track_samples=ac.min_track_samples, player_names=names_map)
    speed_lookup = build_stat_lookup(result)

    # running possession bar
    poss = json.loads(
        (OUTPUTS / f"{STEM}_possession.json").read_text(encoding="utf-8"))
    poss_cum = cumulative_possession(poss["timeline"])

    # per-frame shirt colour (sidesteps id switches)
    print("Classifying per-frame team colours (one video pass)...")
    team_override = mm.classify_per_frame_teams(
        frames, str(VIDEO), roles_map, ball_class_name=ball_cls)

    pose = PoseEstimator(model_path="yolo11m-pose.pt", device="cpu")
    tracks_by_frame = {f.frame_index: f.tracks for f in frames}

    cap = cv2.VideoCapture(str(VIDEO))
    for idx in FRAMES_TO_RENDER:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ok, frame = cap.read()
        if not ok:
            print(f"  ! frame {idx} unreadable")
            continue
        canvas = frame.copy()
        tracks = tracks_by_frame.get(idx, [])
        assigned = pose.detect_on_tracks(frame, tracks)

        for track in tracks:
            role, team_id = roles_map.get(track.track_id, (track.class_name, None))
            if role == "player":
                ov = team_override.get((idx, track.track_id))
                if ov is not None:
                    team_id = ov
            color = mm.color_for_object(role, team_id, track.track_id,
                                        mm.COLOR_MODE_TEAM_ROLE)
            name = names_map.get(track.track_id)
            label = mm._track_label(role, team_id, track.track_id, name)
            if track.track_id in assigned:
                draw_pose(canvas, assigned[track.track_id], color, min_conf=0.5)
                x1, y1 = int(round(track.bbox[0])), int(round(track.bbox[1]))
                cv2.putText(canvas, label, (x1, max(y1 - 4, 10)), _FONT, 0.45,
                            color, 1, cv2.LINE_AA)
            else:
                mm._draw_track_box(canvas, track.bbox, label, color)

        draw_player_stat_bars(canvas, tracks, speed_lookup, idx, names_map=names_map)
        p0, p1 = poss_cum.get(idx, (50.0, 50.0))
        draw_possession_bar(canvas, p0, p1,
                            color0=mm._TEAM_COLORS[0], color1=mm._TEAM_COLORS[1])

        out_path = OUT / f"clasico_final_frame{idx:04d}.png"
        cv2.imwrite(str(out_path), canvas)
        print(f"  + {out_path.name}  ({len(tracks)} tracks, "
              f"poss {p0:.0f}/{p1:.0f})")
    cap.release()
    print("Done.")


if __name__ == "__main__":
    main()
