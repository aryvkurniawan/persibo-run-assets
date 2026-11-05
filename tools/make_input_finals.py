"""Render fresh full-composite STILL FRAMES for input_video (WITH minimap).

The on-disk ``input_video_final_with_minimap.mp4`` was rendered with older
roles (e.g. a referee still labelled "Player 58"). This re-renders selected
frames from the CURRENT ``roles_final`` so referees/goalkeepers are correct,
reusing the exact pipeline drawing (per-frame team colour, skeletons + names,
speed/distance bars, possession bar) plus the tactical minimap.

    python tools/make_input_finals.py            # render candidate frames
    python tools/make_input_finals.py --pick 376 # write thesis figure 12

Output: thesis_figures/_candidates/ for review, or 12_final_composite.png.
"""

from __future__ import annotations

import argparse
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
VIDEO = ROOT / "input_video.mp4"
STEM = "input_video"
CANDIDATES = [300, 376, 450, 500, 600]
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def cumulative_possession(timeline):
    out, c0, c1 = {}, 0, 0
    for entry in timeline:
        frame, team = (entry["frame"], entry.get("team")) if isinstance(
            entry, dict) else (entry[0], entry[1])
        if team == 0:
            c0 += 1
        elif team == 1:
            c1 += 1
        tot = c0 + c1
        out[frame] = (50.0, 50.0) if tot == 0 else (
            100.0 * c0 / tot, 100.0 * c1 / tot)
    return out


def build_context():
    config = load_config(str(ROOT / "config" / "config.yaml"))
    ball_cls = config.tracking.ball_class_name
    frames, _size, meta = load_frame_tracks(
        str(OUTPUTS / f"{STEM}_tracks_swapfixed.json"))
    roles_path = OUTPUTS / f"{STEM}_roles_final.json"
    roles_map = mm.load_roles_map(roles_path)
    names_map = mm.load_player_names(roles_path)
    fps = float(meta.get("video_fps", 25.0))

    positions = json.loads(
        (OUTPUTS / f"{STEM}_field_positions.json").read_text("utf-8"))["frames"]
    ac = config.analytics
    result = compute_analytics(
        positions, roles_map, fps, smoothing_window=ac.smoothing_window,
        max_speed_kmh=ac.max_speed_kmh, min_samples=ac.min_samples,
        ball_class_name=ball_cls, exclude_roles=ac.exclude_roles,
        min_track_samples=ac.min_track_samples, player_names=names_map)
    speed_lookup = build_stat_lookup(result)

    poss = json.loads(
        (OUTPUTS / f"{STEM}_possession.json").read_text("utf-8"))
    poss_cum = cumulative_possession(poss["timeline"])

    print("Classifying per-frame team colours (one video pass)...")
    team_override = mm.classify_per_frame_teams(
        frames, str(VIDEO), roles_map, ball_class_name=ball_cls)

    renderer = mm.MinimapRenderer(
        px_per_meter=config.minimap.px_per_meter,
        player_radius=config.minimap.player_dot_radius,
        ball_radius=config.minimap.ball_dot_radius,
        stripes=config.minimap.stripes,
        hide_referees=config.minimap.hide_referees,
        tactical_style=config.minimap.tactical_style)
    pos_by_frame = {p["frame"]: p["objects"] for p in positions}
    pose = PoseEstimator(model_path="yolo11m-pose.pt", device="cpu")
    return dict(frames={f.frame_index: f.tracks for f in frames},
                roles_map=roles_map, names_map=names_map,
                team_override=team_override, speed_lookup=speed_lookup,
                poss_cum=poss_cum, renderer=renderer, pos_by_frame=pos_by_frame,
                pose=pose, color_mode=config.minimap.color_mode)


def render_frame(idx, ctx, with_minimap=True):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    canvas = frame.copy()
    tracks = ctx["frames"].get(idx, [])
    roles_map, names_map = ctx["roles_map"], ctx["names_map"]
    team_override, color_mode = ctx["team_override"], ctx["color_mode"]
    assigned = ctx["pose"].detect_on_tracks(frame, tracks)

    for track in tracks:
        role, team_id = roles_map.get(track.track_id, (track.class_name, None))
        if role == "player":
            ov = team_override.get((idx, track.track_id))
            if ov is not None:
                team_id = ov
        color = mm.color_for_object(role, team_id, track.track_id, color_mode)
        name = names_map.get(track.track_id)
        label = mm._track_label(role, team_id, track.track_id, name)
        if track.track_id in assigned:
            draw_pose(canvas, assigned[track.track_id], color, min_conf=0.5)
            x1, y1 = int(round(track.bbox[0])), int(round(track.bbox[1]))
            cv2.putText(canvas, label, (x1, max(y1 - 4, 10)), _FONT, 0.45,
                        color, 1, cv2.LINE_AA)
        else:
            mm._draw_track_box(canvas, track.bbox, label, color)

    draw_player_stat_bars(canvas, tracks, ctx["speed_lookup"], idx,
                          names_map=names_map)

    if with_minimap:
        # colour each minimap dot like the pipeline (per-frame shirt colour)
        objs = []
        for o in ctx["pos_by_frame"].get(idx, []):
            role, team = o.get("role"), o.get("team_id")
            if role == "player":
                ov = team_override.get((idx, o["track_id"]))
                if ov is not None:
                    team = ov
            objs.append({**o, "color": mm.color_for_object(
                role, team, o["track_id"], color_mode)})
        mini = ctx["renderer"].render(objs, False)
        canvas = mm.overlay_minimap(canvas, mini, position="top_right",
                                    width_ratio=0.28, margin=20)

    p0, p1 = ctx["poss_cum"].get(idx, (50.0, 50.0))
    draw_possession_bar(canvas, p0, p1,
                        color0=mm._TEAM_COLORS[0], color1=mm._TEAM_COLORS[1])
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", type=int, default=None,
                    help="render this frame straight to thesis figure 12")
    args = ap.parse_args()
    ctx = build_context()

    if args.pick:
        img = render_frame(args.pick, ctx, with_minimap=True)
        if img is None:
            print("frame unreadable")
            return
        from tools.make_thesis_figures import banner
        out = ROOT / "thesis_figures" / "12_final_composite.png"
        cv2.imwrite(str(out), banner(
            img, "Final Output - skeletons, names, minimap, "
            "possession & stat bars", ""))
        print(f"  + {out.name}  (frame {args.pick})")
        return

    cand_dir = ROOT / "thesis_figures" / "_candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    for idx in CANDIDATES:
        img = render_frame(idx, ctx, with_minimap=True)
        if img is not None:
            cv2.imwrite(str(cand_dir / f"composite_frame{idx:04d}.png"), img)
            print(f"  + candidate frame {idx}")
    print(f"\nReview {cand_dir}, then: python tools/make_input_finals.py --pick <frame>")


if __name__ == "__main__":
    main()
