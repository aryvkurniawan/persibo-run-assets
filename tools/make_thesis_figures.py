"""Generate a complete, thesis-ready figure set for the Football AI pipeline.

Produces one clean PNG per pipeline stage into ``thesis_figures/`` so they can
be dropped straight into the dissertation. Reuses the already-computed outputs
for ``input_video`` (tracks, roles, field positions, heatmaps) so nothing heavy
is recomputed except the few model passes needed for a crisp figure.

Run from the football_ai directory:

    python tools/make_thesis_figures.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "thesis_figures"
OUTPUTS = ROOT / "outputs"
VIDEO = ROOT / "input_video.mp4"
STEM = "input_video"
FRAME = 376      # has exactly 22 outfield identities — ideal for the book
KP_FRAME = 501   # penalty box + touchlines fully visible — best calibration view

# Team / role colours (BGR) — consistent across every figure.
TEAM0 = (60, 60, 235)     # red
TEAM1 = (235, 150, 40)    # blue
GK = (40, 200, 255)       # amber
REF = (60, 220, 60)       # green
BALL = (0, 255, 255)      # yellow
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
def grab_frame(idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {idx}")
    return frame


def load_tracks(idx: int):
    path = OUTPUTS / f"{STEM}_tracks_swapfixed.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for fr in data["frames"]:
        if fr["frame"] == idx:
            return fr["tracks"]
    return []


def load_roles():
    path = OUTPUTS / f"{STEM}_roles_final.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for t in data["tracks"]:
        out[t["track_id"]] = {
            "role": t.get("final_role", "player"),
            "team": t.get("team_id"),
            "name": t.get("player_name"),
        }
    return out


def banner(img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Add a clean caption strip at the top of a figure."""
    h, w = img.shape[:2]
    bar = int(h * 0.075)
    canvas = img.copy()
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    cv2.putText(canvas, title, (24, int(bar * 0.62)), FONT,
                w / 1700.0, WHITE, 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (24, int(bar * 0.92)), FONT,
                    w / 3200.0, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def box(img, b, color, label="", thick=2):
    x1, y1, x2, y2 = (int(v) for v in b)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)


def id_color(tid: int):
    """Stable, well-separated colour per track id (golden-ratio hue)."""
    hue = int((tid * 0.61803398875 * 180) % 180)
    hsv = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def save(name: str, img: np.ndarray):
    path = OUT / name
    cv2.imwrite(str(path), img)
    print(f"  + {name}  ({img.shape[1]}x{img.shape[0]})")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_detection(frame, roles):
    """Stage 1 — raw YOLO detections coloured by class."""
    from detection.detector import YOLODetector
    from utils.config_loader import load_config

    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    cfg.model.device = "cpu"
    det = YOLODetector(config=cfg.model, class_names=cfg.classes)
    dets = det.detect(frame)
    img = frame.copy()
    palette = {"player": TEAM1, "goalkeeper": GK, "referee": REF, "ball": BALL}
    counts = {}
    for d in dets:
        c = palette.get(d.class_name, WHITE)
        counts[d.class_name] = counts.get(d.class_name, 0) + 1
        box(img, d.bbox, c, f"{d.class_name} {d.confidence:.2f}")
    sub = "  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    save("01_detection.png", banner(
        img, "Stage 1 - Object Detection (trained YOLO)",
        f"4 classes  |  {sub}"))


def fig_tracking(frame, tracks):
    """Stage 2 — stable identities (one colour + id per track)."""
    img = frame.copy()
    n = 0
    for t in tracks:
        if t["class"] == "ball":
            box(img, t["bbox"], BALL, "ball")
            continue
        n += 1
        box(img, t["bbox"], id_color(t["track_id"]), f"#{t['track_id']}")
    save("02_tracking.png", banner(
        img, "Stage 2 - Multi-Object Tracking (BoT-SORT)",
        f"stable identities across frames  |  {n} tracked players"))


def fig_teams(frame, tracks, roles):
    """Stage 3 — dynamic team / role assignment."""
    img = frame.copy()
    tally = {"Team 0": 0, "Team 1": 0, "GK": 0, "Ref": 0}
    for t in tracks:
        tid = t["track_id"]
        if t["class"] == "ball":
            box(img, t["bbox"], BALL, "ball")
            continue
        r = roles.get(tid, {})
        role, team, name = r.get("role"), r.get("team"), r.get("name")
        if role == "referee":
            col, lab, key = REF, "REF", "Ref"
        elif role == "goalkeeper":
            col, lab, key = GK, f"GK T{team}", "GK"
        else:
            col = TEAM0 if team == 0 else TEAM1
            lab = name if name else f"T{team} #{tid}"
            key = f"Team {team}" if team in (0, 1) else "Team 0"
        if key in tally:
            tally[key] += 1
        box(img, t["bbox"], col, lab)
    sub = "  ".join(f"{k}: {v}" for k, v in tally.items())
    save("03_team_role_assignment.png", banner(
        img, "Stage 3 - Dynamic Team & Role Assignment",
        f"teams discovered by jersey clustering  |  {sub}"))


def _pitch_field_lines(L=105.0, W=68.0, boxes=True, outer=True, circle=True):
    """Canonical pitch line segments in metres (FIFA proportions)."""
    segs = []

    def rect(x1, y1, x2, y2):
        segs.extend([((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                     ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))])

    if outer:
        rect(0, 0, L, W)                   # touchlines / goal lines
    segs.append(((L / 2, 0), (L / 2, W)))  # halfway line
    if boxes:
        rect(0, 13.84, 16.5, 54.16)        # left penalty box
        rect(L - 16.5, 13.84, L, 54.16)    # right penalty box
        rect(0, 24.84, 5.5, 43.16)         # left goal box
        rect(L - 5.5, 24.84, L, 43.16)     # right goal box
    if circle:
        cx, cy, r = L / 2, W / 2, 9.15
        cpts = [(cx + r * np.cos(t), cy + r * np.sin(t))
                for t in np.linspace(0, 2 * np.pi, 48)]
        segs.extend(list(zip(cpts, cpts[1:])))
    return segs


def _ellipse_affine(src_ellipse, dst_ellipse):
    """3-point affine that maps one ellipse (centre+axes+angle) onto another."""
    def pts(el):
        (cx, cy), (aw, ah), ang = el
        t = np.deg2rad(ang)
        ax = np.array([np.cos(t), np.sin(t)]) * (aw / 2.0)
        ay = np.array([-np.sin(t), np.cos(t)]) * (ah / 2.0)
        c = np.array([cx, cy])
        return np.float32([c, c + ax, c + ay])
    return cv2.getAffineTransform(pts(src_ellipse), pts(dst_ellipse))


def _fit_projected_circle(homography, n=72):
    """Project the field centre-circle and fit the resulting image ellipse."""
    cx, cy, r = 52.5, 34.0, 9.15
    pts = []
    for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
        p = homography.project_field_to_image(
            (cx + r * np.cos(t), cy + r * np.sin(t)))
        pts.append([p[0], p[1]])
    return cv2.fitEllipse(np.float32(pts))


def _draw_pitch_overlay(img, homography, circle_ellipse=None):
    """Project the canonical pitch onto the frame (field -> image).

    The keypoint homography aligns the lines near the detected points (here the
    penalty box) very well, but the *far* centre circle is extrapolated and can
    drift a little on one frame. When ``circle_ellipse`` is given, the centre
    circle is drawn from that hand-fit image-space ellipse instead of the
    projected one, so every visible marking sits on the real pitch.
    """
    h, w = img.shape[:2]

    def project(p):
        x, y = homography.project_field_to_image((float(p[0]), float(p[1])))
        return int(round(x)), int(round(y))

    for a, b in _pitch_field_lines(circle=(circle_ellipse is None)):
        try:
            pa, pb = project(a), project(b)
        except Exception:
            continue
        # keep segments that stay near the frame (off-screen boxes would
        # otherwise draw long stray lines across the image)
        if all(-0.3 * w < x < 1.3 * w and -0.3 * h < y < 1.3 * h
               for x, y in (pa, pb)):
            cv2.line(img, pa, pb, (60, 235, 60), 2, cv2.LINE_AA)

    if circle_ellipse is not None:
        (cx, cy), (aw, ah), ang = circle_ellipse
        cv2.ellipse(img, (int(cx), int(cy)), (int(aw), int(ah)), ang,
                    0, 360, (60, 235, 60), 2, cv2.LINE_AA)


def fig_keypoints(frame):
    """Stage 5 — our pitch-keypoint model -> homography.

    Solves the homography **fresh from this exact frame's keypoints** so the
    projected pitch lines align precisely (the saved homography.json is
    sampled/smoothed and can drift a little off a single frame).
    """
    from calibration.pitch_keypoints import (
        KeypointPitchDetector,
        homography_from_keypoints,
    )

    img = frame.copy()
    kps, homography = [], None
    try:
        det = KeypointPitchDetector(
            model_path=str(ROOT / "weights" / "pitch_keypoints_yolo11m_best.pt.pt"),
            device="cpu")
        kps = det.detect(frame)
        # generous error gate: we want the best fit for THIS frame, not a reject
        homography = homography_from_keypoints(
            kps, min_conf=0.5, min_points=4, max_error_m=1e9)
    except Exception as exc:  # pragma: no cover - model/IO guard
        print(f"  ! keypoints skipped ({exc})")

    # Prefer a MANUAL calibration if the user clicked points for this frame —
    # it sits exactly on the pitch (see tools/calibrate_figure.py).
    manual = _manual_homography(KP_FRAME)
    source = "manual click" if manual is not None else "keypoint model"
    if manual is not None:
        homography = manual

    if homography is not None:
        _draw_pitch_overlay(img, homography)
        err = homography.reprojection_error()
    else:
        err = float("nan")

    # detect() returns tuples (class_id, x, y, conf) for all 28 points.
    shown = 0
    for cid, x, y, conf in kps:
        if conf < 0.5:
            continue
        shown += 1
        xi, yi = int(x), int(y)
        cv2.circle(img, (xi, yi), 9, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (xi, yi), 7, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, str(cid), (xi + 8, yi - 8), FONT, 0.5,
                    (0, 255, 255), 2, cv2.LINE_AA)
    err_txt = f"reproj err {err:.2f} m" if err == err else "no solve"
    save("05_pitch_keypoints_homography.png", banner(
        img, "Stage 5 - Pitch Keypoints -> Homography (our model)",
        f"{shown} keypoints (yellow) + projected pitch model (green)  |  "
        f"{err_txt}  ({source})"))


def _manual_homography(frame_idx, max_err_m=2.5, min_points=6):
    """Load a hand-clicked calibration homography for ``frame_idx`` if present.

    Robust to a few mis-clicked / mislabelled landmarks: it solves, then keeps
    dropping the single worst point while the reprojection error is above
    ``max_err_m`` (and enough points remain), so a clean homography survives a
    handful of bad clicks.
    """
    import json

    from calibration.homography import Homography

    path = OUTPUTS / f"{STEM}_pitch_calibration.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        kf = min(data.get("frames", []),
                 key=lambda f: abs(f.get("frame", 0) - frame_idx), default=None)
        if not kf or len(kf.get("points", {})) < 4:
            return None
        items = [(n, p["image"], p["field"]) for n, p in kf["points"].items()]

        while True:
            H = Homography.from_correspondences(
                [im for _, im, _ in items], [fl for _, _, fl in items])
            res = []
            for n, im, fl in items:
                fx, fy = H.project_image_to_field(im)
                res.append((((fx - fl[0]) ** 2 + (fy - fl[1]) ** 2) ** 0.5, n))
            worst, wname = max(res)
            if worst <= max_err_m or len(items) <= min_points:
                kept = len(items)
                print(f"  manual calibration: {kept} pts, "
                      f"err {H.reprojection_error():.2f} m")
                return H
            items = [it for it in items if it[0] != wname]   # drop worst
    except Exception as exc:  # pragma: no cover
        print(f"  ! manual calibration ignored ({exc})")
        return None


def fig_skeleton(frame, tracks):
    """Stage 6 — skeleton-position overlay (per-player pose)."""
    from visualization.pose_overlay import PoseEstimator, draw_pose

    class _T:
        def __init__(self, d):
            self.track_id = d["track_id"]
            self.class_name = d["class"]
            self.bbox = d["bbox"]

    img = frame.copy()
    try:
        pose = PoseEstimator(model_path="yolo11m-pose.pt", device="cpu")
        objs = [_T(t) for t in tracks if t["class"] != "ball"]
        kmap = pose.detect_on_tracks(frame, objs)
    except Exception as exc:  # pragma: no cover
        print(f"  ! pose skipped ({exc})")
        kmap = {}
    if kmap:
        roles = load_roles()
        bbox_of = {t["track_id"]: t["bbox"] for t in tracks}

        def role_color(tid):
            r = roles.get(tid, {})
            role, team = r.get("role"), r.get("team")
            if role == "referee":
                return REF
            if role == "goalkeeper":
                return GK
            return TEAM0 if team == 0 else TEAM1

        for tid, kp in kmap.items():
            draw_pose(img, kp, role_color(tid))
        # Magnified inset of the clearest (largest) OUTFIELD PLAYER skeleton —
        # never the referee — so the body pose is legible in print.
        def _area(tid):
            b = bbox_of.get(tid, (0, 0, 0, 0))
            return (b[2] - b[0]) * (b[3] - b[1])
        players = [tid for tid in kmap
                   if roles.get(tid, {}).get("role") == "player"]
        best = max(players or list(kmap), key=_area, default=None)
        if best is not None:
            img = _skeleton_inset(img, bbox_of.get(best))
        save("06_skeleton_pose.png", banner(
            img, "Stage 6 - Skeleton Position Overlay",
            f"per-player body pose  |  {len(kmap)} skeletons"))
    else:
        # Fall back to a frame from the already-rendered final video, which
        # has the skeletons baked in.
        _frame_from_final("06_skeleton_pose.png",
                          "Stage 6 - Skeleton Position Overlay (final render)")


def _skeleton_inset(img, bbox, inset_h=360):
    """Crop around one skeleton, magnify it and paste a bordered inset."""
    if bbox is None:
        return img
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    padx = int((x2 - x1) * 0.6) + 10
    pady = int((y2 - y1) * 0.25) + 10
    cx1, cy1 = max(x1 - padx, 0), max(y1 - pady, 0)
    cx2, cy2 = min(x2 + padx, w), min(y2 + pady, h)
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return img
    scale = inset_h / crop.shape[0]
    inset = cv2.resize(crop, (max(int(crop.shape[1] * scale), 1), inset_h),
                       interpolation=cv2.INTER_CUBIC)
    ih, iw = inset.shape[:2]
    # bottom-left over grass, clear of the broadcast watermark.
    ox, oy = 30, h - ih - 30
    cv2.rectangle(img, (ox - 4, oy - 4), (ox + iw + 4, oy + ih + 4),
                  (255, 255, 255), 3)
    cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2)
    img[oy:oy + ih, ox:ox + iw] = inset
    cv2.putText(img, "zoom", (ox + 6, oy + 26), FONT, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _frame_from_final(name, title):
    final = OUTPUTS / f"{STEM}_final_with_minimap.mp4"
    if not final.is_file():
        print(f"  ! {name} skipped (no final video)")
        return
    cap = cv2.VideoCapture(str(final))
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME - 1)
    ok, fr = cap.read()
    cap.release()
    if ok:
        save(name, banner(fr, title, ""))


def fig_final_composite():
    """Stage 6/12 — the full broadcast composite (skeletons + minimap + bars)."""
    _frame_from_final(
        "12_final_composite.png",
        "Final Output - skeletons, names, minimap, possession & stat bars")


def fig_detection_stats():
    """Detection-model output analysis over the whole clip (real metrics)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(
        (OUTPUTS / f"{STEM}_detections.json").read_text(encoding="utf-8"))
    order = ["player", "goalkeeper", "referee", "ball"]
    cols = {"player": "#2b7bba", "goalkeeper": "#f0a020",
            "referee": "#3cb44b", "ball": "#e0c020"}
    counts = {k: 0 for k in order}
    confs = {k: [] for k in order}
    for fr in data["frames"]:
        for d in fr["detections"]:
            c = d["class"]
            if c in counts:
                counts[c] += 1
                confs[c].append(d["confidence"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)
    fig.suptitle("Detection Model — output analysis over the match "
                 f"({len(data['frames'])} frames)",
                 fontsize=13, fontweight="bold")

    ax1.bar([k.capitalize() for k in order],
            [counts[k] for k in order],
            color=[cols[k] for k in order])
    ax1.set_ylabel("Total detections")
    ax1.set_title("Detections per class")
    for i, k in enumerate(order):
        ax1.text(i, counts[k], f"{counts[k]:,}", ha="center",
                 va="bottom", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    for k in order:
        if confs[k]:
            ax2.hist(confs[k], bins=20, range=(0, 1), alpha=0.65,
                     color=cols[k], label=f"{k} (μ={np.mean(confs[k]):.2f})")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Detections")
    ax2.set_title("Confidence distribution per class")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "00_detection_model_stats.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  + 00_detection_model_stats.png  (model output analysis)")


def copy_existing():
    """Pull in the figures already rendered by the pipeline."""
    mapping = {
        f"{STEM}_team0_heatmap.png": "07_heatmap_team0.png",
        f"{STEM}_team1_heatmap.png": "08_heatmap_team1.png",
        f"{STEM}_team0_best_player_heatmap.png": "09_heatmap_best_player_team0.png",
        f"{STEM}_team1_best_player_heatmap.png": "10_heatmap_best_player_team1.png",
        "inputvideo_tactical_preview.png": "04_minimap_tactical.png",
        "statbar_preview.png": "11_speed_distance_bar.png",
    }
    for src, dst in mapping.items():
        s = OUTPUTS / src
        if s.is_file():
            shutil.copyfile(s, OUT / dst)
            print(f"  + {dst}  (copied from {src})")
        else:
            print(f"  ! missing {src}")


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    print(f"Writing thesis figures -> {OUT}")
    frame = grab_frame(FRAME)
    tracks = load_tracks(FRAME)
    roles = load_roles()
    print(f"frame {FRAME}: {len(tracks)} tracks, {len(roles)} roles loaded\n")

    fig_detection_stats()
    fig_detection(frame, roles)
    fig_tracking(frame, tracks)
    fig_teams(frame, tracks, roles)
    copy_existing()
    fig_keypoints(grab_frame(KP_FRAME))   # penalty-box view aligns best
    fig_skeleton(frame, tracks)
    fig_final_composite()
    print("\nDone.")


if __name__ == "__main__":
    main()
