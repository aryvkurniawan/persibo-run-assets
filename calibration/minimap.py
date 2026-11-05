"""Top-down pitch minimap + field-position projection (Phase 4).

Given a homography, every tracked object's foot point is projected from
image pixels to pitch meters and drawn as a dot on a synthetic top-down
pitch. Produces a minimap video and a per-frame field-positions JSON.

Nothing here depends on detection/tracking/role-refinement internals — it
consumes the exported tracks JSON and a roles map, so it stays a pure
post-processing layer.
"""

from __future__ import annotations

import colorsys
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from calibration.homography import Homography
from calibration.pitch_model import PitchDimensions
from utils.logger import get_logger

logger = get_logger("calibration.minimap")

Point = Tuple[float, float]

# Color modes: analytics view (team/role) vs. debugging (per-track).
COLOR_MODE_TEAM_ROLE = "team_role"
COLOR_MODE_TRACK_ID = "track_id"
COLOR_MODES = (COLOR_MODE_TEAM_ROLE, COLOR_MODE_TRACK_ID)

# Pitch + object colors (BGR). Two greens give the mown-stripe look.
_PITCH_GREEN = (48, 124, 52)
_PITCH_GREEN_DARK = (40, 108, 44)
_LINE_WHITE = (244, 244, 244)
_DOT_OUTLINE = (250, 250, 250)
_DOT_SHADOW = (24, 40, 24)
# Team / role color mapping (BGR):
#   player team 0 -> blue, team 1 -> orange, goalkeeper -> red,
#   referee -> yellow, ball -> white, unknown -> gray.
_TEAM_COLORS = {0: (255, 128, 0), 1: (0, 128, 255)}
_ROLE_COLORS = {
    "referee": (0, 255, 255),     # yellow
    "goalkeeper": (0, 0, 255),    # red
    "ball": (255, 255, 255),      # white
    "unknown": (170, 170, 170),   # gray
}
_DEFAULT_DOT = (170, 170, 170)
_GOLDEN_RATIO_CONJUGATE = 0.61803398875


def color_for(role: str, team_id: Optional[int]) -> Tuple[int, int, int]:
    """Team/role-based dot color (analytics view)."""
    if role == "player" and team_id in _TEAM_COLORS:
        return _TEAM_COLORS[team_id]
    return _ROLE_COLORS.get(role, _DEFAULT_DOT)


def track_id_color(track_id: int) -> Tuple[int, int, int]:
    """Stable, well-spread per-track color (debugging only)."""
    hue = (int(track_id) * _GOLDEN_RATIO_CONJUGATE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def color_for_object(
    role: str,
    team_id: Optional[int],
    track_id: int,
    color_mode: str = COLOR_MODE_TEAM_ROLE,
) -> Tuple[int, int, int]:
    """Resolve a dot color according to ``color_mode``."""
    if color_mode == COLOR_MODE_TRACK_ID:
        return track_id_color(track_id)
    return color_for(role, team_id)


class MinimapRenderer:
    """Draws a top-down pitch and projects field points onto it."""

    def __init__(
        self,
        dims: PitchDimensions = PitchDimensions(),
        px_per_meter: float = 8.0,
        margin: int = 24,
        player_radius: int = 9,
        ball_radius: int = 6,
        stripes: int = 12,
        hide_referees: bool = False,
        tactical_style: bool = False,
    ) -> None:
        self.dims = dims
        self.scale = float(px_per_meter)
        self.margin = int(margin)
        self.hide_referees = bool(hide_referees)
        self.tactical_style = bool(tactical_style)
        # Tactical view: bigger dots so the team shapes read clearly.
        self.player_radius = int(player_radius) + (3 if tactical_style else 0)
        self.ball_radius = int(ball_radius)
        self.stripes = int(stripes)
        self.width_px = int(round(dims.length * self.scale)) + 2 * self.margin
        self.height_px = int(round(dims.width * self.scale)) + 2 * self.margin
        self._base = self._draw_pitch()

    # ------------------------------------------------------------------
    def field_to_px(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(round(self.margin + x * self.scale)),
            int(round(self.margin + y * self.scale)),
        )

    def on_pitch(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.dims.length and 0.0 <= y <= self.dims.width

    def blank(self) -> np.ndarray:
        return self._base.copy()

    # ------------------------------------------------------------------
    def _draw_pitch(self) -> np.ndarray:
        d = self.dims
        img = np.full((self.height_px, self.width_px, 3), _PITCH_GREEN, np.uint8)

        # Mown vertical stripes (alternating green shades) over the playfield.
        if self.stripes > 0:
            x0, _ = self.field_to_px(0, 0)
            x1, _ = self.field_to_px(d.length, 0)
            y0, y1 = self.margin, self.height_px - self.margin
            band = (x1 - x0) / float(self.stripes)
            for i in range(self.stripes):
                if i % 2 == 0:
                    bx0 = int(round(x0 + i * band))
                    bx1 = int(round(x0 + (i + 1) * band))
                    img[y0:y1, bx0:bx1] = _PITCH_GREEN_DARK
        # Soft vignette border around the whole minimap.
        cv2.rectangle(img, (1, 1), (self.width_px - 2, self.height_px - 2),
                      (30, 30, 30), 2, cv2.LINE_AA)

        lw = 2
        white = _LINE_WHITE

        def line(p1: Point, p2: Point) -> None:
            cv2.line(img, self.field_to_px(*p1), self.field_to_px(*p2),
                     white, lw, cv2.LINE_AA)

        def box(x1: float, y1: float, x2: float, y2: float) -> None:
            cv2.rectangle(img, self.field_to_px(x1, y1), self.field_to_px(x2, y2),
                          white, lw, cv2.LINE_AA)

        def spot(p: Point) -> None:
            cv2.circle(img, self.field_to_px(*p), 3, white, -1, cv2.LINE_AA)

        cx, cy = d.length / 2.0, d.width / 2.0
        pa_half, ga_half = d.penalty_area_width / 2.0, d.goal_area_width / 2.0
        r_px = int(round(d.center_circle_radius * self.scale))

        # Touchlines + halfway line.
        box(0, 0, d.length, d.width)
        line((cx, 0), (cx, d.width))
        # Center circle + spot.
        cv2.circle(img, self.field_to_px(cx, cy), r_px, white, lw, cv2.LINE_AA)
        spot((cx, cy))
        # Penalty + goal areas.
        box(0, cy - pa_half, d.penalty_area_depth, cy + pa_half)
        box(d.length - d.penalty_area_depth, cy - pa_half, d.length, cy + pa_half)
        box(0, cy - ga_half, d.goal_area_depth, cy + ga_half)
        box(d.length - d.goal_area_depth, cy - ga_half, d.length, cy + ga_half)
        # Penalty spots + arcs (the part of the arc outside the penalty box).
        spot((d.penalty_spot_distance, cy))
        spot((d.length - d.penalty_spot_distance, cy))
        half = math.degrees(math.acos(
            (d.penalty_area_depth - d.penalty_spot_distance)
            / d.center_circle_radius))
        cv2.ellipse(img, self.field_to_px(d.penalty_spot_distance, cy),
                    (r_px, r_px), 0, -half, half, white, lw, cv2.LINE_AA)
        cv2.ellipse(img, self.field_to_px(d.length - d.penalty_spot_distance, cy),
                    (r_px, r_px), 0, 180 - half, 180 + half, white, lw, cv2.LINE_AA)
        # Goals (short stubs behind each goal line).
        goal_half = 7.32 / 2.0
        cv2.rectangle(img, self.field_to_px(-1.6, cy - goal_half),
                      self.field_to_px(0, cy + goal_half), white, lw, cv2.LINE_AA)
        cv2.rectangle(img, self.field_to_px(d.length, cy - goal_half),
                      self.field_to_px(d.length + 1.6, cy + goal_half),
                      white, lw, cv2.LINE_AA)
        return img

    def render(
        self, objects: Sequence[dict], show_track_ids: bool = False
    ) -> np.ndarray:
        """Draw dots for ``objects`` (each: field=(x,y), color, optional ball).

        Colors come straight from each object's ``color`` field, so the
        overlay and the standalone minimap stay pixel-identical. Dots have a
        drop shadow + white outline for a clean, professional look; when
        ``show_track_ids`` is set the id is drawn INSIDE the dot. The color
        is always team/role (or per-track) based, never implied by a label.
        """
        img = self.blank()
        drawable = [o for o in objects if self.on_pitch(*o["field"])]
        if self.hide_referees:
            drawable = [o for o in drawable if o.get("role") != "referee"]
        if self.tactical_style:
            self._draw_team_shapes(img, drawable)
        for obj in drawable:
            x, y = obj["field"]
            cx, cy = self.field_to_px(x, y)
            color = obj.get("color", _DEFAULT_DOT)
            is_ball = obj.get("role") == "ball"
            radius = self.ball_radius if is_ball else self.player_radius

            # Drop shadow, filled dot, crisp white rim.
            cv2.circle(img, (cx + 1, cy + 2), radius, _DOT_SHADOW, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), radius, _DOT_OUTLINE, 2, cv2.LINE_AA)
            if is_ball:
                cv2.circle(img, (cx, cy), max(radius - 3, 1), (0, 0, 0), 1,
                           cv2.LINE_AA)
            elif show_track_ids:
                label = str(obj.get("track_id", ""))
                fs = 0.42 if len(label) < 2 else 0.36
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
                txt_color = (20, 20, 20) if sum(color) > 380 else (245, 245, 245)
                cv2.putText(img, label, (cx - tw // 2, cy + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, txt_color, 1, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------
    def _draw_team_shapes(self, img: np.ndarray, objects: Sequence[dict]) -> None:
        """Per-team tactical overlay: a shape hull + an average vertical line.

        Connects each team's outfield players into a convex 'block' (faint
        fill + coloured outline) and marks the team's average horizontal
        position with a dashed vertical line -- the formation read in the
        reference look. Drawn under the dots so the players stay on top.
        """
        for team in (0, 1):
            pts = [self.field_to_px(*o["field"]) for o in objects
                   if o.get("role") == "player" and o.get("team_id") == team]
            if len(pts) < 3:
                continue
            color = _TEAM_COLORS.get(team, _DEFAULT_DOT)
            arr = np.array(pts, dtype=np.int32)
            hull = cv2.convexHull(arr)
            overlay = img.copy()
            cv2.fillConvexPoly(overlay, hull, color)
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            cv2.polylines(img, [hull], True, color, 2, cv2.LINE_AA)
            self._dashed_vline(img, int(round(arr[:, 0].mean())), color)

    def _dashed_vline(self, img: np.ndarray, x: int, color, seg: int = 9) -> None:
        """A dashed vertical line spanning the pitch (team average-position line)."""
        y0, y1 = self.margin, self.height_px - self.margin
        for y in range(y0, y1, seg * 2):
            cv2.line(img, (x, y), (x, min(y + seg, y1)), color, 2, cv2.LINE_AA)


def _foot_point(bbox: Sequence[float], role: str) -> Point:
    """Image-space ground point: bottom-center for people, center for the ball."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    if role == "ball":
        return cx, (y1 + y2) / 2.0
    return cx, y2


# A foot point projecting more than this many metres outside the pitch is
# not a real position: a planar homography solved from pitch-line keypoints
# becomes unstable far from those points (near the camera's horizon), so a
# far-side player can shoot off to hundreds of metres. Such projections are
# rejected; a projection that lands just outside the line is clamped on.
_SANE_MARGIN_M = 12.0


def _within_margin(x: float, y: float, dims: PitchDimensions, margin: float) -> bool:
    return (
        -margin <= x <= dims.length + margin
        and -margin <= y <= dims.width + margin
    )


def classify_per_frame_teams(
    frames: Sequence,
    source_video: str | Path,
    roles_map: Dict[int, Tuple[str, Optional[int]]],
    ball_class_name: str = "ball",
    smooth_window: int = 7,
    random_seed: int = 42,
) -> Dict[Tuple[int, int], int]:
    """Per-frame team id for every PLAYER box, from its jersey colour.

    This sidesteps ID switches for team COLOURING: a box is coloured by what
    the player is actually wearing in THAT frame, not by the (possibly
    switched) track's per-track label. So when a track drifts from a green
    player to a white one, the box colour follows the shirt, not the id.

    One video pass: crop each player's jersey, build a saturation-weighted
    HSV histogram (same space as team clustering), cluster all of them into
    two teams, align the two clusters to the existing ``roles_map`` team
    convention, then smooth each track's team over a short window so a single
    misread frame can't flicker. Returns ``{(frame_index, track_id): team_id}``
    (empty on failure -> caller falls back to per-track colours).
    """
    import numpy as np

    from role_refinement.appearance_extractor import AppearanceExtractor
    from role_refinement.team_clusterer import _spherical_kmeans, _unit
    from utils.config_loader import RoleRefinementConfig
    from utils.video_io import VideoReader

    extractor = AppearanceExtractor(RoleRefinementConfig(ball_class_name=ball_class_name))
    tracks_by_frame = {f.frame_index: f.tracks for f in frames}
    keys: List[Tuple[int, int]] = []
    vecs: List[np.ndarray] = []
    try:
        with VideoReader(source_video) as reader:
            for index, frame in reader.frames():
                h, w = frame.shape[:2]
                for track in tracks_by_frame.get(index, []):
                    role = roles_map.get(track.track_id, (track.class_name, None))[0]
                    if role != "player":
                        continue
                    x1, y1, x2, y2 = track.bbox
                    x1i, y1i = max(0, int(x1)), max(0, int(y1))
                    x2i, y2i = min(w, int(x2)), min(h, int(y2))
                    if x2i <= x1i or y2i <= y1i:
                        continue
                    vec = extractor.vector_from_crop(frame[y1i:y2i, x1i:x2i])
                    if vec is not None:
                        keys.append((index, track.track_id))
                        vecs.append(vec)
    except Exception:
        logger.exception("Per-frame team classification failed; keeping per-track")
        return {}
    if len(vecs) < 2:
        return {}

    matrix = _unit(np.stack(vecs, axis=0))
    _centroids, labels = _spherical_kmeans(matrix, k=2, seed=random_seed)
    raw = {key: int(lbl) for key, lbl in zip(keys, labels)}
    raw = _align_clusters_to_roles(raw, roles_map)
    return _smooth_team_timeline(raw, smooth_window)


def _align_clusters_to_roles(
    raw: Dict[Tuple[int, int], int],
    roles_map: Dict[int, Tuple[str, Optional[int]]],
) -> Dict[Tuple[int, int], int]:
    """Map cluster indices 0/1 onto the roles_map team convention by vote."""
    from collections import Counter

    votes = {0: Counter(), 1: Counter()}
    for (_f, tid), cluster in raw.items():
        team = roles_map.get(tid, (None, None))[1]
        if team in (0, 1):
            votes[cluster][team] += 1
    map0 = votes[0].most_common(1)[0][0] if votes[0] else 0
    map1 = votes[1].most_common(1)[0][0] if votes[1] else 1
    if map0 == map1:                          # degenerate vote -> identity
        map0, map1 = 0, 1
    mapping = {0: map0, 1: map1}
    return {key: mapping[c] for key, c in raw.items()}


def _smooth_team_timeline(
    raw: Dict[Tuple[int, int], int], window: int
) -> Dict[Tuple[int, int], int]:
    """Sliding-window majority of each track's per-frame team (kills flicker)."""
    from collections import Counter, defaultdict

    by_track: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for (frame_index, tid), team in raw.items():
        by_track[tid].append((frame_index, team))
    out: Dict[Tuple[int, int], int] = {}
    for tid, seq in by_track.items():
        seq.sort()
        teams = [t for _f, t in seq]
        for i, (frame_index, _t) in enumerate(seq):
            lo, hi = max(0, i - window), min(len(teams), i + window + 1)
            out[(frame_index, tid)] = Counter(teams[lo:hi]).most_common(1)[0][0]
    return out


def project_tracks_to_field(
    frames: Sequence,
    roles_map: Dict[int, Tuple[str, Optional[int]]],
    homography: Homography,
    renderer: Optional[MinimapRenderer] = None,
    color_mode: str = COLOR_MODE_TEAM_ROLE,
    smooth_alpha: float = 0.5,
    max_step_m: float = 3.0,
    smooth_gap: int = 3,
    team_override: Optional[Dict[Tuple[int, int], int]] = None,
    names_map: Optional[Dict[int, str]] = None,
) -> List[dict]:
    """Project every track on every frame to field coordinates.

    ``frames`` is any iterable of objects exposing ``frame_index`` and
    ``tracks`` (each track with ``track_id``, ``class_name``, ``bbox``).
    ``roles_map`` (``track_id -> (final_role, team_id)``) drives the
    analytics coloring; ``color_mode="track_id"`` switches to per-track
    debug colors. ``homography`` may be a single :class:`Homography` or a
    per-frame selector (anything with ``for_frame``, e.g.
    :class:`MultiHomography`) to follow a panning camera.

    Per-frame homographies jitter, which can teleport a dot across the pitch
    for a frame. ``max_step_m`` clamps the per-frame field movement to a
    physically plausible step and ``smooth_alpha`` applies an exponential
    moving average, so dots glide instead of flickering (continuity is only
    kept across gaps up to ``smooth_gap`` frames; the ball isn't smoothed).
    Returns ``{"frame", "objects":[...]}`` dicts.
    """
    has_selector = hasattr(homography, "for_frame")
    dims = renderer.dims if renderer is not None else PitchDimensions()
    # track_id -> (smoothed_x, smoothed_y, last_frame)
    prev: Dict[int, Tuple[float, float, int]] = {}
    out: List[dict] = []
    for frame in frames:
        frame_h = (
            homography.for_frame(frame.frame_index) if has_selector else homography
        )
        objects = []
        for track in frame.tracks:
            role, team_id = roles_map.get(track.track_id, (track.class_name, None))
            if team_override is not None and role == "player":
                ov = team_override.get((frame.frame_index, track.track_id))
                if ov is not None:
                    team_id = ov          # colour by this frame's jersey, not the id
            fx, fy = frame_h.project_image_to_field(_foot_point(track.bbox, role))
            if not np.isfinite(fx) or not np.isfinite(fy):
                continue

            if role != "ball":
                # Reject horizon blow-ups: a far-outside projection is the
                # homography misfiring, not a real position. Hold the last
                # good spot if it's recent, otherwise drop this track here.
                if not _within_margin(fx, fy, dims, _SANE_MARGIN_M):
                    held = prev.get(track.track_id)
                    if held is not None and 0 < frame.frame_index - held[2] <= smooth_gap:
                        fx, fy = held[0], held[1]
                    else:
                        continue
                fx, fy = _smooth_position(
                    prev, track.track_id, frame.frame_index, fx, fy,
                    smooth_alpha, max_step_m, smooth_gap)
                # Pull a player who lands just outside onto the line so they
                # still show on the minimap.
                fx = min(max(fx, 0.0), dims.length)
                fy = min(max(fy, 0.0), dims.width)

            entry = {
                "track_id": int(track.track_id),
                "role": role,
                "team_id": team_id,
                "field": [round(float(fx), 3), round(float(fy), 3)],
                "color": color_for_object(role, team_id, track.track_id, color_mode),
            }
            name = names_map.get(track.track_id) if names_map else None
            if role != "ball":
                entry["player_display_name"] = name or f"Player {track.track_id}"
            if name:
                entry["player_name"] = name
            if renderer is not None:
                entry["on_pitch"] = renderer.on_pitch(fx, fy)
            objects.append(entry)
        out.append({"frame": frame.frame_index, "objects": objects})
    return out


def _smooth_position(prev, track_id, frame_index, fx, fy,
                     alpha, max_step_m, smooth_gap):
    """Clamp a teleporting projection and EMA-smooth it for a stable dot."""
    p = prev.get(track_id)
    if p is not None and 0 < frame_index - p[2] <= smooth_gap:
        px, py, _ = p
        dx, dy = fx - px, fy - py
        dist = math.hypot(dx, dy)
        if dist > max_step_m and dist > 0:           # reject the teleport
            scale = max_step_m / dist
            fx, fy = px + dx * scale, py + dy * scale
        fx = alpha * fx + (1.0 - alpha) * px         # ease toward measurement
        fy = alpha * fy + (1.0 - alpha) * py
    prev[track_id] = (fx, fy, frame_index)
    return fx, fy


def write_field_positions(
    positions: List[dict], path: str | Path, metadata: Optional[dict] = None
) -> Path:
    """Write the per-frame field-positions JSON (colors stripped)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = [
        {
            "frame": entry["frame"],
            "objects": [
                {k: v for k, v in obj.items() if k != "color"}
                for obj in entry["objects"]
            ],
        }
        for entry in positions
    ]
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata or {}, "frames": clean}, handle, indent=2)
    logger.info("Wrote field positions (%d frames) to %s", len(clean), path)
    return path


def render_minimap_video(
    positions: List[dict],
    renderer: MinimapRenderer,
    out_path: str | Path,
    fps: float = 25.0,
    show_track_ids: bool = False,
) -> Path:
    """Render the projected positions to a top-down minimap video."""
    from utils.video_io import VideoWriter

    out_path = Path(out_path)
    writer = VideoWriter(
        out_path, fps=fps, frame_size=(renderer.width_px, renderer.height_px)
    )
    try:
        for entry in positions:
            writer.write(renderer.render(entry["objects"], show_track_ids))
    finally:
        writer.release()
    logger.info("Wrote minimap video to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Picture-in-picture overlay (minimap composited into the main video)
# ---------------------------------------------------------------------------
OVERLAY_POSITIONS = ("top_right", "top_left", "bottom_right", "bottom_left")
_OVERLAY_PAD = 8       # padding between the minimap and its dark background


def _placement(
    position: str, width: int, height: int, box_w: int, box_h: int, margin: int
) -> Tuple[int, int]:
    """Top-left pixel of the overlay box for a named corner."""
    if position == "top_left":
        x, y = margin, margin
    elif position == "bottom_left":
        x, y = margin, height - box_h - margin
    elif position == "bottom_right":
        x, y = width - box_w - margin, height - box_h - margin
    else:  # top_right (default) — avoids the usual top-left score bug
        x, y = width - box_w - margin, margin
    return max(0, x), max(0, y)


def overlay_geometry(
    base_w: int,
    base_h: int,
    mini_w: int,
    mini_h: int,
    position: str = "top_right",
    width_ratio: float = 0.28,
    margin: int = 20,
    pad: int = _OVERLAY_PAD,
) -> dict:
    """Compute the overlay box + inner-minimap rectangle (pure geometry).

    The minimap is scaled to ``width_ratio`` of the base width (aspect
    preserved), clamped so the padded box always fits inside the frame.
    """
    inner_w = int(round(base_w * width_ratio))
    inner_w = min(max(inner_w, 8), max(8, base_w - 2 * margin - 2 * pad))
    inner_h = int(round(inner_w * mini_h / mini_w))
    max_inner_h = max(8, base_h - 2 * margin - 2 * pad)
    if inner_h > max_inner_h:
        inner_h = max_inner_h
        inner_w = min(inner_w, int(round(inner_h * mini_w / mini_h)))
    box_w = inner_w + 2 * pad
    box_h = inner_h + 2 * pad
    x, y = _placement(position, base_w, base_h, box_w, box_h, margin)
    return {
        "x": x, "y": y, "box_w": box_w, "box_h": box_h,
        "inner_w": inner_w, "inner_h": inner_h, "pad": pad,
    }


def overlay_minimap(
    base: np.ndarray,
    minimap: np.ndarray,
    position: str = "top_right",
    width_ratio: float = 0.28,
    margin: int = 20,
    background_alpha: float = 0.65,
    border: bool = True,
    pad: int = _OVERLAY_PAD,
) -> np.ndarray:
    """Composite ``minimap`` onto a copy of ``base`` as a picture-in-picture.

    Draws a semi-transparent dark backing, the (resized) minimap, and an
    optional thin border. ``base`` is not mutated.
    """
    h, w = base.shape[:2]
    mh, mw = minimap.shape[:2]
    g = overlay_geometry(w, h, mw, mh, position, width_ratio, margin, pad)
    x, y = g["x"], g["y"]
    box_w = min(g["box_w"], w - x)
    box_h = min(g["box_h"], h - y)
    if box_w <= 0 or box_h <= 0:
        return base.copy()

    out = base.copy()
    # Semi-transparent dark background behind the minimap.
    roi = out[y:y + box_h, x:x + box_w]
    dark = np.zeros_like(roi)
    out[y:y + box_h, x:x + box_w] = cv2.addWeighted(
        roi, 1.0 - background_alpha, dark, background_alpha, 0.0)

    inner_w = min(g["inner_w"], box_w - 2 * pad)
    inner_h = min(g["inner_h"], box_h - 2 * pad)
    if inner_w > 0 and inner_h > 0:
        mini = cv2.resize(minimap, (inner_w, inner_h))
        out[y + pad:y + pad + inner_h, x + pad:x + pad + inner_w] = mini
    if border:
        cv2.rectangle(out, (x, y), (x + box_w - 1, y + box_h - 1),
                      (255, 255, 255), 2)
    return out


def render_overlay_video(
    base_video_path: str | Path,
    positions: List[dict],
    renderer: MinimapRenderer,
    out_path: str | Path,
    position: str = "top_right",
    width_ratio: float = 0.28,
    margin: int = 20,
    background_alpha: float = 0.65,
    border: bool = True,
    fps: Optional[float] = None,
    show_track_ids: bool = False,
) -> Path:
    """Overlay the per-frame minimap onto the main annotated video.

    Reads ``base_video_path`` frame by frame and composites the minimap
    rendered from the aligned ``positions`` entry, writing
    ``<...>_final_with_minimap.mp4``. The minimap is rendered with the same
    :meth:`MinimapRenderer.render` as the standalone video, so the dot
    colors are identical.
    """
    from utils.video_io import VideoReader, VideoWriter

    out_path = Path(out_path)
    with VideoReader(base_video_path) as reader:
        writer = VideoWriter(
            out_path,
            fps=float(fps) if fps else reader.fps,
            frame_size=(reader.width, reader.height),
        )
        try:
            for index, frame in reader.frames():   # index is 1-based
                objects = (
                    positions[index - 1]["objects"]
                    if 0 <= index - 1 < len(positions) else []
                )
                mini = renderer.render(objects, show_track_ids)
                writer.write(overlay_minimap(
                    frame, mini, position=position, width_ratio=width_ratio,
                    margin=margin, background_alpha=background_alpha,
                    border=border))
        finally:
            writer.release()
    logger.info("Wrote minimap-overlay video to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Combined final video: team/role-colored player boxes + minimap overlay
# ---------------------------------------------------------------------------
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _track_label(
    role: str, team_id: Optional[int], track_id: int, name: Optional[str] = None
) -> str:
    # A manually-entered name (if any) shows next to the id; otherwise the
    # label reflects the ROLE so referees/keepers are never shown as "Player".
    if name:
        head = f"{name} (#{track_id})"
    elif role == "ball":
        head = f"#{track_id} ball"
    elif role == "referee":
        head = f"Ref #{track_id}"
    elif role == "goalkeeper":
        head = f"GK #{track_id}"
    else:
        head = f"Player {track_id}"
    label = head
    if role in ("player", "goalkeeper") and team_id is not None:
        label += f" T{team_id}"
    return label


def _draw_track_box(
    canvas: np.ndarray,
    bbox: Sequence[float],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
    font_scale: float = 0.5,
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    (tw, th), bl = cv2.getTextSize(label, _FONT, font_scale, 1)
    label_y = y1 - 4
    if label_y - th - bl < 0:
        label_y = y2 + th + bl + 4
    cv2.rectangle(canvas, (x1, label_y - th - bl), (x1 + tw + 4, label_y + 2),
                  color, cv2.FILLED)
    cv2.putText(canvas, label, (x1 + 2, label_y - bl + 2), _FONT, font_scale,
                (0, 0, 0), 1, cv2.LINE_AA)


def render_final_video(
    source_video: str | Path,
    frames: Sequence,
    positions: List[dict],
    roles_map: Dict[int, Tuple[str, Optional[int]]],
    renderer: MinimapRenderer,
    out_path: str | Path,
    color_mode: str = COLOR_MODE_TEAM_ROLE,
    show_track_ids: bool = False,
    position: str = "top_right",
    width_ratio: float = 0.28,
    margin: int = 20,
    background_alpha: float = 0.65,
    border: bool = True,
    fps: Optional[float] = None,
    possession_cum: Optional[Dict[int, Tuple[float, float]]] = None,
    pose_estimator=None,
    pose_min_conf: float = 0.5,
    pose_box_fallback: bool = True,
    team_override: Optional[Dict[Tuple[int, int], int]] = None,
    speed_lookup: Optional[Dict[Tuple[int, int], Tuple[float, float]]] = None,
    names_map: Optional[Dict[int, str]] = None,
) -> Path:
    """One combined output: team/role-colored boxes + labels + minimap PiP.

    Reads the ORIGINAL ``source_video`` and draws each track's box with the
    SAME team/role color as its minimap dot (so players, labels and the
    minimap all agree), then composites the top-down minimap. ``roles_map``
    carries the final (corrected) roles, so corrections are reflected too.
    When ``possession_cum`` (``frame -> (pct0, pct1)``) is given, a running
    possession bar is drawn at the top.
    """
    from utils.video_io import VideoReader, VideoWriter

    out_path = Path(out_path)
    tracks_by_frame = {f.frame_index: f.tracks for f in frames}
    pos_by_frame = {p["frame"]: p["objects"] for p in positions}
    last_pct = (50.0, 50.0)
    with VideoReader(source_video) as reader:
        writer = VideoWriter(
            out_path,
            fps=float(fps) if fps else reader.fps,
            frame_size=(reader.width, reader.height),
        )
        try:
            for index, frame in reader.frames():   # index is 1-based
                canvas = frame.copy()
                tracks = tracks_by_frame.get(index, [])
                assigned = {}
                if pose_estimator is not None:
                    from visualization.pose_overlay import draw_pose
                    assigned = pose_estimator.detect_on_tracks(frame, tracks)
                for track in tracks:
                    role, team_id = roles_map.get(
                        track.track_id, (track.class_name, None))
                    if team_override is not None and role == "player":
                        ov = team_override.get((index, track.track_id))
                        if ov is not None:
                            team_id = ov     # colour by this frame's shirt
                    color = color_for_object(role, team_id, track.track_id, color_mode)
                    name = names_map.get(track.track_id) if names_map else None
                    label = _track_label(role, team_id, track.track_id, name)
                    if track.track_id in assigned:
                        draw_pose(canvas, assigned[track.track_id], color,
                                  min_conf=pose_min_conf)
                        x1, y1, _x2, _y2 = (int(round(v)) for v in track.bbox)
                        cv2.putText(
                            canvas, label,
                            (x1, max(y1 - 4, 10)), _FONT, 0.45, color, 1, cv2.LINE_AA)
                    elif pose_estimator is None or pose_box_fallback:
                        _draw_track_box(canvas, track.bbox, label, color)
                    else:
                        # Skeleton-only: a player the pose model missed gets a
                        # small dot (team colour) instead of a box, so nobody
                        # vanishes from the frame.
                        x1, y1, x2, y2 = track.bbox
                        fcx, fcy = int((x1 + x2) / 2), int(y2)
                        cv2.circle(canvas, (fcx, fcy), 5, color, -1, cv2.LINE_AA)
                        cv2.circle(canvas, (fcx, fcy), 5, (255, 255, 255), 1,
                                   cv2.LINE_AA)
                if speed_lookup is not None:
                    from analytics import draw_player_stat_bars
                    draw_player_stat_bars(
                        canvas, tracks, speed_lookup, index, names_map=names_map)
                mini = renderer.render(pos_by_frame.get(index, []), show_track_ids)
                canvas = overlay_minimap(
                    canvas, mini, position=position, width_ratio=width_ratio,
                    margin=margin, background_alpha=background_alpha, border=border)
                if possession_cum is not None:
                    last_pct = possession_cum.get(index, last_pct)
                    from analytics.possession import draw_possession_bar
                    draw_possession_bar(canvas, last_pct[0], last_pct[1],
                                        color0=_TEAM_COLORS[0], color1=_TEAM_COLORS[1])
                writer.write(canvas)
        finally:
            writer.release()
    logger.info("Wrote combined final video to %s", out_path)
    return out_path


def load_roles_map(path: str | Path) -> Dict[int, Tuple[str, Optional[int]]]:
    """Build ``track_id -> (role, team_id)`` from a roles or final-roles JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    roles: Dict[int, Tuple[str, Optional[int]]] = {}
    for track in data.get("tracks", []):
        role = (
            track.get("final_role")
            or track.get("refined_role")
            or track.get("role")
            or "unknown"
        )
        roles[int(track["track_id"])] = (role, track.get("team_id"))
    return roles


def load_player_names(path: str | Path) -> Dict[int, str]:
    """Build ``track_id -> player_name`` from a final-roles JSON (manual names)."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    names: Dict[int, str] = {}
    for track in data.get("tracks", []):
        name = track.get("player_name")
        if name:
            names[int(track["track_id"])] = str(name)
    return names
