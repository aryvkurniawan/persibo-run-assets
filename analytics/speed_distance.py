"""Phase 5 — speed & distance from field (metre) coordinates.

Pure computation over per-track field trajectories:

* speed between consecutive valid positions: ``d / dt`` (m/s, x3.6 -> km/h),
* a moving-average smoother over the per-frame speed series,
* outlier rejection: a step faster than ``max_speed_kmh`` is an impossible
  jump (homography blip / id swap) -- it is counted as an ``invalid_jump``
  and NOT added to the distance,
* per-track distance / average / max, then team & role distance totals.

Average speed is distance over the *valid moving time* (so standing still
lowers it correctly); max speed is taken from the SMOOTHED series so a single
noisy frame can't inflate it. All functions are I/O-free and unit-tested.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from analytics.analytics_models import AnalyticsResult, TrackAnalytics
from utils.logger import get_logger

logger = get_logger("analytics.speed_distance")

Sample = Tuple[int, float, float]      # (frame_index, x_m, y_m)


def moving_average(values: Sequence[float], window: int) -> List[float]:
    """Centered moving average; window<=1 (or <2 values) returns a copy."""
    vals = list(values)
    if window <= 1 or len(vals) < 2:
        return vals
    half = window // 2
    n = len(vals)
    out: List[float] = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(sum(vals[lo:hi]) / (hi - lo))
    return out


def compute_track_analytics(
    track_id: int,
    role: str,
    team_id: Optional[int],
    samples: Sequence[Sample],
    fps: float,
    smoothing_window: int = 5,
    max_speed_kmh: float = 38.0,
    min_samples: int = 5,
    player_name: Optional[str] = None,
) -> TrackAnalytics:
    """Distance / speed for one track from its ``(frame, x, y)`` samples."""
    result = TrackAnalytics(
        track_id=track_id, role=role, team_id=team_id, player_name=player_name)
    pts = sorted(samples, key=lambda s: s[0])
    fps = float(fps) if fps and fps > 0 else 25.0

    raw: List[Tuple[int, float]] = []      # (frame, speed_kmh) of valid steps
    moving_time = 0.0
    for i in range(1, len(pts)):
        f0, x0, y0 = pts[i - 1]
        f1, x1, y1 = pts[i]
        df = f1 - f0
        if df <= 0:
            continue
        dt = df / fps
        dist = math.hypot(x1 - x0, y1 - y0)
        speed_kmh = (dist / dt) * 3.6
        if speed_kmh > max_speed_kmh:          # impossible jump -> reject
            result.invalid_jumps += 1
            continue
        result.total_distance_m += dist
        result.valid_samples += 1
        moving_time += dt
        raw.append((f1, speed_kmh))
        result.frame_distance_m[f1] = round(result.total_distance_m, 1)

    if raw:
        speeds = [s for _f, s in raw]
        smoothed = moving_average(speeds, smoothing_window)
        result.frame_speed_kmh = {
            f: round(float(s), 2) for (f, _), s in zip(raw, smoothed)}
        result.avg_speed_kmh = (
            (result.total_distance_m / moving_time) * 3.6 if moving_time > 0 else 0.0)
        result.max_speed_kmh = max(smoothed)
    result.insufficient_data = result.valid_samples < min_samples
    return result


def compute_analytics(
    frames: Sequence[dict],
    roles_map: Dict[int, Tuple[str, Optional[int]]],
    fps: float,
    smoothing_window: int = 5,
    max_speed_kmh: float = 38.0,
    min_samples: int = 5,
    ball_class_name: str = "ball",
    on_pitch_only: bool = True,
    exclude_roles: Optional[Sequence[str]] = ("referee", "unknown"),
    min_track_samples: int = 30,
    player_names: Optional[Dict[int, str]] = None,
) -> AnalyticsResult:
    """Build per-track + team + role analytics from field-position frames.

    ``frames`` is the ``field_positions`` structure: ``[{"frame", "objects":
    [{"track_id","role","team_id","field":[x,y],"on_pitch"}]}]``. Roles/teams
    come from ``roles_map`` when present (authoritative, reflects manual
    corrections), else from the embedded values. Only on-pitch, finite
    positions are used (off-pitch points are homography blow-ups).
    """
    samples: Dict[int, List[Sample]] = defaultdict(list)
    embedded: Dict[int, Tuple[str, Optional[int]]] = {}
    for frame in frames:
        f_idx = int(frame.get("frame", 0))
        for obj in frame.get("objects", []):
            role = obj.get("role")
            if role == ball_class_name:
                continue
            if on_pitch_only and not obj.get("on_pitch", True):
                continue
            fx, fy = obj.get("field", (float("nan"), float("nan")))
            if not (math.isfinite(fx) and math.isfinite(fy)):
                continue
            tid = int(obj["track_id"])
            samples[tid].append((f_idx, float(fx), float(fy)))
            embedded[tid] = (role, obj.get("team_id"))

    skip = {r.lower() for r in (exclude_roles or ())}
    # Authoritative role/team per track; gather team-0/1 player points so a
    # goalkeeper (no team of its own) can be assigned to the nearer team.
    track_meta: Dict[int, Tuple[str, Optional[int]]] = {}
    team_points: Dict[int, List[Tuple[float, float]]] = {0: [], 1: []}
    for tid, pts in samples.items():
        role, team_id = roles_map.get(tid) or embedded.get(tid, ("player", None))
        track_meta[tid] = (role, team_id)
        if role == "player" and team_id in (0, 1):
            team_points[team_id].extend((x, y) for _f, x, y in pts)
    team_centroid = {
        t: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        for t, pts in team_points.items() if pts
    }

    result = AnalyticsResult(fps=float(fps))
    for tid, pts in samples.items():
        role, team_id = track_meta[tid]
        if role and role.lower() in skip:          # e.g. referees / unknowns
            continue
        if role == "goalkeeper" and team_id is None:
            team_id = _nearest_team(pts, team_centroid)
        track = compute_track_analytics(
            tid, role, team_id, pts, fps,
            smoothing_window, max_speed_kmh, min_samples,
            player_name=(player_names or {}).get(tid))
        if track.valid_samples < min_track_samples:   # drop noise fragments
            continue
        result.tracks.append(track)
        result.total_distance_m += track.total_distance_m
        result.distance_by_team[team_id] = (
            result.distance_by_team.get(team_id, 0.0) + track.total_distance_m)
        result.distance_by_role[role] = (
            result.distance_by_role.get(role, 0.0) + track.total_distance_m)

    result.tracks.sort(key=lambda t: t.track_id)
    logger.info(
        "Analytics: %d tracks, total distance %.0f m",
        len(result.tracks), result.total_distance_m)
    return result


def _nearest_team(
    samples: Sequence[Sample], team_centroid: Dict[int, Tuple[float, float]]
) -> Optional[int]:
    """Team whose player centroid is closest to this track's mean position.

    A keeper sits at its own goal, where its team's centroid (the defending
    block) is nearer than the opponent's (pushed up the pitch), so the nearest
    centroid is the keeper's own side.
    """
    if not team_centroid:
        return None
    if len(team_centroid) == 1:
        return next(iter(team_centroid))
    gx = sum(s[1] for s in samples) / len(samples)
    gy = sum(s[2] for s in samples) / len(samples)
    return min(team_centroid,
               key=lambda t: math.hypot(gx - team_centroid[t][0],
                                        gy - team_centroid[t][1]))
