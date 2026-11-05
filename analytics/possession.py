"""Ball possession per team (Phase 6 — analytics).

Possession is decided in IMAGE space — it does NOT need the homography:
each frame the ball is attributed to the nearest player (within a gate
scaled by the player's size), and that player's team holds possession.
A "last-touch" model carries possession through loose/airborne frames,
and a small hysteresis avoids flicker when the ball passes between
players. The output is a per-frame timeline + overall percentages.

Consumes only the exported tracks JSON + a roles map (team per track), so
it stays a pure post-processing layer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from utils.logger import get_logger

logger = get_logger("analytics.possession")

Point = Tuple[float, float]
_POSSESSING_ROLES = ("player", "goalkeeper")


@dataclass
class PossessionResult:
    timeline: List[dict]                       # [{"frame", "team"}] (team 0/1/None)
    counts: Dict[int, int]                     # frames held per team
    percentages: Dict[int, float]              # share of possessed frames
    possessed_frames: int
    total_frames: int
    cumulative: List[Tuple[int, float, float]] = field(default_factory=list)
    # per-frame (frame, cum_pct_team0, cum_pct_team1) for a running bar


def _foot(bbox: Sequence[float]) -> Point:
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def frame_possessor(
    frame,
    roles_map: Dict[int, Tuple[str, Optional[int]]],
    ball_class_name: str = "ball",
    gate_factor: float = 2.0,
) -> Optional[Tuple[int, int]]:
    """Return ``(team, player_id)`` holding the ball this frame, or ``None``.

    The nearest player to the ball is the holder, provided the ball is
    within ``gate_factor`` × the player's bbox height of their feet (a
    camera-zoom-invariant gate). Frames with no ball, or a loose ball far
    from everyone, return ``None``.
    """
    ball_center: Optional[Point] = None
    players: List[Tuple[int, "object"]] = []
    for track in frame.tracks:
        role, team = roles_map.get(track.track_id, (track.class_name, None))
        if role == ball_class_name:
            x1, y1, x2, y2 = track.bbox
            ball_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        elif role in _POSSESSING_ROLES and team in (0, 1):
            players.append((team, track))
    if ball_center is None or not players:
        return None

    best: Optional[Tuple[int, int]] = None
    best_dist = float("inf")
    for team, track in players:
        foot = _foot(track.bbox)
        height = max(track.bbox[3] - track.bbox[1], 1.0)
        dist = math.hypot(ball_center[0] - foot[0], ball_center[1] - foot[1])
        if dist <= gate_factor * height and dist < best_dist:
            best_dist, best = dist, (team, int(track.track_id))
    return best


def smooth_possession(
    raw_teams: Sequence[Optional[int]], min_hold: int = 3
) -> List[Optional[int]]:
    """Last-touch model + hysteresis.

    Possession stays with the current team until another team is the nearest
    holder for ``min_hold`` consecutive frames (so a single frame where the
    ball drifts past an opponent doesn't flip it). Loose-ball (``None``)
    frames inherit the current holder.
    """
    current: Optional[int] = None
    pending: Optional[int] = None
    pending_count = 0
    out: List[Optional[int]] = []
    for team in raw_teams:
        if team is None:
            out.append(current)            # loose ball -> keep last holder
            continue
        if current is None:
            current = team                 # first possession establishes at once
        elif team != current:
            # Switching from an established holder needs a few held frames.
            if team == pending:
                pending_count += 1
            else:
                pending, pending_count = team, 1
            if pending_count >= min_hold:
                current, pending, pending_count = team, None, 0
        else:                              # team == current
            pending, pending_count = None, 0
        out.append(current)
    return out


def compute_possession(
    frames: Sequence,
    roles_map: Dict[int, Tuple[str, Optional[int]]],
    ball_class_name: str = "ball",
    gate_factor: float = 2.0,
    min_hold: int = 3,
) -> PossessionResult:
    """Per-team possession over a whole match."""
    raw = [frame_possessor(f, roles_map, ball_class_name, gate_factor)
           for f in frames]
    smoothed = smooth_possession([p[0] if p else None for p in raw], min_hold)

    counts = {0: 0, 1: 0}
    timeline: List[dict] = []
    cumulative: List[Tuple[int, float, float]] = []
    c0 = c1 = 0
    for frame, team in zip(frames, smoothed):
        if team in (0, 1):
            counts[team] += 1
            if team == 0:
                c0 += 1
            else:
                c1 += 1
        total = c0 + c1
        timeline.append({"frame": frame.frame_index, "team": team})
        cumulative.append((
            frame.frame_index,
            100.0 * c0 / total if total else 0.0,
            100.0 * c1 / total if total else 0.0,
        ))

    possessed = counts[0] + counts[1]
    percentages = {
        0: round(100.0 * counts[0] / possessed, 1) if possessed else 0.0,
        1: round(100.0 * counts[1] / possessed, 1) if possessed else 0.0,
    }
    return PossessionResult(
        timeline=timeline,
        counts=counts,
        percentages=percentages,
        possessed_frames=possessed,
        total_frames=len(frames),
        cumulative=cumulative,
    )


def export_possession(
    result: PossessionResult, path: str | Path, metadata: Optional[dict] = None
) -> Path:
    """Write the possession summary + per-frame timeline JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "possession_percent": result.percentages,
        "frames_held": result.counts,
        "possessed_frames": result.possessed_frames,
        "total_frames": result.total_frames,
        "timeline": result.timeline,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Wrote possession (%.1f%% / %.1f%%) to %s",
                result.percentages[0], result.percentages[1], path)
    return path


def export_possession_txt(
    result: PossessionResult, path: str | Path, title: str = ""
) -> Path:
    """Write a human-readable possession report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = f"BALL POSSESSION{' — ' + title if title else ''}"
    lines = [
        head, "=" * max(40, len(head)), "",
        f"  Team 0 : {result.percentages.get(0, 0.0):5.1f} %  "
        f"({result.counts.get(0, 0)} frames)",
        f"  Team 1 : {result.percentages.get(1, 0.0):5.1f} %  "
        f"({result.counts.get(1, 0)} frames)",
        "",
        f"  possessed frames: {result.possessed_frames} / {result.total_frames}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote possession report to %s", path)
    return path


def draw_possession_bar(
    frame,
    pct0: float,
    pct1: float,
    color0: Tuple[int, int, int] = (255, 128, 0),
    color1: Tuple[int, int, int] = (0, 128, 255),
    width_ratio: float = 0.42,
    height: int = 30,
    top: int = 10,
):
    """Draw a running possession bar (team-colored) at the top-center."""
    import cv2

    h, w = frame.shape[:2]
    bar_w = int(w * width_ratio)
    x0 = (w - bar_w) // 2
    split = x0 + int(bar_w * pct0 / 100.0)
    y1 = top + height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, top), (split, y1), color0, -1)
    cv2.rectangle(overlay, (split, top), (x0 + bar_w, y1), color1, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (x0, top), (x0 + bar_w, y1), (255, 255, 255), 1, cv2.LINE_AA)

    f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{pct0:.0f}%", (x0 + 8, y1 - 8), f, 0.6, (255, 255, 255),
                2, cv2.LINE_AA)
    t1 = f"{pct1:.0f}%"
    (tw, _), _ = cv2.getTextSize(t1, f, 0.6, 2)
    cv2.putText(frame, t1, (x0 + bar_w - tw - 8, y1 - 8), f, 0.6, (255, 255, 255),
                2, cv2.LINE_AA)
    label = "POSSESSION"
    (lw, _), _ = cv2.getTextSize(label, f, 0.4, 1)
    cv2.putText(frame, label, (x0 + (bar_w - lw) // 2, top - 3), f, 0.4,
                (235, 235, 235), 1, cv2.LINE_AA)
    return frame
