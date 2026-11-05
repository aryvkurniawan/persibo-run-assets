"""Phase 9 — ball possession from the PROJECTED (field-metre) positions.

A field-space possession estimate: for every frame, find the player nearest
to the ball in pitch metres and credit that player's team with the frame,
then aggregate to team percentages with a timeline and the dominant team.
This is more faithful than an image-space distance (no perspective bias),
since it works on the homography-projected coordinates.

Reads only the ``field_positions`` structure (each object has ``role``,
``team_id``, ``field``, ``on_pitch``). It does NOT touch the tracker or the
existing image-space possession module.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from utils.logger import get_logger

logger = get_logger("analytics.possession_engine")

Point = Tuple[float, float]


@dataclass
class PossessionAnalysis:
    """Field-space possession outcome."""

    percentages: Dict[int, float]                 # team -> %
    counts: Dict[int, int]                        # team -> frames credited
    possessed_frames: int
    total_frames: int
    dominant_team: Optional[int]
    timeline: List[Tuple[int, Optional[int]]] = field(default_factory=list)


def _ball_point(objects: Sequence[dict], ball_class_name: str) -> Optional[Point]:
    for obj in objects:
        if obj.get("role") == ball_class_name:
            x, y = obj.get("field", (float("nan"), float("nan")))
            if math.isfinite(x) and math.isfinite(y):
                return (float(x), float(y))
    return None


def frame_owner(
    objects: Sequence[dict],
    ball_class_name: str = "ball",
    gate_m: Optional[float] = None,
    on_pitch_only: bool = True,
) -> Optional[int]:
    """Team of the player nearest the ball this frame, or ``None``.

    Returns ``None`` when the ball is missing/off-pitch, no eligible player is
    on the field, or (when ``gate_m`` is set) the nearest player is further
    than ``gate_m`` metres from the ball (a loose ball).
    """
    ball = _ball_point(objects, ball_class_name)
    if ball is None:
        return None
    best_team: Optional[int] = None
    best_dist = float("inf")
    for obj in objects:
        role = obj.get("role")
        team = obj.get("team_id")
        if role == ball_class_name or team not in (0, 1):
            continue
        if on_pitch_only and not obj.get("on_pitch", True):
            continue
        x, y = obj.get("field", (float("nan"), float("nan")))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        dist = math.hypot(x - ball[0], y - ball[1])
        if dist < best_dist:
            best_dist, best_team = dist, int(team)
    if best_team is None:
        return None
    if gate_m is not None and best_dist > gate_m:
        return None
    return best_team


def compute_field_possession(
    frames: Sequence[dict],
    ball_class_name: str = "ball",
    gate_m: Optional[float] = None,
) -> PossessionAnalysis:
    """Aggregate per-frame nearest-player ownership into team possession."""
    counts = {0: 0, 1: 0}
    timeline: List[Tuple[int, Optional[int]]] = []
    for frame in frames:
        owner = frame_owner(frame.get("objects", []), ball_class_name, gate_m)
        timeline.append((int(frame.get("frame", 0)), owner))
        if owner in (0, 1):
            counts[owner] += 1

    possessed = counts[0] + counts[1]
    percentages = {
        0: round(100.0 * counts[0] / possessed, 1) if possessed else 0.0,
        1: round(100.0 * counts[1] / possessed, 1) if possessed else 0.0,
    }
    if possessed == 0:
        dominant: Optional[int] = None
    else:
        dominant = 0 if counts[0] >= counts[1] else 1

    logger.info("Field possession: team0 %.1f%% / team1 %.1f%% (%d/%d frames)",
                percentages[0], percentages[1], possessed, len(timeline))
    return PossessionAnalysis(
        percentages=percentages,
        counts=counts,
        possessed_frames=possessed,
        total_frames=len(timeline),
        dominant_team=dominant,
        timeline=timeline,
    )


def export_field_possession(
    result: PossessionAnalysis,
    path: str | Path,
    metadata: Optional[dict] = None,
) -> Path:
    """Write the possession JSON (spec format + timeline + dominant team)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {"phase": "ball_possession", **(metadata or {})},
        "team_0_possession": result.percentages.get(0, 0.0),
        "team_1_possession": result.percentages.get(1, 0.0),
        "dominant_team": result.dominant_team,
        "possessed_frames": result.possessed_frames,
        "total_frames": result.total_frames,
        "frames_held": result.counts,
        "timeline": [[f, t] for f, t in result.timeline],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Wrote field possession to %s", path)
    return path


def export_field_possession_txt(
    result: PossessionAnalysis, path: str | Path, title: str = ""
) -> Path:
    """Write a human-readable possession report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = f"BALL POSSESSION (field){' — ' + title if title else ''}"
    dom = ("Team " + str(result.dominant_team)
           if result.dominant_team is not None else "none")
    lines = [
        head, "=" * max(40, len(head)), "",
        f"  Team 0 : {result.percentages.get(0, 0.0):5.1f} %  "
        f"({result.counts.get(0, 0)} frames)",
        f"  Team 1 : {result.percentages.get(1, 0.0):5.1f} %  "
        f"({result.counts.get(1, 0)} frames)",
        "",
        f"  dominant team   : {dom}",
        f"  possessed frames: {result.possessed_frames} / {result.total_frames}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote field possession report to %s", path)
    return path
