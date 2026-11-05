"""Phase 8 — team tactical analysis from player locations & movement.

Infers explainable team behaviour from the projected field positions (metres)
plus the Phase-5 speed analytics: compactness, the lateral attacking zone,
transition speed, team shape, build-up style and pressure height. Every label
comes with the numeric metric it was derived from and a short reason string,
so the output is auditable rather than a black box. Earlier phases are
untouched — this only reads ``field_positions`` + an ``AnalyticsResult``.

Pitch coordinate system (:class:`~calibration.pitch_model.PitchDimensions`):
``x`` is length 0..105 (goal to goal), ``y`` is width 0..68 (Left .. Right).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple

from analytics.analytics_models import AnalyticsResult
from calibration.pitch_model import PitchDimensions
from utils.logger import get_logger

logger = get_logger("analytics.tactical_analysis")

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Spatial statistics (pure)
# ---------------------------------------------------------------------------
def spatial_stats(
    per_frame_points: Sequence[Sequence[Point]],
    dims: PitchDimensions = PitchDimensions(),
) -> dict:
    """Aggregate per-frame team shape stats over the match.

    ``per_frame_points`` is a list (one per frame) of the team's player
    positions that frame. Returns compactness (mean distance of players to the
    frame centroid, averaged over frames), lateral width and vertical depth
    (averaged per-frame stds), the overall centroid, and the fraction of all
    positions in the left / center / right thirds.
    """
    compact_vals: List[float] = []
    width_vals: List[float] = []
    depth_vals: List[float] = []
    all_x: List[float] = []
    all_y: List[float] = []
    thirds = [0, 0, 0]                      # left, center, right counts
    third = dims.width / 3.0

    for pts in per_frame_points:
        if len(pts) < 2:
            continue
        cx = mean(p[0] for p in pts)
        cy = mean(p[1] for p in pts)
        compact_vals.append(mean(math.hypot(p[0] - cx, p[1] - cy) for p in pts))
        width_vals.append(pstdev([p[1] for p in pts]))
        depth_vals.append(pstdev([p[0] for p in pts]))
        for x, y in pts:
            all_x.append(x)
            all_y.append(y)
            thirds[0 if y < third else 2 if y >= 2 * third else 1] += 1

    total = sum(thirds) or 1
    return {
        "compactness_m": mean(compact_vals) if compact_vals else 0.0,
        "width_m": mean(width_vals) if width_vals else 0.0,
        "depth_m": mean(depth_vals) if depth_vals else 0.0,
        "centroid_x": mean(all_x) if all_x else dims.length / 2.0,
        "centroid_y": mean(all_y) if all_y else dims.width / 2.0,
        "lateral_fractions": (thirds[0] / total, thirds[1] / total, thirds[2] / total),
        "n_frames": len(compact_vals),
    }


# ---------------------------------------------------------------------------
# Classifiers (pure) — each returns (label, reason)
# ---------------------------------------------------------------------------
def classify_compactness(compactness_m: float) -> Tuple[str, str]:
    label = "Compact" if compactness_m < 18.0 else \
        "Wide" if compactness_m > 26.0 else "Balanced"
    return label, f"players average {compactness_m:.1f} m from the team centroid"


def classify_shape(width_m: float) -> Tuple[str, str]:
    label = "Compact" if width_m < 12.0 else \
        "Wide" if width_m > 18.0 else "Balanced"
    return label, f"lateral spread of {width_m:.1f} m across the pitch width"


def attacking_zone(lateral_fractions: Tuple[float, float, float]) -> Tuple[str, str]:
    left, center, right = lateral_fractions
    idx = max(range(3), key=lambda i: lateral_fractions[i])
    label = ("Left", "Center", "Right")[idx]
    return label, (f"{lateral_fractions[idx] * 100:.0f}% of activity in the "
                   f"{label.lower()} channel")


def classify_transition(avg_speed_kmh: float) -> Tuple[str, str]:
    label = "Fast" if avg_speed_kmh > 9.5 else \
        "Slow" if avg_speed_kmh < 7.0 else "Medium"
    return label, f"average player speed {avg_speed_kmh:.1f} km/h"


def classify_buildup(depth_m: float, transition: str) -> Tuple[str, str]:
    if depth_m > 22.0 or transition == "Fast":
        return "Direct Style", (f"stretched {depth_m:.1f} m vertically with "
                                f"{transition.lower()} transitions")
    if depth_m < 16.0 and transition != "Fast":
        return "Short Passing Style", (f"compact {depth_m:.1f} m vertical block, "
                                       "patient transitions")
    return "Mixed", f"{depth_m:.1f} m vertical block with {transition.lower()} tempo"


def classify_pressure(block_height_m: float) -> Tuple[str, str]:
    label = "High Press" if block_height_m > 58.0 else \
        "Low Block" if block_height_m < 40.0 else "Mid Block"
    return label, f"defensive block ~{block_height_m:.0f} m up from its own goal"


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@dataclass
class TacticalProfile:
    team_id: int
    compactness: str
    team_shape: str
    attacking_zone: str
    transition_speed: str
    build_up_style: str
    pressure_style: str
    metrics: Dict[str, float] = field(default_factory=dict)
    reasons: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "team_id": int(self.team_id),
            "compactness": self.compactness,
            "team_shape": self.team_shape,
            "attacking_zone": self.attacking_zone,
            "transition_speed": self.transition_speed,
            "build_up_style": self.build_up_style,
            "pressure_style": self.pressure_style,
            "metrics": {k: round(float(v), 2) for k, v in self.metrics.items()},
            "reasons": self.reasons,
        }


def _team_frame_points(
    frames: Sequence[dict], team_id: int, ball_class_name: str = "ball"
) -> List[List[Point]]:
    """Per-frame list of a team's on-pitch player/keeper field points."""
    out: List[List[Point]] = []
    for frame in frames:
        pts: List[Point] = []
        for obj in frame.get("objects", []):
            if obj.get("team_id") != team_id or obj.get("role") == ball_class_name:
                continue
            if not obj.get("on_pitch", True):
                continue
            x, y = obj.get("field", (float("nan"), float("nan")))
            if math.isfinite(x) and math.isfinite(y):
                pts.append((float(x), float(y)))
        if pts:
            out.append(pts)
    return out


def _team_avg_speed(result: Optional[AnalyticsResult], team_id: int) -> float:
    if result is None:
        return 0.0
    speeds = [t.avg_speed_kmh for t in result.tracks
              if t.team_id == team_id and t.role in ("player", "goalkeeper")
              and t.avg_speed_kmh > 0]
    return mean(speeds) if speeds else 0.0


def compute_team_tactics(
    team_id: int,
    per_frame_points: Sequence[Sequence[Point]],
    avg_speed_kmh: float,
    own_goal_x: float,
    dims: PitchDimensions = PitchDimensions(),
) -> TacticalProfile:
    """Build one team's tactical profile (labels + metrics + reasons)."""
    stats = spatial_stats(per_frame_points, dims)
    block_height = abs(stats["centroid_x"] - own_goal_x)

    compact, r_compact = classify_compactness(stats["compactness_m"])
    shape, r_shape = classify_shape(stats["width_m"])
    zone, r_zone = attacking_zone(stats["lateral_fractions"])
    transition, r_trans = classify_transition(avg_speed_kmh)
    buildup, r_build = classify_buildup(stats["depth_m"], transition)
    pressure, r_press = classify_pressure(block_height)

    return TacticalProfile(
        team_id=team_id,
        compactness=compact,
        team_shape=shape,
        attacking_zone=zone,
        transition_speed=transition,
        build_up_style=buildup,
        pressure_style=pressure,
        metrics={
            "compactness_m": stats["compactness_m"],
            "width_m": stats["width_m"],
            "depth_m": stats["depth_m"],
            "centroid_x": stats["centroid_x"],
            "centroid_y": stats["centroid_y"],
            "block_height_m": block_height,
            "avg_speed_kmh": avg_speed_kmh,
        },
        reasons={
            "compactness": r_compact,
            "team_shape": r_shape,
            "attacking_zone": r_zone,
            "transition_speed": r_trans,
            "build_up_style": r_build,
            "pressure_style": r_press,
        },
    )


def compute_tactical_analysis(
    frames: Sequence[dict],
    result: Optional[AnalyticsResult] = None,
    ball_class_name: str = "ball",
    dims: PitchDimensions = PitchDimensions(),
) -> List[TacticalProfile]:
    """Tactical profiles for team 0 and team 1.

    Attacking direction is inferred from the two teams' average x: the team
    with the higher mean x attacks toward x=length (own goal at 0); the other
    defends the far goal. That sets each team's ``own_goal_x`` so the pressure
    height (block distance from own goal) is meaningful.
    """
    team_points = {t: _team_frame_points(frames, t, ball_class_name) for t in (0, 1)}
    mean_x = {
        t: mean([p[0] for pf in pts for p in pf]) if any(pts) else dims.length / 2.0
        for t, pts in team_points.items()
    }
    # Higher mean x -> attacks right (own goal at x=0); the other defends x=length.
    if mean_x[0] >= mean_x[1]:
        own_goal = {0: 0.0, 1: dims.length}
    else:
        own_goal = {0: dims.length, 1: 0.0}

    profiles: List[TacticalProfile] = []
    for t in (0, 1):
        if not team_points[t]:
            continue
        profiles.append(compute_team_tactics(
            t, team_points[t], _team_avg_speed(result, t), own_goal[t], dims))
    logger.info("Tactical analysis built for %d team(s)", len(profiles))
    return profiles


def export_team_tactical(
    profiles: Sequence[TacticalProfile],
    path: str | Path,
    metadata: Optional[dict] = None,
) -> Path:
    """Write the team tactical JSON (one entry per team)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {"phase": "team_tactical", **(metadata or {})},
        "teams": [p.to_dict() for p in profiles],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Wrote team tactical analysis to %s", path)
    return path


def export_team_tactical_txt(
    profiles: Sequence[TacticalProfile],
    path: str | Path,
    title: str = "",
) -> Path:
    """Write a human-readable team tactical report (label + reason per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = f"TEAM TACTICAL ANALYSIS{' — ' + title if title else ''}"
    lines = [head, "=" * max(56, len(head)), ""]
    rows = [
        ("Compactness", "compactness", lambda p: p.compactness),
        ("Team Shape", "team_shape", lambda p: p.team_shape),
        ("Attacking Zone", "attacking_zone", lambda p: p.attacking_zone),
        ("Transition Speed", "transition_speed", lambda p: p.transition_speed),
        ("Build-up Style", "build_up_style", lambda p: p.build_up_style),
        ("Pressure Style", "pressure_style", lambda p: p.pressure_style),
    ]
    for prof in sorted(profiles, key=lambda p: p.team_id):
        lines.append(f"TEAM {prof.team_id}")
        for label, key, getter in rows:
            lines.append(f"  {label:<16}: {getter(prof):<20} "
                         f"({prof.reasons.get(key, '')})")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote team tactical report to %s", path)
    return path
