"""Canonical 2D football pitch coordinate system (Phase 4).

A real pitch in meters, origin at the top-left corner:

    x axis -> pitch length  (0 .. 105)
    y axis -> pitch width   (0 .. 68)

Named reference points are the clear, human-identifiable intersections a
user can click on a broadcast frame (corners, halfway line, penalty/goal
area corners, penalty spots, center marks). Their field coordinates are
derived from the official dimensions, so calibration only needs the
*image* clicks — the field side is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class PitchDimensions:
    """Official pitch dimensions in meters."""

    length: float = 105.0
    width: float = 68.0
    penalty_area_depth: float = 16.5
    penalty_area_width: float = 40.32
    goal_area_depth: float = 5.5
    goal_area_width: float = 18.32
    penalty_spot_distance: float = 11.0
    center_circle_radius: float = 9.15


# The reference points a user can calibrate, in a sensible click order.
REFERENCE_POINT_NAMES: List[str] = [
    "top_left_corner",
    "top_right_corner",
    "bottom_left_corner",
    "bottom_right_corner",
    "halfway_top_touchline",
    "halfway_bottom_touchline",
    "center_spot",
    "left_penalty_spot",
    "right_penalty_spot",
    "left_penalty_area_top",
    "left_penalty_area_bottom",
    "right_penalty_area_top",
    "right_penalty_area_bottom",
    "left_goal_area_top",
    "left_goal_area_bottom",
    "right_goal_area_top",
    "right_goal_area_bottom",
    "center_circle_top",
    "center_circle_bottom",
]


class PitchModel:
    """Holds the field coordinates of every named reference point."""

    def __init__(self, dims: PitchDimensions = PitchDimensions()) -> None:
        self.dims = dims
        self.points: Dict[str, Point] = self._build_points(dims)

    @staticmethod
    def _build_points(d: PitchDimensions) -> Dict[str, Point]:
        cx = d.length / 2.0          # 52.5
        cy = d.width / 2.0           # 34.0
        pa_half = d.penalty_area_width / 2.0   # 20.16
        ga_half = d.goal_area_width / 2.0      # 9.16
        return {
            "top_left_corner": (0.0, 0.0),
            "top_right_corner": (d.length, 0.0),
            "bottom_left_corner": (0.0, d.width),
            "bottom_right_corner": (d.length, d.width),
            "halfway_top_touchline": (cx, 0.0),
            "halfway_bottom_touchline": (cx, d.width),
            "center_spot": (cx, cy),
            "left_penalty_spot": (d.penalty_spot_distance, cy),
            "right_penalty_spot": (d.length - d.penalty_spot_distance, cy),
            "left_penalty_area_top": (d.penalty_area_depth, cy - pa_half),
            "left_penalty_area_bottom": (d.penalty_area_depth, cy + pa_half),
            "right_penalty_area_top": (d.length - d.penalty_area_depth, cy - pa_half),
            "right_penalty_area_bottom": (
                d.length - d.penalty_area_depth, cy + pa_half),
            "left_goal_area_top": (d.goal_area_depth, cy - ga_half),
            "left_goal_area_bottom": (d.goal_area_depth, cy + ga_half),
            "right_goal_area_top": (d.length - d.goal_area_depth, cy - ga_half),
            "right_goal_area_bottom": (d.length - d.goal_area_depth, cy + ga_half),
            "center_circle_top": (cx, cy - d.center_circle_radius),
            "center_circle_bottom": (cx, cy + d.center_circle_radius),
        }

    def field_point(self, name: str) -> Point:
        if name not in self.points:
            raise KeyError(f"unknown reference point '{name}'")
        return self.points[name]

    def names(self) -> List[str]:
        return list(REFERENCE_POINT_NAMES)

    def contains(self, name: str) -> bool:
        return name in self.points
