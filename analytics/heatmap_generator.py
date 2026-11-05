"""Phase 7 — heatmap density grids from projected field positions.

Pure-ish numpy/OpenCV computation (no model, no I/O beyond reading the given
frames): accumulate on-pitch field points into a grid, Gaussian-smooth it,
and normalize to 0..1. :mod:`analytics.heatmap_renderer` paints the grid over
a top-down pitch. Earlier phases are untouched — this only reads the
``field_positions`` structure produced by the minimap.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from calibration.pitch_model import PitchDimensions
from utils.logger import get_logger

logger = get_logger("analytics.heatmap_generator")

Point = Tuple[float, float]


def collect_positions(
    frames: Sequence[dict],
    track_id: Optional[int] = None,
    team_id: Optional[int] = None,
    role_in: Optional[Sequence[str]] = ("player", "goalkeeper"),
    ball_class_name: str = "ball",
    on_pitch_only: bool = True,
) -> List[Point]:
    """Gather ``(x, y)`` field points matching the filters.

    ``frames`` is the ``field_positions`` list. Filter by a single track id,
    a team, and/or a role set. Off-pitch / non-finite points (homography
    blow-ups) are skipped.
    """
    keep_roles = set(role_in) if role_in else None
    points: List[Point] = []
    for frame in frames:
        for obj in frame.get("objects", []):
            role = obj.get("role")
            if role == ball_class_name:
                continue
            if keep_roles is not None and role not in keep_roles:
                continue
            if track_id is not None and obj.get("track_id") != track_id:
                continue
            if team_id is not None and obj.get("team_id") != team_id:
                continue
            if on_pitch_only and not obj.get("on_pitch", True):
                continue
            fx, fy = obj.get("field", (float("nan"), float("nan")))
            if math.isfinite(fx) and math.isfinite(fy):
                points.append((float(fx), float(fy)))
    return points


def density_grid(
    points: Sequence[Point],
    dims: PitchDimensions = PitchDimensions(),
    px_per_meter: float = 4.0,
    sigma_m: float = 2.5,
) -> np.ndarray:
    """Accumulate points into a Gaussian-smoothed, 0..1 density grid.

    The grid spans the whole pitch (``dims`` metres) at ``px_per_meter``
    resolution; each point drops into its cell, the grid is blurred with a
    Gaussian of ``sigma_m`` metres, then normalized by its peak. An empty
    input yields an all-zero grid.
    """
    w = max(int(round(dims.length * px_per_meter)), 1)
    h = max(int(round(dims.width * px_per_meter)), 1)
    grid = np.zeros((h, w), dtype=np.float32)
    for x, y in points:
        gx = int(x * px_per_meter)
        gy = int(y * px_per_meter)
        if 0 <= gx < w and 0 <= gy < h:
            grid[gy, gx] += 1.0

    if sigma_m > 0 and grid.any():
        sigma_px = max(sigma_m * px_per_meter, 0.8)
        ksize = int(sigma_px * 3) | 1            # odd kernel ~3 sigma
        grid = cv2.GaussianBlur(grid, (ksize, ksize), sigma_px)

    peak = float(grid.max())
    if peak > 0:
        grid /= peak
    return grid
