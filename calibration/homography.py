"""Homography between image pixels and pitch (field) meters.

A single planar homography maps the broadcast image plane to the (flat)
pitch plane. With >= 4 manually-clicked correspondences we can solve it;
with >= 6 we use RANSAC to be robust to a mis-clicked point.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger("calibration.homography")

Point = Tuple[float, float]

# At/above this many points, use RANSAC; below, a plain least-squares solve.
_RANSAC_MIN_POINTS = 6
_MIN_POINTS = 4


def compute_homography(
    image_points: Sequence[Point], field_points: Sequence[Point]
) -> np.ndarray:
    """Solve the image->field homography from correspondences.

    Raises ``ValueError`` if there are fewer than 4 points or the solve
    fails (degenerate / collinear configuration).
    """
    img = np.asarray(image_points, dtype=np.float64)
    fld = np.asarray(field_points, dtype=np.float64)
    if img.shape != fld.shape or img.ndim != 2 or img.shape[1] != 2:
        raise ValueError("image_points and field_points must be matching (N, 2)")
    if len(img) < _MIN_POINTS:
        raise ValueError(
            f"need at least {_MIN_POINTS} correspondences, got {len(img)}"
        )
    method = cv2.RANSAC if len(img) >= _RANSAC_MIN_POINTS else 0
    H, _mask = cv2.findHomography(img, fld, method)
    if H is None:
        raise ValueError("homography solve failed (degenerate point configuration)")
    return H


class Homography:
    """Image<->field projector backed by a 3x3 homography matrix."""

    def __init__(
        self,
        matrix: np.ndarray,
        image_points: Optional[Sequence[Point]] = None,
        field_points: Optional[Sequence[Point]] = None,
    ) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        self.inverse = np.linalg.inv(self.matrix)
        self._image_points = (
            np.asarray(image_points, dtype=np.float64)
            if image_points is not None else None
        )
        self._field_points = (
            np.asarray(field_points, dtype=np.float64)
            if field_points is not None else None
        )

    @classmethod
    def from_correspondences(
        cls, image_points: Sequence[Point], field_points: Sequence[Point]
    ) -> "Homography":
        matrix = compute_homography(image_points, field_points)
        return cls(matrix, image_points, field_points)

    # ------------------------------------------------------------------
    def project_image_to_field(self, point: Point) -> Point:
        return _apply(self.matrix, point)

    def project_field_to_image(self, point: Point) -> Point:
        return _apply(self.inverse, point)

    def project_many_image_to_field(self, points: Sequence[Point]) -> np.ndarray:
        return np.array([self.project_image_to_field(p) for p in points])

    def reprojection_error(self) -> float:
        """Mean Euclidean error (in field meters) over the calibration points.

        Projects each clicked image point to the field and compares against
        the known field coordinate. Returns ``inf`` if no points are stored.
        """
        if self._image_points is None or self._field_points is None:
            return float("inf")
        if len(self._image_points) == 0:
            return float("inf")
        projected = self.project_many_image_to_field(self._image_points)
        errors = np.linalg.norm(projected - self._field_points, axis=1)
        return float(np.mean(errors))

    def to_dict(self) -> dict:
        """Serialise matrix and (when known) the calibration correspondences.

        The points keep ``reprojection_error()`` meaningful after a JSON
        round-trip; older files carrying only ``matrix`` still load.
        """
        data = {"matrix": self.matrix.tolist()}
        if self._image_points is not None and self._field_points is not None:
            data["image_points"] = self._image_points.tolist()
            data["field_points"] = self._field_points.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Homography":
        image_points = data.get("image_points")
        field_points = data.get("field_points")
        if image_points is None or field_points is None:
            image_points = field_points = None
        return cls(np.array(data["matrix"], dtype=np.float64),
                   image_points, field_points)


def _apply(matrix: np.ndarray, point: Point) -> Point:
    """Apply a 3x3 homography to a 2D point (with perspective divide)."""
    vec = matrix @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    w = vec[2]
    if abs(w) < 1e-12:
        return float("nan"), float("nan")
    return float(vec[0] / w), float(vec[1] / w)


class MultiHomography:
    """Per-keyframe homographies with nearest-in-time selection.

    Handles a panning camera: for a given video frame, returns the
    homography of the calibrated keyframe whose frame number is closest.
    Exposes ``for_frame`` so it can be used anywhere a per-frame projector
    is expected (the projection code duck-types on this method).
    """

    def __init__(self, entries: Sequence["tuple[int, Homography]"]) -> None:
        if not entries:
            raise ValueError("MultiHomography needs at least one homography")
        self._entries = sorted(entries, key=lambda e: e[0])

    @classmethod
    def from_keyframes(cls, keyframes: Sequence) -> "MultiHomography":
        """Build from calibration keyframes (each: ``frame``, image/field pts).

        Keyframes with fewer than 4 points are skipped.
        """
        entries = []
        for keyframe in keyframes:
            if keyframe.count < 4:
                continue
            homography = Homography.from_correspondences(
                keyframe.image_points(), keyframe.field_points())
            entries.append((int(keyframe.frame), homography))
        return cls(entries)

    def for_frame(self, frame_index: int) -> Homography:
        return min(self._entries, key=lambda e: abs(e[0] - int(frame_index)))[1]

    @property
    def frame_numbers(self) -> "list[int]":
        return [f for f, _ in self._entries]

    def reprojection_error(self) -> float:
        """Mean reprojection error (meters) across all keyframes."""
        return float(np.mean([h.reprojection_error() for _, h in self._entries]))

    def __len__(self) -> int:
        return len(self._entries)
