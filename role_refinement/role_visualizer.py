"""Draws refined roles onto video frames.

Each box is labelled with the original detector class *and* the refined
role, e.g. ``ID 12 | det: player | role: referee | 0.87`` so the
correction is auditable at a glance. Colors are role-driven: the two
teams get distinct colors, referees/goalkeepers their own, and unknown a
muted gray.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from role_refinement.role_models import (
    ROLE_BALL,
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
    ROLE_REFEREE,
    ROLE_UNKNOWN,
    RoleResult,
)
from tracking.models import Track

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR colors. Team colors only apply to refined players (by team_id).
_TEAM_COLORS = {0: (255, 128, 0), 1: (0, 128, 255)}   # blue-ish / orange-ish
_ROLE_COLORS = {
    ROLE_REFEREE: (0, 255, 255),     # yellow
    ROLE_GOALKEEPER: (0, 0, 255),    # red
    ROLE_BALL: (255, 255, 0),        # cyan
    ROLE_UNKNOWN: (160, 160, 160),   # gray
}
_DEFAULT_COLOR = (200, 200, 200)


class RoleVisualizer:
    """Renders refined-role boxes and labels."""

    def __init__(
        self,
        roles: Mapping[int, RoleResult],
        box_thickness: int = 2,
        font_scale: float = 0.5,
    ) -> None:
        self._roles = roles
        self._box_thickness = box_thickness
        self._font_scale = font_scale

    def color_for(self, result: Optional[RoleResult]) -> Tuple[int, int, int]:
        if result is None:
            return _DEFAULT_COLOR
        if result.refined_role == ROLE_PLAYER and result.team_id in _TEAM_COLORS:
            return _TEAM_COLORS[result.team_id]
        return _ROLE_COLORS.get(result.refined_role, _DEFAULT_COLOR)

    def label_for(self, track: Track, result: Optional[RoleResult]) -> str:
        if result is None:
            return f"ID {track.track_id} | det: {track.class_name} | role: ?"
        label = (
            f"ID {track.track_id} | det: {result.detected_class} "
            f"| role: {result.refined_role}"
        )
        if result.refined_role == ROLE_PLAYER and result.team_id is not None:
            label += f" (T{result.team_id})"
        label += f" | {result.role_confidence:.2f}"
        return label

    def annotate(
        self, frame: np.ndarray, tracks: Sequence[Track]
    ) -> np.ndarray:
        """Return a copy of ``frame`` with every track's refined role drawn."""
        canvas = frame.copy()
        for track in tracks:
            self._draw(canvas, track)
        return canvas

    # ------------------------------------------------------------------
    def _draw(self, canvas: np.ndarray, track: Track) -> None:
        result = self._roles.get(track.track_id)
        color = self.color_for(result)
        x1, y1, x2, y2 = (int(round(v)) for v in track.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self._box_thickness)

        label = self.label_for(track, result)
        (text_w, text_h), baseline = cv2.getTextSize(
            label, _FONT, self._font_scale, 1
        )
        label_y = y1 - 4
        if label_y - text_h - baseline < 0:
            label_y = y2 + text_h + baseline + 4
        cv2.rectangle(
            canvas,
            (x1, label_y - text_h - baseline),
            (x1 + text_w + 4, label_y + 2),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 2, label_y - baseline + 2),
            _FONT,
            self._font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
