"""Draws tracks (with their IDs) onto video frames.

Unlike the Phase 1 detection annotator, colors here are derived from the
``track_id`` rather than a fixed per-class map — each tracked identity
gets its own stable color, which makes ID stability easy to eyeball.
"""

from __future__ import annotations

import colorsys
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from tracking.models import Track
from utils.config_loader import VisualizationConfig
from utils.logger import get_logger

logger = get_logger("tracking.track_visualizer")

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_TEXT_COLOR = (255, 255, 255)
# Golden-ratio hue stepping spreads consecutive IDs into distinct colors.
_GOLDEN_RATIO_CONJUGATE = 0.61803398875


def color_for_track_id(track_id: int) -> Tuple[int, int, int]:
    """Map a track ID to a stable, well-spread BGR color."""
    hue = (track_id * _GOLDEN_RATIO_CONJUGATE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


class TrackVisualizer:
    """Renders track boxes, ``#id class`` labels and an optional debug HUD."""

    def __init__(self, config: VisualizationConfig, debug: bool = False) -> None:
        self._config = config
        self._debug = debug

    def annotate(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        frame_index: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """Return a copy of ``frame`` with all tracks drawn on it."""
        canvas = frame.copy()
        for track in tracks:
            self._draw_track(canvas, track)
        if self._debug:
            self._draw_hud(canvas, tracks, frame_index, fps)
        return canvas

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _draw_track(self, canvas: np.ndarray, track: Track) -> None:
        x1, y1, x2, y2 = (int(round(v)) for v in track.bbox)
        color = color_for_track_id(track.track_id)
        thickness = self._config.box_thickness

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        label = f"#{track.track_id} {track.class_name}"
        if self._config.show_confidence:
            label = f"{label} {track.confidence:.2f}"

        font_scale = self._config.font_scale
        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, font_scale, 1)

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
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def _draw_hud(
        self,
        canvas: np.ndarray,
        tracks: Sequence[Track],
        frame_index: Optional[int],
        fps: Optional[float],
    ) -> None:
        counts: Dict[str, int] = {}
        for track in tracks:
            counts[track.class_name] = counts.get(track.class_name, 0) + 1

        lines = []
        if frame_index is not None:
            lines.append(f"frame: {frame_index}")
        if fps is not None:
            lines.append(f"fps: {fps:.1f}")
        lines.append(f"tracks: {len(tracks)}")
        for name in sorted(counts):
            lines.append(f"  {name}: {counts[name]}")

        line_height = 22
        panel_height = 10 + line_height * len(lines)
        panel_width = 220

        overlay = canvas[0:panel_height, 0:panel_width].copy()
        overlay[:] = (0, 0, 0)
        blended = cv2.addWeighted(
            canvas[0:panel_height, 0:panel_width], 0.4, overlay, 0.6, 0
        )
        canvas[0:panel_height, 0:panel_width] = blended

        for i, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (8, 24 + i * line_height),
                _FONT,
                0.55,
                _HUD_TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
