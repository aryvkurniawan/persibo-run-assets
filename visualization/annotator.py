"""Draws detection results onto video frames.

The annotator never mutates the input frame — it always works on a copy,
so the raw frame stays available for later pipeline stages.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from detection.data_models import Detection
from utils.config_loader import VisualizationConfig
from utils.logger import get_logger

logger = get_logger("visualization.annotator")

_DEFAULT_COLOR: Tuple[int, int, int] = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_TEXT_COLOR = (255, 255, 255)


class DetectionAnnotator:
    """Renders bounding boxes, labels and an optional debug HUD."""

    def __init__(self, config: VisualizationConfig, debug: bool = False) -> None:
        self._config = config
        self._debug = debug

    def annotate(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        frame_index: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """Return a copy of ``frame`` with all detections drawn on it.

        Args:
            frame: BGR image.
            detections: Detections for this frame.
            frame_index: Current frame number (shown in the debug HUD).
            fps: Current processing FPS (shown in the debug HUD).
        """
        canvas = frame.copy()
        for detection in detections:
            self._draw_detection(canvas, detection)
        if self._debug:
            self._draw_hud(canvas, detections, frame_index, fps)
        return canvas

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _color_for(self, class_name: str) -> Tuple[int, int, int]:
        return tuple(self._config.colors.get(class_name, _DEFAULT_COLOR))

    def _draw_detection(self, canvas: np.ndarray, detection: Detection) -> None:
        x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox)
        color = self._color_for(detection.class_name)
        thickness = self._config.box_thickness

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        label = detection.class_name
        if self._config.show_confidence:
            label = f"{label} {detection.confidence:.2f}"

        font_scale = self._config.font_scale
        (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, font_scale, 1)

        # Label background above the box; flip below if it would clip the top.
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
        detections: Sequence[Detection],
        frame_index: Optional[int],
        fps: Optional[float],
    ) -> None:
        counts: Dict[str, int] = {}
        for detection in detections:
            counts[detection.class_name] = counts.get(detection.class_name, 0) + 1

        lines = []
        if frame_index is not None:
            lines.append(f"frame: {frame_index}")
        if fps is not None:
            lines.append(f"fps: {fps:.1f}")
        lines.append(f"detections: {len(detections)}")
        for name in sorted(counts):
            lines.append(f"  {name}: {counts[name]}")

        line_height = 22
        panel_height = 10 + line_height * len(lines)
        panel_width = 220

        # Semi-transparent dark panel so the HUD stays readable on grass.
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
