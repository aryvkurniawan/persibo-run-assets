"""Core data structures for detection results.

These dataclasses are the contract between the detector, the visualizer
and the exporter — no module passes raw model output to another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Detection:
    """A single detected object on one frame.

    Attributes:
        class_id: Integer class index from the model.
        class_name: Human-readable class name ("player", "ball", ...).
        confidence: Detection confidence in [0, 1].
        bbox: Pixel coordinates ``(x1, y1, x2, y2)`` — top-left and
            bottom-right corners.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        """Serialize in the project's JSON export format."""
        return {
            "class": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
        }


@dataclass
class FrameDetections:
    """All detections found on a single video frame."""

    frame_index: int
    detections: List[Detection] = field(default_factory=list)

    def count_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "frame": self.frame_index,
            "detections": [det.to_dict() for det in self.detections],
        }
