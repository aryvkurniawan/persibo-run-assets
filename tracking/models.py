"""Data structures for tracking results.

A :class:`Track` is the tracking-stage analog of detection's
``Detection``: it carries a stable ``track_id`` plus a back-reference
(``source_detection_id``) to the raw detection it came from, so the raw
detections stay usable for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Track:
    """A single tracked object on one frame.

    Attributes:
        frame_id: 1-based frame number this observation belongs to.
        track_id: Stable identity assigned by the tracker.
        class_id: Integer class index.
        class_name: Human-readable class name.
        confidence: Detection confidence carried through the tracker.
        bbox: Pixel ``(x1, y1, x2, y2)``.
        source_detection_id: Index of the originating detection within the
            frame's detection list, or None if it couldn't be matched.
    """

    frame_id: int
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    source_detection_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def to_dict(self) -> dict:
        """Serialize in the project's tracking-JSON format."""
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
        }


@dataclass
class FrameTracks:
    """All tracks present on a single video frame."""

    frame_index: int
    tracks: List[Track] = field(default_factory=list)

    def count_by_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for track in self.tracks:
            counts[track.class_name] = counts.get(track.class_name, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "frame": self.frame_index,
            "tracks": [track.to_dict() for track in self.tracks],
        }
