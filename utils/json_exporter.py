"""Export per-frame detections to a JSON file.

Output schema:

    {
      "metadata": { "video": "...", "model": "...", ... },
      "frames": [
        {
          "frame": 1,
          "detections": [
            {"class": "player", "confidence": 0.93, "bbox": [x1, y1, x2, y2]}
          ]
        }
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from detection.data_models import Detection, FrameDetections
from utils.logger import get_logger

logger = get_logger("utils.json_exporter")


class DetectionJSONExporter:
    """Accumulates per-frame detections and writes them out once at the end."""

    def __init__(self, output_path: str | Path, metadata: Optional[Dict] = None):
        self.output_path = Path(output_path)
        self.metadata: Dict = metadata or {}
        self._frames: List[dict] = []

    def add_frame(self, frame_index: int, detections: Sequence[Detection]) -> None:
        frame = FrameDetections(frame_index=frame_index, detections=list(detections))
        self._frames.append(frame.to_dict())

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def total_detections(self) -> int:
        return sum(len(f["detections"]) for f in self._frames)

    def save(self) -> Path:
        """Write the accumulated detections to disk and return the path."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metadata": self.metadata, "frames": self._frames}
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info(
            "Exported %d frames (%d detections) to %s",
            self.frame_count, self.total_detections, self.output_path,
        )
        return self.output_path
