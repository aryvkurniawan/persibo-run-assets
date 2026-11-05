"""Export per-frame tracks to a JSON file, separate from detections.

Output schema:

    {
      "metadata": { "tracker": "botsort", "weights": "...", ... },
      "frames": [
        {
          "frame": 1,
          "tracks": [
            {"track_id": 12, "class": "player", "confidence": 0.91,
             "bbox": [x1, y1, x2, y2]}
          ]
        }
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tracking.models import FrameTracks, Track
from utils.logger import get_logger

logger = get_logger("tracking.track_exporter")


class TrackJSONExporter:
    """Accumulates per-frame tracks and writes them out once at the end."""

    def __init__(self, output_path: str | Path, metadata: Optional[Dict] = None):
        self.output_path = Path(output_path)
        self.metadata: Dict = metadata or {}
        self._frames: List[dict] = []
        self._unique_ids: set = set()

    def add_frame(self, frame_index: int, tracks: Sequence[Track]) -> None:
        frame = FrameTracks(frame_index=frame_index, tracks=list(tracks))
        self._frames.append(frame.to_dict())
        for track in tracks:
            self._unique_ids.add((track.class_name, track.track_id))

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def total_tracks(self) -> int:
        return sum(len(f["tracks"]) for f in self._frames)

    @property
    def unique_track_count(self) -> int:
        return len(self._unique_ids)

    def save(self) -> Path:
        """Write the accumulated tracks to disk and return the path."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metadata": self.metadata, "frames": self._frames}
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info(
            "Exported %d frames (%d track rows, %d unique IDs) to %s",
            self.frame_count,
            self.total_tracks,
            self.unique_track_count,
            self.output_path,
        )
        return self.output_path
