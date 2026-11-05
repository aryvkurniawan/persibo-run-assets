"""Abstract tracker interface.

Downstream code (the pipeline, exporter, visualizer) depends only on this
contract and on the :class:`Track` dataclass — never on Ultralytics
objects. That keeps the concrete tracker (BoT-SORT today, possibly others
later) swappable, and lets tests inject a fake tracker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from detection.data_models import Detection
from tracking.models import Track


class BaseTracker(ABC):
    """Consumes per-frame detections and returns identity-stable tracks."""

    @abstractmethod
    def update(
        self,
        detections: Sequence[Detection],
        frame: np.ndarray,
        frame_index: int,
    ) -> List[Track]:
        """Advance the tracker by one frame.

        Args:
            detections: Detections for this frame (from ``YOLODetector``).
            frame: The BGR frame, used for motion compensation / ReID.
            frame_index: 1-based frame number, stamped onto each Track.

        Returns:
            The tracks confirmed on this frame.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all tracker state (e.g. when starting a new video)."""
