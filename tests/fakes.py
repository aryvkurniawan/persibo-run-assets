"""Test doubles that mimic the Ultralytics result objects.

Allows testing YOLODetector parsing/threshold logic without loading any
real model weights (and without a GPU).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class FakeTensor:
    """Mimics torch.Tensor's ``.cpu().numpy()`` chain."""

    def __init__(self, data) -> None:
        self._data = np.asarray(data)

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._data


class FakeBoxes:
    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)
        self._n = len(conf)

    def __len__(self) -> int:
        return self._n


class FakeResult:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


class FakeTracker:
    """Stands in for a BaseTracker; assigns one stable track per class.

    Every detection of a given class on a frame is turned into a Track
    whose ``track_id`` is just the class_id + 1, so pipeline wiring can be
    asserted deterministically without any real tracking maths or GPU.
    """

    def __init__(self) -> None:
        self.update_calls = 0
        self.reset_calls = 0

    def update(self, detections, frame, frame_index):
        from tracking.models import Track

        self.update_calls += 1
        tracks = []
        for det_id, det in enumerate(detections):
            tracks.append(
                Track(
                    frame_id=frame_index,
                    track_id=det.class_id + 1,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    source_detection_id=det_id,
                )
            )
        return tracks

    def reset(self) -> None:
        self.reset_calls += 1


class FakeYOLOModel:
    """Stands in for ultralytics.YOLO; returns pre-canned detections.

    Args:
        detections: List of ``(x1, y1, x2, y2, confidence, class_id)``.
        names: Optional class-name mapping, like ``model.names``.
    """

    def __init__(self, detections: Sequence[tuple], names=None) -> None:
        self._detections = list(detections)
        if names is not None:
            self.names = names
        self.predict_calls: List[dict] = []

    def predict(self, source=None, **kwargs) -> List[FakeResult]:
        self.predict_calls.append({"source": source, **kwargs})
        if not self._detections:
            return [FakeResult(boxes=None)]
        xyxy = [d[:4] for d in self._detections]
        conf = [d[4] for d in self._detections]
        cls = [d[5] for d in self._detections]
        return [FakeResult(FakeBoxes(xyxy, conf, cls))]
