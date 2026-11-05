"""Tests for the DirectML ONNX wrapper's post-processing.

These pin the two behaviours that silently broke detection: output-layout
detection and per-class NMS.
"""

import numpy as np
import pytest

from detection.onnx_dml import OnnxDmlModel

NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}


def _model_from_output(output: np.ndarray) -> OnnxDmlModel:
    """An OnnxDmlModel with the session stubbed out."""
    model = OnnxDmlModel.__new__(OnnxDmlModel)
    model.names = dict(NAMES)
    model._imgsz = 320
    model._channels = 4 + len(NAMES)
    model._input = "images"
    model._session = _FakeSession(output)
    return model


class _FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def get_providers(self):
        return ["DmlExecutionProvider"]

    def run(self, _outputs, _feed):
        return [self.output]


def _raw(anchors):
    """Build a (1, C, N) Ultralytics-style output from (cx, cy, w, h, *scores)."""
    arr = np.asarray(anchors, np.float32)
    return arr.T[None]


def test_channels_first_output_is_transposed():
    model = _model_from_output(_raw([
        [100, 100, 20, 40, 0.0, 0.0, 0.9, 0.0],
    ]))
    boxes = model.predict(np.zeros((320, 320, 3), np.uint8), conf=0.5)[0].boxes
    assert len(boxes) == 1
    assert boxes.cls[0] == pytest.approx(2)
    # cx,cy,w,h -> x1,y1,x2,y2 around the same centre.
    assert boxes.xyxy[0] == pytest.approx([90, 80, 110, 120], abs=1.0)


def test_anchors_first_output_is_left_alone():
    model = _model_from_output(_raw([
        [100, 100, 20, 40, 0.0, 0.0, 0.9, 0.0],
    ]).transpose(0, 2, 1))
    boxes = model.predict(np.zeros((320, 320, 3), np.uint8), conf=0.5)[0].boxes
    assert len(boxes) == 1
    assert boxes.xyxy[0] == pytest.approx([90, 80, 110, 120], abs=1.0)


def test_nms_does_not_suppress_across_classes():
    """A confident player must not erase an overlapping referee box."""
    model = _model_from_output(_raw([
        [100, 100, 20, 40, 0.0, 0.0, 0.95, 0.0],   # player
        [101, 101, 20, 40, 0.0, 0.0, 0.0, 0.80],   # referee, same box
    ]))
    boxes = model.predict(np.zeros((320, 320, 3), np.uint8), conf=0.5, iou=0.5)[0].boxes
    assert len(boxes) == 2
    assert sorted(int(c) for c in boxes.cls) == [2, 3]


def test_same_class_overlap_is_suppressed():
    model = _model_from_output(_raw([
        [100, 100, 20, 40, 0.0, 0.0, 0.95, 0.0],
        [101, 101, 20, 40, 0.0, 0.0, 0.80, 0.0],
    ]))
    boxes = model.predict(np.zeros((320, 320, 3), np.uint8), conf=0.5, iou=0.5)[0].boxes
    assert len(boxes) == 1
    assert boxes.conf[0] == pytest.approx(0.95)


def test_output_without_a_channel_axis_raises():
    model = _model_from_output(_raw([[0, 0, 0, 0, 0, 0, 0]])[:1, :7])
    with pytest.raises(ValueError, match="no axis"):
        model.predict(np.zeros((320, 320, 3), np.uint8), conf=0.5)
