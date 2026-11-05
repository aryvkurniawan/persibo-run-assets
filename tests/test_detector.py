import numpy as np
import pytest

from detection.detector import YOLODetector
from tests.fakes import FakeYOLOModel
from utils.config_loader import ModelConfig

CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}


def make_detector(detections, *, conf=0.35, overrides=None, names=None):
    config = ModelConfig(
        device="cpu",
        confidence_threshold=conf,
        class_confidence_overrides=overrides or {},
    )
    model = FakeYOLOModel(detections, names=names)
    return YOLODetector(config, CLASS_NAMES, model=model), model


def test_detect_parses_results():
    detector, _ = make_detector(
        [
            (10, 20, 50, 120, 0.93, 2),   # player
            (300, 310, 312, 322, 0.41, 0),  # ball
        ]
    )
    detections = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))

    assert len(detections) == 2
    player, ball = detections
    assert player.class_name == "player"
    assert player.confidence == pytest.approx(0.93)
    assert player.bbox == (10.0, 20.0, 50.0, 120.0)
    assert ball.class_name == "ball"


def test_global_threshold_filters_low_confidence():
    detector, _ = make_detector(
        [(0, 0, 10, 10, 0.30, 2)],  # player below global 0.35
        conf=0.35,
    )
    assert detector.detect(np.zeros((64, 64, 3), dtype=np.uint8)) == []


def test_per_class_override_keeps_low_confidence_ball():
    detector, model = make_detector(
        [
            (0, 0, 10, 10, 0.25, 0),  # ball, above its 0.20 override
            (0, 0, 10, 10, 0.25, 2),  # player, below global 0.35
        ],
        conf=0.35,
        overrides={"ball": 0.20},
    )
    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    assert [d.class_name for d in detections] == ["ball"]
    # Inference must have run at the lowest threshold so the ball survives NMS.
    assert model.predict_calls[0]["conf"] == pytest.approx(0.20)


def test_empty_result_returns_empty_list():
    detector, _ = make_detector([])
    assert detector.detect(np.zeros((64, 64, 3), dtype=np.uint8)) == []


def test_unknown_class_id_gets_fallback_name():
    detector, _ = make_detector([(0, 0, 10, 10, 0.9, 7)])
    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert detections[0].class_name == "class_7"


def test_missing_weights_raises():
    config = ModelConfig(weights_path="weights/missing.pt", device="cpu")
    with pytest.raises(FileNotFoundError, match="weights"):
        YOLODetector(config, CLASS_NAMES)
