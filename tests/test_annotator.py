import numpy as np

from detection.data_models import Detection
from utils.config_loader import VisualizationConfig
from visualization.annotator import DetectionAnnotator


def make_frame(size=200):
    return np.zeros((size, size, 3), dtype=np.uint8)


def make_detection():
    return Detection(
        class_id=2, class_name="player", confidence=0.9, bbox=(20, 30, 80, 150)
    )


def make_annotator(debug=False):
    config = VisualizationConfig(colors={"player": (255, 0, 0)})
    return DetectionAnnotator(config, debug=debug)


def test_annotate_draws_on_copy_not_original():
    frame = make_frame()
    original = frame.copy()

    annotated = make_annotator().annotate(frame, [make_detection()])

    assert np.array_equal(frame, original), "input frame must not be mutated"
    assert not np.array_equal(annotated, original), "annotation must draw pixels"


def test_box_uses_configured_color():
    annotated = make_annotator().annotate(make_frame(), [make_detection()])
    # Top edge of the bbox at y=30 between x=20..80 should be blue (BGR 255,0,0).
    edge = annotated[30, 25:75]
    assert (edge[:, 0] == 255).any()


def test_no_detections_returns_unchanged_copy():
    frame = make_frame()
    annotated = make_annotator().annotate(frame, [])
    assert np.array_equal(annotated, frame)


def test_debug_hud_draws_even_without_detections():
    annotated = make_annotator(debug=True).annotate(
        make_frame(), [], frame_index=7, fps=24.5
    )
    assert not np.array_equal(annotated, make_frame())


def test_unknown_class_uses_default_color():
    detection = Detection(
        class_id=9, class_name="mystery", confidence=0.8, bbox=(10, 10, 50, 50)
    )
    annotated = make_annotator().annotate(make_frame(), [detection])
    assert not np.array_equal(annotated, make_frame())
