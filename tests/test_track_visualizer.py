import numpy as np

from tracking.models import Track
from tracking.track_visualizer import TrackVisualizer, color_for_track_id
from utils.config_loader import VisualizationConfig


def make_frame(size=200):
    return np.zeros((size, size, 3), dtype=np.uint8)


def make_track(track_id=5, **overrides):
    defaults = dict(
        frame_id=1,
        track_id=track_id,
        class_id=2,
        class_name="player",
        confidence=0.9,
        bbox=(20, 30, 80, 150),
        source_detection_id=0,
    )
    defaults.update(overrides)
    return Track(**defaults)


def make_visualizer(debug=False):
    return TrackVisualizer(VisualizationConfig(), debug=debug)


def test_color_is_stable_and_distinct_per_id():
    # Same ID -> same color (stable); different IDs -> different colors.
    assert color_for_track_id(5) == color_for_track_id(5)
    assert color_for_track_id(5) != color_for_track_id(6)


def test_annotate_draws_on_copy():
    frame = make_frame()
    original = frame.copy()
    annotated = make_visualizer().annotate(frame, [make_track()])

    assert np.array_equal(frame, original), "input frame must not be mutated"
    assert not np.array_equal(annotated, original), "track must be drawn"


def test_two_ids_produce_different_pixels():
    # The two boxes overlap-free; coloring by ID should differ between them.
    a = make_visualizer().annotate(make_frame(), [make_track(track_id=1)])
    b = make_visualizer().annotate(make_frame(), [make_track(track_id=2)])
    assert not np.array_equal(a, b)


def test_no_tracks_returns_unchanged_copy():
    frame = make_frame()
    assert np.array_equal(make_visualizer().annotate(frame, []), frame)


def test_debug_hud_draws():
    annotated = make_visualizer(debug=True).annotate(
        make_frame(), [make_track()], frame_index=3, fps=20.0
    )
    assert not np.array_equal(annotated, make_frame())
