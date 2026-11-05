import pytest

from utils.config_loader import TrackingConfig, load_config

BASE = """
model:
  weights_path: "weights/best.pt"
classes:
  0: ball
  1: goalkeeper
  2: player
  3: referee
"""

TRACKING = """
tracking:
  enabled: true
  tracker_type: botsort
  separate_ball: true
  with_reid: false
  main:
    track_high_thresh: 0.30
    track_buffer: 25
  ball:
    new_track_thresh: 0.08
"""


def write(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_defaults_when_section_absent(tmp_path):
    config = load_config(write(tmp_path, BASE))
    assert isinstance(config.tracking, TrackingConfig)
    assert config.tracking.enabled is False
    assert config.tracking.tracker_type == "botsort"
    # Ball preset is looser than the main tracker by default.
    assert config.tracking.ball.new_track_thresh == 0.10
    assert config.tracking.ball.track_buffer == 60
    assert config.tracking.main.track_buffer == 30


def test_overrides_and_partial_nested_merge(tmp_path):
    config = load_config(write(tmp_path, BASE + TRACKING))
    assert config.tracking.enabled is True
    assert config.tracking.main.track_high_thresh == 0.30
    assert config.tracking.main.track_buffer == 25
    # Unspecified main fields keep their defaults.
    assert config.tracking.main.match_thresh == 0.80
    # Ball: only new_track_thresh overridden; rest keep the looser preset.
    assert config.tracking.ball.new_track_thresh == 0.08
    assert config.tracking.ball.track_buffer == 60


def test_invalid_tracker_type_raises(tmp_path):
    bad = BASE + "tracking:\n  tracker_type: deepsort\n"
    with pytest.raises(ValueError, match="tracker_type"):
        load_config(write(tmp_path, bad))


def test_invalid_threshold_raises(tmp_path):
    bad = BASE + "tracking:\n  main:\n    match_thresh: 1.4\n"
    with pytest.raises(ValueError, match="match_thresh"):
        load_config(write(tmp_path, bad))


def test_invalid_track_buffer_raises(tmp_path):
    bad = BASE + "tracking:\n  ball:\n    track_buffer: 0\n"
    with pytest.raises(ValueError, match="track_buffer"):
        load_config(write(tmp_path, bad))


def test_ball_tracking_defaults(tmp_path):
    config = load_config(write(tmp_path, BASE))
    bt = config.tracking.ball_tracking
    assert bt.enabled is True
    assert bt.matching_box_scale == 3.0
    assert bt.min_box_size == 20.0
    assert bt.max_frame_gap == 30
    assert bt.distance_gate_px == 50.0


def test_ball_tracking_overrides(tmp_path):
    extra = (
        "tracking:\n"
        "  ball_tracking:\n"
        "    enabled: false\n"
        "    matching_box_scale: 4.0\n"
        "    distance_gate_px: 80\n"
    )
    config = load_config(write(tmp_path, BASE + extra))
    bt = config.tracking.ball_tracking
    assert bt.enabled is False
    assert bt.matching_box_scale == 4.0
    assert bt.distance_gate_px == 80
    # Unspecified fields keep defaults.
    assert bt.min_box_size == 20.0


def test_invalid_matching_box_scale_raises(tmp_path):
    bad = BASE + "tracking:\n  ball_tracking:\n    matching_box_scale: 0\n"
    with pytest.raises(ValueError, match="matching_box_scale"):
        load_config(write(tmp_path, bad))


def test_invalid_distance_gate_raises(tmp_path):
    bad = BASE + "tracking:\n  ball_tracking:\n    distance_gate_px: -5\n"
    with pytest.raises(ValueError, match="distance_gate_px"):
        load_config(write(tmp_path, bad))
