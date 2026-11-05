import pytest

from utils.config_loader import AppConfig, load_config

MINIMAL_YAML = """
model:
  weights_path: "weights/best.pt"
  confidence_threshold: 0.4
  class_confidence_overrides:
    ball: 0.15
classes:
  0: ball
  1: goalkeeper
  2: player
  3: referee
visualization:
  colors:
    player: [255, 0, 0]
"""


def write_config(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_minimal_config(tmp_path):
    config = load_config(write_config(tmp_path, MINIMAL_YAML))

    assert isinstance(config, AppConfig)
    assert config.model.confidence_threshold == 0.4
    assert config.model.class_confidence_overrides == {"ball": 0.15}
    assert config.classes == {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
    # Colors become BGR tuples.
    assert config.visualization.colors["player"] == (255, 0, 0)
    # Missing sections fall back to defaults.
    assert config.video.save_video is True
    assert config.debug.enabled is False
    assert config.logging.level == "INFO"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")


def test_invalid_confidence_raises(tmp_path):
    bad = MINIMAL_YAML.replace("confidence_threshold: 0.4",
                               "confidence_threshold: 1.5")
    with pytest.raises(ValueError, match="confidence_threshold"):
        load_config(write_config(tmp_path, bad))


def test_invalid_override_raises(tmp_path):
    bad = MINIMAL_YAML.replace("ball: 0.15", "ball: -0.2")
    with pytest.raises(ValueError, match="override"):
        load_config(write_config(tmp_path, bad))


def test_missing_classes_raises(tmp_path):
    bad = "model:\n  confidence_threshold: 0.4\n"
    with pytest.raises(ValueError, match="classes"):
        load_config(write_config(tmp_path, bad))
