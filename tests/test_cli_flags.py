"""``--save-analytics`` must not turn on minimap video rendering."""

import pytest

from main import apply_overrides, parse_args
from utils.config_loader import load_config


@pytest.fixture
def config():
    return load_config("config/config.yaml")


def test_save_analytics_does_not_enable_minimap_render(config):
    args = parse_args(["--video", "x.mp4", "--save-analytics"])
    config = apply_overrides(config, args)
    assert config.analytics.enabled is True
    assert config.calibration.save_minimap is False


def test_explicit_save_minimap_still_renders(config):
    args = parse_args(["--video", "x.mp4", "--save-minimap"])
    config = apply_overrides(config, args)
    assert config.calibration.save_minimap is True
