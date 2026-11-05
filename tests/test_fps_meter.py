import time

import pytest

from utils.fps_meter import FPSMeter


def test_basic_measurement():
    meter = FPSMeter()
    for _ in range(3):
        meter.start()
        time.sleep(0.01)
        meter.stop()

    assert meter.frame_count == 3
    assert meter.total_time > 0
    assert 0 < meter.average_fps <= 100  # each frame took >= 10 ms
    assert meter.current_fps > 0


def test_stop_without_start_raises():
    with pytest.raises(RuntimeError):
        FPSMeter().stop()


def test_empty_meter_reports_zero():
    meter = FPSMeter()
    assert meter.average_fps == 0.0
    assert meter.current_fps == 0.0
    assert meter.summary() == {"frames": 0, "total_seconds": 0.0, "average_fps": 0.0}


def test_invalid_window_raises():
    with pytest.raises(ValueError):
        FPSMeter(window_size=0)


def test_summary_keys():
    meter = FPSMeter()
    meter.start()
    meter.stop()
    summary = meter.summary()
    assert set(summary) == {"frames", "total_seconds", "average_fps"}
    assert summary["frames"] == 1
