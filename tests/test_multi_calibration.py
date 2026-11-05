"""Tests for multi-frame (panning-camera) pitch calibration."""

from __future__ import annotations

import json

import pytest

from calibration.calibration_store import (
    Calibration,
    MultiCalibration,
    MultiCalibrationStore,
)
from calibration.homography import Homography, MultiHomography
from calibration.minimap import MinimapRenderer, project_tracks_to_field


def _keyframe(frame, offset):
    """A keyframe whose image = field*10 + (offset, offset)."""
    cal = Calibration(video="v.mp4", frame=frame)
    for fx, fy in [(0, 0), (105, 0), (105, 68), (0, 68)]:
        cal.add_point(f"{fx}_{fy}", (fx * 10 + offset, fy * 10 + offset), (fx, fy))
    return cal


# ---------------------------------------------------------------------------
# Multi-calibration save / load (+ backward compatibility)
# ---------------------------------------------------------------------------
def test_multi_calibration_save_load_roundtrip(tmp_path):
    mc = MultiCalibration(video="input_video.mp4",
                          keyframes=[_keyframe(120, 0), _keyframe(600, 5)])
    path = tmp_path / "pitch_calibration.json"
    MultiCalibrationStore(path).save(mc)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["video"] == "input_video.mp4"
    assert len(data["frames"]) == 2
    assert data["frames"][0]["frame"] == 120

    loaded = MultiCalibrationStore(path).load()
    assert len(loaded.keyframes) == 2
    assert loaded.frame_numbers() == [120, 600]
    assert loaded.total_points == 8


def test_multi_calibration_reads_legacy_single_format(tmp_path):
    # A legacy single-frame calibration file must load as one keyframe.
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "video": "input_video.mp4",
        "frame": 250,
        "points": {
            "center_spot": {"image": [960, 540], "field": [52.5, 34.0]},
            "top_left_corner": {"image": [100, 120], "field": [0.0, 0.0]},
        },
    }), encoding="utf-8")
    mc = MultiCalibrationStore(path).load()
    assert len(mc.keyframes) == 1
    assert mc.keyframes[0].frame == 250
    assert mc.total_points == 2


def test_multi_calibration_store_missing_returns_none(tmp_path):
    assert MultiCalibrationStore(tmp_path / "nope.json").load() is None


def test_usable_keyframes_filters_sparse_frames():
    sparse = Calibration(frame=10)
    sparse.add_point("a", (1, 1), (0, 0))         # only 1 point -> unusable
    mc = MultiCalibration(keyframes=[_keyframe(120, 0), sparse])
    usable = mc.usable_keyframes()
    assert len(usable) == 1
    assert usable[0].frame == 120


# ---------------------------------------------------------------------------
# MultiHomography: nearest-frame selection
# ---------------------------------------------------------------------------
def test_multi_homography_selects_nearest_keyframe():
    mc = MultiCalibration(keyframes=[_keyframe(100, 0), _keyframe(500, 1000)])
    mh = MultiHomography.from_keyframes(mc.usable_keyframes())
    assert len(mh) == 2
    assert mh.frame_numbers == [100, 500]

    # Frame 120 is closest to keyframe 100 (offset 0): image=field*10.
    h_near_100 = mh.for_frame(120)
    fx, fy = h_near_100.project_image_to_field((52.5 * 10 + 0, 34 * 10 + 0))
    assert fx == pytest.approx(52.5, abs=1e-2) and fy == pytest.approx(34.0, abs=1e-2)

    # Frame 480 is closest to keyframe 500 (offset 1000).
    h_near_500 = mh.for_frame(480)
    fx, fy = h_near_500.project_image_to_field((52.5 * 10 + 1000, 34 * 10 + 1000))
    assert fx == pytest.approx(52.5, abs=1e-2) and fy == pytest.approx(34.0, abs=1e-2)


def test_multi_homography_requires_at_least_one():
    with pytest.raises(ValueError):
        MultiHomography([])


def test_multi_homography_reprojection_error():
    mc = MultiCalibration(keyframes=[_keyframe(100, 0), _keyframe(500, 50)])
    mh = MultiHomography.from_keyframes(mc.usable_keyframes())
    assert mh.reprojection_error() < 1e-3


# ---------------------------------------------------------------------------
# Projection uses the per-frame homography
# ---------------------------------------------------------------------------
class _Track:
    def __init__(self, tid, name, bbox):
        self.track_id, self.class_name, self.bbox = tid, name, bbox


class _Frame:
    def __init__(self, idx, tracks):
        self.frame_index, self.tracks = idx, tracks


def test_project_tracks_uses_multi_homography_per_frame():
    # Two keyframes with different offsets; a track at the SAME image point
    # must project to the SAME field point on each, because each frame uses
    # its own (correct) homography.
    mc = MultiCalibration(keyframes=[_keyframe(100, 0), _keyframe(500, 300)])
    mh = MultiHomography.from_keyframes(mc.usable_keyframes())
    renderer = MinimapRenderer()

    # Frame near keyframe 100: a foot point that maps to (52.5, 34) under offset 0.
    foot100 = (52.5 * 10 + 0, 34 * 10 + 0)
    bbox100 = (foot100[0] - 5, foot100[1] - 40, foot100[0] + 5, foot100[1])
    # Frame near keyframe 500: foot point under offset 300.
    foot500 = (52.5 * 10 + 300, 34 * 10 + 300)
    bbox500 = (foot500[0] - 5, foot500[1] - 40, foot500[0] + 5, foot500[1])

    frames = [
        _Frame(110, [_Track(1, "player", bbox100)]),   # -> uses keyframe 100
        _Frame(490, [_Track(1, "player", bbox500)]),   # -> uses keyframe 500
    ]
    roles = {1: ("player", 0)}
    positions = project_tracks_to_field(frames, roles, mh, renderer)

    # Both frames should land the player at the same field spot (52.5, 34).
    for entry in positions:
        fx, fy = entry["objects"][0]["field"]
        assert fx == pytest.approx(52.5, abs=1e-1)
        assert fy == pytest.approx(34.0, abs=1e-1)


def test_single_homography_still_works_in_projection():
    # Backward compatibility: a plain Homography (no for_frame) still projects.
    kf = _keyframe(1, 0)
    H = Homography.from_correspondences(kf.image_points(), kf.field_points())
    renderer = MinimapRenderer()
    foot = (52.5 * 10, 34 * 10)
    bbox = (foot[0] - 5, foot[1] - 40, foot[0] + 5, foot[1])
    frames = [_Frame(1, [_Track(1, "player", bbox)])]
    pos = project_tracks_to_field(frames, {1: ("player", 0)}, H, renderer)
    fx, fy = pos[0]["objects"][0]["field"]
    assert fx == pytest.approx(52.5, abs=1e-2) and fy == pytest.approx(34.0, abs=1e-2)
