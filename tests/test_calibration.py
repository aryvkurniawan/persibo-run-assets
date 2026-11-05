"""Tests for Phase 4 — pitch calibration, homography and minimap.

Covered:
    * pitch model coordinates
    * calibration save / load (+ point-in-frame rejection)
    * homography projection (image<->field round-trip)
    * reprojection error
    * minimap drawing + field projection
    * missing-calibration fallback (never crashes)
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from calibration.calibration_store import (
    Calibration,
    CalibrationStore,
    calibration_quality,
    point_in_frame,
)
from calibration.homography import Homography, compute_homography
from calibration.minimap import (
    MinimapRenderer,
    color_for,
    load_roles_map,
    project_tracks_to_field,
)
from calibration.pitch_model import (
    REFERENCE_POINT_NAMES,
    PitchDimensions,
    PitchModel,
)


# ---------------------------------------------------------------------------
# Pitch model
# ---------------------------------------------------------------------------
def test_pitch_model_corner_and_center_coordinates():
    pitch = PitchModel()
    assert pitch.field_point("top_left_corner") == (0.0, 0.0)
    assert pitch.field_point("top_right_corner") == (105.0, 0.0)
    assert pitch.field_point("bottom_left_corner") == (0.0, 68.0)
    assert pitch.field_point("bottom_right_corner") == (105.0, 68.0)
    assert pitch.field_point("center_spot") == (52.5, 34.0)


def test_pitch_model_penalty_and_goal_areas():
    pitch = PitchModel()
    # Penalty spots: 11 m from each goal line.
    assert pitch.field_point("left_penalty_spot") == (11.0, 34.0)
    assert pitch.field_point("right_penalty_spot") == (94.0, 34.0)
    # Penalty area corners: depth 16.5, width 40.32 (half = 20.16).
    assert pitch.field_point("left_penalty_area_top") == pytest.approx((16.5, 13.84))
    assert pitch.field_point("left_penalty_area_bottom") == pytest.approx((16.5, 54.16))
    assert pitch.field_point("right_penalty_area_top") == pytest.approx((88.5, 13.84))
    # Goal area: depth 5.5, width 18.32 (half = 9.16).
    assert pitch.field_point("left_goal_area_top") == pytest.approx((5.5, 24.84))
    assert pitch.field_point("right_goal_area_bottom") == pytest.approx((99.5, 43.16))
    # Center circle marks: radius 9.15.
    assert pitch.field_point("center_circle_top") == pytest.approx((52.5, 24.85))
    assert pitch.field_point("center_circle_bottom") == pytest.approx((52.5, 43.15))


def test_pitch_model_has_all_named_points():
    pitch = PitchModel()
    assert set(pitch.names()) == set(REFERENCE_POINT_NAMES)
    assert len(REFERENCE_POINT_NAMES) == 19
    for name in REFERENCE_POINT_NAMES:
        assert pitch.contains(name)


def test_pitch_dimensions_are_official():
    d = PitchDimensions()
    assert (d.length, d.width) == (105.0, 68.0)
    assert d.penalty_area_depth == 16.5 and d.penalty_area_width == 40.32
    assert d.goal_area_depth == 5.5 and d.goal_area_width == 18.32
    assert d.penalty_spot_distance == 11.0 and d.center_circle_radius == 9.15


# ---------------------------------------------------------------------------
# Calibration store
# ---------------------------------------------------------------------------
def test_point_in_frame():
    assert point_in_frame((10, 10), 100, 100) is True
    assert point_in_frame((0, 0), 100, 100) is True
    assert point_in_frame((100, 100), 100, 100) is True
    assert point_in_frame((-1, 10), 100, 100) is False
    assert point_in_frame((10, 101), 100, 100) is False


def test_calibration_save_load_roundtrip(tmp_path):
    cal = Calibration(video="input_video.mp4", frame=250)
    cal.add_point("center_spot", (960, 540), (52.5, 34.0))
    cal.add_point("top_left_corner", (100, 120), (0.0, 0.0))

    path = tmp_path / "pitch_calibration.json"
    CalibrationStore(path).save(cal)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["video"] == "input_video.mp4"
    assert data["frame"] == 250
    assert data["points"]["center_spot"]["image"] == [960.0, 540.0]
    assert data["points"]["center_spot"]["field"] == [52.5, 34.0]

    loaded = CalibrationStore(path).load()
    assert loaded.count == 2
    assert loaded.points["center_spot"]["field"] == (52.5, 34.0)


def test_calibration_store_missing_returns_none(tmp_path):
    assert CalibrationStore(tmp_path / "nope.json").load() is None


def test_calibration_remove_and_arrays():
    cal = Calibration()
    cal.add_point("a", (1, 2), (10, 20))
    cal.add_point("b", (3, 4), (30, 40))
    assert cal.count == 2
    assert cal.image_points().shape == (2, 2)
    assert cal.field_points().tolist() == [[10, 20], [30, 40]]
    assert cal.remove_point("a") is True
    assert cal.remove_point("a") is False
    assert cal.count == 1


def test_calibration_quality_thresholds():
    assert calibration_quality(2)["usable"] is False
    assert calibration_quality(4)["usable"] is True
    assert calibration_quality(4)["recommended"] is False
    assert calibration_quality(8)["recommended"] is True
    assert calibration_quality(8)["warning"] is None
    assert "recommended" in calibration_quality(5)["warning"]


# ---------------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------------
def _affine_correspondences():
    """Field points and their image points under a known affine map.

    image = field * 10 + (100, 50)  -> a valid (degenerate) homography.
    """
    field = [(0, 0), (105, 0), (105, 68), (0, 68), (52.5, 34), (11, 34)]
    image = [(fx * 10 + 100, fy * 10 + 50) for fx, fy in field]
    return image, field


def test_compute_homography_requires_four_points():
    with pytest.raises(ValueError):
        compute_homography([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])


def test_homography_projection_roundtrip():
    image, field = _affine_correspondences()
    H = Homography.from_correspondences(image, field)
    # image -> field recovers the known field coordinate.
    fx, fy = H.project_image_to_field((52.5 * 10 + 100, 34 * 10 + 50))
    assert fx == pytest.approx(52.5, abs=1e-3)
    assert fy == pytest.approx(34.0, abs=1e-3)
    # field -> image is the inverse.
    ix, iy = H.project_field_to_image((52.5, 34.0))
    assert ix == pytest.approx(52.5 * 10 + 100, abs=1e-2)
    assert iy == pytest.approx(34.0 * 10 + 50, abs=1e-2)


def test_homography_reprojection_error_small_for_consistent_points():
    image, field = _affine_correspondences()
    H = Homography.from_correspondences(image, field)
    assert H.reprojection_error() < 1e-3


def test_homography_reprojection_error_grows_with_bad_point():
    image, field = _affine_correspondences()
    image = list(image)
    image[0] = (image[0][0] + 200, image[0][1] + 200)  # mis-click one corner
    H = Homography.from_correspondences(image, field)
    assert H.reprojection_error() > 1.0


def test_homography_uses_ransac_above_six_points():
    image, field = _affine_correspondences()   # exactly 6 points
    # Should solve without error (RANSAC path).
    H = compute_homography(image, field)
    assert H.shape == (3, 3)


# ---------------------------------------------------------------------------
# Minimap
# ---------------------------------------------------------------------------
def test_minimap_blank_pitch_has_lines():
    renderer = MinimapRenderer(px_per_meter=6)
    img = renderer.blank()
    assert img.shape[2] == 3
    # The white lines must be present somewhere on the green pitch.
    white = np.all(img > 200, axis=2)
    assert white.sum() > 0


def test_minimap_field_to_px_inside_canvas():
    renderer = MinimapRenderer(px_per_meter=8, margin=20)
    px, py = renderer.field_to_px(52.5, 34.0)
    assert 0 <= px < renderer.width_px
    assert 0 <= py < renderer.height_px
    assert renderer.on_pitch(52.5, 34.0) is True
    assert renderer.on_pitch(-1, 34.0) is False


def test_minimap_render_draws_dot():
    renderer = MinimapRenderer(px_per_meter=8)
    blank = renderer.blank()
    out = renderer.render([{"field": (52.5, 34.0), "color": (255, 0, 0),
                            "role": "player"}])
    # Rendering a dot must change some pixels vs the blank pitch.
    assert np.any(out != blank)


def test_minimap_hides_referees():
    renderer = MinimapRenderer(px_per_meter=8, hide_referees=True)
    blank = renderer.blank()
    # A single referee object -> nothing drawn (referees are hidden).
    out = renderer.render([{"field": (52.5, 34.0), "color": (0, 255, 255),
                            "role": "referee"}])
    assert np.array_equal(out, blank)


def test_minimap_tactical_style_draws_team_shape():
    renderer = MinimapRenderer(px_per_meter=8, tactical_style=True)
    plain = MinimapRenderer(px_per_meter=8)
    objs = [{"field": (30.0 + 5*i, 20.0 + 8*(i % 3)), "color": (255, 128, 0),
             "role": "player", "team_id": 0} for i in range(4)]
    out = renderer.render(objs)
    base = plain.render(objs)
    # The hull/lines overlay makes the tactical render differ from the plain one.
    assert not np.array_equal(out, base)


def test_color_for_roles():
    assert color_for("player", 0) == (255, 128, 0)
    assert color_for("player", 1) == (0, 128, 255)
    assert color_for("referee", None) == (0, 255, 255)
    assert color_for("ball", None) == (255, 255, 255)
    assert color_for("unknown", None) == (170, 170, 170)


class _FakeTrack:
    def __init__(self, track_id, class_name, bbox):
        self.track_id = track_id
        self.class_name = class_name
        self.bbox = bbox


class _FakeFrame:
    def __init__(self, frame_index, tracks):
        self.frame_index = frame_index
        self.tracks = tracks


def test_project_tracks_to_field_uses_foot_point():
    image, field = _affine_correspondences()
    H = Homography.from_correspondences(image, field)
    renderer = MinimapRenderer()
    # A player bbox whose foot (bottom-center) maps to a known image point
    # image=(625, 390) -> field (52.5, 34).
    foot_img = (52.5 * 10 + 100, 34 * 10 + 50)
    bbox = (foot_img[0] - 5, foot_img[1] - 40, foot_img[0] + 5, foot_img[1])
    frames = [_FakeFrame(1, [_FakeTrack(7, "player", bbox)])]
    roles = {7: ("player", 1)}
    positions = project_tracks_to_field(frames, roles, H, renderer)

    assert len(positions) == 1
    obj = positions[0]["objects"][0]
    assert obj["track_id"] == 7
    assert obj["role"] == "player" and obj["team_id"] == 1
    assert obj["field"][0] == pytest.approx(52.5, abs=1e-2)
    assert obj["field"][1] == pytest.approx(34.0, abs=1e-2)


def test_load_roles_map_handles_final_and_auto(tmp_path):
    final = tmp_path / "final.json"
    final.write_text(json.dumps({"tracks": [
        {"track_id": 1, "final_role": "referee", "team_id": None},
        {"track_id": 2, "final_role": "player", "team_id": 0},
    ]}), encoding="utf-8")
    roles = load_roles_map(final)
    assert roles[1] == ("referee", None)
    assert roles[2] == ("player", 0)

    auto = tmp_path / "auto.json"
    auto.write_text(json.dumps({"tracks": [
        {"track_id": 9, "refined_role": "goalkeeper", "team_id": None},
    ]}), encoding="utf-8")
    assert load_roles_map(auto)[9] == ("goalkeeper", None)


# ---------------------------------------------------------------------------
# Missing-calibration fallback (the pipeline must never crash)
# ---------------------------------------------------------------------------
def test_missing_calibration_fallback(tmp_path):
    # Loading an absent calibration returns None rather than raising.
    store = CalibrationStore(tmp_path / "absent.json")
    assert store.exists() is False
    assert store.load() is None


def test_run_minimap_skips_without_calibration(tmp_path):
    # The main.run_minimap helper must skip (return None) when calibration
    # is missing — never raise.
    import main
    from utils.config_loader import AppConfig

    config = AppConfig()
    config.calibration.calibration_path = str(tmp_path / "missing.json")
    summary = {"tracks_json": None}
    assert main.run_minimap(config, "input_video.mp4", summary, None) is None
