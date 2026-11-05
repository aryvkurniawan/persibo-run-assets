"""Tests for the picture-in-picture minimap overlay (Phase 4)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from calibration.minimap import (
    OVERLAY_POSITIONS,
    MinimapRenderer,
    overlay_geometry,
    overlay_minimap,
    render_overlay_video,
)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------
def test_overlay_placement_all_corners():
    W, H, margin = 1000, 600, 20
    # minimap 200x130 -> with default ratio we control via width_ratio below.
    # Use a fixed-size mini to get a predictable box.
    mini_w, mini_h = 280, 182          # aspect ~ minimap (105:68)
    common = dict(base_w=W, base_h=H, mini_w=mini_w, mini_h=mini_h,
                  width_ratio=0.28, margin=margin)

    g_tr = overlay_geometry(position="top_right", **common)
    g_tl = overlay_geometry(position="top_left", **common)
    g_br = overlay_geometry(position="bottom_right", **common)
    g_bl = overlay_geometry(position="bottom_left", **common)

    box_w, box_h = g_tr["box_w"], g_tr["box_h"]
    assert (g_tl["x"], g_tl["y"]) == (margin, margin)
    assert (g_tr["x"], g_tr["y"]) == (W - box_w - margin, margin)
    assert (g_br["x"], g_br["y"]) == (W - box_w - margin, H - box_h - margin)
    assert (g_bl["x"], g_bl["y"]) == (margin, H - box_h - margin)


def test_overlay_default_position_is_top_right():
    g = overlay_geometry(1000, 600, 280, 182)   # defaults
    assert g["x"] > 500 and g["y"] < 100        # right-ish, top-ish


def test_all_positions_supported():
    assert set(OVERLAY_POSITIONS) == {
        "top_right", "top_left", "bottom_right", "bottom_left"}


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
def test_overlay_sizing_matches_width_ratio():
    g = overlay_geometry(1920, 1080, 888, 600, width_ratio=0.28, margin=20)
    assert g["inner_w"] == round(1920 * 0.28)            # 538
    # Aspect ratio preserved.
    assert g["inner_h"] == pytest.approx(round(g["inner_w"] * 600 / 888), abs=1)


def test_overlay_sizing_clamped_to_small_frame():
    # Overlay must still fit (no negative / oversized box) on a tiny frame.
    g = overlay_geometry(80, 60, 888, 600, width_ratio=0.9, margin=5)
    assert g["box_w"] <= 80 and g["box_h"] <= 60
    assert g["x"] >= 0 and g["y"] >= 0
    assert g["inner_w"] >= 8 and g["inner_h"] >= 8


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------
def test_overlay_minimap_preserves_shape_and_copies():
    base = np.full((600, 1000, 3), 90, np.uint8)
    renderer = MinimapRenderer(px_per_meter=6)
    mini = renderer.blank()
    out = overlay_minimap(base, mini, position="top_right")
    assert out.shape == base.shape
    # Base is not mutated.
    assert np.all(base == 90)
    # The overlay region changed; a far corner (bottom-left) did not.
    assert np.any(out[20:200, 700:980] != 90)
    assert np.all(out[580:600, 0:200] == 90)


def test_overlay_only_touches_chosen_corner():
    base = np.full((600, 1000, 3), 90, np.uint8)
    mini = MinimapRenderer(px_per_meter=6).blank()
    out = overlay_minimap(base, mini, position="bottom_left", margin=20)
    # Top-right stays untouched; bottom-left changed.
    assert np.all(out[0:150, 850:1000] == 90)
    assert np.any(out[450:590, 20:300] != 90)


def test_overlay_border_can_be_disabled():
    base = np.full((600, 1000, 3), 90, np.uint8)
    mini = MinimapRenderer(px_per_meter=6).blank()
    with_border = overlay_minimap(base, mini, border=True)
    no_border = overlay_minimap(base, mini, border=False)
    # The two differ only along the border pixels.
    assert np.any(with_border != no_border)


# ---------------------------------------------------------------------------
# Output video path / round-trip + standalone unaffected
# ---------------------------------------------------------------------------
def _write_base_video(path, n_frames=5, w=320, h=240, fps=10.0):
    from utils.video_io import VideoWriter

    writer = VideoWriter(path, fps=fps, frame_size=(w, h))
    try:
        for i in range(n_frames):
            frame = np.full((h, w, 3), 60, np.uint8)
            frame[:, :, 0] = i * 10          # vary so frames differ
            writer.write(frame)
    finally:
        writer.release()
    return n_frames


def _positions(n_frames):
    return [
        {"frame": i + 1, "objects": [
            {"field": (52.5, 34.0), "color": (255, 0, 0), "role": "player"}]}
        for i in range(n_frames)
    ]


def test_render_overlay_video_writes_output(tmp_path):
    base = tmp_path / "annotated.mp4"
    n = _write_base_video(base, n_frames=6)
    out = tmp_path / "input_video_final_with_minimap.mp4"
    renderer = MinimapRenderer(px_per_meter=5)
    result = render_overlay_video(base, _positions(n), renderer, out, fps=10.0)

    assert result == out
    assert out.is_file() and out.stat().st_size > 0
    # The output is a readable video with the same resolution as the base.
    from utils.video_io import VideoReader
    with VideoReader(out) as reader:
        assert (reader.width, reader.height) == (320, 240)
        first = next(reader.frames(), None)
        assert first is not None


def test_render_overlay_video_handles_short_positions(tmp_path):
    # Fewer position entries than base frames -> remaining frames get no dots
    # (must not raise / index-error).
    base = tmp_path / "annotated.mp4"
    n = _write_base_video(base, n_frames=5)
    out = tmp_path / "out.mp4"
    renderer = MinimapRenderer(px_per_meter=5)
    render_overlay_video(base, _positions(2), renderer, out, fps=10.0)
    assert out.is_file()


class _Track:
    def __init__(self, tid, name, bbox):
        self.track_id, self.class_name, self.bbox = tid, name, bbox


class _Frame:
    def __init__(self, idx, tracks):
        self.frame_index, self.tracks = idx, tracks


def test_field_positions_export_includes_player_names(tmp_path):
    from calibration.homography import Homography
    from calibration.minimap import project_tracks_to_field, write_field_positions

    frames = [_Frame(1, [_Track(1, "player", (10, 10, 20, 20))])]
    renderer = MinimapRenderer(px_per_meter=4)
    positions = project_tracks_to_field(
        frames,
        {1: ("player", 0)},
        Homography(np.eye(3)),
        renderer,
        names_map={1: "Christine Sinclair"},
        smooth_alpha=1.0,
        max_step_m=1000.0,
    )
    assert positions[0]["objects"][0]["player_name"] == "Christine Sinclair"
    assert positions[0]["objects"][0]["player_display_name"] == "Christine Sinclair"

    out = write_field_positions(positions, tmp_path / "field_positions.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["frames"][0]["objects"][0]["player_name"] == "Christine Sinclair"
    assert data["frames"][0]["objects"][0]["player_display_name"] == (
        "Christine Sinclair")


def test_field_positions_use_display_name_without_manual_name(tmp_path):
    from calibration.homography import Homography
    from calibration.minimap import project_tracks_to_field, write_field_positions

    frames = [_Frame(1, [_Track(4, "player", (10, 10, 20, 20))])]
    renderer = MinimapRenderer(px_per_meter=4)
    positions = project_tracks_to_field(
        frames,
        {4: ("player", 0)},
        Homography(np.eye(3)),
        renderer,
        smooth_alpha=1.0,
        max_step_m=1000.0,
    )

    out = write_field_positions(positions, tmp_path / "field_positions.json")
    obj = json.loads(out.read_text(encoding="utf-8"))["frames"][0]["objects"][0]
    assert obj["player_display_name"] == "Player 4"
    assert "player_name" not in obj


def test_combined_final_video_has_team_boxes_and_minimap(tmp_path):
    from calibration.minimap import (
        MinimapRenderer, color_for_object, render_final_video)
    from utils.video_io import VideoReader

    BLUE = (255, 128, 0)
    GREEN = (45, 110, 45)   # minimap pitch

    source = tmp_path / "input.mp4"
    n = _write_base_video(source, n_frames=3, w=400, h=300)

    frames = [_Frame(i + 1, [_Track(1, "player", (60, 60, 120, 200))])
              for i in range(n)]
    positions = [
        {"frame": i + 1, "objects": [
            {"field": (52.5, 34.0), "role": "player",
             "color": color_for_object("player", 0, 1)}]}
        for i in range(n)
    ]
    roles_map = {1: ("player", 0)}                  # team 0 -> blue box
    renderer = MinimapRenderer(px_per_meter=4)

    out = tmp_path / "input_final_with_minimap.mp4"
    render_final_video(source, frames, positions, roles_map, renderer, out,
                       width_ratio=0.5, background_alpha=0.0)
    assert out.is_file()
    with VideoReader(out) as reader:
        _, frame = next(reader.frames())

    # Lossy mp4 shifts exact pixel values, so match within a tolerance: the
    # team-colored box on the player AND the minimap pitch are both present.
    def has_color(img, color, tol=40):
        diff = np.abs(img.astype(int) - np.array(color)).sum(axis=2)
        return bool(np.any(diff <= tol))

    assert has_color(frame, BLUE)
    assert has_color(frame, GREEN)


def test_standalone_minimap_video_still_works(tmp_path):
    # The standalone minimap renderer is independent of the overlay path.
    from calibration.minimap import render_minimap_video

    renderer = MinimapRenderer(px_per_meter=5)
    out = tmp_path / "input_video_minimap.mp4"
    render_minimap_video(_positions(4), renderer, out, fps=10.0)
    assert out.is_file() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Missing-calibration fallback through main.run_minimap
# ---------------------------------------------------------------------------
def test_overlay_skips_when_calibration_missing(tmp_path):
    import main
    from utils.config_loader import AppConfig

    config = AppConfig()
    config.minimap_overlay.enabled = True
    config.calibration.save_minimap = True
    config.calibration.calibration_path = str(tmp_path / "missing.json")
    summary = {"tracks_json": None, "annotated_video": None}
    # Must return None (skipped) and never raise.
    assert main.run_minimap(config, "input_video.mp4", summary, None) is None


def test_overlay_config_validation_rejects_bad_position():
    from utils.config_loader import MinimapOverlayConfig, _validate_minimap_overlay

    with pytest.raises(ValueError):
        _validate_minimap_overlay(MinimapOverlayConfig(position="middle"))
    with pytest.raises(ValueError):
        _validate_minimap_overlay(MinimapOverlayConfig(width_ratio=1.5))
