"""Tests for Phase 7 — heatmaps."""

from __future__ import annotations

import numpy as np

from analytics.heatmap_generator import collect_positions, density_grid
from analytics.heatmap_renderer import render_heatmap, save_heatmap
from calibration.pitch_model import PitchDimensions


def _frame(idx, objects):
    return {"frame": idx, "objects": objects}


def _obj(tid, x, y, role="player", team=0, on_pitch=True):
    return {"track_id": tid, "role": role, "team_id": team,
            "field": [x, y], "on_pitch": on_pitch}


# ---------------------------------------------------------------------------
# Collecting positions
# ---------------------------------------------------------------------------
def test_collect_positions_filters_team_and_skips_offpitch_and_ball():
    frames = [_frame(1, [
        _obj(1, 30.0, 20.0, team=0),
        _obj(2, 70.0, 40.0, team=1),
        _obj(3, 50.0, 30.0, team=0, on_pitch=False),     # off-pitch, dropped
        _obj(9, 52.5, 34.0, role="ball", team=None),     # ball, dropped
    ])]
    team0 = collect_positions(frames, team_id=0)
    assert team0 == [(30.0, 20.0)]                       # off-pitch one excluded


def test_collect_positions_by_track():
    frames = [
        _frame(1, [_obj(7, 10.0, 10.0)]),
        _frame(2, [_obj(7, 12.0, 11.0), _obj(8, 90.0, 60.0)]),
    ]
    assert collect_positions(frames, track_id=7) == [(10.0, 10.0), (12.0, 11.0)]


# ---------------------------------------------------------------------------
# Heatmap generation + smoothing
# ---------------------------------------------------------------------------
def test_density_grid_shape_and_normalization():
    dims = PitchDimensions()
    grid = density_grid([(52.5, 34.0)], dims=dims, px_per_meter=4.0, sigma_m=2.0)
    assert grid.shape == (round(dims.width * 4), round(dims.length * 4))
    assert grid.max() == 1.0                             # normalized to peak
    assert grid.min() >= 0.0


def test_empty_points_give_zero_grid():
    grid = density_grid([], px_per_meter=4.0)
    assert grid.max() == 0.0


def test_smoothing_spreads_a_single_point():
    pt = [(52.5, 34.0)]
    sharp = density_grid(pt, px_per_meter=4.0, sigma_m=0.0)
    smooth = density_grid(pt, px_per_meter=4.0, sigma_m=3.0)
    # No smoothing -> exactly one hot cell; smoothing spreads it to many.
    assert int((sharp > 0).sum()) == 1
    assert int((smooth > 0.05).sum()) > 5


def test_density_peaks_where_points_cluster():
    pts = [(20.0, 15.0)] * 30 + [(85.0, 55.0)]           # cluster near (20,15)
    grid = density_grid(pts, px_per_meter=4.0, sigma_m=2.0)
    gy, gx = np.unravel_index(int(grid.argmax()), grid.shape)
    assert abs(gx / 4.0 - 20.0) < 5 and abs(gy / 4.0 - 15.0) < 5


# ---------------------------------------------------------------------------
# Rendering + export
# ---------------------------------------------------------------------------
def test_render_heatmap_returns_bgr_image():
    grid = density_grid([(52.5, 34.0)] * 10, px_per_meter=4.0, sigma_m=2.0)
    img = render_heatmap(grid, label="Team 0")
    assert img.ndim == 3 and img.shape[2] == 3
    assert np.any(img != 0)


def test_save_heatmap_writes_png(tmp_path):
    grid = density_grid([(30.0, 20.0)] * 5, px_per_meter=4.0, sigma_m=2.0)
    path = save_heatmap(grid, tmp_path / "team0_heatmap.png", label="Team 0")
    assert path.exists() and path.stat().st_size > 0


def test_render_overlay_keeps_pitch_where_empty():
    # An all-zero grid -> the heatmap is fully transparent -> identical to the
    # bare pitch (lines stay visible).
    from calibration.minimap import MinimapRenderer
    renderer = MinimapRenderer(px_per_meter=6, stripes=12)
    blank = renderer.blank()
    grid = density_grid([], px_per_meter=4.0)
    out = render_heatmap(grid, renderer=renderer)
    assert np.array_equal(out, blank)
