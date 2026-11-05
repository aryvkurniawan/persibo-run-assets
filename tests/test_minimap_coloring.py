"""Tests for minimap team/role coloring (analytics) vs track-id (debug)."""

from __future__ import annotations

import numpy as np
import pytest

from calibration.homography import Homography
from calibration.minimap import (
    COLOR_MODE_TEAM_ROLE,
    COLOR_MODE_TRACK_ID,
    MinimapRenderer,
    color_for_object,
    overlay_minimap,
    project_tracks_to_field,
    render_overlay_video,
)

# Expected BGR colors.
BLUE = (255, 128, 0)
ORANGE = (0, 128, 255)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (170, 170, 170)


def _dot_color_at(renderer, img, field_xy):
    px, py = renderer.field_to_px(*field_xy)
    return tuple(int(v) for v in img[py, px])


# ---------------------------------------------------------------------------
# team/role mapping (the required color table)
# ---------------------------------------------------------------------------
def test_team_role_color_mapping():
    assert color_for_object("player", 0, 5) == BLUE
    assert color_for_object("player", 1, 5) == ORANGE
    assert color_for_object("goalkeeper", None, 5) == RED
    assert color_for_object("referee", None, 5) == YELLOW
    assert color_for_object("ball", None, 5) == WHITE
    assert color_for_object("unknown", None, 5) == GRAY


def test_player_team0_renders_blue():
    r = MinimapRenderer(px_per_meter=8)
    obj = {"field": (30.0, 34.0), "role": "player",
           "color": color_for_object("player", 0, 1)}
    assert _dot_color_at(r, r.render([obj]), (30.0, 34.0)) == BLUE


def test_player_team1_renders_orange():
    r = MinimapRenderer(px_per_meter=8)
    obj = {"field": (75.0, 34.0), "role": "player",
           "color": color_for_object("player", 1, 2)}
    assert _dot_color_at(r, r.render([obj]), (75.0, 34.0)) == ORANGE


def test_goalkeeper_renders_red():
    r = MinimapRenderer(px_per_meter=8)
    obj = {"field": (5.0, 34.0), "role": "goalkeeper",
           "color": color_for_object("goalkeeper", None, 3)}
    assert _dot_color_at(r, r.render([obj]), (5.0, 34.0)) == RED


def test_referee_renders_yellow():
    r = MinimapRenderer(px_per_meter=8)
    obj = {"field": (52.5, 34.0), "role": "referee",
           "color": color_for_object("referee", None, 4)}
    assert _dot_color_at(r, r.render([obj]), (52.5, 34.0)) == YELLOW


def test_ball_renders_white():
    r = MinimapRenderer(px_per_meter=8)
    obj = {"field": (52.5, 40.0), "role": "ball",
           "color": color_for_object("ball", None, 9000)}
    assert _dot_color_at(r, r.render([obj]), (52.5, 40.0)) == WHITE


# ---------------------------------------------------------------------------
# track_id debug mode
# ---------------------------------------------------------------------------
def test_track_id_mode_gives_distinct_colors():
    # Same team, different track ids -> distinct colors in track_id mode...
    c1 = color_for_object("player", 0, 1, COLOR_MODE_TRACK_ID)
    c2 = color_for_object("player", 0, 2, COLOR_MODE_TRACK_ID)
    c3 = color_for_object("player", 0, 3, COLOR_MODE_TRACK_ID)
    assert len({c1, c2, c3}) == 3
    # ...whereas team_role mode collapses them to the same team color.
    same = {color_for_object("player", 0, t, COLOR_MODE_TEAM_ROLE)
            for t in (1, 2, 3)}
    assert same == {BLUE}


def test_track_id_color_is_stable():
    assert color_for_object("player", 0, 7, COLOR_MODE_TRACK_ID) == \
        color_for_object("referee", 1, 7, COLOR_MODE_TRACK_ID)  # id drives color


# ---------------------------------------------------------------------------
# project_tracks_to_field honors color_mode
# ---------------------------------------------------------------------------
class _Track:
    def __init__(self, tid, name, bbox):
        self.track_id, self.class_name, self.bbox = tid, name, bbox


class _Frame:
    def __init__(self, idx, tracks):
        self.frame_index, self.tracks = idx, tracks


def _identity_homography():
    field = [(0, 0), (105, 0), (105, 68), (0, 68)]
    image = [(fx * 10 + 100, fy * 10 + 50) for fx, fy in field]
    return Homography.from_correspondences(image, field)


def test_smoothing_clamps_teleport_jumps():
    # A homography that's fine, but the track's foot point "teleports" 30 m
    # for one frame (simulating a homography spike). Smoothing must keep the
    # per-frame field step physically plausible (no flicker / fake switch).
    H = _identity_homography()

    def foot_to_bbox(fx, fy):
        ix, iy = fx * 10 + 100, fy * 10 + 50           # inverse of the affine
        return (ix - 5, iy - 40, ix + 5, iy)

    frames = [
        _Frame(1, [_Track(1, "player", foot_to_bbox(30.0, 34.0))]),
        _Frame(2, [_Track(1, "player", foot_to_bbox(31.0, 34.0))]),
        _Frame(3, [_Track(1, "player", foot_to_bbox(61.0, 34.0))]),   # +30 m spike
        _Frame(4, [_Track(1, "player", foot_to_bbox(32.0, 34.0))]),
    ]
    roles = {1: ("player", 0)}
    pos = project_tracks_to_field(frames, roles, H, None,
                                  smooth_alpha=0.5, max_step_m=3.0)
    xs = [p["objects"][0]["field"][0] for p in pos]
    steps = [abs(xs[i] - xs[i - 1]) for i in range(1, len(xs))]
    assert max(steps) <= 3.0 + 1e-6                    # teleport clamped


def test_smoothing_off_preserves_raw_projection():
    H = _identity_homography()
    foot = (52.5 * 10 + 100, 34 * 10 + 50)
    bbox = (foot[0] - 5, foot[1] - 40, foot[0] + 5, foot[1])
    frames = [_Frame(1, [_Track(1, "player", bbox)])]
    pos = project_tracks_to_field(frames, {1: ("player", 0)}, H, None,
                                  smooth_alpha=1.0, max_step_m=999)
    fx, fy = pos[0]["objects"][0]["field"]
    assert fx == pytest.approx(52.5, abs=1e-2) and fy == pytest.approx(34.0, abs=1e-2)


def _foot_to_bbox(fx, fy):
    ix, iy = fx * 10 + 100, fy * 10 + 50               # inverse of the affine
    return (ix - 5, iy - 40, ix + 5, iy)


def test_align_clusters_to_roles_keeps_convention():
    from calibration.minimap import _align_clusters_to_roles
    # Cluster 0 is mostly roles-team 1; cluster 1 is mostly roles-team 0 ->
    # the mapping must flip them so colours match the existing convention.
    raw = {(1, 10): 0, (1, 11): 0, (1, 20): 1, (1, 21): 1}
    roles = {10: ("player", 1), 11: ("player", 1),
             20: ("player", 0), 21: ("player", 0)}
    out = _align_clusters_to_roles(raw, roles)
    assert out[(1, 10)] == 1 and out[(1, 20)] == 0


def test_smooth_team_timeline_majority_window():
    from calibration.minimap import _smooth_team_timeline
    # Track 5 reads team 0 except one stray frame -> smoothing fixes the stray.
    raw = {(f, 5): (1 if f == 3 else 0) for f in range(1, 8)}
    out = _smooth_team_timeline(raw, window=3)
    assert out[(3, 5)] == 0                         # the stray flip is smoothed out
    assert all(out[(f, 5)] == 0 for f in range(1, 8))


def test_per_frame_team_override_colours_by_shirt():
    # Track 1's per-track label is team 0, but the override says team 1 in this
    # frame (the shirt) -> the projected dot is coloured team 1.
    H = _identity_homography()
    frames = [_Frame(1, [_Track(1, "player", _foot_to_bbox(52.5, 34.0))])]
    pos = project_tracks_to_field(
        frames, {1: ("player", 0)}, H, None, team_override={(1, 1): 1})
    assert pos[0]["objects"][0]["team_id"] == 1


def test_blowup_projection_on_first_frame_is_dropped():
    # A foot point that projects hundreds of metres off the pitch (a horizon
    # blow-up) on a track's first frame has no history to fall back to, so it
    # must be dropped rather than drawn at a garbage position.
    H = _identity_homography()
    frames = [_Frame(1, [_Track(1, "player", _foot_to_bbox(300.0, 34.0))])]
    pos = project_tracks_to_field(frames, {1: ("player", 0)}, H, None)
    assert pos[0]["objects"] == []


def test_near_edge_player_is_clamped_onto_pitch():
    # A player projecting just outside the touchline (within the sane margin)
    # is pulled onto the line so they still appear on the minimap.
    H = _identity_homography()
    frames = [_Frame(1, [_Track(1, "player", _foot_to_bbox(-5.0, 34.0))])]
    pos = project_tracks_to_field(frames, {1: ("player", 0)}, H, None)
    obj = pos[0]["objects"][0]
    assert obj["field"][0] == pytest.approx(0.0)       # clamped to x=0
    assert obj["field"][1] == pytest.approx(34.0)


def test_blowup_after_good_frame_holds_last_position():
    # A one-frame blow-up after a valid position holds the last good spot
    # instead of teleporting off the pitch.
    H = _identity_homography()
    frames = [
        _Frame(1, [_Track(1, "player", _foot_to_bbox(50.0, 34.0))]),
        _Frame(2, [_Track(1, "player", _foot_to_bbox(900.0, 34.0))]),   # blow-up
    ]
    pos = project_tracks_to_field(frames, {1: ("player", 0)}, H, None)
    assert pos[1]["objects"][0]["field"][0] == pytest.approx(50.0, abs=1.0)


def test_project_uses_team_role_colors_by_default():
    H = _identity_homography()
    # foot point (bottom-center) maps to field (52.5, 34).
    fx, fy = 52.5, 34.0
    foot = (fx * 10 + 100, fy * 10 + 50)
    bbox = (foot[0] - 5, foot[1] - 40, foot[0] + 5, foot[1])
    frames = [_Frame(1, [_Track(1, "player", bbox)])]
    roles = {1: ("player", 1)}                       # team 1 -> orange

    pos = project_tracks_to_field(frames, roles, H)   # default team_role
    assert tuple(pos[0]["objects"][0]["color"]) == ORANGE

    pos_dbg = project_tracks_to_field(frames, roles, H,
                                      color_mode=COLOR_MODE_TRACK_ID)
    assert tuple(pos_dbg[0]["objects"][0]["color"]) != ORANGE  # per-track


# ---------------------------------------------------------------------------
# overlay uses the SAME color mapping as the standalone minimap
# ---------------------------------------------------------------------------
def test_overlay_uses_same_colors_as_standalone():
    r = MinimapRenderer(px_per_meter=8)
    objs = [
        {"field": (30.0, 34.0), "role": "player",
         "color": color_for_object("player", 0, 1)},
        {"field": (75.0, 34.0), "role": "player",
         "color": color_for_object("player", 1, 2)},
    ]
    standalone = r.render(objs)
    # Standalone has both team colors.
    assert _dot_color_at(r, standalone, (30.0, 34.0)) == BLUE
    assert _dot_color_at(r, standalone, (75.0, 34.0)) == ORANGE

    # The overlay composites that very same rendered minimap, so the colors
    # present in the minimap are identical (blue + orange both appear).
    base = np.full((600, 1000, 3), 60, np.uint8)
    out = overlay_minimap(base, standalone, position="top_right",
                          width_ratio=0.5, background_alpha=0.0, border=False)
    assert np.any(np.all(out == BLUE, axis=2))
    assert np.any(np.all(out == ORANGE, axis=2))


def test_render_overlay_video_colors_match_standalone(tmp_path):
    from utils.video_io import VideoReader, VideoWriter

    # A 2-frame base video.
    base = tmp_path / "base.mp4"
    w = VideoWriter(base, fps=10.0, frame_size=(400, 300))
    for _ in range(2):
        w.write(np.full((300, 400, 3), 60, np.uint8))
    w.release()

    r = MinimapRenderer(px_per_meter=4)
    objs = [{"field": (52.5, 34.0), "role": "goalkeeper",
             "color": color_for_object("goalkeeper", None, 1)}]
    positions = [{"frame": 1, "objects": objs}, {"frame": 2, "objects": objs}]

    out = tmp_path / "overlay.mp4"
    render_overlay_video(base, positions, r, out, width_ratio=0.6,
                         background_alpha=0.0, border=False, fps=10.0)
    with VideoReader(out) as reader:
        _, frame = next(reader.frames())
    # The goalkeeper red dot must survive into the overlaid frame.
    assert np.any(np.all(frame == RED, axis=2))
