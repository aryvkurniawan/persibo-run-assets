"""Tests for automatic pitch calibration from the keypoint model."""

from __future__ import annotations

import numpy as np
import pytest

from calibration.homography import Homography
from calibration.pitch_keypoints import (
    NUM_KEYPOINTS,
    PITCH_KEYPOINT_FIELD,
    KeypointHomographyProvider,
    homography_from_keypoints,
)


# ---------------------------------------------------------------------------
# Field template (28 keypoints -> 105x68 meters)
# ---------------------------------------------------------------------------
def test_template_has_28_keypoints():
    assert NUM_KEYPOINTS == 28
    assert set(PITCH_KEYPOINT_FIELD) == set(range(28))


def test_template_known_landmarks():
    F = PITCH_KEYPOINT_FIELD
    assert F[0] == (0.0, 0.0)              # left top corner
    assert F[5] == (0.0, 68.0)            # left bottom corner
    assert F[16] == (105.0, 0.0)          # right top corner
    assert F[11] == (105.0, 68.0)         # right bottom corner
    assert F[17] == (52.5, 0.0)           # halfway - top touchline
    assert F[10] == (52.5, 68.0)          # halfway - bottom touchline
    # Penalty / goal area corners on the goal line.
    assert F[1] == pytest.approx((0.0, 13.84))
    assert F[2] == pytest.approx((0.0, 24.84))
    assert F[8] == pytest.approx((16.5, 13.84))    # left penalty inner top
    assert F[21] == pytest.approx((88.5, 13.84))   # right penalty inner top
    # Center-circle / halfway intersections.
    assert F[18] == pytest.approx((52.5, 24.85))
    assert F[19] == pytest.approx((52.5, 43.15))


def test_template_penalty_arc_intersections():
    F = PITCH_KEYPOINT_FIELD
    # Arc (r=9.15) centered on penalty spot (11,34) meets the 16.5m line.
    assert F[24][0] == pytest.approx(16.5)
    assert F[25][0] == pytest.approx(16.5)
    assert F[24][1] == pytest.approx(41.31, abs=0.05)   # bottom
    assert F[25][1] == pytest.approx(26.69, abs=0.05)   # top


# ---------------------------------------------------------------------------
# Homography from detected keypoints
# ---------------------------------------------------------------------------
def _project(matrix, pt):
    v = matrix @ np.array([pt[0], pt[1], 1.0])
    return v[0] / v[2], v[1] / v[2]


def _synthetic_keypoints(indices, field_to_image):
    """Build detected keypoints (idx, x, y, conf) from a field->image map."""
    out = []
    for i in indices:
        fx, fy = PITCH_KEYPOINT_FIELD[i]
        ix, iy = field_to_image((fx, fy))
        out.append((i, ix, iy, 0.9))
    return out


def test_homography_from_keypoints_recovers_transform():
    # A known field->image transform (affine = valid homography).
    M = np.array([[9.0, 0.0, 80.0], [0.0, 9.0, 40.0], [0.0, 0.0, 1.0]])
    kps = _synthetic_keypoints([0, 16, 11, 5, 17, 10, 8, 21],
                               lambda p: _project(M, p))
    H = homography_from_keypoints(kps, min_points=5)
    assert H is not None
    # field->image round trip via the recovered homography.
    ix, iy = H.project_field_to_image((52.5, 34.0))
    exp = _project(M, (52.5, 34.0))
    assert ix == pytest.approx(exp[0], abs=1e-2)
    assert iy == pytest.approx(exp[1], abs=1e-2)
    assert H.reprojection_error() < 1e-3


def test_homography_from_keypoints_needs_enough_points():
    M = np.array([[9.0, 0.0, 80.0], [0.0, 9.0, 40.0], [0.0, 0.0, 1.0]])
    kps = _synthetic_keypoints([0, 16, 11], lambda p: _project(M, p))
    assert homography_from_keypoints(kps, min_points=5) is None


def test_homography_from_keypoints_ignores_low_confidence():
    M = np.array([[9.0, 0.0, 80.0], [0.0, 9.0, 40.0], [0.0, 0.0, 1.0]])
    kps = _synthetic_keypoints([0, 16, 11, 5, 17, 10], lambda p: _project(M, p))
    # Drop everything below the confidence gate -> not enough points.
    low = [(i, x, y, 0.2) for (i, x, y, _c) in kps]
    assert homography_from_keypoints(low, min_conf=0.5, min_points=5) is None


def test_homography_from_keypoints_rejects_high_error():
    M = np.array([[9.0, 0.0, 80.0], [0.0, 9.0, 40.0], [0.0, 0.0, 1.0]])
    kps = _synthetic_keypoints([0, 16, 11, 5, 17, 10, 8, 21],
                               lambda p: _project(M, p))
    # Corrupt one point badly so the fit can't be sub-4m.
    kps[0] = (kps[0][0], kps[0][1] + 600, kps[0][2] + 600, 0.9)
    assert homography_from_keypoints(kps, min_points=5, max_error_m=4.0) is None


# ---------------------------------------------------------------------------
# Provider: nearest-frame selection
# ---------------------------------------------------------------------------
def _identity_homography():
    M = np.array([[9.0, 0.0, 80.0], [0.0, 9.0, 40.0], [0.0, 0.0, 1.0]])
    kps = _synthetic_keypoints([0, 16, 11, 5, 17, 10, 8, 21],
                               lambda p: _project(M, p))
    return homography_from_keypoints(kps, min_points=5)


def test_provider_nearest_frame_and_stats():
    h1, h2 = _identity_homography(), _identity_homography()
    provider = KeypointHomographyProvider([(100, h1), (500, h2)])
    assert provider.frame_numbers == [100, 500]
    assert provider.for_frame(120) is h1
    assert provider.for_frame(480) is h2
    assert provider.reprojection_error() < 1e-3
    assert len(provider) == 2


def test_provider_requires_entries():
    with pytest.raises(ValueError):
        KeypointHomographyProvider([])


def test_provider_works_as_projection_selector():
    # Duck-typed: a provider has for_frame, so projection uses it per frame.
    from calibration.minimap import MinimapRenderer, project_tracks_to_field

    class _Track:
        def __init__(s, tid, name, bbox):
            s.track_id, s.class_name, s.bbox = tid, name, bbox

    class _Frame:
        def __init__(s, idx, tracks):
            s.frame_index, s.tracks = idx, tracks

    H = _identity_homography()
    provider = KeypointHomographyProvider([(1, H)])
    # foot point that maps (52.5,34) -> image via M=9x+80, 9y+40
    foot = (52.5 * 9 + 80, 34 * 9 + 40)
    bbox = (foot[0] - 5, foot[1] - 40, foot[0] + 5, foot[1])
    frames = [_Frame(1, [_Track(1, "player", bbox)])]
    pos = project_tracks_to_field(frames, {1: ("player", 0)}, provider,
                                  MinimapRenderer())
    fx, fy = pos[0]["objects"][0]["field"]
    assert fx == pytest.approx(52.5, abs=1e-1)
    assert fy == pytest.approx(34.0, abs=1e-1)
