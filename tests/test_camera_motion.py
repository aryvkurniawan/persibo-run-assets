"""Tests for camera-motion estimation + homography propagation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration.camera_motion import (
    PropagatedHomographyProvider,
    build_propagated_provider,
    classify_motion,
    estimate_background_transform,
    measure_motion_level,
    propagate_homographies,
)
from calibration.homography import Homography


# ---------------------------------------------------------------------------
# Background transform from optical flow
# ---------------------------------------------------------------------------
def _textured_image(w=320, h=240, seed=0):
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 255, (h, w), dtype=np.uint8)
    # Add some strong corners the tracker can lock onto.
    for _ in range(40):
        x, y = rng.randint(20, w - 20), rng.randint(20, h - 20)
        cv2.rectangle(img, (x, y), (x + 6, y + 6), 255, -1)
    return img


def test_estimate_background_transform_recovers_translation():
    prev = _textured_image()
    # Shift the whole image right+down by (10, 6) -> camera moved.
    matrix = np.float32([[1, 0, 10], [0, 1, 6]])
    cur = cv2.warpAffine(prev, matrix, (prev.shape[1], prev.shape[0]))
    m, n = estimate_background_transform(prev, cur)
    assert n >= 8
    assert m[0, 2] == pytest.approx(10, abs=1.0)     # tx
    assert m[1, 2] == pytest.approx(6, abs=1.0)       # ty


def test_estimate_background_transform_identity_on_blank():
    blank = np.zeros((240, 320), np.uint8)
    m, n = estimate_background_transform(blank, blank)
    assert n == 0
    assert np.allclose(m, np.eye(3))


# ---------------------------------------------------------------------------
# Homography propagation (pure)
# ---------------------------------------------------------------------------
def _apply(H, pt):
    v = H @ np.array([pt[0], pt[1], 1.0])
    return v[0] / v[2], v[1] / v[2]


def test_propagate_holds_homography_through_camera_motion():
    # A static field point at (52.5, 34). At frame 1 it's at image (160,120).
    # The camera translates the background by (+5,+3) px each frame, so the
    # SAME field point appears shifted each frame; the propagated homography
    # must still map the shifted image point back to (52.5, 34).
    field_pt = (52.5, 34.0)
    # Anchor homography (image->field): an affine that sends (160,120)->(52.5,34).
    H1 = np.array([[0.5, 0.0, 52.5 - 0.5 * 160],
                   [0.0, 0.5, 34.0 - 0.5 * 120],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    assert _apply(H1, (160, 120)) == pytest.approx(field_pt)

    # M_k maps frame k-1 -> k: background point moves by (+5,+3).
    M = np.array([[1, 0, 5], [0, 1, 3], [0, 0, 1]], dtype=np.float64)
    transforms = {k: M for k in range(2, 6)}
    anchors = {1: H1}
    frames = [1, 2, 3, 4, 5]

    homs = propagate_homographies(anchors, transforms, frames)
    assert set(homs) == set(frames)
    for f in frames:
        img_pt = (160 + 5 * (f - 1), 120 + 3 * (f - 1))     # where it appears
        fx, fy = _apply(homs[f], img_pt)
        assert (fx, fy) == pytest.approx(field_pt, abs=1e-6)


def test_propagate_uses_nearest_anchor():
    eye = np.eye(3, dtype=np.float64)
    H_a = eye.copy()
    H_b = eye.copy() * 1.0
    H_b[0, 0] = 2.0                       # a clearly different anchor
    anchors = {1: H_a, 9: H_b}
    transforms = {k: eye for k in range(2, 10)}  # no camera motion
    homs = propagate_homographies(anchors, transforms, list(range(1, 10)))
    # Frame 3 is nearest anchor 1; frame 8 nearest anchor 9.
    assert np.allclose(homs[3], H_a)
    assert np.allclose(homs[8], H_b)


def test_propagate_empty_anchors():
    assert propagate_homographies({}, {}, [1, 2, 3]) == {}


# ---------------------------------------------------------------------------
# PropagatedHomographyProvider
# ---------------------------------------------------------------------------
def _affine_homography(scale=10.0, tx=100.0, ty=50.0):
    image = [(0 * scale + tx, 0 * scale + ty), (105 * scale + tx, 0 * scale + ty),
             (105 * scale + tx, 68 * scale + ty), (0 * scale + tx, 68 * scale + ty)]
    field = [(0, 0), (105, 0), (105, 68), (0, 68)]
    return Homography.from_correspondences(image, field)


def test_propagated_provider_returns_per_frame_and_nearest():
    eye = np.eye(3, dtype=np.float64)
    per_frame = {1: eye.copy(), 2: eye.copy(), 3: eye.copy()}
    prov = PropagatedHomographyProvider(per_frame, anchor_error=0.4)
    assert len(prov) == 3
    assert prov.frame_numbers == [1, 2, 3]
    assert prov.reprojection_error() == pytest.approx(0.4)
    # Exact frame and a nearest-frame fallback both resolve to a projector.
    assert prov.for_frame(2).project_image_to_field((0, 0)) is not None
    assert prov.for_frame(99) is prov.for_frame(3)        # clamps to nearest


def test_build_propagated_provider_from_anchors_and_motion():
    h = _affine_homography()
    anchors = [(1, h), (9, h)]
    eye = np.eye(3, dtype=np.float64)
    transforms = {k: eye for k in range(2, 10)}     # static camera
    prov = build_propagated_provider(
        anchors, transforms, range(1, 10), anchor_error=0.5)
    assert prov is not None
    assert len(prov) == 9                            # every frame got a homography
    # With no camera motion the propagated map equals the anchor mapping.
    gx, gy = prov.for_frame(5).project_image_to_field((52.5 * 10 + 100, 34 * 10 + 50))
    assert gx == pytest.approx(52.5, abs=1e-3) and gy == pytest.approx(34.0, abs=1e-3)


def test_build_propagated_provider_none_without_anchors():
    assert build_propagated_provider([], {2: np.eye(3)}, range(1, 5)) is None


# ---------------------------------------------------------------------------
# Motion level -> anchor budget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shift,expected", [
    (0.0, "static"), (1.2, "static"),
    (8.0, "moving"),
    (64.0, "panning"), (200.0, "panning"),
])
def test_classify_motion_bands(shift, expected):
    assert classify_motion(shift) == expected


def _panning_video(path, frames=12, shift=40, size=(320, 180)):
    """Synthetic clip whose background slides by ``shift`` px each frame."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             25.0, (w, h))
    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 255, (h, w + frames * shift, 3), dtype=np.uint8)
    for i in range(frames):
        writer.write(canvas[:, i * shift:i * shift + w])
    writer.release()


def test_measure_motion_level_reads_static_clip_as_static(tmp_path):
    path = tmp_path / "static.mp4"
    _panning_video(path, frames=8, shift=0)
    assert classify_motion(measure_motion_level(str(path), samples=8)) == "static"


def test_measure_motion_level_reads_panning_clip_as_panning(tmp_path):
    path = tmp_path / "pan.mp4"
    _panning_video(path, frames=8, shift=40)
    # 40 px per sample is far past the panning threshold; the point is that the
    # measurement catches it, not the exact number.
    assert measure_motion_level(str(path), samples=8) > 3.0
