"""Tests for the player pose-skeleton overlay (pure parts)."""

from __future__ import annotations

import numpy as np

from tracking.models import Track
from visualization.pose_overlay import (
    COCO_KEYPOINTS,
    COCO_SKELETON,
    assign_poses_to_tracks,
    draw_pose,
    iou,
)


def _track(tid, bbox):
    return Track(1, tid, 2, "player", 0.9, bbox)


# ---------------------------------------------------------------------------
# Geometry / skeleton sanity
# ---------------------------------------------------------------------------
def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == 1.0 / 3.0   # half overlap


def test_skeleton_indices_valid():
    assert len(COCO_KEYPOINTS) == 17
    for a, b in COCO_SKELETON:
        assert 0 <= a < 17 and 0 <= b < 17


# ---------------------------------------------------------------------------
# Pose -> track assignment
# ---------------------------------------------------------------------------
def test_assign_poses_matches_by_iou():
    kp_a = np.zeros((17, 3), np.float32)
    kp_b = np.ones((17, 3), np.float32)
    poses = [
        ((100, 100, 140, 220), kp_a),    # overlaps track 1
        ((500, 100, 540, 220), kp_b),    # overlaps track 2
    ]
    tracks = [_track(1, (102, 100, 142, 220)), _track(2, (498, 100, 538, 220))]
    assigned = assign_poses_to_tracks(poses, tracks, iou_threshold=0.3)
    assert set(assigned) == {1, 2}
    assert np.array_equal(assigned[1], kp_a)
    assert np.array_equal(assigned[2], kp_b)


def test_assign_poses_ignores_low_iou():
    poses = [((0, 0, 40, 100), np.zeros((17, 3), np.float32))]
    tracks = [_track(1, (900, 500, 940, 600))]      # nowhere near the pose
    assert assign_poses_to_tracks(poses, tracks, iou_threshold=0.3) == {}


def test_assign_poses_is_one_to_one():
    kp = np.zeros((17, 3), np.float32)
    # Two poses overlap the same single track -> only the best one wins.
    poses = [((100, 100, 140, 220), kp), ((105, 100, 145, 220), kp)]
    tracks = [_track(1, (100, 100, 140, 220))]
    assigned = assign_poses_to_tracks(poses, tracks)
    assert list(assigned) == [1]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def test_draw_pose_renders_confident_joints_only():
    canvas = np.zeros((300, 300, 3), np.uint8)
    kpts = np.zeros((17, 3), np.float32)
    # Two confident, connected joints (left shoulder #5, left elbow #7).
    kpts[5] = (100, 100, 0.9)
    kpts[7] = (120, 160, 0.9)
    # A low-confidence joint that must NOT be drawn.
    kpts[9] = (140, 220, 0.1)
    draw_pose(canvas, kpts, color=(255, 128, 0), min_conf=0.5)
    assert np.any(canvas != 0)                      # something was drawn
    # The low-confidence wrist area stays black.
    assert np.all(canvas[215:225, 135:145] == 0)


def test_draw_pose_blank_when_all_low_conf():
    canvas = np.zeros((100, 100, 3), np.uint8)
    kpts = np.zeros((17, 3), np.float32)             # all conf 0
    draw_pose(canvas, kpts, color=(0, 255, 0), min_conf=0.5)
    assert np.all(canvas == 0)
