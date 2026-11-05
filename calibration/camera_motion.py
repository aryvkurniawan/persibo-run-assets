"""Camera-motion estimation + homography propagation (Phase 4 helper).

A broadcast camera pans/tilts/zooms, so the image-to-image map of the
(static) background between two consecutive frames is a transform we can
estimate. With it we can *propagate* a good homography from frames where
the pitch keypoints are confident ("anchors") across the frames where they
aren't — giving a SMOOTH, gap-free homography instead of a per-frame jitter.

This module is the model-independent foundation:
    * :func:`estimate_background_transform` — frame-to-frame motion from
      optical flow, with moving objects (players) masked out.
    * :class:`CameraMotionEstimator` — runs it over a video.
    * :func:`propagate_homographies` — pure: anchors + per-frame transforms
      -> a homography for every frame (nearest-anchor propagation).

Wiring it into the calibration provider is deferred until the re-trained
keypoint model lands (better anchors); the pieces here are ready and tested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger("calibration.camera_motion")

Box = Tuple[float, float, float, float]
_IDENTITY = np.eye(3, dtype=np.float64)


def _affine_to_h(matrix: np.ndarray) -> np.ndarray:
    """Embed a 2x3 affine into a 3x3 homography."""
    h = np.eye(3, dtype=np.float64)
    h[:2, :] = matrix
    return h


def _background_mask(
    shape: Tuple[int, int], exclude_boxes: Optional[Sequence[Box]], pad: int = 12
) -> np.ndarray:
    """White everywhere except (padded) moving-object boxes."""
    h, w = shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    for x1, y1, x2, y2 in exclude_boxes or []:
        ax1 = max(int(x1) - pad, 0)
        ay1 = max(int(y1) - pad, 0)
        ax2 = min(int(x2) + pad, w)
        ay2 = min(int(y2) + pad, h)
        if ax2 > ax1 and ay2 > ay1:
            mask[ay1:ay2, ax1:ax2] = 0
    return mask


def estimate_background_transform(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    exclude_boxes: Optional[Sequence[Box]] = None,
    max_corners: int = 600,
    quality: float = 0.01,
    min_distance: int = 8,
) -> Tuple[np.ndarray, int]:
    """Estimate the background motion ``prev -> cur`` as a 3x3 transform.

    Tracks Shi-Tomasi corners (outside the masked moving objects) with
    Lucas-Kanade optical flow and fits a RANSAC similarity (translation +
    rotation + uniform scale). Returns ``(matrix_3x3, n_inliers)``;
    identity with 0 inliers if it can't be estimated.
    """
    mask = _background_mask(prev_gray.shape[:2], exclude_boxes)
    p0 = cv2.goodFeaturesToTrack(
        prev_gray, max_corners, quality, min_distance, mask=mask)
    if p0 is None or len(p0) < 8:
        return _IDENTITY.copy(), 0
    p1, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None)
    if p1 is None:
        return _IDENTITY.copy(), 0
    ok = status.reshape(-1) == 1
    g0, g1 = p0[ok], p1[ok]
    if len(g0) < 8:
        return _IDENTITY.copy(), 0
    matrix, inliers = cv2.estimateAffinePartial2D(
        g0, g1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return _IDENTITY.copy(), 0
    n = int(inliers.sum()) if inliers is not None else 0
    return _affine_to_h(matrix), n


class CameraMotionEstimator:
    """Estimates per-frame background (camera) transforms over a video."""

    def __init__(
        self,
        max_corners: int = 600,
        quality: float = 0.01,
        min_distance: int = 8,
    ) -> None:
        self.max_corners = max_corners
        self.quality = quality
        self.min_distance = min_distance

    def estimate_video(
        self,
        video_path: str | Path,
        tracks_by_frame: Optional[Dict[int, Sequence]] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[int, np.ndarray]:
        """Return ``{frame_index: M}`` where ``M`` maps frame-1 -> frame.

        ``tracks_by_frame`` (optional) provides per-frame tracks whose
        bboxes are masked out so only the static background drives the
        estimate. Frame index 1 has no predecessor (skipped).
        """
        from utils.video_io import VideoReader

        transforms: Dict[int, np.ndarray] = {}
        prev_gray: Optional[np.ndarray] = None
        with VideoReader(video_path) as reader:
            for index, frame in reader.frames():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    boxes = None
                    if tracks_by_frame is not None:
                        boxes = [t.bbox for t in tracks_by_frame.get(index - 1, [])]
                    matrix, _n = estimate_background_transform(
                        prev_gray, gray, boxes,
                        self.max_corners, self.quality, self.min_distance)
                    transforms[index] = matrix
                prev_gray = gray
                if max_frames is not None and index >= max_frames:
                    break
        logger.info("Estimated camera motion for %d frame pairs", len(transforms))
        return transforms


def propagate_homographies(
    anchors: Dict[int, np.ndarray],
    transforms: Dict[int, np.ndarray],
    frame_indices: Sequence[int],
) -> Dict[int, np.ndarray]:
    """Fill a homography for every frame from sparse anchors + camera motion.

    ``anchors`` maps an anchor frame to its image->field homography (3x3).
    ``transforms`` maps frame ``k`` to ``M_k`` (background motion
    ``k-1 -> k``). Each frame gets the homography propagated from the
    nearest anchor:

        forward step :  H_k   = H_{k-1} @ inv(M_k)
        backward step:  H_{k-1} = H_k   @ M_k

    Returns ``{frame: H_image_to_field}`` for every frame that could be
    reached from some anchor.
    """
    if not anchors:
        return {}
    frames = sorted(frame_indices)
    forward: Dict[int, Tuple[np.ndarray, int]] = {}   # frame -> (H, dist_to_anchor)
    backward: Dict[int, Tuple[np.ndarray, int]] = {}

    # Forward sweep: carry the last anchor's H forward via inv(M).
    cur: Optional[np.ndarray] = None
    anchor_at = 0
    for f in frames:
        if f in anchors:
            cur, anchor_at = anchors[f].astype(np.float64), f
        elif cur is not None and f in transforms:
            cur = cur @ np.linalg.inv(transforms[f])
        else:
            cur = None
        if cur is not None:
            forward[f] = (cur, f - anchor_at)

    # Backward sweep: carry the next anchor's H backward via M.
    cur, anchor_at = None, 0
    for f in reversed(frames):
        if f in anchors:
            cur, anchor_at = anchors[f].astype(np.float64), f
        elif cur is not None and (f + 1) in transforms:
            cur = cur @ transforms[f + 1]
        else:
            cur = None
        if cur is not None:
            backward[f] = (cur, anchor_at - f)

    result: Dict[int, np.ndarray] = {}
    for f in frames:
        fwd, bwd = forward.get(f), backward.get(f)
        if fwd and bwd:
            result[f] = fwd[0] if fwd[1] <= bwd[1] else bwd[0]
        elif fwd:
            result[f] = fwd[0]
        elif bwd:
            result[f] = bwd[0]
    return result


class PropagatedHomographyProvider:
    """Per-frame homographies that follow the camera via motion propagation.

    Drop-in for the other homography providers (exposes ``for_frame``). Built
    from sparse keypoint *anchors* plus frame-to-frame background transforms,
    so every frame gets a homography smoothly carried from the nearest anchor
    instead of snapping to a keyframe — this removes the per-frame jitter that
    teleports minimap dots. ``reprojection_error`` reports the anchors' error
    (the propagated frames carry no calibration points of their own).
    """

    def __init__(
        self,
        per_frame_matrices: Dict[int, np.ndarray],
        anchor_error: float = float("inf"),
    ) -> None:
        from calibration.homography import Homography

        if not per_frame_matrices:
            raise ValueError("no per-frame homographies were propagated")
        self._frames = sorted(per_frame_matrices)
        self._homos = {
            f: Homography(np.asarray(m, dtype=np.float64))
            for f, m in per_frame_matrices.items()
        }
        self._anchor_error = float(anchor_error)

    def for_frame(self, frame_index: int):
        idx = int(frame_index)
        homography = self._homos.get(idx)
        if homography is not None:
            return homography
        nearest = min(self._frames, key=lambda f: abs(f - idx))
        return self._homos[nearest]

    @property
    def frame_numbers(self) -> List[int]:
        return list(self._frames)

    def reprojection_error(self) -> float:
        return self._anchor_error

    def __len__(self) -> int:
        return len(self._homos)


def build_propagated_provider(
    anchor_entries: Sequence[Tuple[int, object]],
    transforms: Dict[int, np.ndarray],
    frame_indices: Sequence[int],
    anchor_error: float = float("inf"),
) -> Optional["PropagatedHomographyProvider"]:
    """Anchors (``[(frame, Homography), ...]``) + motion -> a smooth provider.

    Returns ``None`` if nothing could be propagated (e.g. no anchors).
    """
    anchors = {int(f): np.asarray(h.matrix, dtype=np.float64)
               for f, h in anchor_entries}
    per_frame = propagate_homographies(anchors, transforms, frame_indices)
    if not per_frame:
        return None
    return PropagatedHomographyProvider(per_frame, anchor_error)


# ---------------------------------------------------------------------------
# Camera-motion level: decides how many calibration anchors a clip needs.
# ---------------------------------------------------------------------------

# A broadcast camera that is bolted in place moves <1 px between samples. A
# panning one swings tens of pixels as it follows play. Measured on a 137 min
# match, 85% of samples were static and the remaining 15% were shot changes,
# so the *median* hides all of it -- the high percentile is what matters.
_STATIC_P90_PX = 3.0
_PANNING_P90_PX = 20.0


def measure_motion_level(
    video_path: str | Path,
    samples: int = 60,
    max_frames: Optional[int] = None,
) -> float:
    """90th-percentile background shift in pixels between sampled frames.

    Seeks to each sample instead of decoding the whole file, so a full match
    costs a couple of seconds rather than minutes. Returns 0.0 when the video
    cannot be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 1:
            return 0.0
        limit = min(total - 1, max_frames) if max_frames is not None else total - 1
        step = max(1, limit // max(samples, 1))
        shifts: List[float] = []
        prev: Optional[np.ndarray] = None
        for pos in range(0, limit, step):
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, pos):
                continue
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))
            if prev is not None:
                (dx, dy), _response = cv2.phaseCorrelate(
                    prev.astype(np.float32), gray.astype(np.float32))
                shifts.append(float(np.hypot(dx, dy)))
            prev = gray
        if not shifts:
            return 0.0
        level = float(np.percentile(shifts, 90))
        logger.info(
            "Camera motion: p90 shift %.1f px over %d samples", level, len(shifts))
        return level
    finally:
        cap.release()


def classify_motion(shift_px: float) -> str:
    """``"static"``, ``"moving"``, or ``"panning"`` for a measured p90 shift."""
    if shift_px <= _STATIC_P90_PX:
        return "static"
    if shift_px <= _PANNING_P90_PX:
        return "moving"
    return "panning"
