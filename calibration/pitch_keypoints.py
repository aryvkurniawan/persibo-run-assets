"""Automatic pitch calibration from a keypoint (pose) model.

A YOLO-pose model trained to detect 28 named pitch reference points lets
us compute the image->field homography automatically per frame — no
manual clicking, and it follows a panning camera frame by frame.

The 28 keypoint indices follow the Roboflow *football-field-detection*
ordering; :data:`PITCH_KEYPOINT_FIELD` maps each index to its coordinate
on the canonical 105x68 m pitch (:class:`PitchModel`).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

from calibration.homography import Homography
from calibration.pitch_model import PitchDimensions
from utils.logger import get_logger

logger = get_logger("calibration.pitch_keypoints")

Point = Tuple[float, float]

# A detected keypoint: (index, x_px, y_px, confidence).
Keypoint = Tuple[int, float, float, float]


def _build_field_template(d: PitchDimensions = PitchDimensions()) -> Dict[int, Point]:
    """Map each of the 28 keypoint indices to field meters (105x68)."""
    cx, cy = d.length / 2.0, d.width / 2.0
    pa = d.penalty_area_depth                      # 16.5
    ga = d.goal_area_depth                         # 5.5
    pa_t, pa_b = cy - d.penalty_area_width / 2.0, cy + d.penalty_area_width / 2.0
    ga_t, ga_b = cy - d.goal_area_width / 2.0, cy + d.goal_area_width / 2.0
    cc_t, cc_b = cy - d.center_circle_radius, cy + d.center_circle_radius
    # Penalty-arc intersection with the penalty-area line.
    dx = pa - d.penalty_spot_distance              # 16.5 - 11 = 5.5
    arc_dy = math.sqrt(max(d.center_circle_radius ** 2 - dx ** 2, 0.0))
    arc_t, arc_b = cy - arc_dy, cy + arc_dy
    return {
        0: (0.0, 0.0),                  # left top corner
        1: (0.0, pa_t),                 # left goal line - top penalty corner
        2: (0.0, ga_t),                 # left goal line - top goal-area corner
        3: (0.0, ga_b),                 # left goal line - bottom goal-area corner
        4: (0.0, pa_b),                 # left goal line - bottom penalty corner
        5: (0.0, d.width),              # left bottom corner
        6: (ga, ga_t),                  # left goal area - inner top
        7: (ga, ga_b),                  # left goal area - inner bottom
        8: (pa, pa_t),                  # left penalty area - inner top
        9: (pa, pa_b),                  # left penalty area - inner bottom
        10: (cx, d.width),              # halfway - bottom touchline
        11: (d.length, d.width),        # right bottom corner
        12: (d.length, pa_b),           # right goal line - bottom penalty corner
        13: (d.length, ga_b),           # right goal line - bottom goal-area corner
        14: (d.length, ga_t),           # right goal line - top goal-area corner
        15: (d.length, pa_t),           # right goal line - top penalty corner
        16: (d.length, 0.0),            # right top corner
        17: (cx, 0.0),                  # halfway - top touchline
        18: (cx, cc_t),                 # halfway - top center-circle
        19: (cx, cc_b),                 # halfway - bottom center-circle
        20: (d.length - pa, pa_b),      # right penalty area - inner bottom
        21: (d.length - pa, pa_t),      # right penalty area - inner top
        22: (d.length - ga, ga_t),      # right goal area - inner top
        23: (d.length - ga, ga_b),      # right goal area - inner bottom
        24: (pa, arc_b),                # left penalty arc - bottom
        25: (pa, arc_t),                # left penalty arc - top
        26: (d.length - pa, arc_b),     # right penalty arc - bottom
        27: (d.length - pa, arc_t),     # right penalty arc - top
    }


PITCH_KEYPOINT_FIELD: Dict[int, Point] = _build_field_template()
NUM_KEYPOINTS = 28


def homography_from_keypoints(
    keypoints: Sequence[Keypoint],
    min_conf: float = 0.5,
    min_points: int = 5,
    max_error_m: float = 4.0,
    frame_size: Optional[Tuple[int, int]] = None,
) -> Optional[Homography]:
    """Solve an image->field homography from detected pitch keypoints.

    Uses keypoints above ``min_conf`` whose index is in the template.
    Returns ``None`` when there aren't enough points, the solve is
    degenerate, or the reprojection error exceeds ``max_error_m`` (so a
    poorly-seen frame is rejected rather than producing a bad homography).

    Two guards matter more than the reprojection error, because that error
    only measures how well the homography fits the points it was fitted on --
    an 8-DoF solve through a handful of near-collinear points can swing
    hundreds of metres while still reproducing them almost exactly:

    * ``frame_size`` -- keypoints predicted outside the image (the model
      happily returns x=1234 for a 854 px frame) are hallucinated, not seen.
    * the four frame corners must land near the pitch, otherwise the matrix
      extrapolates far outside its support and every projection is nonsense.
    """
    image: List[Point] = []
    field: List[Point] = []
    for index, x, y, conf in keypoints:
        if conf >= min_conf and index in PITCH_KEYPOINT_FIELD:
            if frame_size is not None:
                w, h = frame_size
                if not (0.0 <= x <= w and 0.0 <= y <= h):
                    continue
            image.append((float(x), float(y)))
            field.append(PITCH_KEYPOINT_FIELD[index])
    if len(image) < max(min_points, 4):
        return None
    try:
        homography = Homography.from_correspondences(image, field)
    except ValueError:
        return None
    if homography.reprojection_error() > max_error_m:
        return None
    if frame_size is not None and not _corners_land_near_pitch(
            homography, frame_size):
        return None
    return homography

# Measured on real broadcast footage, the corner spread is bimodal: a usable
# homography puts the frame corners at 1.5-3.5x the pitch size (the camera sees
# a lot of the ground around the field), while a swung solution lands at 100x+.
# Nothing in between, so this threshold only has to sit in the gap.
_CORNER_LIMIT_FACTOR = 5.0

def _corners_land_near_pitch(homography: Homography,
                             frame_size: Tuple[int, int]) -> bool:
    w, h = frame_size
    corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]],
                       dtype=np.float64).reshape(-1, 1, 2)
    try:
        pts = cv2.perspectiveTransform(corners, homography.matrix).reshape(-1, 2)
    except cv2.error:
        return False
    if not np.all(np.isfinite(pts)):
        return False
    d = PitchDimensions()
    limit_x = d.length * _CORNER_LIMIT_FACTOR
    limit_y = d.width * _CORNER_LIMIT_FACTOR
    return bool(
        abs(pts[:, 0]).max() <= limit_x and abs(pts[:, 1]).max() <= limit_y
    )


class KeypointPitchDetector:
    """Wraps the YOLO-pose pitch model and yields per-frame homographies."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        conf: float = 0.3,
        min_conf: float = 0.5,
        min_points: int = 5,
        max_error_m: float = 4.0,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.conf = conf
        self.min_conf = min_conf
        self.min_points = min_points
        self.max_error_m = max_error_m
        self._model = None

    def _load(self):
        if self._model is None:
            if self.model_path.lower().endswith(".onnx"):
                # AMD GPUs have no CUDA; DirectML is the only route to the GPU.
                from detection.onnx_dml import OnnxPoseDml

                logger.info("Loading ONNX pitch model via DirectML from %s", self.model_path)
                self._model = OnnxPoseDml(self.model_path, imgsz=1280)
            else:
                from ultralytics import YOLO
                logger.info("Loading pitch keypoint model from %s", self.model_path)
                self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame: np.ndarray) -> List[Keypoint]:
        """Detect the 28 pitch keypoints on a frame (best pitch instance)."""
        model = self._load()
        kwargs = {"conf": self.conf, "verbose": False}
        if self.device and self.device != "auto":
            kwargs["device"] = self.device
        result = model.predict(frame, **kwargs)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return []
        best = 0
        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf
            confs = confs.cpu().numpy() if hasattr(confs, "cpu") else np.asarray(confs)
            best = int(np.argmax(confs))
        data = result.keypoints.data[best]
        data = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        return [(i, float(x), float(y), float(c)) for i, (x, y, c) in enumerate(data)]

    def frame_homography(self, frame: np.ndarray) -> Optional[Homography]:
        h, w = frame.shape[:2]
        return homography_from_keypoints(
            self.detect(frame), self.min_conf, self.min_points,
            self.max_error_m, frame_size=(w, h))


class KeypointHomographyProvider:
    """Per-frame homographies from keypoints, with nearest-frame fallback.

    Exposes ``for_frame`` so it drops into the projection code exactly like
    :class:`MultiHomography`. Frames where the pitch wasn't clearly seen get
    the homography of the nearest successfully-calibrated frame.
    """

    def __init__(self, entries: Sequence["Tuple[int, Homography]"]) -> None:
        if not entries:
            raise ValueError("no per-frame homographies were computed")
        self._entries = sorted(entries, key=lambda e: e[0])

    def for_frame(self, frame_index: int) -> Homography:
        return min(self._entries, key=lambda e: abs(e[0] - int(frame_index)))[1]

    @property
    def frame_numbers(self) -> List[int]:
        return [f for f, _ in self._entries]

    def reprojection_error(self) -> float:
        return float(np.mean([h.reprojection_error() for _, h in self._entries]))

    def __len__(self) -> int:
        return len(self._entries)

    @classmethod
    def from_video(
        cls,
        video_path: str | Path,
        detector: KeypointPitchDetector,
        sample_every: int = 5,
        max_frames: Optional[int] = None,
        max_anchors: int = 0,
    ) -> Optional["KeypointHomographyProvider"]:
        """Compute homographies on every ``sample_every``-th frame of a video.

        With ``max_anchors`` > 0 a static-camera clip keeps only the best
        few anchors instead of one per sampled frame. A well-calibrated
        homography for a static shot is the *same* homography -- keeping
        hundreds of noisy near-duplicates just lets the projection jump
        between them, which is what makes speed and distance unusable.
        """
        from utils.video_io import VideoReader

        entries: List[Tuple[int, Homography]] = []
        attempted = 0
        with VideoReader(video_path) as reader:
            for index, frame in reader.frames():
                if (index - 1) % max(sample_every, 1) != 0:
                    continue
                attempted += 1
                homography = detector.frame_homography(frame)
                if homography is not None:
                    entries.append((index, homography))
                if max_frames is not None and index >= max_frames:
                    break
        logger.info(
            "Auto-calibration: %d/%d sampled frames calibrated from keypoints",
            len(entries), attempted)
        if not entries:
            return None
        if max_anchors > 0 and len(entries) > max_anchors:
            entries = _best_anchors(entries, max_anchors)
            logger.info(
                "Auto-calibration: keeping the %d most stable anchors "
                "(static camera, %d candidates were duplicates)",
                len(entries), attempted)
        return cls(entries)


def _anchor_quality(entry: "Tuple[int, Homography]") -> float:
    """Higher is better: how far the frame corners spread *on the pitch*."""
    homography = entry[1]
    probe = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                     dtype=np.float64).reshape(-1, 1, 2)
    try:
        pts = cv2.perspectiveTransform(probe, homography.matrix).reshape(-1, 2)
    except cv2.error:
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    return float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))


def _best_anchors(
    entries: Sequence["Tuple[int, Homography]"], count: int
) -> List["Tuple[int, Homography]"]:
    """Keep the ``count`` anchors whose homography is best constrained.

    Ranked by how widely the frame corners spread across the pitch: a
    homography fitted from points spread over the field has a small null
    space, one fitted from a cluster of near-collinear points does not.
    """
    ranked = sorted(entries, key=_anchor_quality, reverse=True)
    return sorted(ranked[:count], key=lambda e: e[0])
