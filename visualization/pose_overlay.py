"""Player pose-skeleton overlay (optional, cosmetic).

Runs a pretrained human-pose model (YOLO11-pose, COCO 17 keypoints) and
draws a team-colored skeleton on each player instead of a plain box, for a
broadcast-style look. It is purely visual — tracking, teams, roles and the
minimap all keep working off the bounding boxes. Pose detections are tied
to the existing tracks by bbox IoU, so each skeleton inherits that track's
id and team color; players the pose model misses fall back to a box.

The model wrapper loads lazily and fails soft (no skeletons, never a crash)
so this can't break the pipeline. The matching/geometry helpers are pure
and unit-tested without the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger("visualization.pose_overlay")

Box = Tuple[float, float, float, float]

# COCO-17 keypoint order (ultralytics pose output).
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
# Bones to draw (pairs of keypoint indices).
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6),                                  # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),         # arms
    (5, 11), (6, 12), (11, 12),              # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def assign_poses_to_tracks(
    poses: Sequence[Tuple[Box, np.ndarray]],
    tracks: Sequence,
    iou_threshold: float = 0.3,
) -> Dict[int, np.ndarray]:
    """Match pose detections to tracks by best IoU (greedy, 1:1).

    ``poses`` is ``[(bbox, keypoints[17,3]), ...]``; ``tracks`` are objects
    with ``track_id`` and ``bbox``. Returns ``{track_id: keypoints}``.
    """
    pairs = []
    for pi, (pbox, _kpts) in enumerate(poses):
        for track in tracks:
            score = iou(pbox, track.bbox)
            if score >= iou_threshold:
                pairs.append((score, pi, track.track_id))
    pairs.sort(reverse=True)
    used_pose: set = set()
    used_track: set = set()
    assigned: Dict[int, np.ndarray] = {}
    for _score, pi, track_id in pairs:
        if pi in used_pose or track_id in used_track:
            continue
        used_pose.add(pi)
        used_track.add(track_id)
        assigned[track_id] = poses[pi][1]
    return assigned


def draw_pose(
    canvas: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int],
    min_conf: float = 0.5,
    line_thickness: Optional[int] = None,
    joint_radius: Optional[int] = None,
) -> None:
    """Draw a team-colored skeleton + joints on ``canvas``.

    When ``line_thickness``/``joint_radius`` are not given they scale with
    the skeleton's on-screen size, so small/distant players get thin clean
    lines instead of thick blobs.
    """
    import cv2

    pts = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
    if line_thickness is None or joint_radius is None:
        vis = pts[pts[:, 2] >= min_conf]
        extent = (vis[:, 1].max() - vis[:, 1].min()) if len(vis) >= 2 else 60.0
        if line_thickness is None:
            line_thickness = int(min(max(round(extent / 55.0), 1), 4))
        if joint_radius is None:
            joint_radius = int(min(max(round(extent / 70.0), 1), 4))
    for a, b in COCO_SKELETON:
        if a < len(pts) and b < len(pts) and pts[a, 2] >= min_conf \
                and pts[b, 2] >= min_conf:
            pa = (int(pts[a, 0]), int(pts[a, 1]))
            pb = (int(pts[b, 0]), int(pts[b, 1]))
            cv2.line(canvas, pa, pb, color, line_thickness, cv2.LINE_AA)
    for x, y, c in pts:
        if c >= min_conf:
            cv2.circle(canvas, (int(x), int(y)), joint_radius, (255, 255, 255),
                       -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), joint_radius, color, 1, cv2.LINE_AA)


class PoseEstimator:
    """Lazy wrapper over a YOLO11-pose model (fails soft)."""

    def __init__(
        self,
        model_path: str | Path = "yolo11m-pose.pt",
        device: str = "auto",
        conf: float = 0.25,
        crop_height: int = 192,
        ball_class_name: str = "ball",
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.conf = conf
        self.crop_height = int(crop_height)
        self.ball_class_name = ball_class_name
        self._model = None
        self._failed = False

    def _load(self):
        if self._model is None and not self._failed:
            try:
                from ultralytics import YOLO
                logger.info("Loading pose model %s", self.model_path)
                self._model = YOLO(self.model_path)
            except Exception as exc:           # download / load failure
                logger.warning("Pose model unavailable (%s); skeletons off", exc)
                self._failed = True
        return self._model

    def detect(self, frame: np.ndarray) -> List[Tuple[Box, np.ndarray]]:
        """Return ``[(bbox, keypoints[17,3]), ...]`` for the frame."""
        model = self._load()
        if model is None:
            return []
        kwargs = {"conf": self.conf, "verbose": False}
        if self.device and self.device != "auto":
            kwargs["device"] = self.device
        try:
            result = model.predict(frame, **kwargs)[0]
        except Exception as exc:
            logger.warning("Pose inference failed (%s); skeletons off", exc)
            self._failed = True
            return []
        if result.keypoints is None or result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        kpts = result.keypoints.data.cpu().numpy()    # (N, 17, 3)
        out: List[Tuple[Box, np.ndarray]] = []
        for box, kp in zip(boxes, kpts):
            out.append((tuple(float(v) for v in box[:4]), kp))
        return out

    def detect_on_tracks(
        self, frame: np.ndarray, tracks: Sequence, pad: int = 8
    ) -> Dict[int, np.ndarray]:
        """Per-player pose (crop + upscale + batch) -> ``{track_id: keypoints}``.

        Off-the-shelf pose models miss small, distant broadcast players on
        the full frame, so each non-ball track is cropped, upscaled to
        ``crop_height`` and posed in one BATCHED call; the keypoints are then
        mapped back to full-frame pixel coordinates.
        """
        import cv2

        model = self._load()
        if model is None:
            return {}
        h, w = frame.shape[:2]
        crops: List[np.ndarray] = []
        meta: List[Tuple[int, int, int, float]] = []   # (track_id, x1, y1, scale)
        for track in tracks:
            if track.class_name == self.ball_class_name:
                continue
            x1 = max(int(track.bbox[0]) - pad, 0)
            y1 = max(int(track.bbox[1]) - pad, 0)
            x2 = min(int(track.bbox[2]) + pad, w)
            y2 = min(int(track.bbox[3]) + pad, h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 4:
                continue
            scale = self.crop_height / float(crop.shape[0])
            big = cv2.resize(
                crop, (max(int(crop.shape[1] * scale), 1), self.crop_height))
            crops.append(big)
            meta.append((track.track_id, x1, y1, scale))
        if not crops:
            return {}

        kwargs = {"conf": self.conf, "imgsz": 256, "verbose": False}
        if self.device and self.device != "auto":
            kwargs["device"] = self.device
        try:
            results = model.predict(crops, **kwargs)
        except Exception as exc:
            logger.warning("Pose inference failed (%s); skeletons off", exc)
            self._failed = True
            return {}

        out: Dict[int, np.ndarray] = {}
        for (track_id, x1, y1, scale), res in zip(meta, results):
            if res.keypoints is None or len(res.keypoints) == 0:
                continue
            best = 0
            if res.boxes is not None and len(res.boxes) > 0:
                best = int(np.argmax(res.boxes.conf.cpu().numpy()))
            kp = res.keypoints.data[best].cpu().numpy().copy()   # (17,3) crop coords
            kp[:, 0] = x1 + kp[:, 0] / scale                     # map back to frame
            kp[:, 1] = y1 + kp[:, 1] / scale
            out[track_id] = kp
        return out
