"""Automatic pitch calibration from the NBJW HRNet keypoint model.

A second auto-calibration backend alongside the 28-point YOLO-pose detector
in :mod:`calibration.pitch_keypoints`. The model here is an HRNetV2-w48 (the
"No Bells Just Whistles" SoccerNet calibration network) that predicts **57
pitch keypoints** as heatmaps. Measured on a broadcast clasico clip it
calibrates at ~0.30 m mean reprojection error vs ~1.36 m for the YOLO model,
so it is markedly more accurate on hard broadcast footage.

Pipeline per frame: resize to 540x960, run HRNet -> ``(58, 270, 480)``
heatmaps (57 keypoints + background); drop background, take each channel's
peak above ``kp_threshold``; map the peak straight back to frame pixels;
solve the image->field homography (RANSAC, inliers only) against the
published 57-point pitch template.

Decode + solve are pure (numpy / OpenCV) and unit-tested without torch; only
:class:`NBJWPitchDetector` touches the model and accepts an injected model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from calibration.homography import Homography
from utils.logger import get_logger

logger = get_logger("calibration.nbjw_keypoints")

Point = Tuple[float, float]
Keypoint = Tuple[int, float, float, float]   # (index, x_px, y_px, confidence)

NBJW_INPUT_HW: Tuple[int, int] = (540, 960)
NBJW_NUM_JOINTS: int = 58                      # 57 keypoints + 1 background

# 57 keypoint world coordinates on the 105x68 m pitch (NBJW channel order).
_KEYPOINT_WORLD_2D: List[Point] = [
    (0., 0.), (52.5, 0.), (105., 0.), (0., 13.84), (16.5, 13.84), (88.5, 13.84),
    (105., 13.84), (0., 24.84), (5.5, 24.84), (99.5, 24.84), (105., 24.84),
    (0., 30.34), (0., 30.34), (105., 30.34), (105., 30.34), (0., 37.66),
    (0., 37.66), (105., 37.66), (105., 37.66), (0., 43.16), (5.5, 43.16),
    (99.5, 43.16), (105., 43.16), (0., 54.16), (16.5, 54.16), (88.5, 54.16),
    (105., 54.16), (0., 68.), (52.5, 68.), (105., 68.), (16.5, 26.68),
    (52.5, 24.85), (88.5, 26.68), (16.5, 41.31), (52.5, 43.15), (88.5, 41.31),
    (19.99, 32.29), (43.68, 31.53), (61.31, 31.53), (85., 32.29), (19.99, 35.7),
    (43.68, 36.46), (61.31, 36.46), (85., 35.7), (11., 34.), (16.5, 34.),
    (20.15, 34.), (46.03, 27.53), (58.97, 27.53), (43.35, 34.), (52.5, 34.),
    (61.5, 34.), (46.03, 40.47), (58.97, 40.47), (84.85, 34.), (88.5, 34.),
    (94., 34.),
]
# Channels 11..18 are the four goalposts (top/base): each pair shares a 2D
# ground coordinate but can fire on the elevated post top, off the ground
# plane. Dropping them keeps the planar homography clean (49 points remain).
_GOALPOST_INDICES = frozenset(range(11, 19))
NBJW_KEYPOINT_FIELD: Dict[int, Point] = {
    i: p for i, p in enumerate(_KEYPOINT_WORLD_2D) if i not in _GOALPOST_INDICES
}

HRNET_W48_CFG: dict = {
    "MODEL": {
        "IMAGE_SIZE": [960, 540],
        "NUM_JOINTS": NBJW_NUM_JOINTS,
        "PRETRAIN": "",
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK",
                       "NUM_BLOCKS": [4], "NUM_CHANNELS": [64], "FUSE_METHOD": "SUM"},
            "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [4, 4], "NUM_CHANNELS": [48, 96],
                       "FUSE_METHOD": "SUM"},
            "STAGE3": {"NUM_MODULES": 4, "NUM_BRANCHES": 3, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [4, 4, 4], "NUM_CHANNELS": [48, 96, 192],
                       "FUSE_METHOD": "SUM"},
            "STAGE4": {"NUM_MODULES": 3, "NUM_BRANCHES": 4, "BLOCK": "BASIC",
                       "NUM_BLOCKS": [4, 4, 4, 4],
                       "NUM_CHANNELS": [48, 96, 192, 384], "FUSE_METHOD": "SUM"},
        },
    }
}


# ---------------------------------------------------------------------------
# Pure helpers (no torch)
# ---------------------------------------------------------------------------
def decode_heatmaps(
    heatmaps: np.ndarray, frame_width: int, frame_height: int, threshold: float
) -> List[Keypoint]:
    """Turn a ``(C, H, W)`` keypoint-heatmap stack into frame-pixel keypoints.

    Each channel's single strongest pixel is the keypoint, kept only if its
    peak clears ``threshold``. Heatmap coordinates map linearly back to the
    frame (the model input is a plain resize, no letterbox).
    """
    if heatmaps.ndim != 3:
        raise ValueError("heatmaps must be (C, H, W)")
    n_channels, hm_h, hm_w = heatmaps.shape
    sx, sy = frame_width / float(hm_w), frame_height / float(hm_h)
    flat = heatmaps.reshape(n_channels, -1)
    peak_idx = flat.argmax(axis=1)
    peak_val = flat[np.arange(n_channels), peak_idx]
    out: List[Keypoint] = []
    for c in range(n_channels):
        score = float(peak_val[c])
        if score < threshold:
            continue
        row, col = divmod(int(peak_idx[c]), hm_w)
        out.append((c, (col + 0.5) * sx, (row + 0.5) * sy, score))
    return out


def solve_homography(
    keypoints: Sequence[Keypoint],
    template: Dict[int, Point] = NBJW_KEYPOINT_FIELD,
    min_points: int = 5,
    max_error_m: float = 4.0,
    ransac_thresh_m: float = 3.0,
) -> Optional[Homography]:
    """Solve an image->field homography from detected keypoints (RANSAC).

    The HRNet head emits one peak per channel, so a few fire on spurious
    spots; RANSAC discards those and the returned :class:`Homography` keeps
    only the inlier correspondences (its ``reprojection_error`` isn't inflated
    by outliers). Returns ``None`` if too few inliers / degenerate / over
    ``max_error_m``.
    """
    image: List[Point] = []
    field: List[Point] = []
    for index, x, y, _conf in keypoints:
        point = template.get(int(index))
        if point is not None:
            image.append((float(x), float(y)))
            field.append(point)
    need = max(min_points, 4)
    if len(image) < need:
        return None
    img = np.asarray(image, dtype=np.float64)
    fld = np.asarray(field, dtype=np.float64)
    if len(img) >= 6:
        matrix, mask = cv2.findHomography(img, fld, cv2.RANSAC, ransac_thresh_m)
    else:
        matrix, _ = cv2.findHomography(img, fld, 0)
        mask = None
    if matrix is None:
        return None
    if mask is not None:
        inliers = mask.ravel().astype(bool)
        if int(inliers.sum()) < need:
            return None
        img, fld = img[inliers], fld[inliers]
    homography = Homography(matrix, img, fld)
    if homography.reprojection_error() > max_error_m:
        return None
    return homography


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class NBJWPitchDetector:
    """Wraps the HRNet NBJW keypoint model and yields per-frame homographies.

    Drop-in for :class:`calibration.pitch_keypoints.KeypointPitchDetector`
    (same ``detect`` / ``frame_homography`` surface).
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        kp_threshold: float = 0.1486,
        min_points: int = 5,
        max_error_m: float = 4.0,
        input_hw: Tuple[int, int] = NBJW_INPUT_HW,
        model=None,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.kp_threshold = kp_threshold
        self.min_points = min_points
        self.max_error_m = max_error_m
        self.input_hw = input_hw
        self._model = model
        self._device_resolved: Optional[str] = None

    def _resolve_device(self) -> str:
        if self._device_resolved is None:
            if self.device and self.device != "auto":
                self._device_resolved = self.device
            else:
                import torch
                self._device_resolved = (
                    "cuda:0" if torch.cuda.is_available() else "cpu")
        return self._device_resolved

    def _load(self):
        if self._model is None:
            import torch

            from calibration.nbjw import get_cls_net

            logger.info("Loading NBJW HRNet keypoint model from %s", self.model_path)
            model = get_cls_net(HRNET_W48_CFG)
            state = torch.load(
                self.model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            model.to(self._resolve_device())
            model.eval()
            self._model = model
        return self._model

    def detect(self, frame: np.ndarray) -> List[Keypoint]:
        """Detect the pitch keypoints on a BGR frame (original-frame pixels)."""
        import torch

        model = self._load()
        h0, w0 = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        )
        tensor = torch.nn.functional.interpolate(
            tensor, size=self.input_hw, mode="bilinear", align_corners=False)
        tensor = tensor.to(self._resolve_device())
        with torch.no_grad():
            heatmaps = model(tensor)[0]            # (58, 270, 480)
        heatmaps = heatmaps[:-1]                   # drop background -> 57
        return decode_heatmaps(
            heatmaps.detach().cpu().numpy(), w0, h0, self.kp_threshold)

    def frame_homography(self, frame: np.ndarray) -> Optional[Homography]:
        return solve_homography(
            self.detect(frame), NBJW_KEYPOINT_FIELD,
            self.min_points, self.max_error_m)
