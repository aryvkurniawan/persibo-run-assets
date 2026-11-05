"""ONNX Runtime + DirectML inference shim.

Ultralytics' ONNX backend hardcodes ``CPUExecutionProvider``, which leaves an
AMD GPU unused. This module exposes the same ``predict()`` surface Ultralytics
does, but asks ONNX Runtime for DirectML first. Everything downstream
(:class:`detection.detector.YOLODetector`) is unchanged.

Only used when ``model.weights_path`` points at an ``.onnx`` file.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Sequence

import cv2
import numpy as np

_PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]


def _letterbox(img: np.ndarray, size: int, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to a square of ``size``."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


class _Boxes:
    """Mimics the subset of Ultralytics' ``Boxes`` the pipeline touches."""

    __slots__ = ("xyxy", "conf", "cls")

    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return int(len(self.xyxy))


class OnnxDmlModel:
    """Minimal Ultralytics-compatible wrapper over an ONNX detection model."""

    def __init__(self, path: str, names: Dict[int, str], imgsz: int = 1280):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._session = ort.InferenceSession(str(path), so, providers=_PROVIDERS)
        self._input = self._session.get_inputs()[0].name
        self.names = dict(names)
        self._imgsz = imgsz
        # Output channels = 4 box coords + one score per class. Decide the
        # layout from that, not from shape[:2]: (8, 18900) and (18900, 8)
        # satisfy the same "smaller first" test only by luck.
        self._channels = 4 + len(self.names)
        providers = self._session.get_providers()
        if "DmlExecutionProvider" not in providers:
            raise RuntimeError(
                "DirectML provider unavailable; install onnxruntime-directml. "
                f"Available: {providers}"
            )

    @property
    def providers(self) -> Sequence[str]:
        return self._session.get_providers()

    def predict(self, source, conf=0.25, iou=0.5, imgsz=None, device=None, verbose=False):
        size = int(imgsz or self._imgsz)
        frame = source if isinstance(source, np.ndarray) else cv2.imread(str(source))
        padded, scale, pad_x, pad_y = _letterbox(frame, size)

        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]

        out = self._session.run(None, {self._input: blob})[0]
        pred = np.squeeze(out, axis=0)
        # Accept either (C, N) or (N, C). A model with C == 8 classes makes
        # (8, 18900) and (18900, 8) indistinguishable by shape alone, so pick
        # the axis of size C and normalise to (N, C).
        if pred.ndim != 2:
            raise ValueError(f"Unexpected ONNX output {out.shape}: not 2-D")
        if pred.shape[-1] == self._channels:
            pass                                 # already (N, C)
        elif pred.shape[0] == self._channels:
            pred = pred.T
        else:
            raise ValueError(
                f"Unexpected ONNX output {out.shape}: no axis with "
                f"{self._channels} channels (4 box + {len(self.names)} classes)")

        boxes_cxcywh = pred[:, :4]
        scores_all = pred[:, 4:]
        class_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(class_ids)), class_ids]

        keep = scores >= conf
        if not keep.any():
            return [self._empty()]

        boxes_cxcywh = boxes_cxcywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # cx,cy,w,h -> x1,y1,x2,y2 in letterboxed pixels
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        # undo the letterbox
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        h, w = frame.shape[:2]
        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)

        # NMS per class: a class-agnostic pass would let a high-scoring player
        # box suppress an overlapping referee/goalkeeper/ball box.
        kept = []
        for class_id in np.unique(class_ids):
            mask = class_ids == class_id
            idx = cv2.dnn.NMSBoxes(
                xyxy[mask].tolist(), scores[mask].tolist(), float(conf), float(iou)
            )
            if len(idx):
                kept.append(np.flatnonzero(mask)[np.asarray(idx).reshape(-1)])
        if not kept:
            return [self._empty()]
        idx = np.concatenate(kept)

        return [SimpleNamespace(boxes=_Boxes(
            xyxy=xyxy[idx],
            conf=scores[idx],
            cls=class_ids[idx].astype(np.float32),
        ))]

    @staticmethod
    def _empty():
        return SimpleNamespace(boxes=_Boxes(
            xyxy=np.zeros((0, 4), np.float32),
            conf=np.zeros((0,), np.float32),
            cls=np.zeros((0,), np.float32),
        ))


class _Keypoints:
    """Mimics Ultralytics' ``Keypoints`` — only ``.data`` is consumed."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    def __len__(self) -> int:
        return int(len(self.data))


class OnnxPoseDml:
    """Pitch-keypoint YOLO-pose model on ONNX Runtime + DirectML."""

    def __init__(self, path: str, imgsz: int = 1280, num_keypoints: int = 28):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._session = ort.InferenceSession(str(path), so, providers=_PROVIDERS)
        self._input = self._session.get_inputs()[0].name
        self._imgsz = imgsz
        self._nkpt = num_keypoints
        providers = self._session.get_providers()
        if "DmlExecutionProvider" not in providers:
            raise RuntimeError(f"DirectML unavailable: {providers}")

    @property
    def providers(self) -> Sequence[str]:
        return self._session.get_providers()

    def predict(self, source, conf=0.3, imgsz=None, device=None, verbose=False):
        size = int(imgsz or self._imgsz)
        frame = source if isinstance(source, np.ndarray) else cv2.imread(str(source))
        padded, scale, pad_x, pad_y = _letterbox(frame, size)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]

        out = np.squeeze(self._session.run(None, {self._input: blob})[0], axis=0)
        if out.shape[0] < out.shape[1]:
            out = out.T                       # (N, 4 + 1 + k*3)

        boxes = out[:, :4]
        cls_score = out[:, 4]
        kpts = out[:, 5:].reshape(len(out), self._nkpt, 3)

        keep = cls_score >= conf
        if not keep.any():
            return [SimpleNamespace(boxes=_Boxes(np.zeros((0, 4), np.float32),
                                                 np.zeros((0,), np.float32),
                                                 np.zeros((0,), np.float32)),
                                    keypoints=_Keypoints(np.zeros((0, self._nkpt, 3), np.float32)))]

        boxes, cls_score, kpts = boxes[keep], cls_score[keep], kpts[keep]

        x1 = (boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / scale
        y1 = (boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / scale
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        kpts = kpts.copy()
        kpts[..., 0] = (kpts[..., 0] - pad_x) / scale
        kpts[..., 1] = (kpts[..., 1] - pad_y) / scale

        return [SimpleNamespace(
            boxes=_Boxes(np.stack([x1, y1, x2, y2], 1), cls_score, np.zeros_like(cls_score)),
            keypoints=_Keypoints(kpts),
        )]
