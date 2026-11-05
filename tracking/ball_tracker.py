"""Custom ball association — no Ultralytics dependency.

Why the ball needs its own tracker
-----------------------------------
The ball's bounding box is only ~6-7 px wide but it moves ~8-9 px per
frame, so two consecutive ball detections frequently have **zero IoU**.
BoT-SORT associates by IoU and a freshly created track has zero velocity,
so neither IoU nor the Kalman prediction can bridge that first gap, and a
stable ball track never forms.

This tracker fixes that with three ideas:

1. **Inflated-box IoU** — the ball box is enlarged (``matching_box_scale``,
   floored at ``min_box_size``) *for matching only*, so neighbouring
   detections overlap again. The exported/visualized bbox stays the true,
   un-inflated box.
2. **Center-distance fallback** — when even the inflated IoU is zero, a
   match is allowed if the centers are within ``distance_gate_px``.
3. **Velocity prediction** — once a track has moved, its center is
   predicted forward (``center + velocity * dt``) before gating, so it can
   re-acquire the ball across short gaps.

A track is confirmed (and only then exported) after **2 consecutive
matches**, which suppresses one-frame false positives.

The tracker consumes ``Detection`` dataclasses and emits ``Track``
dataclasses; it never mutates the input detections.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence, Tuple

from detection.data_models import Detection
from tracking.models import Track
from utils.config_loader import BallTrackingConfig
from utils.logger import get_logger

logger = get_logger("tracking.ball_tracker")

# Ball IDs are offset so they can never collide with the player/referee
# tracker, whose IDs come from Ultralytics' global BaseTrack counter.
_BALL_TRACK_ID_BASE = 9000
# Consecutive matches required before a ball track is exported.
_MIN_CONSECUTIVE_HITS = 2

Box = Tuple[float, float, float, float]


def _center(bbox: Box) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _inflate(bbox: Box, scale: float, min_size: float) -> Box:
    """Enlarge a box about its center for matching purposes only."""
    cx, cy = _center(bbox)
    w = max((bbox[2] - bbox[0]) * scale, min_size)
    h = max((bbox[3] - bbox[1]) * scale, min_size)
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def _iou(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    if inter <= 0:
        return 0.0
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _BallTrack:
    """Mutable internal state for one ball hypothesis."""

    __slots__ = (
        "track_id", "bbox", "center", "velocity", "consecutive_hits",
        "observations", "confirmed", "last_frame", "confidence", "source_id",
    )

    def __init__(self, track_id: int, detection: Detection, frame_index: int,
                 source_id: int):
        self.track_id = track_id
        self.bbox: Box = detection.bbox
        self.center = _center(detection.bbox)
        self.velocity = (0.0, 0.0)
        self.consecutive_hits = 1
        self.observations = 1
        self.confirmed = False
        self.last_frame = frame_index
        self.confidence = detection.confidence
        self.source_id = source_id

    def predicted_center(self, frame_index: int) -> Tuple[float, float]:
        dt = frame_index - self.last_frame
        return (
            self.center[0] + self.velocity[0] * dt,
            self.center[1] + self.velocity[1] * dt,
        )

    def predicted_bbox(self, frame_index: int) -> Box:
        dt = frame_index - self.last_frame
        dx, dy = self.velocity[0] * dt, self.velocity[1] * dt
        return (
            self.bbox[0] + dx, self.bbox[1] + dy,
            self.bbox[2] + dx, self.bbox[3] + dy,
        )

    def register_match(self, detection: Detection, frame_index: int,
                       source_id: int) -> None:
        dt = max(frame_index - self.last_frame, 1)
        new_center = _center(detection.bbox)
        measured_v = (
            (new_center[0] - self.center[0]) / dt,
            (new_center[1] - self.center[1]) / dt,
        )
        if self.observations == 1:
            self.velocity = measured_v          # first motion estimate
        else:
            self.velocity = (
                0.5 * self.velocity[0] + 0.5 * measured_v[0],
                0.5 * self.velocity[1] + 0.5 * measured_v[1],
            )
        self.bbox = detection.bbox              # TRUE bbox, never inflated
        self.center = new_center
        self.last_frame = frame_index
        self.confidence = detection.confidence
        self.source_id = source_id
        self.consecutive_hits += 1
        self.observations += 1
        if self.consecutive_hits >= _MIN_CONSECUTIVE_HITS:
            self.confirmed = True

    def register_miss(self) -> None:
        # Confirmation is sticky; only the consecutive-hit streak resets.
        self.consecutive_hits = 0


class BallTracker:
    """Center-distance + velocity + inflated-IoU tracker for the ball only."""

    def __init__(
        self,
        config: BallTrackingConfig,
        ball_class_id: int,
        ball_class_name: str,
        debug: bool = False,
    ) -> None:
        self._config = config
        self._ball_class_id = ball_class_id
        self._ball_class_name = ball_class_name
        self._tracks: List[_BallTrack] = []
        self._next_id = _BALL_TRACK_ID_BASE + 1
        if debug:
            logger.setLevel(logging.DEBUG)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = _BALL_TRACK_ID_BASE + 1

    def update(
        self,
        indexed_detections: Sequence[Tuple[int, Detection]],
        frame_index: int,
    ) -> List[Track]:
        """Advance the ball tracker by one frame.

        Args:
            indexed_detections: ``(source_detection_id, Detection)`` pairs
                for the ball-class detections on this frame.
            frame_index: 1-based frame number.

        Returns:
            Confirmed ball tracks matched on this frame.
        """
        confs = [round(d.confidence, 2) for _, d in indexed_detections]
        logger.debug(
            "frame %d: %d ball detection(s) conf=%s",
            frame_index, len(indexed_detections), confs,
        )

        self._drop_stale_tracks(frame_index)

        pre_existing_ids = {t.track_id for t in self._tracks}
        matched_track_ids = self._associate(indexed_detections, frame_index)

        # Reset the streak on pre-existing tracks not matched this frame.
        # Tracks newly created this frame are NOT treated as a miss.
        for track in self._tracks:
            if (
                track.track_id in pre_existing_ids
                and track.track_id not in matched_track_ids
            ):
                track.register_miss()

        emitted: List[Track] = []
        for track in self._tracks:
            if track.track_id in matched_track_ids and track.confirmed:
                emitted.append(self._to_track(track, frame_index))
        return emitted

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------
    def _associate(
        self,
        indexed_detections: Sequence[Tuple[int, Detection]],
        frame_index: int,
    ) -> set:
        scale = self._config.matching_box_scale
        min_size = self._config.min_box_size
        gate = self._config.distance_gate_px

        # Build all viable (cost, track_idx, det_idx) candidates.
        candidates: List[Tuple[float, int, int, float, float]] = []
        for ti, track in enumerate(self._tracks):
            pred_box = _inflate(track.predicted_bbox(frame_index), scale, min_size)
            pred_center = track.predicted_center(frame_index)
            for di, (_, det) in enumerate(indexed_detections):
                det_box = _inflate(det.bbox, scale, min_size)
                iou = _iou(pred_box, det_box)
                dcx, dcy = _center(det.bbox)
                dist = math.hypot(pred_center[0] - dcx, pred_center[1] - dcy)
                if iou > 0:
                    cost = 1.0 - iou           # IoU matches preferred
                elif dist <= gate:
                    cost = 1.0 + dist / gate    # ranked after any IoU match
                else:
                    continue                    # outside both gates
                candidates.append((cost, ti, di, iou, dist))

        candidates.sort(key=lambda c: c[0])
        used_tracks: set = set()
        used_dets: set = set()
        matched_track_ids: set = set()

        for cost, ti, di, iou, dist in candidates:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            track = self._tracks[ti]
            source_id, det = indexed_detections[di]
            track.register_match(det, frame_index, source_id)
            matched_track_ids.add(track.track_id)
            logger.debug(
                "  matched ball -> track %d (inflated_iou=%.3f, "
                "center_dist=%.1fpx, hits=%d, confirmed=%s)",
                track.track_id, iou, dist, track.consecutive_hits,
                track.confirmed,
            )

        # Unmatched detections start new tentative tracks.
        for di, (source_id, det) in enumerate(indexed_detections):
            if di in used_dets:
                continue
            track = _BallTrack(self._next_id, det, frame_index, source_id)
            self._next_id += 1
            self._tracks.append(track)
            logger.debug(
                "  created ball track %d at center=(%.1f, %.1f) (unconfirmed)",
                track.track_id, track.center[0], track.center[1],
            )

        return matched_track_ids

    def _drop_stale_tracks(self, frame_index: int) -> None:
        max_gap = self._config.max_frame_gap
        kept: List[_BallTrack] = []
        for track in self._tracks:
            if frame_index - track.last_frame > max_gap:
                logger.debug(
                    "  dropping ball track %d (no match for %d frames)",
                    track.track_id, frame_index - track.last_frame,
                )
            else:
                kept.append(track)
        self._tracks = kept

    def _to_track(self, track: _BallTrack, frame_index: int) -> Track:
        return Track(
            frame_id=frame_index,
            track_id=track.track_id,
            class_id=self._ball_class_id,
            class_name=self._ball_class_name,
            confidence=track.confidence,
            bbox=track.bbox,                    # true, un-inflated box
            source_detection_id=track.source_id,
        )
