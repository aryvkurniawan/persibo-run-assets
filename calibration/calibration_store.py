"""Persist manual pitch calibration to ``outputs/pitch_calibration.json``.

On-disk format::

    {
      "video": "input_video.mp4",
      "frame": 250,
      "points": {
        "center_spot": {"image": [960, 540], "field": [52.5, 34.0]}
      }
    }

The ``field`` coordinates are stored for transparency/debugging but are
always derivable from :class:`PitchModel` by name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from calibration.pitch_model import PitchModel
from utils.logger import get_logger

logger = get_logger("calibration.calibration_store")

Point = Tuple[float, float]


def point_in_frame(point: Point, width: int, height: int) -> bool:
    """True if a pixel point lies within ``[0, width] x [0, height]``."""
    x, y = point
    return 0 <= x <= width and 0 <= y <= height


@dataclass
class Calibration:
    """A set of image<->field correspondences for one video frame."""

    video: str = ""
    frame: int = 0
    # name -> {"image": (x, y), "field": (x, y)}
    points: Dict[str, Dict[str, Point]] = field(default_factory=dict)

    def add_point(self, name: str, image_xy: Point, field_xy: Point) -> None:
        self.points[name] = {
            "image": (float(image_xy[0]), float(image_xy[1])),
            "field": (float(field_xy[0]), float(field_xy[1])),
        }

    def remove_point(self, name: str) -> bool:
        return self.points.pop(name, None) is not None

    @property
    def count(self) -> int:
        return len(self.points)

    def ordered_names(self) -> List[str]:
        return list(self.points.keys())

    def image_points(self) -> np.ndarray:
        return np.array(
            [self.points[n]["image"] for n in self.ordered_names()], dtype=np.float64
        )

    def field_points(self) -> np.ndarray:
        return np.array(
            [self.points[n]["field"] for n in self.ordered_names()], dtype=np.float64
        )

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "frame": int(self.frame),
            "points": {
                name: {
                    "image": [round(float(v), 2) for v in p["image"]],
                    "field": [round(float(v), 4) for v in p["field"]],
                }
                for name, p in self.points.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        points: Dict[str, Dict[str, Point]] = {}
        for name, p in (data.get("points") or {}).items():
            points[name] = {
                "image": tuple(float(v) for v in p["image"]),
                "field": tuple(float(v) for v in p["field"]),
            }
        return cls(
            video=str(data.get("video", "")),
            frame=int(data.get("frame", 0)),
            points=points,
        )


class CalibrationStore:
    """Reads/writes the pitch calibration JSON."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> Optional[Calibration]:
        """Return the stored calibration, or ``None`` if the file is absent."""
        if not self.path.is_file():
            logger.info("No calibration file at %s", self.path)
            return None
        with self.path.open("r", encoding="utf-8") as handle:
            return Calibration.from_dict(json.load(handle))

    def save(self, calibration: Calibration) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(calibration.to_dict(), handle, indent=2)
        logger.info("Saved calibration (%d points) to %s",
                    calibration.count, self.path)
        return self.path


@dataclass
class MultiCalibration:
    """Several per-keyframe calibrations for a panning broadcast camera.

    A single fixed homography drifts as the camera pans/zooms, so the user
    calibrates a few keyframes (e.g. one at each goal + midfield). Each
    keyframe gets its own homography; the minimap later picks the keyframe
    nearest in time to the frame being rendered.

    Backward compatible: a legacy single-frame calibration file (with a
    top-level ``points``) loads as a one-keyframe :class:`MultiCalibration`.
    """

    video: str = ""
    keyframes: List[Calibration] = field(default_factory=list)

    @property
    def total_points(self) -> int:
        return sum(k.count for k in self.keyframes)

    def usable_keyframes(self, min_points: int = 4) -> List[Calibration]:
        """Keyframes with enough points to solve a homography."""
        return [k for k in self.keyframes if k.count >= min_points]

    def frame_numbers(self) -> List[int]:
        return [k.frame for k in self.keyframes]

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "frames": [
                {"frame": int(k.frame), "points": k.to_dict()["points"]}
                for k in self.keyframes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MultiCalibration":
        video = str(data.get("video", ""))
        keyframes: List[Calibration] = []
        if "frames" in data:                       # multi-frame format
            for entry in data["frames"]:
                keyframes.append(Calibration.from_dict({
                    "video": video,
                    "frame": entry.get("frame", 0),
                    "points": entry.get("points", {}),
                }))
        elif "points" in data:                     # legacy single-frame format
            keyframes.append(Calibration.from_dict(data))
        return cls(video=video, keyframes=keyframes)


class MultiCalibrationStore:
    """Reads/writes a multi-keyframe calibration (also reads legacy files)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> Optional[MultiCalibration]:
        if not self.path.is_file():
            logger.info("No calibration file at %s", self.path)
            return None
        with self.path.open("r", encoding="utf-8") as handle:
            return MultiCalibration.from_dict(json.load(handle))

    def save(self, calibration: MultiCalibration) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(calibration.to_dict(), handle, indent=2)
        logger.info("Saved multi-calibration (%d keyframes, %d points) to %s",
                    len(calibration.keyframes), calibration.total_points, self.path)
        return self.path


def calibration_quality(count: int, min_points: int = 4, recommended: int = 8) -> dict:
    """Summarize whether a calibration has enough points."""
    return {
        "count": count,
        "usable": count >= min_points,
        "recommended": count >= recommended,
        "warning": (
            None if count >= recommended
            else f"Only {count} point(s); {recommended}+ recommended for accuracy"
            if count >= min_points
            else f"Need at least {min_points} points (have {count})"
        ),
    }
