"""OpenCV-based video reading and writing.

Both classes are context managers so resources are always released:

    with VideoReader("match.mp4") as reader:
        for frame_index, frame in reader.frames():
            ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger("utils.video_io")


class VideoReader:
    """Iterates over the frames of a video file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise IOError(f"OpenCV could not open video: {self.path}")

        self.fps: float = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.width: int = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # CAP_PROP_FRAME_COUNT can be unreliable / negative for some codecs.
        raw_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count: Optional[int] = raw_count if raw_count > 0 else None

        logger.info(
            "Opened video %s (%dx%d @ %.2f fps, %s frames)",
            self.path.name, self.width, self.height, self.fps,
            self.frame_count if self.frame_count else "unknown",
        )

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(frame_index, frame)`` pairs. Frame indices are 1-based."""
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            index += 1
            yield index, frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


class VideoWriter:
    """Writes frames to a video file, creating parent directories as needed."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise IOError(
                f"OpenCV could not open writer for {self.path} (codec={codec})"
            )
        self._frames_written = 0
        logger.info("Writing output video to %s", self.path)

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
        self._frames_written += 1

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()
