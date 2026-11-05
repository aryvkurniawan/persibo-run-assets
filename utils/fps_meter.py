"""Frame-rate measurement utility.

Usage:
    meter = FPSMeter()
    meter.start()
    ...process one frame...
    meter.stop()
    print(meter.current_fps, meter.average_fps)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional


class FPSMeter:
    """Measures per-frame processing time and derives FPS statistics.

    ``current_fps`` uses a rolling window (smooth, good for an on-screen
    HUD); ``average_fps`` uses all recorded frames (good for the final
    report).
    """

    def __init__(self, window_size: int = 30) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._window: Deque[float] = deque(maxlen=window_size)
        self._start_time: Optional[float] = None
        self._total_time: float = 0.0
        self._frame_count: int = 0

    def start(self) -> None:
        """Mark the beginning of one frame's processing."""
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Mark the end of one frame's processing. Returns the duration (s)."""
        if self._start_time is None:
            raise RuntimeError("FPSMeter.stop() called before start()")
        duration = time.perf_counter() - self._start_time
        self._start_time = None
        self._window.append(duration)
        self._total_time += duration
        self._frame_count += 1
        return duration

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def total_time(self) -> float:
        return self._total_time

    @property
    def current_fps(self) -> float:
        """FPS over the rolling window."""
        if not self._window:
            return 0.0
        window_time = sum(self._window)
        return len(self._window) / window_time if window_time > 0 else 0.0

    @property
    def average_fps(self) -> float:
        """FPS over the whole run."""
        if self._frame_count == 0 or self._total_time <= 0:
            return 0.0
        return self._frame_count / self._total_time

    def summary(self) -> Dict[str, float]:
        return {
            "frames": self._frame_count,
            "total_seconds": round(self._total_time, 3),
            "average_fps": round(self.average_fps, 2),
        }
