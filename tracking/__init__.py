"""Phase 2 — multi-object tracking (BoT-SORT).

Public surface:
    Track, FrameTracks      — tracking data models
    BaseTracker             — abstract tracker contract
    BoTSORTTracker          — BoT-SORT implementation fed by our detections
    TrackJSONExporter       — per-frame tracking JSON export
    TrackVisualizer         — draws track IDs onto frames
"""

from tracking.models import FrameTracks, Track
from tracking.tracker import BaseTracker

__all__ = ["Track", "FrameTracks", "BaseTracker"]
