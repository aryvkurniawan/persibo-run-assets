"""Data structures for Phase 5 — speed & distance analytics.

A :class:`TrackAnalytics` is the per-track summary (distance, average/max
speed, sample counts); :class:`AnalyticsResult` aggregates them with the
team / role distance breakdowns. These are the contract between
:mod:`analytics.speed_distance` (computation), :mod:`analytics.analytics_exporter`
(JSON) and :mod:`analytics.analytics_visualizer` (overlay) — no module passes
raw positions to another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrackAnalytics:
    """Speed / distance summary for one track identity.

    ``frame_speed_kmh`` holds the smoothed per-frame speed used only by the
    video overlay; it is intentionally excluded from :meth:`to_dict`.
    """

    track_id: int
    role: str
    team_id: Optional[int]
    total_distance_m: float = 0.0
    avg_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    valid_samples: int = 0
    invalid_jumps: int = 0
    insufficient_data: bool = False
    player_name: Optional[str] = None      # manual name (from corrections)
    # Per-frame smoothed speed (km/h) and running cumulative distance (m),
    # used only by the on-video stat bar; excluded from to_dict.
    frame_speed_kmh: Dict[int, float] = field(default_factory=dict)
    frame_distance_m: Dict[int, float] = field(default_factory=dict)

    @property
    def player_display_name(self) -> str:
        return self.player_name or f"Player {self.track_id}"

    def to_dict(self) -> dict:
        data = {
            "track_id": int(self.track_id),
            "role": self.role,
            "team_id": (None if self.team_id is None else int(self.team_id)),
            "player_display_name": self.player_display_name,
            "total_distance_m": round(float(self.total_distance_m), 2),
            "avg_speed_kmh": round(float(self.avg_speed_kmh), 2),
            "max_speed_kmh": round(float(self.max_speed_kmh), 2),
            "valid_samples": int(self.valid_samples),
            "invalid_jumps": int(self.invalid_jumps),
            "insufficient_data": bool(self.insufficient_data),
        }
        if self.player_name:
            data["player_name"] = self.player_name
        return data


@dataclass
class AnalyticsResult:
    """All per-track analytics plus distance breakdowns for a video."""

    fps: float
    tracks: List[TrackAnalytics] = field(default_factory=list)
    total_distance_m: float = 0.0
    distance_by_team: Dict[Optional[int], float] = field(default_factory=dict)
    distance_by_role: Dict[str, float] = field(default_factory=dict)

    def team_distance(self, team_id: Optional[int]) -> float:
        return float(self.distance_by_team.get(team_id, 0.0))
