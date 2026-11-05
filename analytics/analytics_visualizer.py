"""Phase 5 — speed overlay for the final video (optional, cosmetic).

Draws each player's current smoothed speed next to them, e.g. ``#7 | 18.4
km/h``. Pure drawing on a frame; the speed values come from the analytics
result via :func:`build_speed_lookup`, so this module never recomputes
anything. Disabled unless ``analytics.show_speed_overlay`` is set.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from analytics.analytics_models import AnalyticsResult

# (frame_index, track_id) -> speed_kmh
SpeedLookup = Dict[Tuple[int, int], float]
# (frame_index, track_id) -> (speed_kmh, cumulative_distance_m)
StatLookup = Dict[Tuple[int, int], Tuple[float, float]]

_FONT_SCALE = 0.5
_BALL = "ball"
# A speed at/above this fills the mini speed bar (km/h).
_BAR_FULL_KMH = 34.0


def build_speed_lookup(result: AnalyticsResult) -> SpeedLookup:
    """Flatten per-track smoothed speeds to ``{(frame, track_id): km/h}``."""
    lookup: SpeedLookup = {}
    for track in result.tracks:
        for frame_index, kmh in track.frame_speed_kmh.items():
            lookup[(int(frame_index), int(track.track_id))] = float(kmh)
    return lookup


def build_stat_lookup(result: AnalyticsResult) -> StatLookup:
    """``{(frame, track_id): (speed_kmh, cumulative_distance_m)}`` for the bar."""
    lookup: StatLookup = {}
    for track in result.tracks:
        for frame_index, kmh in track.frame_speed_kmh.items():
            dist = track.frame_distance_m.get(frame_index, 0.0)
            lookup[(int(frame_index), int(track.track_id))] = (float(kmh), float(dist))
    return lookup


def draw_player_stat_bars(
    frame,
    tracks: Sequence,
    lookup: StatLookup,
    frame_index: int,
    ball_class_name: str = _BALL,
    names_map: Optional[Dict[int, str]] = None,
) -> None:
    """Draw a small stat panel UNDER each player: a speed bar + speed + distance.

    The panel is a semi-transparent rounded box just below the player's feet
    with a coloured speed bar (filled by speed) and two lines, ``<n> km/h`` and
    ``<n> m`` (distance run so far). Players without a speed this frame (e.g.
    the pose/track gap) simply get no panel.
    """
    import cv2

    h_img, w_img = frame.shape[:2]
    pw, ph = 96, 42
    for track in tracks:
        if track.class_name == ball_class_name:
            continue
        stat = lookup.get((int(frame_index), int(track.track_id)))
        if stat is None:
            continue
        speed_kmh, dist_m = stat
        name = names_map.get(track.track_id) if names_map else None
        label = name or f"Player {track.track_id}"
        x1, _y1, x2, y2 = (int(round(v)) for v in track.bbox)
        cx = (x1 + x2) // 2
        px = min(max(cx - pw // 2, 2), w_img - pw - 2)
        py = min(max(y2 + 4, 2), h_img - ph - 2)

        panel = frame[py:py + ph, px:px + pw]
        if panel.shape[:2] != (ph, pw):
            continue
        dark = panel.copy()
        dark[:] = (28, 28, 28)
        cv2.addWeighted(dark, 0.55, panel, 0.45, 0, panel)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (210, 210, 210), 1)

        # Speed bar (top), green->red by speed.
        frac = max(0.0, min(speed_kmh / _BAR_FULL_KMH, 1.0))
        bar_w = int((pw - 8) * frac)
        bar_col = (0, int(220 * (1 - frac)) + 30, int(40 + 215 * frac))  # BGR
        cv2.rectangle(frame, (px + 4, py + 3), (px + 4 + bar_w, py + 7),
                      bar_col, -1, cv2.LINE_AA)

        cv2.putText(frame, _fit_label(label, 12), (px + 4, py + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1,
                    cv2.LINE_AA)
        text_y = py + 29

        cv2.putText(frame, f"{speed_kmh:.0f} km/h", (px + 4, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{dist_m:.0f} m", (px + 4, text_y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 225, 255), 1, cv2.LINE_AA)


def draw_speed_labels(
    frame,
    tracks: Sequence,
    lookup: SpeedLookup,
    frame_index: int,
    ball_class_name: str = _BALL,
    color: Tuple[int, int, int] = (255, 255, 255),
    names_map: Optional[Dict[int, str]] = None,
) -> None:
    """Draw ``#id | X km/h`` above each non-ball track that has a speed."""
    import cv2

    for track in tracks:
        if track.class_name == ball_class_name:
            continue
        kmh = lookup.get((int(frame_index), int(track.track_id)))
        if kmh is None:
            continue
        x1, y1, _x2, _y2 = (int(round(v)) for v in track.bbox)
        name = names_map.get(track.track_id) if names_map else None
        label = name if name else f"Player {track.track_id}"
        text = f"{label} | {kmh:.1f} km/h"
        org = (x1, max(y1 - 18, 12))
        # Shadow for readability over grass, then the text.
        cv2.putText(frame, text, (org[0] + 1, org[1] + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, org,
                    cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, color, 1, cv2.LINE_AA)


def _fit_label(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars - 1] + "."
