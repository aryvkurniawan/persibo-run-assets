"""Phase 6 — pure rating / classification logic for player performance.

All functions here are deterministic and take already-normalized inputs
(0..1, relative to the squad) so they are trivially unit-testable. The
thresholds are tuned for football: high distance + steady involvement = high
work rate, high speed + distance = high activity, distance + a late speed
drop = fatigue. ``player_rating`` is a weighted blend on a 0..10 scale and
``generate_insight`` turns the labels into a short scouting-style sentence.
"""

from __future__ import annotations

from typing import Tuple

WorkRate = str       # "High" | "Medium" | "Low"
ActivityLevel = str  # "High" | "Medium" | "Low"
FatigueLevel = str   # "Low" | "Medium" | "High"
PerformanceStatus = str  # "Excellent" | "Good" | "Average" | "Poor"

_HI, _LO = 0.66, 0.33


def _clamp01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _three(score: float, labels: Tuple[str, str, str]) -> str:
    """Map a 0..1 score to a 3-way label (labels = high, medium, low)."""
    if score >= _HI:
        return labels[0]
    if score >= _LO:
        return labels[1]
    return labels[2]


# ---------------------------------------------------------------------------
# Component scores (0..1)
# ---------------------------------------------------------------------------
def work_rate_score(distance_norm: float, consistency: float) -> float:
    """Distance covered + how consistently the player kept moving."""
    return _clamp01(0.6 * _clamp01(distance_norm) + 0.4 * _clamp01(consistency))


def activity_score(speed_norm: float, distance_norm: float) -> float:
    """How busy: average speed blended with total ground covered."""
    return _clamp01(0.5 * _clamp01(speed_norm) + 0.5 * _clamp01(distance_norm))


def fatigue_score(distance_norm: float, speed_drop: float, activity: float) -> float:
    """Rises with distance run, a late-match speed drop, and intensity."""
    return _clamp01(
        0.45 * _clamp01(distance_norm)
        + 0.40 * _clamp01(speed_drop)
        + 0.15 * _clamp01(activity)
    )


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def classify_work_rate(distance_norm: float, consistency: float) -> WorkRate:
    return _three(work_rate_score(distance_norm, consistency),
                  ("High", "Medium", "Low"))


def classify_activity(speed_norm: float, distance_norm: float) -> ActivityLevel:
    return _three(activity_score(speed_norm, distance_norm),
                  ("High", "Medium", "Low"))


def classify_fatigue(distance_norm: float, speed_drop: float,
                     activity: float) -> FatigueLevel:
    # Note the label order: a HIGH score means HIGH fatigue.
    return _three(fatigue_score(distance_norm, speed_drop, activity),
                  ("High", "Medium", "Low"))


def player_rating(distance_norm: float, speed_norm: float,
                  activity: float, work_rate_s: float) -> float:
    """Weighted, normalized blend on a 0..10 scale (one decimal)."""
    rating = (
        0.35 * _clamp01(distance_norm)
        + 0.25 * _clamp01(speed_norm)
        + 0.20 * _clamp01(activity)
        + 0.20 * _clamp01(work_rate_s)
    ) * 10.0
    return round(min(max(rating, 0.0), 10.0), 1)


def performance_status(rating: float) -> PerformanceStatus:
    if rating >= 8.0:
        return "Excellent"
    if rating >= 6.0:
        return "Good"
    if rating >= 4.0:
        return "Average"
    return "Poor"


def generate_insight(
    role: str,
    work_rate: WorkRate,
    activity: ActivityLevel,
    fatigue: FatigueLevel,
    rating: float,
    max_speed_kmh: float,
) -> str:
    """Short scouting-style summary built from the classified labels."""
    if role == "goalkeeper":
        base = ("Commanded the box with steady involvement."
                if activity != "Low" else "Limited involvement, as expected for a keeper.")
        return base

    if activity == "High":
        lead = "Very active and frequently involved in play."
    elif activity == "Low":
        lead = "Low movement contribution."
    else:
        lead = "Steady involvement across the match."

    if fatigue == "Low" and work_rate == "High":
        tail = " Maintained high intensity throughout."
    elif fatigue == "High":
        tail = " Intensity dropped in the later stages."
    elif work_rate == "High":
        tail = " Covered a lot of ground."
    else:
        tail = ""

    if max_speed_kmh >= 30.0:
        tail += " Showed strong top-end pace."

    return (lead + tail).strip()
