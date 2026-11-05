"""Data structures for Phase 3 — track-level role refinement.

A :class:`RoleResult` is the refinement-stage analog of detection's
``Detection`` and tracking's ``Track``: it carries the *corrected* role
for a whole track identity, plus the evidence used to reach it.

A :class:`TrackAppearance` is the intermediate, per-track appearance and
trajectory summary that the clusterer and refiner reason over. It is the
contract between :mod:`appearance_extractor`, :mod:`team_clusterer` and
:mod:`role_refiner` — no module passes raw crops or histograms to another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Allowed refined roles (the only legal values of ``RoleResult.refined_role``)
# ---------------------------------------------------------------------------
ROLE_PLAYER = "player"
ROLE_GOALKEEPER = "goalkeeper"
ROLE_REFEREE = "referee"
ROLE_BALL = "ball"
ROLE_UNKNOWN = "unknown"

ALLOWED_ROLES = (
    ROLE_PLAYER,
    ROLE_GOALKEEPER,
    ROLE_REFEREE,
    ROLE_BALL,
    ROLE_UNKNOWN,
)


@dataclass(frozen=True)
class RoleResult:
    """The refined role of a single track identity.

    Attributes:
        track_id: Stable identity from the tracker.
        detected_class: The class the detector/tracker originally assigned.
        refined_role: One of :data:`ALLOWED_ROLES`.
        role_confidence: Confidence of the refinement decision in [0, 1].
        team_id: 0 or 1 for players; ``None`` for everything else.
        role_reason: Short machine-readable explanation of the decision.
    """

    track_id: int
    detected_class: str
    refined_role: str
    role_confidence: float
    team_id: Optional[int]
    role_reason: str

    def __post_init__(self) -> None:
        if self.refined_role not in ALLOWED_ROLES:
            raise ValueError(
                f"refined_role must be one of {ALLOWED_ROLES}, "
                f"got '{self.refined_role}'"
            )

    def to_dict(self) -> dict:
        """Serialize in the Phase 3 roles-JSON format."""
        return {
            "track_id": int(self.track_id),
            "detected_class": self.detected_class,
            "refined_role": self.refined_role,
            "role_confidence": round(float(self.role_confidence), 4),
            "team_id": (None if self.team_id is None else int(self.team_id)),
            "role_reason": self.role_reason,
        }


@dataclass
class TrackAppearance:
    """Per-track appearance + image-space trajectory summary.

    Attributes:
        track_id: Stable identity from the tracker.
        detected_class: Class name carried from tracking.
        vector: Stable, L1-normalized mean appearance vector for the track
            (concatenated jersey HSV histograms). Empty array if no usable
            crop could be sampled.
        sample_vectors: One appearance vector per successfully sampled
            frame, shape ``(n_samples, D)``. Used for temporal voting.
        n_samples: Number of frames that yielded a usable appearance vector.
        track_length: Number of frames the track was observed on.
        centers: Per-observation bbox centers in image space, ``(x, y)``.
        frame_size: ``(width, height)`` of the source video, for normalizing
            image-space position features.
    """

    track_id: int
    detected_class: str
    vector: np.ndarray
    sample_vectors: np.ndarray
    n_samples: int
    track_length: int
    centers: List[Tuple[float, float]] = field(default_factory=list)
    frame_size: Tuple[int, int] = (1, 1)

    # ------------------------------------------------------------------
    # Derived image-space trajectory features (no homography — pixels only)
    # ------------------------------------------------------------------
    @property
    def has_appearance(self) -> bool:
        return self.vector.size > 0 and self.n_samples > 0

    @property
    def mean_x_norm(self) -> float:
        """Mean horizontal position, normalized to [0, 1] (0=left, 1=right)."""
        if not self.centers:
            return 0.5
        w = max(self.frame_size[0], 1)
        return float(np.mean([c[0] for c in self.centers])) / w

    @property
    def mean_y_norm(self) -> float:
        """Mean vertical position, normalized to [0, 1] (0=top, 1=bottom)."""
        if not self.centers:
            return 0.5
        h = max(self.frame_size[1], 1)
        return float(np.mean([c[1] for c in self.centers])) / h

    @property
    def x_spread_norm(self) -> float:
        """Std-dev of horizontal position, normalized by frame width."""
        if len(self.centers) < 2:
            return 0.0
        w = max(self.frame_size[0], 1)
        return float(np.std([c[0] for c in self.centers])) / w

    @property
    def y_spread_norm(self) -> float:
        """Std-dev of vertical position, normalized by frame height."""
        if len(self.centers) < 2:
            return 0.0
        h = max(self.frame_size[1], 1)
        return float(np.std([c[1] for c in self.centers])) / h

    @property
    def coverage_norm(self) -> float:
        """Fraction of the frame area spanned by the trajectory bbox."""
        if len(self.centers) < 2:
            return 0.0
        xs = [c[0] for c in self.centers]
        ys = [c[1] for c in self.centers]
        w = max(self.frame_size[0], 1)
        h = max(self.frame_size[1], 1)
        return ((max(xs) - min(xs)) / w) * ((max(ys) - min(ys)) / h)

    @property
    def edge_proximity(self) -> float:
        """How close the mean horizontal position sits to a left/right edge.

        1.0 at either touchline (x≈0 or x≈1), 0.0 dead-center. Goalkeepers
        live near the horizontal extremes; central play sits near 0.
        """
        return 2.0 * abs(self.mean_x_norm - 0.5)
