"""Phase 3 — track-level role refinement.

Runs after tracking. Corrects each track's role adaptively per match by
discovering the two main team appearance clusters from the tracked
players, flagging appearance outliers, and separating referee vs
goalkeeper using image-space position and motion plus temporal voting.
No fixed color rules and no homography.

Public surface:
    RoleResult, TrackAppearance      — role-refinement data models
    AppearanceExtractor              — per-track jersey HSV appearance vectors
    TeamClusterer, ClusterResult     — discover teams, flag outliers
    RoleRefiner                      — pure role-decision logic
    RoleRefinementRunner             — end-to-end tracks JSON -> roles JSON
    RoleJSONExporter                 — roles JSON export
    RoleVisualizer                   — annotated role video
"""

from role_refinement.role_models import (
    ALLOWED_ROLES,
    ROLE_BALL,
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
    ROLE_REFEREE,
    ROLE_UNKNOWN,
    RoleResult,
    TrackAppearance,
)

__all__ = [
    "RoleResult",
    "TrackAppearance",
    "ALLOWED_ROLES",
    "ROLE_PLAYER",
    "ROLE_GOALKEEPER",
    "ROLE_REFEREE",
    "ROLE_BALL",
    "ROLE_UNKNOWN",
]
