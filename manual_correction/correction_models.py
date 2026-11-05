"""Data structures for Phase 3.5 — human-in-the-loop role correction.

Three contracts:

* :class:`ReviewCandidate` — an ambiguous track served to the reviewer
  (the UI's input). Carries everything needed to make a call without
  scrubbing the video: the automatic verdict plus representative crops.
* :class:`Correction` — one reviewer decision (the UI's output / the
  corrections-file entries).
* :class:`FinalRole` — the reconciled per-track result after overlaying
  corrections on the automatic roles (the final export).

Both the automatic and the final role are preserved on :class:`FinalRole`
so a correction is always reversible and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from role_refinement.role_models import ALLOWED_ROLES

# The reviewer may also choose to leave a track untouched.
IGNORE = "ignore"
# Legal values for a stored correction's ``role`` field.
CORRECTION_ROLES = tuple(ALLOWED_ROLES) + (IGNORE,)


@dataclass(frozen=True)
class ReviewCandidate:
    """An ambiguous track presented for manual review.

    Attributes:
        track_id: Stable identity from the tracker.
        detected_class: Original detector/tracker class label.
        current_role: The automatic refined role (Phase 3).
        role_confidence: Confidence of the automatic decision.
        team_id: Automatic team assignment (``None`` for non-players).
        track_length: Number of frames the track was observed on.
        frame_ids: Representative frame numbers (start / middle / end ...).
        crop_paths: Saved jersey-crop image paths, aligned with ``frame_ids``.
        frame_paths: Saved full-frame image paths, aligned with ``frame_ids``
            (entry is "" when that frame's full image wasn't saved).
        bboxes: Pixel ``[x1, y1, x2, y2]`` of the track on each sampled
            frame, aligned with ``frame_ids`` — lets the UI draw the box on
            the full-frame context so ID switches are easy to spot.
        role_reason: The automatic ``role_reason`` (for context).
    """

    track_id: int
    detected_class: str
    current_role: str
    role_confidence: float
    team_id: Optional[int]
    track_length: int
    frame_ids: List[int] = field(default_factory=list)
    crop_paths: List[str] = field(default_factory=list)
    frame_paths: List[str] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list)
    role_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "track_id": int(self.track_id),
            "detected_class": self.detected_class,
            "current_role": self.current_role,
            "role_confidence": round(float(self.role_confidence), 4),
            "team_id": (None if self.team_id is None else int(self.team_id)),
            "track_length": int(self.track_length),
            "frame_ids": [int(f) for f in self.frame_ids],
            "crop_paths": list(self.crop_paths),
            "frame_paths": list(self.frame_paths),
            "bboxes": [[float(v) for v in b] for b in self.bboxes],
            "role_reason": self.role_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewCandidate":
        return cls(
            track_id=int(data["track_id"]),
            detected_class=str(data.get("detected_class", "unknown")),
            current_role=str(data.get("current_role", "unknown")),
            role_confidence=float(data.get("role_confidence", 0.0)),
            team_id=_opt_int(data.get("team_id")),
            track_length=int(data.get("track_length", 0)),
            frame_ids=[int(f) for f in data.get("frame_ids", [])],
            crop_paths=list(data.get("crop_paths", [])),
            frame_paths=list(data.get("frame_paths", [])),
            bboxes=[list(b) for b in data.get("bboxes", [])],
            role_reason=str(data.get("role_reason", "")),
        )


@dataclass(frozen=True)
class Correction:
    """A single reviewer decision.

    Attributes:
        role: One of :data:`CORRECTION_ROLES`. ``"ignore"`` means "leave the
            automatic role as-is" (recorded but applied as a no-op).
        team_id: Overriding team id (0/1 for players, ``None`` otherwise).
        id_switch: True if the reviewer observed an identity switch on this
            track (two real people share the id).
        switch_note: Free-text note describing the switch.
        merge_with_track_id: Optional id of the track this one should
            continue as. With ``id_switch`` and ``switch_frame`` set, this
            drives a manual ID-swap: from ``switch_frame`` to the end, this
            track and ``merge_with_track_id`` exchange ids (see
            :mod:`manual_correction.manual_swap`).
        switch_frame: Frame number at which the identity switch happens
            (1-based, as shown in the review UI). Used together with
            ``id_switch`` + ``merge_with_track_id`` to apply the swap.

    Serialization is backward compatible: a correction with only role +
    team_id round-trips to the original ``{"role": ..., "team_id": ...}``
    shape, and the ID-switch fields are written only when actually used —
    so old corrections files load unchanged and old readers (the apply
    layer) keep working.
    """

    role: str
    team_id: Optional[int] = None
    id_switch: bool = False
    switch_note: str = ""
    merge_with_track_id: Optional[int] = None
    switch_frame: Optional[int] = None     # frame the swap takes effect from
    user_initialized: bool = False
    player_name: Optional[str] = None      # reviewer-typed name (manual)

    def __post_init__(self) -> None:
        if self.role not in CORRECTION_ROLES:
            raise ValueError(
                f"correction role must be one of {CORRECTION_ROLES}, "
                f"got '{self.role}'"
            )

    @property
    def is_noop(self) -> bool:
        """An ``ignore`` correction does not change the automatic role."""
        return self.role == IGNORE

    @property
    def has_switch_info(self) -> bool:
        return bool(
            self.id_switch
            or self.switch_note
            or self.merge_with_track_id is not None
            or self.switch_frame is not None
        )

    @property
    def is_manual_swap(self) -> bool:
        """True when this correction fully specifies a manual ID-swap."""
        return bool(
            self.id_switch
            and self.merge_with_track_id is not None
            and self.switch_frame is not None
        )

    def to_dict(self) -> dict:
        data = {"role": self.role, "team_id": self.team_id}
        # Only emit the optional fields when there's something to record,
        # so simple corrections keep the original two-key shape.
        if self.has_switch_info:
            data["id_switch"] = bool(self.id_switch)
            data["switch_note"] = self.switch_note
            data["merge_with_track_id"] = self.merge_with_track_id
            data["switch_frame"] = self.switch_frame
        if self.user_initialized:
            data["user_initialized"] = True
        if self.player_name:
            data["player_name"] = self.player_name
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Correction":
        name = data.get("player_name")
        return cls(
            role=str(data["role"]),
            team_id=_opt_int(data.get("team_id")),
            id_switch=bool(data.get("id_switch", False)),
            switch_note=str(data.get("switch_note", "")),
            merge_with_track_id=_opt_int(data.get("merge_with_track_id")),
            switch_frame=_opt_int(data.get("switch_frame")),
            user_initialized=bool(data.get("user_initialized", False)),
            player_name=(str(name) if name else None),
        )


@dataclass(frozen=True)
class FinalRole:
    """The reconciled role for one track after applying corrections."""

    track_id: int
    detected_class: str
    auto_role: str
    final_role: str
    role_confidence: float
    team_id: Optional[int]
    auto_team_id: Optional[int]
    corrected_by_user: bool
    role_reason: str = ""
    user_initialized: bool = False
    player_name: Optional[str] = None      # reviewer-typed name (manual)

    @property
    def player_display_name(self) -> str:
        return self.player_name or f"Player {self.track_id}"

    def to_dict(self) -> dict:
        data = {
            "track_id": int(self.track_id),
            "detected_class": self.detected_class,
            "auto_role": self.auto_role,
            "final_role": self.final_role,
            "role_confidence": round(float(self.role_confidence), 4),
            "team_id": (None if self.team_id is None else int(self.team_id)),
            "auto_team_id": (
                None if self.auto_team_id is None else int(self.auto_team_id)
            ),
            "corrected_by_user": bool(self.corrected_by_user),
            "user_initialized": bool(self.user_initialized),
            "role_reason": self.role_reason,
        }
        if self.final_role != "ball":
            data["player_display_name"] = self.player_display_name
        if self.player_name:
            data["player_name"] = self.player_name
        return data


def _opt_int(value) -> Optional[int]:
    if value is None:
        return None
    return int(value)
