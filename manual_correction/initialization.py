"""User-guided initialization for manual role correction.

Instead of correcting ambiguous tracks one by one, the reviewer first
labels the boxes they can clearly see in a few representative frames
(assigning each to Team 0 / Team 1 / Referee / Goalkeeper / Ignore).
Those labels become high-confidence *seeds* that are propagated to the
whole track, dramatically cutting the per-track work.

This module is pure (stdlib only) so all the logic — label storage,
multi-select assignment, representative-frame selection, and propagation —
is unit-testable without a display or a video.

On-disk format (``outputs/manual_initialization_labels.json``)::

    {
      "frames": {
        "1":   {"team_0": [4, 10, 17], "team_1": [1, 7, 24],
                "referee": [20], "goalkeeper": [115], "ignore": []},
        "250": {"team_0": [...], "team_1": [...], ...}
      }
    }
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from manual_correction.correction_models import Correction
from utils.logger import get_logger

logger = get_logger("manual_correction.initialization")

# The buckets a reviewer can drop a box into.
TEAM_0 = "team_0"
TEAM_1 = "team_1"
REFEREE = "referee"
GOALKEEPER = "goalkeeper"
IGNORE = "ignore"
GROUPS = [TEAM_0, TEAM_1, REFEREE, GOALKEEPER, IGNORE]

# How each group maps onto a (role, team_id) correction.
GROUP_TO_ROLE_TEAM: Dict[str, "tuple[str, Optional[int]]"] = {
    TEAM_0: ("player", 0),
    TEAM_1: ("player", 1),
    REFEREE: ("referee", None),
    GOALKEEPER: ("goalkeeper", None),
    IGNORE: ("ignore", None),
}

# Marker used when a track was labelled inconsistently across frames — a
# sign of an ID switch, which must NOT be propagated as a single team/role.
NEEDS_SPLIT_NOTE = "needs split / tracking issue"


def empty_frame_labels() -> Dict[str, List[int]]:
    return {group: [] for group in GROUPS}


def assign_group(
    frame_labels: Dict[str, List[int]],
    track_ids: Iterable[int],
    group: Optional[str],
) -> Dict[str, List[int]]:
    """Assign ``track_ids`` to ``group`` within one frame's labels.

    A track belongs to at most one group per frame, so the ids are first
    removed from every group, then added to the target. ``group=None``
    just clears the assignment (used to deselect). Mutates and returns
    ``frame_labels``.
    """
    for missing in GROUPS:
        frame_labels.setdefault(missing, [])
    ids = {int(t) for t in track_ids}
    for existing in GROUPS:
        frame_labels[existing] = [t for t in frame_labels[existing] if t not in ids]
    if group is not None:
        if group not in GROUPS:
            raise ValueError(f"unknown init group '{group}'")
        frame_labels[group] = sorted(set(frame_labels[group]) | ids)
    return frame_labels


@dataclass
class InitializationLabels:
    """Per-frame box-to-group assignments."""

    frames: Dict[int, Dict[str, List[int]]] = field(default_factory=dict)

    def ensure_frame(self, frame: int) -> Dict[str, List[int]]:
        return self.frames.setdefault(int(frame), empty_frame_labels())

    def assign(
        self, frame: int, track_ids: Iterable[int], group: Optional[str]
    ) -> None:
        assign_group(self.ensure_frame(frame), track_ids, group)

    def group_of(self, frame: int, track_id: int) -> Optional[str]:
        labels = self.frames.get(int(frame))
        if not labels:
            return None
        for group in GROUPS:
            if int(track_id) in labels.get(group, []):
                return group
        return None

    def frame_count(self) -> int:
        return len(self.frames)

    def to_dict(self) -> dict:
        return {
            "frames": {
                str(frame): {g: list(labels.get(g, [])) for g in GROUPS}
                for frame, labels in sorted(self.frames.items())
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InitializationLabels":
        frames: Dict[int, Dict[str, List[int]]] = {}
        for key, labels in (data.get("frames") or {}).items():
            try:
                frame = int(key)
            except (ValueError, TypeError):
                continue
            entry = empty_frame_labels()
            for group in GROUPS:
                entry[group] = [int(t) for t in (labels or {}).get(group, [])]
            frames[frame] = entry
        return cls(frames=frames)


class InitializationStore:
    """Reads/writes ``manual_initialization_labels.json``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> InitializationLabels:
        if not self.path.is_file():
            return InitializationLabels()
        with self.path.open("r", encoding="utf-8") as handle:
            return InitializationLabels.from_dict(json.load(handle))

    def save(self, labels: InitializationLabels) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(labels.to_dict(), handle, indent=2)
        logger.info("Saved initialization labels (%d frames) to %s",
                    labels.frame_count(), self.path)
        return self.path


def select_initialization_frames(
    frames_present: Iterable[int],
    gk_frames: Iterable[int] = (),
    custom: Iterable[int] = (),
) -> List[int]:
    """Choose representative frames to initialize on.

    Picks the first frame, an early frame (~10% in), a midfield/middle
    frame, the first goalkeeper-visible frame (if any), plus any custom
    frame numbers the user asked for.
    """
    present = sorted({int(f) for f in frames_present})
    picks = set(int(c) for c in custom)
    if present:
        n = len(present)
        picks.add(present[0])                       # first
        picks.add(present[min(n - 1, n // 10)])     # early (~10%)
        picks.add(present[n // 2])                  # midfield (middle)
    gk = sorted({int(f) for f in gk_frames})
    if gk:
        picks.add(gk[0])                            # goalkeeper-visible
    return sorted(picks)


def propagate_initialization(
    labels: InitializationLabels,
) -> Dict[int, Correction]:
    """Turn initialization labels into per-track correction seeds.

    * A track labelled with a single (non-ignore) group across all frames
      is locked to that team/role (``user_initialized=True``).
    * A track labelled ``ignore`` only becomes an ``ignore`` seed.
    * A track labelled with conflicting groups across frames is an ID switch
      (the box was a different player in some frame). Rather than discard the
      reviewer's work as ``unknown``, the **majority** label wins and the
      track is still flagged ``id_switch`` so the switch is recorded. Only a
      genuine tie (no majority) falls back to ``unknown``.
    """
    track_to_counts: Dict[int, Counter] = {}
    for labels_for_frame in labels.frames.values():
        for group in GROUPS:
            for track_id in labels_for_frame.get(group, []):
                track_to_counts.setdefault(int(track_id), Counter())[group] += 1

    seeds: Dict[int, Correction] = {}
    for track_id, counts in track_to_counts.items():
        non_ignore = Counter({g: c for g, c in counts.items() if g != IGNORE})
        if not non_ignore:
            seeds[track_id] = Correction(
                role="ignore", team_id=None, user_initialized=True
            )
            continue
        ranked = non_ignore.most_common()
        winner, win_count = ranked[0]
        conflict = len(non_ignore) > 1
        tie = conflict and ranked[1][1] == win_count
        if tie:
            # No majority -> genuinely ambiguous, leave it for a manual split.
            seeds[track_id] = Correction(
                role="unknown",
                team_id=None,
                id_switch=True,
                switch_note=NEEDS_SPLIT_NOTE,
                user_initialized=True,
            )
        else:
            # Majority label wins; note the switch when there was a conflict.
            role, team_id = GROUP_TO_ROLE_TEAM[winner]
            seeds[track_id] = Correction(
                role=role,
                team_id=team_id,
                id_switch=conflict,
                switch_note=NEEDS_SPLIT_NOTE if conflict else "",
                user_initialized=True,
            )
    return seeds


def merge_seeds(
    corrections: Dict[int, Correction],
    seeds: Dict[int, Correction],
    overwrite: bool = True,
) -> Dict[int, Correction]:
    """Merge initialization seeds into an existing corrections map."""
    merged = dict(corrections)
    for track_id, seed in seeds.items():
        if overwrite or track_id not in merged:
            merged[track_id] = seed
    return merged
