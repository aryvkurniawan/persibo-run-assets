"""Post-tracking ID stitching (re-link broken track identities).

BoT-SORT sometimes drops a track during an occlusion / missed detection
and then assigns a *brand-new* ID to the same person when they reappear.
This module re-links those: if a new ID appears, shortly after an old ID
disappeared, at (roughly) the position the old one was heading to, it is
treated as the SAME identity and remapped back to the OLD id.

It is a pure post-process over the exported tracks — the BoT-SORT tracker
itself is never touched. Stitching runs before role refinement, so the
resulting longer, stable tracks also improve team clustering and reduce
"unknown" short tracks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tracking.models import FrameTracks, Track
from utils.logger import get_logger

logger = get_logger("tracking.track_stitcher")

Point = Tuple[float, float]

# How many trailing observations to use when estimating a track's velocity.
_VELOCITY_WINDOW = 5


@dataclass
class _Obs:
    frame_id: int
    center: Point
    size: Tuple[float, float]


@dataclass
class TrackSummary:
    """Compact lifetime summary of one track id."""

    track_id: int
    class_name: str
    first_frame: int
    last_frame: int
    first_center: Point
    last_center: Point
    velocity: Point
    avg_size: Tuple[float, float]


@dataclass
class _Chain:
    """A growing identity chain; its tail is the latest stitched segment."""

    root_id: int
    class_name: str
    tail_frame: int
    tail_center: Point
    velocity: Point
    size: Tuple[float, float]


@dataclass
class StitchResult:
    """Outcome of stitching."""

    remapping: Dict[int, int]            # original id -> kept (old) id
    frames: List[FrameTracks]            # tracks with ids remapped
    merges: int                          # how many ids were re-linked
    ids_before: int
    ids_after: int
    chains: Dict[int, List[int]] = field(default_factory=dict)  # root -> merged ids


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def _observations(frames: Sequence[FrameTracks]) -> Dict[int, List[_Obs]]:
    obs: Dict[int, List[_Obs]] = {}
    for frame in frames:
        for track in frame.tracks:
            x1, y1, x2, y2 = track.bbox
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            size = (x2 - x1, y2 - y1)
            obs.setdefault(track.track_id, []).append(
                _Obs(track.frame_id, center, size))
    for seq in obs.values():
        seq.sort(key=lambda o: o.frame_id)
    return obs


def _velocity(seq: List[_Obs]) -> Point:
    if len(seq) < 2:
        return (0.0, 0.0)
    tail = seq[-min(_VELOCITY_WINDOW, len(seq)):]
    first, last = tail[0], tail[-1]
    span = last.frame_id - first.frame_id
    if span <= 0:
        return (0.0, 0.0)
    return ((last.center[0] - first.center[0]) / span,
            (last.center[1] - first.center[1]) / span)


def summarize_tracks(frames: Sequence[FrameTracks]) -> Dict[int, TrackSummary]:
    """One :class:`TrackSummary` per track id."""
    class_of: Dict[int, str] = {}
    for frame in frames:
        for track in frame.tracks:
            class_of.setdefault(track.track_id, track.class_name)

    summaries: Dict[int, TrackSummary] = {}
    for track_id, seq in _observations(frames).items():
        sizes = [o.size for o in seq]
        avg_size = (
            sum(s[0] for s in sizes) / len(sizes),
            sum(s[1] for s in sizes) / len(sizes),
        )
        summaries[track_id] = TrackSummary(
            track_id=track_id,
            class_name=class_of.get(track_id, "unknown"),
            first_frame=seq[0].frame_id,
            last_frame=seq[-1].frame_id,
            first_center=seq[0].center,
            last_center=seq[-1].center,
            velocity=_velocity(seq),
            avg_size=avg_size,
        )
    return summaries


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------
def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _size_ok(a: Tuple[float, float], b: Tuple[float, float], ratio: float) -> bool:
    aw, ah = max(a[0], 1e-3), max(a[1], 1e-3)
    bw, bh = max(b[0], 1e-3), max(b[1], 1e-3)
    rw = max(aw / bw, bw / aw)
    rh = max(ah / bh, bh / ah)
    return rw <= ratio and rh <= ratio


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two appearance vectors (0 for a zero vector)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_stitch_map(
    frames: Sequence[FrameTracks],
    max_frame_gap: int = 30,
    distance_gate_px: float = 100.0,
    size_ratio_gate: float = 1.6,
    match_same_class: bool = True,
    ball_class_name: str = "ball",
    appearance_of: Optional[Dict[int, np.ndarray]] = None,
    appearance_min_sim: float = 0.5,
) -> Dict[int, int]:
    """Return ``{original_id -> kept_old_id}`` for re-linked tracks.

    A new track is merged into an older chain when it starts within
    ``max_frame_gap`` frames of the chain ending, its first position is
    within ``distance_gate_px`` of where the chain was predicted to be
    (last position + velocity * gap), the bbox sizes are compatible, and
    (optionally) the class matches. Balls are never stitched (their IDs are
    handled by the dedicated ball tracker).

    When ``appearance_of`` (``track_id -> mean jersey vector``) is given, a
    merge is additionally **gated on jersey colour**: a candidate whose colour
    is less than ``appearance_min_sim`` similar to the chain's is rejected, so
    two different-team players are never stitched into one id even if their
    paths line up. This makes wider position gates safe.
    """
    summaries = summarize_tracks(frames)
    # Process identities in birth order so older tracks anchor the chains.
    ordered = sorted(summaries.values(),
                     key=lambda s: (s.first_frame, s.track_id))

    chains: List[_Chain] = []
    remapping: Dict[int, int] = {}

    for summary in ordered:
        if summary.class_name == ball_class_name:
            remapping[summary.track_id] = summary.track_id
            continue

        best: Optional[_Chain] = None
        best_dist = float("inf")
        for chain in chains:
            if match_same_class and chain.class_name != summary.class_name:
                continue
            gap = summary.first_frame - chain.tail_frame
            if gap <= 0 or gap > max_frame_gap:
                continue
            predicted = (
                chain.tail_center[0] + chain.velocity[0] * gap,
                chain.tail_center[1] + chain.velocity[1] * gap,
            )
            dist = _distance(predicted, summary.first_center)
            if dist > distance_gate_px:
                continue
            if not _size_ok(chain.size, summary.avg_size, size_ratio_gate):
                continue
            if appearance_of is not None:
                va = appearance_of.get(chain.root_id)
                vb = appearance_of.get(summary.track_id)
                if va is not None and vb is not None \
                        and _cosine(va, vb) < appearance_min_sim:
                    continue                     # different jersey -> not the same id
            if dist < best_dist:
                best_dist, best = dist, chain

        if best is not None:
            remapping[summary.track_id] = best.root_id
            # Extend the chain's tail so further segments can re-link too.
            best.tail_frame = summary.last_frame
            best.tail_center = summary.last_center
            best.velocity = summary.velocity
            best.size = summary.avg_size
        else:
            remapping[summary.track_id] = summary.track_id
            chains.append(_Chain(
                root_id=summary.track_id,
                class_name=summary.class_name,
                tail_frame=summary.last_frame,
                tail_center=summary.last_center,
                velocity=summary.velocity,
                size=summary.avg_size,
            ))
    return remapping


def apply_remapping(
    frames: Sequence[FrameTracks], remapping: Dict[int, int]
) -> List[FrameTracks]:
    """Rewrite track ids using ``remapping`` (drops in-frame duplicates)."""
    out: List[FrameTracks] = []
    for frame in frames:
        seen: set = set()
        new_tracks: List[Track] = []
        for track in frame.tracks:
            new_id = remapping.get(track.track_id, track.track_id)
            if new_id in seen:
                # Two segments collided on one frame (shouldn't happen for
                # non-overlapping merges) — keep the first, skip the dup.
                continue
            seen.add(new_id)
            new_tracks.append(
                track if new_id == track.track_id
                else replace(track, track_id=new_id))
        out.append(FrameTracks(frame_index=frame.frame_index, tracks=new_tracks))
    return out


def drop_short_tracks(
    frames: Sequence[FrameTracks],
    min_length: int,
    ball_class_name: str = "ball",
) -> List[FrameTracks]:
    """Remove tracks observed on fewer than ``min_length`` frames (noise).

    The ball is never dropped (it legitimately fragments). ``min_length <= 1``
    is a no-op.
    """
    if min_length <= 1:
        return list(frames)
    counts: Dict[int, int] = {}
    is_ball: Dict[int, bool] = {}
    for frame in frames:
        for track in frame.tracks:
            counts[track.track_id] = counts.get(track.track_id, 0) + 1
            is_ball[track.track_id] = track.class_name == ball_class_name
    keep = {
        tid for tid, c in counts.items()
        if c >= min_length or is_ball.get(tid, False)
    }
    return [
        FrameTracks(frame_index=f.frame_index,
                    tracks=[t for t in f.tracks if t.track_id in keep])
        for f in frames
    ]


def stitch_tracks(
    frames: Sequence[FrameTracks],
    max_frame_gap: int = 30,
    distance_gate_px: float = 100.0,
    size_ratio_gate: float = 1.6,
    match_same_class: bool = True,
    ball_class_name: str = "ball",
    min_segment_length: int = 0,
    appearance_of: Optional[Dict[int, np.ndarray]] = None,
    appearance_min_sim: float = 0.5,
) -> StitchResult:
    """Stitch broken identities and return the remapped tracks + stats."""
    remapping = compute_stitch_map(
        frames, max_frame_gap, distance_gate_px, size_ratio_gate,
        match_same_class, ball_class_name,
        appearance_of=appearance_of, appearance_min_sim=appearance_min_sim)
    stitched = apply_remapping(frames, remapping)
    if min_segment_length > 1:
        stitched = drop_short_tracks(stitched, min_segment_length, ball_class_name)

    merges = sum(1 for k, v in remapping.items() if k != v)
    chains: Dict[int, List[int]] = {}
    for original, root in remapping.items():
        if original != root:
            chains.setdefault(root, []).append(original)
    surviving = {t.track_id for f in stitched for t in f.tracks}
    result = StitchResult(
        remapping=remapping,
        frames=stitched,
        merges=merges,
        ids_before=len(remapping),
        ids_after=len(surviving),
        chains=chains,
    )
    logger.info(
        "Track stitching: %d ids -> %d ids (%d re-linked, %d short dropped)",
        result.ids_before, result.ids_after, result.merges,
        len(set(remapping.values())) - len(surviving))
    return result
