"""Manual ID-swap surgery driven by reviewer corrections.

The automatic :class:`tracking.swap_corrector.SwapCorrector` fixes crossing
ID-swaps it can detect from jersey appearance. Some swaps slip through (the
crossing is too far apart, the colours are ambiguous, ...). When a reviewer
spots one in the correction UI they record it on the track:

* ``id_switch`` = True
* ``merge_with_track_id`` = the *other* track it swapped identities with
* ``switch_frame`` = the frame the swap takes effect from

This module turns those records into concrete swaps and applies them to the
tracks by **reusing the same, already-tested** :func:`apply_swaps` the
automatic corrector uses. From ``switch_frame`` to the last frame the two
ids are exchanged, so each real player keeps one consistent id for the whole
clip.

Idempotency: callers should always apply manual swaps to the stable
(auto) swap-corrected tracks, never to this module's own output -- a swap is
its own inverse, so applying it twice cancels out. :func:`resolve_swap_source`
encodes that precedence (it never returns the ``_manualswap`` file).

The pure pieces (:func:`build_manual_swaps`) take plain data so they're
unit-testable without any video or files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from manual_correction.correction_models import Correction
from tracking.models import FrameTracks
from tracking.swap_corrector import Swap, apply_swaps
from utils.logger import get_logger

logger = get_logger("manual_correction.manual_swap")


def build_manual_swaps(
    corrections: Dict[int, Correction], last_frame: int
) -> List[Swap]:
    """Derive ``(id_a, id_b, start, end)`` swaps from reviewer corrections.

    One swap is emitted per *pair* of tracks. If both sides of a pair are
    flagged (track A says "continue as B" and B says "continue as A"), the
    pair is swapped once, using the lower id's ``switch_frame``. Entries
    missing a partner or a frame, or pointing at themselves, are ignored.
    """
    swaps: List[Swap] = []
    seen: set = set()
    for track_id in sorted(corrections):
        correction = corrections[track_id]
        if not correction.is_manual_swap:
            continue
        partner = int(correction.merge_with_track_id)  # type: ignore[arg-type]
        if partner == track_id:
            logger.warning(
                "Track %d marked to continue as itself; ignoring swap",
                track_id)
            continue
        pair = (min(track_id, partner), max(track_id, partner))
        if pair in seen:
            continue
        seen.add(pair)
        start = max(int(correction.switch_frame), 0)  # type: ignore[arg-type]
        if start > last_frame:
            logger.warning(
                "Swap %d<->%d switch_frame %d is past the last frame %d; "
                "ignoring", track_id, partner, start, last_frame)
            continue
        swaps.append((track_id, partner, start, int(last_frame)))
    return swaps


def apply_manual_swaps(
    frames: Sequence[FrameTracks], corrections: Dict[int, Correction]
) -> Tuple[List[FrameTracks], List[Swap]]:
    """Apply every reviewer-marked manual swap to ``frames``.

    Returns the corrected frames and the list of swaps applied (empty when
    there was nothing to do, in which case the frames are returned as-is).
    """
    if not frames:
        return list(frames), []
    last_frame = max(f.frame_index for f in frames)
    swaps = build_manual_swaps(corrections, last_frame)
    if not swaps:
        return list(frames), []
    corrected = apply_swaps(frames, swaps)
    logger.info("Applied %d manual ID-swap(s): %s", len(swaps),
                ", ".join(f"#{a}<->#{b}@{s}-{e}" for a, b, s, e in swaps))
    return corrected, swaps


# Source files in the same precedence the pipeline produces them, but
# EXCLUDING the manual-swap output -- manual swaps are always re-derived
# from the stable auto-corrected base so re-running is idempotent.
_SWAP_SOURCE_NAMES = (
    "{stem}_tracks_swapfixed.json",
    "{stem}_tracks_stitched.json",
    "{stem}_tracks.json",
)


def resolve_swap_source(output_dir: str | Path, stem: str) -> Path | None:
    """Most-processed tracks file to apply manual swaps to (never manualswap)."""
    out = Path(output_dir)
    for name in _SWAP_SOURCE_NAMES:
        candidate = out / name.format(stem=stem)
        if candidate.is_file():
            return candidate
    return None
