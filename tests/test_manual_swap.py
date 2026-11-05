"""Tests for reviewer-driven manual ID-switch swaps."""

from __future__ import annotations

from manual_correction.correction_models import Correction
from manual_correction.correction_store import CorrectionStore
from manual_correction.manual_swap import (
    apply_manual_swaps,
    build_manual_swaps,
    resolve_swap_source,
)
from tracking.models import FrameTracks, Track


# ---------------------------------------------------------------------------
# Model round-trip (switch_frame is backward compatible)
# ---------------------------------------------------------------------------
def test_correction_without_switch_keeps_minimal_shape():
    c = Correction(role="player", team_id=0)
    assert c.to_dict() == {"role": "player", "team_id": 0}
    assert Correction.from_dict(c.to_dict()) == c
    assert not c.is_manual_swap


def test_correction_switch_frame_roundtrips():
    c = Correction(role="player", team_id=1, id_switch=True,
                   merge_with_track_id=4, switch_frame=466)
    data = c.to_dict()
    assert data["switch_frame"] == 466
    assert Correction.from_dict(data) == c
    assert c.is_manual_swap


def test_old_corrections_file_without_switch_frame_loads():
    # An entry written before switch_frame existed must still load.
    legacy = {"role": "player", "team_id": 0, "id_switch": True,
              "switch_note": "swap", "merge_with_track_id": 4}
    c = Correction.from_dict(legacy)
    assert c.switch_frame is None
    assert not c.is_manual_swap            # incomplete -> not applied


# ---------------------------------------------------------------------------
# build_manual_swaps (pure)
# ---------------------------------------------------------------------------
def test_build_basic_swap():
    corrections = {
        1: Correction("player", 1, id_switch=True, merge_with_track_id=4,
                      switch_frame=5),
        4: Correction("player", 0),
    }
    assert build_manual_swaps(corrections, last_frame=10) == [(1, 4, 5, 10)]


def test_build_dedups_mirrored_pair():
    # Both sides flagged -> swap once, using the lower id's switch frame.
    corrections = {
        1: Correction("player", 1, id_switch=True, merge_with_track_id=4,
                      switch_frame=5),
        4: Correction("player", 0, id_switch=True, merge_with_track_id=1,
                      switch_frame=6),
    }
    swaps = build_manual_swaps(corrections, last_frame=10)
    assert swaps == [(1, 4, 5, 10)]


def test_build_skips_incomplete_and_self():
    corrections = {
        1: Correction("player", 1, id_switch=True, merge_with_track_id=4),  # no frame
        2: Correction("player", 1, id_switch=True, switch_frame=5),         # no partner
        3: Correction("player", 1, id_switch=True, merge_with_track_id=3,
                      switch_frame=5),                                       # self
        7: Correction("player", 1),                                         # plain
    }
    assert build_manual_swaps(corrections, last_frame=10) == []


def test_build_skips_frame_past_end():
    corrections = {
        1: Correction("player", 1, id_switch=True, merge_with_track_id=4,
                      switch_frame=999),
    }
    assert build_manual_swaps(corrections, last_frame=10) == []


# ---------------------------------------------------------------------------
# apply_manual_swaps (end-to-end on synthetic frames)
# ---------------------------------------------------------------------------
def _t(frame, tid, cx):
    return Track(frame, tid, 2, "player", 0.9, (cx - 10, 280, cx + 10, 330))


def _swapped_clip(n=10, switch=5):
    """Tracker output where ids 1 and 4 exchange physical players at ``switch``.

    Player P sits at cx=100, player Q at cx=900. Before ``switch`` id 1->P and
    id 4->Q; from ``switch`` on the ids are swapped (id 1->Q, id 4->P).
    """
    frames = []
    for f in range(1, n + 1):
        if f < switch:
            id1_cx, id4_cx = 100, 900
        else:
            id1_cx, id4_cx = 900, 100
        frames.append(FrameTracks(f, [_t(f, 1, id1_cx), _t(f, 4, id4_cx)]))
    return frames


def _cx(track):
    return (track.bbox[0] + track.bbox[2]) / 2


def test_apply_manual_swap_makes_ids_consistent():
    frames = _swapped_clip(n=10, switch=5)
    corrections = {
        1: Correction("player", 1, id_switch=True, merge_with_track_id=4,
                      switch_frame=5),
    }
    corrected, swaps = apply_manual_swaps(frames, corrections)
    assert swaps == [(1, 4, 5, 10)]
    by = {fr.frame_index: {t.track_id: _cx(t) for t in fr.tracks}
          for fr in corrected}
    # After the swap each id holds ONE physical player for the whole clip.
    assert all(by[f][1] == 100 for f in range(1, 11))   # id 1 -> player P
    assert all(by[f][4] == 900 for f in range(1, 11))   # id 4 -> player Q


def test_apply_is_self_inverse():
    # A swap is its own inverse -> applying twice returns the original.
    # This is why callers always source from the stable swap-fixed tracks.
    frames = _swapped_clip()
    corrections = {
        1: Correction("player", 1, id_switch=True, merge_with_track_id=4,
                      switch_frame=5),
    }
    once, _ = apply_manual_swaps(frames, corrections)
    twice, _ = apply_manual_swaps(once, corrections)
    orig = {fr.frame_index: {t.track_id: _cx(t) for t in fr.tracks}
            for fr in frames}
    back = {fr.frame_index: {t.track_id: _cx(t) for t in fr.tracks}
            for fr in twice}
    assert back == orig


def test_apply_no_swaps_returns_frames_unchanged():
    frames = _swapped_clip()
    corrected, swaps = apply_manual_swaps(frames, {1: Correction("player", 1)})
    assert swaps == []
    assert corrected == frames


# ---------------------------------------------------------------------------
# resolve_swap_source (never returns the manual-swap output)
# ---------------------------------------------------------------------------
def test_resolve_swap_source_precedence(tmp_path):
    stem = "clip"
    (tmp_path / f"{stem}_tracks.json").write_text("{}", encoding="utf-8")
    (tmp_path / f"{stem}_tracks_stitched.json").write_text("{}", encoding="utf-8")
    assert resolve_swap_source(tmp_path, stem).name == f"{stem}_tracks_stitched.json"
    (tmp_path / f"{stem}_tracks_swapfixed.json").write_text("{}", encoding="utf-8")
    assert resolve_swap_source(tmp_path, stem).name == f"{stem}_tracks_swapfixed.json"


def test_resolve_swap_source_ignores_manualswap(tmp_path):
    stem = "clip"
    (tmp_path / f"{stem}_tracks_swapfixed.json").write_text("{}", encoding="utf-8")
    # Even if a manualswap file exists, the source stays the stable base.
    (tmp_path / f"{stem}_tracks_manualswap.json").write_text("{}", encoding="utf-8")
    assert resolve_swap_source(tmp_path, stem).name == f"{stem}_tracks_swapfixed.json"


def test_resolve_swap_source_none_when_empty(tmp_path):
    assert resolve_swap_source(tmp_path, "missing") is None


# ---------------------------------------------------------------------------
# Store round-trip with switch metadata
# ---------------------------------------------------------------------------
def test_store_persists_switch_frame(tmp_path):
    path = tmp_path / "corr.json"
    store = CorrectionStore(path)
    store.save({1: Correction("player", 1, id_switch=True,
                              merge_with_track_id=4, switch_frame=466)})
    loaded = CorrectionStore(path).load()
    assert loaded[1].switch_frame == 466
    assert loaded[1].is_manual_swap
