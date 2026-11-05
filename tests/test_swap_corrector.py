"""Tests for appearance-based ID-swap correction (pure logic)."""

from __future__ import annotations

from tracking.models import FrameTracks, Track
from tracking.swap_corrector import (
    SwapInterval,
    apply_splits,
    apply_swaps,
    find_swap_intervals,
    match_swaps,
    plan_splits,
    unpaired_intervals,
)


# ---------------------------------------------------------------------------
# Swap-interval detection (handles temporary cross-then-cross-back swaps)
# ---------------------------------------------------------------------------
def test_find_intervals_temporary_swap():
    # Dominant team 0, but a burst of team 1 in the middle (a temporary swap).
    timeline = (
        [(f, 0) for f in range(0, 60, 10)]
        + [(f, 1) for f in range(60, 120, 10)]
        + [(f, 0) for f in range(120, 200, 10)]
    )
    dominant, intervals = find_swap_intervals(timeline, min_run=2)
    assert dominant == 0
    assert len(intervals) == 1
    assert intervals[0].own_team == 0 and intervals[0].other_team == 1
    assert intervals[0].start == 60 and intervals[0].end == 110


def test_find_intervals_none_for_consistent_track():
    timeline = [(f, 1) for f in range(0, 120, 10)]
    dominant, intervals = find_swap_intervals(timeline)
    assert dominant == 1 and intervals == []


def test_find_intervals_ignores_single_sample_blip():
    # One stray label (run length 1) is below min_run -> not an interval.
    timeline = [(f, 0) for f in range(0, 100, 10)]
    timeline[5] = (50, 1)
    _dom, intervals = find_swap_intervals(timeline, min_run=2)
    assert intervals == []


def test_find_intervals_ignores_none_labels():
    timeline = [(0, 0), (10, None), (20, 0), (30, 1), (40, 1), (50, None), (60, 0)]
    dom, intervals = find_swap_intervals(timeline, min_run=2)
    assert dom == 0 and len(intervals) == 1
    assert intervals[0].start == 30 and intervals[0].end == 40


# ---------------------------------------------------------------------------
# Matching opposite intervals
# ---------------------------------------------------------------------------
def test_match_pairs_opposite_overlapping_intervals():
    a = SwapInterval(track_id=22, start=60, end=110, own_team=0, other_team=1)
    b = SwapInterval(track_id=12, start=58, end=108, own_team=1, other_team=0)
    centers = {22: (500, 300), 12: (510, 305)}
    swaps = match_swaps([a, b], lambda t, f: centers[t], distance_gate_px=200)
    assert len(swaps) == 1
    ida, idb, start, end = swaps[0]
    assert {ida, idb} == {22, 12}                # the pair (order by interval start)
    assert (start, end) == (58, 110)             # union window


def test_match_rejects_non_overlapping_in_time():
    a = SwapInterval(22, 60, 110, 0, 1)
    b = SwapInterval(12, 400, 450, 1, 0)
    centers = {22: (500, 300), 12: (510, 305)}
    assert match_swaps([a, b], lambda t, f: centers[t], frame_tol=25) == []


def test_match_rejects_far_apart():
    a = SwapInterval(22, 60, 110, 0, 1)
    b = SwapInterval(12, 58, 108, 1, 0)
    centers = {22: (100, 300), 12: (1500, 300)}
    assert match_swaps([a, b], lambda t, f: centers[t], distance_gate_px=200) == []


def test_match_rejects_same_direction():
    a = SwapInterval(22, 60, 110, 0, 1)
    b = SwapInterval(12, 58, 108, 0, 1)          # same dominant/other, not a mirror
    centers = {22: (500, 300), 12: (510, 305)}
    assert match_swaps([a, b], lambda t, f: centers[t]) == []


# ---------------------------------------------------------------------------
# Applying interval swaps (and reverting outside the window)
# ---------------------------------------------------------------------------
def _t(frame, tid, cx=500):
    return Track(frame, tid, 2, "player", 0.9, (cx - 10, 280, cx + 10, 330))


def test_apply_swaps_only_within_window():
    frames = [FrameTracks(f, [_t(f, 22), _t(f, 12)]) for f in (50, 80, 100, 150)]
    out = apply_swaps(frames, [(22, 12, 60, 110)])
    by = {fr.frame_index: [t.track_id for t in fr.tracks] for fr in out}
    assert by[50] == [22, 12]            # before window: unchanged
    assert sorted(by[80]) == [12, 22]    # inside window: swapped (both present)
    assert sorted(by[100]) == [12, 22]   # inside window
    assert by[150] == [22, 12]           # after window: reverted


def test_apply_swaps_empty_is_identity():
    frames = [FrameTracks(1, [_t(1, 5)]), FrameTracks(2, [_t(2, 5)])]
    out = apply_swaps(frames, [])
    assert [t.track_id for fr in out for t in fr.tracks] == [5, 5]


# ---------------------------------------------------------------------------
# Drift splitting (a flip interval with no crossing partner)
# ---------------------------------------------------------------------------
def _iv(tid, start, end, own, other, n):
    return SwapInterval(tid, start, end, own, other, n_samples=n)


def test_unpaired_intervals_excludes_paired_and_short():
    paired = _iv(22, 60, 110, 0, 1, n=5)
    drift = _iv(21, 70, 95, 1, 0, n=4)          # no partner -> unpaired
    blip = _iv(7, 30, 32, 0, 1, n=2)            # too few samples -> dropped
    swaps = [(22, 12, 60, 110)]                 # consumes track 22's interval
    out = unpaired_intervals([paired, drift, blip], swaps, min_samples=3)
    assert [iv.track_id for iv in out] == [21]


def test_plan_splits_assigns_unique_ids():
    ivs = [_iv(21, 70, 95, 1, 0, n=4), _iv(30, 200, 240, 0, 1, n=4)]
    splits = plan_splits(ivs, start_id=7000)
    assert splits == [(21, 70, 95, 7000), (30, 200, 240, 7001)]


def test_apply_splits_reassigns_only_inside_window():
    # Track 21 is one id throughout; the drift window [70,95] becomes a new id.
    frames = [FrameTracks(f, [_t(f, 21)]) for f in (50, 80, 100)]
    out = apply_splits(frames, [(21, 70, 95, 7000)])
    by = {fr.frame_index: [t.track_id for t in fr.tracks] for fr in out}
    assert by[50] == [21]        # before the drift: original id
    assert by[80] == [7000]      # inside the drift: split to a new id
    assert by[100] == [21]       # after the drift: original id again
