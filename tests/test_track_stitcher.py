"""Tests for post-tracking ID stitching (re-linking broken identities)."""

from __future__ import annotations

from tracking.models import FrameTracks, Track
from tracking.track_stitcher import (
    apply_remapping,
    compute_stitch_map,
    stitch_tracks,
    summarize_tracks,
)


def _track(frame, tid, cx, cy, name="player", w=20, h=50):
    return Track(
        frame_id=frame, track_id=tid, class_id=2, class_name=name,
        confidence=0.9, bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def _moving_track(frames, tid, x0, y, vx, name="player"):
    """A track present on `frames` moving at vx px/frame."""
    out = {}
    for i, f in enumerate(frames):
        out.setdefault(f, []).append(_track(f, tid, x0 + vx * i, y, name))
    return out


def _build(frames_dict):
    """frames_dict: {frame_index: [Track, ...]} -> list[FrameTracks]."""
    return [FrameTracks(frame_index=f, tracks=frames_dict.get(f, []))
            for f in sorted(frames_dict)]


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def test_summarize_tracks_velocity_and_extents():
    frames = {}
    for f in range(1, 11):                  # track 1 moves right at 10 px/frame
        frames.setdefault(f, []).append(_track(f, 1, 100 + 10 * f, 300))
    summaries = summarize_tracks(_build(frames))
    s = summaries[1]
    assert s.first_frame == 1 and s.last_frame == 10
    assert s.velocity[0] > 9 and abs(s.velocity[1]) < 1e-6


# ---------------------------------------------------------------------------
# Stitching: the core re-link case
# ---------------------------------------------------------------------------
def test_new_id_at_old_position_is_relinked():
    # Track 1 moves right and disappears at frame 10 (x=300).
    # Track 7 appears at frame 14 near where 1 was heading (~x=340) -> merge.
    frames = {}
    for f in range(1, 11):
        frames.setdefault(f, []).append(_track(f, 1, 200 + 10 * f, 300))  # ends x=300
    for f in range(14, 24):
        frames.setdefault(f, []).append(_track(f, 7, 340 + 10 * (f - 14), 300))

    remap = compute_stitch_map(_build(frames), max_frame_gap=30,
                               distance_gate_px=80)
    assert remap[7] == 1            # new id 7 kept as old id 1
    assert remap[1] == 1


def test_appearance_gate_blocks_different_colour_merge():
    # Geometry says "merge" (track 7 starts where 1 was heading), but their
    # jersey colours differ -> the appearance gate must reject the merge.
    import numpy as np
    frames = {}
    for f in range(1, 11):
        frames.setdefault(f, []).append(_track(f, 1, 200 + 10 * f, 300))
    for f in range(14, 24):
        frames.setdefault(f, []).append(_track(f, 7, 340 + 10 * (f - 14), 300))
    appearance = {1: np.array([1.0, 0.0, 0.0]), 7: np.array([0.0, 1.0, 0.0])}
    remap = compute_stitch_map(
        _build(frames), max_frame_gap=30, distance_gate_px=80,
        appearance_of=appearance, appearance_min_sim=0.5)
    assert remap[7] == 7            # different colour -> NOT merged


def test_appearance_gate_allows_same_colour_merge():
    import numpy as np
    frames = {}
    for f in range(1, 11):
        frames.setdefault(f, []).append(_track(f, 1, 200 + 10 * f, 300))
    for f in range(14, 24):
        frames.setdefault(f, []).append(_track(f, 7, 340 + 10 * (f - 14), 300))
    appearance = {1: np.array([1.0, 0.1, 0.0]), 7: np.array([0.97, 0.12, 0.0])}
    remap = compute_stitch_map(
        _build(frames), max_frame_gap=30, distance_gate_px=80,
        appearance_of=appearance, appearance_min_sim=0.5)
    assert remap[7] == 1            # same colour + geometry -> merged


def test_no_relink_when_position_far():
    frames = {}
    for f in range(1, 11):
        frames.setdefault(f, []).append(_track(f, 1, 200, 300))     # stays left
    for f in range(14, 24):
        frames.setdefault(f, []).append(_track(f, 7, 1500, 300))    # far right
    remap = compute_stitch_map(_build(frames), distance_gate_px=100)
    assert remap[7] == 7            # not merged


def test_no_relink_when_gap_too_large():
    frames = {}
    for f in range(1, 11):
        frames.setdefault(f, []).append(_track(f, 1, 300, 300))
    for f in range(100, 110):       # reappears far in time
        frames.setdefault(f, []).append(_track(f, 7, 300, 300))
    remap = compute_stitch_map(_build(frames), max_frame_gap=30)
    assert remap[7] == 7


def test_no_relink_across_classes():
    frames = {}
    for f in range(1, 11):
        frames.setdefault(f, []).append(_track(f, 1, 300, 300, name="referee"))
    for f in range(13, 23):
        frames.setdefault(f, []).append(_track(f, 7, 305, 300, name="player"))
    remap = compute_stitch_map(_build(frames), match_same_class=True)
    assert remap[7] == 7            # referee != player -> no merge


def test_referee_relinks_to_referee():
    frames = {}
    for f in range(1, 11):
        frames.setdefault(f, []).append(_track(f, 3, 300, 300, name="referee"))
    for f in range(13, 23):
        frames.setdefault(f, []).append(_track(f, 9, 305, 300, name="referee"))
    remap = compute_stitch_map(_build(frames))
    assert remap[9] == 3            # referee re-linked


def test_no_relink_when_size_mismatch():
    frames = {}
    for f in range(1, 11):
        frames.setdefault(f, []).append(_track(f, 1, 300, 300, w=20, h=50))
    for f in range(13, 23):
        # Much bigger box at the same place -> different person.
        frames.setdefault(f, []).append(_track(f, 7, 305, 300, w=120, h=300))
    remap = compute_stitch_map(_build(frames), size_ratio_gate=1.6)
    assert remap[7] == 7


def test_ball_is_never_stitched():
    frames = {}
    for f in range(1, 11):
        frames.setdefault(f, []).append(_track(f, 9001, 300, 300, name="ball"))
    for f in range(13, 23):
        frames.setdefault(f, []).append(_track(f, 9002, 305, 300, name="ball"))
    remap = compute_stitch_map(_build(frames), ball_class_name="ball")
    assert remap[9001] == 9001 and remap[9002] == 9002


# ---------------------------------------------------------------------------
# Chaining + apply
# ---------------------------------------------------------------------------
def test_three_segments_chain_to_one_id():
    # 1 (f1-10) -> 7 (f13-22) -> 12 (f25-34), all on a rightward path.
    frames = {}
    for f in range(1, 11):
        frames.setdefault(f, []).append(_track(f, 1, 200 + 10 * f, 300))
    for f in range(13, 23):
        frames.setdefault(f, []).append(_track(f, 7, 300 + 10 * (f - 12), 300))
    for f in range(25, 35):
        frames.setdefault(f, []).append(_track(f, 12, 420 + 10 * (f - 24), 300))
    remap = compute_stitch_map(_build(frames), max_frame_gap=30,
                               distance_gate_px=120)
    assert remap[7] == 1 and remap[12] == 1


def test_drop_short_tracks_removes_noise_keeps_ball():
    from tracking.track_stitcher import drop_short_tracks

    frames = {}
    for f in range(1, 21):                       # long player id 1 (20 frames)
        frames.setdefault(f, []).append(_track(f, 1, 300, 300))
    for f in range(1, 4):                        # short noise id 2 (3 frames)
        frames.setdefault(f, []).append(_track(f, 2, 800, 400))
    for f in range(1, 4):                        # short BALL id 9001 (kept)
        frames.setdefault(f, []).append(_track(f, 9001, 500, 500, name="ball"))
    out = drop_short_tracks(_build(frames), min_length=8, ball_class_name="ball")
    ids = {t.track_id for fr in out for t in fr.tracks}
    assert 1 in ids            # long track kept
    assert 2 not in ids        # short noise dropped
    assert 9001 in ids         # short ball kept


def test_stitch_tracks_drops_short_segments():
    frames = {}
    for f in range(1, 21):
        frames.setdefault(f, []).append(_track(f, 1, 200 + 5 * f, 300))
    for f in range(1, 4):                        # isolated 3-frame noise
        frames.setdefault(f, []).append(_track(f, 9, 1500, 800))
    result = stitch_tracks(_build(frames), min_segment_length=8)
    ids = {t.track_id for fr in result.frames for t in fr.tracks}
    assert ids == {1}          # noise id 9 dropped


def test_apply_remapping_rewrites_ids():
    frames = _build({
        1: [_track(1, 1, 100, 300)],
        5: [_track(5, 7, 150, 300)],
    })
    out = apply_remapping(frames, {1: 1, 7: 1})
    assert out[0].tracks[0].track_id == 1
    assert out[1].tracks[0].track_id == 1     # 7 rewritten to 1


def test_run_stitching_writes_stitched_json_and_repoints(tmp_path):
    import json

    import main
    from utils.config_loader import AppConfig

    # A tracks JSON with a stitchable pair (id 1 ends, id 7 reappears near it).
    payload = {"metadata": {"resolution": [1920, 1080], "video_fps": 25.0},
               "frames": []}
    for f in range(1, 11):
        payload["frames"].append({"frame": f, "tracks": [
            {"track_id": 1, "class": "player", "confidence": 0.9,
             "bbox": [200 + 10 * f, 280, 220 + 10 * f, 330]}]})
    for f in range(14, 24):
        payload["frames"].append({"frame": f, "tracks": [
            {"track_id": 7, "class": "player", "confidence": 0.9,
             "bbox": [340 + 10 * (f - 14), 280, 360 + 10 * (f - 14), 330]}]})
    tracks_json = tmp_path / "input_video_tracks.json"
    tracks_json.write_text(json.dumps(payload), encoding="utf-8")

    config = AppConfig()
    config.video.output_dir = str(tmp_path)
    config.tracking.stitching.enabled = True
    config.tracking.stitching.distance_gate_px = 80.0
    summary = {"tracks_json": str(tracks_json)}

    stats = main.run_stitching(config, "input_video.mp4", summary)
    assert stats["merges"] == 1
    # Downstream pointer now references the stitched file, which exists.
    assert summary["tracks_json"].endswith("_tracks_stitched.json")
    out = json.loads((tmp_path / "input_video_tracks_stitched.json")
                     .read_text(encoding="utf-8"))
    assert out["metadata"]["stitched"] is True
    ids = {t["track_id"] for fr in out["frames"] for t in fr["tracks"]}
    assert ids == {1}        # id 7 re-linked to 1


def test_stitch_tracks_stats_and_result():
    frames = {}
    for f in range(1, 11):
        frames.setdefault(f, []).append(_track(f, 1, 200 + 10 * f, 300))
    for f in range(14, 24):
        frames.setdefault(f, []).append(_track(f, 7, 340 + 10 * (f - 14), 300))
    result = stitch_tracks(_build(frames), distance_gate_px=80)
    assert result.ids_before == 2
    assert result.ids_after == 1
    assert result.merges == 1
    assert result.chains == {1: [7]}
    # Every frame's track now carries id 1.
    all_ids = {t.track_id for fr in result.frames for t in fr.tracks}
    assert all_ids == {1}
