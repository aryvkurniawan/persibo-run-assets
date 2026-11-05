"""Unit tests for Phase 3 — track-level role refinement.

Covered:
    * appearance extraction (jersey HSV vectors, grass fallback, sampling)
    * team clustering + outlier detection
    * role voting (player via temporal vote)
    * unknown role fallback (short / ambiguous tracks)
    * preserving the ball role
    * referee vs goalkeeper separation from image-space motion
    * roles JSON export
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from role_refinement.appearance_extractor import (
    FEATURE_DIM,
    AppearanceExtractor,
    DictCropProvider,
)
from role_refinement.role_exporter import RoleJSONExporter
from role_refinement.role_models import (
    ROLE_BALL,
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
    ROLE_REFEREE,
    ROLE_UNKNOWN,
    RoleResult,
    TrackAppearance,
)
from role_refinement.role_refiner import RoleRefiner
from role_refinement.team_clusterer import ClusterResult, TeamClusterer
from tracking.models import Track
from utils.config_loader import RoleRefinementConfig

# BGR solid colors used as synthetic jersey crops.
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)     # grass-like; should be masked then fall back
BLACK = (0, 0, 0)
MAGENTA = (255, 0, 255)


def solid_crop(color, h=80, w=40):
    crop = np.zeros((h, w, 3), dtype=np.uint8)
    crop[:, :] = color
    return crop


def cfg(**overrides):
    base = dict(min_track_length=15, samples_per_track=20)
    base.update(overrides)
    return RoleRefinementConfig(**base)


def make_track(track_id, frame_id, cx=640, cy=360, name="player"):
    return Track(
        frame_id=frame_id,
        track_id=track_id,
        class_id=2,
        class_name=name,
        confidence=0.9,
        bbox=(cx - 10, cy - 30, cx + 10, cy + 30),
    )


def make_history(track_id, n_frames, name="player", center=(640, 360)):
    cx, cy = center
    return {track_id: [make_track(track_id, f, cx, cy, name) for f in range(1, n_frames + 1)]}


# ---------------------------------------------------------------------------
# Appearance extraction
# ---------------------------------------------------------------------------
def test_vector_from_crop_shape_and_normalization():
    ext = AppearanceExtractor(cfg())
    vec = ext.vector_from_crop(solid_crop(RED))
    assert vec is not None
    assert vec.shape == (FEATURE_DIM,)
    assert vec.sum() == pytest.approx(1.0, abs=1e-5)


def test_vector_from_crop_distinguishes_colors():
    ext = AppearanceExtractor(cfg())
    red = ext.vector_from_crop(solid_crop(RED))
    blue = ext.vector_from_crop(solid_crop(BLUE))
    # Different kit colors -> different appearance vectors.
    assert not np.allclose(red, blue)


def test_vector_from_crop_grass_fallback():
    # A pure-grass crop has all pixels masked; extraction must still
    # produce a vector (falls back to using all pixels) rather than None.
    ext = AppearanceExtractor(cfg())
    vec = ext.vector_from_crop(solid_crop(GREEN))
    assert vec is not None
    assert vec.sum() == pytest.approx(1.0, abs=1e-5)


def test_vector_from_crop_rejects_empty():
    ext = AppearanceExtractor(cfg())
    assert ext.vector_from_crop(None) is None
    assert ext.vector_from_crop(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_sample_frames_even_spacing_and_cap():
    ext = AppearanceExtractor(cfg(samples_per_track=5))
    history = make_history(1, 100)
    plan = ext.sample_frames(history)
    assert len(plan[1]) == 5
    assert plan[1][0] == 1 and plan[1][-1] == 100  # spans the whole track
    assert plan[1] == sorted(plan[1])


def test_sample_frames_short_track_returns_all():
    ext = AppearanceExtractor(cfg(samples_per_track=20))
    history = make_history(1, 4)
    plan = ext.sample_frames(history)
    assert plan[1] == [1, 2, 3, 4]


def test_extract_skips_appearance_for_ball():
    ext = AppearanceExtractor(cfg(ball_class_name="ball"))
    history = make_history(99, 30, name="ball")
    plan = ext.sample_frames(history)
    # No crops supplied; ball needs none.
    appearances = ext.extract(history, plan, DictCropProvider({}), (1280, 720))
    ball = appearances[99]
    assert ball.detected_class == "ball"
    assert not ball.has_appearance
    assert ball.track_length == 30


# ---------------------------------------------------------------------------
# Team clustering + outliers
# ---------------------------------------------------------------------------
def _appearance_from_color(ext, track_id, color, length=30, center=(640, 360)):
    vec = ext.vector_from_crop(solid_crop(color))
    samples = np.stack([vec, vec, vec], axis=0)
    return TrackAppearance(
        track_id=track_id,
        detected_class="player",
        vector=vec,
        sample_vectors=samples,
        n_samples=3,
        track_length=length,
        centers=[center] * length,
        frame_size=(1280, 720),
    )


def test_team_clusterer_splits_two_teams_and_flags_outlier():
    ext = AppearanceExtractor(cfg())
    appearances = {}
    for tid in range(1, 6):
        appearances[tid] = _appearance_from_color(ext, tid, RED)
    for tid in range(6, 11):
        appearances[tid] = _appearance_from_color(ext, tid, BLUE)
    # A clearly different (dark) appearance -> outlier candidate.
    appearances[11] = _appearance_from_color(ext, 11, BLACK)

    result = TeamClusterer(cfg()).cluster(appearances)
    assert result.ok
    assert result.centroids.shape == (2, FEATURE_DIM)

    red_team = {result.team_of.get(t) for t in range(1, 6)}
    blue_team = {result.team_of.get(t) for t in range(6, 11)}
    assert red_team == {next(iter(red_team))}      # all reds share a team
    assert blue_team == {next(iter(blue_team))}     # all blues share a team
    assert red_team != blue_team                    # and the teams differ
    assert 11 in result.outliers                    # dark kit is an outlier


def test_team_clusterer_skips_when_too_few_stable_tracks():
    ext = AppearanceExtractor(cfg())
    appearances = {1: _appearance_from_color(ext, 1, RED, length=5)}  # too short
    result = TeamClusterer(cfg()).cluster(appearances)
    assert not result.ok


# ---------------------------------------------------------------------------
# Role voting / fallbacks (RoleRefiner is pure)
# ---------------------------------------------------------------------------
def two_team_cluster():
    centroids = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return ClusterResult(centroids=centroids)


def appearance(track_id, vecs, length=30, name="player", centers=None, fs=(1280, 720)):
    arr = np.asarray(vecs, dtype=np.float32)
    mean = arr.mean(axis=0) if len(arr) else np.empty((0,), np.float32)
    return TrackAppearance(
        track_id=track_id,
        detected_class=name,
        vector=mean,
        sample_vectors=arr,
        n_samples=len(arr),
        track_length=length,
        centers=centers or [(640, 360)] * length,
        frame_size=fs,
    )


def test_vote_assigns_player_to_team():
    refiner = RoleRefiner(cfg())
    app = appearance(1, [[1, 0, 0]] * 5)  # always nearest team 0
    result = refiner.refine({1: app}, two_team_cluster())[1]
    assert result.refined_role == ROLE_PLAYER
    assert result.team_id == 0
    assert result.role_confidence == pytest.approx(1.0)
    assert result.role_reason == "team_cluster_match"


def test_ambiguous_team_votes_fall_back_to_unknown():
    # With reconciliation off, an ambiguous vote is the cautious "unknown".
    refiner = RoleRefiner(cfg(temporal_vote_threshold=0.7, resolve_unknowns=False))
    # Half the samples vote team 0, half team 1 -> no clear winner.
    app = appearance(1, [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
    result = refiner.refine({1: app}, two_team_cluster())[1]
    assert result.refined_role == ROLE_UNKNOWN
    assert result.role_reason == "ambiguous_team_low_vote"


def test_stable_player_assigned_team_despite_ambiguous_vote():
    # A long, stable track with a split team vote is a real player on one
    # team -> assigned to its leaning team, not left "unknown".
    refiner = RoleRefiner(cfg(temporal_vote_threshold=0.7,
                              assign_team_for_stable_players=True,
                              stable_min_length=45))
    # 3 votes team 0, 2 votes team 1 -> fraction 0.6 (< 0.7), leans team 0.
    app = appearance(1, [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
                     length=200)
    result = refiner.refine({1: app}, two_team_cluster())[1]
    assert result.refined_role == ROLE_PLAYER
    assert result.team_id == 0
    assert result.role_reason == "team_nearest_stable"


def test_short_ambiguous_track_still_unknown():
    # Below stable_min_length, an ambiguous track stays unknown (reconcile off).
    refiner = RoleRefiner(cfg(temporal_vote_threshold=0.7,
                              assign_team_for_stable_players=True,
                              stable_min_length=45, resolve_unknowns=False))
    app = appearance(1, [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], length=20)
    result = refiner.refine({1: app}, two_team_cluster())[1]
    assert result.refined_role == ROLE_UNKNOWN
    assert result.role_reason == "ambiguous_team_low_vote"


def test_short_track_is_unknown():
    refiner = RoleRefiner(cfg(min_track_length=15, resolve_unknowns=False))
    app = appearance(1, [[1, 0, 0]] * 3, length=5)
    result = refiner.refine({1: app}, two_team_cluster())[1]
    assert result.refined_role == ROLE_UNKNOWN
    assert result.role_reason == "short_track_low_confidence"


def test_ball_role_is_preserved():
    refiner = RoleRefiner(cfg(ball_class_name="ball"))
    # Even with no appearance at all, a ball stays a ball.
    app = appearance(7, [], length=30, name="ball")
    result = refiner.refine({7: app}, two_team_cluster())[7]
    assert result.refined_role == ROLE_BALL
    assert result.role_confidence == pytest.approx(1.0)
    assert result.team_id is None


def test_outlier_central_motion_is_referee():
    refiner = RoleRefiner(cfg())
    # Far from both teams (votes outlier) + central, wide-roaming trajectory.
    centers = [(x, 360) for x in range(100, 1200, 100)]
    app = appearance(1, [[0, 0, 1]] * 6, centers=centers * 3)
    result = refiner.refine({1: app}, two_team_cluster())[1]
    assert result.refined_role == ROLE_REFEREE
    assert result.role_reason == "appearance_outlier_central_motion"


def test_outlier_goal_side_is_goalkeeper():
    refiner = RoleRefiner(cfg())
    # Far from both teams + hugs the left edge with tiny spread.
    centers = [(60 + (i % 3), 360) for i in range(30)]
    app = appearance(1, [[0, 0, 1]] * 6, centers=centers)
    result = refiner.refine({1: app}, two_team_cluster())[1]
    assert result.refined_role == ROLE_GOALKEEPER
    assert result.role_reason == "appearance_outlier_goal_side"


def test_still_outlier_at_moderate_edge_is_goalkeeper():
    # A STILL outlier toward a side (x~0.82 -> edge~0.64, tiny spread) is a
    # goalkeeper even though its edge is below goalkeeper_min_edge -- a keeper
    # barely moves. (Regression: this used to be forced to referee.)
    refiner = RoleRefiner(cfg(goalkeeper_min_edge=0.70, goalkeeper_max_roam=0.30))
    centers = [(1050 + (i % 3), 360) for i in range(30)]   # x_norm ~0.82, still
    app = appearance(1, [[0, 0, 1]] * 6, centers=centers)
    result = refiner.refine({1: app}, two_team_cluster())[1]
    assert result.refined_role == ROLE_GOALKEEPER


def test_roaming_outlier_near_side_is_referee():
    # An outlier toward a side that ROAMS a lot is the assistant referee, not
    # a keeper -- motion separates them when position can't.
    refiner = RoleRefiner(cfg(goalkeeper_min_edge=0.70, goalkeeper_max_roam=0.30))
    # Mean x ~0.82 (edge ~0.64) but a big horizontal spread (patrols the line):
    # alternating 900/1200 px on a 1280-wide frame -> mean 1050 (x_norm 0.82).
    centers = [(900 + (i % 2) * 300, 360) for i in range(30)]
    app = appearance(1, [[0, 0, 1]] * 6, centers=centers)
    result = refiner.refine({1: app}, two_team_cluster())[1]
    assert result.refined_role == ROLE_REFEREE


def test_clustering_unavailable_yields_unknown():
    refiner = RoleRefiner(cfg())
    empty = ClusterResult(centroids=np.empty((0, 0), dtype=np.float32))
    app = appearance(1, [[1, 0, 0]] * 5)
    result = refiner.refine({1: app}, empty)[1]
    assert result.refined_role == ROLE_UNKNOWN
    assert result.role_reason == "clustering_unavailable"


# ---------------------------------------------------------------------------
# Automatic reconciliation — never ship a person track as "unknown"
# ---------------------------------------------------------------------------
def test_resolve_unknowns_ambiguous_vote_to_nearest_team():
    # A short, ambiguous track (would be unknown) is force-resolved to the
    # team its samples lean toward when reconciliation is on (the default).
    refiner = RoleRefiner(cfg(temporal_vote_threshold=0.7))
    # 3 votes team 0, 2 votes team 1 -> leans team 0, but below threshold.
    app = appearance(1, [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]],
                     length=20)
    result = refiner.refine({1: app}, two_team_cluster())[1]
    assert result.refined_role == ROLE_PLAYER
    assert result.team_id == 0
    assert result.role_reason == "unknown_resolved_nearest_team"


def test_resolve_unknowns_short_track_to_nearest_team():
    # A track too short to be "stable" still has color -> nearest team.
    refiner = RoleRefiner(cfg(min_track_length=15))
    app = appearance(1, [[0, 1, 0]] * 3, length=5)   # nearest team 1
    result = refiner.refine({1: app}, two_team_cluster())[1]
    assert result.refined_role == ROLE_PLAYER
    assert result.team_id == 1
    assert result.role_reason == "unknown_resolved_nearest_team"


def test_resolve_unknowns_off_keeps_unknown():
    # The escape hatch: with reconciliation disabled the cautious role stands.
    refiner = RoleRefiner(cfg(min_track_length=15, resolve_unknowns=False))
    app = appearance(1, [[0, 1, 0]] * 3, length=5)
    result = refiner.refine({1: app}, two_team_cluster())[1]
    assert result.refined_role == ROLE_UNKNOWN


def test_resolve_unknowns_colorless_track_inherits_neighbor_team():
    # A track with no usable appearance can't be colored; it inherits the
    # team of the nearest already-labeled teammate by image position.
    refiner = RoleRefiner(cfg())
    near = appearance(1, [[0, 1, 0]] * 5, length=40, centers=[(200, 360)] * 40)
    colorless = appearance(2, [], length=8, centers=[(210, 360)] * 8)
    results = refiner.refine({1: near, 2: colorless}, two_team_cluster())
    assert results[1].team_id == 1
    assert results[2].refined_role == ROLE_PLAYER
    assert results[2].team_id == 1
    assert results[2].role_reason == "unknown_resolved_neighbor"


def test_resolve_unknowns_never_relabels_committed_referee():
    # Genuine referees committed by the outlier split are left untouched.
    refiner = RoleRefiner(cfg())
    centers = [(x, 360) for x in range(100, 1200, 100)]
    ref = appearance(1, [[0, 0, 1]] * 6, centers=centers * 3)
    result = refiner.refine({1: ref}, two_team_cluster())[1]
    assert result.refined_role == ROLE_REFEREE


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------
def test_roles_json_export_roundtrip(tmp_path):
    roles = {
        12: RoleResult(12, "player", ROLE_REFEREE, 0.87, None,
                       "appearance_outlier_central_motion"),
        3: RoleResult(3, "player", ROLE_PLAYER, 0.95, 1, "team_cluster_match"),
    }
    path = tmp_path / "out" / "vid_roles.json"
    exporter = RoleJSONExporter(
        path,
        metadata={"phase": "role_refinement",
                  "method": "appearance_clustering_temporal_voting"},
    )
    saved = exporter.save(roles)
    assert saved == path

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"]["phase"] == "role_refinement"
    # Sorted by track_id.
    assert [t["track_id"] for t in data["tracks"]] == [3, 12]
    ref = data["tracks"][1]
    assert ref == {
        "track_id": 12,
        "detected_class": "player",
        "refined_role": "referee",
        "role_confidence": 0.87,
        "team_id": None,
        "role_reason": "appearance_outlier_central_motion",
    }
    assert data["tracks"][0]["team_id"] == 1


def test_roleresult_rejects_illegal_role():
    with pytest.raises(ValueError):
        RoleResult(1, "player", "linesman", 0.5, None, "nope")

def test_run_role_refinement_reuses_cached_roles(tmp_path, monkeypatch):
    import main
    from role_refinement import role_refiner

    tracks = tmp_path / "m_tracks.json"
    tracks.write_text("{}", encoding="utf-8")
    roles = tmp_path / "m_roles.json"
    roles.write_text(json.dumps({"tracks": [{"refined_role": "player"}]}),
                     encoding="utf-8")

    def boom(*_a, **_k):
        raise AssertionError("refinement re-ran despite cached roles")

    monkeypatch.setattr(role_refiner.RoleRefinementRunner, "run", boom)
    config = main.load_config("config/config.yaml")
    config.video.output_dir = str(tmp_path)
    config.role_refinement.output_path = None
    summary = main.run_role_refinement(
        config, "m.mp4", {"tracks_json": str(tracks)})
    assert summary["roles_json"] == str(roles)
    assert summary["n_tracks"] == 1
    assert summary["roles_by_type"] == {"player": 1}
