"""Tests for the detector referee prior (Phase 3).

A detector "referee" that role refinement flipped to "player" with only
moderate confidence should revert to referee — unless confidence is very
high, the prior is disabled, or a manual / user-initialized correction
overrides it.
"""

from __future__ import annotations

import numpy as np

from manual_correction.correction_models import Correction
from manual_correction.correction_runner import apply_corrections
from role_refinement.role_models import (
    ROLE_BALL,
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
    ROLE_REFEREE,
    RoleResult,
    TrackAppearance,
)
from role_refinement.role_refiner import RoleRefiner, apply_referee_prior
from role_refinement.team_clusterer import ClusterResult
from utils.config_loader import RoleRefinementConfig


def player_result(track_id, detected_class, conf, team=0):
    return RoleResult(track_id, detected_class, ROLE_PLAYER, conf, team,
                      "team_cluster_match")


# ---------------------------------------------------------------------------
# Pure function
# ---------------------------------------------------------------------------
def test_detected_referee_not_converted_below_threshold():
    result = player_result(7, "referee", 0.62, team=0)
    out = apply_referee_prior(result, enabled=True, threshold=0.90)
    assert out.refined_role == ROLE_REFEREE
    assert out.team_id is None
    assert out.role_reason == "detector_referee_prior"
    assert out.detected_class == "referee"


def test_detected_referee_stays_player_above_threshold():
    result = player_result(7, "referee", 0.95, team=0)
    out = apply_referee_prior(result, enabled=True, threshold=0.90)
    assert out.refined_role == ROLE_PLAYER          # high confidence wins
    assert out.team_id == 0


def test_threshold_is_strict_inequality_at_boundary():
    # Exactly at the threshold is NOT below it -> stays player.
    out = apply_referee_prior(player_result(7, "referee", 0.90), True, 0.90)
    assert out.refined_role == ROLE_PLAYER


def test_prior_disabled_leaves_player():
    out = apply_referee_prior(player_result(7, "referee", 0.50), False, 0.90)
    assert out.refined_role == ROLE_PLAYER


def test_prior_only_affects_referee_detected_players():
    # A detector "player" flipped to player -> untouched.
    out = apply_referee_prior(player_result(7, "player", 0.40), True, 0.90)
    assert out.refined_role == ROLE_PLAYER

    # Ball and goalkeeper refined roles are never touched (rule needs player).
    ball = RoleResult(9, "ball", ROLE_BALL, 1.0, None, "ball_passthrough")
    assert apply_referee_prior(ball, True, 0.90).refined_role == ROLE_BALL
    gk = RoleResult(5, "referee", ROLE_GOALKEEPER, 0.5, None, "x")
    assert apply_referee_prior(gk, True, 0.90).refined_role == ROLE_GOALKEEPER


# ---------------------------------------------------------------------------
# Integration through RoleRefiner.refine
# ---------------------------------------------------------------------------
def _appearance(track_id, vecs, detected="referee", length=30):
    arr = np.asarray(vecs, dtype=np.float32)
    return TrackAppearance(
        track_id=track_id,
        detected_class=detected,
        vector=arr.mean(axis=0),
        sample_vectors=arr,
        n_samples=len(arr),
        track_length=length,
        centers=[(640, 360)] * length,
        frame_size=(1280, 720),
    )


def _two_team_cluster():
    centroids = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return ClusterResult(centroids=centroids)


def test_refine_reverts_low_confidence_referee_player():
    cfg = RoleRefinementConfig(referee_to_player_override_confidence=0.90)
    refiner = RoleRefiner(cfg)
    # 7 of 8 samples vote team 0 -> player at 0.875 confidence (< 0.90).
    vecs = [[1, 0, 0]] * 7 + [[0, 1, 0]]
    result = refiner.refine({7: _appearance(7, vecs)}, _two_team_cluster())[7]
    assert result.refined_role == ROLE_REFEREE
    assert result.role_reason == "detector_referee_prior"


def test_refine_keeps_high_confidence_referee_player():
    cfg = RoleRefinementConfig(referee_to_player_override_confidence=0.90)
    refiner = RoleRefiner(cfg)
    # All samples vote team 0 -> player at 1.0 confidence (>= 0.90).
    result = refiner.refine(
        {7: _appearance(7, [[1, 0, 0]] * 8)}, _two_team_cluster())[7]
    assert result.refined_role == ROLE_PLAYER


# ---------------------------------------------------------------------------
# Manual / user-initialized corrections still override the prior
# ---------------------------------------------------------------------------
def test_manual_correction_overrides_referee_prior():
    # Auto role (after the prior) is referee; the reviewer says it's a player.
    auto_roles = {
        7: RoleResult(7, "referee", ROLE_REFEREE, 0.62, None,
                      "detector_referee_prior"),
    }
    finals = apply_corrections(auto_roles, {7: Correction("player", 0)})
    assert finals[7].final_role == ROLE_PLAYER
    assert finals[7].team_id == 0
    assert finals[7].auto_role == ROLE_REFEREE      # prior preserved as auto
    assert finals[7].corrected_by_user is True


def test_user_initialized_correction_overrides_referee_prior():
    auto_roles = {
        7: RoleResult(7, "referee", ROLE_REFEREE, 0.62, None,
                      "detector_referee_prior"),
    }
    seed = Correction("player", 1, user_initialized=True)
    finals = apply_corrections(auto_roles, {7: seed})
    assert finals[7].final_role == ROLE_PLAYER
    assert finals[7].team_id == 1
    assert finals[7].user_initialized is True
