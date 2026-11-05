"""Integration test for Phase 3 using fake tracks + synthetic crops.

Builds a tiny synthetic "match": two color-coded teams, a centrally
roaming referee, an edge-hugging goalkeeper, a ball, and a short noise
track — then runs the whole extraction -> clustering -> refinement chain
through :class:`RoleRefinementRunner` with an injected crop provider (no
real video needed).
"""

from __future__ import annotations

import numpy as np

from role_refinement.appearance_extractor import DictCropProvider
from role_refinement.role_models import (
    ROLE_BALL,
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
    ROLE_REFEREE,
)
from role_refinement.role_refiner import RoleRefinementRunner
from tracking.models import Track
from utils.config_loader import RoleRefinementConfig

FRAME_SIZE = (1280, 720)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
MAGENTA = (255, 0, 255)


def solid_crop(color, h=80, w=40):
    crop = np.zeros((h, w, 3), dtype=np.uint8)
    crop[:, :] = color
    return crop


def track(track_id, frame_id, cx, cy, name="player"):
    return Track(
        frame_id=frame_id,
        track_id=track_id,
        class_id=2,
        class_name=name,
        confidence=0.9,
        bbox=(cx - 10, cy - 30, cx + 10, cy + 30),
    )


def build_match():
    """Return (track_history, color_by_track, expected_role)."""
    history = {}
    color_by_track = {}
    n = 30

    # Team A (red) and Team B (blue): 6 players each, scattered, stationary.
    next_id = 1
    for team_color, base_x in ((RED, 300), (BLUE, 900)):
        for k in range(6):
            tid = next_id
            next_id += 1
            cx, cy = base_x + k * 20, 200 + k * 60
            history[tid] = [track(tid, f, cx, cy) for f in range(1, n + 1)]
            color_by_track[tid] = team_color

    # Referee: distinct dark kit, roams centrally across the whole width.
    ref_id = next_id
    next_id += 1
    history[ref_id] = [
        track(ref_id, f, 150 + (f * 35) % 1000, 360) for f in range(1, n + 1)
    ]
    color_by_track[ref_id] = BLACK

    # Goalkeeper: distinct kit, hugs the left edge with tiny movement.
    gk_id = next_id
    next_id += 1
    history[gk_id] = [
        track(gk_id, f, 55 + (f % 3), 360) for f in range(1, n + 1)
    ]
    color_by_track[gk_id] = MAGENTA

    # Ball: color irrelevant (no appearance used).
    ball_id = next_id
    next_id += 1
    history[ball_id] = [track(ball_id, f, 640, 360, name="ball") for f in range(1, n + 1)]

    # Short noise track (length 4): must end up "unknown".
    short_id = next_id
    history[short_id] = [track(short_id, f, 500, 500) for f in range(1, 5)]
    color_by_track[short_id] = RED

    ids = {
        "red": list(range(1, 7)),
        "blue": list(range(7, 13)),
        "referee": ref_id,
        "goalkeeper": gk_id,
        "ball": ball_id,
        "short": short_id,
    }
    return history, color_by_track, ids


def build_provider(extractor, history, color_by_track):
    plan = extractor.sample_frames(history)
    crops = {}
    for tid, frame_ids in plan.items():
        if tid not in color_by_track:
            continue
        for fid in frame_ids:
            crops[(tid, fid)] = solid_crop(color_by_track[tid])
    return plan, DictCropProvider(crops)


def test_end_to_end_role_refinement():
    history, color_by_track, ids = build_match()
    config = RoleRefinementConfig(enabled=True, min_track_length=15, samples_per_track=10)
    runner = RoleRefinementRunner(config)

    plan, provider = build_provider(runner._extractor, history, color_by_track)
    roles = runner.refine_with_provider(history, plan, provider, FRAME_SIZE)

    # Ball role is preserved untouched.
    assert roles[ids["ball"]].refined_role == ROLE_BALL

    # Each team is internally consistent and the two teams differ.
    red_teams = {roles[t].team_id for t in ids["red"]}
    blue_teams = {roles[t].team_id for t in ids["blue"]}
    for t in ids["red"] + ids["blue"]:
        assert roles[t].refined_role == ROLE_PLAYER
    assert len(red_teams) == 1 and len(blue_teams) == 1
    assert red_teams != blue_teams

    # Outliers are not classified as players.
    assert roles[ids["referee"]].refined_role != ROLE_PLAYER
    assert roles[ids["goalkeeper"]].refined_role != ROLE_PLAYER

    # And they should separate by motion: central roamer = referee,
    # edge-hugger = goalkeeper.
    assert roles[ids["referee"]].refined_role == ROLE_REFEREE
    assert roles[ids["goalkeeper"]].refined_role == ROLE_GOALKEEPER

    # Short track has too little history to commit on its own, but automatic
    # reconciliation (on by default) still resolves it to the team its red
    # kit is nearest to instead of leaving a grey "unknown" on the pitch.
    short = roles[ids["short"]]
    assert short.refined_role == ROLE_PLAYER
    assert short.team_id == next(iter(red_teams))
    assert short.role_reason == "unknown_resolved_nearest_team"


def test_roles_cover_every_track():
    history, color_by_track, _ = build_match()
    config = RoleRefinementConfig(min_track_length=15, samples_per_track=10)
    runner = RoleRefinementRunner(config)
    plan, provider = build_provider(runner._extractor, history, color_by_track)
    roles = runner.refine_with_provider(history, plan, provider, FRAME_SIZE)
    assert set(roles) == set(history)
