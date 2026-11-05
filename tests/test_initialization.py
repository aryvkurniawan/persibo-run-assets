"""Tests for user-guided initialization (Phase 3.5).

Covered:
    * initialization label save / load (exact JSON format)
    * multi-select assignment logic
    * representative-frame selection
    * propagation: initialized tracks override auto roles
    * ID-switch tracks are NOT blindly propagated
    * backward compatibility with existing manual_role_corrections.json
"""

from __future__ import annotations

import json

from manual_correction.correction_models import Correction
from manual_correction.correction_runner import apply_corrections
from manual_correction.correction_store import CorrectionStore
from manual_correction.initialization import (
    GOALKEEPER,
    GROUPS,
    NEEDS_SPLIT_NOTE,
    REFEREE,
    TEAM_0,
    TEAM_1,
    InitializationLabels,
    InitializationStore,
    assign_group,
    empty_frame_labels,
    merge_seeds,
    propagate_initialization,
    select_initialization_frames,
)
from role_refinement.role_models import RoleResult


def auto(track_id, role, conf=0.5, team=None, det="player"):
    return RoleResult(track_id, det, role, conf, team, "r")


# ---------------------------------------------------------------------------
# Multi-select assignment logic
# ---------------------------------------------------------------------------
def test_assign_group_moves_ids_into_one_group():
    frame = empty_frame_labels()
    assign_group(frame, [4, 10, 17], TEAM_0)
    assert frame[TEAM_0] == [4, 10, 17]

    # Re-assigning some ids to another group removes them from the first.
    assign_group(frame, [10, 24], TEAM_1)
    assert frame[TEAM_0] == [4, 17]
    assert frame[TEAM_1] == [10, 24]

    # A box belongs to at most one group.
    all_assigned = sum((frame[g] for g in GROUPS), [])
    assert all_assigned.count(10) == 1


def test_assign_group_none_clears():
    frame = empty_frame_labels()
    assign_group(frame, [1, 2, 3], REFEREE)
    assign_group(frame, [2], None)          # deselect track 2
    assert frame[REFEREE] == [1, 3]


def test_labels_assign_and_group_of():
    labels = InitializationLabels()
    labels.assign(1, [4, 10], TEAM_0)
    labels.assign(1, [20], REFEREE)
    assert labels.group_of(1, 4) == TEAM_0
    assert labels.group_of(1, 20) == REFEREE
    assert labels.group_of(1, 999) is None


# ---------------------------------------------------------------------------
# Save / load (exact format)
# ---------------------------------------------------------------------------
def test_init_labels_save_load_roundtrip(tmp_path):
    labels = InitializationLabels()
    labels.assign(1, [4, 10, 17], TEAM_0)
    labels.assign(1, [1, 7, 24], TEAM_1)
    labels.assign(1, [20], REFEREE)
    labels.assign(1, [115], GOALKEEPER)
    labels.assign(250, [5, 6], TEAM_0)

    path = tmp_path / "init.json"
    InitializationStore(path).save(labels)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["frames"]["1"]["team_0"] == [4, 10, 17]
    assert data["frames"]["1"]["team_1"] == [1, 7, 24]
    assert data["frames"]["1"]["referee"] == [20]
    assert data["frames"]["1"]["goalkeeper"] == [115]
    assert data["frames"]["1"]["ignore"] == []

    reloaded = InitializationStore(path).load()
    assert reloaded.group_of(1, 17) == TEAM_0
    assert reloaded.group_of(250, 5) == TEAM_0


def test_init_store_missing_file_is_empty(tmp_path):
    labels = InitializationStore(tmp_path / "nope.json").load()
    assert labels.frame_count() == 0


# ---------------------------------------------------------------------------
# Representative frame selection
# ---------------------------------------------------------------------------
def test_select_initialization_frames():
    present = list(range(1, 1001))     # frames 1..1000
    frames = select_initialization_frames(present, gk_frames=[640], custom=[123])
    assert 1 in frames                 # first
    assert 500 in frames or 501 in frames  # midfield (middle)
    assert 640 in frames               # goalkeeper-visible
    assert 123 in frames               # custom
    assert frames == sorted(frames)


def test_select_initialization_frames_empty_present():
    assert select_initialization_frames([], custom=[7, 3]) == [3, 7]


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------
def test_propagation_seeds_override_auto_roles():
    labels = InitializationLabels()
    labels.assign(1, [4], TEAM_0)
    labels.assign(1, [7], TEAM_1)
    labels.assign(1, [20], REFEREE)
    labels.assign(1, [115], GOALKEEPER)
    seeds = propagate_initialization(labels)

    assert seeds[4] == Correction("player", 0, user_initialized=True)
    assert seeds[7] == Correction("player", 1, user_initialized=True)
    assert seeds[20] == Correction("referee", None, user_initialized=True)
    assert seeds[115] == Correction("goalkeeper", None, user_initialized=True)

    # Applied over auto roles, the seed wins and is flagged user_initialized.
    auto_roles = {
        4: auto(4, "unknown", 0.2),
        7: auto(7, "player", 0.9, 0),
        20: auto(20, "player", 0.5),
        115: auto(115, "unknown", 0.3),
    }
    finals = apply_corrections(auto_roles, seeds)
    assert finals[4].final_role == "player" and finals[4].team_id == 0
    assert finals[4].user_initialized is True
    assert finals[20].final_role == "referee"
    assert finals[20].auto_role == "player"        # auto preserved
    assert finals[7].final_role == "player" and finals[7].team_id == 1


def test_id_switch_track_is_not_blindly_propagated():
    labels = InitializationLabels()
    # Same track id labelled Team 0 in one frame, Team 1 in another.
    labels.assign(1, [9], TEAM_0)
    labels.assign(250, [9], TEAM_1)
    seeds = propagate_initialization(labels)

    seed = seeds[9]
    assert seed.id_switch is True
    assert seed.switch_note == NEEDS_SPLIT_NOTE
    # NOT locked to a single team.
    assert seed.role == "unknown"
    assert seed.team_id is None
    assert seed.user_initialized is True


def test_majority_label_wins_over_a_switch_frame():
    # A track labelled Team 1 in three frames and Team 0 in one (the frame
    # where the box had switched players): the majority wins so the reviewer's
    # work isn't thrown away, but the switch is still flagged.
    labels = InitializationLabels()
    labels.assign(1, [21], TEAM_1)
    labels.assign(10, [21], TEAM_1)
    labels.assign(76, [21], TEAM_0)       # the switched frame
    labels.assign(376, [21], TEAM_1)
    seed = propagate_initialization(labels)[21]
    assert seed.role == "player"
    assert seed.team_id == 1               # majority Team 1, not unknown
    assert seed.id_switch is True          # switch still recorded
    assert seed.switch_note == NEEDS_SPLIT_NOTE


def test_tie_label_stays_unknown():
    # A genuine tie (no majority) is still left unknown for a manual split.
    labels = InitializationLabels()
    labels.assign(1, [9], TEAM_0)
    labels.assign(2, [9], TEAM_1)
    seed = propagate_initialization(labels)[9]
    assert seed.role == "unknown"
    assert seed.id_switch is True


def test_ignore_only_track_becomes_ignore_seed():
    labels = InitializationLabels()
    labels.assign(1, [42], "ignore")
    seeds = propagate_initialization(labels)
    assert seeds[42].role == "ignore"
    assert seeds[42].user_initialized is True
    # Applied as a no-op: keeps the auto role.
    finals = apply_corrections({42: auto(42, "player", 0.9, 1)}, seeds)
    assert finals[42].final_role == "player"
    assert finals[42].corrected_by_user is False
    assert finals[42].user_initialized is True


# ---------------------------------------------------------------------------
# Merge + backward compatibility
# ---------------------------------------------------------------------------
def test_merge_seeds_into_existing_corrections():
    existing = {1: Correction("player", 0), 2: Correction("referee", None)}
    seeds = {2: Correction("goalkeeper", None, user_initialized=True),
             5: Correction("player", 1, user_initialized=True)}
    merged = merge_seeds(existing, seeds, overwrite=True)
    assert merged[1] == Correction("player", 0)            # untouched
    assert merged[2].role == "goalkeeper"                  # seed overwrote
    assert merged[5].role == "player"                      # seed added

    # Without overwrite, existing entries win.
    merged2 = merge_seeds(existing, seeds, overwrite=False)
    assert merged2[2].role == "referee"


def test_user_initialized_is_backward_compatible(tmp_path):
    # A plain old corrections file (no user_initialized) still loads, and a
    # simple correction still serializes without the new key.
    path = tmp_path / "corr.json"
    path.write_text(json.dumps({"1": {"role": "player", "team_id": 0}}),
                    encoding="utf-8")
    store = CorrectionStore(path)
    loaded = store.load()
    assert loaded[1] == Correction("player", 0)
    assert loaded[1].user_initialized is False

    # Round-trip a simple correction: no extra keys leak in.
    store.save({1: Correction("referee", None)})
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["1"] == {"role": "referee", "team_id": None}

    # An initialized seed DOES record the flag.
    store.save({2: Correction("player", 1, user_initialized=True)})
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["2"]["user_initialized"] is True


def test_runner_apply_initialization_merges_into_corrections(tmp_path):
    from manual_correction.correction_runner import RoleCorrectionRunner
    from utils.config_loader import ManualCorrectionConfig

    labels = InitializationLabels()
    labels.assign(1, [4], TEAM_0)
    labels.assign(1, [20], REFEREE)
    labels.assign(1, [9], TEAM_0)
    labels.assign(250, [9], TEAM_1)        # track 9 conflicts -> id switch
    labels_path = tmp_path / "init.json"
    InitializationStore(labels_path).save(labels)

    corr_path = tmp_path / "corr.json"
    # A pre-existing manual correction that must survive the merge.
    CorrectionStore(corr_path).save({99: Correction("player", 0)})

    runner = RoleCorrectionRunner(ManualCorrectionConfig())
    result = runner.apply_initialization(labels_path, corr_path)
    assert result["n_seeds"] == 3
    assert result["n_id_switch"] == 1

    corrections = CorrectionStore(corr_path).load()
    assert corrections[4] == Correction("player", 0, user_initialized=True)
    assert corrections[20].role == "referee"
    assert corrections[9].id_switch is True            # not propagated as a team
    assert corrections[99] == Correction("player", 0)  # untouched


def test_init_seed_in_corrections_store_roundtrip(tmp_path):
    path = tmp_path / "corr.json"
    store = CorrectionStore(path)
    store.save({9: Correction("unknown", None, id_switch=True,
                              switch_note=NEEDS_SPLIT_NOTE, user_initialized=True)})
    reloaded = store.load()[9]
    assert reloaded.id_switch is True
    assert reloaded.switch_note == NEEDS_SPLIT_NOTE
    assert reloaded.user_initialized is True
