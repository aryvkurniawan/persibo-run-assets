"""Tests for Phase 3.5 — human-in-the-loop role correction.

Covered:
    * correction file loading (incl. malformed-entry tolerance)
    * correction application (override layer, priority)
    * team override
    * missing / extra correction entries
    * final-export integrity (auto + final both preserved)
    * candidate discovery + review-dataset (UI data) generation
    * reversibility / ignore no-op
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from manual_correction.correction_models import (
    Correction,
    FinalRole,
    ReviewCandidate,
)
from manual_correction.correction_runner import (
    RoleCorrectionRunner,
    apply_corrections,
    find_candidate_ids,
    find_review_ids,
    load_auto_roles,
    representative_frame_ids,
)
from manual_correction.correction_store import CorrectionStore
from role_refinement.role_models import RoleResult
from tracking.models import Track
from utils.config_loader import ManualCorrectionConfig


def auto(track_id, role, conf, team=None, det="player", reason="r"):
    return RoleResult(track_id, det, role, conf, team, reason)


def mc_cfg(**kw):
    base = dict(review_confidence_threshold=0.65, representative_frames=3)
    base.update(kw)
    return ManualCorrectionConfig(**base)


# ---------------------------------------------------------------------------
# Correction store: loading
# ---------------------------------------------------------------------------
def test_store_load_missing_file_returns_empty(tmp_path):
    store = CorrectionStore(tmp_path / "nope.json")
    assert store.load() == {}


def test_store_load_parses_entries(tmp_path):
    path = tmp_path / "corr.json"
    path.write_text(json.dumps({
        "1": {"role": "player", "team_id": 0},
        "107": {"role": "referee", "team_id": None},
    }), encoding="utf-8")
    corrections = CorrectionStore(path).load()
    assert corrections[1] == Correction("player", 0)
    assert corrections[107] == Correction("referee", None)


def test_store_skips_malformed_entries(tmp_path):
    path = tmp_path / "corr.json"
    path.write_text(json.dumps({
        "1": {"role": "player", "team_id": 0},
        "2": {"team_id": 0},               # missing 'role'
        "3": {"role": "linesman"},          # illegal role
        "bad": {"role": "referee"},         # non-int key
    }), encoding="utf-8")
    corrections = CorrectionStore(path).load()
    assert set(corrections) == {1}


def test_store_save_roundtrip_sorted(tmp_path):
    path = tmp_path / "out" / "corr.json"
    store = CorrectionStore(path)
    store.save({107: Correction("referee", None), 1: Correction("player", 1)})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data.keys()) == ["1", "107"]              # sorted by id
    assert data["1"] == {"role": "player", "team_id": 1}
    assert data["107"] == {"role": "referee", "team_id": None}


def test_store_upsert_and_remove(tmp_path):
    store = CorrectionStore(tmp_path / "corr.json")
    store.upsert(5, "goalkeeper", None)
    assert store.load()[5] == Correction("goalkeeper", None)
    store.remove(5)
    assert 5 not in store.load()       # reversible


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------
def test_find_candidates_unknown_and_low_confidence():
    auto_roles = {
        1: auto(1, "player", 1.0, 0),          # confident player -> skip
        2: auto(2, "unknown", 0.2),            # unknown -> candidate
        3: auto(3, "referee", 0.45),           # low confidence -> candidate
        4: auto(4, "player", 0.66, 1),         # just above threshold -> skip
        5: auto(5, "ball", 1.0, None, det="ball"),  # ball never reviewed
    }
    ids = find_candidate_ids(auto_roles, mc_cfg())
    assert ids == [2, 3]


def test_find_candidates_respects_review_roles_config():
    auto_roles = {
        1: auto(1, "goalkeeper", 0.9, None),
        2: auto(2, "player", 0.99, 0),
    }
    cfg = mc_cfg(review_roles=["goalkeeper"], review_confidence_threshold=0.0)
    assert find_candidate_ids(auto_roles, cfg) == [1]


def test_find_review_ids_includes_every_non_ball_track_for_name_entry():
    auto_roles = {
        1: auto(1, "player", 1.0, 0),
        2: auto(2, "unknown", 0.2),
        3: auto(3, "referee", 0.9, None),
        4: auto(4, "ball", 1.0, None, det="ball"),
    }
    assert find_review_ids(auto_roles, mc_cfg()) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Override layer
# ---------------------------------------------------------------------------
def test_apply_corrections_overrides_and_preserves_auto():
    auto_roles = {
        1: auto(1, "player", 0.9, 0),
        119: auto(119, "unknown", 0.45, None, det="referee"),
    }
    corrections = {119: Correction("referee", None)}
    finals = apply_corrections(auto_roles, corrections)

    # Corrected track: final changes, auto preserved, flagged.
    f = finals[119]
    assert f.auto_role == "unknown"
    assert f.final_role == "referee"
    assert f.corrected_by_user is True
    assert f.role_confidence == 0.45            # auto confidence preserved
    assert f.detected_class == "referee"

    # Untouched track keeps auto role and is not flagged.
    assert finals[1].final_role == "player"
    assert finals[1].corrected_by_user is False


def test_apply_team_override():
    auto_roles = {1: auto(1, "player", 0.5, 0)}
    finals = apply_corrections(auto_roles, {1: Correction("player", 1)})
    assert finals[1].final_role == "player"
    assert finals[1].team_id == 1                # overridden
    assert finals[1].auto_team_id == 0           # original preserved
    assert finals[1].corrected_by_user is True


def test_player_name_roundtrips_and_is_optional():
    named = Correction("player", 0, player_name="Messi")
    assert named.to_dict()["player_name"] == "Messi"
    assert Correction.from_dict(named.to_dict()).player_name == "Messi"
    # No name -> key omitted (backward-compatible) and loads as None.
    plain = Correction("player", 0)
    assert "player_name" not in plain.to_dict()
    assert Correction.from_dict({"role": "player", "team_id": 0}).player_name is None


def test_apply_propagates_player_name():
    auto_roles = {7: auto(7, "player", 0.5, 0)}
    finals = apply_corrections(
        auto_roles, {7: Correction("player", 0, player_name="Ronaldo")})
    assert finals[7].player_name == "Ronaldo"
    assert finals[7].to_dict()["player_name"] == "Ronaldo"
    assert finals[7].to_dict()["player_display_name"] == "Ronaldo"


def test_empty_player_name_has_display_name_not_manual_name():
    auto_roles = {7: auto(7, "player", 0.5, 0)}
    finals = apply_corrections(auto_roles, {7: Correction("player", 0)})
    data = finals[7].to_dict()
    assert data["player_display_name"] == "Player 7"
    assert "player_name" not in data


def test_apply_ignore_is_noop():
    auto_roles = {1: auto(1, "unknown", 0.3, None)}
    finals = apply_corrections(auto_roles, {1: Correction("ignore", None)})
    assert finals[1].final_role == "unknown"
    assert finals[1].corrected_by_user is False  # ignore leaves auto untouched


def test_apply_missing_entries_use_auto():
    auto_roles = {1: auto(1, "player", 0.9, 0), 2: auto(2, "referee", 0.8, None)}
    finals = apply_corrections(auto_roles, {})   # no corrections at all
    assert all(not f.corrected_by_user for f in finals.values())
    assert finals[2].final_role == "referee"


def test_apply_ignores_corrections_for_unknown_tracks():
    auto_roles = {1: auto(1, "player", 0.9, 0)}
    finals = apply_corrections(auto_roles, {999: Correction("referee", None)})
    assert set(finals) == {1}                    # phantom track 999 dropped


# ---------------------------------------------------------------------------
# Final export integrity
# ---------------------------------------------------------------------------
def test_export_final_roundtrip(tmp_path):
    runner = RoleCorrectionRunner(mc_cfg())
    finals = {
        119: FinalRole(119, "referee", "unknown", "referee", 0.45, None, None,
                       True, "appearance_outlier_central_motion"),
        1: FinalRole(1, "player", "player", "player", 0.95, 0, 0, False, "team"),
    }
    path = runner.export_final(finals, tmp_path / "vid_roles_final.json",
                               metadata={"phase": "manual_role_correction"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [t["track_id"] for t in data["tracks"]] == [1, 119]
    corrected = data["tracks"][1]
    assert corrected["auto_role"] == "unknown"
    assert corrected["final_role"] == "referee"
    assert corrected["corrected_by_user"] is True
    assert corrected["role_confidence"] == 0.45


# ---------------------------------------------------------------------------
# Representative frame selection + review-dataset (UI data) generation
# ---------------------------------------------------------------------------
def test_representative_frame_ids_picks_start_mid_end():
    obs = [Track(f, 7, 2, "player", 0.9, (0, 0, 10, 10)) for f in range(1, 31)]
    assert representative_frame_ids(obs, 3) == [1, 15, 30]
    assert representative_frame_ids(obs, 1) == [16]
    short = obs[:2]
    assert representative_frame_ids(short, 3) == [1, 2]


def _write_roles_json(path, roles):
    payload = {"metadata": {}, "tracks": [r.to_dict() for r in roles]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_tracks_json(path, history, resolution=(1920, 1080)):
    frames = {}
    for obs in history.values():
        for t in obs:
            frames.setdefault(t.frame_id, []).append(t)
    payload = {
        "metadata": {"resolution": list(resolution)},
        "frames": [
            {"frame": fi, "tracks": [
                {"track_id": t.track_id, "class": t.class_name,
                 "confidence": t.confidence, "bbox": list(t.bbox)}
                for t in frames[fi]
            ]}
            for fi in sorted(frames)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_review_dataset_generates_ui_data(tmp_path, monkeypatch):
    # Auto roles: all non-ball tracks are shown so names can be entered.
    roles = [auto(1, "player", 0.99, 0), auto(2, "unknown", 0.2, None)]
    roles_json = tmp_path / "vid_roles.json"
    _write_roles_json(roles_json, roles)

    history = {
        1: [Track(f, 1, 2, "player", 0.9, (10, 10, 30, 70)) for f in range(1, 21)],
        2: [Track(f, 2, 2, "player", 0.3, (40, 40, 60, 100)) for f in range(1, 21)],
    }
    tracks_json = tmp_path / "vid_tracks.json"
    _write_tracks_json(tracks_json, history)

    cfg = mc_cfg(review_dir=str(tmp_path / "review"), representative_frames=3,
                 save_review_frames=False)
    runner = RoleCorrectionRunner(cfg)

    # Stub the (video-reading) image collection so the test needs no video.
    def fake_collect(self, video_path, requests_by_frame, review_root):
        crop_paths = {}
        for fid, reqs in requests_by_frame.items():
            for tid, _ in reqs:
                p = review_root / f"track_{tid:04d}_f{fid:06d}.jpg"
                p.write_bytes(b"fakejpg")
                crop_paths[(tid, fid)] = str(p)
        return crop_paths, {}

    monkeypatch.setattr(RoleCorrectionRunner, "_collect_images", fake_collect)

    dataset = runner.build_review_dataset(
        "vid.mp4", tracks_json, roles_json
    )
    assert dataset["n_candidates"] == 2
    assert [c.track_id for c in dataset["candidates"]] == [1, 2]
    cand = dataset["candidates"][1]
    assert isinstance(cand, ReviewCandidate)
    assert cand.track_id == 2
    assert cand.current_role == "unknown"
    assert cand.track_length == 20
    assert cand.frame_ids == [1, 11, 20]   # start / mid / end of a 20-frame track
    assert len(cand.crop_paths) == 3

    # Manifest is written and self-describing.
    manifest = json.loads((tmp_path / "review" / "vid_review.json").read_text())
    assert manifest["metadata"]["phase"] == "manual_role_review"
    assert len(manifest["candidates"]) == 2
    assert manifest["metadata"]["review_scope"] == "all_non_ball_tracks"


def test_load_auto_roles_reads_phase3_json(tmp_path):
    roles = [auto(12, "referee", 0.87, None, det="player")]
    path = tmp_path / "r.json"
    _write_roles_json(path, roles)
    loaded = load_auto_roles(path)
    assert loaded[12].refined_role == "referee"
    assert loaded[12].detected_class == "player"
    assert loaded[12].team_id is None


def test_run_manual_correction_launches_ui_before_final_export(tmp_path, monkeypatch):
    import main
    from manual_correction.initialization import (
        InitializationLabels,
        InitializationStore,
        TEAM_1,
    )
    from utils.config_loader import AppConfig

    output_dir = tmp_path / "outputs"
    corrections = output_dir / "vid_role_corrections.json"
    init_labels = output_dir / "vid_init_labels.json"
    roles_json = output_dir / "vid_roles.json"
    tracks_json = output_dir / "vid_tracks.json"
    output_dir.mkdir()

    _write_roles_json(roles_json, [auto(1, "player", 0.99, 0)])
    _write_tracks_json(
        tracks_json,
        {1: [Track(f, 1, 2, "player", 0.9, (10, 10, 30, 70))
             for f in range(1, 6)]},
    )
    labels = InitializationLabels()
    labels.assign(1, [1], TEAM_1)
    InitializationStore(init_labels).save(labels)

    def fake_collect(self, video_path, requests_by_frame, review_root):
        crop_paths = {}
        for fid, reqs in requests_by_frame.items():
            for tid, _ in reqs:
                p = review_root / f"track_{tid:04d}_f{fid:06d}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"fakejpg")
                crop_paths[(tid, fid)] = str(p)
        return crop_paths, {}

    class FakeCorrectionApp:
        def __init__(self, session):
            self.session = session

        def run(self):
            self.session.save_correction(
                1, "player", 0, player_name="Lionel Messi")
            return True

    monkeypatch.setattr(RoleCorrectionRunner, "_collect_images", fake_collect)
    monkeypatch.setattr(
        "manual_correction.correction_tk_ui.CorrectionApp", FakeCorrectionApp)

    config = AppConfig()
    config.video.output_dir = str(output_dir)
    config.manual_correction = mc_cfg(
        review_dir=str(output_dir / "manual_review"),
        corrections_file=str(corrections),
        initialization_labels_file=str(init_labels),
    )
    args = main.parse_args(["--video", "vid.mp4", "--manual-role-review"])
    result = main.run_manual_correction(
        config,
        args,
        "vid.mp4",
        {"tracks_json": str(tracks_json)},
        {"roles_json": str(roles_json)},
    )

    final = json.loads((output_dir / "vid_roles_final.json").read_text("utf-8"))
    assert result["final_roles_json"].endswith("vid_roles_final.json")
    assert final["tracks"][0]["player_name"] == "Lionel Messi"
    assert final["tracks"][0]["player_display_name"] == "Lionel Messi"
    assert final["tracks"][0]["team_id"] == 0
    assert result["n_named"] == 1


def test_run_manual_correction_stops_when_tk_review_not_confirmed(
    tmp_path, monkeypatch
):
    import main
    from utils.config_loader import AppConfig

    output_dir = tmp_path / "outputs"
    corrections = output_dir / "vid_role_corrections.json"
    roles_json = output_dir / "vid_roles.json"
    tracks_json = output_dir / "vid_tracks.json"
    output_dir.mkdir()

    _write_roles_json(roles_json, [auto(1, "player", 0.99, 0)])
    _write_tracks_json(
        tracks_json,
        {1: [Track(f, 1, 2, "player", 0.9, (10, 10, 30, 70))
             for f in range(1, 6)]},
    )

    def fake_collect(self, video_path, requests_by_frame, review_root):
        crop_paths = {}
        for fid, reqs in requests_by_frame.items():
            for tid, _ in reqs:
                p = review_root / f"track_{tid:04d}_f{fid:06d}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"fakejpg")
                crop_paths[(tid, fid)] = str(p)
        return crop_paths, {}

    class FakeCorrectionApp:
        def __init__(self, session):
            self.session = session

        def run(self):
            return False

    monkeypatch.setattr(RoleCorrectionRunner, "_collect_images", fake_collect)
    monkeypatch.setattr(
        "manual_correction.correction_tk_ui.CorrectionApp", FakeCorrectionApp)

    config = AppConfig()
    config.video.output_dir = str(output_dir)
    config.manual_correction = mc_cfg(
        review_dir=str(output_dir / "manual_review"),
        corrections_file=str(corrections),
    )
    args = main.parse_args(["--video", "vid.mp4", "--manual-role-review"])

    with pytest.raises(RuntimeError, match="closed without confirmation"):
        main.run_manual_correction(
            config,
            args,
            "vid.mp4",
            {"tracks_json": str(tracks_json)},
            {"roles_json": str(roles_json)},
        )

    assert not (output_dir / "vid_roles_final.json").exists()


def test_unrequested_review_keeps_automatic_roles(tmp_path, monkeypatch):
    import main
    from manual_correction import correction_runner as cr

    def boom(*_a, **_k):
        raise AssertionError("review dataset built without the flag")

    monkeypatch.setattr(cr.RoleCorrectionRunner, "build_review_dataset", boom)
    config = main.load_config("config/config.yaml")
    config.manual_correction = mc_cfg(
        review_dir=str(tmp_path),
        corrections_file=str(tmp_path / "none.json"),
    )
    args = main.parse_args(["--video", "vid.mp4", "--role-refinement"])
    result = main.run_manual_correction(
        config,
        args,
        "vid.mp4",
        {"tracks_json": str(tmp_path / "tracks.json")},
        {"roles_json": str(tmp_path / "roles.json")},
    )
    assert result is None

# ---------------------------------------------------------------------------
# Per-video output paths (no collision when testing another video)
# ---------------------------------------------------------------------------
def test_corrections_and_init_paths_are_per_video():
    import main

    args = main.parse_args(["--video", "matchA.mp4", "--all"])
    config = main.apply_overrides(main.load_config("config/config.yaml"), args)
    assert config.manual_correction.corrections_file == \
        "outputs/matchA_role_corrections.json"
    assert config.manual_correction.initialization_labels_file == \
        "outputs/matchA_init_labels.json"

    args2 = main.parse_args(["--video", "matchB.mp4", "--all"])
    config2 = main.apply_overrides(main.load_config("config/config.yaml"), args2)
    assert config2.manual_correction.corrections_file == \
        "outputs/matchB_role_corrections.json"


def test_explicit_corrections_file_is_respected():
    import main

    args = main.parse_args(
        ["--video", "matchA.mp4", "--corrections-file", "outputs/my_corr.json"])
    config = main.apply_overrides(main.load_config("config/config.yaml"), args)
    assert config.manual_correction.corrections_file == "outputs/my_corr.json"
