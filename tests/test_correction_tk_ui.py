"""Tests for the native Tkinter manual-correction UI.

Only the Tk-free core is exercised (manifest loading, save/delete,
keyboard role mapping, manifest auto-detection) — no display required, so
these run headless in CI.
"""

from __future__ import annotations

import json

import pytest

from manual_correction.correction_models import Correction
from manual_correction.correction_tk_ui import (
    FILTER_ALL,
    FILTER_ID_SWITCH,
    FILTER_LOW_CONF,
    FILTER_UNCORRECTED,
    FILTER_UNKNOWN,
    KEY_ROLE_MAP,
    CorrectionApp,
    CorrectionSession,
    filter_indices,
    find_latest_manifest,
    load_manifest,
    normalize_team,
    parse_team_choice,
    role_for_key,
    summarize,
    team_display_for,
    team_dropdown_options,
)


def _candidate(track_id, role="unknown", conf=0.3, team=None,
               frame_ids=None, crop_paths=None, frame_paths=None, bboxes=None):
    frame_ids = frame_ids if frame_ids is not None else [1, 15, 30]
    return {
        "track_id": track_id,
        "detected_class": "player",
        "current_role": role,
        "role_confidence": conf,
        "team_id": team,
        "track_length": 30,
        "frame_ids": frame_ids,
        "crop_paths": crop_paths if crop_paths is not None else [],
        "frame_paths": frame_paths if frame_paths is not None else [],
        "bboxes": bboxes if bboxes is not None else [],
        "role_reason": "ambiguous_team_low_vote",
    }


def _write_manifest(path, candidates, corrections_file=None, team_legend=None):
    meta = {"phase": "manual_role_review"}
    if corrections_file is not None:
        meta["corrections_file"] = str(corrections_file)
    payload = {"metadata": meta, "candidates": candidates}
    if team_legend is not None:
        payload["team_legend"] = team_legend
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


_LEGEND = {
    "0": {"team_id": 0, "label": "mostly white/black kit",
          "color_rgb": [240, 240, 240], "swatch": "#f0f0f0", "crop_paths": []},
    "1": {"team_id": 1, "label": "mostly green kit",
          "color_rgb": [40, 160, 40], "swatch": "#28a028", "crop_paths": []},
}


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _Root:
    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


class _FakeWidget:
    def __init__(self, kind, parent=None, **kwargs):
        self.kind = kind
        self.parent = parent
        self.kwargs = kwargs
        self.children = []
        self.grid_options = None
        self.pack_options = None
        self.column_weights = {}
        if parent is not None:
            parent.children.append(self)

    def grid(self, *args, **kwargs):
        self.grid_options = kwargs

    def pack(self, *args, **kwargs):
        self.pack_options = kwargs

    def columnconfigure(self, column, weight):
        self.column_weights[column] = weight


class _FakeTtk:
    def Frame(self, parent=None, **kwargs):
        return _FakeWidget("Frame", parent, **kwargs)

    def Label(self, parent=None, **kwargs):
        return _FakeWidget("Label", parent, **kwargs)

    def Button(self, parent=None, **kwargs):
        return _FakeWidget("Button", parent, **kwargs)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------
def test_load_manifest_reads_candidates(tmp_path):
    path = _write_manifest(tmp_path / "m.json", [_candidate(2), _candidate(3)])
    data = load_manifest(path)
    assert len(data["candidates"]) == 2
    assert data["candidates"][0]["track_id"] == 2


def test_load_manifest_rejects_non_manifest(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"something": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(bad)


def test_load_manifest_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.json")


def test_find_latest_manifest_picks_newest(tmp_path):
    review = tmp_path / "manual_review"
    (review / "vidA").mkdir(parents=True)
    older = _write_manifest(review / "vidA" / "manifest.json", [_candidate(1)])
    newer = _write_manifest(review / "vidB_review.json", [_candidate(2)])
    # Ensure distinct mtimes regardless of filesystem resolution.
    import os
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    found = find_latest_manifest(review)
    assert found == newer


def test_find_latest_manifest_none_when_empty(tmp_path):
    assert find_latest_manifest(tmp_path / "does_not_exist") is None


# ---------------------------------------------------------------------------
# Keyboard role mapping
# ---------------------------------------------------------------------------
def test_keyboard_role_mapping():
    assert KEY_ROLE_MAP == {
        "1": "player",
        "2": "goalkeeper",
        "3": "referee",
        "4": "ball",
        "5": "unknown",
        "6": "ignore",
    }
    assert role_for_key("1") == "player"
    assert role_for_key("3") == "referee"
    assert role_for_key("6") == "ignore"
    assert role_for_key("9") is None
    assert role_for_key("x") is None


def test_team_helpers():
    assert parse_team_choice("None") is None
    assert parse_team_choice("0") == 0
    assert parse_team_choice("1") == 1
    # Team only applies to players.
    assert normalize_team("player", 1) == 1
    assert normalize_team("referee", 1) is None
    assert normalize_team("goalkeeper", 0) is None


# ---------------------------------------------------------------------------
# Session: save / delete / navigation
# ---------------------------------------------------------------------------
def test_session_save_correction_writes_store_format(tmp_path):
    corr_file = tmp_path / "corr.json"
    manifest = _write_manifest(
        tmp_path / "m.json", [_candidate(2), _candidate(3)], corr_file
    )
    session = CorrectionSession(manifest)

    session.save_correction(2, "referee", None)
    # Player keeps its team; non-player team is normalized to None.
    session.save_correction(3, "player", 1)

    on_disk = json.loads(corr_file.read_text(encoding="utf-8"))
    assert on_disk == {
        "2": {"role": "referee", "team_id": None},
        "3": {"role": "player", "team_id": 1},
    }


def test_session_save_correction_persists_player_name(tmp_path):
    corr_file = tmp_path / "corr.json"
    manifest = _write_manifest(tmp_path / "m.json", [_candidate(9)], corr_file)
    session = CorrectionSession(manifest)

    session.save_correction(9, "player", 0, player_name="Alex Morgan")

    on_disk = json.loads(corr_file.read_text(encoding="utf-8"))
    assert on_disk["9"]["player_name"] == "Alex Morgan"
    assert session.existing(9) == Correction("player", 0, player_name="Alex Morgan")


def test_finish_review_commits_current_name_before_final_roles(tmp_path):
    from manual_correction.correction_runner import RoleCorrectionRunner
    from utils.config_loader import ManualCorrectionConfig

    corr_file = tmp_path / "corr.json"
    roles_json = tmp_path / "roles.json"
    final_json = tmp_path / "roles_final.json"
    manifest = _write_manifest(
        tmp_path / "m.json",
        [_candidate(1, role="player", conf=0.99, team=1)],
        corr_file,
        _LEGEND,
    )
    roles_json.write_text(json.dumps({
        "tracks": [{
            "track_id": 1,
            "detected_class": "player",
            "refined_role": "player",
            "role_confidence": 0.99,
            "team_id": 1,
            "role_reason": "test",
        }]
    }), encoding="utf-8")
    session = CorrectionSession(manifest)

    app = CorrectionApp.__new__(CorrectionApp)
    app.session = session
    app._team_display_to_id = {
        display: team_id
        for display, team_id in team_dropdown_options(session.team_legend)
    }
    app.role_var = _Var("player")
    app.team_var = _Var(team_display_for(session.team_legend, 1))
    app.name_var = _Var("Mohamed wael")
    app.switch_var = _Var(False)
    app.note_var = _Var("")
    app.merge_var = _Var("")
    app.switch_frame_var = _Var("")
    app._dirty = False
    app.confirmed = False
    app.root = _Root()
    app._refresh_summary = lambda: None
    app._set_status = lambda message=None: None

    app._on_finish()

    on_disk = json.loads(corr_file.read_text(encoding="utf-8"))
    assert on_disk["1"]["player_name"] == "Mohamed wael"
    assert app.confirmed is True
    assert app.root.destroyed is True

    RoleCorrectionRunner(ManualCorrectionConfig()).run_apply(
        video_path="input_video.mp4",
        roles_json_path=roles_json,
        corrections_file=corr_file,
        final_output_path=final_json,
    )
    final = json.loads(final_json.read_text(encoding="utf-8"))
    assert final["tracks"][0]["player_name"] == "Mohamed wael"
    assert final["tracks"][0]["player_display_name"] == "Mohamed wael"


def test_action_footer_buttons_are_created_in_fixed_footer_container():
    app = CorrectionApp.__new__(CorrectionApp)
    app._ttk = _FakeTtk()
    app.root = _FakeWidget("Root")
    app.status_var = _Var("")
    for _label, handler_name in CorrectionApp._ACTION_FOOTER_BUTTONS:
        setattr(app, handler_name, lambda: None)

    app._build_action_footer()

    assert app.action_footer.parent is app.root
    assert app.action_footer.grid_options == {"row": 2, "column": 0, "sticky": "ew"}
    assert app.action_footer.column_weights == {0: 1}
    assert app.footer_status_label.parent is app.action_footer
    assert app.footer_buttons_container.parent is app.action_footer
    assert list(app.footer_buttons) == [
        "Save Current",
        "Previous",
        "Next",
        "Finish Review",
    ]
    assert all(
        button.parent is app.footer_buttons_container
        for button in app.footer_buttons.values()
    )


def test_session_save_team_override_only_for_players(tmp_path):
    corr_file = tmp_path / "corr.json"
    manifest = _write_manifest(tmp_path / "m.json", [_candidate(5)], corr_file)
    session = CorrectionSession(manifest)
    c = session.save_correction(5, "referee", 1)   # team should be dropped
    assert c.team_id is None
    assert session.existing(5) == Correction("referee", None)


def test_session_delete_correction_reverts(tmp_path):
    corr_file = tmp_path / "corr.json"
    manifest = _write_manifest(tmp_path / "m.json", [_candidate(7)], corr_file)
    session = CorrectionSession(manifest)

    session.save_correction(7, "goalkeeper", None)
    assert session.existing(7) is not None

    assert session.delete_correction(7) is True
    assert session.existing(7) is None
    assert json.loads(corr_file.read_text(encoding="utf-8")) == {}
    # Deleting again is a no-op.
    assert session.delete_correction(7) is False


def test_session_preloads_existing_corrections(tmp_path):
    corr_file = tmp_path / "corr.json"
    corr_file.write_text(json.dumps({"2": {"role": "referee", "team_id": None}}),
                         encoding="utf-8")
    manifest = _write_manifest(tmp_path / "m.json", [_candidate(2)], corr_file)
    session = CorrectionSession(manifest)
    assert session.existing(2) == Correction("referee", None)


def test_session_navigation(tmp_path):
    manifest = _write_manifest(
        tmp_path / "m.json", [_candidate(1), _candidate(2), _candidate(3)]
    )
    session = CorrectionSession(manifest, corrections_path=tmp_path / "c.json")
    assert session.current["track_id"] == 1
    assert session.go_next()["track_id"] == 2
    assert session.go_next()["track_id"] == 3
    assert session.go_next()["track_id"] == 3      # clamps at the end
    assert session.go_prev()["track_id"] == 2
    assert session.go_to(0)["track_id"] == 1
    assert session.go_prev()["track_id"] == 1      # clamps at the start


# ---------------------------------------------------------------------------
# Multiple sampled frames + frame navigation
# ---------------------------------------------------------------------------
def test_loads_all_sampled_frames_and_frame_nav(tmp_path):
    cand = _candidate(
        2,
        frame_ids=[10, 20, 30, 40, 50],
        crop_paths=[f"c{i}.jpg" for i in range(5)],
        frame_paths=[f"f{i}.jpg" for i in range(5)],
        bboxes=[[0, 0, 5, 5]] * 5,
    )
    manifest = _write_manifest(tmp_path / "m.json", [cand])
    session = CorrectionSession(manifest, corrections_path=tmp_path / "c.json")

    assert session.frame_count == 5          # all sampled frames available
    assert session.frame_pos == 0
    assert session.next_frame() == 1
    assert session.next_frame() == 2
    assert session.prev_frame() == 1
    assert session.select_frame(4) == 4
    assert session.next_frame() == 4         # clamps at the last frame
    # Jump-to-frame picks the nearest sampled frame id.
    assert session.jump_to_frame_number(31) == 2   # nearest to 30
    assert session.jump_to_frame_number(48) == 4   # nearest to 50


# ---------------------------------------------------------------------------
# ID-switch metadata save / load + backward compatibility
# ---------------------------------------------------------------------------
def test_id_switch_metadata_save_and_load(tmp_path):
    corr_file = tmp_path / "corr.json"
    manifest = _write_manifest(tmp_path / "m.json", [_candidate(1)], corr_file)
    session = CorrectionSession(manifest)

    session.save_correction(
        1, "player", 1,
        id_switch=True,
        switch_note="green -> white around frame 369",
        merge_with_track_id=88,
        switch_frame=369,
    )
    on_disk = json.loads(corr_file.read_text(encoding="utf-8"))
    assert on_disk["1"] == {
        "role": "player",
        "team_id": 1,
        "id_switch": True,
        "switch_note": "green -> white around frame 369",
        "merge_with_track_id": 88,
        "switch_frame": 369,
    }

    # Reload into a fresh session: metadata survives the round-trip.
    reloaded = CorrectionSession(manifest)
    c = reloaded.existing(1)
    assert c.id_switch is True
    assert c.switch_note == "green -> white around frame 369"
    assert c.merge_with_track_id == 88
    assert c.switch_frame == 369
    assert c.is_manual_swap


def test_simple_correction_keeps_old_two_key_format(tmp_path):
    corr_file = tmp_path / "corr.json"
    manifest = _write_manifest(tmp_path / "m.json", [_candidate(1)], corr_file)
    session = CorrectionSession(manifest)
    session.save_correction(1, "referee", None)        # no switch info
    on_disk = json.loads(corr_file.read_text(encoding="utf-8"))
    assert on_disk["1"] == {"role": "referee", "team_id": None}   # unchanged shape


def test_backward_compatible_old_corrections_load(tmp_path):
    # An old corrections file written before ID-switch fields existed.
    corr_file = tmp_path / "corr.json"
    corr_file.write_text(json.dumps({
        "1": {"role": "player", "team_id": 0},
        "2": {"role": "referee", "team_id": None},
    }), encoding="utf-8")
    manifest = _write_manifest(
        tmp_path / "m.json", [_candidate(1), _candidate(2)], corr_file)
    session = CorrectionSession(manifest)
    assert session.existing(1) == Correction("player", 0)
    assert session.existing(2).id_switch is False     # defaults applied


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def test_filter_indices_modes():
    candidates = [
        _candidate(1, role="unknown", conf=0.2),
        _candidate(2, role="referee", conf=0.6),
        _candidate(3, role="player", conf=0.9),
    ]
    corrections = {
        2: Correction("referee", None, id_switch=True),   # corrected + switch
    }
    assert filter_indices(candidates, corrections, FILTER_ALL, 0.65) == [0, 1, 2]
    assert filter_indices(candidates, corrections, FILTER_UNKNOWN, 0.65) == [0]
    # confidence < 0.65 -> tracks 1 and 2
    assert filter_indices(candidates, corrections, FILTER_LOW_CONF, 0.65) == [0, 1]
    # uncorrected -> tracks 1 and 3 (track 2 has a correction)
    assert filter_indices(candidates, corrections, FILTER_UNCORRECTED, 0.65) == [0, 2]
    assert filter_indices(candidates, corrections, FILTER_ID_SWITCH, 0.65) == [1]


def test_session_set_filter_changes_visible_view(tmp_path):
    candidates = [
        _candidate(1, role="unknown", conf=0.2),
        _candidate(2, role="player", conf=0.9),
        _candidate(3, role="unknown", conf=0.3),
    ]
    manifest = _write_manifest(tmp_path / "m.json", candidates)
    session = CorrectionSession(manifest, corrections_path=tmp_path / "c.json")
    assert session.count == 3
    session.set_filter(FILTER_UNKNOWN)
    assert session.count == 2
    assert [c["track_id"] for c in session.visible_candidates] == [1, 3]
    session.set_filter(FILTER_ALL)
    assert session.count == 3


def test_filter_recomputes_after_save(tmp_path):
    candidates = [_candidate(1, role="unknown"), _candidate(2, role="unknown")]
    manifest = _write_manifest(tmp_path / "m.json", candidates)
    session = CorrectionSession(manifest, corrections_path=tmp_path / "c.json")
    session.set_filter(FILTER_UNCORRECTED)
    assert session.count == 2
    session.save_correction(1, "player", 0)     # now track 1 is corrected
    assert [c["track_id"] for c in session.visible_candidates] == [2]


# ---------------------------------------------------------------------------
# Summary counts
# ---------------------------------------------------------------------------
def test_summary_counts(tmp_path):
    candidates = [_candidate(i) for i in range(1, 6)]   # 5 candidates
    corrections = {
        1: Correction("player", 0),
        2: Correction("player", 1),
        3: Correction("referee", None, id_switch=True),
        4: Correction("ignore", None),
    }
    s = summarize(candidates, corrections)
    assert s["total"] == 5
    assert s["corrected"] == 4
    assert s["remaining"] == 1
    assert s["players"] == 2
    assert s["referees"] == 1
    assert s["goalkeepers"] == 0
    assert s["ignored"] == 1
    assert s["id_switches"] == 1


def test_session_summary_after_edits(tmp_path):
    candidates = [_candidate(1), _candidate(2), _candidate(3)]
    manifest = _write_manifest(tmp_path / "m.json", candidates)
    session = CorrectionSession(manifest, corrections_path=tmp_path / "c.json")
    assert session.summary()["corrected"] == 0
    session.save_correction(1, "referee", None, id_switch=True)
    session.save_correction(2, "player", 0)
    s = session.summary()
    assert s["corrected"] == 2
    assert s["remaining"] == 1
    assert s["referees"] == 1
    assert s["players"] == 1
    assert s["id_switches"] == 1


# ---------------------------------------------------------------------------
# Team legend dropdown labels + numeric team_id preservation
# ---------------------------------------------------------------------------
def test_team_dropdown_options_use_visual_labels():
    legend = {0: _LEGEND["0"], 1: _LEGEND["1"]}
    options = team_dropdown_options(legend)
    assert options == [
        ("No team / not applicable", None),
        ("Team 0 — mostly white/black kit", 0),
        ("Team 1 — mostly green kit", 1),
    ]


def test_team_dropdown_options_fallback_without_legend():
    options = team_dropdown_options({})
    assert options == [
        ("No team / not applicable", None),
        ("Team 0", 0),
        ("Team 1", 1),
    ]


def test_team_display_for_roundtrip():
    legend = {0: _LEGEND["0"], 1: _LEGEND["1"]}
    assert team_display_for(legend, 0) == "Team 0 — mostly white/black kit"
    assert team_display_for(legend, 1) == "Team 1 — mostly green kit"
    assert team_display_for(legend, None) == "No team / not applicable"


def test_session_loads_team_legend(tmp_path):
    manifest = _write_manifest(
        tmp_path / "m.json", [_candidate(1)], team_legend=_LEGEND)
    session = CorrectionSession(manifest, corrections_path=tmp_path / "c.json")
    assert session.team_legend[0]["label"] == "mostly white/black kit"
    assert session.team_legend[1]["swatch"] == "#28a028"


def test_saved_team_id_stays_numeric(tmp_path):
    # Simulate what the UI does: map a friendly display string back to the
    # numeric team_id before saving, then verify the JSON is numeric/null.
    corr_file = tmp_path / "corr.json"
    manifest = _write_manifest(
        tmp_path / "m.json", [_candidate(1), _candidate(2), _candidate(3)],
        corr_file, team_legend=_LEGEND)
    session = CorrectionSession(manifest)
    options = team_dropdown_options(session.team_legend)
    display_to_id = {disp: tid for disp, tid in options}

    # Player on "Team 1 — mostly green kit" -> numeric 1.
    session.save_correction(
        1, "player", display_to_id["Team 1 — mostly green kit"])
    # Player on Team 0 -> numeric 0.
    session.save_correction(
        2, "player", display_to_id["Team 0 — mostly white/black kit"])
    # Referee with "No team" -> null.
    session.save_correction(
        3, "referee", display_to_id["No team / not applicable"])

    on_disk = json.loads(corr_file.read_text(encoding="utf-8"))
    assert on_disk["1"]["team_id"] == 1
    assert on_disk["2"]["team_id"] == 0
    assert on_disk["3"]["team_id"] is None
    # Types are real ints / null — never the display strings.
    assert isinstance(on_disk["1"]["team_id"], int)
    assert isinstance(on_disk["2"]["team_id"], int)
