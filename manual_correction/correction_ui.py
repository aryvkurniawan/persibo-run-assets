"""Lightweight Streamlit UI for human-in-the-loop role correction.

Launch (after a ``--manual-role-review`` run has produced a manifest)::

    streamlit run manual_correction/correction_ui.py

The manifest and corrections-file paths are read from the environment
variables ``FOOTBALL_AI_REVIEW_MANIFEST`` and
``FOOTBALL_AI_CORRECTIONS_FILE`` (both also editable in the sidebar), so
``main.py`` can point the UI at the right files when it prints the launch
command.

The reviewer only ever sees the ambiguous tracks the runner selected
(``unknown`` / low confidence) — never the whole video.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make the project root importable when launched via `streamlit run`.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from manual_correction.correction_models import IGNORE  # noqa: E402
from manual_correction.correction_store import CorrectionStore  # noqa: E402

_NO_CHANGE = "(no change)"
_ROLE_OPTIONS = [
    _NO_CHANGE,
    "player",
    "goalkeeper",
    "referee",
    "ball",
    "unknown",
    IGNORE,
]
_TEAM_OPTIONS = ["auto", "none", "0", "1"]


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _team_default_index(team_id) -> int:
    if team_id == 0:
        return _TEAM_OPTIONS.index("0")
    if team_id == 1:
        return _TEAM_OPTIONS.index("1")
    return _TEAM_OPTIONS.index("auto")


def _resolve_team(team_choice: str, auto_team):
    if team_choice == "auto":
        return auto_team
    if team_choice == "none":
        return None
    return int(team_choice)


def main() -> None:
    st.set_page_config(page_title="Role Correction", layout="wide")
    st.title("⚽ Manual Role Correction")
    st.caption("Review only the ambiguous tracks. Empty / unset = keep automatic role.")

    default_manifest = os.environ.get("FOOTBALL_AI_REVIEW_MANIFEST", "")
    manifest_path = st.sidebar.text_input("Review manifest", value=default_manifest)
    if not manifest_path:
        st.info("Set the review manifest path in the sidebar to begin.")
        return
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        st.error(f"Manifest not found: {manifest_file}")
        return

    manifest = _load_manifest(manifest_file)
    meta = manifest.get("metadata", {})
    candidates = manifest.get("candidates", [])

    default_corrections = os.environ.get(
        "FOOTBALL_AI_CORRECTIONS_FILE", meta.get("corrections_file", "")
    )
    corrections_path = st.sidebar.text_input(
        "Corrections file", value=default_corrections
    )
    store = CorrectionStore(corrections_path)
    existing = store.load()

    st.sidebar.markdown(f"**Candidates:** {len(candidates)}")
    st.sidebar.markdown(f"**Video:** {meta.get('source_video', '?')}")
    st.sidebar.markdown(
        f"**Threshold:** conf < {meta.get('review_confidence_threshold', '?')}"
    )

    if not candidates:
        st.success("No ambiguous tracks — nothing to review. 🎉")
        return

    with st.form("corrections_form"):
        selections = {}
        for cand in candidates:
            track_id = int(cand["track_id"])
            st.subheader(
                f"Track {track_id} — det: {cand['detected_class']} | "
                f"auto role: {cand['current_role']} "
                f"({cand['role_confidence']:.2f})"
            )
            cols = st.columns([3, 2])
            with cols[0]:
                crops = [p for p in cand.get("crop_paths", []) if Path(p).is_file()]
                if crops:
                    st.image(crops, width=120,
                             caption=[f"f{f}" for f in cand.get("frame_ids", [])][:len(crops)])
                else:
                    st.caption("(no crops saved)")
            with cols[1]:
                st.write(f"team_id: {cand['team_id']}")
                st.write(f"track_length: {cand['track_length']}")
                st.write(f"reason: {cand.get('role_reason', '')}")

                prior = existing.get(track_id)
                role_index = (
                    _ROLE_OPTIONS.index(prior.role)
                    if prior and prior.role in _ROLE_OPTIONS
                    else 0
                )
                role_choice = st.selectbox(
                    "Set role", _ROLE_OPTIONS, index=role_index,
                    key=f"role_{track_id}",
                )
                team_default = (
                    _team_default_index(prior.team_id)
                    if prior
                    else _team_default_index(cand["team_id"])
                )
                team_choice = st.selectbox(
                    "Team (players only)", _TEAM_OPTIONS, index=team_default,
                    key=f"team_{track_id}",
                )
            selections[track_id] = (role_choice, team_choice, cand["team_id"])
            st.divider()

        submitted = st.form_submit_button("💾 Save corrections")

    if submitted:
        corrections = store.load()
        changed = 0
        for track_id, (role_choice, team_choice, auto_team) in selections.items():
            if role_choice == _NO_CHANGE:
                # Revert: drop any existing correction for this track.
                if track_id in corrections:
                    del corrections[track_id]
                    changed += 1
                continue
            team_id = (
                _resolve_team(team_choice, auto_team)
                if role_choice == "player"
                else None
            )
            from manual_correction.correction_models import Correction
            corrections[track_id] = Correction(role=role_choice, team_id=team_id)
            changed += 1
        store.save(corrections)
        st.success(f"Saved {len(corrections)} correction(s) to {corrections_path}.")


if __name__ == "__main__":  # `streamlit run` executes the file as __main__.
    main()
