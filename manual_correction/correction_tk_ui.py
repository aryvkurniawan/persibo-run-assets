"""Fast native (Tkinter) desktop UI for manual role correction.

A snappier, Windows-friendly alternative to the Streamlit UI
(:mod:`manual_correction.correction_ui`) — both are kept. It reads the same
review manifest produced by::

    python main.py --video input_video.mp4 --tracking --role-refinement \
        --manual-role-review

and writes corrections in the exact same ``CorrectionStore`` format
(``outputs/manual_role_corrections.json``), so review work is
interchangeable between the two UIs.

Run it::

    python manual_correction/correction_tk_ui.py \
        --manifest outputs/manual_review/input_video_review.json

If ``--manifest`` is omitted, the latest manifest under
``outputs/manual_review/`` is auto-detected.

Features:
    * every sampled frame shown in a scrollable thumbnail strip (click to
      enlarge), not just start/mid/end — essential for spotting ID switches
    * full-frame context preview with the track's bbox highlighted
    * ID-switch tools (checkbox + note + "continue as id" hint), stored in
      the correction JSON without breaking the old format
    * filters (all / unknown / low-confidence / uncorrected / id-switch)
    * live correction summary and Saved / Unsaved status

Keyboard shortcuts:
    1=player 2=goalkeeper 3=referee 4=ball 5=unknown 6=ignore
    A/D = previous/next sampled frame    Left/Right = previous/next track
    Ctrl+S = save

Only the Python standard library + Pillow are used. All decision logic
lives in the Tk-free :class:`CorrectionSession` + module helpers so it is
unit-testable without a display; :class:`CorrectionApp` is a thin GUI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Make the project root importable when launched as a bare script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from manual_correction import initialization as initmod  # noqa: E402
from manual_correction.correction_models import (  # noqa: E402
    IGNORE,
    Correction,
)
from manual_correction.correction_store import CorrectionStore  # noqa: E402

# Role choices and their number-key shortcuts.
ROLE_CHOICES = ["player", "goalkeeper", "referee", "ball", "unknown", IGNORE]
KEY_ROLE_MAP: Dict[str, str] = {
    "1": "player",
    "2": "goalkeeper",
    "3": "referee",
    "4": "ball",
    "5": "unknown",
    "6": IGNORE,
}
TEAM_CHOICES = ["None", "0", "1"]

# Filter modes for the candidate list.
FILTER_ALL = "all"
FILTER_UNKNOWN = "unknown"
FILTER_LOW_CONF = "low_confidence"
FILTER_UNCORRECTED = "uncorrected"
FILTER_ID_SWITCH = "id_switch"
FILTER_MODES = [
    FILTER_ALL,
    FILTER_UNKNOWN,
    FILTER_LOW_CONF,
    FILTER_UNCORRECTED,
    FILTER_ID_SWITCH,
]

_DEFAULT_REVIEW_DIR = "outputs/manual_review"
_DEFAULT_CORRECTIONS = "outputs/manual_role_corrections.json"
_DEFAULT_INIT_LABELS = "outputs/manual_initialization_labels.json"
_DEFAULT_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# Pure helpers (no Tk — unit-testable)
# ---------------------------------------------------------------------------
def load_manifest(path: str | Path) -> dict:
    """Load and validate a review manifest JSON."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Review manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "candidates" not in data:
        raise ValueError(f"{path} is not a valid review manifest (no 'candidates')")
    return data


def find_latest_manifest(
    review_dir: str | Path = _DEFAULT_REVIEW_DIR,
) -> Optional[Path]:
    """Return the most recently modified manifest under ``review_dir``.

    Accepts both ``<stem>_review.json`` files written by the runner and
    ``<video>/manifest.json`` layouts. A file qualifies only if it parses
    as JSON and contains a ``candidates`` key.
    """
    root = Path(review_dir)
    if not root.is_dir():
        return None
    best: Optional[Path] = None
    best_mtime = -1.0
    for candidate in root.rglob("*.json"):
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        if not (isinstance(data, dict) and "candidates" in data):
            continue
        mtime = candidate.stat().st_mtime
        if mtime > best_mtime:
            best_mtime, best = mtime, candidate
    return best


def role_for_key(key: str) -> Optional[str]:
    """Map a keyboard key ('1'..'6') to a role, or ``None``."""
    return KEY_ROLE_MAP.get(key)


def normalize_team(role: str, team_id: Optional[int]) -> Optional[int]:
    """Team id only applies to players; everything else stores ``None``."""
    return team_id if role == "player" else None


def parse_team_choice(value: str) -> Optional[int]:
    """Map a team dropdown string ('None'/'0'/'1') to ``Optional[int]``."""
    if value in ("None", "", None):
        return None
    return int(value)


_NO_TEAM_LABEL = "No team / not applicable"


def team_dropdown_options(
    team_legend: Dict[int, dict]
) -> List["tuple[str, Optional[int]]"]:
    """Friendly dropdown entries: ``[(display_label, team_id), ...]``.

    Uses the auto-generated visual label from the team legend so reviewers
    see e.g. *"Team 0 — mostly white/black kit"* instead of a bare id. The
    stored value is still the numeric ``team_id`` (or ``None``).
    """
    options: List[tuple] = [(_NO_TEAM_LABEL, None)]
    for team in (0, 1):
        info = team_legend.get(team)
        label = info.get("label") if info else None
        if label:
            options.append((f"Team {team} — {label}", team))
        else:
            options.append((f"Team {team}", team))
    return options


def team_display_for(team_legend: Dict[int, dict], team_id: Optional[int]) -> str:
    """The dropdown display string for a numeric ``team_id``."""
    for display, value in team_dropdown_options(team_legend):
        if value == team_id:
            return display
    return _NO_TEAM_LABEL


def _parse_team_legend(manifest: dict) -> Dict[int, dict]:
    """Read the manifest's ``team_legend`` into an int-keyed dict."""
    raw = manifest.get("team_legend", {}) or {}
    legend: Dict[int, dict] = {}
    for key, info in raw.items():
        try:
            legend[int(key)] = info
        except (ValueError, TypeError):
            continue
    return legend


def filter_indices(
    candidates: List[dict],
    corrections: Dict[int, Correction],
    mode: str,
    threshold: float = _DEFAULT_THRESHOLD,
) -> List[int]:
    """Indices of candidates matching ``mode`` (see ``FILTER_*``)."""
    result = []
    for i, cand in enumerate(candidates):
        track_id = int(cand["track_id"])
        correction = corrections.get(track_id)
        if mode == FILTER_ALL:
            keep = True
        elif mode == FILTER_UNKNOWN:
            keep = cand.get("current_role") == "unknown"
        elif mode == FILTER_LOW_CONF:
            keep = float(cand.get("role_confidence", 0.0)) < threshold
        elif mode == FILTER_UNCORRECTED:
            keep = correction is None
        elif mode == FILTER_ID_SWITCH:
            keep = correction is not None and correction.id_switch
        else:
            keep = True
        if keep:
            result.append(i)
    return result


def summarize(
    candidates: List[dict], corrections: Dict[int, Correction]
) -> Dict[str, int]:
    """Counts for the summary panel."""
    total = len(candidates)
    corrected = 0
    by_role = {"player": 0, "referee": 0, "goalkeeper": 0,
               "ball": 0, "unknown": 0, IGNORE: 0}
    id_switches = 0
    for cand in candidates:
        correction = corrections.get(int(cand["track_id"]))
        if correction is None:
            continue
        corrected += 1
        if correction.role in by_role:
            by_role[correction.role] += 1
        if correction.id_switch:
            id_switches += 1
    return {
        "total": total,
        "corrected": corrected,
        "remaining": total - corrected,
        "players": by_role["player"],
        "referees": by_role["referee"],
        "goalkeepers": by_role["goalkeeper"],
        "ball": by_role["ball"],
        "unknown": by_role["unknown"],
        "ignored": by_role[IGNORE],
        "id_switches": id_switches,
    }


# ---------------------------------------------------------------------------
# Tk-free controller
# ---------------------------------------------------------------------------
class CorrectionSession:
    """Holds candidates + corrections and persists changes (no Tkinter).

    Navigation operates over the currently *filtered* view; the full
    candidate list and the corrections map are always available for the
    summary panel.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        corrections_path: Optional[str | Path] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = load_manifest(self.manifest_path)
        self.candidates: List[dict] = list(self.manifest.get("candidates", []))
        self.team_legend: Dict[int, dict] = _parse_team_legend(self.manifest)

        meta = self.manifest.get("metadata", {})
        self.threshold = float(
            meta.get("review_confidence_threshold", _DEFAULT_THRESHOLD)
        )
        resolved = (
            corrections_path
            or meta.get("corrections_file")
            or _DEFAULT_CORRECTIONS
        )
        self.store = CorrectionStore(resolved)
        self.corrections: Dict[int, Correction] = self.store.load()

        self.filter_mode = FILTER_ALL
        self._visible: List[int] = list(range(len(self.candidates)))
        self.pos = 0          # position within the filtered view
        self.frame_pos = 0    # selected sampled frame within the current track

    # -- filtering -----------------------------------------------------
    def set_filter(self, mode: str) -> None:
        if mode not in FILTER_MODES:
            raise ValueError(f"unknown filter mode '{mode}'")
        self.filter_mode = mode
        self._recompute_visible()
        self.pos = 0
        self.frame_pos = 0

    def _recompute_visible(self) -> None:
        self._visible = filter_indices(
            self.candidates, self.corrections, self.filter_mode, self.threshold
        )
        if self.pos >= len(self._visible):
            self.pos = max(len(self._visible) - 1, 0)

    @property
    def visible_candidates(self) -> List[dict]:
        return [self.candidates[i] for i in self._visible]

    # -- track navigation ----------------------------------------------
    @property
    def count(self) -> int:
        return len(self._visible)

    @property
    def index(self) -> int:
        return self.pos

    @property
    def current(self) -> Optional[dict]:
        if 0 <= self.pos < len(self._visible):
            return self.candidates[self._visible[self.pos]]
        return None

    def go_next(self) -> Optional[dict]:
        if self.pos < self.count - 1:
            self.pos += 1
            self.frame_pos = 0
        return self.current

    def go_prev(self) -> Optional[dict]:
        if self.pos > 0:
            self.pos -= 1
            self.frame_pos = 0
        return self.current

    def go_to(self, index: int) -> Optional[dict]:
        if 0 <= index < self.count:
            self.pos = index
            self.frame_pos = 0
        return self.current

    # -- frame navigation (within the current track) -------------------
    @property
    def frame_count(self) -> int:
        cand = self.current
        return len(cand.get("crop_paths", [])) if cand else 0

    def select_frame(self, frame_pos: int) -> int:
        if 0 <= frame_pos < self.frame_count:
            self.frame_pos = frame_pos
        return self.frame_pos

    def next_frame(self) -> int:
        if self.frame_pos < self.frame_count - 1:
            self.frame_pos += 1
        return self.frame_pos

    def prev_frame(self) -> int:
        if self.frame_pos > 0:
            self.frame_pos -= 1
        return self.frame_pos

    def jump_to_frame_number(self, frame_number: int) -> int:
        """Select the sampled frame whose frame id is nearest ``frame_number``."""
        cand = self.current
        if not cand or not cand.get("frame_ids"):
            return self.frame_pos
        frame_ids = cand["frame_ids"]
        nearest = min(range(len(frame_ids)),
                      key=lambda i: abs(frame_ids[i] - frame_number))
        self.frame_pos = nearest
        return self.frame_pos

    # -- corrections ---------------------------------------------------
    def existing(self, track_id: int) -> Optional[Correction]:
        return self.corrections.get(int(track_id))

    def save_correction(
        self,
        track_id: int,
        role: str,
        team_id: Optional[int] = None,
        id_switch: bool = False,
        switch_note: str = "",
        merge_with_track_id: Optional[int] = None,
        switch_frame: Optional[int] = None,
        player_name: Optional[str] = None,
    ) -> Correction:
        """Add/replace a correction (incl. ID-switch metadata) and persist."""
        name = (player_name or "").strip() or None
        correction = Correction(
            role=role,
            team_id=normalize_team(role, team_id),
            id_switch=bool(id_switch),
            switch_note=switch_note or "",
            merge_with_track_id=merge_with_track_id,
            switch_frame=switch_frame,
            player_name=name,
        )
        self.corrections[int(track_id)] = correction
        self.store.save(self.corrections)
        self._recompute_visible()
        return correction

    def delete_correction(self, track_id: int) -> bool:
        """Remove a correction (revert the track to its auto role)."""
        track_id = int(track_id)
        if track_id in self.corrections:
            del self.corrections[track_id]
            self.store.save(self.corrections)
            self._recompute_visible()
            return True
        return False

    def summary(self) -> Dict[str, int]:
        return summarize(self.candidates, self.corrections)


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------
class CorrectionApp:
    """Thin Tkinter front-end over a :class:`CorrectionSession`."""

    _THUMB = (72, 96)
    _BIG_CROP = (180, 220)
    _CONTEXT = (500, 220)
    _ACTION_FOOTER_BUTTONS = (
        ("Save Current", "_on_save"),
        ("Previous", "_on_prev"),
        ("Next", "_on_next"),
        ("Finish Review", "_on_finish"),
    )

    def __init__(self, session: CorrectionSession) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self._ttk = ttk
        self.session = session
        self.root = tk.Tk()
        self.root.title("⚽ Manual Role Correction (native)")
        self.root.geometry("1180x720")
        self.root.minsize(980, 640)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._photo_refs: list = []
        self._legend_photo_refs: list = []   # persistent; not cleared per track
        self._dirty = False
        self.confirmed = False

        # Friendly team dropdown options derived from the auto-generated legend.
        self._team_options = team_dropdown_options(session.team_legend)
        self._team_display_to_id = {disp: tid for disp, tid in self._team_options}
        self._team_displays = [disp for disp, _ in self._team_options]

        self.role_var = tk.StringVar(value="unknown")
        self.team_var = tk.StringVar(value=_NO_TEAM_LABEL)
        self.name_var = tk.StringVar(value="")
        self.switch_var = tk.BooleanVar(value=False)
        self.note_var = tk.StringVar(value="")
        self.merge_var = tk.StringVar(value="")
        self.switch_frame_var = tk.StringVar(value="")
        self.filter_var = tk.StringVar(value=FILTER_ALL)
        self.jump_var = tk.StringVar(value="")

        self.header_var = tk.StringVar(value="")
        self.progress_var = tk.StringVar(value="")
        self.info_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")

        self._build_widgets()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_current()

    # -- layout --------------------------------------------------------
    def _build_widgets(self) -> None:
        tk, ttk = self._tk, self._ttk

        top = ttk.Frame(self.root, padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, textvariable=self.header_var,
                  font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Label(top, textvariable=self.progress_var,
                  font=("Segoe UI", 11)).pack(side="right")

        body = ttk.Frame(self.root, padding=8)
        body.grid(row=1, column=0, sticky="nsew")

        # ---- left: images ----
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="Full-frame context (bbox highlighted)",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.context_label = ttk.Label(left)
        self.context_label.pack(anchor="w", pady=(2, 6))

        mid = ttk.Frame(left)
        mid.pack(anchor="w", fill="x")
        ttk.Label(mid, text="Selected crop", font=("Segoe UI", 10, "bold")).pack(
            anchor="w")
        self.big_crop_label = ttk.Label(mid)
        self.big_crop_label.pack(anchor="w", pady=2)

        ttk.Label(left, text="All sampled frames  (A / D to step)",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(6, 0))
        frame_nav = ttk.Frame(left)
        frame_nav.pack(anchor="w", pady=(2, 4))
        ttk.Button(frame_nav, text="Prev frame (A)",
                   command=self._on_prev_frame).pack(side="left")
        ttk.Button(frame_nav, text="Next frame (D)",
                   command=self._on_next_frame).pack(side="left", padx=4)
        strip_wrap = ttk.Frame(left)
        strip_wrap.pack(anchor="w", fill="x")
        self.strip_canvas = tk.Canvas(strip_wrap, height=self._THUMB[1] + 26,
                                      highlightthickness=0)
        hbar = ttk.Scrollbar(strip_wrap, orient="horizontal",
                             command=self.strip_canvas.xview)
        self.strip_canvas.configure(xscrollcommand=hbar.set)
        self.strip_canvas.pack(side="top", fill="x", expand=True)
        hbar.pack(side="top", fill="x")
        self.strip_inner = ttk.Frame(self.strip_canvas)
        self.strip_canvas.create_window((0, 0), window=self.strip_inner, anchor="nw")
        self.strip_inner.bind(
            "<Configure>",
            lambda e: self.strip_canvas.configure(
                scrollregion=self.strip_canvas.bbox("all")),
        )

        # ---- right: metadata + controls ----
        right = ttk.Frame(body, padding=(12, 0))
        right.pack(side="right", fill="y")

        ttk.Label(right, textvariable=self.info_var, justify="left",
                  font=("Consolas", 10)).pack(anchor="w", pady=(0, 8))

        ttk.Label(right, text="Role  (keys 1-6)",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        for i, role in enumerate(ROLE_CHOICES, start=1):
            ttk.Radiobutton(right, text=f"{i}  {role}", value=role,
                            variable=self.role_var,
                            command=self._mark_dirty).pack(anchor="w")

        ttk.Label(right, text="Team (players only)",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        ttk.OptionMenu(right, self.team_var, self.team_var.get(),
                       *self._team_displays,
                       command=lambda *_: self._mark_dirty()).pack(anchor="w", fill="x")

        self._build_legend_panel(right)

        ttk.Separator(right).pack(fill="x", pady=8)
        ttk.Label(right, text="Player name").pack(anchor="w")
        name_entry = ttk.Entry(right, textvariable=self.name_var, width=30)
        name_entry.pack(anchor="w")
        name_entry.bind("<KeyRelease>", lambda e: self._mark_dirty())

        ttk.Separator(right).pack(fill="x", pady=8)
        ttk.Checkbutton(right, text="ID switch observed", variable=self.switch_var,
                        command=self._mark_dirty).pack(anchor="w")
        ttk.Label(right, text="Switch notes").pack(anchor="w")
        note_entry = ttk.Entry(right, textvariable=self.note_var, width=30)
        note_entry.pack(anchor="w")
        note_entry.bind("<KeyRelease>", lambda e: self._mark_dirty())
        ttk.Label(right, text="Continue as track id").pack(anchor="w")
        merge_entry = ttk.Entry(right, textvariable=self.merge_var, width=12)
        merge_entry.pack(anchor="w")
        merge_entry.bind("<KeyRelease>", lambda e: self._mark_dirty())

        # Frame the identity switch happens at. The two tracks exchange ids
        # from here to the end, so each player keeps one id for the whole clip.
        ttk.Label(right, text="Switch at frame").pack(anchor="w")
        switchf = ttk.Frame(right)
        switchf.pack(anchor="w", fill="x")
        switch_entry = ttk.Entry(switchf, textvariable=self.switch_frame_var,
                                 width=8)
        switch_entry.pack(side="left")
        switch_entry.bind("<KeyRelease>", lambda e: self._mark_dirty())
        ttk.Button(switchf, text="Use current frame",
                   command=self._on_use_current_frame).pack(side="left", padx=4)

        ttk.Separator(right).pack(fill="x", pady=8)
        ttk.Label(right, text="Filter").pack(anchor="w")
        ttk.OptionMenu(right, self.filter_var, self.filter_var.get(), *FILTER_MODES,
                       command=self._on_filter).pack(anchor="w")

        jumpf = ttk.Frame(right)
        jumpf.pack(anchor="w", pady=(8, 0))
        ttk.Label(jumpf, text="Jump to frame#").pack(side="left")
        je = ttk.Entry(jumpf, textvariable=self.jump_var, width=8)
        je.pack(side="left", padx=4)
        je.bind("<Return>", lambda e: self._on_jump())
        ttk.Button(jumpf, text="Go", command=self._on_jump).pack(side="left")

        ttk.Button(right, text="Delete correction",
                   command=self._on_delete).pack(anchor="w", pady=(8, 0))

        ttk.Separator(right).pack(fill="x", pady=8)
        ttk.Label(right, text="Summary", font=("Segoe UI", 10, "bold")).pack(
            anchor="w")
        ttk.Label(right, textvariable=self.summary_var, justify="left",
                  font=("Consolas", 9)).pack(anchor="w")

        self._build_action_footer()

    def _build_action_footer(self) -> None:
        """Pinned bottom bar; never placed inside the main content area."""
        ttk = self._ttk
        footer = ttk.Frame(self.root, padding=(8, 6))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.action_footer = footer

        self.footer_status_label = ttk.Label(
            footer,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=4,
        )
        self.footer_status_label.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        buttons = ttk.Frame(footer)
        buttons.grid(row=0, column=1, sticky="e")
        self.footer_buttons_container = buttons
        self.footer_buttons = {}
        for text, handler_name in self._ACTION_FOOTER_BUTTONS:
            button = ttk.Button(buttons, text=text, command=getattr(self, handler_name))
            button.pack(side="left", padx=(0, 4))
            self.footer_buttons[text] = button

    def _build_legend_panel(self, parent) -> None:
        """Static panel: per-team swatch, auto color label, and 3 crops."""
        tk, ttk = self._tk, self._ttk
        legend = self.session.team_legend
        box = ttk.LabelFrame(parent, text="Team legend", padding=6)
        box.pack(anchor="w", fill="x", pady=(6, 0))
        if not legend:
            ttk.Label(box, text="(no team legend in manifest)").pack(anchor="w")
            return

        pil = self._pil()
        for team in (0, 1):
            info = legend.get(team)
            if not info:
                continue
            row = ttk.Frame(box)
            row.pack(anchor="w", fill="x", pady=3)
            header = ttk.Frame(row)
            header.pack(anchor="w", fill="x")
            swatch = info.get("swatch", "#808080")
            try:
                tk.Frame(header, width=18, height=18, background=swatch,
                         relief="solid", borderwidth=1).pack(side="left", padx=(0, 6))
            except tk.TclError:
                pass  # invalid color string -> skip the swatch, keep the label
            ttk.Label(header, text=f"Team {team} — {info.get('label', '')}",
                      font=("Segoe UI", 9, "bold")).pack(side="left")

            crops = ttk.Frame(row)
            crops.pack(anchor="w")
            crop_paths = info.get("crop_paths", [])[:3]
            if pil is not None and crop_paths:
                Image, _, ImageTk = pil
                for path in crop_paths:
                    p = Path(path)
                    if not p.is_file():
                        continue
                    img = Image.open(p)
                    img.thumbnail((44, 64))
                    photo = ImageTk.PhotoImage(img)
                    self._legend_photo_refs.append(photo)
                    ttk.Label(crops, image=photo).pack(side="left", padx=2)

    def _bind_keys(self) -> None:
        for key in KEY_ROLE_MAP:
            self.root.bind(key, self._on_role_key)
        self.root.bind("<Left>", lambda e: self._on_prev())
        self.root.bind("<Right>", lambda e: self._on_next())
        self.root.bind("a", lambda e: self._on_prev_frame())
        self.root.bind("d", lambda e: self._on_next_frame())
        self.root.bind("A", lambda e: self._on_prev_frame())
        self.root.bind("D", lambda e: self._on_next_frame())
        self.root.bind("<Control-s>", lambda e: self._on_save())
        self.root.bind("<Control-S>", lambda e: self._on_save())
        self.root.bind("<Control-Return>", lambda e: self._on_finish())
        self.root.bind("<Control-KP_Enter>", lambda e: self._on_finish())

    # -- state helpers -------------------------------------------------
    def _mark_dirty(self) -> None:
        self._dirty = True
        self._set_status()

    def _set_status(self, message: Optional[str] = None) -> None:
        state = "● Unsaved changes" if self._dirty else "✓ Saved"
        self.status_var.set(f"{state}    {message or ''}".rstrip())

    def _refresh_summary(self) -> None:
        s = self.session.summary()
        self.summary_var.set(
            f"total       : {s['total']}\n"
            f"corrected   : {s['corrected']}\n"
            f"remaining   : {s['remaining']}\n"
            f"players     : {s['players']}\n"
            f"referees    : {s['referees']}\n"
            f"goalkeepers : {s['goalkeepers']}\n"
            f"ignored     : {s['ignored']}\n"
            f"id switches : {s['id_switches']}"
        )
        self.progress_var.set(
            f"Track {self.session.index + 1}/{self.session.count}    "
            f"Corrected {s['corrected']}/{s['total']}"
        )

    # -- rendering -----------------------------------------------------
    def _show_current(self) -> None:
        self._photo_refs.clear()
        self._dirty = False
        cand = self.session.current

        self._refresh_summary()
        if cand is None:
            self.header_var.set("— no candidates in this filter —")
            self.info_var.set("")
            self._clear_images()
            self._set_status()
            return

        self.header_var.set(f"Track id {cand['track_id']}")
        self.info_var.set(
            f"track_id      : {cand['track_id']}\n"
            f"detected_class: {cand['detected_class']}\n"
            f"current_role  : {cand['current_role']}\n"
            f"confidence    : {cand['role_confidence']:.2f}\n"
            f"team_id       : {cand['team_id']}\n"
            f"track_length  : {cand['track_length']}\n"
            f"reason        : {cand.get('role_reason', '')}"
        )

        legend = self.session.team_legend
        existing = self.session.existing(cand["track_id"])
        if existing is not None:
            self.role_var.set(existing.role)
            self.team_var.set(team_display_for(legend, existing.team_id))
            self.name_var.set(existing.player_name or "")
            self.switch_var.set(existing.id_switch)
            self.note_var.set(existing.switch_note)
            self.merge_var.set(
                "" if existing.merge_with_track_id is None
                else str(existing.merge_with_track_id))
            self.switch_frame_var.set(
                "" if existing.switch_frame is None
                else str(existing.switch_frame))
        else:
            self.role_var.set(cand["current_role"]
                              if cand["current_role"] in ROLE_CHOICES else "unknown")
            self.team_var.set(team_display_for(legend, cand["team_id"]))
            self.name_var.set("")
            self.switch_var.set(False)
            self.note_var.set("")
            self.merge_var.set("")
            self.switch_frame_var.set("")

        self._render_strip(cand)
        self._render_selected(cand)
        self._set_status("(existing correction loaded)" if existing else "")

    def _clear_images(self) -> None:
        self.context_label.configure(image="")
        self.big_crop_label.configure(image="")
        for child in self.strip_inner.winfo_children():
            child.destroy()

    def _pil(self):
        try:
            from PIL import Image, ImageDraw, ImageTk
            return Image, ImageDraw, ImageTk
        except ImportError:
            return None

    def _render_strip(self, cand: dict) -> None:
        ttk = self._ttk
        for child in self.strip_inner.winfo_children():
            child.destroy()
        pil = self._pil()
        crop_paths = cand.get("crop_paths", [])
        frame_ids = cand.get("frame_ids", [])
        if not crop_paths:
            ttk.Label(self.strip_inner, text="(no crops saved)").pack()
            return
        if pil is None:
            ttk.Label(self.strip_inner,
                      text="Pillow not installed — pip install Pillow").pack()
            return
        Image, _, ImageTk = pil
        for idx, path in enumerate(crop_paths):
            cell = ttk.Frame(self.strip_inner, padding=2,
                             relief=("solid" if idx == self.session.frame_pos else "flat"),
                             borderwidth=(2 if idx == self.session.frame_pos else 0))
            cell.pack(side="left", padx=2)
            p = Path(path)
            if p.is_file():
                img = Image.open(p)
                img.thumbnail(self._THUMB)
                photo = ImageTk.PhotoImage(img)
                self._photo_refs.append(photo)
                lbl = ttk.Label(cell, image=photo, cursor="hand2")
                lbl.pack()
                lbl.bind("<Button-1>", lambda e, i=idx: self._on_thumb_click(i))
            else:
                ttk.Label(cell, text="(missing)").pack()
            fid = frame_ids[idx] if idx < len(frame_ids) else "?"
            ttk.Label(cell, text=f"f{fid}").pack()

    def _render_selected(self, cand: dict) -> None:
        pil = self._pil()
        if pil is None:
            return
        Image, ImageDraw, ImageTk = pil
        i = self.session.frame_pos
        crop_paths = cand.get("crop_paths", [])
        frame_paths = cand.get("frame_paths", [])
        bboxes = cand.get("bboxes", [])

        # Enlarged crop.
        if i < len(crop_paths) and Path(crop_paths[i]).is_file():
            crop = Image.open(crop_paths[i])
            crop.thumbnail(self._BIG_CROP)
            cphoto = ImageTk.PhotoImage(crop)
            self._photo_refs.append(cphoto)
            self.big_crop_label.configure(image=cphoto)
        else:
            self.big_crop_label.configure(image="")

        # Full-frame context with bbox drawn.
        fpath = frame_paths[i] if i < len(frame_paths) else ""
        if fpath and Path(fpath).is_file():
            frame_img = Image.open(fpath).convert("RGB")
            if i < len(bboxes) and len(bboxes[i]) == 4:
                draw = ImageDraw.Draw(frame_img)
                x1, y1, x2, y2 = bboxes[i]
                for w in range(3):  # thick rectangle
                    draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w],
                                   outline=(0, 255, 0))
            frame_img.thumbnail(self._CONTEXT)
            fphoto = ImageTk.PhotoImage(frame_img)
            self._photo_refs.append(fphoto)
            self.context_label.configure(image=fphoto)
        else:
            self.context_label.configure(image="")

    # -- event handlers ------------------------------------------------
    def _on_thumb_click(self, idx: int) -> None:
        self.session.select_frame(idx)
        self._render_strip(self.session.current)
        self._render_selected(self.session.current)

    def _on_role_key(self, event) -> None:
        role = role_for_key(event.char)
        if role:
            self.role_var.set(role)
            self._mark_dirty()

    def _on_prev(self) -> None:
        if not self._commit_before_track_change():
            return
        self.session.go_prev()
        self._show_current()

    def _on_next(self) -> None:
        if not self._commit_before_track_change():
            return
        self.session.go_next()
        self._show_current()

    def _on_prev_frame(self) -> None:
        self.session.prev_frame()
        self._render_strip(self.session.current)
        self._render_selected(self.session.current)

    def _on_next_frame(self) -> None:
        self.session.next_frame()
        self._render_strip(self.session.current)
        self._render_selected(self.session.current)

    def _on_jump(self) -> None:
        try:
            n = int(self.jump_var.get())
        except ValueError:
            self._set_status("Enter a frame number to jump to.")
            return
        self.session.jump_to_frame_number(n)
        self._render_strip(self.session.current)
        self._render_selected(self.session.current)

    def _on_filter(self, *_args) -> None:
        if not self._commit_before_track_change():
            return
        self.session.set_filter(self.filter_var.get())
        self._show_current()

    def _merge_value(self) -> Optional[int]:
        raw = self.merge_var.get().strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _switch_frame_value(self) -> Optional[int]:
        raw = self.switch_frame_var.get().strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _on_use_current_frame(self) -> None:
        """Record the currently-shown sampled frame as the switch frame."""
        cand = self.session.current
        frame_ids = cand.get("frame_ids", []) if cand else []
        pos = self.session.frame_pos
        if not frame_ids or not (0 <= pos < len(frame_ids)):
            self._set_status("No sampled frame selected.")
            return
        self.switch_frame_var.set(str(int(frame_ids[pos])))
        self._mark_dirty()
        self._set_status(f"Switch frame set to {frame_ids[pos]}")

    def _commit_current_form(self) -> bool:
        cand = self.session.current
        if cand is None:
            return False
        team_id = self._team_display_to_id.get(self.team_var.get())
        correction = self.session.save_correction(
            cand["track_id"],
            self.role_var.get(),
            team_id,
            id_switch=bool(self.switch_var.get()),
            switch_note=self.note_var.get().strip(),
            merge_with_track_id=self._merge_value(),
            switch_frame=self._switch_frame_value(),
            player_name=self.name_var.get(),
        )
        self._dirty = False
        self._refresh_summary()
        self._set_status("Saved current track")
        return True

    def _commit_before_track_change(self) -> bool:
        return True if self.session.current is None else self._commit_current_form()

    def _on_save(self) -> bool:
        return self._commit_current_form()

    def _on_save_next(self) -> None:
        if self._commit_current_form():           # auto-save, then advance
            self.session.go_next()
            self._show_current()

    def _on_delete(self) -> None:
        cand = self.session.current
        if cand is None:
            return
        if self.session.delete_correction(cand["track_id"]):
            self._dirty = False
            self._show_current()
            self._set_status(f"Deleted correction for track {cand['track_id']}")
        else:
            self._set_status("No correction to delete for this track.")

    def _on_finish(self) -> None:
        if self.session.current is not None and not self._commit_current_form():
            return
        self.session.corrections = self.session.store.load()
        self.confirmed = True
        self.root.destroy()

    def _on_close(self) -> None:
        if self._dirty:
            from tkinter import messagebox
            if not messagebox.askokcancel(
                "Unsaved changes",
                "You have unsaved changes. Close without saving?",
            ):
                return
        self.root.destroy()

    def run(self) -> bool:
        self.root.mainloop()
        return bool(self.confirmed)


# ===========================================================================
# Initialization mode (label boxes in a few frames as seeds)
# ===========================================================================
# Friendly button labels + box outline colors per init group.
INIT_GROUP_BUTTONS = [
    ("Team 0", initmod.TEAM_0),
    ("Team 1", initmod.TEAM_1),
    ("Referee", initmod.REFEREE),
    ("Goalkeeper", initmod.GOALKEEPER),
    ("Ignore", initmod.IGNORE),
]
INIT_GROUP_COLORS = {
    initmod.TEAM_0: "#3b82f6",      # blue
    initmod.TEAM_1: "#f59e0b",      # orange
    initmod.REFEREE: "#eab308",     # yellow
    initmod.GOALKEEPER: "#ef4444",  # red
    initmod.IGNORE: "#9ca3af",      # gray
}
_INIT_UNASSIGNED = "#ffffff"
_INIT_SELECTED = "#22c55e"          # green outline for the current selection


def load_init_manifest(path: str | Path) -> dict:
    """Load and validate an initialization manifest."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Initialization manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "frames" not in data:
        raise ValueError(f"{path} is not a valid initialization manifest")
    return data


class InitializationSession:
    """Tk-free controller for initialization labelling (unit-testable)."""

    def __init__(
        self,
        init_manifest_path: str | Path,
        labels_path: Optional[str | Path] = None,
    ) -> None:
        self.manifest_path = Path(init_manifest_path)
        self.manifest = load_init_manifest(self.manifest_path)
        self.frames: List[dict] = list(self.manifest.get("frames", []))
        meta = self.manifest.get("metadata", {})
        self.source_video = meta.get("source_video")
        self.source_tracks = meta.get("source_tracks")
        self.source_roles = meta.get("source_roles")

        resolved = (
            labels_path
            or meta.get("labels_file")
            or _DEFAULT_INIT_LABELS
        )
        self.store = initmod.InitializationStore(resolved)
        self.labels = self.store.load()
        self.pos = 0
        self.selected: set = set()

    # -- navigation ----------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.frames)

    @property
    def current(self) -> Optional[dict]:
        if 0 <= self.pos < len(self.frames):
            return self.frames[self.pos]
        return None

    @property
    def frame_number(self) -> Optional[int]:
        cand = self.current
        return int(cand["frame"]) if cand else None

    @property
    def tracks(self) -> List[dict]:
        cand = self.current
        return cand.get("tracks", []) if cand else []

    def go_to(self, index: int) -> Optional[dict]:
        if 0 <= index < self.count:
            self.pos = index
            self.selected = set()
        return self.current

    def go_next(self) -> Optional[dict]:
        return self.go_to(min(self.pos + 1, self.count - 1))

    def go_prev(self) -> Optional[dict]:
        return self.go_to(max(self.pos - 1, 0))

    def frame_numbers(self) -> List[int]:
        return [int(f["frame"]) for f in self.frames]

    def add_frame(self, entry: dict) -> int:
        """Append a (custom) frame entry; return its index."""
        existing = self.frame_numbers()
        if int(entry["frame"]) in existing:
            return existing.index(int(entry["frame"]))
        self.frames.append(entry)
        self.frames.sort(key=lambda f: int(f["frame"]))
        return self.frame_numbers().index(int(entry["frame"]))

    # -- selection -----------------------------------------------------
    def toggle_select(self, track_id: int) -> None:
        track_id = int(track_id)
        if track_id in self.selected:
            self.selected.discard(track_id)
        else:
            self.selected.add(track_id)

    def clear_selection(self) -> None:
        self.selected = set()

    # -- labelling -----------------------------------------------------
    def group_of(self, track_id: int) -> Optional[str]:
        if self.frame_number is None:
            return None
        return self.labels.group_of(self.frame_number, track_id)

    def assign_selected(self, group: Optional[str]) -> int:
        """Assign the current selection to ``group``; return how many."""
        if self.frame_number is None or not self.selected:
            return 0
        n = len(self.selected)
        self.labels.assign(self.frame_number, self.selected, group)
        self.clear_selection()
        return n

    def save(self) -> Path:
        return self.store.save(self.labels)

    def summary(self) -> Dict[str, int]:
        counts = {g: 0 for g in initmod.GROUPS}
        labelled_frames = 0
        for labels in self.labels.frames.values():
            if any(labels.get(g) for g in initmod.GROUPS):
                labelled_frames += 1
            for g in initmod.GROUPS:
                counts[g] += len(labels.get(g, []))
        counts["frames_labelled"] = labelled_frames
        return counts


class InitializationApp:
    """Canvas-based UI to label visible boxes in representative frames."""

    _MAX_W = 1040
    _MAX_H = 600

    def __init__(self, session: InitializationSession) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self._ttk = ttk
        self.session = session
        self.root = tk.Tk()
        self.root.title("⚽ Initialization — label boxes to seed teams/roles")
        self.root.geometry("1280x820")

        self._img_ref = None
        self._scale = 1.0
        self._box_items: Dict[int, int] = {}     # track_id -> canvas rect id
        self._scaled_bbox: Dict[int, tuple] = {}  # track_id -> (x1,y1,x2,y2)
        self._dirty = False

        self.status_var = tk.StringVar(value="")
        self.header_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self.frame_var = tk.StringVar(value="")
        self.custom_var = tk.StringVar(value="")
        self.autosave_var = tk.BooleanVar(value=False)

        self._build_widgets()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._render_frame()

    # -- layout --------------------------------------------------------
    def _build_widgets(self) -> None:
        tk, ttk = self._tk, self._ttk

        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.header_var,
                  font=("Segoe UI", 15, "bold")).pack(side="left")
        # Big, obvious Save button (top-right).
        self._big_save_top = tk.Button(
            top, text="💾  SAVE  (Ctrl+S)", command=self._on_save,
            bg="#16a34a", fg="white", font=("Segoe UI", 12, "bold"),
            padx=14, pady=6, relief="raised")
        self._big_save_top.pack(side="right")

        bar = ttk.Frame(self.root, padding=(10, 0))
        bar.pack(fill="x")
        ttk.Button(bar, text="◀ Prev frame", command=self._on_prev).pack(side="left")
        ttk.Button(bar, text="Next frame ▶", command=self._on_next).pack(
            side="left", padx=4)
        ttk.Label(bar, text="  Frame:").pack(side="left")
        self._frame_menu = ttk.OptionMenu(bar, self.frame_var, "", "",
                                          command=self._on_pick_frame)
        self._frame_menu.pack(side="left")
        ttk.Label(bar, text="   Custom frame#:").pack(side="left")
        ce = ttk.Entry(bar, textvariable=self.custom_var, width=8)
        ce.pack(side="left", padx=4)
        ce.bind("<Return>", lambda e: self._on_load_custom())
        ttk.Button(bar, text="Load", command=self._on_load_custom).pack(side="left")
        ttk.Checkbutton(bar, text="Autosave", variable=self.autosave_var).pack(
            side="right")

        body = ttk.Frame(self.root, padding=8)
        body.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(body, background="#111111",
                                highlightthickness=0, cursor="hand2")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        side = ttk.Frame(body, padding=(12, 0))
        side.pack(side="right", fill="y")
        ttk.Label(side, text="Assign selected boxes to:",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        for label, group in INIT_GROUP_BUTTONS:
            color = INIT_GROUP_COLORS[group]
            tk.Button(side, text=label, command=lambda g=group: self._on_assign(g),
                      bg=color, fg="black", width=16,
                      font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=2)
        ttk.Button(side, text="Clear selection",
                   command=self._on_clear).pack(anchor="w", pady=(6, 0))
        ttk.Button(side, text="Unassign selected",
                   command=lambda: self._on_assign(None)).pack(anchor="w")

        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Label(side, text="Legend", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        for label, group in INIT_GROUP_BUTTONS:
            row = ttk.Frame(side)
            row.pack(anchor="w")
            tk.Frame(row, width=14, height=14, background=INIT_GROUP_COLORS[group],
                     relief="solid", borderwidth=1).pack(side="left", padx=(0, 6))
            ttk.Label(row, text=label).pack(side="left")

        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Label(side, textvariable=self.summary_var, justify="left",
                  font=("Consolas", 9)).pack(anchor="w")
        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Button(side, text="✅ Apply as seeds → corrections",
                   command=self._on_apply_seeds).pack(anchor="w")

        bottom = ttk.Frame(self.root, padding=(10, 6))
        bottom.pack(fill="x", side="bottom")
        ttk.Label(bottom, textvariable=self.status_var, relief="sunken",
                  anchor="w", padding=4).pack(side="left", fill="x", expand=True)
        # Big, obvious Save button (bottom-right).
        tk.Button(bottom, text="💾  SAVE  (Ctrl+S)", command=self._on_save,
                  bg="#16a34a", fg="white", font=("Segoe UI", 12, "bold"),
                  padx=14, pady=6).pack(side="right")

    def _bind_keys(self) -> None:
        self.root.bind("<Control-s>", lambda e: self._on_save())
        self.root.bind("<Control-S>", lambda e: self._on_save())
        self.root.bind("<Left>", lambda e: self._on_prev())
        self.root.bind("<Right>", lambda e: self._on_next())
        self.root.bind("<Escape>", lambda e: self._on_clear())
        for i, (_, group) in enumerate(INIT_GROUP_BUTTONS, start=1):
            self.root.bind(str(i), lambda e, g=group: self._on_assign(g))

    # -- state ---------------------------------------------------------
    def _mark_dirty(self) -> None:
        self._dirty = True
        self._set_status()

    def _set_status(self, message: Optional[str] = None) -> None:
        state = "● Unsaved changes" if self._dirty else "✓ Saved successfully"
        self.status_var.set(f"{state}    {message or ''}".rstrip())

    def _refresh_summary(self) -> None:
        s = self.session.summary()
        self.summary_var.set(
            f"frames labelled : {s['frames_labelled']}/{self.session.count}\n"
            f"team 0          : {s[initmod.TEAM_0]}\n"
            f"team 1          : {s[initmod.TEAM_1]}\n"
            f"referee         : {s[initmod.REFEREE]}\n"
            f"goalkeeper      : {s[initmod.GOALKEEPER]}\n"
            f"ignore          : {s[initmod.IGNORE]}"
        )

    # -- rendering -----------------------------------------------------
    def _render_frame(self) -> None:
        self.canvas.delete("all")
        self._box_items.clear()
        self._scaled_bbox.clear()
        cand = self.session.current
        self._sync_frame_menu()
        self._refresh_summary()
        if cand is None:
            self.header_var.set("— no initialization frames —")
            self._set_status()
            return

        self.header_var.set(
            f"Frame {cand['frame']}   ({self.session.pos + 1}/{self.session.count})"
            f"   —  click boxes to select, then assign"
        )
        self.frame_var.set(str(cand["frame"]))

        from PIL import Image, ImageTk
        image = Image.open(cand["image"])
        w, h = image.size
        self._scale = min(self._MAX_W / w, self._MAX_H / h, 1.0)
        disp = image.resize((max(1, int(w * self._scale)),
                             max(1, int(h * self._scale))))
        self._img_ref = ImageTk.PhotoImage(disp)
        self.canvas.configure(width=disp.size[0], height=disp.size[1],
                              scrollregion=(0, 0, disp.size[0], disp.size[1]))
        self.canvas.create_image(0, 0, anchor="nw", image=self._img_ref)

        for track in cand.get("tracks", []):
            tid = int(track["track_id"])
            x1, y1, x2, y2 = (v * self._scale for v in track["bbox"])
            self._scaled_bbox[tid] = (x1, y1, x2, y2)
            rect = self.canvas.create_rectangle(x1, y1, x2, y2, width=2)
            self.canvas.create_text(x1 + 3, y1 + 8, anchor="w", text=str(tid),
                                    fill="#ffffff", font=("Segoe UI", 8, "bold"))
            self._box_items[tid] = rect
        self._refresh_boxes()
        self._set_status()

    def _refresh_boxes(self) -> None:
        """Recolor each box by its current group / selection state."""
        for tid, rect in self._box_items.items():
            if tid in self.session.selected:
                self.canvas.itemconfigure(rect, outline=_INIT_SELECTED, width=4)
            else:
                group = self.session.group_of(tid)
                color = INIT_GROUP_COLORS.get(group, _INIT_UNASSIGNED)
                self.canvas.itemconfigure(rect, outline=color, width=2)

    def _sync_frame_menu(self) -> None:
        menu = self._frame_menu["menu"]
        menu.delete(0, "end")
        for i, fn in enumerate(self.session.frame_numbers()):
            menu.add_command(
                label=str(fn),
                command=lambda v=str(fn), idx=i: self._select_frame_index(idx, v))

    # -- event handlers ------------------------------------------------
    def _select_frame_index(self, index: int, label: str) -> None:
        self.frame_var.set(label)
        self.session.go_to(index)
        self._render_frame()

    def _on_pick_frame(self, value) -> None:
        nums = self.session.frame_numbers()
        try:
            idx = nums.index(int(value))
        except (ValueError, TypeError):
            return
        self.session.go_to(idx)
        self._render_frame()

    def _on_canvas_click(self, event) -> None:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        hit = self._box_at(x, y)
        if hit is not None:
            self.session.toggle_select(hit)
            self._refresh_boxes()

    def _box_at(self, x: float, y: float) -> Optional[int]:
        """Smallest box containing the point (handles overlapping boxes)."""
        best, best_area = None, None
        for tid, (x1, y1, x2, y2) in self._scaled_bbox.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best_area is None or area < best_area:
                    best, best_area = tid, area
        return best

    def _on_assign(self, group: Optional[str]) -> None:
        n = self.session.assign_selected(group)
        if n:
            self._mark_dirty()
            name = group if group else "unassigned"
            self._set_status(f"Assigned {n} box(es) -> {name}")
            self._refresh_boxes()
            self._refresh_summary()
            if self.autosave_var.get():
                self._on_save()

    def _on_clear(self) -> None:
        self.session.clear_selection()
        self._refresh_boxes()

    def _on_prev(self) -> None:
        self.session.go_prev()
        self._render_frame()

    def _on_next(self) -> None:
        self.session.go_next()
        self._render_frame()

    def _on_load_custom(self) -> None:
        try:
            n = int(self.custom_var.get())
        except ValueError:
            self._set_status("Enter a valid frame number to load.")
            return
        if n in self.session.frame_numbers():
            self._select_frame_index(self.session.frame_numbers().index(n), str(n))
            return
        if not self.session.source_video or not self.session.source_tracks:
            self._set_status("Custom frame needs source_video/tracks in the manifest.")
            return
        from manual_correction.correction_runner import RoleCorrectionRunner
        from utils.config_loader import ManualCorrectionConfig

        runner = RoleCorrectionRunner(ManualCorrectionConfig())
        entry = runner.extract_init_frame(
            self.session.source_video, self.session.source_tracks, n,
            self.session.source_roles)
        if entry is None:
            self._set_status(f"Frame {n} has no tracks / could not be read.")
            return
        idx = self.session.add_frame(entry)
        self._select_frame_index(idx, str(n))

    def _on_save(self) -> None:
        path = self.session.save()
        self._dirty = False
        self._set_status(f"Saved to {path}")

    def _on_apply_seeds(self) -> None:
        self.session.save()
        self._dirty = False
        from manual_correction.correction_runner import RoleCorrectionRunner
        from manual_correction.initialization import (
            merge_seeds, propagate_initialization)

        seeds = propagate_initialization(self.session.labels)
        corrections_file = _DEFAULT_CORRECTIONS
        store = CorrectionStore(corrections_file)
        merged = merge_seeds(store.load(), seeds, overwrite=True)
        store.save(merged)
        n_switch = sum(1 for c in seeds.values() if c.id_switch)
        self._set_status(
            f"Applied {len(seeds)} seed(s) ({n_switch} ID-switch) -> {corrections_file}")

    def _on_close(self) -> None:
        if self._dirty:
            from tkinter import messagebox
            if not messagebox.askokcancel(
                "Unsaved changes",
                "You have unsaved initialization labels. Close without saving?",
            ):
                return
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="correction_tk_ui",
        description="Native Tkinter UI for manual role correction.",
    )
    parser.add_argument(
        "--manifest",
        help="Path to a review manifest JSON. If omitted, the latest manifest "
        f"under {_DEFAULT_REVIEW_DIR}/ is auto-detected.",
    )
    parser.add_argument(
        "--review-dir",
        default=_DEFAULT_REVIEW_DIR,
        help="Directory to auto-detect a manifest in (when --manifest is omitted).",
    )
    parser.add_argument(
        "--corrections",
        help="Path to the corrections JSON to write "
        f"(default: from the manifest, else {_DEFAULT_CORRECTIONS}).",
    )
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="Launch initialization mode (label boxes in a few frames as seeds) "
        "instead of per-track correction.",
    )
    return parser


def launch_initialization_from_review(
    review_manifest_path: str | Path,
    labels_path: Optional[str] = None,
) -> int:
    """Build an init dataset from a review manifest's sources and run the UI."""
    from manual_correction.correction_runner import RoleCorrectionRunner
    from utils.config_loader import ManualCorrectionConfig

    review = load_manifest(review_manifest_path)
    meta = review.get("metadata", {})
    video = meta.get("source_video")
    tracks = meta.get("source_tracks")
    roles = meta.get("source_roles")
    if not video or not tracks:
        raise SystemExit(
            "Review manifest lacks source_video/source_tracks; "
            "regenerate it with --manual-role-review."
        )
    config = ManualCorrectionConfig()
    if labels_path:
        config.initialization_labels_file = labels_path
    runner = RoleCorrectionRunner(config)
    info = runner.build_initialization_dataset(video, tracks, roles)
    print(f"Initialization frames: {info['frames']}")
    session = InitializationSession(info["manifest"], labels_path=labels_path)
    InitializationApp(session).run()
    return 0


def resolve_manifest(args) -> Path:
    if args.manifest:
        return Path(args.manifest)
    found = find_latest_manifest(args.review_dir)
    if found is None:
        raise SystemExit(
            f"No review manifest found under {args.review_dir}/. Run "
            "`--manual-role-review` first, or pass --manifest."
        )
    return found


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = resolve_manifest(args)
    if args.initialize:
        print(f"Initialization mode from: {manifest_path}")
        return launch_initialization_from_review(manifest_path, args.corrections)
    print(f"Loading manifest: {manifest_path}")
    session = CorrectionSession(manifest_path, corrections_path=args.corrections)
    print(f"{session.count} candidate(s); writing corrections to {session.store.path}")
    CorrectionApp(session).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
