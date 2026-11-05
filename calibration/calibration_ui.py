"""Native Tkinter UI for manual pitch calibration (Phase 4).

The reviewer picks a named reference point from a list, then clicks its
location on a chosen video frame. Points accumulate into a calibration
that is saved to ``outputs/pitch_calibration.json`` and later used to build
the image<->field homography for the minimap.

Run it::

    python -m calibration.calibration_ui --video input_video.mp4

The decision logic lives in the Tk-free :class:`CalibrationSession` so it
is unit-testable without a display; :class:`CalibrationApp` is the GUI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Make the project root importable when launched as a bare script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from calibration.calibration_store import (  # noqa: E402
    Calibration,
    MultiCalibration,
    MultiCalibrationStore,
    calibration_quality,
    point_in_frame,
)
from calibration.homography import Homography  # noqa: E402
from calibration.pitch_model import PitchModel  # noqa: E402

_DEFAULT_CALIBRATION = "outputs/pitch_calibration.json"
_MIN_POINTS = 4
_RECOMMENDED_POINTS = 8

Point = Tuple[float, float]


def grab_frame(video_path: str | Path, frame_number: int) -> Optional[np.ndarray]:
    """Random-access read of a single 1-based frame; ``None`` on failure."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(frame_number) - 1, 0))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


class CalibrationSession:
    """Tk-free controller for multi-keyframe calibration point picking.

    Points are grouped per video frame ("keyframe"). Switching the frame
    switches which keyframe is being edited; ``self.calibration`` always
    points at the current frame's keyframe, while ``self.multi`` holds all
    of them. This lets the user calibrate several frames (one per goal /
    midfield) to handle a panning camera.
    """

    def __init__(
        self,
        video_path: str | Path,
        frame_number: int = 1,
        calibration_path: str | Path = _DEFAULT_CALIBRATION,
        pitch: Optional[PitchModel] = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.frame_number = int(frame_number)
        self.pitch = pitch or PitchModel()
        self.store = MultiCalibrationStore(calibration_path)

        self.multi = self.store.load() or MultiCalibration(video=self.video_path.name)
        self.frame_size: Tuple[int, int] = (0, 0)
        self.selected_name: str = self.pitch.names()[0]
        self.calibration: Calibration = self._keyframe_for(self.frame_number)
        self._order: List[str] = list(self.calibration.points.keys())

    # -- keyframes -----------------------------------------------------
    def _keyframe_for(self, frame: int) -> Calibration:
        for keyframe in self.multi.keyframes:
            if keyframe.frame == frame:
                return keyframe
        keyframe = Calibration(video=self.video_path.name, frame=int(frame))
        self.multi.keyframes.append(keyframe)
        return keyframe

    def keyframe_summary(self) -> List[Tuple[int, int]]:
        """``(frame, point_count)`` for every non-empty keyframe."""
        return [(k.frame, k.count) for k in self.multi.keyframes if k.count > 0]

    @property
    def total_points(self) -> int:
        return self.multi.total_points

    # -- names / selection ---------------------------------------------
    def names(self) -> List[str]:
        return self.pitch.names()

    def select(self, name: str) -> None:
        if self.pitch.contains(name):
            self.selected_name = name

    def has(self, name: str) -> bool:
        return name in self.calibration.points

    # -- frame ---------------------------------------------------------
    def set_frame_size(self, width: int, height: int) -> None:
        self.frame_size = (int(width), int(height))

    def set_frame_number(self, frame: int) -> None:
        """Switch to (or start) the keyframe for ``frame``."""
        self.frame_number = int(frame)
        self.calibration = self._keyframe_for(self.frame_number)
        self._order = list(self.calibration.points.keys())

    # -- points --------------------------------------------------------
    def add_point(self, image_xy: Point, name: Optional[str] = None) -> bool:
        """Assign an image click to a reference point. Rejects out-of-frame."""
        name = name or self.selected_name
        if not self.pitch.contains(name):
            return False
        w, h = self.frame_size
        if w and h and not point_in_frame(image_xy, w, h):
            return False
        field_xy = self.pitch.field_point(name)
        self.calibration.add_point(name, image_xy, field_xy)
        if name in self._order:
            self._order.remove(name)
        self._order.append(name)
        return True

    def remove_point(self, name: str) -> bool:
        removed = self.calibration.remove_point(name)
        if removed and name in self._order:
            self._order.remove(name)
        return removed

    def undo(self) -> Optional[str]:
        """Remove the most recently added point on the current keyframe."""
        if not self._order:
            return None
        name = self._order[-1]
        self.remove_point(name)
        return name

    @property
    def count(self) -> int:
        """Point count on the CURRENT keyframe."""
        return self.calibration.count

    # -- persistence ---------------------------------------------------
    def save(self) -> Path:
        self.multi.video = self.video_path.name
        # Drop empty keyframes so we never persist junk.
        self.multi.keyframes = [
            k for k in self.multi.keyframes
            if k.count > 0 or k.frame == self.frame_number
        ]
        return self.store.save(self.multi)

    def reload(self) -> bool:
        existing = self.store.load()
        if existing is None:
            return False
        self.multi = existing
        self.calibration = self._keyframe_for(self.frame_number)
        self._order = list(self.calibration.points.keys())
        return True

    # -- homography / quality ------------------------------------------
    def quality(self) -> dict:
        return calibration_quality(self.count, _MIN_POINTS, _RECOMMENDED_POINTS)

    def homography(self) -> Optional[Homography]:
        """Homography for the CURRENT keyframe (for live quality feedback)."""
        if self.count < _MIN_POINTS:
            return None
        try:
            return Homography.from_correspondences(
                self.calibration.image_points(), self.calibration.field_points())
        except ValueError:
            return None

    def reprojection_error(self) -> Optional[float]:
        homography = self.homography()
        return homography.reprojection_error() if homography else None


class CalibrationApp:
    """Tkinter canvas UI for clicking pitch reference points."""

    _MAX_W = 1100
    _MAX_H = 640

    def __init__(self, session: CalibrationSession, frame: np.ndarray) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self._ttk = ttk
        self.session = session
        self._frame = frame
        h, w = frame.shape[:2]
        self.session.set_frame_size(w, h)

        self.root = tk.Tk()
        self.root.title("⚽ Pitch Calibration — click reference points")
        self.root.geometry("1380x820")
        self._img_ref = None
        self._scale = 1.0
        self._dirty = False

        self.status_var = tk.StringVar(value="")
        self.quality_var = tk.StringVar(value="")

        self._build_widgets()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._render()

    # -- layout --------------------------------------------------------
    def _build_widgets(self) -> None:
        tk, ttk = self._tk, self._ttk

        top = ttk.Frame(self.root, padding=(10, 6))
        top.pack(fill="x")
        ttk.Label(top, text=f"Frame {self.session.frame_number} — "
                  f"select a point on the right, then click its spot",
                  font=("Segoe UI", 13, "bold")).pack(side="left")
        tk.Button(top, text="💾  SAVE  (Ctrl+S)", command=self._on_save,
                  bg="#16a34a", fg="white", font=("Segoe UI", 12, "bold"),
                  padx=12, pady=6).pack(side="right")

        body = ttk.Frame(self.root, padding=8)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, background="#111111",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_click)

        side = ttk.Frame(body, padding=(12, 0))
        side.pack(side="right", fill="y")
        ttk.Label(side, text="Reference point",
                  font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(side, height=20, width=28,
                                  exportselection=False)
        self.listbox.pack(anchor="w")
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        btns = ttk.Frame(side)
        btns.pack(anchor="w", pady=6)
        ttk.Button(btns, text="Undo (Ctrl+Z)", command=self._on_undo).pack(
            side="left")
        ttk.Button(btns, text="Delete point", command=self._on_delete).pack(
            side="left", padx=4)

        fr = ttk.Frame(side)
        fr.pack(anchor="w", pady=4)
        ttk.Label(fr, text="Frame #:").pack(side="left")
        self.frame_var = tk.StringVar(value=str(self.session.frame_number))
        fe = ttk.Entry(fr, textvariable=self.frame_var, width=8)
        fe.pack(side="left", padx=4)
        fe.bind("<Return>", lambda e: self._on_change_frame())
        ttk.Button(fr, text="Load frame", command=self._on_change_frame).pack(
            side="left")

        lb = ttk.Frame(side)
        lb.pack(anchor="w", pady=4)
        ttk.Button(lb, text="Load calibration", command=self._on_load).pack(
            side="left")

        ttk.Label(side, textvariable=self.quality_var, justify="left",
                  font=("Consolas", 9)).pack(anchor="w", pady=8)

        ttk.Label(self.root, textvariable=self.status_var, relief="sunken",
                  anchor="w", padding=4).pack(fill="x", side="bottom")

    def _bind_keys(self) -> None:
        self.root.bind("<Control-s>", lambda e: self._on_save())
        self.root.bind("<Control-S>", lambda e: self._on_save())
        self.root.bind("<Control-z>", lambda e: self._on_undo())
        self.root.bind("<Control-Z>", lambda e: self._on_undo())

    # -- rendering -----------------------------------------------------
    def _render(self) -> None:
        from PIL import Image, ImageTk

        self.canvas.delete("all")
        h, w = self._frame.shape[:2]
        rgb = self._frame[:, :, ::-1]
        image = Image.fromarray(rgb)
        self._scale = min(self._MAX_W / w, self._MAX_H / h, 1.0)
        disp = image.resize((max(1, int(w * self._scale)),
                             max(1, int(h * self._scale))))
        self._img_ref = ImageTk.PhotoImage(disp)
        self.canvas.configure(width=disp.size[0], height=disp.size[1])
        self.canvas.create_image(0, 0, anchor="nw", image=self._img_ref)

        for name, p in self.session.calibration.points.items():
            x, y = p["image"]
            sx, sy = x * self._scale, y * self._scale
            color = "#22c55e" if name == self.session.selected_name else "#ef4444"
            self.canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5,
                                    outline=color, width=2)
            self.canvas.create_text(sx + 7, sy, anchor="w", text=name,
                                    fill="#ffff00", font=("Segoe UI", 7))
        self._refresh_listbox()
        self._refresh_quality()

    def _refresh_listbox(self) -> None:
        self.listbox.delete(0, "end")
        for i, name in enumerate(self.session.names()):
            mark = "✓ " if self.session.has(name) else "   "
            self.listbox.insert("end", f"{mark}{name}")
            if name == self.session.selected_name:
                self.listbox.selection_clear(0, "end")
                self.listbox.selection_set(i)

    def _refresh_quality(self) -> None:
        q = self.session.quality()
        err = self.session.reprojection_error()
        lines = [
            f"FRAME {self.session.frame_number}",
            f"points (this frame): {q['count']}",
            f"minimum (4)     : {'OK' if q['usable'] else 'NOT MET'}",
            f"recommended (8) : {'OK' if q['recommended'] else 'no'}",
        ]
        if err is not None:
            lines.append(f"reproj error    : {err:.2f} m")
        if q["warning"]:
            lines.append(f"⚠ {q['warning']}")
        if err is not None and err > 5.0:
            lines.append("⚠ high reproj error — re-check points")
        lines.append("")
        lines.append("KEYFRAMES (frame: pts):")
        kf = self.session.keyframe_summary()
        lines.append(f"  {kf}" if kf else "  (none yet)")
        lines.append(f"total points: {self.session.total_points}")
        lines.append("Tip: calibrate one frame")
        lines.append("per goal + midfield, then")
        lines.append("change Frame# and repeat.")
        self.quality_var.set("\n".join(lines))

    # -- events --------------------------------------------------------
    def _on_select(self, _event) -> None:
        sel = self.listbox.curselection()
        if sel:
            self.session.select(self.session.names()[sel[0]])
            self._render()

    def _on_click(self, event) -> None:
        x = self.canvas.canvasx(event.x) / self._scale
        y = self.canvas.canvasy(event.y) / self._scale
        if self.session.add_point((x, y)):
            self._dirty = True
            self._advance_selection()
            self._render()
            self._set_status(f"Set {self.session.selected_name}")
        else:
            self._set_status("Point rejected (outside frame).")

    def _advance_selection(self) -> None:
        names = self.session.names()
        # Move to the next not-yet-calibrated point for a smooth workflow.
        start = names.index(self.session.selected_name)
        for offset in range(1, len(names) + 1):
            cand = names[(start + offset) % len(names)]
            if not self.session.has(cand):
                self.session.select(cand)
                return

    def _on_undo(self) -> None:
        name = self.session.undo()
        if name:
            self._dirty = True
            self.session.select(name)
            self._render()
            self._set_status(f"Removed {name}")

    def _on_delete(self) -> None:
        name = self.session.selected_name
        if self.session.remove_point(name):
            self._dirty = True
            self._render()
            self._set_status(f"Deleted {name}")

    def _on_change_frame(self) -> None:
        try:
            n = int(self.frame_var.get())
        except ValueError:
            self._set_status("Enter a valid frame number.")
            return
        frame = grab_frame(self.session.video_path, n)
        if frame is None:
            self._set_status(f"Could not read frame {n}.")
            return
        self._frame = frame
        self.session.set_frame_number(n)        # switch/start this keyframe
        h, w = frame.shape[:2]
        self.session.set_frame_size(w, h)
        self._render()
        kf = self.session.keyframe_summary()
        self._set_status(f"Frame {n} (keyframes so far: {kf})")

    def _on_save(self) -> None:
        if (self.session.count < _MIN_POINTS
                and not self.session.multi.usable_keyframes()):
            self._set_status(f"Need at least {_MIN_POINTS} points to save.")
            return
        path = self.session.save()
        self._dirty = False
        kf = self.session.keyframe_summary()
        self._set_status(
            f"Saved {len(kf)} keyframe(s) {kf}, {self.session.total_points} "
            f"points total to {path}")

    def _on_load(self) -> None:
        if self.session.reload():
            self._render()
            self._set_status("Loaded existing calibration.")
        else:
            self._set_status("No saved calibration to load.")

    def _set_status(self, message: str) -> None:
        state = "● Unsaved" if self._dirty else "✓ Saved"
        self.status_var.set(f"{state}    {message}")

    def _on_close(self) -> None:
        if self._dirty:
            from tkinter import messagebox
            if not messagebox.askokcancel(
                "Unsaved changes", "Close without saving the calibration?"):
                return
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m calibration.calibration_ui",
        description="Manual pitch calibration: click reference points on a frame.",
    )
    parser.add_argument("--video", required=True, help="Path to the source video")
    parser.add_argument("--frame", type=int, default=1,
                        help="Frame number to calibrate on (1-based)")
    parser.add_argument("--calibration", default=None,
                        help="Calibration JSON to load/save "
                        "(default: outputs/<video-name>_pitch_calibration.json)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    video = Path(args.video)
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        return 2
    # Save under the video's name by default, so the pipeline auto-loads this
    # video's calibration (no flags) on later runs.
    if not args.calibration:
        args.calibration = f"outputs/{video.stem}_pitch_calibration.json"
        print(f"Calibration will be saved to {args.calibration}")
    frame = grab_frame(video, args.frame)
    if frame is None:
        print(f"Could not read frame {args.frame} from {video}", file=sys.stderr)
        return 2
    session = CalibrationSession(video, args.frame, args.calibration)
    CalibrationApp(session, frame).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
