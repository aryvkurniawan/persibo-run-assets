"""Open the manual pitch-calibration UI on the calibration-figure frame.

Click the visible pitch landmarks on frame 501 of ``input_video`` (select a
name on the right, then click its exact spot), then press Ctrl+S. The points
are saved to ``outputs/input_video_pitch_calibration.json`` and the thesis
figure generator will automatically use this hand-placed homography for
``05_pitch_keypoints_homography.png`` — giving a pixel-perfect overlay.

    python tools/calibrate_figure.py            # frame 501 (default)
    python tools/calibrate_figure.py --frame N  # any other frame

Afterwards, regenerate the figure:

    python tools/make_thesis_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibration.calibration_ui import (
    CalibrationApp,
    CalibrationSession,
    grab_frame,
)

VIDEO = ROOT / "input_video.mp4"
DEFAULT_FRAME = 501   # matches KP_FRAME in make_thesis_figures.py

# Landmarks clearly visible on frame 501 (right penalty box + centre area).
SUGGESTED = [
    "right_penalty_area_top", "right_penalty_area_bottom",
    "right_goal_area_top", "right_goal_area_bottom",
    "top_right_corner", "bottom_right_corner",
    "halfway_top_touchline", "halfway_bottom_touchline",
    "center_circle_top", "center_circle_bottom", "center_spot",
    "right_penalty_spot",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=DEFAULT_FRAME)
    args = ap.parse_args()

    out = ROOT / "outputs" / "input_video_pitch_calibration.json"
    print(f"Calibrating frame {args.frame} of {VIDEO.name}")
    print(f"Will save to {out}")
    print("\nClick these visible landmarks (select on the right, then click):")
    for n in SUGGESTED:
        print(f"   - {n}")
    print("\nClick as many as you clearly see (>= 6 recommended), then Ctrl+S.")
    print("Then run:  python tools/make_thesis_figures.py\n")

    frame = grab_frame(VIDEO, args.frame)
    if frame is None:
        print(f"Could not read frame {args.frame}", file=sys.stderr)
        return 2
    session = CalibrationSession(VIDEO, args.frame, str(out))
    CalibrationApp(session, frame).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
