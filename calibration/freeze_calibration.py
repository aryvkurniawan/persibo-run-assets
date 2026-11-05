"""Analyse a video once and FREEZE a per-video homography by name.

Runs the pitch-keypoint detector across the video, keeps only the
well-conditioned frames (a degenerate / compressing homography is dropped),
and writes ``outputs/<stem>_homography.json`` (per-keyframe matrices). The
main pipeline then auto-loads that file by the video's name -- so the minimap
uses a fixed, analysed-once calibration instead of recomputing a shaky one
every run.

Usage:
    python -m calibration.freeze_calibration --video clasico_clip.mp4
    python -m calibration.freeze_calibration --video X --backend nbjw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from calibration.pitch_model import PitchDimensions
from utils.logger import get_logger

logger = get_logger("calibration.freeze")


def _well_conditioned(
    homography, frame_w: int, frame_h: int, dims: PitchDimensions = PitchDimensions()
) -> bool:
    """True if the homography maps the frame to a sane (non-collapsed) pitch area.

    Projects the four image corners to the field; a good homography yields a
    quad that spans a real chunk of the pitch in BOTH axes. A compressing /
    degenerate homography collapses it to a thin sliver -> rejected.
    """
    corners = [(0, 0), (frame_w, 0), (frame_w, frame_h), (0, frame_h)]
    pts = np.array([homography.project_image_to_field(c) for c in corners])
    if not np.all(np.isfinite(pts)):
        return False
    x_span = pts[:, 0].max() - pts[:, 0].min()
    y_span = pts[:, 1].max() - pts[:, 1].min()
    # The visible area should cover a decent width AND height of the pitch.
    return x_span >= 0.30 * dims.length and y_span >= 0.30 * dims.width


def freeze(video: str, backend: str, sample_every: int, out_path: str,
           device: str = "cpu") -> int:
    from calibration.pitch_keypoints import KeypointHomographyProvider
    from utils.video_io import VideoReader

    with VideoReader(video) as reader:
        frame_w, frame_h = reader.width, reader.height

    if backend == "nbjw":
        from calibration.nbjw_keypoints import NBJWPitchDetector
        detector = NBJWPitchDetector("weights/keypoints_best.pt.pt", device=device)
    else:
        detector = KeypointPitchDetector_default(device)

    provider = KeypointHomographyProvider.from_video(
        video, detector, sample_every=sample_every)
    if provider is None:
        print("No frames could be calibrated; nothing frozen.", file=sys.stderr)
        return 1

    kept = [(f, h) for f, h in provider._entries
            if _well_conditioned(h, frame_w, frame_h)]
    if not kept:
        print("Every calibrated frame was degenerate/compressed; nothing frozen.",
              file=sys.stderr)
        return 1

    payload = {
        "video": video,
        "frame_size": [frame_w, frame_h],
        "backend": backend,
        "frames": [{"frame": int(f), "matrix": h.matrix.tolist()} for f, h in kept],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload), encoding="utf-8")
    print(f"Froze {len(kept)}/{len(provider)} well-conditioned keyframes -> {out_path}")
    print("The pipeline will now auto-use this calibration for the video by name.")
    return 0


def KeypointPitchDetector_default(device):
    from calibration.pitch_keypoints import KeypointPitchDetector
    return KeypointPitchDetector(
        "weights/pitch_keypoints_yolo11m_best.pt.pt", device=device,
        min_conf=0.5, min_points=5, max_error_m=4.0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m calibration.freeze_calibration",
        description="Analyse a video and freeze its pitch homography by name.")
    p.add_argument("--video", required=True, help="Path to the source video")
    p.add_argument("--backend", choices=["yolo_pose", "nbjw"], default="yolo_pose",
                   help="Keypoint backend (yolo_pose default; better on broadcast)")
    p.add_argument("--sample-every", type=int, default=5,
                   help="Analyse every Nth frame")
    p.add_argument("--device", default="cpu", help="cpu or cuda:0")
    p.add_argument("--output", default=None,
                   help="Output JSON (default outputs/<stem>_homography.json)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out = args.output or f"outputs/{Path(args.video).stem}_homography.json"
    return freeze(args.video, args.backend, args.sample_every, out, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
