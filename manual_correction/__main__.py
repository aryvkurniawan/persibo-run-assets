"""Standalone CLI for Phase 3.5 manual role correction.

Lets you run the review / apply steps directly against EXISTING Phase 2
tracks JSON and Phase 3 roles JSON, without re-running the (slow)
detection + tracking + refinement pipeline.

Run from inside the ``football_ai`` directory:

    # 1) Build the review dataset for ambiguous tracks, then it prints the
    #    Streamlit launch command:
    python -m manual_correction review --video input_video.mp4

    # 2) After reviewing & saving corrections in the UI, apply them:
    python -m manual_correction apply  --video input_video.mp4 --final-video

Defaults are derived from the video stem and the outputs directory, so the
common case needs only ``--video``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from manual_correction.correction_runner import RoleCorrectionRunner
from utils.config_loader import ManualCorrectionConfig
from utils.logger import setup_logging


def _resolve_tracks(out: Path, stem: str) -> Path:
    """Pick the most-processed tracks file (swapfixed > stitched > raw).

    The pipeline rewrites track ids during stitching/swap correction, so the
    review/initialize tools must label the SAME identities that ``--all`` and
    ``--reuse-tracks`` end up using -- otherwise the ids won't match.
    """
    for name in (f"{stem}_tracks_manualswap.json",
                 f"{stem}_tracks_swapfixed.json",
                 f"{stem}_tracks_stitched.json",
                 f"{stem}_tracks.json"):
        candidate = out / name
        if candidate.is_file():
            return candidate
    return out / f"{stem}_tracks.json"


def _defaults(args) -> tuple[Path, Path, Path, Path]:
    """Resolve tracks/roles/corrections/final paths from the video stem."""
    out = Path(args.output_dir)
    stem = Path(args.video).stem
    tracks = Path(args.tracks) if args.tracks else _resolve_tracks(out, stem)
    roles = Path(args.roles) if args.roles else out / f"{stem}_roles.json"
    corrections = (
        Path(args.corrections)
        if args.corrections
        else out / f"{stem}_role_corrections.json"   # per-video, no collisions
    )
    final = Path(args.final) if args.final else out / f"{stem}_roles_final.json"
    return tracks, roles, corrections, final


def _config(args) -> ManualCorrectionConfig:
    stem = Path(args.video).stem
    out = Path(args.output_dir)
    return ManualCorrectionConfig(
        enabled=True,
        review_confidence_threshold=args.threshold,
        review_dir=args.review_dir,
        corrections_file=str(_defaults(args)[2]),
        initialization_labels_file=str(out / f"{stem}_init_labels.json"),
    )


def cmd_review(args) -> int:
    tracks, roles, corrections, _ = _defaults(args)
    for label, path in (("tracks", tracks), ("roles", roles)):
        if not path.is_file():
            print(f"ERROR: {label} JSON not found: {path}", file=sys.stderr)
            print("Run the pipeline first (or pass --tracks/--roles).", file=sys.stderr)
            return 2

    runner = RoleCorrectionRunner(_config(args))
    ds = runner.build_review_dataset(args.video, tracks, roles)

    print("\n=== Manual review dataset ===")
    print(f"  candidates      : {ds['n_candidates']}")
    print(f"  review manifest : {ds['manifest']}")
    if ds["n_candidates"]:
        print("\nNow launch the review UI:")
        print(f'  $env:FOOTBALL_AI_REVIEW_MANIFEST="{ds["manifest"]}"')
        print(f'  $env:FOOTBALL_AI_CORRECTIONS_FILE="{corrections}"')
        print("  streamlit run manual_correction/correction_ui.py")
    else:
        print("  No ambiguous tracks — nothing to review.")
    return 0


def cmd_apply(args) -> int:
    tracks, roles, corrections, final = _defaults(args)
    if not roles.is_file():
        print(f"ERROR: roles JSON not found: {roles}", file=sys.stderr)
        return 2
    if not corrections.is_file():
        print(f"WARNING: no corrections file at {corrections}; "
              "every track will keep its automatic role.", file=sys.stderr)

    cfg = _config(args)
    cfg.save_final_video = args.final_video
    runner = RoleCorrectionRunner(cfg)
    # Auto-merge saved initialization labels (multi-select team seeds), but
    # only when they were made on THESE tracks: init labels are keyed to
    # track ids, so labels older than the tracks file are stale (a different
    # detection run) and would mislabel whoever now holds those ids.
    init_labels = Path(cfg.initialization_labels_file)
    if init_labels.is_file():
        fresh = (
            tracks.is_file()
            and init_labels.stat().st_mtime >= tracks.stat().st_mtime
        )
        if fresh:
            seeds = runner.apply_initialization(init_labels, corrections)
            print(f"Applied {seeds['n_seeds']} initialization seed(s) "
                  f"from {init_labels}")
        else:
            print(f"Skipped stale initialization labels ({init_labels} "
                  f"predates the tracks; re-label to apply)", file=sys.stderr)
    result = runner.run_apply(
        video_path=args.video,
        roles_json_path=roles,
        corrections_file=corrections,
        final_output_path=final,
        tracks_json_path=tracks if tracks.is_file() else None,
        render_video=args.final_video,
    )

    print("\n=== Final roles ===")
    for name, count in sorted(result["final_by_role"].items()):
        print(f"    {name:<12}: {count}")
    print(f"  tracks           : {result['n_tracks']}")
    print(f"  user-corrected   : {result['n_corrected']}")
    print(f"  final roles JSON : {result['final_roles_json']}")
    if result.get("final_video"):
        print(f"  final role video : {result['final_video']}")
    return 0


def cmd_swap(args) -> int:
    """Apply reviewer-marked ID-switch swaps and write corrected tracks JSON."""
    from manual_correction.correction_store import CorrectionStore
    from manual_correction.manual_swap import (
        build_manual_swaps,
        resolve_swap_source,
    )
    from role_refinement.role_refiner import load_frame_tracks
    from tracking.swap_corrector import apply_swaps
    from tracking.track_exporter import TrackJSONExporter

    out = Path(args.output_dir)
    stem = Path(args.video).stem
    _, _, corrections_path, _ = _defaults(args)
    if not corrections_path.is_file():
        print(f"ERROR: corrections file not found: {corrections_path}",
              file=sys.stderr)
        print("Run 'review' first and mark the ID switch(es).", file=sys.stderr)
        return 2

    source = resolve_swap_source(out, stem)
    if source is None:
        print(f"ERROR: no tracks JSON found in {out}", file=sys.stderr)
        return 2

    corrections = CorrectionStore(corrections_path).load()
    frames, _frame_size, meta = load_frame_tracks(str(source))
    last_frame = max((f.frame_index for f in frames), default=0)
    swaps = build_manual_swaps(corrections, last_frame)
    if not swaps:
        print("No manual ID-switch swaps to apply "
              "(need id_switch + 'continue as' + switch frame).")
        return 0

    corrected = apply_swaps(frames, swaps)
    out_path = out / f"{stem}_tracks_manualswap.json"
    fixed_meta = dict(meta)
    fixed_meta.update({"manual_swap": True, "manual_swaps": len(swaps)})
    exporter = TrackJSONExporter(out_path, metadata=fixed_meta)
    for frame in corrected:
        exporter.add_frame(frame.frame_index, frame.tracks)
    exporter.save()
    # Keep corrections newer than the tracks file so the apply step's
    # freshness gate still accepts them.
    CorrectionStore(corrections_path).save(corrections)

    print("\n=== Manual ID-switch swaps ===")
    print(f"  source tracks    : {source}")
    for a, b, s, e in swaps:
        print(f"    #{a} <-> #{b}  frames {s}-{e}")
    print(f"  corrected tracks : {out_path}")
    print("\nNow re-run the pipeline to pick up the corrected ids, e.g.:")
    print(f"  python main.py --video {args.video} --all --reuse-tracks")
    return 0


def cmd_initialize(args) -> int:
    tracks, roles, corrections, _ = _defaults(args)
    if not tracks.is_file():
        print(f"ERROR: tracks JSON not found: {tracks}", file=sys.stderr)
        return 2
    roles_arg = str(roles) if roles.is_file() else None

    cfg = _config(args)
    runner = RoleCorrectionRunner(cfg)
    custom = [int(f) for f in (args.frames or [])]
    info = runner.build_initialization_dataset(
        args.video, tracks, roles_arg, custom_frames=custom)

    print("\n=== Initialization dataset ===")
    print(f"  frames          : {info['frames']}")
    print(f"  init manifest   : {info['manifest']}")
    print(f"  labels file     : {info['labels_file']}")

    if args.no_ui:
        print("  (--no-ui) skipping the UI; launch it later with:")
        print("    python manual_correction/correction_tk_ui.py --initialize "
              f"--manifest <review_manifest>")
        return 0

    # Launch the canvas labelling UI.
    from manual_correction.correction_tk_ui import (
        InitializationApp,
        InitializationSession,
    )

    session = InitializationSession(
        info["manifest"], labels_path=cfg.initialization_labels_file)
    InitializationApp(session).run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m manual_correction",
        description="Phase 3.5 manual role correction (review / apply / "
        "initialize) over existing tracks + roles JSON.",
    )
    parser.add_argument("command",
                        choices=["review", "apply", "initialize", "swap"],
                        help="review = build dataset; apply = write final roles; "
                        "initialize = label boxes in a few frames as seeds; "
                        "swap = apply marked ID-switch swaps to the tracks")
    parser.add_argument("--video", required=True, help="Path to the source video")
    parser.add_argument("--output-dir", default="outputs",
                        help="Where the *_tracks.json / *_roles.json live")
    parser.add_argument("--tracks", help="Override tracks JSON path")
    parser.add_argument("--roles", help="Override roles JSON path")
    parser.add_argument("--corrections", help="Override corrections JSON path")
    parser.add_argument("--final", help="Override final roles JSON path")
    parser.add_argument("--review-dir", default="outputs/manual_review",
                        help="Where review crops/manifests are written")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Review tracks with confidence below this")
    parser.add_argument("--final-video", action="store_true",
                        help="(apply) render the final annotated video")
    parser.add_argument("--frames", nargs="*", type=int,
                        help="(initialize) extra custom frame numbers to label")
    parser.add_argument("--no-ui", action="store_true",
                        help="(initialize) build the dataset but don't launch the UI")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level="INFO", log_to_file=False)
    if args.command == "review":
        return cmd_review(args)
    if args.command == "initialize":
        return cmd_initialize(args)
    if args.command == "swap":
        return cmd_swap(args)
    return cmd_apply(args)


if __name__ == "__main__":
    sys.exit(main())
