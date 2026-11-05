"""Phase 3.5 orchestration: review dataset, override layer, final export.

Pipeline position::

    Detection -> Tracking -> Team Assignment -> Role Refinement
        -> Manual Role Correction -> Final Output

Two independent entry points:

* :meth:`RoleCorrectionRunner.build_review_dataset` — discover the tracks
  shown in the mandatory review/name-entry step, save representative
  crops/frames, and write a review manifest the UI consumes.
* :meth:`RoleCorrectionRunner.apply` / :meth:`run_apply` — overlay the
  reviewer's corrections on the automatic roles (manual > automatic > raw
  detector) and write ``outputs/<video>_roles_final.json``.

The pure decision pieces (:func:`find_candidate_ids`,
:func:`apply_corrections`) take plain data so they're unit-testable
without any video or files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from manual_correction.correction_models import (
    Correction,
    FinalRole,
    ReviewCandidate,
)
from role_refinement.role_models import ROLE_BALL, ROLE_PLAYER, RoleResult
from role_refinement.role_refiner import (
    _safe_crop,
    build_track_history,
    load_frame_tracks,
)
from role_refinement.role_visualizer import RoleVisualizer
from tracking.models import Track
from utils.config_loader import ManualCorrectionConfig
from utils.logger import get_logger

logger = get_logger("manual_correction.correction_runner")


# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------
def find_candidate_ids(
    auto_roles: Dict[int, RoleResult], config: ManualCorrectionConfig
) -> List[int]:
    """Track ids that warrant manual review.

    A track is a candidate when its automatic role is in
    ``config.review_roles`` (e.g. ``unknown``) OR its confidence is below
    ``config.review_confidence_threshold``. The ball is never reviewed.
    """
    review_roles = set(config.review_roles)
    threshold = config.review_confidence_threshold
    candidates = []
    for track_id, role in auto_roles.items():
        if role.refined_role == ROLE_BALL:
            continue
        if role.refined_role in review_roles or role.role_confidence < threshold:
            candidates.append(track_id)
    return sorted(candidates)


def find_review_ids(
    auto_roles: Dict[int, RoleResult], config: ManualCorrectionConfig
) -> List[int]:
    """All non-ball tracks to show in the mandatory review/name-entry stage."""
    ids = set(find_candidate_ids(auto_roles, config))
    for track_id, role in auto_roles.items():
        if role.refined_role != ROLE_BALL:
            ids.add(track_id)
    return sorted(ids)


def apply_corrections(
    auto_roles: Dict[int, RoleResult],
    corrections: Dict[int, Correction],
) -> Dict[int, FinalRole]:
    """Overlay corrections on automatic roles (manual > automatic).

    The automatic role is always preserved on the result, so the override
    is auditable and reversible. Corrections referencing tracks that don't
    exist in ``auto_roles`` are ignored with a warning.
    """
    finals: Dict[int, FinalRole] = {}
    for track_id, auto in auto_roles.items():
        correction = corrections.get(track_id)
        if correction is not None and not correction.is_noop:
            final_role = correction.role
            final_team = correction.team_id
            corrected = True
        else:
            final_role = auto.refined_role
            final_team = auto.team_id
            corrected = False
        # ``user_initialized`` reflects the reviewer's seed even for an
        # ``ignore`` no-op (they explicitly looked at and kept this track).
        initialized = correction.user_initialized if correction is not None else False
        player_name = correction.player_name if correction is not None else None
        finals[track_id] = FinalRole(
            track_id=track_id,
            detected_class=auto.detected_class,
            auto_role=auto.refined_role,
            final_role=final_role,
            role_confidence=auto.role_confidence,
            team_id=final_team,
            auto_team_id=auto.team_id,
            corrected_by_user=corrected,
            role_reason=auto.role_reason,
            user_initialized=initialized,
            player_name=player_name,
        )

    for track_id in corrections:
        if track_id not in auto_roles:
            logger.warning(
                "Correction for unknown track_id %d ignored (no auto role)",
                track_id,
            )
    return finals


def representative_frame_ids(observations: List[Track], n: int) -> List[int]:
    """Pick ``n`` representative frame ids (start, ..., end), evenly spaced."""
    frame_ids = [obs.frame_id for obs in observations]
    if not frame_ids:
        return []
    if len(frame_ids) <= n:
        return list(frame_ids)
    if n == 1:
        return [frame_ids[len(frame_ids) // 2]]
    step = (len(frame_ids) - 1) / (n - 1)
    idx = sorted({int(round(i * step)) for i in range(n)})
    return [frame_ids[i] for i in idx]


def load_auto_roles(roles_json_path: str | Path) -> Dict[int, RoleResult]:
    """Load Phase 3 roles JSON into ``{track_id: RoleResult}``."""
    path = Path(roles_json_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    roles: Dict[int, RoleResult] = {}
    for raw in payload.get("tracks", []):
        track_id = int(raw["track_id"])
        roles[track_id] = RoleResult(
            track_id=track_id,
            detected_class=str(raw.get("detected_class", "unknown")),
            refined_role=str(raw["refined_role"]),
            role_confidence=float(raw.get("role_confidence", 0.0)),
            team_id=(None if raw.get("team_id") is None else int(raw["team_id"])),
            role_reason=str(raw.get("role_reason", "")),
        )
    return roles


# ---------------------------------------------------------------------------
# Final-roles visualizer (re-uses Phase 3 colors, USER/AUTO label)
# ---------------------------------------------------------------------------
class FinalRoleVisualizer(RoleVisualizer):
    """Draws final roles, tagging manually-corrected tracks with ``USER``."""

    def __init__(self, finals: Dict[int, FinalRole], **kwargs) -> None:
        roles = {
            tid: RoleResult(
                track_id=tid,
                detected_class=f.detected_class,
                refined_role=f.final_role,
                role_confidence=f.role_confidence,
                team_id=f.team_id,
                role_reason=f.role_reason,
            )
            for tid, f in finals.items()
        }
        super().__init__(roles, **kwargs)
        self._finals = finals

    def label_for(self, track, result):  # type: ignore[override]
        final = self._finals.get(track.track_id)
        if final is None:
            return f"ID {track.track_id} | role: ?"
        label = (
            f"{final.player_name} (ID {track.track_id})"
            if final.player_name
            else final.player_display_name
        )
        label += f" | role: {final.final_role}"
        if final.final_role == ROLE_PLAYER and final.team_id is not None:
            label += f" (T{final.team_id})"
        tag = "USER" if final.corrected_by_user else "AUTO"
        return f"{label} | {tag}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class RoleCorrectionRunner:
    """Builds review datasets and applies corrections to produce final roles."""

    def __init__(self, config: ManualCorrectionConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Review dataset
    # ------------------------------------------------------------------
    def build_review_dataset(
        self,
        video_path: str | Path,
        tracks_json_path: str | Path,
        roles_json_path: str | Path,
    ) -> Dict:
        """Save crops/frames for every non-ball review track and write the manifest."""
        video_path = Path(video_path)
        stem = video_path.stem
        auto_roles = load_auto_roles(roles_json_path)
        frames, frame_size, _ = load_frame_tracks(tracks_json_path)
        history = build_track_history(frames)

        candidate_ids = find_review_ids(auto_roles, self._config)
        logger.info(
            "Found %d review track(s) out of %d tracks",
            len(candidate_ids),
            len(auto_roles),
        )

        review_root = Path(self._config.review_dir) / stem
        review_root.mkdir(parents=True, exist_ok=True)

        # Plan which (track, frame) crops we need, then collect in one pass.
        requests_by_frame: Dict[int, List[Tuple[int, Tuple]]] = {}
        rep_frames: Dict[int, List[int]] = {}
        for track_id in candidate_ids:
            observations = history.get(track_id, [])
            rep = representative_frame_ids(
                observations, self._config.representative_frames
            )
            rep_frames[track_id] = rep
            bbox_by_frame = {o.frame_id: o.bbox for o in observations}
            for frame_id in rep:
                requests_by_frame.setdefault(frame_id, []).append(
                    (track_id, bbox_by_frame[frame_id])
                )

        # Team legend: a few confident players per team, so the reviewer can
        # SEE what "Team 0" / "Team 1" look like instead of guessing from ids.
        legend_plan = self._plan_team_legend(auto_roles, history)
        for entries in legend_plan.values():
            for track_id, frame_id, bbox in entries:
                requests_by_frame.setdefault(frame_id, []).append((track_id, bbox))

        crop_paths, frame_paths = self._collect_images(
            video_path, requests_by_frame, review_root
        )

        candidates: List[ReviewCandidate] = []
        for track_id in candidate_ids:
            auto = auto_roles[track_id]
            observations = history.get(track_id, [])
            bbox_by_frame = {o.frame_id: o.bbox for o in observations}
            # Keep frame_ids / crop_paths / frame_paths / bboxes index-aligned:
            # only include a frame when its crop was actually saved.
            f_ids: List[int] = []
            c_paths: List[str] = []
            fr_paths: List[str] = []
            bxs: List[List[float]] = []
            for frame_id in rep_frames[track_id]:
                crop = crop_paths.get((track_id, frame_id))
                if crop is None:
                    continue
                f_ids.append(frame_id)
                c_paths.append(crop)
                fr_paths.append(frame_paths.get(frame_id, ""))
                bbox = bbox_by_frame.get(frame_id, (0.0, 0.0, 0.0, 0.0))
                bxs.append([round(float(v), 2) for v in bbox])
            candidates.append(
                ReviewCandidate(
                    track_id=track_id,
                    detected_class=auto.detected_class,
                    current_role=auto.refined_role,
                    role_confidence=auto.role_confidence,
                    team_id=auto.team_id,
                    track_length=len(observations),
                    frame_ids=f_ids,
                    crop_paths=c_paths,
                    frame_paths=fr_paths,
                    bboxes=bxs,
                    role_reason=auto.role_reason,
                )
            )

        team_legend = self._build_team_legend(legend_plan, crop_paths)

        manifest_path = Path(self._config.review_dir) / f"{stem}_review.json"
        manifest = {
            "metadata": {
                "phase": "manual_role_review",
                "source_video": str(video_path),
                "source_roles": str(roles_json_path),
                "source_tracks": str(tracks_json_path),
                "corrections_file": self._config.corrections_file,
                "frame_size": list(frame_size),
                "review_confidence_threshold": (
                    self._config.review_confidence_threshold
                ),
                "review_roles": list(self._config.review_roles),
                "review_scope": "all_non_ball_tracks",
                "ambiguous_candidate_count": len(
                    find_candidate_ids(auto_roles, self._config)),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            "team_legend": team_legend,
            "candidates": [c.to_dict() for c in candidates],
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        logger.info("Wrote review manifest (%d candidates) to %s",
                    len(candidates), manifest_path)

        return {
            "manifest": str(manifest_path),
            "n_candidates": len(candidates),
            "corrections_file": self._config.corrections_file,
            "candidates": candidates,
        }

    def _collect_images(
        self,
        video_path: Path,
        requests_by_frame: Dict[int, List[Tuple[int, Tuple]]],
        review_root: Path,
    ) -> Tuple[Dict[Tuple[int, int], str], Dict[int, str]]:
        """One sequential video pass: save jersey crops (+ optional frames)."""
        import cv2

        from utils.video_io import VideoReader

        crop_paths: Dict[Tuple[int, int], str] = {}
        frame_paths: Dict[int, str] = {}
        if not requests_by_frame:
            return crop_paths, frame_paths

        last_needed = max(requests_by_frame)
        with VideoReader(video_path) as reader:
            for frame_index, frame in reader.frames():
                reqs = requests_by_frame.get(frame_index)
                if reqs:
                    h, w = frame.shape[:2]
                    if self._config.save_review_frames:
                        fpath = review_root / f"frame_{frame_index:06d}.jpg"
                        cv2.imwrite(str(fpath), frame)
                        frame_paths[frame_index] = str(fpath)
                    for track_id, bbox in reqs:
                        crop = _safe_crop(frame, bbox, w, h)
                        if crop is None or crop.size == 0:
                            continue
                        cpath = (
                            review_root
                            / f"track_{track_id:04d}_f{frame_index:06d}.jpg"
                        )
                        cv2.imwrite(str(cpath), crop)
                        crop_paths[(track_id, frame_index)] = str(cpath)
                if frame_index >= last_needed:
                    break
        return crop_paths, frame_paths

    # ------------------------------------------------------------------
    # Team legend
    # ------------------------------------------------------------------
    def _plan_team_legend(
        self,
        auto_roles: Dict[int, RoleResult],
        history: Dict[int, List[Track]],
        per_team: int = 3,
    ) -> Dict[int, List[Tuple[int, int, Tuple]]]:
        """Pick up to ``per_team`` confident players per team for the legend.

        Returns ``team -> [(track_id, frame_id, bbox), ...]`` using each
        chosen player's middle frame as its representative crop.
        """
        players: Dict[int, List[Tuple[int, float, int]]] = {0: [], 1: []}
        for track_id, role in auto_roles.items():
            if role.refined_role == ROLE_PLAYER and role.team_id in (0, 1):
                players[role.team_id].append(
                    (track_id, role.role_confidence, len(history.get(track_id, [])))
                )

        plan: Dict[int, List[Tuple[int, int, Tuple]]] = {}
        for team, candidates in players.items():
            if not candidates:
                continue
            candidates.sort(key=lambda c: (c[1], c[2]), reverse=True)
            entries: List[Tuple[int, int, Tuple]] = []
            for track_id, _, _ in candidates[:per_team]:
                observations = history.get(track_id, [])
                mid = representative_frame_ids(observations, 1)
                if not mid:
                    continue
                frame_id = mid[0]
                bbox = next(
                    (o.bbox for o in observations if o.frame_id == frame_id), None
                )
                if bbox is not None:
                    entries.append((track_id, frame_id, bbox))
            if entries:
                plan[team] = entries
        return plan

    def _build_team_legend(
        self,
        legend_plan: Dict[int, List[Tuple[int, int, Tuple]]],
        crop_paths: Dict[Tuple[int, int], str],
    ) -> Dict[str, dict]:
        """Auto-generate a readable color label + swatch per team."""
        import cv2

        from manual_correction.team_legend import build_label_from_hsv_list
        from role_refinement.appearance_extractor import AppearanceExtractor
        from utils.config_loader import RoleRefinementConfig

        extractor = AppearanceExtractor(RoleRefinementConfig())
        legend: Dict[str, dict] = {}
        for team, entries in legend_plan.items():
            crops: List[str] = []
            hsv_list: List[Tuple[float, float, float]] = []
            for track_id, frame_id, _ in entries:
                path = crop_paths.get((track_id, frame_id))
                if not path:
                    continue
                crops.append(path)
                image = cv2.imread(path)
                if image is None:
                    continue
                hsv = extractor.dominant_hsv(image)
                if hsv is not None:
                    hsv_list.append((float(hsv[0]), float(hsv[1]), float(hsv[2])))
            info = build_label_from_hsv_list(hsv_list)
            legend[str(team)] = {
                "team_id": team,
                "label": info["label"],
                "color_rgb": info["color_rgb"],
                "swatch": info["swatch"],
                "crop_paths": crops,
            }
        return legend

    # ------------------------------------------------------------------
    # Apply corrections -> final roles
    # ------------------------------------------------------------------
    def apply(
        self,
        auto_roles: Dict[int, RoleResult],
        corrections: Dict[int, Correction],
    ) -> Dict[int, FinalRole]:
        """Pure override step (thin wrapper around :func:`apply_corrections`)."""
        return apply_corrections(auto_roles, corrections)

    def run_apply(
        self,
        video_path: str | Path,
        roles_json_path: str | Path,
        corrections_file: str | Path,
        final_output_path: str | Path,
        tracks_json_path: Optional[str | Path] = None,
        render_video: bool = False,
    ) -> Dict:
        """Load auto roles + corrections, write the final roles JSON (+ video)."""
        from manual_correction.correction_store import CorrectionStore

        auto_roles = load_auto_roles(roles_json_path)
        corrections = CorrectionStore(corrections_file).load()
        finals = self.apply(auto_roles, corrections)

        final_path = self.export_final(
            finals,
            final_output_path,
            metadata={
                "phase": "manual_role_correction",
                "source_roles": str(roles_json_path),
                "corrections_file": str(corrections_file),
                "priority": "manual > automatic > detector",
            },
        )

        video_out = None
        if render_video and tracks_json_path is not None:
            video_out = self._render_final_video(
                video_path, tracks_json_path, finals, Path(video_path).stem
            )

        return {
            "final_roles_json": str(final_path),
            "final_video": str(video_out) if video_out else None,
            "n_tracks": len(finals),
            "n_corrected": sum(1 for f in finals.values() if f.corrected_by_user),
            "n_named": sum(1 for f in finals.values() if f.player_name),
            "final_by_role": _count_final_roles(finals),
        }

    def export_final(
        self,
        finals: Dict[int, FinalRole],
        output_path: str | Path,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Write ``<video>_roles_final.json`` (sorted by track id)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata or {},
            "tracks": [finals[tid].to_dict() for tid in sorted(finals)],
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Exported %d final roles to %s", len(finals), path)
        return path

    def _render_final_video(self, video_path, tracks_json_path, finals, stem):
        from utils.video_io import VideoReader, VideoWriter

        frames, _, _ = load_frame_tracks(tracks_json_path)
        tracks_by_frame = {f.frame_index: f.tracks for f in frames}
        visualizer = FinalRoleVisualizer(finals)
        out_path = Path(self._config.review_dir).parent / f"{stem}_roles_final.mp4"
        with VideoReader(video_path) as reader:
            writer = VideoWriter(
                out_path,
                fps=reader.fps,
                frame_size=(reader.width, reader.height),
            )
            try:
                for frame_index, frame in reader.frames():
                    tracks = tracks_by_frame.get(frame_index, [])
                    writer.write(visualizer.annotate(frame, tracks))
            finally:
                writer.release()
        logger.info("Wrote final role video to %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # User-guided initialization (Phase 3.5)
    # ------------------------------------------------------------------
    def build_initialization_dataset(
        self,
        video_path: str | Path,
        tracks_json_path: str | Path,
        roles_json_path: Optional[str | Path] = None,
        custom_frames: Iterable[int] = (),
    ) -> Dict:
        """Pick representative frames and save them with all visible boxes.

        Writes ``<stem>_init.json`` describing each initialization frame
        (full-frame image + every visible track's bbox), which the
        initialization UI consumes.
        """
        from manual_correction.initialization import select_initialization_frames
        from role_refinement.role_models import ROLE_GOALKEEPER

        video_path = Path(video_path)
        stem = video_path.stem
        frames, frame_size, _ = load_frame_tracks(tracks_json_path)
        tracks_by_frame = {f.frame_index: f.tracks for f in frames}
        frames_present = list(tracks_by_frame.keys())

        auto_roles = (
            load_auto_roles(roles_json_path) if roles_json_path else {}
        )
        gk_ids = {
            tid for tid, r in auto_roles.items()
            if r.refined_role == ROLE_GOALKEEPER
        }
        gk_frames = [
            fi for fi, tracks in tracks_by_frame.items()
            if any(t.track_id in gk_ids for t in tracks)
        ]

        chosen = select_initialization_frames(
            frames_present, gk_frames, custom_frames
        )
        review_root = Path(self._config.review_dir) / stem
        review_root.mkdir(parents=True, exist_ok=True)
        images = self._grab_frames(video_path, chosen, review_root)

        frame_entries = []
        for frame_id in chosen:
            if frame_id not in images:
                continue
            frame_entries.append(
                self._init_frame_entry(
                    frame_id, tracks_by_frame.get(frame_id, []),
                    images[frame_id], auto_roles,
                )
            )

        manifest = {
            "metadata": {
                "phase": "manual_initialization",
                "source_video": str(video_path),
                "source_tracks": str(tracks_json_path),
                "source_roles": str(roles_json_path) if roles_json_path else None,
                "frame_size": list(frame_size),
                "labels_file": self._config.initialization_labels_file,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            "frames": frame_entries,
        }
        manifest_path = Path(self._config.review_dir) / f"{stem}_init.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        logger.info("Wrote initialization manifest (%d frames) to %s",
                    len(frame_entries), manifest_path)
        return {"manifest": str(manifest_path), "n_frames": len(frame_entries),
                "frames": chosen, "labels_file": self._config.initialization_labels_file}

    def extract_init_frame(
        self,
        video_path: str | Path,
        tracks_json_path: str | Path,
        frame_number: int,
        roles_json_path: Optional[str | Path] = None,
    ) -> Optional[dict]:
        """Build a single initialization-frame entry on demand (custom frame)."""
        video_path = Path(video_path)
        frames, _, _ = load_frame_tracks(tracks_json_path)
        tracks_by_frame = {f.frame_index: f.tracks for f in frames}
        if frame_number not in tracks_by_frame:
            return None
        auto_roles = (
            load_auto_roles(roles_json_path) if roles_json_path else {}
        )
        review_root = Path(self._config.review_dir) / video_path.stem
        review_root.mkdir(parents=True, exist_ok=True)
        images = self._grab_frames(video_path, [frame_number], review_root)
        if frame_number not in images:
            return None
        return self._init_frame_entry(
            frame_number, tracks_by_frame[frame_number],
            images[frame_number], auto_roles,
        )

    def _init_frame_entry(self, frame_id, tracks, image_path, auto_roles) -> dict:
        entry_tracks = []
        for t in tracks:
            auto = auto_roles.get(t.track_id)
            entry_tracks.append({
                "track_id": t.track_id,
                "bbox": [round(float(v), 2) for v in t.bbox],
                "auto_role": auto.refined_role if auto else t.class_name,
                "team_id": auto.team_id if auto else None,
            })
        return {"frame": int(frame_id), "image": image_path, "tracks": entry_tracks}

    @staticmethod
    def _grab_frames(
        video_path: Path, frame_numbers: Iterable[int], review_root: Path
    ) -> Dict[int, str]:
        """Random-access grab of specific 1-based frames; returns saved paths."""
        import cv2

        wanted = sorted({int(f) for f in frame_numbers})
        saved: Dict[int, str] = {}
        if not wanted:
            return saved
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Could not open video for init frames: %s", video_path)
            return saved
        try:
            for frame_number in wanted:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number - 1, 0))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                path = review_root / f"init_frame_{frame_number:06d}.jpg"
                cv2.imwrite(str(path), frame)
                saved[frame_number] = str(path)
        finally:
            cap.release()
        return saved

    def apply_initialization(
        self,
        labels_path: str | Path,
        corrections_file: str | Path,
        overwrite: bool = True,
    ) -> Dict:
        """Propagate initialization seeds into the corrections file."""
        from manual_correction.correction_store import CorrectionStore
        from manual_correction.initialization import (
            InitializationStore,
            merge_seeds,
            propagate_initialization,
        )

        labels = InitializationStore(labels_path).load()
        seeds = propagate_initialization(labels)
        store = CorrectionStore(corrections_file)
        merged = merge_seeds(store.load(), seeds, overwrite=overwrite)
        store.save(merged)

        n_switch = sum(1 for c in seeds.values() if c.id_switch)
        logger.info("Applied %d initialization seed(s) (%d flagged as ID switch)",
                    len(seeds), n_switch)
        return {
            "corrections_file": str(corrections_file),
            "n_seeds": len(seeds),
            "n_id_switch": n_switch,
            "n_corrections": len(merged),
        }


def _count_final_roles(finals: Dict[int, FinalRole]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for f in finals.values():
        counts[f.final_role] = counts.get(f.final_role, 0) + 1
    return counts
