"""Phase 3 orchestration: turn tracks into refined roles.

Two layers:

* :class:`RoleRefiner` — pure decision logic. Given per-track appearance
  summaries and the team-cluster result, it decides each track's role
  using temporal voting (team membership) and image-space trajectory
  analysis (referee vs goalkeeper). No I/O, fully unit-testable.

* :class:`RoleRefinementRunner` — I/O glue. Loads the tracking JSON,
  samples crops from the source video, runs extraction → clustering →
  refinement, writes the roles JSON and (optionally) an annotated video.

Role decisions are always multi-frame: team membership comes from a vote
across sampled frames, and referee/goalkeeper separation comes from the
whole track's position/motion distribution — never a single frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from role_refinement.appearance_extractor import (
    AppearanceExtractor,
    CropProvider,
    build_track_history,
)
from role_refinement.role_models import (
    ROLE_BALL,
    ROLE_GOALKEEPER,
    ROLE_PLAYER,
    ROLE_REFEREE,
    ROLE_UNKNOWN,
    RoleResult,
    TrackAppearance,
)
from role_refinement.team_clusterer import ClusterResult, TeamClusterer
from tracking.models import FrameTracks, Track
from utils.config_loader import RoleRefinementConfig
from utils.logger import get_logger

logger = get_logger("role_refinement.role_refiner")

# Trajectory-feature scaling for the referee/goalkeeper split. Image-space
# only — these are typical broadcast magnitudes, not pitch coordinates.
_SPREAD_SCALE = 0.20      # horizontal std (frac of width) that counts as "roams"
_COVERAGE_SCALE = 0.15    # trajectory-bbox area fraction that counts as "roams"
# Position (where the track lives) vs motion (how much it roams). Position
# is the primary cue: keepers sit at a horizontal extreme, referees stay
# central. Motion is a *modifier* — lots of roaming pushes toward referee,
# but little roaming is only weak evidence (a short clip has no motion at
# all), so it must never by itself force a goalkeeper verdict.
_W_POSITION = 0.6
_W_MOTION = 0.4
_MIN_RG_MARGIN = 0.08     # normalized margin below which the call is too close


class RoleRefiner:
    """Pure role-decision logic (no I/O)."""

    def __init__(self, config: RoleRefinementConfig) -> None:
        self._config = config

    def refine(
        self,
        appearances: Dict[int, TrackAppearance],
        cluster: ClusterResult,
    ) -> Dict[int, RoleResult]:
        """Decide a :class:`RoleResult` for every track."""
        results: Dict[int, RoleResult] = {}
        for track_id, appearance in appearances.items():
            result = self._refine_one(appearance, cluster)
            results[track_id] = apply_referee_prior(
                result,
                enabled=self._config.referee_detector_prior_enabled,
                threshold=self._config.referee_to_player_override_confidence,
            )
        if self._config.resolve_unknowns:
            results = self._resolve_unknowns(results, appearances, cluster)
        return results

    # ------------------------------------------------------------------
    # Final reconciliation: never ship a person track as "unknown"
    # ------------------------------------------------------------------
    def _resolve_unknowns(
        self,
        results: Dict[int, RoleResult],
        appearances: Dict[int, TrackAppearance],
        cluster: ClusterResult,
    ) -> Dict[int, RoleResult]:
        """Force every still-``unknown`` person track to its best guess.

        Refinement is deliberately cautious and leaves a track ``unknown``
        when no cue clears its confidence bar. That is exactly what made a
        fresh match come out with a handful of grey players needing a manual
        fix. This pass closes that gap automatically, cheapest cue first:

        1. **Color** — a track that still has a jersey signal is assigned to
           the team its sampled frames land nearest to (majority vote, no
           outlier gate). This catches the common short / fragmented track.
        2. **Position** — an *ambiguous outlier* (referee/keeper candidate the
           split could not call) is resolved by where it lives: a horizontal
           extreme reads goalkeeper, central reads referee.
        3. **Neighbors** — a colorless leftover inherits the team of the
           nearest already-labeled teammate, so it still appears on a side.

        Genuine referees and goalkeepers were already committed by
        :meth:`_split_outlier` and are untouched here. Manual / initialized
        corrections run downstream and still override any guess.
        """
        ball = self._config.ball_class_name
        out = dict(results)

        # Passes 1 & 2 — color, then ambiguous-outlier by position.
        for tid, res in results.items():
            if res.refined_role != ROLE_UNKNOWN:
                continue
            app = appearances.get(tid)
            if app is None or app.detected_class == ball:
                continue

            if res.role_reason == "appearance_outlier_ambiguous":
                if app.edge_proximity >= 0.5:
                    role, reason = ROLE_GOALKEEPER, "outlier_resolved_goal_side"
                else:
                    role, reason = ROLE_REFEREE, "outlier_resolved_central"
                out[tid] = RoleResult(
                    track_id=tid,
                    detected_class=app.detected_class,
                    refined_role=role,
                    role_confidence=round(float(max(res.role_confidence, 0.4)), 4),
                    team_id=None,
                    role_reason=reason,
                )
                continue

            team = self._nearest_team_majority(app, cluster)
            if team >= 0:
                out[tid] = RoleResult(
                    track_id=tid,
                    detected_class=app.detected_class,
                    refined_role=ROLE_PLAYER,
                    role_confidence=round(float(max(res.role_confidence, 0.4)), 4),
                    team_id=int(team),
                    role_reason="unknown_resolved_nearest_team",
                )

        # Pass 3 — colorless leftovers inherit the nearest teammate's side.
        anchors = [
            (appearances[t].mean_x_norm, appearances[t].mean_y_norm, r.team_id)
            for t, r in out.items()
            if r.refined_role == ROLE_PLAYER
            and r.team_id is not None
            and appearances.get(t) is not None
            and appearances[t].centers
        ]
        if anchors:
            for tid, res in out.items():
                if res.refined_role != ROLE_UNKNOWN:
                    continue
                app = appearances.get(tid)
                if app is None or app.detected_class == ball or not app.centers:
                    continue
                px, py = app.mean_x_norm, app.mean_y_norm
                _, _, team = min(
                    anchors, key=lambda q: (q[0] - px) ** 2 + (q[1] - py) ** 2
                )
                out[tid] = RoleResult(
                    track_id=tid,
                    detected_class=app.detected_class,
                    refined_role=ROLE_PLAYER,
                    role_confidence=0.3,
                    team_id=int(team),
                    role_reason="unknown_resolved_neighbor",
                )
        return out

    def _nearest_team_majority(
        self, appearance: TrackAppearance, cluster: ClusterResult
    ) -> int:
        """Team index this track's samples land nearest to, or -1 if none.

        Unlike the confidence-gated vote in :meth:`refine`, this ignores the
        outlier threshold: the goal is the *closest* team, not whether that
        closeness is strong enough to trust. Falls back to the mean vector
        when no per-sample vectors are available.
        """
        if not cluster.ok:
            return -1
        votes: Dict[int, int] = {}
        for sample in appearance.sample_vectors:
            idx, _ = cluster.nearest(sample)
            if idx >= 0:
                votes[idx] = votes.get(idx, 0) + 1
        if votes:
            return max(votes, key=votes.get)
        idx, _ = cluster.nearest(appearance.vector)
        return idx

    # ------------------------------------------------------------------
    # Per-track decision
    # ------------------------------------------------------------------
    def _refine_one(
        self, appearance: TrackAppearance, cluster: ClusterResult
    ) -> RoleResult:
        cfg = self._config

        # 1) Ball is never re-classified — pass it straight through.
        if appearance.detected_class == cfg.ball_class_name:
            return RoleResult(
                track_id=appearance.track_id,
                detected_class=appearance.detected_class,
                refined_role=ROLE_BALL,
                role_confidence=1.0,
                team_id=None,
                role_reason="ball_passthrough",
            )

        # 2) Need a usable appearance and a stable track to say anything.
        if not appearance.has_appearance:
            return self._unknown(appearance, "no_appearance_signal", 0.0)
        if appearance.track_length < cfg.min_track_length:
            return self._unknown(appearance, "short_track_low_confidence", 0.2)
        if not cluster.ok:
            return self._unknown(appearance, "clustering_unavailable", 0.0)

        # 3) Temporal vote: classify each sampled frame as a team or outlier.
        team_votes, outlier_votes = self._vote(appearance, cluster)
        n = appearance.n_samples
        outlier_fraction = outlier_votes / n if n else 0.0

        # Consistently far from BOTH teams -> referee / goalkeeper candidate.
        if outlier_fraction >= cfg.temporal_vote_threshold:
            return self._split_outlier(appearance)

        # Otherwise vote on a single team.
        if team_votes:
            best_team = max(team_votes, key=team_votes.get)
            team_fraction = team_votes[best_team] / n if n else 0.0
        else:
            best_team, team_fraction = -1, 0.0

        if team_fraction >= cfg.temporal_vote_threshold:
            return RoleResult(
                track_id=appearance.track_id,
                detected_class=appearance.detected_class,
                refined_role=ROLE_PLAYER,
                role_confidence=round(float(team_fraction), 4),
                team_id=int(best_team),
                role_reason="team_cluster_match",
            )

        # A long, stable, non-outlier track is a real player on one of the two
        # teams even when the vote was split (e.g. a white kit whose hue is
        # weak). Assign it to its leaning team rather than leaving it unknown.
        stable = appearance.track_length >= cfg.stable_min_length
        if cfg.assign_team_for_stable_players and best_team >= 0 and stable:
            return RoleResult(
                track_id=appearance.track_id,
                detected_class=appearance.detected_class,
                refined_role=ROLE_PLAYER,
                role_confidence=round(float(max(team_fraction, 0.5)), 4),
                team_id=int(best_team),
                role_reason="team_nearest_stable",
            )

        # Ambiguous: split between teams / partly outlier and no clear winner.
        if cfg.unknown_if_low_confidence:
            return self._unknown(
                appearance, "ambiguous_team_low_vote", float(team_fraction)
            )
        return RoleResult(
            track_id=appearance.track_id,
            detected_class=appearance.detected_class,
            refined_role=ROLE_PLAYER,
            role_confidence=round(float(team_fraction), 4),
            team_id=int(best_team) if best_team >= 0 else None,
            role_reason="team_cluster_match_weak",
        )

    # ------------------------------------------------------------------
    # Referee vs goalkeeper (image-space trajectory only — no homography)
    # ------------------------------------------------------------------
    def _split_outlier(self, appearance: TrackAppearance) -> RoleResult:
        edge = appearance.edge_proximity                       # 0 center .. 1 edge
        spread = _clip01(appearance.x_spread_norm / _SPREAD_SCALE)
        coverage = _clip01(appearance.coverage_norm / _COVERAGE_SCALE)
        roam = max(spread, coverage)                           # 0 still .. 1 roams

        # Position cue: keepers at an extreme (edge≈1), referees central.
        # Motion cue: roaming favors referee. With no roaming the motion term
        # is neutral, so the verdict falls back to position alone — a central
        # motionless outlier stays referee-leaning, not auto-goalkeeper.
        gk_score = _W_POSITION * edge + _W_MOTION * (1.0 - roam)
        ref_score = _W_POSITION * (1.0 - edge) + _W_MOTION * roam
        total = gk_score + ref_score
        margin = abs(gk_score - ref_score) / total if total > 0 else 0.0

        if margin < _MIN_RG_MARGIN and self._config.unknown_if_low_confidence:
            return self._unknown(
                appearance,
                "appearance_outlier_ambiguous",
                round(float(max(gk_score, ref_score) / total) if total else 0.0, 4),
            )

        confidence = max(gk_score, ref_score) / total if total else 0.0
        # Goalkeeper vs (assistant) referee — both can sit toward a side, so
        # position alone isn't enough; MOTION is the tell. A keeper barely
        # moves (low roam); a referee patrols and roams. So accept goalkeeper
        # when the outlier is either clearly AT the goal line (strong edge) OR
        # essentially still; a side outlier that ROAMS is the assistant referee.
        strong_edge = edge >= self._config.goalkeeper_min_edge
        still = roam <= self._config.goalkeeper_max_roam
        if gk_score >= ref_score and (strong_edge or still):
            role, reason = ROLE_GOALKEEPER, "appearance_outlier_goal_side"
        else:
            role, reason = ROLE_REFEREE, "appearance_outlier_central_motion"
        return RoleResult(
            track_id=appearance.track_id,
            detected_class=appearance.detected_class,
            refined_role=role,
            role_confidence=round(float(confidence), 4),
            team_id=None,
            role_reason=reason,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _vote(
        self, appearance: TrackAppearance, cluster: ClusterResult
    ) -> Tuple[Dict[int, int], int]:
        """Per-sample team/outlier votes for one track."""
        team_votes: Dict[int, int] = {}
        outlier_votes = 0
        threshold = self._config.outlier_distance_threshold
        for sample in appearance.sample_vectors:
            idx, dist = cluster.nearest(sample)
            if idx < 0 or dist > threshold:
                outlier_votes += 1
            else:
                team_votes[idx] = team_votes.get(idx, 0) + 1
        return team_votes, outlier_votes

    def _unknown(
        self, appearance: TrackAppearance, reason: str, confidence: float
    ) -> RoleResult:
        return RoleResult(
            track_id=appearance.track_id,
            detected_class=appearance.detected_class,
            refined_role=ROLE_UNKNOWN,
            role_confidence=round(float(confidence), 4),
            team_id=None,
            role_reason=reason,
        )


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def apply_referee_prior(
    result: RoleResult, enabled: bool, threshold: float
) -> RoleResult:
    """Trust the detector's "referee" over a low-confidence player flip.

    Assistant referees are sometimes detected correctly as ``referee`` but
    refined to ``player`` because their kit resembles a team. When that
    happens and the player decision wasn't highly confident, revert to
    referee. Only ``detected_class == "referee"`` tracks that refinement
    turned into a ``player`` are affected — ball, goalkeeper, and any other
    role are left untouched. Manual / user-initialized corrections are
    applied downstream and still override this prior.
    """
    if not enabled:
        return result
    if (
        result.detected_class == ROLE_REFEREE
        and result.refined_role == ROLE_PLAYER
        and result.role_confidence < threshold
    ):
        return RoleResult(
            track_id=result.track_id,
            detected_class=result.detected_class,
            refined_role=ROLE_REFEREE,
            role_confidence=result.role_confidence,
            team_id=None,
            role_reason="detector_referee_prior",
        )
    return result


# ---------------------------------------------------------------------------
# I/O glue
# ---------------------------------------------------------------------------
class VideoCropProvider:
    """Crops sampled bboxes out of a video in a single sequential pass.

    Only the requested ``(track_id, frame_id)`` crops are kept in memory,
    so peak memory is bounded by the number of samples, not the video.
    """

    def __init__(self, crops: Dict[Tuple[int, int], np.ndarray]) -> None:
        self._crops = crops

    def get(self, track_id, frame_id, bbox):
        return self._crops.get((track_id, frame_id))

    @classmethod
    def from_video(
        cls,
        video_path: str | Path,
        requests_by_frame: Dict[int, List[Tuple[int, Tuple]]],
    ) -> "VideoCropProvider":
        from utils.video_io import VideoReader

        crops: Dict[Tuple[int, int], np.ndarray] = {}
        if not requests_by_frame:
            return cls(crops)
        last_needed = max(requests_by_frame)
        with VideoReader(video_path) as reader:
            for frame_index, frame in reader.frames():
                requests = requests_by_frame.get(frame_index)
                if requests:
                    h, w = frame.shape[:2]
                    for track_id, bbox in requests:
                        crop = _safe_crop(frame, bbox, w, h)
                        if crop is not None:
                            crops[(track_id, frame_index)] = crop
                if frame_index >= last_needed:
                    break
        return cls(crops)


def _safe_crop(frame, bbox, w, h):
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(round(x1), w - 1)))
    y1 = int(max(0, min(round(y1), h - 1)))
    x2 = int(max(0, min(round(x2), w)))
    y2 = int(max(0, min(round(y2), h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


class RoleRefinementRunner:
    """End-to-end Phase 3: tracks JSON + video -> roles JSON (+ video)."""

    def __init__(self, config: RoleRefinementConfig) -> None:
        self._config = config
        self._extractor = AppearanceExtractor(config)
        self._clusterer = TeamClusterer(config)
        self._refiner = RoleRefiner(config)

    def run(
        self,
        video_path: str | Path,
        tracks_json_path: str | Path,
        roles_output_path: str | Path,
        role_video_path: Optional[str | Path] = None,
    ) -> Dict:
        """Refine roles for one tracked video and write the outputs."""
        video_path = Path(video_path)
        frames, frame_size, source_meta = load_frame_tracks(tracks_json_path)
        track_history = build_track_history(frames)

        roles = self.refine_history(track_history, frame_size, video_path)

        from role_refinement.role_exporter import RoleJSONExporter

        exporter = RoleJSONExporter(
            roles_output_path,
            metadata={
                "phase": "role_refinement",
                "source_tracks": str(tracks_json_path),
                "method": "appearance_clustering_temporal_voting",
                "source_video": str(video_path),
                "frame_size": list(frame_size),
                "params": {
                    "min_track_length": self._config.min_track_length,
                    "samples_per_track": self._config.samples_per_track,
                    "team_cluster_count": self._config.team_cluster_count,
                    "outlier_distance_threshold": (
                        self._config.outlier_distance_threshold
                    ),
                    "temporal_vote_threshold": (
                        self._config.temporal_vote_threshold
                    ),
                },
            },
        )
        roles_path = exporter.save(roles)

        role_video_out = None
        if role_video_path is not None:
            role_video_out = self._render_video(
                video_path, frames, roles, Path(role_video_path)
            )

        return {
            "roles_json": str(roles_path),
            "role_video": str(role_video_out) if role_video_out else None,
            "n_tracks": len(roles),
            "roles_by_type": _count_roles(roles),
        }

    def refine_history(
        self,
        track_history: Dict[int, List[Track]],
        frame_size: Tuple[int, int],
        video_path: str | Path,
    ) -> Dict[int, RoleResult]:
        """Extract appearance, cluster teams and refine roles."""
        sample_plan = self._extractor.sample_frames(track_history)
        requests_by_frame = _invert_plan(track_history, sample_plan, self._config)
        provider: CropProvider = VideoCropProvider.from_video(
            video_path, requests_by_frame
        )
        return self.refine_with_provider(
            track_history, sample_plan, provider, frame_size
        )

    def refine_with_provider(
        self,
        track_history: Dict[int, List[Track]],
        sample_plan: Dict[int, List[int]],
        provider: CropProvider,
        frame_size: Tuple[int, int],
    ) -> Dict[int, RoleResult]:
        """Same as :meth:`refine_history` but with an injected crop provider."""
        appearances = self._extractor.extract(
            track_history, sample_plan, provider, frame_size
        )
        cluster = self._clusterer.cluster(appearances)
        return self._refiner.refine(appearances, cluster)

    # ------------------------------------------------------------------
    def _render_video(self, video_path, frames, roles, out_path):
        from role_refinement.role_visualizer import RoleVisualizer
        from utils.video_io import VideoReader, VideoWriter

        tracks_by_frame = {f.frame_index: f.tracks for f in frames}
        visualizer = RoleVisualizer(roles)
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
        logger.info("Wrote role video to %s", out_path)
        return out_path


# ---------------------------------------------------------------------------
# Tracks-JSON loading / sample-plan inversion
# ---------------------------------------------------------------------------
def load_frame_tracks(
    tracks_json_path: str | Path,
) -> Tuple[List[FrameTracks], Tuple[int, int], dict]:
    """Reconstruct :class:`FrameTracks` from a Phase 2 tracks JSON file."""
    path = Path(tracks_json_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    meta = payload.get("metadata", {})
    resolution = meta.get("resolution") or [1, 1]
    frame_size = (int(resolution[0]), int(resolution[1]))

    # class_name -> class_id is informational here; tracking carries names.
    frames: List[FrameTracks] = []
    for raw_frame in payload.get("frames", []):
        frame_index = int(raw_frame.get("frame", 0))
        tracks: List[Track] = []
        for raw in raw_frame.get("tracks", []):
            bbox = tuple(float(v) for v in raw["bbox"])
            tracks.append(
                Track(
                    frame_id=frame_index,
                    track_id=int(raw["track_id"]),
                    class_id=-1,
                    class_name=str(raw["class"]),
                    confidence=float(raw.get("confidence", 0.0)),
                    bbox=bbox,  # type: ignore[arg-type]
                )
            )
        frames.append(FrameTracks(frame_index=frame_index, tracks=tracks))
    return frames, frame_size, meta


def _invert_plan(
    track_history: Dict[int, List[Track]],
    sample_plan: Dict[int, List[int]],
    config: RoleRefinementConfig,
) -> Dict[int, List[Tuple[int, Tuple]]]:
    """Build ``frame_id -> [(track_id, bbox), ...]`` from a sample plan.

    Ball tracks are skipped: their appearance is never used.
    """
    requests: Dict[int, List[Tuple[int, Tuple]]] = {}
    for track_id, frame_ids in sample_plan.items():
        observations = track_history.get(track_id, [])
        if observations and observations[0].class_name == config.ball_class_name:
            continue
        bbox_by_frame = {obs.frame_id: obs.bbox for obs in observations}
        for frame_id in frame_ids:
            bbox = bbox_by_frame.get(frame_id)
            if bbox is None:
                continue
            requests.setdefault(frame_id, []).append((track_id, bbox))
    return requests


def _count_roles(roles: Dict[int, RoleResult]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for result in roles.values():
        counts[result.refined_role] = counts.get(result.refined_role, 0) + 1
    return counts
