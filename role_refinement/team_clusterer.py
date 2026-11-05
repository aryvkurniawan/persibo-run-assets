"""Discover the two main team appearance clusters and flag outliers.

The core Phase 3 idea: a team (and, by exclusion, a referee or
goalkeeper) is *not* defined by a fixed color. Instead we cluster the
appearance vectors of the stable player tracks, take the largest two
clusters as Team 1 and Team 2, and treat tracks that sit far from *both*
team centroids as outlier candidates (referee / goalkeeper).

Clustering is a small, dependency-free spherical (cosine) k-means so the
project keeps its numpy-only footprint — no scikit-learn. Cosine
distance is bounded in [0, 1] for the non-negative histogram features,
which is what makes ``outlier_distance_threshold`` interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from role_refinement.role_models import TrackAppearance
from utils.config_loader import RoleRefinementConfig
from utils.logger import get_logger

logger = get_logger("role_refinement.team_clusterer")


@dataclass
class ClusterResult:
    """Outcome of team clustering.

    Attributes:
        centroids: Unit-normalized team centroids, shape ``(k, D)``. Empty
            if clustering could not run.
        team_of: ``track_id -> team index`` for tracks assigned to a team
            (i.e. inliers within ``outlier_distance_threshold``).
        distance_of: ``track_id -> cosine distance`` to the nearest centroid.
        outliers: track ids whose nearest-centroid distance exceeds the
            outlier threshold (referee / goalkeeper candidates).
        team_sizes: number of inlier tracks per team index.
    """

    centroids: np.ndarray
    team_of: Dict[int, int] = field(default_factory=dict)
    distance_of: Dict[int, float] = field(default_factory=dict)
    outliers: List[int] = field(default_factory=list)
    team_sizes: Dict[int, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.centroids.size > 0

    def nearest(self, vector: np.ndarray) -> tuple[int, float]:
        """Return ``(team_index, cosine_distance)`` for an appearance vector."""
        if not self.ok or vector.size == 0:
            return -1, 1.0
        unit = _unit(vector.reshape(1, -1))
        sims = unit @ self.centroids.T
        idx = int(np.argmax(sims[0]))
        return idx, float(1.0 - sims[0, idx])


class TeamClusterer:
    """Spherical k-means over stable player appearance vectors."""

    def __init__(self, config: RoleRefinementConfig) -> None:
        self._config = config

    def cluster(
        self, appearances: Dict[int, TrackAppearance]
    ) -> ClusterResult:
        """Fit team centroids from stable, appearance-bearing person tracks.

        Tracks shorter than ``min_track_length`` or without a usable
        appearance vector are excluded from fitting (they cannot be
        trusted to anchor a team), but every track is still scored against
        the resulting centroids by the caller via :meth:`ClusterResult.nearest`.
        """
        stable = self._stable_tracks(appearances)
        k = self._config.team_cluster_count
        if len(stable) < k:
            logger.warning(
                "Not enough stable tracks (%d) for %d team clusters; "
                "skipping team clustering",
                len(stable),
                k,
            )
            return ClusterResult(centroids=np.empty((0, 0), dtype=np.float32))

        track_ids = [t.track_id for t in stable]
        matrix = _unit(np.stack([t.vector for t in stable], axis=0))

        centroids, labels = _spherical_kmeans(
            matrix, k=k, seed=self._config.random_seed
        )

        # One refinement pass: drop outliers, refit centroids on inliers
        # only, so a stray referee/keeper doesn't drag a team centroid.
        sims = matrix @ centroids.T
        nearest = np.argmax(sims, axis=1)
        nearest_dist = 1.0 - sims[np.arange(len(matrix)), nearest]
        inlier_mask = nearest_dist <= self._config.outlier_distance_threshold
        if inlier_mask.sum() >= k:
            centroids = _recompute_centroids(matrix[inlier_mask],
                                             nearest[inlier_mask], k, centroids)

        result = ClusterResult(centroids=centroids)
        for tid, vec in zip(track_ids, matrix):
            sims_row = vec @ centroids.T
            idx = int(np.argmax(sims_row))
            dist = float(1.0 - sims_row[idx])
            result.distance_of[tid] = dist
            if dist <= self._config.outlier_distance_threshold:
                result.team_of[tid] = idx
                result.team_sizes[idx] = result.team_sizes.get(idx, 0) + 1
            else:
                result.outliers.append(tid)

        logger.info(
            "Team clustering: %d stable tracks -> team sizes %s, %d outliers",
            len(stable),
            dict(sorted(result.team_sizes.items())),
            len(result.outliers),
        )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _stable_tracks(
        self, appearances: Dict[int, TrackAppearance]
    ) -> List[TrackAppearance]:
        ball = self._config.ball_class_name
        min_len = self._config.min_track_length
        return [
            a
            for a in appearances.values()
            if a.detected_class != ball
            and a.has_appearance
            and a.track_length >= min_len
        ]


# ---------------------------------------------------------------------------
# Numpy-only spherical k-means helpers
# ---------------------------------------------------------------------------
def _unit(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero rows stay zero (cosine sim 0 to everything)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def _kmeanspp_init(matrix: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    """k-means++ seeding using cosine distance."""
    n = matrix.shape[0]
    first = rng.randint(n)
    centers = [matrix[first]]
    for _ in range(1, k):
        sims = matrix @ np.stack(centers, axis=0).T
        # distance to the nearest chosen center
        dist = 1.0 - np.max(sims, axis=1)
        dist = np.clip(dist, 0, None)
        total = dist.sum()
        if total <= 0:
            centers.append(matrix[rng.randint(n)])
            continue
        probs = dist / total
        centers.append(matrix[rng.choice(n, p=probs)])
    return _unit(np.stack(centers, axis=0))


def _recompute_centroids(
    matrix: np.ndarray,
    labels: np.ndarray,
    k: int,
    previous: np.ndarray,
) -> np.ndarray:
    centers = []
    for j in range(k):
        members = matrix[labels == j]
        if len(members) == 0:
            centers.append(previous[j])  # keep the old centroid if orphaned
        else:
            centers.append(members.mean(axis=0))
    return _unit(np.stack(centers, axis=0))


def _spherical_kmeans(
    matrix: np.ndarray,
    k: int,
    seed: int,
    max_iter: int = 50,
    n_init: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine k-means with k-means++ restarts; returns (centroids, labels)."""
    best_centroids: Optional[np.ndarray] = None
    best_labels: Optional[np.ndarray] = None
    best_inertia = -np.inf  # maximize total cosine similarity to own centroid

    for init in range(n_init):
        rng = np.random.RandomState(seed + init)
        centroids = _kmeanspp_init(matrix, k, rng)
        labels = np.zeros(len(matrix), dtype=int)
        for _ in range(max_iter):
            sims = matrix @ centroids.T
            new_labels = np.argmax(sims, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            centroids = _recompute_centroids(matrix, labels, k, centroids)

        inertia = float(np.sum(matrix * centroids[labels]))
        if inertia > best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels

    assert best_centroids is not None and best_labels is not None
    return best_centroids, best_labels
