"""Export refined roles to ``outputs/<video>_roles.json``.

Output schema (Phase 3):

    {
      "metadata": {
        "phase": "role_refinement",
        "source_tracks": "...",
        "method": "appearance_clustering_temporal_voting"
      },
      "tracks": [
        {
          "track_id": 12,
          "detected_class": "player",
          "refined_role": "referee",
          "role_confidence": 0.87,
          "team_id": null,
          "role_reason": "appearance_outlier_central_motion"
        }
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional

from role_refinement.role_models import RoleResult
from utils.logger import get_logger

logger = get_logger("role_refinement.role_exporter")


class RoleJSONExporter:
    """Writes the per-track refined-role JSON."""

    def __init__(
        self, output_path: str | Path, metadata: Optional[Dict] = None
    ) -> None:
        self.output_path = Path(output_path)
        self.metadata: Dict = metadata or {}

    def save(self, roles: Mapping[int, RoleResult]) -> Path:
        """Serialize ``roles`` (sorted by track id) and return the path."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = [roles[tid].to_dict() for tid in sorted(roles)]
        payload = {"metadata": self.metadata, "tracks": ordered}
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info(
            "Exported %d refined roles to %s", len(ordered), self.output_path
        )
        return self.output_path
