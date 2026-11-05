"""Load and persist manual role corrections.

The corrections file is the single source of truth for human input and is
deliberately separate from the automatic results, so:

* automatic results are never overwritten, and
* a correction is reversible — delete an entry (or the whole file) and the
  track reverts to its automatic role.

On-disk format (``outputs/manual_role_corrections.json``):

    {
      "1":   {"role": "player",  "team_id": 0},
      "107": {"role": "referee", "team_id": null},
      "119": {"role": "referee", "team_id": null}
    }

Keys are track ids as strings (JSON object keys must be strings).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from manual_correction.correction_models import Correction
from utils.logger import get_logger

logger = get_logger("manual_correction.correction_store")


class CorrectionStore:
    """Reads/writes the manual corrections JSON, keyed by integer track id."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    def load(self) -> Dict[int, Correction]:
        """Return ``{track_id: Correction}``; empty dict if the file is absent.

        Malformed individual entries are skipped with a warning rather than
        aborting the whole load — a half-finished review should still apply.
        """
        if not self.path.is_file():
            logger.info("No corrections file at %s (using auto roles only)", self.path)
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError(
                f"Corrections file {self.path} must be a JSON object, "
                f"got {type(raw).__name__}"
            )

        corrections: Dict[int, Correction] = {}
        for key, value in raw.items():
            try:
                track_id = int(key)
                corrections[track_id] = Correction.from_dict(value)
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Skipping invalid correction for '%s': %s", key, exc)
        logger.info("Loaded %d corrections from %s", len(corrections), self.path)
        return corrections

    def save(self, corrections: Dict[int, Correction]) -> Path:
        """Write the corrections map to disk (sorted by track id)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            str(track_id): corrections[track_id].to_dict()
            for track_id in sorted(corrections)
        }
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Saved %d corrections to %s", len(payload), self.path)
        return self.path

    # ------------------------------------------------------------------
    # Convenience mutators (used by the UI)
    # ------------------------------------------------------------------
    def upsert(
        self, track_id: int, role: str, team_id: Optional[int] = None
    ) -> Dict[int, Correction]:
        """Set/replace one correction and persist the whole file."""
        corrections = self.load()
        corrections[int(track_id)] = Correction(role=role, team_id=team_id)
        self.save(corrections)
        return corrections

    def remove(self, track_id: int) -> Dict[int, Correction]:
        """Delete one correction (reverting that track to its auto role)."""
        corrections = self.load()
        corrections.pop(int(track_id), None)
        self.save(corrections)
        return corrections
