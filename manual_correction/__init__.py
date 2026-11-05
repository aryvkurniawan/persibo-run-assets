"""Phase 3.5 — human-in-the-loop role correction.

Runs after automatic role refinement. Surfaces ONLY ambiguous tracks
(role ``unknown`` or low confidence) for manual review, stores the
reviewer's choices separately, and overlays them on the automatic results
to produce a final roles JSON. Automatic results are never overwritten and
corrections are reversible.

Two review UIs are provided (both optional, neither imported here so the
package never hard-requires their deps):
    * :mod:`correction_ui`     — Streamlit (`streamlit run ...`)
    * :mod:`correction_tk_ui`  — native Tkinter (faster; stdlib + Pillow)

Public surface:

    ReviewCandidate, Correction, FinalRole   — data models
    CorrectionStore                          — corrections JSON load/save
    RoleCorrectionRunner                     — review dataset + override layer
    find_candidate_ids, apply_corrections    — pure decision helpers
"""

from manual_correction.correction_models import (
    CORRECTION_ROLES,
    IGNORE,
    Correction,
    FinalRole,
    ReviewCandidate,
)
from manual_correction.correction_runner import (
    RoleCorrectionRunner,
    apply_corrections,
    find_candidate_ids,
    load_auto_roles,
)
from manual_correction.correction_store import CorrectionStore
from manual_correction.initialization import (
    InitializationLabels,
    InitializationStore,
    propagate_initialization,
    select_initialization_frames,
)

__all__ = [
    "ReviewCandidate",
    "Correction",
    "FinalRole",
    "CORRECTION_ROLES",
    "IGNORE",
    "CorrectionStore",
    "RoleCorrectionRunner",
    "find_candidate_ids",
    "apply_corrections",
    "load_auto_roles",
    "InitializationLabels",
    "InitializationStore",
    "propagate_initialization",
    "select_initialization_frames",
]
