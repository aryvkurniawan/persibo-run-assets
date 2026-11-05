"""Quantitative evaluation of the pipeline (Phase 7).

Two standard benchmarks, kept separate from the runtime pipeline so they
only ever *consume* its outputs (the trained weights, the exported tracks):

* :mod:`evaluation.detection_map` — detection accuracy as **mAP** (mean
  Average Precision) via Ultralytics' COCO-style validator on a held-out set.
* :mod:`evaluation.mota_eval` — multi-object **tracking** accuracy as
  **MOTA / MOTP / IDF1** via ``py-motmetrics``, comparing the exported
  tracks against a hand-annotated ground truth.

Both write a machine-readable JSON and a human-readable Markdown report into
``outputs/evaluation/`` so the numbers can drop straight into the thesis.
"""

from __future__ import annotations

__all__ = ["detection_map", "mota_eval", "mot_format"]
