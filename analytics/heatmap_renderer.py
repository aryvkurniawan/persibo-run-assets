"""Phase 7 — paint a density grid over a top-down pitch and save it.

Reuses the Phase-4 :class:`~calibration.minimap.MinimapRenderer` ONLY to draw
the pitch background (stripes + lines) — it is not modified. The normalized
density grid is colour-mapped and alpha-blended with per-pixel opacity
proportional to density, so empty areas keep the pitch (lines stay visible)
and hot zones glow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from calibration.minimap import MinimapRenderer
from utils.logger import get_logger

logger = get_logger("analytics.heatmap_renderer")

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def render_heatmap(
    grid: np.ndarray,
    renderer: Optional[MinimapRenderer] = None,
    alpha: float = 0.65,
    colormap: int = cv2.COLORMAP_JET,
    label: str = "",
) -> np.ndarray:
    """Blend a 0..1 density ``grid`` onto a top-down pitch image (BGR)."""
    if renderer is None:
        renderer = MinimapRenderer(px_per_meter=9, stripes=12)
    pitch = renderer.blank()
    height, width = pitch.shape[:2]
    margin = renderer.margin
    field_w = max(width - 2 * margin, 1)
    field_h = max(height - 2 * margin, 1)

    resized = cv2.resize(
        grid.astype(np.float32), (field_w, field_h), interpolation=cv2.INTER_LINEAR)
    dens = np.zeros((height, width), dtype=np.float32)
    dens[margin:margin + field_h, margin:margin + field_w] = np.clip(resized, 0.0, 1.0)

    heat = cv2.applyColorMap((dens * 255).astype(np.uint8), colormap)
    opacity = (dens * float(alpha))[..., None]            # per-pixel, 0..alpha
    out = (pitch.astype(np.float32) * (1.0 - opacity)
           + heat.astype(np.float32) * opacity).astype(np.uint8)

    if label:
        org = (margin, max(margin - 8, 16))
        cv2.putText(out, label, (org[0] + 1, org[1] + 1), _FONT, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, org, _FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_heatmap(
    grid: np.ndarray,
    path: str | Path,
    renderer: Optional[MinimapRenderer] = None,
    alpha: float = 0.65,
    colormap: int = cv2.COLORMAP_JET,
    label: str = "",
) -> Path:
    """Render and write a heatmap PNG; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = render_heatmap(grid, renderer, alpha=alpha, colormap=colormap, label=label)
    cv2.imwrite(str(path), image)
    logger.info("Wrote heatmap to %s", path)
    return path
