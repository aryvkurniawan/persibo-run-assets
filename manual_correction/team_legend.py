"""Human-friendly team legend generation.

Turns the (already-decided) team appearance into something a non-technical
reviewer can read: a color name and a swatch, e.g. *"mostly white/black
kit"*. This is purely descriptive — it labels teams the clustering already
discovered, it does NOT classify anyone by color.

All functions here are pure (stdlib only, no OpenCV/numpy), so they're
trivially unit-testable. The caller supplies dominant HSV samples (one per
representative crop), computed upstream with the same jersey-region logic
used by role refinement.
"""

from __future__ import annotations

import colorsys
from typing import List, Optional, Tuple

# OpenCV HSV ranges: H in 0..180, S in 0..255, V in 0..255.
HSV = Tuple[float, float, float]

_SAT_GRAY = 35      # below this saturation a pixel reads as white/gray/black
_VAL_WHITE = 175    # above this value (and low sat) -> white
_VAL_BLACK = 60     # below this value (and low sat) -> black


def describe_hsv(h: float, s: float, v: float) -> str:
    """Map a single dominant HSV color to a plain color name."""
    if s < _SAT_GRAY:
        if v >= _VAL_WHITE:
            return "white"
        if v <= _VAL_BLACK:
            return "black"
        return "gray"
    if h < 10 or h >= 170:
        return "red"
    if h < 20:
        return "orange"
    if h < 33:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    return "purple"


def hsv_to_rgb255(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert an OpenCV-range HSV color to an 8-bit RGB tuple."""
    r, g, b = colorsys.hsv_to_rgb((h * 2.0) / 360.0, s / 255.0, v / 255.0)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = (max(0, min(255, int(c))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_label_from_hsv_list(hsv_list: List[HSV]) -> dict:
    """Summarize per-crop dominant colors into a team legend entry.

    Returns ``{"label", "color_rgb", "swatch"}``. The label names the one or
    two most common colors (so a white+black kit reads "white/black"); the
    swatch is a real observed color from the dominant group.
    """
    if not hsv_list:
        gray = (128, 128, 128)
        return {"label": "unknown kit", "color_rgb": list(gray),
                "swatch": rgb_to_hex(gray)}

    labelled = [(describe_hsv(*hsv), hsv) for hsv in hsv_list]

    # Frequency of each color name, preserving first-seen order for ties.
    order: List[str] = []
    counts: dict = {}
    for name, _ in labelled:
        if name not in counts:
            counts[name] = 0
            order.append(name)
        counts[name] += 1
    ranked = sorted(order, key=lambda n: (-counts[n], order.index(n)))

    top = ranked[0]
    names = [top]
    # Include a second color only if it's clearly present (>= 1/3 of samples).
    if len(ranked) > 1 and counts[ranked[1]] >= max(1, len(hsv_list) // 3):
        names.append(ranked[1])

    # Swatch: a representative observed color from the dominant group
    # (median by value, avoiding fragile circular-hue averaging).
    top_group = sorted((hsv for name, hsv in labelled if name == top),
                       key=lambda hsv: hsv[2])
    rep = top_group[len(top_group) // 2]
    rgb = hsv_to_rgb255(*rep)

    return {
        "label": f"mostly {'/'.join(names)} kit",
        "color_rgb": list(rgb),
        "swatch": rgb_to_hex(rgb),
    }
