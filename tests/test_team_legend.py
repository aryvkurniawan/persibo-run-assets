"""Tests for the human-friendly team legend (color naming + labels)."""

from __future__ import annotations

from manual_correction.team_legend import (
    build_label_from_hsv_list,
    describe_hsv,
    hsv_to_rgb255,
    rgb_to_hex,
)


# ---------------------------------------------------------------------------
# Color naming (OpenCV HSV ranges: H 0-180, S/V 0-255)
# ---------------------------------------------------------------------------
def test_describe_hsv_achromatic():
    assert describe_hsv(0, 0, 255) == "white"     # bright, unsaturated
    assert describe_hsv(0, 0, 10) == "black"      # dark, unsaturated
    assert describe_hsv(0, 0, 120) == "gray"      # mid, unsaturated


def test_describe_hsv_hues():
    assert describe_hsv(0, 230, 230) == "red"
    assert describe_hsv(178, 230, 230) == "red"   # wraps around
    assert describe_hsv(60, 230, 200) == "green"
    assert describe_hsv(118, 230, 200) == "blue"
    assert describe_hsv(25, 230, 200) == "yellow"


def test_rgb_hex_roundtrip_helpers():
    assert rgb_to_hex((255, 0, 0)) == "#ff0000"
    assert rgb_to_hex((0, 128, 0)) == "#008000"
    # Pure green in OpenCV HSV is H=60.
    r, g, b = hsv_to_rgb255(60, 255, 255)
    assert g > r and g > b


# ---------------------------------------------------------------------------
# Label building
# ---------------------------------------------------------------------------
def test_build_label_single_dominant_color():
    greens = [(60, 220, 180), (62, 210, 190), (58, 230, 175)]
    info = build_label_from_hsv_list(greens)
    assert info["label"] == "mostly green kit"
    assert info["swatch"].startswith("#")
    assert len(info["color_rgb"]) == 3


def test_build_label_two_colors_white_black():
    # A white+black kit: half the crops read white, half black.
    samples = [(0, 10, 240), (0, 8, 235), (0, 10, 20), (0, 12, 25)]
    info = build_label_from_hsv_list(samples)
    assert info["label"].startswith("mostly ")
    assert "white" in info["label"] and "black" in info["label"]
    assert "/" in info["label"]


def test_build_label_empty_is_unknown():
    info = build_label_from_hsv_list([])
    assert info["label"] == "unknown kit"
    assert info["swatch"] == "#808080"


def test_build_label_ignores_rare_secondary_color():
    # 5 green, 1 stray red -> red is below the 1/3 threshold, dropped.
    samples = [(60, 220, 180)] * 5 + [(0, 230, 200)]
    info = build_label_from_hsv_list(samples)
    assert info["label"] == "mostly green kit"
