"""Phase 4 — manual pitch calibration, homography and top-down minimap.

A post-processing layer: the user manually clicks pitch reference points
on a frame (avoiding automatic keypoint errors), which yields an
image<->field homography used to project tracked objects onto a top-down
minimap. It consumes only the exported tracks/roles JSON, so it never
touches detection / tracking / role-refinement / manual-correction.

Public surface (the Tk UI in :mod:`calibration_ui` is intentionally NOT
imported here, so the package doesn't require Pillow/Tk to import):

    PitchModel, PitchDimensions           — canonical pitch coordinates
    Calibration, CalibrationStore         — calibration data + JSON I/O
    Homography, compute_homography        — image<->field projection
    MinimapRenderer                       — top-down pitch drawing
"""

from calibration.calibration_store import (
    Calibration,
    CalibrationStore,
    MultiCalibration,
    MultiCalibrationStore,
    calibration_quality,
    point_in_frame,
)
from calibration.homography import (
    Homography,
    MultiHomography,
    compute_homography,
)
from calibration.minimap import (
    COLOR_MODES,
    MinimapRenderer,
    color_for,
    color_for_object,
    load_roles_map,
    overlay_geometry,
    overlay_minimap,
    project_tracks_to_field,
    render_final_video,
    render_overlay_video,
    track_id_color,
)
from calibration.pitch_keypoints import (
    PITCH_KEYPOINT_FIELD,
    KeypointHomographyProvider,
    KeypointPitchDetector,
    homography_from_keypoints,
)
from calibration.pitch_model import (
    REFERENCE_POINT_NAMES,
    PitchDimensions,
    PitchModel,
)

__all__ = [
    "PitchModel",
    "PitchDimensions",
    "REFERENCE_POINT_NAMES",
    "Calibration",
    "CalibrationStore",
    "MultiCalibration",
    "MultiCalibrationStore",
    "calibration_quality",
    "point_in_frame",
    "Homography",
    "MultiHomography",
    "compute_homography",
    "MinimapRenderer",
    "project_tracks_to_field",
    "load_roles_map",
    "color_for",
    "color_for_object",
    "track_id_color",
    "COLOR_MODES",
    "overlay_minimap",
    "overlay_geometry",
    "render_overlay_video",
    "render_final_video",
    "PITCH_KEYPOINT_FIELD",
    "homography_from_keypoints",
    "KeypointPitchDetector",
    "KeypointHomographyProvider",
]
