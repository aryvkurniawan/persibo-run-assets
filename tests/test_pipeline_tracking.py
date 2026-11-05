"""Pipeline wiring tests for detection + tracking.

A fake tracker is injected so we can assert the pipeline produces BOTH
the detection JSON (unchanged from Phase 1) and the tracking JSON, plus
an annotated video — without any real tracking maths or GPU.
"""

import json

import cv2
import numpy as np

from detection.detector import YOLODetector
from detection.pipeline import DetectionPipeline
from tests.fakes import FakeTracker, FakeYOLOModel
from utils.config_loader import AppConfig, ModelConfig, VideoConfig

CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}


def create_test_video(path, n_frames=5, size=(160, 120), fps=20.0):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path


def make_pipeline(tmp_path, detections, tracker):
    config = AppConfig(
        model=ModelConfig(device="cpu"),
        classes=CLASS_NAMES,
        video=VideoConfig(output_dir=str(tmp_path / "outputs"), output_codec="MJPG"),
    )
    detector = YOLODetector(
        config.model, config.classes, model=FakeYOLOModel(detections)
    )
    return DetectionPipeline(config, detector=detector, tracker=tracker), config


def test_pipeline_with_tracking_writes_both_jsons(tmp_path):
    video = create_test_video(tmp_path / "clip.avi", n_frames=5)
    tracker = FakeTracker()
    pipeline, _ = make_pipeline(
        tmp_path,
        detections=[
            (10, 10, 40, 80, 0.9, 2),    # player
            (100, 90, 108, 98, 0.4, 0),  # ball
        ],
        tracker=tracker,
    )

    summary = pipeline.run(video)

    # Tracking ran on every frame.
    assert summary["tracking"] is True
    assert tracker.update_calls == 5
    assert summary["frames_processed"] == 5
    assert summary["total_tracks"] == 10  # 2 tracks x 5 frames
    # FakeTracker: player -> id 3, ball -> id 1 -> two unique IDs.
    assert summary["unique_track_ids"] == 2

    outputs = tmp_path / "outputs"
    # Detection JSON still produced (Phase 1 not broken).
    det_data = json.loads((outputs / "clip_detections.json").read_text("utf-8"))
    assert len(det_data["frames"]) == 5
    assert det_data["frames"][0]["detections"][0]["class"] == "player"

    # Tracking JSON produced in the spec format.
    trk_data = json.loads((outputs / "clip_tracks.json").read_text("utf-8"))
    assert trk_data["metadata"]["tracker"] == "botsort"
    assert len(trk_data["frames"]) == 5
    first = trk_data["frames"][0]["tracks"][0]
    assert set(first) == {"track_id", "class", "confidence", "bbox"}

    assert (outputs / "clip_annotated.mp4").is_file()


def test_detection_only_writes_no_tracks_json(tmp_path):
    video = create_test_video(tmp_path / "clip.avi", n_frames=3)
    pipeline, _ = make_pipeline(
        tmp_path, detections=[(10, 10, 40, 80, 0.9, 2)], tracker=None
    )

    summary = pipeline.run(video)

    assert summary["tracking"] is False
    assert "tracks_json" not in summary
    assert (tmp_path / "outputs" / "clip_detections.json").is_file()
    assert not (tmp_path / "outputs" / "clip_tracks.json").exists()


def test_tracking_output_path_override(tmp_path):
    video = create_test_video(tmp_path / "clip.avi", n_frames=2)
    custom = tmp_path / "custom" / "my_tracks.json"
    pipeline, config = make_pipeline(
        tmp_path, detections=[(10, 10, 40, 80, 0.9, 2)], tracker=FakeTracker()
    )
    config.tracking.output_path = str(custom)

    summary = pipeline.run(video)

    assert summary["tracks_json"] == str(custom)
    assert custom.is_file()
