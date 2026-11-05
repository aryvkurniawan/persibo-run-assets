"""End-to-end pipeline test using a synthetic video and a fake detector."""

import json

import cv2
import numpy as np

from detection.detector import YOLODetector
from detection.pipeline import DetectionPipeline
from tests.fakes import FakeYOLOModel
from utils.config_loader import AppConfig, ModelConfig, VideoConfig

CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}


def create_test_video(path, n_frames=6, size=(160, 120), fps=20.0):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()
    return path


def make_pipeline(tmp_path, detections):
    config = AppConfig(
        model=ModelConfig(device="cpu"),
        classes=CLASS_NAMES,
        video=VideoConfig(output_dir=str(tmp_path / "outputs"), output_codec="MJPG"),
    )
    fake_model = FakeYOLOModel(detections)
    detector = YOLODetector(config.model, config.classes, model=fake_model)
    return DetectionPipeline(config, detector=detector), config


def test_pipeline_end_to_end(tmp_path):
    video = create_test_video(tmp_path / "clip.avi", n_frames=6)
    pipeline, config = make_pipeline(
        tmp_path,
        detections=[
            (10, 10, 40, 80, 0.9, 2),   # player
            (100, 90, 108, 98, 0.5, 0),  # ball
        ],
    )

    summary = pipeline.run(video)

    assert summary["frames_processed"] == 6
    assert summary["total_detections"] == 12  # 2 detections x 6 frames
    assert summary["detections_by_class"] == {"player": 6, "ball": 6}
    assert summary["average_fps"] > 0
    assert summary["stopped_early"] is False

    json_path = tmp_path / "outputs" / "clip_detections.json"
    assert json_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data["frames"]) == 6
    assert data["frames"][0]["frame"] == 1
    assert data["frames"][0]["detections"][0]["class"] == "player"
    assert data["metadata"]["resolution"] == [160, 120]

    # Annotated video exists; note the pipeline always names it .mp4 but the
    # MJPG codec still produces a readable file for the assertion below.
    assert (tmp_path / "outputs" / "clip_annotated.mp4").is_file()


def test_pipeline_without_video_output(tmp_path):
    video = create_test_video(tmp_path / "clip.avi", n_frames=3)
    pipeline, config = make_pipeline(tmp_path, detections=[])
    config.video.save_video = False

    summary = pipeline.run(video)

    assert summary["frames_processed"] == 3
    assert summary["total_detections"] == 0
    assert summary["annotated_video"] is None
    assert not (tmp_path / "outputs" / "clip_annotated.mp4").exists()
    assert (tmp_path / "outputs" / "clip_detections.json").is_file()
