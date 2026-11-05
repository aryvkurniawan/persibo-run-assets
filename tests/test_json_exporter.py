import json

from detection.data_models import Detection
from utils.json_exporter import DetectionJSONExporter


def make_detection(name="player", conf=0.93):
    return Detection(
        class_id=2, class_name=name, confidence=conf, bbox=(10.0, 20.0, 50.0, 120.0)
    )


def test_export_roundtrip(tmp_path):
    path = tmp_path / "out" / "detections.json"
    exporter = DetectionJSONExporter(path, metadata={"video": "match.mp4"})

    exporter.add_frame(1, [make_detection(), make_detection("ball", 0.4)])
    exporter.add_frame(2, [])  # frames without detections must still appear
    saved_path = exporter.save()

    assert saved_path == path
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["metadata"]["video"] == "match.mp4"
    assert len(data["frames"]) == 2
    assert data["frames"][0] == {
        "frame": 1,
        "detections": [
            {"class": "player", "confidence": 0.93, "bbox": [10.0, 20.0, 50.0, 120.0]},
            {"class": "ball", "confidence": 0.4, "bbox": [10.0, 20.0, 50.0, 120.0]},
        ],
    }
    assert data["frames"][1] == {"frame": 2, "detections": []}


def test_counters(tmp_path):
    exporter = DetectionJSONExporter(tmp_path / "d.json")
    exporter.add_frame(1, [make_detection()])
    exporter.add_frame(2, [make_detection(), make_detection()])

    assert exporter.frame_count == 2
    assert exporter.total_detections == 3
