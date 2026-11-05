import json

from tracking.models import Track
from tracking.track_exporter import TrackJSONExporter


def make_track(track_id, name="player", class_id=2, conf=0.9):
    return Track(
        frame_id=1,
        track_id=track_id,
        class_id=class_id,
        class_name=name,
        confidence=conf,
        bbox=(10.0, 20.0, 50.0, 120.0),
        source_detection_id=0,
    )


def test_export_roundtrip_matches_spec(tmp_path):
    path = tmp_path / "out" / "tracks.json"
    exporter = TrackJSONExporter(
        path, metadata={"tracker": "botsort", "weights": "weights/best.pt"}
    )

    exporter.add_frame(1, [make_track(12), make_track(7, "referee", 3)])
    exporter.add_frame(2, [])  # empty frames must still be recorded
    saved = exporter.save()

    assert saved == path
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metadata"] == {"tracker": "botsort", "weights": "weights/best.pt"}
    assert len(data["frames"]) == 2
    assert data["frames"][0] == {
        "frame": 1,
        "tracks": [
            {"track_id": 12, "class": "player", "confidence": 0.9,
             "bbox": [10.0, 20.0, 50.0, 120.0]},
            {"track_id": 7, "class": "referee", "confidence": 0.9,
             "bbox": [10.0, 20.0, 50.0, 120.0]},
        ],
    }
    assert data["frames"][1] == {"frame": 2, "tracks": []}


def test_counters_and_unique_ids(tmp_path):
    exporter = TrackJSONExporter(tmp_path / "t.json")
    exporter.add_frame(1, [make_track(12), make_track(7, "referee", 3)])
    # Same player ID 12 reappears next frame -> still one unique player ID.
    exporter.add_frame(2, [make_track(12)])

    assert exporter.frame_count == 2
    assert exporter.total_tracks == 3
    assert exporter.unique_track_count == 2
