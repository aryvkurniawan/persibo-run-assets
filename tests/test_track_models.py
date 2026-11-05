from tracking.models import FrameTracks, Track


def make_track(**overrides):
    defaults = dict(
        frame_id=1,
        track_id=12,
        class_id=2,
        class_name="player",
        confidence=0.91234,
        bbox=(10.111, 20.222, 110.333, 220.444),
        source_detection_id=3,
    )
    defaults.update(overrides)
    return Track(**defaults)


def test_to_dict_matches_spec_format():
    track = make_track()
    assert track.to_dict() == {
        "track_id": 12,
        "class": "player",
        "confidence": 0.9123,
        "bbox": [10.11, 20.22, 110.33, 220.44],
    }


def test_to_dict_omits_internal_fields():
    # frame_id and source_detection_id are debugging aids, not part of the
    # exported tracking schema.
    data = make_track().to_dict()
    assert "frame_id" not in data
    assert "source_detection_id" not in data


def test_center_property():
    track = make_track(bbox=(10.0, 20.0, 110.0, 220.0))
    assert track.center == (60.0, 120.0)


def test_source_detection_id_optional():
    track = make_track(source_detection_id=None)
    assert track.source_detection_id is None


def test_frame_tracks_to_dict_and_counts():
    frame = FrameTracks(
        frame_index=7,
        tracks=[
            make_track(track_id=1),
            make_track(track_id=2),
            make_track(track_id=3, class_name="referee", class_id=3),
        ],
    )
    data = frame.to_dict()
    assert data["frame"] == 7
    assert len(data["tracks"]) == 3
    assert frame.count_by_class() == {"player": 2, "referee": 1}
