from detection.data_models import Detection, FrameDetections


def make_detection(**overrides):
    defaults = dict(
        class_id=2,
        class_name="player",
        confidence=0.93456,
        bbox=(10.123, 20.456, 110.789, 220.012),
    )
    defaults.update(overrides)
    return Detection(**defaults)


def test_to_dict_matches_export_format():
    det = make_detection()
    data = det.to_dict()

    assert data == {
        "class": "player",
        "confidence": 0.9346,
        "bbox": [10.12, 20.46, 110.79, 220.01],
    }


def test_geometry_properties():
    det = make_detection(bbox=(10.0, 20.0, 110.0, 220.0))
    assert det.center == (60.0, 120.0)
    assert det.width == 100.0
    assert det.height == 200.0


def test_frame_detections_to_dict():
    frame = FrameDetections(
        frame_index=1,
        detections=[make_detection(), make_detection(class_name="ball", class_id=0)],
    )
    data = frame.to_dict()

    assert data["frame"] == 1
    assert len(data["detections"]) == 2
    assert data["detections"][1]["class"] == "ball"


def test_count_by_class():
    frame = FrameDetections(
        frame_index=5,
        detections=[
            make_detection(),
            make_detection(),
            make_detection(class_name="referee", class_id=3),
        ],
    )
    assert frame.count_by_class() == {"player": 2, "referee": 1}
