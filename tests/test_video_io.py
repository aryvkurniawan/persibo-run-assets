import cv2
import numpy as np
import pytest

from utils.video_io import VideoReader, VideoWriter


def create_test_video(path, n_frames=10, size=(64, 48), fps=20.0):
    """Write a tiny MJPG/AVI video (codec available on all platforms)."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    assert writer.isOpened()
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_reader_iterates_all_frames(tmp_path):
    video = create_test_video(tmp_path / "clip.avi", n_frames=10)

    with VideoReader(video) as reader:
        assert reader.width == 64
        assert reader.height == 48
        assert reader.fps == pytest.approx(20.0)

        frames = list(reader.frames())

    assert len(frames) == 10
    indices = [i for i, _ in frames]
    assert indices == list(range(1, 11)), "frame indices must be 1-based"
    assert frames[0][1].shape == (48, 64, 3)


def test_reader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoReader(tmp_path / "missing.mp4")


def test_writer_roundtrip(tmp_path):
    out = tmp_path / "nested" / "out.avi"

    with VideoWriter(out, fps=20.0, frame_size=(64, 48), codec="MJPG") as writer:
        for _ in range(5):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        assert writer.frames_written == 5

    assert out.is_file()
    with VideoReader(out) as reader:
        assert len(list(reader.frames())) == 5
