"""Video processing loop tests — no model required.

`process_video` does not know what work is being done; it calls the `on_frame`
it was given. So a fake `on_frame` is enough to test the loop itself.
"""

import cv2
import numpy as np
import pytest

from src.video import process_video, video_info


def test_video_info_reads_correctly(synthetic_video):
    source = synthetic_video(frames=12, size=(64, 48))
    info = video_info(source)

    assert info["frames"] == 12
    assert info["width"] == 64
    assert info["height"] == 48
    assert info["fps"] == pytest.approx(10.0)


def test_video_info_raises_for_a_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="Could not open"):
        video_info(tmp_path / "yok.mp4")


def test_process_video_raises_for_a_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="Could not open"):
        process_video(tmp_path / "yok.mp4", tmp_path / "cikti.mp4", lambda f: f)


def test_output_has_the_same_frame_count(synthetic_video, tmp_path):
    source = synthetic_video(frames=12)
    target = tmp_path / "cikti.mp4"

    stats = process_video(source, target, lambda frame: frame)

    assert stats["frames"] == 12
    assert target.exists()

    capture = cv2.VideoCapture(str(target))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    finally:
        capture.release()


def test_stride_1_processes_every_frame(synthetic_video, tmp_path):
    source = synthetic_video(frames=10)
    calls = []

    process_video(source, tmp_path / "c.mp4", lambda f: (calls.append(1), f)[1], stride=1)

    assert len(calls) == 10


def test_stride_3_processes_every_third_frame(synthetic_video, tmp_path):
    source = synthetic_video(frames=9)
    calls = []

    stats = process_video(source, tmp_path / "c.mp4", lambda f: (calls.append(1), f)[1], stride=3)

    # 9 frames, 0/3/6 are processed; the output is still 9 (the rest repeat).
    assert len(calls) == 3
    assert stats["frames"] == 9


def test_skipped_frames_repeat_the_last_annotated_frame(synthetic_video, tmp_path):
    """With stride > 1 the frames in between must repeat the last processed one."""
    source = synthetic_video(frames=4, size=(32, 32))
    target = tmp_path / "c.mp4"

    # Paint each processed frame a flat colour to see which one got written.
    colour = [50]

    def on_frame(frame):
        colour[0] += 50
        return np.full_like(frame, colour[0])

    process_video(source, target, on_frame, stride=2)

    capture = cv2.VideoCapture(str(target))
    try:
        written = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            written.append(int(frame[0, 0, 0]))
    finally:
        capture.release()

    # 4 frames / stride 2 -> processed twice, each value written twice.
    assert len(written) == 4
    assert written[0] == written[1]
    assert written[2] == written[3]
    assert written[0] != written[2]


def test_progress_callback_reaches_the_end(synthetic_video, tmp_path):
    source = synthetic_video(frames=8)
    progress = []

    process_video(source, tmp_path / "c.mp4", lambda f: f, on_progress=progress.append)

    assert len(progress) == 8
    assert progress[-1] == pytest.approx(1.0)
    assert progress == sorted(progress)  # must increase monotonically
    assert all(0 < p <= 1.0 for p in progress)


def test_progress_callback_is_optional(synthetic_video, tmp_path):
    source = synthetic_video(frames=5)
    stats = process_video(source, tmp_path / "c.mp4", lambda f: f)
    assert stats["frames"] == 5


def test_stats_report_the_source_properties(synthetic_video, tmp_path):
    source = synthetic_video(frames=6, size=(64, 48))
    stats = process_video(source, tmp_path / "c.mp4", lambda f: f)

    assert stats["size"] == (64, 48)
    assert stats["fps"] == pytest.approx(10.0)
