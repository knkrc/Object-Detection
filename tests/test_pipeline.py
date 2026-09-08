"""Tests for the batch pipeline behind the CLI.

The shaping functions are pure, so most of this needs no model at all. The two
`slow` tests at the end run the real thing end to end.
"""

from pathlib import Path

import pytest

from src.detector import Detection
from src.pipeline import (
    LineSpec,
    as_json,
    detection_rows,
    image_result,
    is_video,
    iter_sources,
    parse_line,
    video_result,
)


def make_detection(label: str, conf: float = 0.9, box=(0, 0, 10, 10)) -> Detection:
    return Detection(label=label, confidence=conf, box=box)


# --- parse_line ----------------------------------------------------------
# The CLI turns whatever this raises into an argparse error, so the messages
# matter as much as the values.


def test_parse_line_reads_orientation_and_position():
    assert parse_line("horizontal:0.5") == LineSpec("horizontal", 0.5)
    assert parse_line("vertical:0.25") == LineSpec("vertical", 0.25)


def test_parse_line_is_case_insensitive_and_trims():
    assert parse_line(" Horizontal :0.5") == LineSpec("horizontal", 0.5)


def test_parse_line_rejects_unknown_orientation():
    with pytest.raises(ValueError, match="orientation must be one of"):
        parse_line("diagonal:0.5")


def test_parse_line_rejects_missing_position():
    with pytest.raises(ValueError, match="missing position"):
        parse_line("horizontal")


def test_parse_line_rejects_non_numeric_position():
    with pytest.raises(ValueError, match="must be a number"):
        parse_line("horizontal:half")


@pytest.mark.parametrize("position", ["0", "1", "1.5", "-0.2"])
def test_parse_line_rejects_positions_outside_the_frame(position):
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_line(f"horizontal:{position}")


# --- iter_sources --------------------------------------------------------


def test_iter_sources_expands_a_directory(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.mp4").touch()

    assert iter_sources([tmp_path]) == [tmp_path / "a.jpg", tmp_path / "b.mp4"]


def test_iter_sources_skips_non_media_inside_a_directory(tmp_path):
    """A README next to the images must not become an error later."""
    (tmp_path / "a.jpg").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "sub").mkdir()

    assert iter_sources([tmp_path]) == [tmp_path / "a.jpg"]


def test_iter_sources_rejects_an_explicit_non_media_file(tmp_path):
    """Naming a file directly is a clear intent, so a wrong type is an error."""
    notes = tmp_path / "notes.txt"
    notes.touch()

    with pytest.raises(ValueError, match="unsupported file type"):
        iter_sources([notes])


def test_iter_sources_raises_for_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file"):
        iter_sources([tmp_path / "gone.jpg"])


def test_iter_sources_deduplicates_and_sorts(tmp_path):
    (tmp_path / "b.jpg").touch()
    (tmp_path / "a.jpg").touch()

    # The same file named twice, once directly and once through its folder.
    assert iter_sources([tmp_path, tmp_path / "a.jpg"]) == [
        tmp_path / "a.jpg",
        tmp_path / "b.jpg",
    ]


def test_iter_sources_is_case_insensitive_about_extensions(tmp_path):
    (tmp_path / "PHOTO.JPG").touch()
    assert iter_sources([tmp_path]) == [tmp_path / "PHOTO.JPG"]


def test_is_video_distinguishes_by_extension():
    assert is_video(Path("clip.mp4")) is True
    assert is_video(Path("clip.MOV")) is True
    assert is_video(Path("photo.jpg")) is False


# --- result shaping ------------------------------------------------------


def test_detection_rows_are_json_friendly():
    rows = detection_rows([make_detection("car", 0.87654, (1, 2, 3, 4))])

    assert rows == [{"label": "car", "confidence": 0.8765, "box": [1, 2, 3, 4]}]


def test_image_result_carries_size_and_counts():
    result = image_result(
        Path("a.jpg"), (640, 480), [make_detection("person"), make_detection("person")]
    )

    assert result["type"] == "image"
    assert (result["width"], result["height"]) == (640, 480)
    assert result["counts"] == {"person": 2}
    assert len(result["detections"]) == 2


def test_video_result_without_tracking_reports_per_frame_counts():
    stats = {"frames": 100, "fps": 25.0, "size": (640, 480)}

    result = video_result(Path("c.mp4"), stats, None, {"person": 300})

    assert result["tracking"] is None
    assert result["counts"] == {"person": 300}
    assert result["frames"] == 100


def test_as_json_records_the_settings_used():
    import json

    document = json.loads(as_json([], "yolov8n.pt", 0.35, ["person"]))

    assert document == {
        "model": "yolov8n.pt",
        "confidence": 0.35,
        "classes": ["person"],
        "results": [],
    }


def test_as_json_writes_null_for_an_empty_class_filter():
    import json

    assert json.loads(as_json([], "yolov8n.pt", 0.35, []))["classes"] is None


# --- end to end ----------------------------------------------------------


@pytest.mark.slow
def test_run_image_detects_and_writes_the_annotated_copy(tmp_path):
    from src.config import SAMPLES_DIR
    from src.detector import Detector
    from src.pipeline import run_image

    source = SAMPLES_DIR / "bus.jpg"
    if not source.exists():
        pytest.skip("no sample image: python scripts/download_samples.py")

    result = run_image(Detector("yolov8n.pt"), source, 0.35, None, tmp_path)

    assert result["counts"].get("person", 0) >= 1
    assert Path(result["output"]).exists()


@pytest.mark.slow
def test_run_video_without_output_leaves_no_file_behind(tmp_path, synthetic_video):
    """Frames still have to be walked, but nothing should be left on disk."""
    from src.detector import Detector
    from src.pipeline import run_video

    source = synthetic_video(frames=6)

    result = run_video(Detector("yolov8n.pt"), source, 0.35, None, output_dir=None)

    assert result["frames"] == 6
    assert "output" not in result
    assert list(source.parent.glob(".*discard*")) == []
