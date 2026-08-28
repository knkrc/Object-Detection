"""Line counter and tracking session tests.

The line direction logic was wrong once in M2 (rightward movement on a vertical
line was counted as backward); these tests keep that bug from coming back.
"""

import numpy as np
import pytest
from conftest import make_detector

from src.tracker import LineCounter, TrackSession, color_for, line_from_ratio

# --- LineCounter ---------------------------------------------------------


@pytest.fixture
def vertical_line() -> LineCounter:
    """A vertical line at x=100 drawn bottom to top, so right is positive."""
    return LineCounter((100, 200), (100, 0), names=("right", "left"))


def test_first_sighting_is_not_a_crossing(vertical_line):
    assert vertical_line.update(1, (50, 100)) is None
    assert vertical_line.counts == {"right": 0, "left": 0}


def test_staying_on_one_side_does_not_count(vertical_line):
    vertical_line.update(1, (50, 100))
    for x in (60, 70, 80, 99):
        assert vertical_line.update(1, (x, 100)) is None
    assert sum(vertical_line.counts.values()) == 0


def test_left_to_right_counts_as_right(vertical_line):
    vertical_line.update(1, (50, 100))
    assert vertical_line.update(1, (150, 100)) == "right"
    assert vertical_line.counts == {"right": 1, "left": 0}


def test_right_to_left_counts_as_left(vertical_line):
    vertical_line.update(1, (150, 100))
    assert vertical_line.update(1, (50, 100)) == "left"
    assert vertical_line.counts == {"right": 0, "left": 1}


def test_going_back_and_forth_counts_twice(vertical_line):
    vertical_line.update(1, (50, 100))
    vertical_line.update(1, (150, 100))
    vertical_line.update(1, (50, 100))
    assert vertical_line.counts == {"right": 1, "left": 1}


def test_different_ids_are_tracked_separately(vertical_line):
    vertical_line.update(1, (50, 100))
    vertical_line.update(2, (150, 100))
    # Object 2 starts on the right; its first sighting is not a crossing.
    assert vertical_line.counts == {"right": 0, "left": 0}
    vertical_line.update(1, (150, 100))
    assert vertical_line.counts["right"] == 1


def test_a_point_exactly_on_the_line_is_ignored(vertical_line):
    vertical_line.update(1, (50, 100))
    assert vertical_line.update(1, (100, 100)) is None
    # The side must survive intact: a left-to-right crossing still counts.
    assert vertical_line.update(1, (150, 100)) == "right"


def test_crossing_ids_are_recorded(vertical_line):
    vertical_line.update(7, (50, 100))
    vertical_line.update(7, (150, 100))
    assert vertical_line.crossed == {7}


# --- line_from_ratio: regression tests for the direction rules ------------


def test_rightward_movement_on_a_vertical_line_counts_as_right():
    """M2 bug: drawn top to bottom, rightward movement came out as 'left'."""
    line = line_from_ratio(640, 480, "vertical", 0.5)
    line.update(1, (100, 240))
    assert line.update(1, (500, 240)) == "right"


def test_downward_movement_on_a_horizontal_line_counts_as_down():
    line = line_from_ratio(640, 480, "horizontal", 0.5)
    line.update(1, (320, 50))
    assert line.update(1, (320, 400)) == "down"


def test_line_is_placed_by_ratio():
    line = line_from_ratio(640, 480, "horizontal", 0.25)
    assert line.p1 == (0, 120)
    assert line.p2 == (640, 120)


# --- color_for -----------------------------------------------------------


def test_same_id_always_gets_the_same_colour():
    assert color_for(5) == color_for(5)


def test_different_ids_get_different_colours():
    colours = {color_for(i) for i in range(10)}
    assert len(colours) > 1


# --- TrackSession --------------------------------------------------------


def test_one_id_across_many_frames_counts_once(blank_frame):
    # The same object (id=1) is seen across three frames.
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]] * 3)
    session = TrackSession(detector=detector, fps=10.0)

    for _ in range(3):
        session.step(blank_frame)

    assert session.unique_counts() == {"car": 1}
    assert session.summary()["total_objects"] == 1


def test_different_ids_are_counted_separately(blank_frame):
    detector = make_detector(
        [
            [(1, 0, 10, 10, 50, 50)],
            [(1, 0, 12, 10, 52, 50), (2, 0, 100, 100, 140, 140)],
            [(2, 0, 102, 100, 142, 140), (3, 1, 200, 200, 240, 240)],
        ]
    )
    session = TrackSession(detector=detector, fps=10.0)
    for _ in range(3):
        session.step(blank_frame)

    assert session.unique_counts() == {"car": 2, "person": 1}


def test_boxes_without_an_id_are_skipped(blank_frame):
    """The tracker leaves some boxes without an ID; those must not be counted."""
    detector = make_detector([[(None, 0, 10, 10, 50, 50), (1, 0, 60, 60, 90, 90)]])
    session = TrackSession(detector=detector, fps=10.0)
    _, tracks = session.step(blank_frame)

    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_duration_is_derived_from_fps(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]] * 20)
    session = TrackSession(detector=detector, fps=10.0)
    for _ in range(20):
        session.step(blank_frame)

    row = session.durations()[0]
    assert row["frames"] == 20
    assert row["seconds"] == pytest.approx(2.0)  # 20 frames / 10 fps
    assert row["first_frame"] == 0
    assert row["last_frame"] == 19


def test_duration_table_is_sorted_longest_first(blank_frame):
    detector = make_detector(
        [
            [(1, 0, 10, 10, 50, 50), (2, 0, 60, 60, 90, 90)],
            [(1, 0, 10, 10, 50, 50)],
            [(1, 0, 10, 10, 50, 50)],
        ]
    )
    session = TrackSession(detector=detector, fps=10.0)
    for _ in range(3):
        session.step(blank_frame)

    rows = session.durations()
    assert [r["id"] for r in rows] == [1, 2]


def test_trail_length_is_capped(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]] * 20)
    session = TrackSession(detector=detector, fps=10.0, trail_length=5)
    for _ in range(20):
        session.step(blank_frame)

    assert len(session.trails[1]) == 5


def test_trail_collects_centre_points(blank_frame):
    detector = make_detector(
        [
            [(1, 0, 0, 0, 100, 100)],  # merkez (50, 50)
            [(1, 0, 100, 0, 200, 100)],  # merkez (150, 50)
        ]
    )
    session = TrackSession(detector=detector, fps=10.0)
    session.step(blank_frame)
    session.step(blank_frame)

    assert list(session.trails[1]) == [(50, 50), (150, 50)]


def test_class_filter_is_passed_to_the_model(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]])
    session = TrackSession(detector=detector, fps=10.0, keep_classes=["person"])
    session.step(blank_frame)

    assert detector.model.calls[0]["classes"] == [1]


def test_no_filter_means_all_classes(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]])
    session = TrackSession(detector=detector, fps=10.0)
    session.step(blank_frame)

    assert detector.model.calls[0]["classes"] is None


def test_line_counter_is_wired_into_the_session(blank_frame):
    detector = make_detector(
        [
            [(1, 0, 0, 0, 100, 100)],  # merkez x=50, cizginin solunda
            [(1, 0, 500, 0, 600, 100)],  # merkez x=550, saginda -> gecis
        ]
    )
    line = line_from_ratio(640, 480, "vertical", 0.5)
    session = TrackSession(detector=detector, fps=10.0, line=line)
    session.step(blank_frame)
    session.step(blank_frame)

    assert session.summary()["line"] == {"right": 1, "left": 0}


def test_summary_line_is_none_without_a_counter(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]])
    session = TrackSession(detector=detector, fps=10.0)
    session.step(blank_frame)

    assert session.summary()["line"] is None


def test_reset_clears_the_state(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]] * 5)
    session = TrackSession(detector=detector, fps=10.0)
    for _ in range(5):
        session.step(blank_frame)

    session.reset()

    assert session.frame_index == 0
    assert session.unique_counts() == {}
    assert session.trails == {}
    assert session.durations() == []


def test_step_returns_an_annotated_frame(blank_frame):
    detector = make_detector([[(1, 0, 10, 10, 50, 50)]])
    session = TrackSession(detector=detector, fps=10.0)
    annotated, _ = session.step(blank_frame)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == blank_frame.shape
