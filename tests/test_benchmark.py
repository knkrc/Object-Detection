"""Tests for the benchmark script's pure parts.

The measuring itself depends on the machine and cannot be asserted on, but the
statistics and the table it produces can.
"""

import pytest

from scripts.benchmark import as_badge, as_markdown, summarize_times


def test_summarize_uses_the_median_not_the_mean():
    """One slow run must not move the headline number."""
    times = [0.010, 0.010, 0.010, 0.010, 1.000]

    assert summarize_times(times)["ms"] == 10.0


def test_summarize_reports_fps_as_the_inverse_of_the_median():
    assert summarize_times([0.02, 0.02, 0.02])["fps"] == 50.0


def test_summarize_keeps_the_spread():
    """A wide spread is the signal that the figure is not to be trusted."""
    result = summarize_times([0.010, 0.020, 0.030])

    assert result["min_ms"] == 10.0
    assert result["max_ms"] == 30.0
    assert result["runs"] == 3


def test_summarize_rejects_an_empty_run():
    with pytest.raises(ValueError, match="no timings"):
        summarize_times([])


def test_markdown_has_a_row_per_measurement():
    rows = [
        {"model": "yolov8n.pt", "device": "mps", "ms": 9.5, "fps": 105.8},
        {"model": "yolov8n.pt", "device": "cpu", "ms": 25.6, "fps": 39.1},
    ]

    table = as_markdown(rows, "test machine")

    assert "| yolov8n.pt | mps | 9.5 | 105.8 |" in table
    assert "| yolov8n.pt | cpu | 25.6 | 39.1 |" in table
    assert "test machine" in table


def test_markdown_says_what_was_timed():
    """The number includes drawing, and the table has to admit that."""
    assert "drawing the boxes" in as_markdown([], "test machine")


# --- as_badge ------------------------------------------------------------
# The badge is generated from the measurement so it cannot quietly go stale.


def test_badge_reports_the_cpu_number():
    """MPS is Apple-only; the badge has to mean something on any machine."""
    rows = [
        {"model": "yolov8n.pt", "device": "mps", "ms": 9.3, "fps": 107.2},
        {"model": "yolov8n.pt", "device": "cpu", "ms": 25.5, "fps": 39.2},
    ]

    assert "~40_FPS_on_CPU" in as_badge(rows)


def test_badge_picks_the_fastest_cpu_model():
    """The default the app ships with, not the slowest one measured."""
    rows = [
        {"model": "yolov8m.pt", "device": "cpu", "ms": 106.9, "fps": 9.4},
        {"model": "yolov8n.pt", "device": "cpu", "ms": 25.5, "fps": 39.2},
    ]

    badge = as_badge(rows)

    assert "yolov8n" in badge
    assert "yolov8m" not in badge


def test_badge_rounds_so_it_does_not_need_editing_every_run():
    """39.2 and 40.7 are the same measurement on different days."""

    def badge_fps(fps):
        return as_badge([{"model": "m.pt", "device": "cpu", "ms": 1.0, "fps": fps}])

    assert badge_fps(39.2) == badge_fps(40.7)
    assert "~40_FPS" in badge_fps(39.2)


def test_badge_escapes_hyphens_in_the_model_name():
    """shields.io reads a bare hyphen as the separator between label and value."""
    rows = [{"model": "african-wildlife.pt", "device": "cpu", "ms": 25.0, "fps": 40.0}]

    assert "african--wildlife" in as_badge(rows)


def test_badge_is_empty_without_a_cpu_measurement():
    rows = [{"model": "yolov8n.pt", "device": "mps", "ms": 9.3, "fps": 107.2}]

    assert as_badge(rows) == ""


def test_markdown_includes_the_badge_to_paste():
    rows = [{"model": "yolov8n.pt", "device": "cpu", "ms": 25.5, "fps": 39.2}]

    table = as_markdown(rows, "test machine")

    assert "README badge" in table
    assert "~40_FPS_on_CPU" in table
