"""Tests for the benchmark script's pure parts.

The measuring itself depends on the machine and cannot be asserted on, but the
statistics and the table it produces can.
"""

import pytest

from scripts.benchmark import as_markdown, summarize_times


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
