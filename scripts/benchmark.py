"""Measures how fast the models run, per device, and writes a table.

The project reports mAP but never said how fast anything is, which is the other
half of choosing a model size. This fills that in.

What is timed is `Detector.detect()` — the whole path the app takes, including
drawing the boxes. That is the number a user actually waits for, and it is
larger than raw inference.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --models yolov8n.pt yolov8s.pt --devices cpu mps
    python scripts/benchmark.py --markdown docs/benchmark.md
"""

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import AVAILABLE_MODELS, DEFAULT_CONF, SAMPLES_DIR  # noqa: E402
from src.detector import Detector  # noqa: E402
from src.tracker import TrackSession  # noqa: E402

DEFAULT_RUNS = 20
DEFAULT_WARMUP = 3


def available_devices() -> list[str]:
    """Devices worth measuring on this machine, fastest first."""
    import torch

    devices = []
    if torch.cuda.is_available():
        devices.append("0")
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def summarize_times(times: list[float]) -> dict:
    """Median rather than mean: one slow run should not move the number.

    Also reports the spread, because a wide one means the figure is not to be
    trusted — thermal throttling and background load both show up here.
    """
    if not times:
        raise ValueError("no timings to summarize")

    median = statistics.median(times)
    return {
        "runs": len(times),
        "ms": round(median * 1000, 1),
        "fps": round(1.0 / median, 1) if median else 0.0,
        "min_ms": round(min(times) * 1000, 1),
        "max_ms": round(max(times) * 1000, 1),
    }


def time_images(detector: Detector, image, conf: float, runs: int, warmup: int) -> dict:
    """Times repeated detection on one image.

    The warm-up runs are thrown away: the first call through a device pays for
    lazy initialisation and would dominate the median otherwise.
    """
    for _ in range(warmup):
        detector.detect(image, conf)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        detector.detect(image, conf)
        times.append(time.perf_counter() - start)
    return summarize_times(times)


def time_video_frames(detector: Detector, frames: list, conf: float, track: bool) -> dict:
    """Times a run over prepared frames, with or without tracking.

    Tracking adds association work on top of detection; comparing the two tells
    you what that costs. The frames are held in memory so decoding does not
    enter the measurement.

    Each mode warms up over the whole clip before being measured. A few frames
    are not enough: the first attempt warmed up on three and reported tracking
    as *faster* than plain detection, which cannot be true. With a full pass
    discarded, tracking lands about 9% slower, which is the association work.
    """
    warm = TrackSession(detector=detector, conf=conf) if track else None
    for frame in frames:
        warm.step(frame) if warm is not None else detector.detect(frame, conf)

    session = TrackSession(detector=detector, conf=conf) if track else None

    times = []
    for frame in frames:
        start = time.perf_counter()
        if session is not None:
            session.step(frame)
        else:
            detector.detect(frame, conf)
        times.append(time.perf_counter() - start)
    return summarize_times(times)


def make_frames(image, count: int, width: int = 640) -> list:
    """A short synthetic clip: the sample image panned across the frame.

    Real footage would be better, but the repo has none that is ours to ship,
    and for timing what matters is that every frame is different work.
    """
    height = int(image.shape[0] * width / image.shape[1])
    scaled = cv2.resize(image, (width, height))
    return [np.roll(scaled, i * 8, axis=1) for i in range(count)]


def as_badge(rows: list[dict]) -> str:
    """The README badge line, derived from what was just measured.

    CPU rather than MPS: the hosted demo runs on CPU, and a number that only
    holds on Apple Silicon means nothing to someone reading from Linux. The
    smallest model, because that is the default the app ships with.

    Emitting it here keeps the badge honest — re-run the benchmark and you get
    the line to paste, instead of a claim that quietly goes stale.

    Rounded to the nearest 5 and marked "~": run-to-run variance moves the exact
    figure by a frame or two, and a badge that has to be edited every time it is
    measured is a badge nobody keeps current. The precise table is one click away.
    """
    cpu = [r for r in rows if r["device"] == "cpu"]
    if not cpu:
        return ""

    row = min(cpu, key=lambda r: r["ms"])
    label = row["model"].replace(".pt", "").replace("-", "--")
    fps = max(5, round(row["fps"] / 5) * 5)
    return (
        f"[![Speed](https://img.shields.io/badge/{label}-~{fps}_FPS_on_CPU-success)]"
        "(#how-fast-is-it)"
    )


def as_markdown(rows: list[dict], machine: str) -> str:
    """The table that goes in the README."""
    lines = [
        f"Measured on {machine}. Timings cover `Detector.detect()` end to end,",
        "including drawing the boxes — the wait a user actually sees.",
        "",
        "| Model | Device | ms / image | FPS |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['device']} | {row['ms']} | {row['fps']} |")

    badge = as_badge(rows)
    if badge:
        lines += ["", "README badge for these numbers:", "", "```markdown", badge, "```"]
    return "\n".join(lines) + "\n"


def describe_machine() -> str:
    return (
        f"{platform.machine()} {platform.system()} {platform.release()}, "
        f"Python {platform.python_version()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=list(AVAILABLE_MODELS.values()))
    parser.add_argument("--devices", nargs="+", default=None, help="Default: everything available")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--video-frames", type=int, default=30, help="0 skips the video comparison")
    parser.add_argument("--markdown", type=Path, help="Also write the table here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = args.devices or available_devices()

    source = SAMPLES_DIR / "bus.jpg"
    if not source.exists():
        raise SystemExit("no sample image: python scripts/download_samples.py")
    image = cv2.imread(str(source))

    print(describe_machine())
    print(
        f"image: {source.name} {image.shape[1]}x{image.shape[0]} | "
        f"conf {args.conf} | {args.runs} runs after {args.warmup} warm-up\n"
    )
    print(f"{'model':<28} {'device':<7} {'ms':>8} {'FPS':>7} {'min-max ms':>14}")
    print("-" * 68)

    rows = []
    for weights in args.models:
        for device in devices:
            detector = Detector(weights, device=device)
            result = time_images(detector, image, args.conf, args.runs, args.warmup)
            rows.append({"model": weights, "device": device, **result})
            print(
                f"{weights:<28} {device:<7} {result['ms']:>8} {result['fps']:>7} "
                f"{result['min_ms']:>6}-{result['max_ms']:<7}"
            )

    if args.video_frames:
        print(f"\nvideo, {args.video_frames} frames, {args.models[0]} on {devices[0]}:")
        frames = make_frames(image, args.video_frames)
        detector = Detector(args.models[0], device=devices[0])
        for label, track in (("detection", False), ("tracking", True)):
            result = time_video_frames(detector, frames, args.conf, track)
            print(f"  {label:<10} {result['ms']:>7} ms/frame   {result['fps']:>6} FPS")

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(as_markdown(rows, describe_machine()))
        print(f"\nwrote {args.markdown}")


if __name__ == "__main__":
    main()
