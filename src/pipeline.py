"""Batch processing shared by the command line interface.

`detect.py` is kept to argument parsing and printing; everything that turns a
file into results lives here, so it can be tested without spawning a process.
Results come back as plain dicts because the CLI's other job is to emit JSON.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.config import IMAGE_TYPES, VIDEO_TYPES
from src.detector import Detection, Detector, summarize
from src.tracker import TrackSession, line_from_ratio
from src.video import process_video, video_info

ORIENTATIONS = ("horizontal", "vertical")


@dataclass
class LineSpec:
    """A `--line horizontal:0.5` argument, already validated."""

    orientation: str
    position: float


def parse_line(text: str) -> LineSpec:
    """Turns 'horizontal:0.5' into a LineSpec, or explains what went wrong.

    Raises ValueError so the CLI can turn it into a clean argparse error rather
    than a traceback.
    """
    orientation, _, raw = text.partition(":")
    orientation = orientation.strip().lower()

    if orientation not in ORIENTATIONS:
        allowed = ", ".join(ORIENTATIONS)
        raise ValueError(f"orientation must be one of {allowed}, got {orientation!r}")
    if not raw:
        raise ValueError("missing position, expected something like 'horizontal:0.5'")

    try:
        position = float(raw)
    except ValueError:
        raise ValueError(f"position must be a number, got {raw!r}") from None
    if not 0.0 < position < 1.0:
        raise ValueError(f"position must be between 0 and 1 (exclusive), got {position}")

    return LineSpec(orientation, position)


def iter_sources(paths: Iterable[Path]) -> list[Path]:
    """Expands the given paths into the media files we can actually process.

    A directory contributes the media files directly inside it; a file is taken
    as-is only if its extension is one we handle, so a stray README in a folder
    does not become an error later. Sorted for reproducible output.
    """
    suffixes = {f".{ext}" for ext in (*IMAGE_TYPES, *VIDEO_TYPES)}
    found: list[Path] = []

    for path in paths:
        if path.is_dir():
            found.extend(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
        elif path.is_file():
            if path.suffix.lower() not in suffixes:
                raise ValueError(f"unsupported file type: {path}")
            found.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")

    return sorted(set(found))


def is_video(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in VIDEO_TYPES


def detection_rows(detections: list[Detection]) -> list[dict]:
    """Detections as JSON-friendly dicts, in the order the model returned them."""
    return [
        {
            "label": d.label,
            "confidence": round(d.confidence, 4),
            "box": list(d.box),
        }
        for d in detections
    ]


def image_result(path: Path, size: tuple[int, int], detections: list[Detection]) -> dict:
    """Shapes one image's outcome. Separate from the model call so it is testable."""
    return {
        "source": str(path),
        "type": "image",
        "width": size[0],
        "height": size[1],
        "counts": summarize(detections),
        "detections": detection_rows(detections),
    }


def video_result(path: Path, stats: dict, session: TrackSession | None, counts: dict) -> dict:
    """Shapes one video's outcome.

    With tracking on, the interesting numbers are per-object and per-line; with
    it off, all we can honestly report is how often each class was seen, which
    counts the same object once per frame.
    """
    result = {
        "source": str(path),
        "type": "video",
        "frames": stats["frames"],
        "fps": round(stats["fps"], 2),
        "width": stats["size"][0],
        "height": stats["size"][1],
    }

    if session is None:
        result["tracking"] = None
        result["counts"] = counts
        return result

    summary = session.summary()
    result["counts"] = summary["unique"]
    result["tracking"] = {
        "unique": summary["unique"],
        "total_objects": summary["total_objects"],
        "line": summary["line"],
        "objects": session.durations(),
    }
    return result


def run_image(
    detector: Detector,
    path: Path,
    conf: float,
    keep_classes: list[str] | None,
    output_dir: Path | None,
) -> dict:
    """Detects on one image, optionally writing the annotated copy."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not read image: {path}")

    annotated, detections = detector.detect(image, conf, keep_classes)
    height, width = image.shape[:2]

    result = image_result(path, (width, height), detections)
    if output_dir is not None:
        target = output_dir / f"detected_{path.stem}.jpg"
        cv2.imwrite(str(target), annotated)
        result["output"] = str(target)
    return result


def run_video(
    detector: Detector,
    path: Path,
    conf: float,
    keep_classes: list[str] | None,
    output_dir: Path | None,
    track: bool = False,
    line: LineSpec | None = None,
    stride: int = 1,
    trail_length: int = 32,
    on_progress=None,
) -> dict:
    """Processes one video, with or without tracking.

    Writing the annotated video is optional, but the frames still have to be
    walked either way, so `output_dir=None` saves disk rather than time.
    """
    info = video_info(path)
    session = None
    counts: dict[str, int] = {}

    if track:
        counter = None
        if line is not None:
            counter = line_from_ratio(
                info["width"], info["height"], line.orientation, line.position
            )
        session = TrackSession(
            detector=detector,
            conf=conf,
            keep_classes=keep_classes,
            fps=info["fps"] / stride,
            trail_length=trail_length,
            line=counter,
        )

        def on_frame(frame):
            return session.step(frame)[0]
    else:

        def on_frame(frame):
            annotated, detections = detector.detect(frame, conf, keep_classes)
            for label, n in summarize(detections).items():
                counts[label] = counts.get(label, 0) + n
            return annotated

    target = (output_dir / f"detected_{path.stem}.mp4") if output_dir else _discard_path(path)
    stats = process_video(path, target, on_frame, stride=stride, on_progress=on_progress)

    result = video_result(path, stats, session, counts)
    if output_dir is not None:
        result["output"] = str(target)
    else:
        target.unlink(missing_ok=True)
    return result


def _discard_path(path: Path) -> Path:
    """Where to write frames we are going to throw away.

    `process_video` always writes a file; when the caller only wants numbers we
    still need somewhere to put it, next to the source so the temp file lands on
    the same filesystem.
    """
    return path.with_name(f".{path.stem}.discard.mp4")


def as_json(results: list[dict], model: str, conf: float, keep_classes: list[str] | None) -> str:
    """The document written by --json: the settings used, then the results."""
    return json.dumps(
        {
            "model": model,
            "confidence": conf,
            "classes": keep_classes or None,
            "results": results,
        },
        indent=2,
    )
