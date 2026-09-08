"""Run detection or tracking from the command line, on files or whole folders.

Examples:
    python detect.py samples/                          # detect, print a summary
    python detect.py samples/bus.jpg --output out/     # write the annotated copy
    python detect.py clip.mp4 --track --line horizontal:0.5
    python detect.py samples/ --json results.json --classes person car

The same `src/` modules the Streamlit app uses do the work here; this file only
parses arguments and prints. Use `--json` when something else has to read the
results — the human-readable output is not meant to be parsed.
"""

import argparse
import sys
from pathlib import Path

from src.config import AVAILABLE_MODELS, DEFAULT_CONF, DEFAULT_TRAIL_LENGTH, custom_models
from src.detector import Detector
from src.pipeline import as_json, is_video, iter_sources, parse_line, run_image, run_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Images, videos, or folders containing them",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Weights file. Anything under models/ works; the pretrained ones "
        "download on first use (default: %(default)s)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help="Confidence threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        metavar="NAME",
        help="Only look for these classes, e.g. --classes person car",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="DIR",
        help="Write annotated images and videos here (created if missing)",
    )
    parser.add_argument("--json", type=Path, metavar="FILE", help="Write the results as JSON")
    parser.add_argument(
        "--track",
        action="store_true",
        help="Track objects across frames instead of detecting each one on its "
        "own. Videos only; gives unique counts rather than per-frame ones",
    )
    parser.add_argument(
        "--line",
        metavar="ORIENTATION:POSITION",
        help="Count crossings of a virtual line, e.g. horizontal:0.5. Implies --track",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process one in every N video frames (default: %(default)s)",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=DEFAULT_TRAIL_LENGTH,
        help="Frames of motion trail to draw when tracking (default: %(default)s)",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print errors")

    args = parser.parse_args(argv)

    if args.line is not None:
        try:
            args.line = parse_line(args.line)
        except ValueError as exc:
            parser.error(f"--line: {exc}")
        args.track = True

    if args.stride < 1:
        parser.error("--stride must be 1 or more")

    return args


def describe(result: dict) -> str:
    """One line per source, in the shape a person would want to read."""
    counts = result.get("counts") or {}
    found = ", ".join(f"{n}x {label}" for label, n in counts.items()) or "nothing"

    if result["type"] == "image":
        return f"{result['source']}: {found}"

    tracking = result.get("tracking")
    if tracking is None:
        return f"{result['source']}: {result['frames']} frames, {found} (per frame)"

    line = tracking["line"]
    crossings = ""
    if line:
        crossings = "  |  crossed: " + ", ".join(f"{name} {n}" for name, n in line.items())
    return (
        f"{result['source']}: {result['frames']} frames, "
        f"{tracking['total_objects']} distinct objects — {found}{crossings}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        sources = iter_sources(args.paths)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not sources:
        print("error: nothing to process", file=sys.stderr)
        return 2

    if args.track and not any(is_video(p) for p in sources):
        print("error: --track needs at least one video", file=sys.stderr)
        return 2

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    known = {**AVAILABLE_MODELS, **custom_models()}.values()
    if not args.quiet and args.model not in known and not (Path("models") / args.model).exists():
        print(f"note: {args.model} is not in models/, ultralytics will try to fetch it")

    detector = Detector(args.model)
    results = []

    for path in sources:
        try:
            if is_video(path):
                results.append(
                    run_video(
                        detector,
                        path,
                        conf=args.conf,
                        keep_classes=args.classes,
                        output_dir=args.output,
                        track=args.track,
                        line=args.line,
                        stride=args.stride,
                        trail_length=args.trail_length,
                        on_progress=None if args.quiet else _progress(path),
                    )
                )
            else:
                results.append(run_image(detector, path, args.conf, args.classes, args.output))
        except (ValueError, RuntimeError) as exc:
            # One unreadable file should not lose the results of everything else.
            print(f"error: {exc}", file=sys.stderr)
            continue

        if not args.quiet:
            print(describe(results[-1]))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(as_json(results, args.model, args.conf, args.classes))
        if not args.quiet:
            print(f"\nwrote {args.json}")

    if not results:
        return 1
    return 0


def _progress(path: Path):
    """Overwrites one line with the percentage; videos are the slow part.

    Only when stdout is a terminal: piped or redirected, the carriage returns
    do nothing and every percentage would end up on its own line in the log.
    """
    if not sys.stdout.isatty():
        return None

    def report(fraction: float) -> None:
        print(f"\r{path.name}: {fraction * 100:5.1f}%", end="", flush=True)
        if fraction >= 1.0:
            print("\r", end="")

    return report


if __name__ == "__main__":
    sys.exit(main())
