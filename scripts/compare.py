"""Compares the pretrained COCO model with our own on the same images.

Example:
    python scripts/compare.py --custom models/african-wildlife.pt

Output: side-by-side images and a summary grid under docs/comparison/.
The point is to show what the pretrained model does not know and ours does.
"""

import argparse
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATASETS_DIR, DOCS_DIR  # noqa: E402
from src.detector import Detector, summarize  # noqa: E402

BANNER_HEIGHT = 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom", default="models/african-wildlife.pt")
    parser.add_argument(
        "--baseline", default="yolov8n.pt", help="Pretrained model to compare against"
    )
    parser.add_argument(
        "--images",
        default=None,
        help="Folder holding the images (default: the dataset's val split)",
    )
    parser.add_argument("--dataset", default="african-wildlife")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


# Datasets lay out their validation split differently; try all the common ones.
VAL_LAYOUTS = [
    ("images", "val"),
    ("images", "valid"),
    ("valid", "images"),
    ("val", "images"),
]


def find_folder(args) -> Path:
    if args.images:
        folder = Path(args.images)
    else:
        root = DATASETS_DIR / args.dataset
        candidates = [root.joinpath(*parts) for parts in VAL_LAYOUTS]
        folder = next((c for c in candidates if c.is_dir()), candidates[0])

    if not folder.exists():
        raise SystemExit(
            f"Image folder not found: {folder}\nPass --images to point at one yourself."
        )
    return folder


def label_path(image: Path) -> Path:
    """An image's YOLO label file: .../images/val/x.jpg -> .../labels/val/x.txt"""
    parts = [("labels" if part == "images" else part) for part in image.parts]
    return Path(*parts).with_suffix(".txt")


def first_class(image: Path) -> int | None:
    """Class id of the first object in an image; None when there is no label."""
    labels = label_path(image)
    if not labels.exists():
        return None
    for line in labels.read_text().splitlines():
        if line.strip():
            return int(line.split()[0])
    return None


def find_images(args) -> list[Path]:
    """Picks samples from every class.

    Random sampling piles up on whichever class dominates the dataset (elephant,
    here), while the interesting cases are the classes the pretrained model does
    *not* know. Spreading evenly is both more representative and more useful.
    """
    folder = find_folder(args)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"No images in folder: {folder}")

    random.seed(args.seed)

    by_class: dict[int, list[Path]] = {}
    for image in images:
        cls = first_class(image)
        if cls is not None:
            by_class.setdefault(cls, []).append(image)

    if not by_class:
        return random.sample(images, min(args.count, len(images)))

    picked: list[Path] = []
    classes = sorted(by_class)
    # Siniflar arasinda sirayla dolasarak istenen sayiya ulasiriz.
    while len(picked) < args.count:
        added = False
        for cls in classes:
            pool = [p for p in by_class[cls] if p not in picked]
            if pool and len(picked) < args.count:
                picked.append(random.choice(pool))
                added = True
        if not added:
            break
    return picked


def output_name(detections, used: set[str]) -> str:
    """Names the output file after its content: rhino-1.jpg, buffalo-2.jpg...

    Filenames in the dataset contain spaces and parentheses, which cause trouble
    in the README and in URLs. A meaningful name also tells you which image you
    are looking at in the app's picker.
    """
    counts = summarize(detections)
    base = next(iter(counts), "no-detection")
    base = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-") or "image"

    name, index = base, 1
    while name in used:
        index += 1
        name = f"{base}-{index}"
    used.add(name)
    return name


def as_labels(detections) -> str:
    """A short summary like "2x elephant, 1x cow"."""
    counts = summarize(detections)
    return ", ".join(f"{n}x {label}" for label, n in counts.items()) or "nothing"


def with_banner(image: np.ndarray, text: str) -> np.ndarray:
    """Adds a banner above the image saying which model produced it."""
    banner = np.full((BANNER_HEIGHT, image.shape[1], 3), 30, np.uint8)
    # Scale the text to the width so a long label list still fits the banner.
    scale = min(0.7, max(0.4, image.shape[1] / (len(text) * 22)))
    cv2.putText(banner, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return np.vstack([banner, image])


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Puts two images side by side at the same height."""
    height = min(left.shape[0], right.shape[0])

    def fit(image):
        scale = height / image.shape[0]
        return cv2.resize(image, (int(image.shape[1] * scale), height))

    return np.hstack([fit(left), fit(right)])


def main() -> None:
    args = parse_args()

    custom_path = Path(args.custom)
    if not custom_path.exists():
        raise SystemExit(f"Custom model not found: {custom_path}\nRun scripts/train.py first.")

    baseline = Detector(args.baseline)
    is_in_models_dir = custom_path.parent.name == "models"
    custom = Detector(custom_path.name if is_in_models_dir else str(custom_path))

    target = DOCS_DIR / "comparison"
    target.mkdir(parents=True, exist_ok=True)

    panels = []
    used_names: set[str] = set()
    for path in find_images(args):
        image = cv2.imread(str(path))
        if image is None:
            continue

        base_drawn, base_hits = baseline.detect(image, args.conf)
        own_drawn, own_hits = custom.detect(image, args.conf)

        # The real difference is the label, not the count: the pretrained
        # model thinks a rhino is a "cow".
        base_text = as_labels(base_hits)
        own_text = as_labels(own_hits)

        panel = side_by_side(
            with_banner(base_drawn, f"Pretrained COCO model: {base_text}"),
            with_banner(own_drawn, f"Our own model: {own_text}"),
        )
        cv2.imwrite(str(target / f"{output_name(own_hits, used_names)}.jpg"), panel)
        panels.append(panel)

        mark = "  <-- differs" if base_text != own_text else ""
        print(f"{path.name:16} pretrained: {base_text:28} custom: {own_text}{mark}")

    if panels:
        width = min(p.shape[1] for p in panels)
        stacked = np.vstack(
            [cv2.resize(p, (width, int(p.shape[0] * width / p.shape[1]))) for p in panels]
        )
        grid_path = target / "summary.jpg"
        cv2.imwrite(str(grid_path), stacked)
        print(f"\n{len(panels)} comparisons written to: {target}")
        print(f"Summary grid: {grid_path}")


if __name__ == "__main__":
    main()
