"""Fine-tunes a pretrained YOLOv8 model on our own dataset.

Examples:
    python scripts/train.py --epochs 30
    python scripts/train.py --data brain-tumor.yaml --model yolov8s.pt --epochs 50

When training finishes the best weights are copied to `models/<name>.pt`, and
the UI picks them up automatically as "Custom: <name>".
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODELS_DIR, RUNS_DIR  # noqa: E402
from src.detector import resolve_weights, stash_weights  # noqa: E402


def pick_device() -> str:
    """Picks the fastest available device (mps on Apple Silicon)."""
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="african-wildlife.yaml",
        help="Dataset config (a built-in ultralytics one, or your own data.yaml)",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Starting weights")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default=None, help="Run name (defaults to the dataset name)")
    parser.add_argument(
        "--device", default=None, help="mps / cpu / 0. Chosen automatically if omitted"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Stop early if there is no improvement for this many epochs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO  # must be imported after torch

    name = args.name or Path(args.data).stem
    device = args.device or pick_device()

    print(f"dataset : {args.data}")
    print(f"model   : {args.model}")
    print(f"device  : {device}")
    print(f"epochs  : {args.epochs}  imgsz: {args.imgsz}  batch: {args.batch}")
    print()

    # Load from models/ if the weights are there; otherwise let ultralytics
    # download them and move the file into models/ (or the root gets littered).
    model = YOLO(resolve_weights(args.model))
    stash_weights(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=True,
        plots=True,  # produces results.png, confusion_matrix.png and friends
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Best weights not found: {best}")

    target = MODELS_DIR / f"{name}.pt"
    shutil.copy2(best, target)

    print()
    print(f"Training done. Output: {results.save_dir}")
    print(f"Best weights copied to: {target}")
    print(f"The UI will show this as 'Custom: {name}'.")


if __name__ == "__main__":
    main()
