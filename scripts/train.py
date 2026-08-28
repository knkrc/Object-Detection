"""Hazir YOLOv8 modelini kendi veri setimizle fine-tune eder.

Ornek:
    python scripts/train.py --epochs 30
    python scripts/train.py --data brain-tumor.yaml --model yolov8s.pt --epochs 50

Egitim bitince en iyi agirlik `models/<isim>.pt` olarak kopyalanir; arayuz
bunu "Ozel: <isim>" seklinde otomatik olarak model listesine ekler.
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
    """En hizli kullanilabilir cihazi secer (Apple Silicon'da mps)."""
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
        help="Veri seti yapilandirmasi (ultralytics hazir seti veya kendi data.yaml'in)",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Baslangic agirligi")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default=None, help="Calisma adi (varsayilan: veri setinin adi)")
    parser.add_argument(
        "--device", default=None, help="mps / cpu / 0. Bos birakilirsa otomatik secilir"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Bu kadar epoch boyunca iyilesme yoksa erken durdur",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO  # torch'tan sonra import edilmeli

    name = args.name or Path(args.data).stem
    device = args.device or pick_device()

    print(f"veri seti : {args.data}")
    print(f"model     : {args.model}")
    print(f"cihaz     : {device}")
    print(f"epoch     : {args.epochs}  imgsz: {args.imgsz}  batch: {args.batch}")
    print()

    # models/ altinda varsa oradan yukle; yoksa ultralytics indirsin ve
    # indirdigini models/'a tasiyalim (yoksa proje koku kirleniyor).
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
        plots=True,  # results.png, confusion_matrix.png vs. uretir
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"En iyi agirlik bulunamadi: {best}")

    target = MODELS_DIR / f"{name}.pt"
    shutil.copy2(best, target)

    print()
    print(f"Egitim bitti. Ciktilar: {results.save_dir}")
    print(f"En iyi agirlik kopyalandi: {target}")
    print(f"Arayuzde 'Ozel: {name}' olarak gorunecek.")


if __name__ == "__main__":
    main()
