"""Egitilmis modeli dogrulama setinde olcer ve metrikleri docs/ altina yazar.

Ornek:
    python scripts/evaluate.py --model models/african-wildlife.pt

Ciktilar:
    docs/metrics.json  — ham sayilar
    docs/metrics.md    — README'ye yapistirilabilir tablo
    docs/plots/        — egitim grafikleri ve confusion matrix (runs/ git'e girmiyor)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DOCS_DIR, RUNS_DIR  # noqa: E402

# Ultralytics'in urettigi, README'de gostermeye deger grafikler
PLOT_FILES = [
    "results.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "val_batch0_pred.jpg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/african-wildlife.pt")
    parser.add_argument("--data", default="african-wildlife.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--run",
        default=None,
        help="Grafiklerin kopyalanacagi egitim klasoru (varsayilan: runs/<veri seti adi>)",
    )
    return parser.parse_args()


def collect_metrics(metrics, names: dict[int, str]) -> dict:
    """Ultralytics'in metrik nesnesinden ise yarayan sayilari cikarir."""
    box = metrics.box
    overall = {
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
    }

    per_class = []
    for i, class_index in enumerate(box.ap_class_index):
        per_class.append(
            {
                "class": names[int(class_index)],
                "mAP50": round(float(box.ap50[i]), 4),
                "mAP50-95": round(float(box.ap[i].mean()), 4),
                "precision": round(float(box.p[i]), 4),
                "recall": round(float(box.r[i]), 4),
            }
        )

    return {"overall": overall, "per_class": per_class}


def as_markdown(model_name: str, data: str, results: dict) -> str:
    lines = [
        f"# Training results — `{model_name}`",
        "",
        f"Dataset: `{data}`",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key, value in results["overall"].items():
        lines.append(f"| {key} | {value:.3f} |")

    lines += [
        "",
        "## Per class",
        "",
        "| Class | mAP50 | mAP50-95 | Precision | Recall |",
        "|---|---|---|---|---|",
    ]
    for row in results["per_class"]:
        lines.append(
            f"| {row['class']} | {row['mAP50']:.3f} | {row['mAP50-95']:.3f} "
            f"| {row['precision']:.3f} | {row['recall']:.3f} |"
        )

    return "\n".join(lines) + "\n"


def copy_plots(run_dir: Path) -> list[str]:
    """Egitim grafiklerini docs/plots'a kopyalar — runs/ git'e girmiyor."""
    target = DOCS_DIR / "plots"
    target.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in PLOT_FILES:
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, target / name)
            copied.append(name)
    return copied


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}\nRun scripts/train.py first.")

    model = YOLO(str(model_path))
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device, verbose=False)
    results = collect_metrics(metrics, model.names)

    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "metrics.json").write_text(
        json.dumps(
            {"model": model_path.name, "data": args.data, **results},
            indent=2,
            ensure_ascii=False,
        )
    )
    (DOCS_DIR / "metrics.md").write_text(as_markdown(model_path.name, args.data, results))

    run_dir = Path(args.run) if args.run else RUNS_DIR / Path(args.data).stem
    copied = copy_plots(run_dir) if run_dir.exists() else []

    print(as_markdown(model_path.name, args.data, results))
    print(f"Yazildi: {DOCS_DIR / 'metrics.json'}, {DOCS_DIR / 'metrics.md'}")
    print(f"Kopyalanan grafikler ({len(copied)}): {', '.join(copied) or 'yok'}")


if __name__ == "__main__":
    main()
