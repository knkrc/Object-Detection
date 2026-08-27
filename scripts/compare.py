"""Hazir COCO modeli ile kendi egittigimiz modeli ayni goresellerde karsilastirir.

Ornek:
    python scripts/compare.py --custom models/african-wildlife.pt

Cikti: docs/comparison/ altinda yan yana gorseller ve bir ozet izgara.
Amac, "hazir model bunu bilmiyor, benim modelim biliyor" farkini gostermek.
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
    parser.add_argument("--baseline", default="yolov8n.pt", help="Karsilastirilacak hazir model")
    parser.add_argument("--images", default=None,
                        help="Gorsellerin bulundugu klasor (varsayilan: veri setinin val bolumu)")
    parser.add_argument("--dataset", default="african-wildlife")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


# Veri setleri dogrulama bolumunu farkli duzenlerde tutuyor; hepsini deneriz.
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
            f"Gorsel klasoru bulunamadi: {folder}\n"
            "--images ile klasoru elle verebilirsin."
        )
    return folder


def label_path(image: Path) -> Path:
    """Bir gorselin YOLO etiket dosyasi: .../images/val/x.jpg -> .../labels/val/x.txt"""
    parts = [("labels" if part == "images" else part) for part in image.parts]
    return Path(*parts).with_suffix(".txt")


def first_class(image: Path) -> int | None:
    """Gorseldeki ilk nesnenin sinif id'si; etiket yoksa None."""
    labels = label_path(image)
    if not labels.exists():
        return None
    for line in labels.read_text().splitlines():
        if line.strip():
            return int(line.split()[0])
    return None


def find_images(args) -> list[Path]:
    """Her siniftan ornek secer.

    Rastgele secim, veri setinde cok olan sinifa (burada fil) yigilabiliyor;
    oysa asil ilginc olan hazir modelin *bilmedigi* siniflar. Sinif basina
    esit dagitmak hem daha temsili hem de karsilastirmayi anlamli kiliyor.
    """
    folder = find_folder(args)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"Klasorde gorsel yok: {folder}")

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
    """Cikti dosyasini iceriginden adlandirir: rhino-1.jpg, buffalo-2.jpg...

    Veri setindeki dosya adlarinda bosluk ve parantez var; bunlar README'de
    ve URL'de sorun cikariyor. Ayrica anlamli isim, arayuzdeki secim
    kutusunda hangi gorsele baktigini gosteriyor.
    """
    counts = summarize(detections)
    base = next(iter(counts), "tespit-yok")
    base = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-") or "gorsel"

    name, index = base, 1
    while name in used:
        index += 1
        name = f"{base}-{index}"
    used.add(name)
    return name


def as_labels(detections) -> str:
    """'2x elephant, 1x cow' seklinde kisa bir ozet."""
    counts = summarize(detections)
    return ", ".join(f"{n}x {label}" for label, n in counts.items()) or "hicbir sey"


def with_banner(image: np.ndarray, text: str) -> np.ndarray:
    """Gorselin ustune hangi modelin sonucu oldugunu yazan bir serit ekler."""
    banner = np.full((BANNER_HEIGHT, image.shape[1], 3), 30, np.uint8)
    # Uzun etiket listesi serite sigsin diye yaziyi genislige gore olcekliyoruz.
    scale = min(0.7, max(0.4, image.shape[1] / (len(text) * 22)))
    cv2.putText(banner, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return np.vstack([banner, image])


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Iki gorseli ayni yukseklige getirip yan yana koyar."""
    height = min(left.shape[0], right.shape[0])

    def fit(image):
        scale = height / image.shape[0]
        return cv2.resize(image, (int(image.shape[1] * scale), height))

    return np.hstack([fit(left), fit(right)])


def main() -> None:
    args = parse_args()

    custom_path = Path(args.custom)
    if not custom_path.exists():
        raise SystemExit(f"Ozel model bulunamadi: {custom_path}\nOnce scripts/train.py calistir.")

    baseline = Detector(args.baseline)
    custom = Detector(custom_path.name if (custom_path.parent.name == "models") else str(custom_path))

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

        # Asil fark sayida degil etikette: hazir model gergedani "cow" saniyor.
        base_text = as_labels(base_hits)
        own_text = as_labels(own_hits)

        panel = side_by_side(
            with_banner(base_drawn, f"Hazir COCO modeli: {base_text}"),
            with_banner(own_drawn, f"Kendi modelimiz: {own_text}"),
        )
        cv2.imwrite(str(target / f"{output_name(own_hits, used_names)}.jpg"), panel)
        panels.append(panel)

        mark = "  <-- farkli" if base_text != own_text else ""
        print(f"{path.name:16} hazir: {base_text:28} ozel: {own_text}{mark}")

    if panels:
        width = min(p.shape[1] for p in panels)
        stacked = np.vstack([
            cv2.resize(p, (width, int(p.shape[0] * width / p.shape[1]))) for p in panels
        ])
        grid_path = target / "ozet.jpg"
        cv2.imwrite(str(grid_path), stacked)
        print(f"\n{len(panels)} karsilastirma yazildi: {target}")
        print(f"Ozet izgara: {grid_path}")


if __name__ == "__main__":
    main()
