"""README icin uygulamanin ekran goruntulerini alir.

Onkosul: `playwright` kurulu olmali ve chromium indirilmis olmali:
    pip install -r requirements-dev.txt
    playwright install chromium

Kullanim:
    python scripts/screenshot.py            # streamlit'i kendi baslatir
    python scripts/screenshot.py --url http://localhost:8501   # calisani kullan

Ciktilar docs/screenshots/ altina yazilir. Arayuz degistikce yeniden
calistirilabilir olsun diye script haline getirildi.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._preview import open_tab, select_model, start_app, wait_for  # noqa: E402
from src.config import DOCS_DIR  # noqa: E402

TARGET_DIR = DOCS_DIR / "screenshots"
# Yukseklik, tespit karsilastirmasinin tamami sigsin diye genis tutuldu.
VIEWPORT = {"width": 1440, "height": 1080}
# 2 kat retina keskin ama dosyayi gereksiz buyutuyor; 1.5 README icin yeterli.
SCALE = 1.5
# Ekran goruntulerinin buyuk kismi fotograf; PNG gereksiz sisiyor (1.4 MB -> ~300 KB).
JPEG_QUALITY = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="Calisan bir uygulamanin adresi")
    parser.add_argument(
        "--port", type=int, default=8599, help="Kendi baslatirken kullanilacak port"
    )
    return parser.parse_args()


def shoot(page, name: str) -> Path:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    path = TARGET_DIR / f"{name}.jpg"
    page.screenshot(path=str(path), type="jpeg", quality=JPEG_QUALITY)
    print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")
    return path


def main() -> None:
    args = parse_args()
    from playwright.sync_api import sync_playwright

    process = None
    if args.url:
        url = args.url
        if not wait_for(url, timeout=10):
            raise SystemExit(f"Uygulamaya ulasilamadi: {url}")
    else:
        print("Streamlit baslatiliyor...")
        process, url = start_app(args.port)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=SCALE)
            page.goto(url)
            # Ilk acilista model yukleniyor; acele etmeyelim.
            page.wait_for_timeout(6000)

            print("Ekran goruntuleri:")
            open_tab(page, "Samples")
            page.wait_for_timeout(4000)  # tespit calisiyor
            shoot(page, "detection")

            # Metrikler egitilmis modele ait; kenar cubugunun da onu gostermesi
            # icin once modeli secip sonra sekmeye geciyoruz.
            select_model(page, "Custom: african-wildlife")
            open_tab(page, "Model performance")
            shoot(page, "model-performance")

            browser.close()
    finally:
        if process:
            process.terminate()
            process.wait(timeout=10)

    print(f"\nBitti: {TARGET_DIR}")


if __name__ == "__main__":
    main()
