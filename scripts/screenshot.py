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
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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


def wait_for(url: str, timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    return False


def start_app(port: int) -> tuple[subprocess.Popen, str]:
    """Streamlit'i arka planda baslatir ve hazir olmasini bekler."""
    url = f"http://localhost:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            f"--server.port={port}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            # "Deploy" butonu ve menu ekran goruntusunde gereksiz gurultu
            "--client.toolbarMode=viewer",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for(url):
        process.terminate()
        raise SystemExit(f"Uygulama {port} portunda ayaga kalkmadi.")
    return process, url


def open_tab(page, label: str) -> None:
    """Sekme basligina tiklar ve icerigin yerlesmesini bekler."""
    page.get_by_role("tab", name=label).click()
    page.wait_for_timeout(2500)


def select_model(page, label: str) -> None:
    """Kenar cubugundaki model secim kutusundan bir secenek secer."""
    page.get_by_role("combobox").first.click()
    page.wait_for_timeout(500)
    page.get_by_text(label, exact=True).click()
    # Model yuklenip sayfa yeniden cizilene kadar bekle
    page.wait_for_timeout(6000)


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
            open_tab(page, "Ornekler")
            page.wait_for_timeout(4000)  # tespit calisiyor
            shoot(page, "tespit")

            # Metrikler egitilmis modele ait; kenar cubugunun da onu gostermesi
            # icin once modeli secip sonra sekmeye geciyoruz.
            select_model(page, "Ozel: african-wildlife")
            open_tab(page, "Model performansi")
            shoot(page, "model-performansi")

            browser.close()
    finally:
        if process:
            process.terminate()
            process.wait(timeout=10)

    print(f"\nBitti: {TARGET_DIR}")


if __name__ == "__main__":
    main()
