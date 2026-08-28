"""Takes the app screenshots used in the README.

Requires `playwright` with chromium installed:
    pip install -r requirements-dev.txt
    playwright install chromium

Usage:
    python scripts/screenshot.py            # starts streamlit itself
    python scripts/screenshot.py --url http://localhost:8501   # use a running one

Output goes to docs/screenshots/. It is a script rather than a manual step so
the images can be refreshed with one command whenever the UI changes.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._preview import open_tab, select_model, start_app, wait_for  # noqa: E402
from src.config import DOCS_DIR  # noqa: E402

TARGET_DIR = DOCS_DIR / "screenshots"
# The height is generous so the whole detection comparison fits.
VIEWPORT = {"width": 1440, "height": 1080}
# 2x retina is sharp but needlessly large; 1.5 is plenty for a README.
SCALE = 1.5
# Most of a screenshot here is photo; PNG bloats it (1.4 MB -> ~300 KB).
JPEG_QUALITY = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="Address of an already running app")
    parser.add_argument("--port", type=int, default=8599, help="Port to use when starting the app")
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
            raise SystemExit(f"Could not reach the app at: {url}")
    else:
        print("Starting Streamlit...")
        process, url = start_app(args.port)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=SCALE)
            page.goto(url)
            # The model loads on first open; give it time.
            page.wait_for_timeout(6000)

            print("Screenshots:")
            open_tab(page, "Samples")
            page.wait_for_timeout(4000)  # detection is running
            shoot(page, "detection")

            # The metrics belong to the trained model, so select it first and
            # then open the tab — otherwise the sidebar contradicts the table.
            select_model(page, "Custom: african-wildlife")
            open_tab(page, "Model performance")
            shoot(page, "model-performance")

            browser.close()
    finally:
        if process:
            process.terminate()
            process.wait(timeout=10)

    print(f"\nDone: {TARGET_DIR}")


if __name__ == "__main__":
    main()
