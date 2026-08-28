"""Shared helpers for starting the app and driving it with a browser.

screenshot.py and make_demo_gif.py do the same setup: bring streamlit up, wait
until it is ready, walk through the tabs. Pulled out here to avoid repeating it.
The leading underscore marks it as a helper, not a script you run directly.
"""

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def wait_for(url: str, timeout: int = 90) -> bool:
    """Waits for Streamlit's health endpoint to respond."""
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
    """Starts Streamlit in the background and waits until it is ready."""
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
            # The "Deploy" button and menu are just noise in a screenshot
            "--client.toolbarMode=viewer",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for(url):
        process.terminate()
        raise SystemExit(f"The app did not come up on port {port}.")
    return process, url


def open_tab(page, label: str, settle: int = 2500) -> None:
    """Clicks a tab and waits for its content to settle."""
    page.get_by_role("tab", name=label).click()
    page.wait_for_timeout(settle)


def select_model(page, label: str, settle: int = 6000) -> None:
    """Picks an option from the model selectbox in the sidebar."""
    page.get_by_role("combobox").first.click()
    page.wait_for_timeout(500)
    page.get_by_text(label, exact=True).click()
    page.wait_for_timeout(settle)
