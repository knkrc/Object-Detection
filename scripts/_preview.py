"""Uygulamayi baslatip tarayiciyla gezmek icin ortak yardimcilar.

screenshot.py ve make_demo_gif.py ayni isi yapiyor: streamlit'i ayaga kaldir,
hazir olmasini bekle, sekmelerde gez. Tekrar etmemek icin buraya alindi.
Alt cizgiyle basliyor cunku dogrudan calistirilan bir script degil.
"""

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def wait_for(url: str, timeout: int = 90) -> bool:
    """Streamlit'in saglik ucuna cevap vermesini bekler."""
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
            sys.executable, "-m", "streamlit", "run", "app.py",
            f"--server.port={port}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
            # "Deploy" butonu ve menu goruntude gereksiz gurultu
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


def open_tab(page, label: str, settle: int = 2500) -> None:
    """Sekme basligina tiklar ve icerigin yerlesmesini bekler."""
    page.get_by_role("tab", name=label).click()
    page.wait_for_timeout(settle)


def select_model(page, label: str, settle: int = 6000) -> None:
    """Kenar cubugundaki model secim kutusundan bir secenek secer."""
    page.get_by_role("combobox").first.click()
    page.wait_for_timeout(500)
    page.get_by_text(label, exact=True).click()
    page.wait_for_timeout(settle)
