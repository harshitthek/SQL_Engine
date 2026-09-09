"""
Main entrypoint to run the Text-to-SQL Gradio application.
Ensures FastAPI backend inference server is active before launching UI.
"""
from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import time
from urllib.parse import urlparse
import requests

# Add current directory to path if needed
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from gradio_app import launch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [app] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")


def is_backend_healthy(url: str) -> bool:
    """Check if FastAPI /health responds with 200."""
    try:
        s = requests.Session()
        s.trust_env = False
        s.proxies = {"http": None, "https": None}
        r = s.get(f"{url.rstrip('/')}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def ensure_backend() -> subprocess.Popen | None:
    """
    Ensure the FastAPI backend is running if configured for loopback.
    Auto-spawns uvicorn in a background process if not already running.
    """
    if os.getenv("FASTAPI_NO_AUTOSTART") == "1":
        return None

    api_url = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
    parsed = urlparse(api_url)
    hostname = (parsed.hostname or "").lower()

    if hostname not in ("127.0.0.1", "localhost", "0.0.0.0"):
        return None

    if is_backend_healthy(api_url):
        logger.info(f"FastAPI backend is already running and healthy at {api_url}")
        return None

    logger.info(f"No backend detected at {api_url}. Spawning local FastAPI uvicorn gateway...")
    port = str(parsed.port or 8000)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", port, "--workers", "1"],
        cwd=REPO_DIR,
    )

    def _cleanup():
        if proc.poll() is None:
            logger.info("Stopping background FastAPI server...")
            proc.terminate()
            try:
                proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(_cleanup)

    # Wait up to 15s for the backend to complete startup
    t_start = time.time()
    while time.time() - t_start < 15.0:
        if is_backend_healthy(api_url):
            logger.info(f"FastAPI gateway became ready in {time.time() - t_start:.1f}s at {api_url}")
            return proc
        if proc.poll() is not None:
            logger.warning(f"FastAPI process exited prematurely with returncode {proc.returncode}")
            break
        time.sleep(0.5)

    logger.warning(f"FastAPI did not respond at {api_url} within 15s. Launching Gradio UI anyway.")
    return proc


if __name__ == "__main__":
    ensure_backend()
    launch()
