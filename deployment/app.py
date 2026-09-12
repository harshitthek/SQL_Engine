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
    """Check if FastAPI /health responds with 200 and model status is healthy."""
    try:
        s = requests.Session()
        s.trust_env = False
        s.proxies = {"http": None, "https": None}
        r = s.get(f"{url.rstrip('/')}/health", timeout=2.0)
        if r.status_code == 200:
            try:
                data = r.json()
                if isinstance(data, dict) and "status" in data:
                    return data["status"] == "healthy"
            except Exception:
                pass
            return True
        return False
    except Exception:
        return False


def ensure_backend() -> subprocess.Popen | None:
    """
    Ensure the FastAPI backend is running if configured for loopback.
    Auto-spawns uvicorn in a background process if not already running,
    and waits until the model is loaded and /health returns healthy.
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

    # Wait for the backend to complete startup (downloading model + loading weights).
    # Model download can take several minutes on first run (2.45GB weights).
    startup_timeout = float(os.getenv("FASTAPI_STARTUP_TIMEOUT", "600"))
    t_start = time.time()
    last_log = t_start
    logger.info(f"Waiting for FastAPI backend to load model at {api_url} (timeout={int(startup_timeout)}s)...")
    while time.time() - t_start < startup_timeout:
        if is_backend_healthy(api_url):
            logger.info(f"FastAPI gateway and model became ready in {time.time() - t_start:.1f}s at {api_url}")
            return proc
        if proc.poll() is not None:
            logger.error(f"FastAPI process exited prematurely with returncode {proc.returncode}")
            break
        if time.time() - last_log >= 15.0:
            logger.info(f"Still waiting for model to load at {api_url} ({int(time.time() - t_start)}s elapsed)...")
            last_log = time.time()
        time.sleep(1.0)

    if not is_backend_healthy(api_url):
        logger.error(f"FastAPI model server at {api_url} failed to become ready within {startup_timeout}s.")
    return proc


if __name__ == "__main__":
    if os.getenv("SPACE_ID"):
        # HF Spaces: skip FastAPI subprocess, use direct in-process inference with @spaces.GPU
        logger.info("HF Spaces detected (SPACE_ID set). Using direct inference mode — skipping FastAPI subprocess.")
        try:
            from gradio_app import _get_direct_engine
            logger.info("Preloading Text2SQLEngine into memory for HF Spaces...")
            _get_direct_engine()
        except Exception as exc:
            logger.warning(f"Could not preload direct engine during startup (will load on demand): {exc}")
        launch()
    else:
        ensure_backend()
        api_url = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
        if os.getenv("FASTAPI_NO_AUTOSTART") != "1" and not is_backend_healthy(api_url):
            logger.error(
                f"Model is not loaded. Cannot launch Gradio UI because FastAPI inference server at {api_url} is not ready. "
                "Aborting startup."
            )
            sys.exit(1)
        logger.info("Model verified healthy. Launching Gradio UI...")
        launch()
