"""
Unit tests for app.py entrypoint and process orchestration.
"""
import os
import subprocess
from unittest.mock import MagicMock, patch

import app
import requests
from app import ensure_backend, is_backend_healthy


def test_is_backend_healthy():
    with patch("requests.Session.get") as mock_get:
        # Healthy 200
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp
        assert is_backend_healthy("http://127.0.0.1:8000") is True

        # Non-200
        mock_resp.status_code = 503
        assert is_backend_healthy("http://127.0.0.1:8000") is False

        # Exception
        mock_get.side_effect = requests.ConnectionError("Connection refused")
        assert is_backend_healthy("http://127.0.0.1:8000") is False


def test_ensure_backend_disabled():
    with patch.dict(os.environ, {"FASTAPI_NO_AUTOSTART": "1"}):
        assert ensure_backend() is None


def test_ensure_backend_remote_host():
    with patch.dict(os.environ, {"FASTAPI_URL": "http://api.production.example.com:8000"}):
        assert ensure_backend() is None


def test_ensure_backend_already_healthy():
    with patch.dict(os.environ, {"FASTAPI_URL": "http://127.0.0.1:8000"}), \
         patch("app.is_backend_healthy", return_value=True):
        assert ensure_backend() is None


def test_ensure_backend_spawns_and_becomes_ready():
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None

    registered_cleanup = []

    def fake_register(fn):
        registered_cleanup.append(fn)

    with patch.dict(os.environ, {"FASTAPI_URL": "http://127.0.0.1:8000"}, clear=False), \
         patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
         patch("atexit.register", side_effect=fake_register), \
         patch("time.sleep"), \
         patch("app.is_backend_healthy", side_effect=[False, True]):
        proc = ensure_backend()
        assert proc == mock_proc
        mock_popen.assert_called_once()
        assert len(registered_cleanup) == 1

        # Test registered cleanup normal termination
        cleanup_fn = registered_cleanup[0]
        cleanup_fn()
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=4)


def test_ensure_backend_cleanup_timeout_kills_proc():
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="uvicorn", timeout=4)

    registered_cleanup = []

    with patch.dict(os.environ, {"FASTAPI_URL": "http://127.0.0.1:8000"}, clear=False), \
         patch("subprocess.Popen", return_value=mock_proc), \
         patch("atexit.register", side_effect=lambda fn: registered_cleanup.append(fn)), \
         patch("app.is_backend_healthy", side_effect=[False, True]):
        ensure_backend()
        assert len(registered_cleanup) == 1

        cleanup_fn = registered_cleanup[0]
        cleanup_fn()
        mock_proc.kill.assert_called_once()


def test_ensure_backend_cleanup_already_stopped():
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 0  # already exited

    registered_cleanup = []
    with patch.dict(os.environ, {"FASTAPI_URL": "http://127.0.0.1:8000"}, clear=False), \
         patch("subprocess.Popen", return_value=mock_proc), \
         patch("atexit.register", side_effect=lambda fn: registered_cleanup.append(fn)), \
         patch("app.is_backend_healthy", side_effect=[False, True]):
        ensure_backend()
        cleanup_fn = registered_cleanup[0]
        cleanup_fn()
        mock_proc.terminate.assert_not_called()


def test_ensure_backend_proc_exits_prematurely():
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1  # exited with error
    mock_proc.returncode = 1

    with patch.dict(os.environ, {"FASTAPI_URL": "http://127.0.0.1:8000"}, clear=False), \
         patch("subprocess.Popen", return_value=mock_proc), \
         patch("atexit.register"), \
         patch("time.sleep"), \
         patch("app.is_backend_healthy", return_value=False):
        proc = ensure_backend()
        assert proc == mock_proc


def test_ensure_backend_timeout_15s():
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None

    current_time = 0.0

    def fake_time():
        nonlocal current_time
        t = current_time
        current_time += 6.0
        return t

    with patch.dict(os.environ, {"FASTAPI_URL": "http://127.0.0.1:8000"}, clear=False), \
         patch("subprocess.Popen", return_value=mock_proc), \
         patch("atexit.register"), \
         patch("time.sleep"), \
         patch("time.time", side_effect=fake_time), \
         patch("app.is_backend_healthy", return_value=False):
        proc = ensure_backend()
        assert proc == mock_proc


def test_app_main_and_repo_dir():
    import runpy
    import sys
    repo_dir = app.REPO_DIR
    while repo_dir in sys.path:
        sys.path.remove(repo_dir)

    with patch.dict(os.environ, {"FASTAPI_NO_AUTOSTART": "1"}), \
         patch("gradio_app.launch") as mock_launch:
        runpy.run_path(os.path.join(repo_dir, "app.py"), run_name="__main__")
        mock_launch.assert_called_once()
        assert repo_dir in sys.path



