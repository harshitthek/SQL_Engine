"""
Text-to-SQL Workstation - Gradio Frontend Application.

Connects to existing FastAPI inference gateway at http://0.0.0.0:8000.
Owns database management, schema introspection, UI state, and safe query execution.
Matches developer-tool technical green terminal aesthetic from reference wireframes.
"""
from __future__ import annotations

import os
import re

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
# Ensure localhost/loopback bypasses proxy in sandboxed/corporate environments
for _k in ("no_proxy", "NO_PROXY"):
    _cur = os.environ.get(_k, "")
    if not _cur:
        os.environ[_k] = "127.0.0.1,localhost,0.0.0.0"
    elif "127.0.0.1" not in _cur:
        os.environ[_k] = f"{_cur},127.0.0.1,localhost,0.0.0.0"

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib")
    os.makedirs(mpl_cache, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_cache

import json
import logging
import sqlite3
import sys
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from typing import Any

import gradio as gr
import pandas as pd
from api_client import (
    FastAPIClient,
    FastAPIUnavailableError,
    InferenceBusyError,
    InferenceFailedError,
    ModelNotReadyError,
    RateLimitExceededError,
    RequestValidationError,
)
from database import DatabaseConfig, DatabaseManager, adapt_sql_dialect

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [gradio_app] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gradio_app")

# ---------------------------------------------------------------------------
# Inference Mode: "api" (default, via FastAPI) or "direct" (HF Spaces ZeroGPU)
# ---------------------------------------------------------------------------
# On HF Spaces with ZeroGPU, GPU access is only available during @spaces.GPU
# decorated function calls in the main process. Since our FastAPI subprocess
# cannot receive GPU from ZeroGPU, we bypass it and do direct in-process
# inference when running on HF Spaces.
# ---------------------------------------------------------------------------

IS_HF_SPACE = bool(os.getenv("SPACE_ID"))
INFERENCE_MODE = os.getenv("INFERENCE_MODE", "direct" if IS_HF_SPACE else "api")

# Global FastAPI client instance (used in "api" mode)
API_BASE_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
fastapi_client = FastAPIClient(base_url=API_BASE_URL)

# Direct inference engine (used in "direct" mode on HF Spaces)
_direct_engine = None
_direct_engine_lock = None

def _get_direct_engine():
    """Lazy-load the Text2SQLEngine singleton for direct inference mode."""
    global _direct_engine
    if _direct_engine is None:
        from prediction import Text2SQLEngine
        logger.info("Direct mode: Loading Text2SQLEngine in-process...")
        _direct_engine = Text2SQLEngine()
        logger.info(f"Direct mode: Engine loaded on device='{_direct_engine.device}'")
    return _direct_engine

# ZeroGPU-decorated inference function (only active on HF Spaces)
try:
    import spaces as _spaces_module

    @_spaces_module.GPU
    def _gpu_generate_sql(question: str, schema: str, dialect: str = "sqlite") -> str:
        """Run inference with temporary ZeroGPU access."""
        import torch
        engine = _get_direct_engine()
        # Move model to GPU if ZeroGPU made CUDA available
        if torch.cuda.is_available() and engine.device != "cuda":
            engine.device = "cuda"
            engine.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            engine.model = engine.model.to(device=engine.device, dtype=engine.torch_dtype)
            logger.info(f"Direct mode: Moved model to {engine.device} ({engine.torch_dtype})")
        return engine.generate_sql(question, schema, dialect=dialect, max_new_tokens=256, temperature=0.0)

    logger.info("ZeroGPU @spaces.GPU decorator registered for direct inference.")
except (ImportError, Exception):
    _spaces_module = None

    def _gpu_generate_sql(question: str, schema: str, dialect: str = "sqlite") -> str:
        """Fallback: run inference on CPU without ZeroGPU."""
        engine = _get_direct_engine()
        return engine.generate_sql(question, schema, dialect=dialect, max_new_tokens=256, temperature=0.0)


def direct_generate_sql(question: str, schema: str, dialect: str = "sqlite") -> dict:
    """Direct inference wrapper that returns a response dict matching FastAPI format."""
    import time
    import uuid
    t0 = time.perf_counter()
    sql = _gpu_generate_sql(question, schema, dialect)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "sql": sql,
        "model": "text2sql-v1",
        "generation_time_ms": elapsed_ms,
        "request_id": str(uuid.uuid4()),
    }

SAMPLE_DB_PATH = "sample_company.db"

def get_repo_url() -> str:
    """Return repository URL, dynamic via REPO_URL environment variable with fallback to friend's upstream repository."""
    return os.getenv("REPO_URL", "https://github.com/here-2007/SQL_Engine")


REPO_URL = get_repo_url()


def get_top_banner_html(repo_url: str | None = None) -> str:
    url = repo_url or get_repo_url()
    return (
        '<div id="top-announcement-banner" class="terminal-banner" '
        'data-banner-text="You can run it locally for even Better Experience Github" '
        'aria-label="You can run it locally for even Better Experience Github">'
        '<div class="banner-content">'
        '<span class="banner-prompt">&gt;_</span>'
        '<span class="banner-text">'
        'You can run it locally for even Better Experience '
        f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="banner-repo-link" id="banner-repo-link">Github</a>'
        '</span>'
        '</div>'
        '<button type="button" id="banner-dismiss-btn" class="banner-close-btn" '
        'onclick="document.getElementById(\'top-announcement-banner\').style.display=\'none\'; '
        'var w = document.getElementById(\'top_announcement_banner_wrapper\'); if (w) w.style.display = \'none\'; '
        'document.querySelectorAll(\'.terminal-banner, .banner-wrapper\').forEach(function(el) { el.style.display = \'none\'; }); '
        'try { sessionStorage.setItem(\'dismiss_local_run_banner\', \'1\'); } catch (e) {}" '
        'aria-label="Dismiss banner" title="Dismiss banner">✕</button>'
        '</div>'
    )


def get_permanent_github_html(repo_url: str | None = None) -> str:
    url = repo_url or get_repo_url()
    return (
        f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
        'class="permanent-github-link" id="permanent-github-link" '
        'aria-label="GitHub Repository" title="GitHub Repository">'
        '<svg height="24" width="24" viewBox="0 0 16 16" fill="currentColor" class="github-icon" aria-hidden="true">'
        '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>'
        '</svg>'
        '</a>'
    )


TOP_BANNER_HTML = get_top_banner_html()
PERMANENT_GITHUB_HTML = get_permanent_github_html()

BANNER_DISMISS_SCRIPT = """
(function() {
    function dismissTopBanner() {
        try {
            sessionStorage.setItem('dismiss_local_run_banner', '1');
        } catch (e) {}
        var banner = document.getElementById('top-announcement-banner');
        if (banner) banner.style.display = 'none';
        var wrapper = document.getElementById('top_announcement_banner_wrapper');
        if (wrapper) wrapper.style.display = 'none';
        document.querySelectorAll('.terminal-banner, .banner-wrapper').forEach(function(el) {
            el.style.display = 'none';
        });
        try {
            if (!document.getElementById('banner-dismiss-style')) {
                var s = document.createElement('style');
                s.id = 'banner-dismiss-style';
                s.textContent = '.banner-wrapper, #top_announcement_banner_wrapper, .terminal-banner, #top-announcement-banner { display: none !important; }';
                document.head.appendChild(s);
            }
        } catch (e) {}
    }

    function initBannerDismiss() {
        try {
            if (sessionStorage.getItem('dismiss_local_run_banner') === '1') {
                dismissTopBanner();
            }
        } catch (e) {}

        if (window.__sql_engine_banner_listener_attached) return;
        window.__sql_engine_banner_listener_attached = true;

        document.addEventListener('click', function(e) {
            var target = e.target && e.target.nodeType === 3 ? e.target.parentElement : e.target;
            var btn = (target && target.closest) ? target.closest('#banner-dismiss-btn, .banner-close-btn') : null;
            if (!btn && target && (target.id === 'banner-dismiss-btn' || (target.classList && target.classList.contains('banner-close-btn')))) {
                btn = target;
            }
            if (btn) {
                e.preventDefault();
                e.stopPropagation();
                dismissTopBanner();
            }
        }, true);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initBannerDismiss);
    } else {
        initBannerDismiss();
    }
})();
"""

BANNER_DISMISS_HEAD = f"<script>{BANNER_DISMISS_SCRIPT}</script>"


# ---------------------------------------------------------------------------
# Database Utilities & Fixtures
# ---------------------------------------------------------------------------

def create_sample_sqlite_db(path: str = SAMPLE_DB_PATH) -> str:
    """Creates a realistic SQLite database fixture for immediate testing."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            location TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER,
            salary REAL NOT NULL,
            hire_date DATE,
            FOREIGN KEY (department_id) REFERENCES departments (id)
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY,
            employee_id INTEGER,
            amount REAL NOT NULL,
            sale_date DATE,
            FOREIGN KEY (employee_id) REFERENCES employees (id)
        );
    """)

    cur.executemany("INSERT OR IGNORE INTO departments VALUES (?, ?, ?);", [
        (1, "Engineering", "San Francisco"),
        (2, "Sales", "New York"),
        (3, "Marketing", "London"),
        (4, "Product", "Seattle"),
    ])
    cur.executemany("INSERT OR IGNORE INTO employees VALUES (?, ?, ?, ?, ?);", [
        (101, "Alice Chen", 1, 135000.0, "2021-03-15"),
        (102, "Bob Smith", 1, 115000.0, "2022-06-01"),
        (103, "Charlie Davis", 2, 88000.0, "2020-01-10"),
        (104, "Diana Prince", 2, 94000.0, "2021-11-20"),
        (105, "Evan Wright", 3, 76000.0, "2023-02-14"),
        (106, "Fiona Gallagher", 4, 120000.0, "2022-08-19"),
    ])
    cur.executemany("INSERT OR IGNORE INTO sales VALUES (?, ?, ?, ?);", [
        (1, 103, 16500.0, "2024-01-15"),
        (2, 104, 24000.0, "2024-02-10"),
        (3, 103, 19500.0, "2024-03-05"),
        (4, 104, 32000.0, "2024-03-22"),
        (5, 103, 14200.0, "2024-04-12"),
    ])
    conn.commit()
    conn.close()
    return path


def format_log_entry(message: str) -> str:
    """Format a timestamped log line for the UI console."""
    ts = datetime.now().strftime("%H:%M:%S")
    return f"[{ts}] {message}"


def get_initial_state() -> dict[str, Any]:
    """Returns the default uninitialized application state."""
    return {
        "db_manager": None,
        "config": None,
        "is_connected": False,
        "db_type": None,
        "database_name": "None",
        "table_names": [],
        "table_count": 0,
        "schema": "",
        "dialect": None,
        "last_sql": "",
        "last_metadata": {},
        "logs": [
            format_log_entry("Application initialized."),
            format_log_entry("FastAPI Target: " + API_BASE_URL),
            format_log_entry("Ready. Connect a database in 'Set Database' to begin."),
        ],
    }


def _safe_port(val: Any, default: int | None = None) -> int | None:
    """Safely parse a port value into an integer, falling back to default on error."""
    if val is None:
        return default
    try:
        s = str(val).strip()
        return int(s) if s else default
    except (ValueError, TypeError):
        return default


def redact_credentials(text: str, password: str | None = None) -> str:
    """
    Redacts sensitive credentials, passwords, and connection URIs from strings
    destined for terminal logs, UI status banners, or server error messages.
    """
    if not text or not isinstance(text, str):
        return str(text) if text is not None else ""

    sanitized = text

    # Redact explicit password if supplied
    if password and isinstance(password, str) and password.strip():
        sanitized = sanitized.replace(password, "••••••")
        try:
            from urllib.parse import quote_plus
            sanitized = sanitized.replace(quote_plus(password), "••••••")
        except Exception:
            pass

    # Redact URI passwords (e.g. postgresql://user:pass@host:5432/db)
    sanitized = re.sub(r"://([^:@\s/]+):([^@\s/]+)@", r"://\1:••••••@", sanitized)

    # Redact key-value password assignments (password=..., pass=..., pwd=...)
    sanitized = re.sub(
        r"\b(password|passwd|pwd|pass)\s*=\s*([\'\"][^\'\"]*[\'\"]|[^\s;,&]+)",
        r"\1=••••••",
        sanitized,
        flags=re.IGNORECASE,
    )

    # Redact JSON style "password": "..."
    sanitized = re.sub(
        r'([\'"](password|passwd|pwd|pass)[\'"]\s*:\s*)([\'"][^\'"]*[\'"])',
        r'\1"••••••"',
        sanitized,
        flags=re.IGNORECASE,
    )

    # Redact Supabase Personal Access Tokens (sbp_...)
    sanitized = re.sub(r'\bsbp_[a-zA-Z0-9_]+\b', 'sbp_••••••••', sanitized)

    # Redact Bearer / apikey auth tokens
    sanitized = re.sub(r'\b(Bearer|apikey)\s+([a-zA-Z0-9_\-\.]+)', r'\1 ••••••', sanitized, flags=re.IGNORECASE)

    return sanitized


# ---------------------------------------------------------------------------
# Custom CSS Layer - Terminal Green Developer Tool Aesthetic
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Developer Workstation Dark Theme */
:root {
    --bg-main: #070a09;
    --bg-card: #0b110f;
    --bg-card-hover: #0e1714;
    --border-green: #15803d;
    --border-bright: #22c55e;
    --border-subtle: #143825;
    --text-primary: #ecfdf5;
    --text-accent: #4ade80;
    --text-muted: #6ee7b7;
    --text-dim: #059669;
    --font-mono: ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace;
}

body, .gradio-container {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
}

/* Header status cards */
.status-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-green) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    box-shadow: 0 0 10px rgba(22, 163, 74, 0.1) !important;
    transition: all 0.2s ease-in-out;
}
.status-card:hover {
    border-color: var(--border-bright) !important;
    box-shadow: 0 0 14px rgba(34, 197, 94, 0.2) !important;
}

.status-label {
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-dim) !important;
    margin-bottom: 4px !important;
}

.status-value {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: var(--text-accent) !important;
}

/* Nav & Primary Buttons */
.btn-primary-green {
    background-color: #166534 !important;
    color: #f0fdf4 !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: var(--font-mono) !important;
    transition: all 0.2s !important;
}
.btn-primary-green:hover {
    background-color: #15803d !important;
    box-shadow: 0 0 12px rgba(34, 197, 94, 0.4) !important;
}

.btn-secondary-green {
    background-color: #0b1411 !important;
    color: var(--text-accent) !important;
    border: 1px solid var(--border-green) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    transition: all 0.2s !important;
}
.btn-secondary-green:hover {
    border-color: var(--border-bright) !important;
    background-color: #13221c !important;
}

/* Terminal container */
.terminal-panel {
    background: #050807 !important;
    border: 1px solid var(--border-green) !important;
    border-radius: 10px !important;
    padding: 14px !important;
    box-shadow: inset 0 0 16px rgba(0, 0, 0, 0.8) !important;
}

/* Textboxes and Inputs */
textarea, input[type="text"], input[type="password"], input[type="number"], .gr-input {
    background-color: #080d0b !important;
    color: #f0fdf4 !important;
    border: 1px solid var(--border-green) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
}
textarea:focus, input:focus {
    border-color: var(--border-bright) !important;
    outline: none !important;
    box-shadow: 0 0 8px rgba(34, 197, 94, 0.3) !important;
}

/* Logs panel */
.logs-box textarea {
    background-color: #040706 !important;
    color: #4ade80 !important;
    border: 1px solid var(--border-green) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    line-height: 1.4 !important;
}

/* Tab bar */
.tabs > .tab-nav, div[role="tablist"] {
    border-bottom: 1px solid var(--border-subtle) !important;
}
.tab-nav button, div[role="tablist"] button {
    font-family: var(--font-mono) !important;
    font-weight: 600 !important;
    color: #86efac !important;
}
.tab-nav button.selected, div[role="tablist"] button[aria-selected="true"], div[role="tablist"] button.selected {
    color: #4ade80 !important;
    border-bottom: 2px solid var(--border-bright) !important;
}

/* Header layout */
.header-row {
    align-items: stretch !important;
    gap: 10px !important;
    margin-bottom: 12px !important;
}
.nav-button-container {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.header-nav-btn {
    height: 100% !important;
    min-height: 52px !important;
    width: 100% !important;
}

/* Permanent Top-Right GitHub Logo */
#permanent_github_logo,
.permanent-github-container {
    position: fixed !important;
    top: 14px !important;
    right: 18px !important;
    z-index: 9999 !important;
    width: auto !important;
    height: auto !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;
    min-width: 0 !important;
    pointer-events: none !important;
}

.permanent-github-link {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 38px !important;
    height: 38px !important;
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-green) !important;
    border-radius: 8px !important;
    color: var(--text-accent) !important;
    text-decoration: none !important;
    box-shadow: 0 0 10px rgba(22, 163, 74, 0.15) !important;
    transition: all 0.2s ease-in-out !important;
    cursor: pointer !important;
    pointer-events: auto !important;
}

.permanent-github-link:hover {
    border-color: var(--border-bright) !important;
    background-color: var(--bg-card-hover) !important;
    color: #f0fdf4 !important;
    box-shadow: 0 0 14px rgba(34, 197, 94, 0.4) !important;
    transform: translateY(-1px) !important;
}

.permanent-github-link svg,
.permanent-github-link .github-icon {
    width: 22px !important;
    height: 22px !important;
    fill: currentColor !important;
    transition: transform 0.2s ease-in-out !important;
}

.permanent-github-link:hover svg,
.permanent-github-link:hover .github-icon {
    transform: scale(1.08) !important;
}

/* Top Announcement Banner (Dismissible) */
.banner-wrapper,
#top_announcement_banner_wrapper {
    margin: 0 0 12px 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}

.terminal-banner,
#top-announcement-banner {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    background: linear-gradient(90deg, #09140f 0%, #0d1e16 50%, #09140f 100%) !important;
    border: 1px solid var(--border-green) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    margin-right: 56px !important;
    box-shadow: 0 0 14px rgba(22, 163, 74, 0.15), inset 0 0 12px rgba(0, 0, 0, 0.5) !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
    color: var(--text-primary) !important;
    transition: all 0.25s ease-in-out !important;
}

.terminal-banner:hover,
#top-announcement-banner:hover {
    border-color: var(--border-bright) !important;
    box-shadow: 0 0 18px rgba(34, 197, 94, 0.25), inset 0 0 12px rgba(0, 0, 0, 0.4) !important;
}

.banner-content {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    flex-grow: 1 !important;
    overflow: hidden !important;
}

.banner-prompt {
    color: var(--text-accent) !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    text-shadow: 0 0 6px rgba(74, 222, 128, 0.5) !important;
    user-select: none !important;
}

.banner-text {
    color: #d1fae5 !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
}

.banner-repo-link,
#banner-repo-link {
    color: var(--text-accent) !important;
    font-weight: 700 !important;
    text-decoration: underline !important;
    text-underline-offset: 3px !important;
    transition: all 0.15s ease-in-out !important;
}

.banner-repo-link:hover,
#banner-repo-link:hover {
    color: #86efac !important;
    text-shadow: 0 0 8px rgba(74, 222, 128, 0.6) !important;
}

.banner-close-btn,
#banner-dismiss-btn {
    position: relative !important;
    z-index: 101 !important;
    pointer-events: auto !important;
    cursor: pointer !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--text-muted) !important;
    font-size: 14px !important;
    line-height: 1 !important;
    border-radius: 6px !important;
    padding: 4px 8px !important;
    margin-left: 12px !important;
    transition: all 0.2s ease-in-out !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.banner-close-btn:hover,
#banner-dismiss-btn:hover {
    background-color: rgba(239, 68, 68, 0.15) !important;
    border-color: rgba(239, 68, 68, 0.4) !important;
    color: #f87171 !important;
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.3) !important;
}

@media (max-width: 640px) {
    #permanent_github_logo,
    .permanent-github-container {
        top: 10px !important;
        right: 10px !important;
    }
    .permanent-github-link {
        width: 32px !important;
        height: 32px !important;
    }
    .permanent-github-link svg,
    .permanent-github-link .github-icon {
        width: 18px !important;
        height: 18px !important;
    }
    .terminal-banner,
    #top-announcement-banner {
        margin-right: 44px !important;
        font-size: 11px !important;
        padding: 8px 10px !important;
    }
}
"""


# ---------------------------------------------------------------------------
# Business Logic Handlers
# ---------------------------------------------------------------------------

def refresh_health() -> tuple[str, str]:
    """Query health status and return UI status and details strings."""
    if INFERENCE_MODE == "direct":
        # In direct mode, report engine status (no FastAPI to query)
        if _direct_engine is not None:
            dev = getattr(_direct_engine, "device", "cpu").upper()
            return "● Model Healthy", f"Device: {dev} | Model: text2sql-v1 | Mode: Direct"
        else:
            return "○ Model Loading", "Direct inference mode — model loads on first query"

    data = fastapi_client.health()
    st = data.get("status", "offline")
    dev = data.get("device", "none")
    ver = data.get("model_version", "unknown")

    if st == "healthy":
        status_md = "● Model Healthy"
        detail_md = f"Device: {dev.upper()} | Model: {ver}"
    elif st == "degraded":
        status_md = "○ Model Degraded"
        err = data.get("error") or "Unknown error"
        detail_md = f"Degraded ({err[:30]}...)"
    else:
        status_md = "✕ FastAPI Offline"
        detail_md = f"Target: {fastapi_client.base_url}"

    return status_md, detail_md


def switch_db_type(db_type: str, saved_profiles_json: str | None = None) -> tuple[Any, Any, Any, Any, Any, str]:
    """Dynamically adjust field visibility, interactability, and defaults based on DB type."""
    normalized = (db_type or "sqlite").strip().lower()

    profile: dict[str, Any] = {}
    if saved_profiles_json and isinstance(saved_profiles_json, str):
        try:
            data = json.loads(saved_profiles_json)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(k, str) and k.strip().lower() == normalized and isinstance(v, dict):
                        profile = v
                        break
        except Exception:
            profile = {}

    if normalized == "sqlite":
        db_val = profile.get("database") if profile.get("database") is not None else SAMPLE_DB_PATH
        return (
            gr.update(visible=False, interactive=False, value=""),  # username
            gr.update(visible=False, interactive=False, value=""),  # host
            gr.update(visible=False, interactive=False, value=None),  # port
            gr.update(visible=False, interactive=False, value=""),  # password
            gr.update(label="Database File Path", placeholder="e.g. sample_company.db or chinook.db", value=db_val),
            "SQLite mode: Enter the database file path. Credentials are not required.",
        )
    elif normalized == "postgresql":
        user_val = profile.get("username") or ""
        host_val = profile.get("host") or ""
        port_val = _safe_port(profile.get("port"), 5432)
        pw_val = profile.get("password") or ""
        db_val = profile.get("database") or ""
        return (
            gr.update(visible=True, interactive=True, value=user_val, placeholder="postgres"),
            gr.update(visible=True, interactive=True, value=host_val, placeholder="localhost"),
            gr.update(visible=True, interactive=True, value=port_val),
            gr.update(visible=True, interactive=True, value=pw_val, placeholder="••••••••"),
            gr.update(label="Database Name", value=db_val, placeholder="e.g. company_db"),
            "PostgreSQL mode: Enter host, port (default 5432), database name, and credentials.",
        )
    elif normalized == "mysql":
        user_val = profile.get("username") or ""
        host_val = profile.get("host") or ""
        port_val = _safe_port(profile.get("port"), 3306)
        pw_val = profile.get("password") or ""
        db_val = profile.get("database") or ""
        return (
            gr.update(visible=True, interactive=True, value=user_val, placeholder="root"),
            gr.update(visible=True, interactive=True, value=host_val, placeholder="localhost"),
            gr.update(visible=True, interactive=True, value=port_val),
            gr.update(visible=True, interactive=True, value=pw_val, placeholder="••••••••"),
            gr.update(label="Database Name", value=db_val, placeholder="e.g. company_db"),
            "MySQL mode: Enter host, port (default 3306), database name, and credentials.",
        )
    elif normalized in ("supabase (api)", "supabase_api", "supabase-api"):
        pw_val = profile.get("password") or ""
        db_val = profile.get("database") or ""
        return (
            gr.update(visible=False, interactive=False, value=""),  # username
            gr.update(visible=False, interactive=False, value=""),  # host
            gr.update(visible=False, interactive=False, value=None),  # port
            gr.update(visible=True, interactive=True, value=pw_val, placeholder="anon / service_role key or Personal Access Token (sbp_...)"),
            gr.update(label="Supabase Project URL or Ref ID", value=db_val, placeholder="e.g. https://<project-ref>.supabase.co or <project-ref>"),
            "Supabase (API) mode: Enter your Supabase Project URL (or Ref ID) and API Key or Personal Access Token (sbp_...). Direct database password, host, and port are not required!",
        )
    elif normalized in ("supabase", "supabase (direct)", "supabase-direct"):
        user_val = profile.get("username") or "postgres"
        host_val = profile.get("host") or ""
        port_val = _safe_port(profile.get("port"), 5432)
        pw_val = profile.get("password") or ""
        db_val = profile.get("database") or "postgres"
        return (
            gr.update(visible=True, interactive=True, value=user_val, placeholder="postgres"),
            gr.update(visible=True, interactive=True, value=host_val, placeholder="e.g. db.<ref>.supabase.co or aws-0-xx.pooler.supabase.com"),
            gr.update(visible=True, interactive=True, value=port_val),
            gr.update(visible=True, interactive=True, value=pw_val, placeholder="••••••••"),
            gr.update(label="Database Name", value=db_val, placeholder="postgres"),
            "Supabase mode: Enter Supabase host (direct db.<project-ref>.supabase.co or connection pooler), port (default 5432), database name (default 'postgres'), username (default 'postgres'), and password. SSL is automatically enforced (sslmode=require).",
        )
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        "",
    )


def handle_connect(
    db_type: str,
    database: str,
    host: str | None,
    port: Any | None,
    username: str | None,
    password: str | None,
    state: dict[str, Any],
) -> tuple[str, str, str, str, str, dict[str, Any]]:
    """
    Validates credentials, establishes a real connection via DatabaseManager,
    introspects table names and schema, and updates state and terminal logs.
    """
    cleaned_type = (db_type or "sqlite").strip().lower()
    cleaned_db = (database or "").strip().strip("'\"")
    cleaned_host = (host or "").strip() or None
    cleaned_user = (username or "").strip() or None
    cleaned_pw = password if password is not None and str(password).strip() else None

    # Parse port safely
    port_val: int | None = None
    if port is not None and str(port).strip():
        try:
            port_val = int(str(port).strip())
        except ValueError:
            port_val = None

    logs = list(state.get("logs", []))
    logs.append(format_log_entry(f"Initiating connection to {cleaned_type.upper()}..."))

    # SQLite-specific validation
    if cleaned_type == "sqlite":
        if not cleaned_db:
            err_msg = "Database file path is required for SQLite."
            logs.append(format_log_entry(f"Validation failed: {err_msg}"))
            return (
                format_conn_status(state),
                state.get("database_name", "None"),
                format_table_count(state),
                "\n".join(logs),
                f"⚠️ {err_msg}",
                state,
            )
        if not os.path.exists(cleaned_db):
            logs.append(format_log_entry(f"Notice: SQLite file '{cleaned_db}' does not exist on disk (new database will be created)."))
        config = DatabaseConfig(db_type="sqlite", database=cleaned_db)
    elif cleaned_type in ("supabase (api)", "supabase_api", "supabase-api"):
        if not cleaned_db or cleaned_pw is None or not str(cleaned_pw).strip():
            err_msg = "Project URL/Ref ID and API Key/Token are required for Supabase (API)."
            logs.append(format_log_entry(f"Validation failed: {err_msg}"))
            return (
                format_conn_status(state),
                state.get("database_name", "None"),
                format_table_count(state),
                "\n".join(logs),
                f"⚠️ {err_msg}",
                state,
            )
        config = DatabaseConfig(
            db_type="supabase_api",
            database=cleaned_db,
            password=cleaned_pw,
        )
    elif cleaned_type in ("postgresql", "mysql", "supabase"):
        if cleaned_type == "supabase":
            cleaned_db = cleaned_db or "postgres"
            cleaned_user = cleaned_user or "postgres"
            port_val = port_val or 5432
            if not cleaned_host or cleaned_pw is None:
                err_msg = "Host and password are required for supabase."
                logs.append(format_log_entry(f"Validation failed: {err_msg}"))
                return (
                    format_conn_status(state),
                    state.get("database_name", "None"),
                    format_table_count(state),
                    "\n".join(logs),
                    f"⚠️ {err_msg}",
                    state,
                )
            config = DatabaseConfig(
                db_type="supabase",
                database=cleaned_db,
                host=cleaned_host,
                port=port_val,
                username=cleaned_user,
                password=cleaned_pw,
            )
        else:
            if not cleaned_db:
                err_msg = f"Database name is required for {cleaned_type}."
                logs.append(format_log_entry(f"Validation failed: {err_msg}"))
                return (
                    format_conn_status(state),
                    state.get("database_name", "None"),
                    format_table_count(state),
                    "\n".join(logs),
                    f"⚠️ {err_msg}",
                    state,
                )
            if not cleaned_host or not cleaned_user or cleaned_pw is None:
                err_msg = f"Host, username, and password are required for {cleaned_type}."
                logs.append(format_log_entry(f"Validation failed: {err_msg}"))
                return (
                    format_conn_status(state),
                    state.get("database_name", "None"),
                    format_table_count(state),
                    "\n".join(logs),
                    f"⚠️ {err_msg}",
                    state,
                )
            default_port = 5432 if cleaned_type == "postgresql" else 3306
            config = DatabaseConfig(
                db_type=cleaned_type,
                database=cleaned_db,
                host=cleaned_host,
                port=port_val or default_port,
                username=cleaned_user,
                password=cleaned_pw,
            )
    else:
        err_msg = f"Unsupported database type: {cleaned_type}"
        logs.append(format_log_entry(f"Error: {err_msg}"))
        return (
            format_conn_status(state),
            state.get("database_name", "None"),
            format_table_count(state),
            "\n".join(logs),
            f"⚠️ {err_msg}",
            state,
        )

    # Attempt connection and schema extraction
    try:
        new_manager = DatabaseManager(config)
        new_manager.connect()
        tables = new_manager.get_table_names()
        schema_text = new_manager.get_schema()

        # Update logs safely without credentials
        safe_db = redact_credentials(cleaned_db, cleaned_pw)
        logs.append(format_log_entry(f"Connected successfully to {cleaned_type.upper()} ({safe_db})"))
        logs.append(format_log_entry(f"Tables discovered ({len(tables)}): {', '.join(tables) if tables else 'None'}"))
        logs.append(format_log_entry(f"Schema introspected ({len(schema_text)} chars). Cached in workspace state."))

        dialect_name = (
            "postgresql"
            if getattr(new_manager, "is_api_mode", False)
            else (
                new_manager.engine.dialect.name
                if (new_manager.engine and hasattr(new_manager.engine, "dialect"))
                else cleaned_type
            )
        )

        # Update application state
        state["db_manager"] = new_manager
        state["config"] = config
        state["is_connected"] = True
        state["db_type"] = cleaned_type
        state["database_name"] = cleaned_db
        state["table_names"] = tables
        state["table_count"] = len(tables)
        state["schema"] = schema_text
        state["dialect"] = dialect_name
        state["logs"] = logs

        status_text = f"● Connected to {cleaned_type.title()}"
        tables_text = f"{len(tables)} tables"
        return (
            status_text,
            cleaned_db,
            tables_text,
            "\n".join(logs),
            f"✓ Connected to {cleaned_type.title()} ({safe_db})",
            state,
        )

    except Exception as exc:
        # Preserve previous valid connection on failure
        sanitized_err = redact_credentials(str(exc), cleaned_pw)
        logs.append(format_log_entry(f"Connection failed: {sanitized_err}"))
        state["logs"] = logs

        return (
            format_conn_status(state),
            state.get("database_name", "None"),
            format_table_count(state),
            "\n".join(logs),
            f"✗ Connection error: {sanitized_err}",
            state,
        )


def format_conn_status(state: dict[str, Any]) -> str:
    """Format connection status string from state."""
    if state.get("is_connected"):
        db_type = (state.get("db_type") or "SQL").title()
        return f"● Connected to {db_type}"
    return "○ No database connected"


def format_table_count(state: dict[str, Any]) -> str:
    """Format table count string from state."""
    count = state.get("table_count", 0)
    return f"{count} tables"


def handle_generate_sql(
    question: str,
    state: dict[str, Any],
) -> tuple[str, str, str, Any, dict[str, Any]]:
    """
    Sends natural language question and cached database schema to FastAPI /v1/tosql.
    Updates UI output terminal and metadata.
    """
    cleaned_question = (question or "").strip()
    mgr: DatabaseManager | None = state.get("db_manager")

    # Pre-flight check: database connection
    if not state.get("is_connected") or not state.get("schema") or mgr is None:
        msg = "⚠️ Database schema unavailable. Connect a database in 'Set Database' first."
        return (
            "-- No database schema available --\n-- Connect to a database in 'Set Database' to introspect schema.",
            "Metadata: Unavailable (Database disconnected)",
            msg,
            gr.update(interactive=False),  # Disable Run SQL
            state,
        )

    # Check connection liveness
    if not mgr.test_connection():
        state["is_connected"] = False
        msg = "⚠️ Database connection lost. Please reconnect in 'Set Database'."
        return (
            "-- Database connection lost --\n-- Please reconnect in 'Set Database'.",
            "Metadata: Connection Lost",
            msg,
            gr.update(interactive=False),
            state,
        )

    if not cleaned_question:
        msg = "⚠️ Please enter a question about your database."
        return (
            "-- Please enter a question above --",
            "Metadata: Ready",
            msg,
            gr.update(interactive=False),
            state,
        )

    schema = state.get("schema", "")
    active_dialect = state.get("dialect") or state.get("db_type") or "sqlite"

    try:
        if INFERENCE_MODE == "direct":
            logger.info(f"Direct inference: '{cleaned_question}' (dialect={active_dialect})")
            resp = direct_generate_sql(
                question=cleaned_question,
                schema=schema,
                dialect=active_dialect,
            )
        else:
            logger.info(f"Submitting question to FastAPI /v1/tosql: '{cleaned_question}' (dialect={active_dialect})")
            resp = fastapi_client.generate_sql(
                question=cleaned_question,
                schema=schema,
                dialect=active_dialect,
            )

        sql = resp.get("sql", "").strip()
        model_name = resp.get("model", "text2sql-v1")
        gen_time = resp.get("generation_time_ms", 0.0)
        req_id = resp.get("request_id", "n/a")

        state["last_sql"] = sql
        state["last_metadata"] = resp

        dialect_display = {
            "postgresql": "PostgreSQL",
            "postgres": "PostgreSQL",
            "supabase": "PostgreSQL",
            "mysql": "MySQL",
            "sqlite": "SQLite",
        }.get(str(active_dialect).lower().strip(), str(active_dialect).strip().title())

        meta_line = f"Model: {model_name}  |  Dialect: {dialect_display}  |  Generation Time: {gen_time} ms  |  Request ID: {req_id}"
        status_msg = f"✓ SQL generated successfully ({gen_time} ms)"

        return (
            sql,
            meta_line,
            status_msg,
            gr.update(interactive=True),  # Enable Run SQL
            state,
        )

    except RequestValidationError as exc:
        err_msg = f"Validation Error: {exc}"
        logger.warning(err_msg)
        return (
            f"-- Validation Error --\n-- {exc}",
            "Metadata: Validation Failure",
            f"⚠️ {exc}",
            gr.update(interactive=False),
            state,
        )
    except RateLimitExceededError as exc:
        err_msg = f"Rate limit exceeded: {exc}"
        logger.warning(err_msg)
        return (
            f"-- Rate Limited --\n-- {exc}",
            "Metadata: Rate Limit Exceeded (429)",
            "⚠️ Rate limit exceeded. Too many requests, please wait before submitting more queries.",
            gr.update(interactive=False),
            state,
        )
    except ModelNotReadyError as exc:
        err_msg = f"Model is currently unavailable: {exc}"
        logger.warning(err_msg)
        return (
            f"-- Service Degraded --\n-- {exc}",
            "Metadata: Model Not Ready",
            "⚠️ Model is currently unavailable. Please check FastAPI health.",
            gr.update(interactive=False),
            state,
        )
    except InferenceBusyError as exc:
        err_msg = f"Server busy: {exc}"
        logger.warning(err_msg)
        return (
            f"-- Server Busy --\n-- {exc}",
            "Metadata: Inference Timeout / Queue Full",
            "⚠️ SQL generation timed out. The server is busy, please try again.",
            gr.update(interactive=False),
            state,
        )
    except FastAPIUnavailableError as exc:
        err_msg = f"FastAPI service is unavailable: {exc}"
        logger.warning(err_msg)
        return (
            f"-- FastAPI Unavailable --\n-- Cannot connect to {fastapi_client.base_url}",
            "Metadata: FastAPI Offline",
            f"⚠️ Text-to-SQL service is unavailable at {fastapi_client.base_url}.",
            gr.update(interactive=False),
            state,
        )
    except InferenceFailedError as exc:
        err_msg = f"Inference execution failed: {exc}"
        logger.error(err_msg)
        return (
            f"-- Inference Error --\n-- {exc}",
            "Metadata: Model Inference Error",
            "⚠️ Model inference encountered an internal error.",
            gr.update(interactive=False),
            state,
        )
    except Exception as exc:
        err_msg = f"Unexpected error: {exc}"
        logger.error(err_msg, exc_info=True)
        return (
            f"-- Error --\n-- {exc}",
            "Metadata: Error",
            f"⚠️ Generation failed: {exc}",
            gr.update(interactive=False),
            state,
        )


def handle_run_sql(
    sql_text: str,
    state: dict[str, Any],
) -> tuple[Any, Any, str]:
    """
    Validates and executes generated SQL against the active database connection.
    Enforces read-only safety validation: only SELECT, WITH, and EXPLAIN are allowed.
    Renders results into a Gradio DataFrame.
    """
    query = (sql_text or state.get("last_sql") or "").strip()

    if not query or query.startswith("--"):
        return (
            gr.update(visible=False, value=pd.DataFrame()),
            gr.update(visible=False),
            "⚠️ No SQL query to execute. Generate a query first.",
        )

    mgr: DatabaseManager | None = state.get("db_manager")
    if mgr is None or not state.get("is_connected"):
        return (
            gr.update(visible=False, value=pd.DataFrame()),
            gr.update(visible=False),
            "⚠️ Database disconnected. Reconnect in 'Set Database'.",
        )

    # Check connection liveness
    if not mgr.test_connection():
        state["is_connected"] = False
        return (
            gr.update(visible=False, value=pd.DataFrame()),
            gr.update(visible=True, value="⚠️ Database connection lost."),
            "⚠️ Database connection lost. Please reconnect in 'Set Database'.",
        )

    # Apply dialect adaptation if query still contains SQLite-isms before sending to target DB
    active_dialect = state.get("dialect") or state.get("db_type") or "sqlite"
    query = adapt_sql_dialect(query, dialect=active_dialect)

    # 1. Safety validation via DatabaseManager
    try:
        mgr.validate_sql(query)
    except ValueError as exc:
        logger.warning(f"SQL validation blocked execution: {exc} | Query: {query}")
        return (
            gr.update(visible=False, value=pd.DataFrame()),
            gr.update(visible=True, value=f"⛔ Safety Block: {exc}"),
            f"⛔ Execution rejected: {exc}",
        )

    # 2. Execute safe read query
    try:
        res = mgr.execute_query(query, max_rows=500)
        cols = res.get("columns", [])
        rows = res.get("rows", [])
        row_count = res.get("row_count", 0)

        df = pd.DataFrame(rows, columns=cols)
        success_msg = f"✓ Query executed successfully: {row_count} row(s) returned."

        return (
            gr.update(visible=True, value=df),
            gr.update(visible=True, value=success_msg),
            success_msg,
        )
    except Exception as exc:
        sanitized_err = redact_credentials(str(exc))
        logger.error(f"Execution error on query '{query}': {sanitized_err}")
        return (
            gr.update(visible=False, value=pd.DataFrame()),
            gr.update(visible=True, value=f"✗ SQL Error: {sanitized_err}"),
            f"✗ Execution failed: {sanitized_err}",
        )


def handle_clear() -> tuple[str, str, str, str, Any, Any, Any]:
    """Clear question input, output terminal, status line, and query results."""
    return (
        "",  # question
        "-- Generated SQL will appear here --",  # code output
        "Metadata: Ready",  # meta line
        "Ready",  # status line
        gr.update(interactive=False),  # disable Run SQL
        gr.update(visible=False, value=pd.DataFrame()),  # results df
        gr.update(visible=False, value=""),  # results info
    )


def on_copy_sql(sql_text: str) -> str:
    """Provides user feedback when copying SQL to clipboard."""
    cleaned = (sql_text or "").strip()
    if not cleaned or cleaned.startswith("--"):
        return "⚠️ No SQL query to copy. Generate a query first."
    return "✓ SQL copied to clipboard"


def handle_load_sample(state: dict[str, Any]) -> tuple[str, str, Any, Any, Any, Any, str, str, str, str, str, dict[str, Any]]:
    """Loads and connects the built-in sample SQLite company database with 1 click."""
    db_path = create_sample_sqlite_db()
    st, db_n, tc, log_out, banner, new_state = handle_connect(
        db_type="sqlite",
        database=db_path,
        host=None,
        port=None,
        username=None,
        password=None,
        state=state,
    )
    return (
        "SQLite",  # db_type dropdown
        db_path,   # database name
        gr.update(visible=False, interactive=False, value=""),  # host
        gr.update(visible=False, interactive=False, value=None),  # port
        gr.update(visible=False, interactive=False, value=""),  # username
        gr.update(visible=False, interactive=False, value=""),  # password
        st,        # status
        db_n,      # db name
        tc,        # table count
        log_out,   # logs
        banner,    # connect status banner
        new_state, # updated state
    )


def populate_from_client_storage(
    storage_json: str,
    current_db_type: str = "SQLite",
) -> tuple[Any, Any, Any, Any, Any, str]:
    """
    Parses client-side localStorage payload and populates connection form fields
    for the selected database type via the hidden bridge component.
    """
    profiles: dict[str, Any] = {}
    if storage_json and isinstance(storage_json, str):
        try:
            data = json.loads(storage_json)
            if isinstance(data, dict):
                profiles = data
        except Exception:
            profiles = {}

    norm = (current_db_type or "sqlite").strip().lower()
    profile: dict[str, Any] = {}
    for k, v in profiles.items():
        if isinstance(k, str) and k.strip().lower() == norm and isinstance(v, dict):
            profile = v
            break

    if norm == "sqlite":
        db_val = profile.get("database") if profile.get("database") is not None else SAMPLE_DB_PATH
        return (
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(value=""),
            gr.update(value=db_val),
            storage_json or "{}",
        )
    elif norm in ("supabase (api)", "supabase_api", "supabase-api"):
        return (
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(value=profile.get("password") or ""),
            gr.update(value=profile.get("database") or ""),
            storage_json or "{}",
        )
    elif norm == "supabase":
        user_val = profile.get("username") or "postgres"
        db_val = profile.get("database") or "postgres"
        port_val = _safe_port(profile.get("port"), 5432)
        return (
            gr.update(value=user_val),
            gr.update(value=profile.get("host") or ""),
            gr.update(value=port_val),
            gr.update(value=profile.get("password") or ""),
            gr.update(value=db_val),
            storage_json or "{}",
        )
    elif norm == "postgresql":
        port_val = _safe_port(profile.get("port"), 5432)
        return (
            gr.update(value=profile.get("username") or ""),
            gr.update(value=profile.get("host") or ""),
            gr.update(value=port_val),
            gr.update(value=profile.get("password") or ""),
            gr.update(value=profile.get("database") or ""),
            storage_json or "{}",
        )
    elif norm == "mysql":
        port_val = _safe_port(profile.get("port"), 3306)
        return (
            gr.update(value=profile.get("username") or ""),
            gr.update(value=profile.get("host") or ""),
            gr.update(value=port_val),
            gr.update(value=profile.get("password") or ""),
            gr.update(value=profile.get("database") or ""),
            storage_json or "{}",
        )
    return (
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        storage_json or "{}",
    )


def handle_clear_credentials(db_type: str = "SQLite") -> tuple[Any, Any, Any, Any, Any, str, str]:
    """Clears form fields and resets bridge component when saved credentials are purged."""
    norm = (db_type or "sqlite").strip().lower()
    default_port = None if norm in ("sqlite", "supabase (api)", "supabase_api", "supabase-api") else (5432 if norm in ("postgresql", "supabase") else 3306)
    default_db = SAMPLE_DB_PATH if norm == "sqlite" else ("postgres" if norm == "supabase" else "")
    default_user = "postgres" if norm == "supabase" else ""

    return (
        gr.update(value=default_user),   # username
        gr.update(value=""),             # host
        gr.update(value=default_port),    # port
        gr.update(value=""),             # password
        gr.update(value=default_db),      # db name / path
        "✓ Saved browser credentials purged from localStorage.",  # banner
        "{}",                            # client_storage_bridge
    )


# ---------------------------------------------------------------------------
# Gradio Application Layout
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Build the complete Gradio interface for SQL Engine."""
    with gr.Blocks(title="Text-to-SQL Workstation") as demo:
        # Application state store
        state = gr.State(value=get_initial_state())

        # Hidden bridge component for client-side localStorage syncing
        client_storage_bridge = gr.Textbox(
            value="{}",
            visible=False,
            elem_id="client_storage_bridge",
        )

        # Permanent GitHub Logo (Top Right)
        _top_github_logo = gr.HTML(
            value=get_permanent_github_html(),
            elem_id="permanent_github_logo",
            elem_classes=["permanent-github-container"],
        )

        # Dismissible Top Announcement Banner
        _top_banner = gr.HTML(
            value=get_top_banner_html(),
            elem_id="top_announcement_banner_wrapper",
            elem_classes=["banner-wrapper"],
            head=BANNER_DISMISS_HEAD,
        )

        # Top Navigation & Status Bar (Matching Reference Screen 1)
        with gr.Row(elem_classes=["header-row"]):
            # Card 1: Connection Status with SQL
            with gr.Column(scale=3, elem_classes=["status-card"]):
                gr.Markdown("<div class='status-label'>Connection Status with SQL</div>")
                conn_status_md = gr.Markdown("○ No database connected", elem_classes=["status-value"])

            # Card 2: Database
            with gr.Column(scale=2, elem_classes=["status-card"]):
                gr.Markdown("<div class='status-label'>Database</div>")
                database_name_md = gr.Markdown("None", elem_classes=["status-value"])

            # Card 3: table
            with gr.Column(scale=2, elem_classes=["status-card"]):
                gr.Markdown("<div class='status-label'>table</div>")
                table_count_md = gr.Markdown("0 tables", elem_classes=["status-value"])

            # Card 4: Quick Navigation Action to Set Database
            with gr.Column(scale=2, min_width=140, elem_classes=["nav-button-container"]):
                btn_nav_set_db = gr.Button(
                    "⚙ Set Database",
                    elem_classes=["btn-primary-green", "header-nav-btn"],
                    size="lg",
                )

            # Card 5: Model health
            with gr.Column(scale=3, elem_classes=["status-card"]):
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown("<div class='status-label'>Model health</div>")
                        model_health_md = gr.Markdown("Checking...", elem_classes=["status-value"])
                        model_detail_md = gr.Markdown("Connecting...", elem_classes=["status-label"])
                    with gr.Column(scale=1, min_width=36):
                        btn_refresh_health = gr.Button("⟳", elem_classes=["btn-secondary-green"], size="sm")

        # Tabbed Logical Pages
        with gr.Tabs(selected="workspace") as tabs:
            # -------------------------------------------------------------------
            # PAGE 1: SQL WORKSPACE
            # -------------------------------------------------------------------
            with gr.Tab("SQL Workspace", id="workspace"):
                gr.Markdown("### Natural Language SQL Generation")

                # Question Input Area
                with gr.Group():
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="Ask a question about your database (e.g., 'What is the total revenue for the year 2024?' or 'Show average salary by department')...",
                        lines=3,
                        max_lines=6,
                    )
                    with gr.Row():
                        btn_generate = gr.Button(
                            "Generate SQL",
                            elem_classes=["btn-primary-green"],
                            size="lg",
                            interactive=False,
                        )
                        btn_clear = gr.Button(
                            "Clear",
                            elem_classes=["btn-secondary-green"],
                            size="lg",
                        )

                    status_line = gr.Markdown("Ready", elem_classes=["status-label"])

                # Output Area (Terminal Style)
                with gr.Group(elem_classes=["terminal-panel"]):
                    gr.Markdown("#### Output Terminal")
                    sql_output = gr.Code(
                        value="-- Generated SQL will appear here --",
                        language="sql",
                        lines=7,
                        label="Generated SQL",
                        interactive=False,
                    )
                    metadata_line = gr.Markdown(
                        "Metadata: Ready",
                        elem_classes=["status-label"],
                    )

                    with gr.Row():
                        btn_run_sql = gr.Button(
                            "▶ Run SQL",
                            elem_classes=["btn-primary-green"],
                            interactive=False,
                            size="md",
                        )
                        btn_copy_sql = gr.Button(
                            "📋 Copy SQL",
                            elem_classes=["btn-secondary-green"],
                            size="md",
                        )

                    # Query execution feedback and dataframe
                    execution_info = gr.Markdown(visible=False)
                    results_table = gr.DataFrame(
                        label="Query Results",
                        visible=False,
                        interactive=False,
                    )

            # -------------------------------------------------------------------
            # PAGE 2: SET DATABASE
            # -------------------------------------------------------------------
            with gr.Tab("Set Database", id="set_db"):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### Set Database")
                    with gr.Column(scale=1, min_width=200):
                        btn_back_to_workspace = gr.Button(
                            "← Back to SQL Workspace",
                            elem_classes=["btn-secondary-green"],
                            size="md",
                        )

                # Two-Column Credentials Layout (Matching Reference Screen 2)
                with gr.Row():
                    # Column 1
                    with gr.Column():
                        db_type_menu = gr.Dropdown(
                            choices=["SQLite", "PostgreSQL", "MySQL", "Supabase (API)", "Supabase"],
                            value="SQLite",
                            label="DB type(menu)",
                        )
                        username_input = gr.Textbox(
                            label="username",
                            placeholder="postgres",
                            visible=False,
                        )
                        port_input = gr.Textbox(
                            label="port",
                            placeholder="5432",
                            visible=False,
                        )

                    # Column 2
                    with gr.Column():
                        db_name_input = gr.Textbox(
                            label="DB Name / Path",
                            placeholder="e.g. sample_company.db",
                            value=SAMPLE_DB_PATH,
                        )
                        host_input = gr.Textbox(
                            label="host",
                            placeholder="localhost",
                            visible=False,
                        )
                        password_input = gr.Textbox(
                            label="password",
                            type="password",
                            placeholder="••••••••",
                            visible=False,
                        )

                db_mode_hint = gr.Markdown(
                    "SQLite mode: Enter database file path. Credentials are not required.",
                    elem_classes=["status-label"],
                )

                with gr.Row():
                    btn_connect = gr.Button(
                        "Connect",
                        elem_classes=["btn-primary-green"],
                        size="lg",
                    )
                    btn_sample_db = gr.Button(
                        "Load Sample SQLite DB",
                        elem_classes=["btn-secondary-green"],
                        size="lg",
                    )
                    btn_clear_creds = gr.Button(
                        "🗑️ Clear Saved Credentials",
                        elem_classes=["btn-secondary-green"],
                        size="lg",
                    )

                connect_banner = gr.Markdown(
                    "Ready to connect. Choose a database or click 'Load Sample SQLite DB'.",
                    elem_classes=["status-label"],
                )

                # Logs Area at Bottom (Matching Reference Screen 2)
                with gr.Group(elem_classes=["terminal-panel"]):
                    gr.Markdown("#### Logs")
                    logs_terminal = gr.Textbox(
                        label="Connection & Schema Logs",
                        lines=8,
                        max_lines=15,
                        value="\n".join(get_initial_state()["logs"]),
                        interactive=False,
                        elem_classes=["logs-box"],
                    )

        # -------------------------------------------------------------------
        # Event Bindings & Interactivity
        # -------------------------------------------------------------------

        # 1. Navigation Actions
        btn_nav_set_db.click(
            fn=lambda: gr.Tabs(selected="set_db"),
            outputs=[tabs],
            show_progress="hidden",
            js="() => { const b = document.querySelector('button[data-tab-id=\"set_db\"]'); if (b) b.click(); }",
        )
        btn_back_to_workspace.click(
            fn=lambda: gr.Tabs(selected="workspace"),
            outputs=[tabs],
            show_progress="hidden",
            js="() => { const b = document.querySelector('button[data-tab-id=\"workspace\"]'); if (b) b.click(); }",
        )

        # 2. Dynamic DB Type changes
        db_type_menu.change(
            fn=switch_db_type,
            inputs=[db_type_menu, client_storage_bridge],
            outputs=[
                username_input,
                host_input,
                port_input,
                password_input,
                db_name_input,
                db_mode_hint,
            ],
            js="""(db_type, bridge) => {
                try {
                    const raw = window.localStorage['sql_engine_client_connections'] || window.localStorage.getItem('sql_engine_client_connections') || '{}';
                    return [db_type, raw];
                } catch (err) {
                    console.error('Error reading localStorage on db change:', err);
                    return [db_type, bridge || '{}'];
                }
            }""",
        )

        # 3. Connect Button
        btn_connect.click(
            fn=handle_connect,
            inputs=[
                db_type_menu,
                db_name_input,
                host_input,
                port_input,
                username_input,
                password_input,
                state,
            ],
            outputs=[
                conn_status_md,
                database_name_md,
                table_count_md,
                logs_terminal,
                connect_banner,
                state,
            ],
            js="""(db_type, database, host, port, username, password, state) => {
                try {
                    let raw = window.localStorage['sql_engine_client_connections'] || window.localStorage.getItem('sql_engine_client_connections');
                    let profiles = raw ? JSON.parse(raw) : {};
                    if (typeof profiles !== 'object' || profiles === null || Array.isArray(profiles)) {
                        profiles = {};
                    }
                    if (db_type) {
                        profiles[db_type] = {
                            db_type: db_type,
                            database: database || '',
                            host: host || '',
                            port: port || '',
                            username: username || '',
                            password: password || ''
                        };
                        const serialized = JSON.stringify(profiles);
                        window.localStorage['sql_engine_client_connections'] = serialized;
                        window.localStorage.setItem('sql_engine_client_connections', serialized);
                        const bridge = document.querySelector('#client_storage_bridge textarea, #client_storage_bridge input');
                        if (bridge) {
                            bridge.value = serialized;
                            bridge.dispatchEvent(new Event('input', { bubbles: true }));
                        }
                    }
                } catch (err) {
                    console.error('Error saving credentials to localStorage:', err);
                }
                return [db_type, database, host, port, username, password, state];
            }""",
        ).then(
            fn=lambda s: gr.update(interactive=bool(s.get("is_connected") and s.get("schema"))),
            inputs=[state],
            outputs=[btn_generate],
        )

        # 3b. Clear Saved Credentials Button
        btn_clear_creds.click(
            fn=handle_clear_credentials,
            inputs=[db_type_menu],
            outputs=[
                username_input,
                host_input,
                port_input,
                password_input,
                db_name_input,
                connect_banner,
                client_storage_bridge,
            ],
            js="""(db_type) => {
                try {
                    delete window.localStorage['sql_engine_client_connections'];
                    window.localStorage.removeItem('sql_engine_client_connections');
                    const bridge = document.querySelector('#client_storage_bridge textarea, #client_storage_bridge input');
                    if (bridge) {
                        bridge.value = '{}';
                        bridge.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                } catch (err) {
                    console.error('Failed to clear localStorage:', err);
                }
                return [db_type];
            }""",
        )

        # 4. Load Sample SQLite Database Button
        btn_sample_db.click(
            fn=handle_load_sample,
            inputs=[state],
            outputs=[
                db_type_menu,
                db_name_input,
                host_input,
                port_input,
                username_input,
                password_input,
                conn_status_md,
                database_name_md,
                table_count_md,
                logs_terminal,
                connect_banner,
                state,
            ],
        ).then(
            fn=lambda s: gr.update(interactive=bool(s.get("is_connected") and s.get("schema"))),
            inputs=[state],
            outputs=[btn_generate],
        )

        # 5. Generate SQL Button
        btn_generate.click(
            fn=lambda: (gr.update(visible=False, value=pd.DataFrame()), gr.update(visible=False, value="")),
            outputs=[results_table, execution_info],
        ).then(
            fn=handle_generate_sql,
            inputs=[question_input, state],
            outputs=[
                sql_output,
                metadata_line,
                status_line,
                btn_run_sql,
                state,
            ],
        )

        # 6. Run SQL Button
        btn_run_sql.click(
            fn=handle_run_sql,
            inputs=[sql_output, state],
            outputs=[
                results_table,
                execution_info,
                status_line,
            ],
        )

        # 7. Copy SQL Action
        btn_copy_sql.click(
            fn=on_copy_sql,
            inputs=[sql_output],
            outputs=[status_line],
            js="(sql) => { if (sql && !sql.startsWith('--')) { navigator.clipboard.writeText(sql); } }",
        )

        # 8. Clear Button
        btn_clear.click(
            fn=handle_clear,
            outputs=[
                question_input,
                sql_output,
                metadata_line,
                status_line,
                btn_run_sql,
                results_table,
                execution_info,
            ],
        )

        # 9. Refresh Health Button & Initial Load
        btn_refresh_health.click(
            fn=refresh_health,
            outputs=[model_health_md, model_detail_md],
        )

        demo.load(
            fn=populate_from_client_storage,
            inputs=[client_storage_bridge, db_type_menu],
            outputs=[
                username_input,
                host_input,
                port_input,
                password_input,
                db_name_input,
                client_storage_bridge,
            ],
            js="""() => {
                try {
                    function dismissTopBanner() {
                        try {
                            sessionStorage.setItem('dismiss_local_run_banner', '1');
                        } catch (e) {}
                        var banner = document.getElementById('top-announcement-banner');
                        if (banner) banner.style.display = 'none';
                        var wrapper = document.getElementById('top_announcement_banner_wrapper');
                        if (wrapper) wrapper.style.display = 'none';
                        document.querySelectorAll('.terminal-banner, .banner-wrapper').forEach(function(el) {
                            el.style.display = 'none';
                        });
                        try {
                            if (!document.getElementById('banner-dismiss-style')) {
                                var s = document.createElement('style');
                                s.id = 'banner-dismiss-style';
                                s.textContent = '.banner-wrapper, #top_announcement_banner_wrapper, .terminal-banner, #top-announcement-banner { display: none !important; }';
                                document.head.appendChild(s);
                            }
                        } catch (e) {}
                    }

                    if (!window.__sql_engine_banner_listener_attached) {
                        window.__sql_engine_banner_listener_attached = true;
                        document.addEventListener('click', function(e) {
                            var target = e.target && e.target.nodeType === 3 ? e.target.parentElement : e.target;
                            var btn = (target && target.closest) ? target.closest('#banner-dismiss-btn, .banner-close-btn') : null;
                            if (!btn && target && (target.id === 'banner-dismiss-btn' || (target.classList && target.classList.contains('banner-close-btn')))) {
                                btn = target;
                            }
                            if (btn) {
                                e.preventDefault();
                                e.stopPropagation();
                                dismissTopBanner();
                            }
                        }, true);
                    }

                    if (sessionStorage.getItem('dismiss_local_run_banner') === '1') {
                        dismissTopBanner();
                    }
                } catch (bannerErr) {
                    console.error('Error initializing banner dismiss listener in demo.load:', bannerErr);
                }

                try {
                    const raw = window.localStorage['sql_engine_client_connections'] || window.localStorage.getItem('sql_engine_client_connections') || '{}';
                    return [raw, 'SQLite'];
                } catch (err) {
                    console.error('Error reading localStorage on load:', err);
                    return ['{}', 'SQLite'];
                }
            }""",
            show_progress="hidden",
        ).then(
            fn=refresh_health,
            outputs=[model_health_md, model_detail_md],
            show_progress="hidden",
        ).then(
            fn=lambda s: gr.update(interactive=bool(s.get("is_connected") and s.get("schema"))),
            inputs=[state],
            outputs=[btn_generate],
            show_progress="hidden",
        )

    return demo


def get_app_theme() -> gr.Theme:
    """Returns the custom emerald monospace developer workstation theme."""
    return gr.themes.Default(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.zinc,
        font=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    )


def launch(
    host: str | None = None,
    port: int | None = None,
    share: bool = False,
):
    """Launch the Gradio application."""
    # HF Spaces health probe needs 0.0.0.0; detect via SPACE_ID env var
    default_host = "0.0.0.0" if os.getenv("SPACE_ID") else "127.0.0.1"
    server_name = host or os.getenv("GRADIO_HOST", default_host)
    raw_port = port or os.getenv("GRADIO_PORT", 7860)
    server_port = int(raw_port)

    logger.info(f"Starting Gradio UI on {server_name}:{server_port}")
    logger.info(f"FastAPI Backend Target: {API_BASE_URL}")

    demo = build_app()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        css=CUSTOM_CSS,
        theme=get_app_theme(),
        head=BANNER_DISMISS_HEAD,
        js=BANNER_DISMISS_SCRIPT,
    )


if __name__ == "__main__":
    launch()
