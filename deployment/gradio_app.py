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

    function syncThemeUI(isDark) {
        if (isDark) {
            document.documentElement.classList.add('dark');
            document.documentElement.classList.remove('light');
            document.documentElement.setAttribute('data-theme', 'dark');
            if (document.body) {
                document.body.classList.add('dark');
                document.body.classList.remove('light');
                document.body.setAttribute('data-theme', 'dark');
            }
        } else {
            document.documentElement.classList.remove('dark');
            document.documentElement.classList.add('light');
            document.documentElement.setAttribute('data-theme', 'light');
            if (document.body) {
                document.body.classList.remove('dark');
                document.body.classList.add('light');
                document.body.setAttribute('data-theme', 'light');
            }
        }
        document.querySelectorAll('.snow-theme-toggle-btn, #snow-theme-toggle-btn').forEach(function(btn) {
            btn.textContent = isDark ? '☀️' : '🌙';
            btn.setAttribute('title', isDark ? 'Switch to Light Theme' : 'Switch to Dark Theme');
        });
    }

    function initTheme() {
        try {
            var saved = localStorage.getItem('color-theme');
            var prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            var isDark = saved === 'dark' || (!saved && prefersDark);
            syncThemeUI(isDark);
        } catch (e) {}
    }

    function setNavActive(mode) {
        var sideBtns = document.querySelectorAll('.snow-side-btn');
        sideBtns.forEach(function(btn) {
            btn.classList.remove('active');
        });
        if (mode === 'workspace' && sideBtns[0]) {
            sideBtns[0].classList.add('active');
        } else if (mode === 'set_db' && sideBtns[1]) {
            sideBtns[1].classList.add('active');
        } else if (mode === 'logs' && sideBtns[3]) {
            sideBtns[3].classList.add('active');
        }
        var breadcrumbCurrent = document.querySelector('.snow-breadcrumb-current');
        if (breadcrumbCurrent) {
            breadcrumbCurrent.textContent = (mode === 'set_db' || mode === 'logs') ? 'Database Manager' : 'Query Studio';
        }
    }

    window.__sql_engine_sync_theme = syncThemeUI;
    window.__sql_engine_init_theme = initTheme;
    window.__sql_engine_set_nav_active = setNavActive;

    function initBannerDismiss() {
        try {
            if (sessionStorage.getItem('dismiss_local_run_banner') === '1') {
                dismissTopBanner();
            }
        } catch (e) {}

        initTheme();

        if (window.__sql_engine_banner_listener_attached) return;
        window.__sql_engine_banner_listener_attached = true;

        // Global shortcut ⌘K / Ctrl+K and search submit on Enter
        document.addEventListener('keydown', function(e) {
            if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
                e.preventDefault();
                var searchInput = document.getElementById('snow-search-input') || document.querySelector('.snow-search-input');
                if (searchInput) {
                    searchInput.focus();
                    searchInput.select();
                }
            }
            if (e.key === 'Enter') {
                var searchInput = document.getElementById('snow-search-input') || document.querySelector('.snow-search-input');
                if (document.activeElement === searchInput && searchInput && searchInput.value.trim()) {
                    e.preventDefault();
                    var val = searchInput.value.trim();
                    var questionBox = document.querySelector('#question_input textarea') || document.querySelector('.snow-center-content textarea');
                    if (questionBox) {
                        var qText = val;
                        var low = val.toLowerCase();
                        if (low === 'employees' || low === 'departments' || low === 'sales') {
                            qText = 'Show all records from ' + low;
                        }
                        questionBox.value = qText;
                        questionBox.dispatchEvent(new Event('input', { bubbles: true }));
                        var wsTab = document.querySelector('button[data-tab-id="workspace"]');
                        if (wsTab) wsTab.click();
                        setNavActive('workspace');
                        questionBox.focus();
                        questionBox.style.transition = 'box-shadow 0.3s ease';
                        questionBox.style.boxShadow = '0 0 0 3px rgba(220, 38, 38, 0.45)';
                        setTimeout(function() { questionBox.style.boxShadow = ''; }, 1200);
                        searchInput.value = '';
                    }
                }
            }
        });

        document.addEventListener('click', function(e) {
            var target = e.target && e.target.nodeType === 3 ? e.target.parentElement : e.target;
            if (!target) return;
            
            // Banner dismiss button
            var btn = (target && target.closest) ? target.closest('#banner-dismiss-btn, .banner-close-btn') : null;
            if (!btn && target && (target.id === 'banner-dismiss-btn' || (target.classList && target.classList.contains('banner-close-btn')))) {
                btn = target;
            }
            if (btn) {
                e.preventDefault();
                e.stopPropagation();
                dismissTopBanner();
                return;
            }

            // Theme toggle button
            var themeBtn = (target && target.closest) ? target.closest('.snow-theme-toggle-btn, #snow-theme-toggle-btn') : null;
            if (themeBtn) {
                e.preventDefault();
                e.stopPropagation();
                var currentIsDark = document.documentElement.classList.contains('dark') || (document.body && document.body.classList.contains('dark'));
                var nextIsDark = !currentIsDark;
                try {
                    localStorage.setItem('color-theme', nextIsDark ? 'dark' : 'light');
                } catch (err) {}
                syncThemeUI(nextIsDark);
                return;
            }

            // Direct Gradio Tab button clicks -> sync sidebar and breadcrumb
            var tabBtn = target.closest ? target.closest('button[data-tab-id]') : null;
            if (tabBtn) {
                var tabId = tabBtn.getAttribute('data-tab-id');
                if (tabId === 'workspace' || tabId === 'set_db') {
                    setNavActive(tabId);
                }
            }

            // KPI card interactive clicks
            var kpiCard = target.closest ? target.closest('.status-card') : null;
            if (kpiCard) {
                if (kpiCard.classList.contains('kpi-card-coral')) {
                    var refBtn = document.querySelector('.snow-refresh-btn');
                    if (refBtn) refBtn.click();
                } else {
                    var setTab = document.querySelector('button[data-tab-id="set_db"]') || document.querySelector('.header-nav-btn');
                    if (setTab) setTab.click();
                    setNavActive('set_db');
                }
                return;
            }

            // Notifications card click -> switch to Set Database to check connection
            var notifCard = target.closest ? target.closest('.snow-notif-card') : null;
            if (notifCard) {
                var setTab2 = document.querySelector('button[data-tab-id="set_db"]');
                if (setTab2) setTab2.click();
                setNavActive('set_db');
                return;
            }

            // Activities card click -> jump to Activity & Logs
            var actCard = target.closest ? target.closest('.snow-activity-card') : null;
            if (actCard) {
                var sideBtns = document.querySelectorAll('.snow-side-btn');
                if (sideBtns && sideBtns[3]) sideBtns[3].click();
                return;
            }

            // Pro Tips card click -> trigger first chip
            var tipCard = target.closest ? target.closest('.snow-pro-tip-card') : null;
            if (tipCard) {
                var firstChip = document.querySelector('.snow-chip-btn');
                if (firstChip) firstChip.click();
                var wsTab2 = document.querySelector('button[data-tab-id="workspace"]');
                if (wsTab2) wsTab2.click();
                setNavActive('workspace');
                return;
            }
        }, true);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            initBannerDismiss();
        });
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
/* ==========================================================================
   Snow Dashboard UI Kit - Dual-Theme Architecture (Light & Dark)
   White Canvas in Light Mode, Obsidian in Dark Mode, Warm Yellow/Red in Both
   ========================================================================== */

:root, .light, html.light, body.light, [data-theme="light"] {
    --snow-bg-canvas: #FFFFFF;
    --snow-bg-card: #FFFFFF;
    --snow-bg-card-hover: #FFFDF5;
    --snow-border: #E5E7EB;
    --snow-border-hover: #FCD34D;
    --snow-border-focus: #DC2626;
    --snow-text-primary: #18181B;
    --snow-text-secondary: #3F3F46;
    --snow-text-muted: #71717A;
    --snow-accent: #DC2626;
    --snow-accent-hover: #B91C1C;
    --snow-accent-tint: #FEF2F2;
    --snow-accent-text: #DC2626;
    --snow-kpi-yellow-bg: #FEF9C3;
    --snow-kpi-yellow-border: #FDE047;
    --snow-kpi-yellow-text: #854D0E;
    --snow-kpi-red-bg: #FFE4E6;
    --snow-kpi-red-border: #FECDD3;
    --snow-kpi-red-text: #9F1239;
    --snow-kpi-amber-bg: #FEF3C7;
    --snow-kpi-amber-border: #FCD34D;
    --snow-kpi-coral-bg: #FEE2E2;
    --snow-kpi-coral-border: #FECACA;
    --snow-search-bg: #FFFDF5;
    --snow-search-border: #FDE68A;
    --snow-search-icon: #92400E;
    --snow-banner-bg: #FEF9C3;
    --snow-banner-border: #FDE68A;
    --snow-banner-text: #78350F;
    --snow-btn-sec-bg: #FEF9C3;
    --snow-btn-sec-border: #FDE047;
    --snow-btn-sec-text: #78350F;
    --snow-btn-sec-hover: #FEF08A;
    --snow-avatar-bg: linear-gradient(135deg, #FEF9C3 0%, #FFE4E6 100%);
    --snow-avatar-border: #FDE047;
    --snow-chip-bg: #FEF9C3;
    --snow-chip-border: #FDE047;
    --snow-chip-text: #854D0E;
    --snow-chip-hover-bg: #FEE2E2;
    --snow-chip-hover-border: #FCA5A5;
    --snow-chip-hover-text: #DC2626;
    --snow-quick-bg: linear-gradient(135deg, #FEF9C3 0%, #FFE4E6 100%);
    --snow-quick-border: #FDE68A;
    --snow-tab-hover-bg: #FEF9C3;
    --snow-tab-active-bg: #FEE2E2;
    --snow-font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --snow-font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    --snow-shadow-card: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 4px 12px -2px rgba(0, 0, 0, 0.03);
    --snow-shadow-hover: 0 4px 12px -1px rgba(220, 38, 38, 0.08);
}

:root.dark, html.dark, body.dark, .dark, .dark .gradio-container, body.dark .gradio-container, html.dark .gradio-container, [data-theme="dark"], [data-theme="dark"] .gradio-container {
    --snow-bg-canvas: #0B0F19;
    --snow-bg-card: #111827;
    --snow-bg-card-hover: #1F2937;
    --snow-border: #1F2937;
    --snow-border-hover: #D97706;
    --snow-border-focus: #EF4444;
    --snow-text-primary: #F9FAFB;
    --snow-text-secondary: #E5ECF6;
    --snow-text-muted: #9CA3AF;
    --snow-accent: #EF4444;
    --snow-accent-hover: #DC2626;
    --snow-accent-tint: #2B1117;
    --snow-accent-text: #FCA5A5;
    --snow-kpi-yellow-bg: #2B2105;
    --snow-kpi-yellow-border: #B45309;
    --snow-kpi-yellow-text: #FEF08A;
    --snow-kpi-red-bg: #311018;
    --snow-kpi-red-border: #9F1239;
    --snow-kpi-red-text: #FECDD3;
    --snow-kpi-amber-bg: #281905;
    --snow-kpi-amber-border: #D97706;
    --snow-kpi-coral-bg: #2D0F14;
    --snow-kpi-coral-border: #BE123C;
    --snow-search-bg: #111827;
    --snow-search-border: #374151;
    --snow-search-icon: #FBBF24;
    --snow-banner-bg: #2B2105;
    --snow-banner-border: #B45309;
    --snow-banner-text: #FEF08A;
    --snow-btn-sec-bg: #1F2937;
    --snow-btn-sec-border: #374151;
    --snow-btn-sec-text: #FEF08A;
    --snow-btn-sec-hover: #374151;
    --snow-avatar-bg: linear-gradient(135deg, #2B2105 0%, #311018 100%);
    --snow-avatar-border: #B45309;
    --snow-chip-bg: #2B2105;
    --snow-chip-border: #B45309;
    --snow-chip-text: #FEF08A;
    --snow-chip-hover-bg: #311018;
    --snow-chip-hover-border: #9F1239;
    --snow-chip-hover-text: #FECDD3;
    --snow-quick-bg: linear-gradient(135deg, #2B2105 0%, #311018 100%);
    --snow-quick-border: #B45309;
    --snow-tab-hover-bg: #2B2105;
    --snow-tab-active-bg: #311018;
    --snow-shadow-card: 0 1px 3px 0 rgba(0, 0, 0, 0.4), 0 4px 12px -2px rgba(0, 0, 0, 0.3);
    --snow-shadow-hover: 0 4px 12px -1px rgba(239, 68, 68, 0.15);
}

body, .gradio-container {
    background-color: var(--snow-bg-canvas) !important;
    color: var(--snow-text-primary) !important;
    font-family: var(--snow-font-sans) !important;
    letter-spacing: -0.01em !important;
}

.gradio-container {
    max-width: 1540px !important;
    padding: 16px 24px !important;
    margin: 0 auto !important;
}

/* -------------------------------------------------------------
   Snow Top Header Bar (Breadcrumbs, Search Bar, Header Actions)
   ------------------------------------------------------------- */
.snow-header-bar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    background: var(--snow-bg-card) !important;
    border: 1px solid var(--snow-border) !important;
    border-radius: 14px !important;
    padding: 8px 18px !important;
    margin-bottom: 16px !important;
    box-shadow: var(--snow-shadow-card) !important;
}

.snow-breadcrumb-nav {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 500;
}

.snow-breadcrumb-icon {
    font-size: 16px;
}

.snow-breadcrumb-root {
    color: var(--snow-text-muted);
}

.snow-breadcrumb-sep {
    color: #CBD5E1;
}

.snow-breadcrumb-current {
    color: var(--snow-text-primary);
    font-weight: 700;
}

.snow-search-box {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--snow-search-bg) !important;
    border: 1px solid var(--snow-search-border) !important;
    border-radius: 10px;
    padding: 6px 14px;
    width: 100%;
    max-width: 380px;
}

.snow-search-box .search-icon {
    color: var(--snow-search-icon) !important;
    flex-shrink: 0;
}

.snow-search-input {
    border: none !important;
    background: transparent !important;
    color: var(--snow-text-primary) !important;
    font-size: 12px !important;
    outline: none !important;
    width: 100% !important;
    padding: 0 !important;
    box-shadow: none !important;
}

.snow-kbd-shortcut {
    font-size: 10px;
    font-weight: 600;
    color: var(--snow-search-icon) !important;
    background: var(--snow-btn-sec-bg) !important;
    border: 1px solid var(--snow-search-border) !important;
    border-radius: 6px;
    padding: 2px 6px;
    flex-shrink: 0;
}

.snow-header-actions {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
}

.snow-header-actions-row {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 8px !important;
    width: auto !important;
}

.snow-header-actions-row > div,
.snow-header-actions-row > .block {
    min-width: 0 !important;
    width: auto !important;
    flex: 0 0 auto !important;
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

.nav-button-container {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.header-nav-btn {
    height: 36px !important;
    min-height: 36px !important;
    padding: 0 16px !important;
    font-size: 12px !important;
    width: auto !important;
    min-width: 140px !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}

.snow-refresh-btn {
    height: 36px !important;
    min-height: 36px !important;
    width: 36px !important;
    min-width: 36px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 15px !important;
    flex-shrink: 0 !important;
}

/* -------------------------------------------------------------
   Snow App Layout (3-Column Architecture)
   ------------------------------------------------------------- */
.snow-app-layout {
    display: flex !important;
    gap: 16px !important;
    align-items: flex-start !important;
}

/* Left Sidebar */
.snow-sidebar {
    background: var(--snow-bg-card) !important;
    border: 1px solid var(--snow-border) !important;
    border-radius: 16px !important;
    padding: 16px 12px !important;
    box-shadow: var(--snow-shadow-card) !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
}

.snow-sidebar > div,
.snow-sidebar > .block {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

.snow-profile-card {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 8px;
    margin-bottom: 4px;
}

.snow-avatar {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: var(--snow-avatar-bg) !important;
    border: 1px solid var(--snow-avatar-border) !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}

.snow-user-name {
    font-size: 13px;
    font-weight: 700;
    color: var(--snow-text-primary);
    line-height: 1.2;
}

.snow-user-badge {
    font-size: 11px;
    color: var(--snow-text-muted);
}

.snow-nav-divider {
    height: 1px;
    background: var(--snow-border);
    margin: 6px 0;
}

.snow-nav-group-title {
    font-size: 10px;
    font-weight: 700;
    color: var(--snow-text-muted);
    letter-spacing: 0.08em;
    padding: 4px 8px;
    text-transform: uppercase;
}

.snow-side-btn {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--snow-text-secondary) !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    width: 100% !important;
    min-height: 34px !important;
}

.snow-side-btn:hover {
    background: var(--snow-tab-hover-bg) !important;
    color: var(--snow-chip-text) !important;
}

.snow-side-btn.active {
    background: var(--snow-tab-active-bg) !important;
    color: var(--snow-accent) !important;
    font-weight: 600 !important;
    border-left: 3px solid var(--snow-accent) !important;
}

.snow-sidebar-footer {
    margin-top: 20px;
    padding: 8px 8px 2px 8px;
    border-top: 1px solid var(--snow-border);
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.snow-footer-tag {
    font-size: 11px;
    font-weight: 600;
    color: var(--snow-text-primary);
}

.snow-footer-sub {
    font-size: 10px;
    color: var(--snow-text-muted);
}

/* -------------------------------------------------------------
   4 Pastel KPI Cards (Alternating Lightish Yellow & Soft Red)
   ------------------------------------------------------------- */
.header-row,
.snow-kpi-grid {
    display: grid !important;
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
    gap: 12px !important;
    margin-bottom: 16px !important;
    width: 100% !important;
    align-items: stretch !important;
}

.snow-kpi-grid > .column,
.snow-kpi-grid > div {
    min-width: 0 !important;
    width: 100% !important;
    margin: 0 !important;
}

.status-card {
    border-radius: 16px !important;
    padding: 12px 14px !important;
    min-height: 94px !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 4px 10px -2px rgba(0, 0, 0, 0.03) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    cursor: pointer !important;
}

.status-card .block,
.status-card > div,
.status-card .prose {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

.status-card .prose p,
.status-card p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.25 !important;
}

.kpi-card-yellow,
.kpi-card-blue {
    background-color: var(--snow-kpi-yellow-bg) !important;
    border: 1px solid var(--snow-kpi-yellow-border) !important;
}

.kpi-card-red,
.kpi-card-periwinkle {
    background-color: var(--snow-kpi-red-bg) !important;
    border: 1px solid var(--snow-kpi-red-border) !important;
}

.kpi-card-amber {
    background-color: var(--snow-kpi-amber-bg) !important;
    border: 1px solid var(--snow-kpi-amber-border) !important;
}

.kpi-card-coral {
    background-color: var(--snow-kpi-coral-bg) !important;
    border: 1px solid var(--snow-kpi-coral-border) !important;
}

.status-card:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--snow-shadow-hover) !important;
}

.status-label {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--snow-text-muted) !important;
    margin-bottom: 2px !important;
    display: block !important;
}

.status-value {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: var(--snow-text-primary) !important;
    line-height: 1.25 !important;
    letter-spacing: -0.01em !important;
    font-variant-numeric: tabular-nums !important;
    word-break: break-word !important;
}

.status-value p {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: var(--snow-text-primary) !important;
    margin: 0 !important;
    line-height: 1.25 !important;
}

.kpi-card-yellow .status-value,
.kpi-card-yellow .status-value p,
.kpi-card-amber .status-value,
.kpi-card-amber .status-value p {
    color: var(--snow-kpi-yellow-text) !important;
}

.kpi-card-red .status-value,
.kpi-card-red .status-value p,
.kpi-card-coral .status-value,
.kpi-card-coral .status-value p,
.kpi-card-periwinkle .status-value,
.kpi-card-periwinkle .status-value p {
    color: var(--snow-kpi-red-text) !important;
}

.kpi-badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    background: rgba(245, 158, 11, 0.15) !important;
    color: var(--snow-kpi-yellow-text) !important;
    margin-top: 4px;
    width: fit-content;
}

.kpi-card-red .kpi-badge-pill,
.kpi-card-coral .kpi-badge-pill,
.kpi-card-periwinkle .kpi-badge-pill {
    background: rgba(239, 68, 68, 0.15) !important;
    color: var(--snow-kpi-red-text) !important;
}

/* -------------------------------------------------------------
   Snow Workspace Cards & Prompt Studio
   ------------------------------------------------------------- */
.snow-card,
.terminal-panel {
    background: var(--snow-bg-card) !important;
    border: 1px solid var(--snow-border) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: var(--snow-shadow-card) !important;
    margin-bottom: 16px !important;
}

/* Remove default Gradio block background inside .snow-card */
.snow-card > .block,
.snow-card .form,
.snow-card .group,
.snow-card .gr-group,
.snow-card .styler,
.snow-card > .styler,
.gr-group .styler,
.terminal-panel .styler,
.snow-card .prose,
.snow-card .markdown {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Prompt chips styling */
.chips-header {
    margin-bottom: 8px !important;
}

.chips-label {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: var(--snow-text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

.snow-chips-row {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-bottom: 12px !important;
}

.snow-chips-row > div,
.snow-chips-row > .block {
    min-width: 0 !important;
    width: auto !important;
    flex: 0 0 auto !important;
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

.snow-chip-btn {
    border-radius: 9999px !important;
    background: var(--snow-chip-bg) !important;
    border: 1px solid var(--snow-chip-border) !important;
    color: var(--snow-chip-text) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    padding: 5px 12px !important;
    white-space: nowrap !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    height: auto !important;
    min-height: 28px !important;
}

.snow-chip-btn:hover {
    background: var(--snow-chip-hover-bg) !important;
    border-color: var(--snow-chip-hover-border) !important;
    color: var(--snow-chip-hover-text) !important;
    transform: translateY(-1px) !important;
}

/* Primary Action Buttons - Vibrant Crimson Red */
.btn-primary-green {
    background-color: var(--snow-accent) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--snow-accent) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: var(--snow-font-sans) !important;
    font-size: 13px !important;
    box-shadow: 0 1px 3px 0 rgba(220, 38, 38, 0.25) !important;
    transition: all 0.15s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.btn-primary-green:hover {
    background-color: var(--snow-accent-hover) !important;
    border-color: var(--snow-accent-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 10px 0 rgba(220, 38, 38, 0.35) !important;
}

.btn-primary-green:active {
    transform: scale(0.98) !important;
}

.btn-primary-green:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.35) !important;
}

/* Secondary Buttons - Lightish Yellow / Slate Dark */
.btn-secondary-green {
    background-color: var(--snow-btn-sec-bg) !important;
    color: var(--snow-btn-sec-text) !important;
    border: 1px solid var(--snow-btn-sec-border) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: var(--snow-font-sans) !important;
    font-size: 13px !important;
    box-shadow: var(--snow-shadow-card) !important;
    transition: all 0.15s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.btn-secondary-green:hover {
    background-color: var(--snow-btn-sec-hover) !important;
    border-color: var(--snow-border-hover) !important;
    color: var(--snow-text-primary) !important;
    transform: translateY(-1px) !important;
}

.btn-secondary-green:active {
    transform: scale(0.98) !important;
}

/* Textboxes and Inputs */
textarea, input[type="text"], input[type="password"], input[type="number"], .gr-input {
    background-color: var(--snow-bg-card) !important;
    color: var(--snow-text-primary) !important;
    border: 1px solid var(--snow-border) !important;
    border-radius: 10px !important;
    font-family: var(--snow-font-sans) !important;
    font-size: 13px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

textarea:focus, input:focus, .gr-input:focus {
    border-color: var(--snow-border-focus) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.18) !important;
}

/* Code block & Output Display */
.gr-code, .cm-editor, pre {
    font-family: var(--snow-font-mono) !important;
    border-radius: 10px !important;
}

/* Logs panel */
.logs-box textarea {
    background-color: #0B0F19 !important;
    color: #38BDF8 !important;
    border: 1px solid #1E293B !important;
    font-family: var(--snow-font-mono) !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
    border-radius: 10px !important;
}

/* Tab Bar */
.tabs > .tab-nav, div[role="tablist"] {
    border-bottom: 1px solid var(--snow-border) !important;
    gap: 8px !important;
    margin-bottom: 16px !important;
}

.tab-nav button, div[role="tablist"] button {
    font-family: var(--snow-font-sans) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    color: var(--snow-text-secondary) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 16px !important;
    transition: all 0.15s ease !important;
}

.tab-nav button:hover, div[role="tablist"] button:hover {
    color: var(--snow-accent) !important;
    background: var(--snow-tab-hover-bg) !important;
}

.tab-nav button.selected, div[role="tablist"] button[aria-selected="true"], div[role="tablist"] button.selected {
    color: var(--snow-accent) !important;
    border-bottom: 2px solid var(--snow-accent) !important;
    background: var(--snow-tab-active-bg) !important;
}

/* Quick sample database box */
.snow-quick-sample-box {
    background: var(--snow-quick-bg) !important;
    border: 1px solid var(--snow-quick-border) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    margin-bottom: 18px !important;
}

/* -------------------------------------------------------------
   Right Rail (Activity, Notifications, Tips)
   ------------------------------------------------------------- */
.snow-right-rail {
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
}

.snow-right-rail > div,
.snow-right-rail > .block {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

.snow-rail-card {
    background: var(--snow-bg-card);
    border: 1px solid var(--snow-border);
    border-radius: 16px;
    padding: 14px 16px;
    box-shadow: var(--snow-shadow-card);
    margin-bottom: 10px;
}

.snow-rail-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--snow-text-muted);
    margin-bottom: 10px;
}

.snow-notif-item {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    margin-bottom: 8px;
}

.snow-notif-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}

.snow-notif-dot.blue,
.snow-notif-dot.red {
    background: #DC2626 !important;
}

.snow-notif-dot.green,
.snow-notif-dot.yellow {
    background: #EAB308 !important;
}

.snow-notif-msg {
    font-size: 12px;
    font-weight: 600;
    color: var(--snow-text-primary);
}

.snow-notif-meta {
    font-size: 11px;
    color: var(--snow-text-muted);
}

.snow-activity-item {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
    font-size: 12px;
    color: var(--snow-text-secondary);
}

.snow-tip-text {
    font-size: 12px;
    color: var(--snow-text-muted);
    line-height: 1.4;
    margin: 0;
}

/* Permanent GitHub Link */
#permanent_github_logo,
.permanent-github-container {
    position: fixed !important;
    top: 14px !important;
    right: 18px !important;
    z-index: 9999 !important;
    pointer-events: none !important;
}

.permanent-github-link {
    pointer-events: auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 38px !important;
    height: 38px !important;
    background-color: var(--snow-bg-card) !important;
    border: 1px solid var(--snow-border) !important;
    border-radius: 10px !important;
    color: var(--snow-text-secondary) !important;
    box-shadow: var(--snow-shadow-card) !important;
    text-decoration: none !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.permanent-github-link:hover {
    background-color: var(--snow-bg-card-hover) !important;
    color: var(--snow-accent) !important;
    box-shadow: var(--snow-shadow-hover) !important;
    transform: translateY(-1px) !important;
}

.permanent-github-link svg,
.permanent-github-link .github-icon {
    width: 20px !important;
    height: 20px !important;
    fill: currentColor !important;
    transition: transform 0.2s ease-in-out !important;
}

.permanent-github-link:hover svg,
.permanent-github-link:hover .github-icon {
    transform: scale(1.08) !important;
}

/* Top Announcement Banner (Dismissible - Lightish Yellow & Red) */
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
    background: var(--snow-banner-bg) !important;
    border: 1px solid var(--snow-banner-border) !important;
    border-radius: 10px !important;
    padding: 6px 14px !important;
    margin: 0 auto 12px auto !important;
    max-width: 900px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    font-family: var(--snow-font-sans) !important;
    font-size: 12px !important;
    color: var(--snow-banner-text) !important;
    transition: all 0.2s ease-in-out !important;
}

.terminal-banner:hover,
#top-announcement-banner:hover {
    border-color: var(--snow-border-hover) !important;
}

.banner-content {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    flex-grow: 1 !important;
    overflow: hidden !important;
}

.banner-prompt {
    color: var(--snow-accent) !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    user-select: none !important;
}

.banner-text {
    color: var(--snow-banner-text) !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
}

.banner-repo-link,
#banner-repo-link {
    color: var(--snow-accent) !important;
    font-weight: 700 !important;
    text-decoration: underline !important;
    text-underline-offset: 3px !important;
    transition: all 0.15s ease-in-out !important;
}

.banner-repo-link:hover,
#banner-repo-link:hover {
    color: var(--snow-accent-hover) !important;
}

.banner-close-btn,
#banner-dismiss-btn {
    position: relative !important;
    z-index: 101 !important;
    pointer-events: auto !important;
    cursor: pointer !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--snow-banner-text) !important;
    font-size: 13px !important;
    line-height: 1 !important;
    border-radius: 6px !important;
    padding: 3px 6px !important;
    margin-left: 12px !important;
    transition: all 0.15s ease-in-out !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.banner-close-btn:hover,
#banner-dismiss-btn:hover {
    background-color: var(--snow-accent-tint) !important;
    border-color: var(--snow-accent) !important;
    color: var(--snow-accent) !important;
}

.snow-card h1, .snow-card h2, .snow-card h3, .snow-card h4,
.terminal-panel h1, .terminal-panel h2, .terminal-panel h3, .terminal-panel h4,
.snow-center-content h1, .snow-center-content h2, .snow-center-content h3, .snow-center-content h4 {
    color: var(--snow-text-primary) !important;
}

.snow-card p, .terminal-panel p {
    color: var(--snow-text-secondary) !important;
}

label, .gr-form > label, .gr-box label, .block > label, span[data-testid="block-info"] {
    color: var(--snow-text-secondary) !important;
    font-weight: 600 !important;
}

.snow-theme-toggle-btn {
    cursor: pointer !important;
    font-size: 15px !important;
}

.snow-pro-tip-card {
    cursor: pointer !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

.snow-pro-tip-card:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--snow-shadow-hover) !important;
}

@media (max-width: 1100px) {
    .snow-app-layout {
        flex-direction: column !important;
    }
    .snow-sidebar, .snow-right-rail {
        width: 100% !important;
        max-width: 100% !important;
    }
    .snow-kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    }
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
        padding: 6px 8px !important;
    }
    .snow-kpi-grid {
        grid-template-columns: 1fr !important;
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
            gr.update(visible=False, interactive=False, value="", label="password", info=None),  # password
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
            gr.update(visible=True, interactive=True, value=pw_val, placeholder="••••••••", label="password", info=None),
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
            gr.update(visible=True, interactive=True, value=pw_val, placeholder="••••••••", label="password", info=None),
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
            gr.update(
                visible=True,
                interactive=True,
                value=pw_val,
                label="PAT",
                info="Personal Access Token (PAT) : [GO TO](https://supabase.com/dashboard/account/tokens)",
                placeholder="Personal Access Token (sbp_...)",
            ),
            gr.update(label="SUPABASE PROJECT ID", value=db_val, placeholder="e.g. adzuykgtwbajaktnesbk or https://<project-ref>.supabase.co"),
            "Supabase (API) mode: Enter your SUPABASE PROJECT ID and Personal Access Token (PAT, sbp_...). Direct database password, host, and port are not required!",
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
            gr.update(visible=True, interactive=True, value=pw_val, placeholder="••••••••", label="password", info=None),
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
            err_msg = "SUPABASE PROJECT ID and Personal Access Token (PAT) are required for Supabase (API)."
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

        # Snow Top Navigation Bar (Breadcrumbs, Search Bar, Header Actions)
        with gr.Row(elem_classes=["snow-header-bar"]):
            with gr.Column(scale=3, min_width=200, elem_classes=["snow-breadcrumbs"]):
                gr.HTML("""
                    <div class="snow-breadcrumb-nav">
                        <span class="snow-breadcrumb-icon">❄️</span>
                        <span class="snow-breadcrumb-root">Dashboards</span>
                        <span class="snow-breadcrumb-sep">/</span>
                        <span class="snow-breadcrumb-current">Query Studio</span>
                    </div>
                """)
            with gr.Column(scale=4, min_width=260, elem_classes=["snow-search-col"]):
                gr.HTML("""
                    <div class="snow-search-box">
                        <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                        <input type="text" class="snow-search-input" id="snow-search-input" placeholder="Search tables, columns, queries (press Enter to prompt)..." autocomplete="off" />
                        <span class="snow-kbd-shortcut" title="Press ⌘K or Ctrl+K to search">⌘K</span>
                    </div>
                """)
            with gr.Column(scale=2, min_width=180, elem_classes=["snow-header-actions"]):
                with gr.Row(elem_classes=["snow-header-actions-row"]):
                    btn_nav_set_db = gr.Button(
                        "⚙ Set Database",
                        elem_classes=["btn-primary-green", "header-nav-btn"],
                        size="sm",
                        min_width=0,
                    )
                    btn_refresh_health = gr.Button(
                        "⟳",
                        elem_classes=["btn-secondary-green", "snow-refresh-btn"],
                        size="sm",
                        min_width=0,
                    )
                    btn_theme_toggle = gr.Button(
                        "🌙",
                        elem_classes=["btn-secondary-green", "snow-refresh-btn", "snow-theme-toggle-btn"],
                        size="sm",
                        min_width=0,
                        elem_id="snow-theme-toggle-btn",
                    )

        # Main 3-Column Snow Dashboard Layout
        with gr.Row(elem_classes=["snow-app-layout"]):
            # -------------------------------------------------------------
            # COLUMN 1: LEFT NAVIGATION SIDEBAR
            # -------------------------------------------------------------
            with gr.Column(scale=1, min_width=180, elem_classes=["snow-sidebar"]):
                gr.HTML("""
                    <div class="snow-profile-card">
                        <div class="snow-avatar">❄️</div>
                        <div class="snow-user-info">
                            <div class="snow-user-name">SQL Engine</div>
                            <div class="snow-user-badge">Workstation v1.0</div>
                        </div>
                    </div>
                    <div class="snow-nav-divider"></div>
                    <div class="snow-nav-group-title">DASHBOARDS</div>
                """)
                btn_side_workspace = gr.Button("⚡ Query Studio", elem_classes=["snow-side-btn", "active"], size="sm")
                btn_side_set_db = gr.Button("🗄️ Database Manager", elem_classes=["snow-side-btn"], size="sm")
                gr.HTML("""
                    <div class="snow-nav-divider"></div>
                    <div class="snow-nav-group-title">PAGES</div>
                """)
                btn_side_sample = gr.Button("📦 Sample SQLite DB", elem_classes=["snow-side-btn"], size="sm")
                btn_side_logs = gr.Button("📋 Activity & Logs", elem_classes=["snow-side-btn"], size="sm")
                gr.HTML("""
                    <div class="snow-sidebar-footer">
                        <span class="snow-footer-tag">⚡ warm snow ui</span>
                        <span class="snow-footer-sub">Light Contrast • Yellow & Red</span>
                    </div>
                """)

            # -------------------------------------------------------------
            # COLUMN 2: CENTER WORKSPACE (MAIN DASHBOARD CONTENT)
            # -------------------------------------------------------------
            with gr.Column(scale=5, elem_classes=["snow-center-content"]):
                # 4 Pastel KPI Cards Row (Alternating Lightish Yellow & Soft Red)
                with gr.Row(elem_classes=["header-row", "snow-kpi-grid"]):
                    # Card 1: Connection Status (Lightish Yellow)
                    with gr.Column(scale=1, min_width=0, elem_classes=["status-card", "kpi-card-yellow"]):
                        gr.Markdown("<div class='status-label'>Connection Status with SQL</div>")
                        conn_status_md = gr.Markdown("○ No database connected", elem_classes=["status-value"])
                        gr.HTML("<div class='kpi-badge-pill'>● Database</div>")

                    # Card 2: Database (Soft Red)
                    with gr.Column(scale=1, min_width=0, elem_classes=["status-card", "kpi-card-red"]):
                        gr.Markdown("<div class='status-label'>Database</div>")
                        database_name_md = gr.Markdown("None", elem_classes=["status-value"])
                        gr.HTML("<div class='kpi-badge-pill'>Target Source</div>")

                    # Card 3: table count (Warm Amber Yellow)
                    with gr.Column(scale=1, min_width=0, elem_classes=["status-card", "kpi-card-amber"]):
                        gr.Markdown("<div class='status-label'>table</div>")
                        table_count_md = gr.Markdown("0 tables", elem_classes=["status-value"])
                        gr.HTML("<div class='kpi-badge-pill'>Schema Synced</div>")

                    # Card 4: Model health (Soft Coral Red)
                    with gr.Column(scale=1, min_width=0, elem_classes=["status-card", "kpi-card-coral"]):
                        gr.Markdown("<div class='status-label'>Model health</div>")
                        model_health_md = gr.Markdown("Checking...", elem_classes=["status-value"])
                        model_detail_md = gr.Markdown("Connecting...", elem_classes=["status-label"])

                # Tabbed Logical Pages
                with gr.Tabs(selected="workspace", elem_classes=["snow-tabs"]) as tabs:
                    # -------------------------------------------------------------------
                    # PAGE 1: SQL WORKSPACE (Query Studio)
                    # -------------------------------------------------------------------
                    with gr.Tab("SQL Workspace", id="workspace"):
                        # Natural Language Query Prompt Card
                        with gr.Group(elem_classes=["snow-card"]):
                            gr.Markdown("### Natural Language SQL Generation")

                            # Prompt suggestion chips
                            gr.HTML("<div class='chips-header'><span class='chips-label'>Try asking:</span></div>")
                            with gr.Row(elem_classes=["snow-chips-row"]):
                                chip1 = gr.Button("Total sales by department", elem_classes=["snow-chip-btn"], size="sm", min_width=0)
                                chip2 = gr.Button("Employees hired after 2021", elem_classes=["snow-chip-btn"], size="sm", min_width=0)
                                chip3 = gr.Button("Top 5 sales by amount", elem_classes=["snow-chip-btn"], size="sm", min_width=0)
                                chip4 = gr.Button("Average salary by department", elem_classes=["snow-chip-btn"], size="sm", min_width=0)

                            question_input = gr.Textbox(
                                label="Question",
                                placeholder="Ask a question about your database (e.g., 'What is the total revenue for the year 2024?' or 'Show average salary by department')...",
                                lines=3,
                                max_lines=6,
                                elem_id="question_input",
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
                        with gr.Group(elem_classes=["terminal-panel", "snow-card"]):
                            with gr.Row():
                                with gr.Column(scale=3):
                                    gr.Markdown("#### Output Terminal")
                                with gr.Column(scale=1, min_width=140):
                                    btn_copy_sql = gr.Button(
                                        "📋 Copy SQL",
                                        elem_classes=["btn-secondary-green"],
                                        size="sm",
                                    )

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
                        with gr.Group(elem_classes=["snow-card"]):
                            with gr.Row():
                                with gr.Column(scale=3):
                                    gr.Markdown("### Set Database")
                                with gr.Column(scale=1, min_width=200):
                                    btn_back_to_workspace = gr.Button(
                                        "← Back to SQL Workspace",
                                        elem_classes=["btn-secondary-green"],
                                        size="md",
                                    )

                            # Quick sample DB loader box
                            with gr.Group(elem_classes=["snow-quick-sample-box"]):
                                with gr.Row():
                                    with gr.Column(scale=3):
                                        gr.Markdown("**Quick Start**: Test queries instantly with the pre-populated SQLite company database.")
                                    with gr.Column(scale=1, min_width=180):
                                        btn_sample_db = gr.Button(
                                            "Load Sample SQLite DB",
                                            elem_classes=["btn-primary-green"],
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
                            with gr.Group(elem_classes=["terminal-panel", "snow-logs-panel"]):
                                with gr.Row():
                                    with gr.Column(scale=3):
                                        gr.Markdown("#### Logs")
                                    with gr.Column(scale=1, min_width=110):
                                        btn_copy_logs = gr.Button(
                                            "📋 Copy",
                                            elem_classes=["btn-secondary-green"],
                                            size="sm",
                                        )
                                    with gr.Column(scale=1, min_width=110):
                                        btn_clear_logs = gr.Button(
                                            "🗑️ Clear",
                                            elem_classes=["btn-secondary-green"],
                                            size="sm",
                                        )
                                logs_terminal = gr.Textbox(
                                    label="Connection & Schema Logs",
                                    lines=8,
                                    max_lines=15,
                                    value="\n".join(get_initial_state()["logs"]),
                                    interactive=False,
                                    elem_classes=["logs-box"],
                                    elem_id="logs_terminal",
                                )

            # -------------------------------------------------------------
            # COLUMN 3: RIGHT RAIL (ACTIVITIES & NOTIFICATIONS)
            # -------------------------------------------------------------
            with gr.Column(scale=1, min_width=200, elem_classes=["snow-right-rail"]):
                gr.HTML("""
                    <div class="snow-rail-card snow-notif-card" style="cursor: pointer;" title="Click to view connection and engine status">
                        <div class="snow-rail-title">🔔 Notifications</div>
                        <div class="snow-notif-item">
                            <div class="snow-notif-dot red"></div>
                            <div class="snow-notif-body">
                                <div class="snow-notif-msg">ZeroGPU / Direct Mode</div>
                                <div class="snow-notif-meta">In-process inference active</div>
                            </div>
                        </div>
                        <div class="snow-notif-item">
                            <div class="snow-notif-dot yellow"></div>
                            <div class="snow-notif-body">
                                <div class="snow-notif-msg">Schema Cache Ready</div>
                                <div class="snow-notif-meta">Dynamic introspector active</div>
                            </div>
                        </div>
                    </div>

                    <div class="snow-rail-card snow-activity-card" style="cursor: pointer;" title="Click to jump to Activity & Logs">
                        <div class="snow-rail-title">⚡ Activities</div>
                        <div class="snow-activity-item">
                            <span class="activity-icon">📊</span>
                            <div class="activity-text">Database manager initialized</div>
                        </div>
                        <div class="snow-activity-item">
                            <span class="activity-icon">🔒</span>
                            <div class="activity-text">Read-only SQL safety guards enabled</div>
                        </div>
                        <div class="snow-activity-item">
                            <span class="activity-icon">⚡</span>
                            <div class="activity-text">Warm Snow UI active</div>
                        </div>
                    </div>

                    <div class="snow-rail-card snow-pro-tip-card" style="cursor: pointer;" title="Click to test a sample prompt chip">
                        <div class="snow-rail-title">💡 Pro Tips</div>
                        <p class="snow-tip-text">Click any query chip in the prompt studio to test without manual typing. <span style="color: var(--snow-accent); font-weight: 600;">(Try now →)</span></p>
                    </div>
                """)

        # -------------------------------------------------------------------
        # Event Bindings & Interactivity
        # -------------------------------------------------------------------

        # 1. Navigation Actions (MUST REMAIN FIRST TWO FOR TESTS fn0 and fn1)
        btn_nav_set_db.click(
            fn=lambda: gr.Tabs(selected="set_db"),
            outputs=[tabs],
            show_progress="hidden",
            js="() => { const b = document.querySelector('button[data-tab-id=\"set_db\"]'); if (b) b.click(); if (window.__sql_engine_set_nav_active) window.__sql_engine_set_nav_active('set_db'); }",
        )
        btn_back_to_workspace.click(
            fn=lambda: gr.Tabs(selected="workspace"),
            outputs=[tabs],
            show_progress="hidden",
            js="() => { const b = document.querySelector('button[data-tab-id=\"workspace\"]'); if (b) b.click(); if (window.__sql_engine_set_nav_active) window.__sql_engine_set_nav_active('workspace'); }",
        )

        # Sidebar navigation buttons
        btn_side_workspace.click(
            fn=lambda: gr.Tabs(selected="workspace"),
            outputs=[tabs],
            show_progress="hidden",
            js="() => { const b = document.querySelector('button[data-tab-id=\"workspace\"]'); if (b) b.click(); if (window.__sql_engine_set_nav_active) window.__sql_engine_set_nav_active('workspace'); }",
        )
        btn_side_set_db.click(
            fn=lambda: gr.Tabs(selected="set_db"),
            outputs=[tabs],
            show_progress="hidden",
            js="() => { const b = document.querySelector('button[data-tab-id=\"set_db\"]'); if (b) b.click(); if (window.__sql_engine_set_nav_active) window.__sql_engine_set_nav_active('set_db'); }",
        )

        # 1b. Sidebar Sample DB Quick Start (Loads sample DB, updates UI, switches to workspace)
        def handle_side_sample_quickstart(state: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any, Any, str, str, str, str, str, str, gr.Tabs, dict[str, Any]]:
            (
                db_menu, db_name, host, port, user, pwd,
                conn_st, db_md, tbl_md, logs_txt, banner_txt, updated_state
            ) = handle_load_sample(state)
            sample_q = "What is the total sales amount by department?"
            return (
                db_menu, db_name, host, port, user, pwd,
                conn_st, db_md, tbl_md, logs_txt, banner_txt,
                sample_q,
                gr.Tabs(selected="workspace"),
                updated_state,
            )

        btn_side_sample.click(
            fn=handle_side_sample_quickstart,
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
                question_input,
                tabs,
                state,
            ],
            show_progress="hidden",
            js="() => { const b = document.querySelector('button[data-tab-id=\"workspace\"]'); if (b) b.click(); if (window.__sql_engine_set_nav_active) window.__sql_engine_set_nav_active('workspace'); }",
        ).then(
            fn=lambda s: gr.update(interactive=bool(s.get("is_connected") and s.get("schema"))),
            inputs=[state],
            outputs=[btn_generate],
        )

        # 1c. Sidebar Logs Navigation (Switches to Set Database, refreshes logs, scrolls to terminal)
        def handle_side_logs_jump(state: dict[str, Any]) -> tuple[gr.Tabs, str]:
            logs_content = "\n".join(state.get("logs", []))
            return gr.Tabs(selected="set_db"), logs_content

        btn_side_logs.click(
            fn=handle_side_logs_jump,
            inputs=[state],
            outputs=[tabs, logs_terminal],
            show_progress="hidden",
            js="""() => {
                const b = document.querySelector('button[data-tab-id="set_db"]');
                if (b) b.click();
                if (window.__sql_engine_set_nav_active) window.__sql_engine_set_nav_active('logs');
                setTimeout(() => {
                    const el = document.querySelector('.logs-box') || document.querySelector('#logs_terminal');
                    if (el) {
                        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        el.style.transition = 'box-shadow 0.3s ease, border-color 0.3s ease';
                        el.style.boxShadow = '0 0 0 3px rgba(220, 38, 38, 0.45)';
                        setTimeout(() => { el.style.boxShadow = ''; }, 1500);
                    }
                }, 150);
            }""",
        )

        # Prompt suggestion chips
        chip1.click(
            fn=lambda s: ("What is the total sales amount by department?", gr.update(interactive=bool(s.get("is_connected") and s.get("schema")))),
            inputs=[state],
            outputs=[question_input, btn_generate],
        )
        chip2.click(
            fn=lambda s: ("Show all employees hired after 2021 along with their department", gr.update(interactive=bool(s.get("is_connected") and s.get("schema")))),
            inputs=[state],
            outputs=[question_input, btn_generate],
        )
        chip3.click(
            fn=lambda s: ("List the top 5 sales ordered by amount descending", gr.update(interactive=bool(s.get("is_connected") and s.get("schema")))),
            inputs=[state],
            outputs=[question_input, btn_generate],
        )
        chip4.click(
            fn=lambda s: ("Calculate average employee salary grouped by department name", gr.update(interactive=bool(s.get("is_connected") and s.get("schema")))),
            inputs=[state],
            outputs=[question_input, btn_generate],
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

        # 10. Copy and Clear Logs Actions
        btn_copy_logs.click(
            fn=lambda: "Logs copied to clipboard.",
            outputs=[status_line],
            js="() => { const box = document.querySelector('.logs-box textarea') || document.querySelector('#logs_terminal textarea'); if (box && box.value) navigator.clipboard.writeText(box.value); }",
        )

        def handle_clear_logs(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            msg = format_log_entry("Logs cleared by user.")
            state["logs"] = [msg]
            return msg, state

        btn_clear_logs.click(
            fn=handle_clear_logs,
            inputs=[state],
            outputs=[logs_terminal, state],
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

                    if (window.__sql_engine_init_theme) {
                        window.__sql_engine_init_theme();
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
    """Returns the modern Snow Dashboard UI theme with high light contrast and warm red/yellow palette."""
    return gr.themes.Default(
        primary_hue=gr.themes.colors.red,
        secondary_hue=gr.themes.colors.amber,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
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
