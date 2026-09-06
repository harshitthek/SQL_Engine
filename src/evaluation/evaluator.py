"""SQL Evaluation Suite for Text-to-SQL.

Implements:
- Task 01: Database resolver for Spider SQLite databases
- Task 02: Exact Match (EM) metric with column sorting and whitespace/case normalization
- Task 03: Execution Accuracy (EX) comparing result sets regardless of row/column order
- Task 04: Execution error categorization (SyntaxError, RuntimeError, EmptyResult, Timeout, SandboxViolation)
- Task 05: 3-second query timeout wrapper using concurrent.futures
- Task 08: Sandbox guardrail permitting only SELECT/WITH queries and rejecting modifications
"""

from collections import Counter
from dataclasses import dataclass
import concurrent.futures
import math
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_DB_ROOTS = [
    os.path.join(REPO_ROOT, "data/spider_data/database"),
    os.path.join(REPO_ROOT, "data/spider_data/test_database"),
    "data/spider_data/database",
    "data/spider_data/test_database",
]

PROHIBITED_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "attach",
    "detach",
    "vacuum",
    "pragma",
}


class SandboxViolationError(ValueError):
    """Raised when a SQL query violates the read-only sandbox policy."""
    pass


@dataclass
class ExecutionResult:
    """Result of running a query on SQLite."""
    success: bool
    data: Optional[List[Tuple[Any, ...]]] = None
    error_type: Optional[str] = None  # SyntaxError, RuntimeError, Timeout, SandboxViolation
    error_message: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class EvalItemResult:
    """Evaluation output for a single SQL example."""
    em_correct: bool
    ex_correct: bool
    error_type: Optional[str] = None  # None, SyntaxError, RuntimeError, EmptyResult, Timeout, SandboxViolation, ResultMismatch
    error_message: Optional[str] = None
    pred_sql: str = ""
    gold_sql: str = ""
    db_id: str = ""
    hardness: Optional[str] = None
    execution_time: float = 0.0


# ---------------------------------------------------------------------------
# Task 01: Database Resolver
# ---------------------------------------------------------------------------

def get_db_path(db_id: str, db_root_dirs: Optional[List[str]] = None) -> str:
    """Resolve the SQLite database file path for a given Spider db_id.
    
    Searches both train/dev database and test database directories.
    """
    candidates = []
    roots = db_root_dirs or DEFAULT_DB_ROOTS
    for root in roots:
        # Check {root}/{db_id}/{db_id}.sqlite
        candidates.append(os.path.join(root, db_id, f"{db_id}.sqlite"))
        # Check {root}/{db_id}.sqlite
        candidates.append(os.path.join(root, f"{db_id}.sqlite"))

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        f"Database '{db_id}' not found in candidate paths:\n" + "\n".join(candidates[:6])
    )


# ---------------------------------------------------------------------------
# Task 08: Sandbox Guardrail
# ---------------------------------------------------------------------------

def validate_sandbox(query: str) -> str:
    """Validate that query is safe and read-only (SELECT / WITH).
    
    Rejects any generated SQL that begins with INSERT/UPDATE/DELETE/DROP or other
    modifying/dangerous statements.
    """
    if not query or not query.strip():
        raise SandboxViolationError("Empty SQL query")

    # Strip markdown formatting
    clean_q = re.sub(r"^```(?:sql)?\s*", "", query.strip(), flags=re.IGNORECASE)
    clean_q = re.sub(r"\s*```$", "", clean_q.strip())

    # Strip comments
    clean_q = re.sub(r"--.*$", "", clean_q, flags=re.MULTILINE)
    clean_q = re.sub(r"/\*.*?\*/", "", clean_q, flags=re.DOTALL).strip()

    if not clean_q:
        raise SandboxViolationError("Empty SQL query after stripping comments")

    # Mask string literals so semicolons and keywords inside strings do not trigger rules
    strings = []
    def save_str(m):
        strings.append(m.group(0))
        return f"__STR_{len(strings)-1}__"

    str_pattern = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"")
    masked_q = str_pattern.sub(save_str, clean_q).strip()

    # Strip trailing semicolons
    masked_q = masked_q.rstrip(";").strip()

    # Check for multi-statement queries (semicolon outside quotes)
    if ";" in masked_q:
        raise SandboxViolationError("Multi-statement queries are forbidden in sandbox")

    # Find first keyword token (skipping any leading open parentheses)
    match = re.search(r"\b([a-zA-Z]+)\b", masked_q)
    if not match:
        raise SandboxViolationError("Invalid SQL structure: missing leading keyword")

    first_word = match.group(1).lower()
    if first_word in PROHIBITED_KEYWORDS:
        raise SandboxViolationError(
            f"Sandbox guardrail violation: query begins with forbidden statement '{first_word.upper()}'"
        )

    if first_word not in ("select", "with"):
        raise SandboxViolationError(
            f"Sandbox guardrail violation: evaluator only permits SELECT/WITH queries, got '{first_word.upper()}'"
        )

    # Check all word tokens outside strings for prohibited keywords (catches CTEs or subqueries containing modifications)
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", masked_q.lower()))
    violating = tokens.intersection(PROHIBITED_KEYWORDS)
    if violating:
        bad_kw = sorted(violating)[0].upper()
        raise SandboxViolationError(
            f"Sandbox guardrail violation: query contains forbidden keyword '{bad_kw}'"
        )

    return clean_q


# ---------------------------------------------------------------------------
# Task 02: Exact Match (EM) Normalization
# ---------------------------------------------------------------------------

def sort_select_columns(sql: str) -> str:
    """Sort column expressions in SELECT clauses alphabetically.
    
    Handles nested subqueries recursively, DISTINCT qualifiers, functions with commas,
    and queries without a FROM clause, preserving string literals and quotes.
    """
    strings = []
    def save_str(m):
        strings.append(m.group(0))
        return f"__STR_{len(strings)-1}__"

    str_pattern = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"")
    s = str_pattern.sub(save_str, sql)

    paren_map = {}
    paren_counter = [0]

    def mask_parens(text: str) -> str:
        while True:
            m = re.search(r"\([^()]*\)", text)
            if not m:
                break
            placeholder = f"__PAREN_{paren_counter[0]}__"
            paren_counter[0] += 1
            inner = m.group(0)[1:-1]
            if re.search(r"\bselect\b", inner, re.IGNORECASE):
                inner = process_single_level(inner)
            paren_map[placeholder] = f"({inner})"
            text = text[:m.start()] + placeholder + text[m.end():]
        return text

    def unmask(text: str) -> str:
        changed = True
        while changed:
            changed = False
            for k, v in list(paren_map.items()):
                if k in text:
                    text = text.replace(k, v)
                    changed = True
        for i, val in enumerate(strings):
            text = text.replace(f"__STR_{i}__", val)
        return text

    def process_single_level(text: str) -> str:
        masked = mask_parens(text)

        pattern = re.compile(
            r"\b(select(?:\s+distinct)?\s+)(.+?)(?=\s+(?:from|where|group\s+by|having|order\s+by|limit|union|intersect|except)\b|$)",
            re.IGNORECASE | re.DOTALL,
        )

        def repl(match):
            prefix = match.group(1)
            cols_str = match.group(2).strip()
            cols = [c.strip() for c in cols_str.split(","
# [WIP: evaluation engine]
