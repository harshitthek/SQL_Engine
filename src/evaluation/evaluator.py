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
            cols = [c.strip() for c in cols_str.split(",") if c.strip()]
            unmasked_cols = [unmask(c) for c in cols]
            sorted_cols = sorted(unmasked_cols, key=str.lower)
            return prefix + ", ".join(sorted_cols) + " "

        replaced = pattern.sub(repl, masked)
        return unmask(replaced)

    result = process_single_level(s)
    return " ".join(result.split())


def normalize_sql(query: str, sort_columns: bool = True) -> str:
    """Normalize SQL query for fair comparison.
    
    - Strips markdown code blocks
    - Strips comments
    - Strips trailing semicolons and whitespace
    - Normalizes internal whitespace
    - Removes backticks around identifiers
    - Normalizes commas and operators
    - Sorts SELECT columns (if sort_columns is True)
    """
    if not query:
        return ""

    # Strip markdown fences
    query = re.sub(r"^```(?:sql)?\s*", "", query.strip(), flags=re.IGNORECASE)
    query = re.sub(r"\s*```$", "", query.strip())

    # Strip comments
    query = re.sub(r"--.*$", "", query, flags=re.MULTILINE)
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)

    # Strip trailing semicolon and whitespace
    query = query.strip().rstrip(";").strip()

    # Remove backticks
    query = query.replace("`", "")

    # Normalize whitespace before commas
    query = re.sub(r"\s*,\s*", ", ", query)

    # Normalize internal whitespace
    query = " ".join(query.split())

    # Sort columns if requested
    if sort_columns:
        query = sort_select_columns(query)
        query = " ".join(query.split())

    return query.strip()


def normalize_for_em(query: str) -> str:
    """Normalize SQL query for Exact Match comparison (lowercased, columns sorted)."""
    norm = normalize_sql(query, sort_columns=True).lower()
    norm = re.sub(r"\s*,\s*", ", ", norm)
    norm = re.sub(r"\(\s+", "(", norm)
    norm = re.sub(r"\s+\)", ")", norm)
    return " ".join(norm.split())


def compute_exact_match(
    predictions: List[str], references: List[str]
) -> Dict[str, Any]:
    """Compute Exact Match (EM) percentage between normalized predictions and labels."""
    if not predictions or not references or len(predictions) != len(references):
        return {"exact_match": 0.0, "total": 0, "correct": 0}

    correct = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        norm_pred = normalize_for_em(pred)
        norm_ref = normalize_for_em(ref)
        if norm_pred == norm_ref:
            correct += 1

    em = (correct / total * 100.0) if total > 0 else 0.0
    return {
        "exact_match": round(em, 2),
        "total": total,
        "correct": correct,
    }


# ---------------------------------------------------------------------------
# Task 03: Result Set Comparison (Execution Accuracy)
# ---------------------------------------------------------------------------

def normalize_val(val: Any) -> Any:
    """Normalize individual SQL result values for fair comparison."""
    if val is None:
        return None
    if isinstance(val, float):
        return "__SQL_NAN__" if math.isnan(val) else round(val, 3)
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        cleaned = val.strip()
        if not any(c.isdigit() for c in cleaned):
            return cleaned.lower()
        if cleaned.isdigit() and len(cleaned) > 1 and cleaned.startswith("0"):
            return cleaned
        try:
            f = float(cleaned)
            return round(f, 3) if "." in cleaned else int(f)
        except ValueError:
            return cleaned.lower()
    return val


def compare_result_sets(
    res1: Optional[List[Tuple[Any, ...]]],
    res2: Optional[List[Tuple[Any, ...]]],
) -> bool:
    """Compare two SQLite result sets regardless of row and column order.
    
    Treats results as a multiset (bag) of rows. Checks all column permutations
    so that column reordering (e.g. SELECT a, b vs SELECT b, a) is treated as a match.
    Uses column multiset pruning to achieve exact matching efficiently without false positives.
    """
    if res1 is None and res2 is None:
        return True
    if res1 is None or res2 is None:
        return False
    if len(res1) != len(res2):
        return False
    if len(res1) == 0 and len(res2) == 0:
        return True
    if len(res1[0]) != len(res2[0]):
        return False

    num_cols = len(res1[0])
    norm1 = [tuple(normalize_val(x) for x in r) for r in res1]
    norm2 = [tuple(normalize_val(x) for x in r) for r in res2]

    target_counter = Counter(norm2)

    # Fast path: columns are already in identical order
    if Counter(norm1) == target_counter:
        return True

    # If single column and did not match above, no permutation will help
    if num_cols == 1:
        return False

    # Extract column multisets across all rows to prune the search space
    col_counts1 = [Counter(row[j] for row in norm1) for j in range(num_cols)]
    col_counts2 = [Counter(row[i] for row in norm2) for i in range(num_cols)]

    # Candidate source column indices in norm1 for each target column position i in norm2
    candidates = []
    for i in range(num_cols):
        cands = [j for j in range(num_cols) if col_counts1[j] == col_counts2[i]]
        if not cands:
            return False
        candidates.append(cands)

    def search(target_idx: int, used_src: set, perm: list) -> bool:
        if target_idx == num_cols:
            perm_counter = Counter(tuple(row[perm[k]] for k in range(num_cols)) for row in norm1)
            return perm_counter == target_counter
        for src_col in candidates[target_idx]:
            if src_col not in used_src:
                used_src.add(src_col)
                perm.append(src_col)
                if search(target_idx + 1, used_src, perm):
                    return True
                perm.pop()
                used_src.remove(src_col)
        return False

    return search(0, set(), [])


# ---------------------------------------------------------------------------
# Task 04 & 05: Safe Execution with Timeout & Error Categorization
# ---------------------------------------------------------------------------

def execute_query(
    db_path: str,
    query: str,
    timeout: float = 3.0,
    check_sandbox: bool = True,
) -> ExecutionResult:
    """Execute SQL query against SQLite database with timeout and sandboxing.
    
    Wraps execution in concurrent.futures with a hard timeout and connection interruption.
    Opens DB in read-only mode. Categorizes errors into SyntaxError, RuntimeError, Timeout, SandboxViolation.
    """
    t0 = time.time()

    if check_sandbox:
        try:
            query = validate_sandbox(query)
        except SandboxViolationError as e:
            return ExecutionResult(
                success=False,
                error_type="SandboxViolation",
                error_message=str(e),
                execution_time=round(time.time() - t0, 4),
            )

    conn_box = [None]

    def _worker():
        abs_db = os.path.abspath(db_path)
        uri = f"file:{abs_db}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn_box[0] = conn
        try:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            return rows
        finally:
            conn.close()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_worker)
    try:
        data = future.result(timeout=timeout)
        executor.shutdown(wait=False)
        return ExecutionResult(
            success=True,
            data=data,
            execution_time=round(time.time() - t0, 4),
        )
    except concurrent.futures.TimeoutError:
        conn = conn_box[0]
        if conn is not None:
            try:
                conn.interrupt()
            except Exception:
                pass
        executor.shutdown(wait=False, cancel_futures=True)
        return ExecutionResult(
            success=False,
            error_type