"""Evaluation package for Text-to-SQL."""

from .evaluator import (
    EvalItemResult,
    ExecutionResult,
    SandboxViolationError,
    SQLEvaluator,
    compare_result_sets,
    compute_exact_match,
    execute_query,
    get_db_path,
    normalize_for_em,
    normalize_sql,
    sort_select_columns,
    validate_sandbox,
)

__all__ = [
    "EvalItemResult",
    "ExecutionResult",
    "SQLEvaluator",
    "SandboxViolationError",
    "compare_result_sets",
    "compute_exact_match",
    "execute_query",
    "get_db_path",
    "normalize_for_em",
    "normalize_sql",
    "sort_select_columns",
    "validate_sandbox",
]
