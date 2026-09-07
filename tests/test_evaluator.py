"""Unit tests for SQLEvaluator and SQL Evaluation Suite.

Tests cover:
- Task 01: Database resolver for Spider SQLite databases
- Task 02: Exact Match (EM) normalization & SELECT column sorting
- Task 03: Execution Accuracy (EX) multiset & column permutation comparison
- Task 04: Error categorization (SyntaxError, RuntimeError, EmptyResult, Timeout, SandboxViolation)
- Task 05: 3-second query timeout wrapper using concurrent.futures
- Task 08: Sandbox guardrail for read-only SELECT queries
"""

import os
import pytest

from src.evaluation.evaluator import (
    SQLEvaluator,
    SandboxViolationError,
    compare_result_sets,
    compute_exact_match,
    execute_query,
    get_db_path,
    normalize_for_em,
    normalize_sql,
    sort_select_columns,
    validate_sandbox,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DB_PATH = os.path.join(REPO_ROOT, "data/spider_data/database/concert_singer/concert_singer.sqlite")


# ---------------------------------------------------------------------------
# Task 01: Database Resolver Tests
# ---------------------------------------------------------------------------

def test_db_resolver_existing():
    """Verify that existing Spider databases are resolved properly."""
    path = get_db_path("concert_singer")
    assert os.path.exists(path)
    assert path.endswith("concert_singer.sqlite")


def test_db_resolver_missing():
    """Verify that non-existent database throws FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        get_db_path("non_existent_db_12345")


# ---------------------------------------------------------------------------
# Task 02: Exact Match Normalization & Column Sorting Tests
# ---------------------------------------------------------------------------

def test_normalize_sql_fences_and_comments():
    raw = "```sql\n-- get singers\nSELECT name FROM singer; /* trailing */\n```"
    normalized = normalize_sql(raw)
    assert normalized == "SELECT name FROM singer"


def test_normalize_sql_backticks_and_spacing():
    raw = "SELECT  `name`  ,  `country`   FROM   `singer`  ;"
    normalized = normalize_sql(raw)
    assert normalized == "SELECT country, name FROM singer"


def test_sort_select_columns_basic():
    q1 = "SELECT name, country, age FROM singer"
    sorted_q1 = sort_select_columns(q1)
    assert sorted_q1 == "SELECT age, country, name FROM singer"


def test_sort_select_columns_distinct():
    q = "SELECT DISTINCT country, age FROM singer"
    sorted_q = sort_select_columns(q)
    assert sorted_q == "SELECT DISTINCT age, country FROM singer"


def test_sort_select_columns_nested_subquery():
    q = "SELECT b, a FROM (SELECT y, x FROM t) WHERE z > (SELECT max(age), min(age) FROM p)"
    sorted_q = sort_select_columns(q)
    assert "SELECT a, b FROM" in sorted_q
    assert "SELECT x, y FROM" in sorted_q
    assert "SELECT max(age), min(age) FROM" in sorted_q


def test_exact_match_column_order_invariance():
    pred = "SELECT country, name FROM singer WHERE age > 25;"
    gold = "SELECT name, country FROM singer WHERE age > 25"
    assert normalize_for_em(pred) == normalize_for_em(gold)

    res = compute_exact_match([pred], [gold])
    assert res["exact_match"] == 100.0
    assert res["correct"] == 1
    assert res["total"] == 1


def test_exact_match_mismatch():
    pred = "SELECT name FROM singer WHERE age > 25"
    gold = "SELECT name FROM singer WHERE age > 30"
    assert normalize_for_em(pred) != normalize_for_em(gold)

    res = compute_exact_match([pred], [gold])
    assert res["exact_match"] == 0.0


# ---------------------------------------------------------------------------
# Task 03: Result Set Comparison (Execution Accuracy) Tests
# ---------------------------------------------------------------------------

def test_compare_result_sets_row_order_invariance():
    res1 = [(1, "Alice"), (2, "Bob")]
    res2 = [(2, "Bob"), (1, "Alice")]
    assert compare_result_sets(res1, res2) is True


def test_compare_result_sets_column_order_invariance():
    res1 = [(1, "Alice"), (2, "Bob")]
    res2 = [("Alice", 1), ("Bob", 2)]
    assert compare_result_sets(res1, res2) is True


def test_compare_result_sets_multiset_difference():
    # Frequency mismatch must fail
    res1 = [(1, "Alice"), (1, "Alice")]
    res2 = [(1, "Alice")]
    assert compare_result_sets(res1, res2) is False


def test_compare_result_sets_float_rounding():
    res1 = [(25.0004,)]
    res2 = [(25.0,)]
    assert compare_result_sets(res1, res2) is True


def test_compare_result_sets_empty():
    assert compare_result_sets([], []) is True
    assert compare_result_sets([(1,)], []) is False
    assert compare_result_sets(None, None) is True
    assert compare_result_sets([(1,)], None) is False


# ---------------------------------------------------------------------------
# Task 04 & 05: Safe Execution, Timeout & Error Categorization Tests
# ---------------------------------------------------------------------------

def test_execute_query_success():
    res = execute_query(TEST_DB_PATH, "SELECT count(*) FROM singer")
    assert res.success is True
    assert res.error_type is None
    assert len(res.data) == 1
    assert res.data[0][0] == 6


def test_execute_query_syntax_error():
    res = execute_query(TEST_DB_PATH, "SELECT , FROM singer")
    assert res.success is False
    assert res.error_type == "SyntaxError"
    assert "syntax error" in res.error_message.lower()

    res2 = execute_query(TEST_DB_PATH, "SELECT 1 FROM")
    assert res2.success is False
    assert res2.error_type == "SyntaxError"


def test_execute_query_runtime_error_no_table():
    res = execute_query(TEST_DB_PATH, "SELECT * FROM nonexistent_table")
    assert res.success is False
    assert res.error_type == "RuntimeError"
    assert "no such table" in res.error_message.lower()


def test_execute_query_runtime_error_no_column():
    res = execute_query(TEST_DB_PATH, "SELECT nonexistent_column FROM singer")
    assert res.success is False
    assert res.error_type == "RuntimeError"
    assert "no such column" in res.error_message.lower()


def test_execute_query_timeout():
    # An infinite recursive CTE
    infinite_query = (
        "WITH RECURSIVE r(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM r) "
        "SELECT * FROM r"
    )
    res = execute_query(TEST_DB_PATH, infinite_query, timeout=0.5)
    assert res.success is False
    assert res.error_type == "Timeout"
    assert "timed out" in res.error_message.lower()


# ---------------------------------------------------------------------------
# Task 08: Sandbox Guardrail Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_query", [
    "INSERT INTO singer VALUES (99, 'Test', 'USA', 'Song', '2020', 20, 'T')",
    "UPDATE singer SET Age = 100 WHERE Singer_ID = 1",
    "DELETE FROM singer WHERE Singer_ID = 1",
    "DROP TABLE singer",
    "ALTER TABLE singer ADD COLUMN bio TEXT",
    "CREATE TABLE hack (id INT)",
    "TRUNCATE TABLE singer",
    "-- comment\nDELETE FROM singer",
    "SELECT * FROM singer; DROP TABLE singer",
])
def test_sandbox_rejects_modifications(bad_query):
    with pytest.raises(SandboxViolationError):
        validate_sandbox(bad_query)

    # execute_query returns SandboxViolation error type
    res = execute_query(TEST_DB_PATH, bad_query, check_sandbox=True)
    assert res.success is False
    assert res.error_type == "SandboxViolation"


def test_sandbox_allows_valid_select():
    assert validate_sandbox("SELECT * FROM singer") == "SELECT * FROM singer"
    assert (
        validate_sandbox("WITH cte AS (SELECT * FROM singer) SELECT count(*) FROM cte")
        == "WITH cte AS (SELECT * FROM singer) SELECT count(*) FROM cte"
    )


# ---------------------------------------------------------------------------
# SQLEvaluator End-to-End Tests
# ---------------------------------------------------------------------------

def test_evaluator_single_empty_result():
    evaluator = SQLEvaluator()
    # Query that returns 0 rows when gold returns non-empty
    pred_sql = "SELECT name FROM singer WHERE country = 'NonExistentCountry'"
    gold_sql = "SELECT name FROM singer"
    res = evaluator.evaluate_single(pred_sql, gold_sql, db_id="concert_singer")
    assert res.em_correct is False
    assert res.ex_correct is False
    assert res.error_type == "EmptyResult"


def test_evaluator_single_success():
    evaluator = SQLEvaluator()
    pred_sql = "SELECT country, name FROM singer"
    gold_sql = "SELECT name, country FROM singer"
    res = evaluator.evaluate_single(pred_sql, gold_sql, db_id="concert_singer")
    assert res.em_correct is True
    assert res.ex_correct is True
    assert res.error_type is None


def test_evaluator_batch_with_hardness():
    evaluator = SQLEvaluator()
    preds = [
        "SELECT count(*) FROM singer",
        "SELECT country, name FROM singer",
        "SELECT * FROM non_existent",
    ]
    golds = [
        "SELECT count(*) FROM singer",
        "SELECT name, country FROM singer",
        "SELECT count(*) FROM singer",
    ]
    dbs = ["concert_singer", "concert_singer", "concert_singer"]
    hardness = ["simple", "medium", "hard"]

    batch_metrics = evaluator.evaluate_batch(preds, golds, dbs, hardness_list=hardness)
    assert batch_metrics["total"] == 3
    assert batch_metrics["exact_match"] == 66.67
    assert batch_metrics["execution_accuracy"] == 66.67
    assert "simple" in batch_metrics["by_hardness"]
    assert "medium" in batch_metrics["by_hardness"]
    assert "hard" in batch_metrics["by_hardness"]
    assert batch_metrics["by_hardness"]["simple"]["ex_percent"] == 100.0
    assert batch_metrics["by_hardness"]["hard"]["ex_percent"] == 0.0
    assert "RuntimeError" in batch_metrics["error_breakdown"]


# ---------------------------------------------------------------------------
# Deep Edge-Case Tests (Task 02, 03, 04, 06, 08)
# ---------------------------------------------------------------------------

def test_sort_select_columns_scalar_subquery():
    """Verify that scalar subqueries in SELECT do not suck adjacent columns inside."""
    q = "SELECT name, (SELECT count(*) FROM singer) FROM singer"
    sorted_q = sort_select_columns(q)
    assert sorted_q == "SELECT (SELECT count(*) FROM singer), name FROM singer"


def test_sort_select_columns_from_in_string_literal():
    """Verify that 'from' inside string literals does not break projection sorting."""
    q = "SELECT name, 'coming from home' FROM singer"
    sorted_q = sort_select_columns(q)
    assert sorted_q == "SELECT 'coming from home', name FROM singer"


def test_sort_select_columns_without_from_clause():
    """Verify that SELECT queries without a FROM clause have columns sorted properly."""
    q = "SELECT 2, 1"
    sorted_q = sort_select_columns(q)
    assert sorted_q == "SELECT 1, 2"


def test_sort_select_columns_function_with_multiple_args():
    """Verify that commas inside function arguments do not split column expressions."""
    q = "SELECT SUBSTR(name, 1, 3), age FROM singer"
    sorted_q = sort_select_columns(q)
    assert sorted_q == "SELECT age, SUBSTR(name, 1, 3) FROM singer"


def test_validate_sandbox_semicolon_in_string_literal():
    """Verify that semicolons inside string literals are not flagged as multi-statement."""
    q = "SELECT * FROM singer WHERE notes = 'Song; Album'"
    assert validate_sandbox(q) == q


def test_validate_sandbox_parenthesized_select():
    """Verify that parenthesized SELECT queries are permitted by sandbox."""
    q = "(SELECT * FROM singer)"
    assert validate_sandbox(q) == q


def test_validate_sandbox_trailing_semicolons():
    """Verify that trailing semicolons are safely handled."""
    assert validate_sandbox("SELECT * FROM singer;") == "SELECT * FROM singer;"
    assert validate_sandbox("SELECT * FROM singer;;") == "SELECT * FROM singer;;"


def test_validate_sandbox_rejects_modifications_in_cte():
    """Verify that modifying statements inside CTEs are caught and rejected."""
    q = "WITH cte AS (DELETE FROM singer RETURNING *) SELECT * FROM cte"
    with pytest.raises(SandboxViolationError):
        validate_sandbox(q)


def test_validate_sandbox_allows_keyword_inside_string():
    """Verify that prohibited keyword names inside string literals are allowed."""
    q = "SELECT * FROM singer WHERE notes = 'delete'"
    assert validate_sandbox(q) == q


def test_compare_result_sets_many_columns_no_false_positive():
    """Verify that queries with >6 columns do not trigger false positive matches."""
    res1 = [
        (1, 0, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0, 0),
    ]
    res2 = [
        (1, 0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0),
    ]
    assert compare_result_sets(res1, res2) is False


def test_compare_result_sets_many_columns_true_match():
    """Verify that permuted columns with >6 columns correctly match."""
    res1 = [
        (1, 0, 0, 0, 0, 0, 2),
        (0, 1, 0, 0, 0, 0, 3),
    ]
    # Swap column 0 and column 6
    res2 = [
        (2, 0, 0, 0, 0, 0, 1),
        (3, 1, 0, 0, 0, 0, 0),
    ]
    assert compare_result_sets(res1, res2) is True


def test_compare_result_sets_nan_handling():
    """Verify that float NaN values compare as equal in result sets."""
    res1 = [(float("nan"), 1)]
    res2 = [(1, float("nan"))]
    assert compare_result_sets(res1, res2) is True


def test_compare_result_sets_string_with_leading_zero():
    """Verify that strings with leading zeroes (e.g. zip codes) are not coerced to ints."""
    res1 = [("0123",)]
    res2 = [("123",)]
    assert compare_result_sets(res1, res2) is False


def test_evaluator_gold_exec_failure():
    """Verify that failure to execute gold SQL is reported as an error."""
    evaluator = SQLEvaluator()
    res = evaluator.evaluate_single(
        pred_sql="SELECT count(*) FROM singer",
        gold_sql="SELECT nonexistent FROM broken_table",
        db_id="concert_singer",
    )
    assert res.ex_correct is False
    assert res.error_type == "RuntimeError"
    assert "Gold SQL execution failed" in res.error_message


def test_load_eval_data_from_dev_json():
    """Verify that Spider dev.json can be loaded directly with complexity classification."""
    from scripts.run_eval import load_eval_data
    records = load_eval_data("data/spider_data/dev.json", max_samples=5)
    assert len(records) == 5
    assert isinstance(records[0]["sql"], str)
    assert records[0]["sql"].startswith("SELECT")
    assert records[0]["hardness"] in ("simple", "medium", "hard", "extra-hard")

