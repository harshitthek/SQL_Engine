"""Unit tests for the Text-to-SQL data preparation pipeline."""

import os
import pytest
from datasets import load_from_disk

from data.processing.complexity import eval_hardness
from data.processing.prompt_templates import TEMPLATES
from data.processing.schema_serializer import SchemaSerializer
from data.processing.tokenizer_utils import QwenTokenizerWrapper


def test_complexity_classifier():
    # Simple query
    simple_sql = {
        "select": [False, [[3, [0, [0, 0, False], None]]]],
        "from": {"table_units": [["table_unit", 1]], "conds": []},
        "where": [[False, 3, [0, [0, 10, False], None], 56.0, None]],
        "groupBy": [],
        "having": [],
        "orderBy": [],
        "limit": None,
        "intersect": None,
        "union": None,
        "except": None,
    }
    assert eval_hardness(simple_sql) == "simple"

    # Complex query with subquery/intersect
    complex_sql = {
        "select": [False, [[0, [0, [0, 1, False], None]]]],
        "from": {"table_units": [["table_unit", 0], ["table_unit", 1]], "conds": []},
        "where": [],
        "groupBy": [[0, 1, False]],
        "having": [],
        "orderBy": ("asc", [[0, [0, 2, False], None]]),
        "limit": 1,
        "intersect": {"select": [False, []], "from": {"table_units": [], "conds": []}, "where": [], "groupBy": [], "having": [], "orderBy": [], "limit": None, "intersect": None, "union": None, "except": None},
        "union": None,
        "except": None,
    }
    assert eval_hardness(complex_sql) in ("hard", "extra-hard")


def test_schema_serializer():
    serializer = SchemaSerializer(
        tables_json_path="data/spider_data/tables.json",
        db_root_dir="data/spider_data/database",
    )
    assert "perpetrator" in serializer.get_db_ids()
    ddl = serializer.serialize_ddl("perpetrator")
    assert "CREATE TABLE `perpetrator`" in ddl
    assert "CREATE TABLE `people`" in ddl
    assert "PRIMARY KEY" in ddl
    assert "FOREIGN KEY" in ddl


def test_prompt_templates():
    schema = "CREATE TABLE `users` (`id` NUMBER, `name` TEXT);"
    question = "List all user names."
    sql = "SELECT name FROM users;"

    for name in ["markdown", "chatml", "code_comment"]:
        template = TEMPLATES[name]
        full_p = template.format_full(schema, question, sql)
        input_p = template.format_input(schema, question)
        assert question in full_p
        assert sql in full_p
        assert question in input_p
        assert sql not in input_p


def test_arrow_dataset_loading():
    arrow_dir = "data/processed_arrow"
    assert os.path.exists(arrow_d