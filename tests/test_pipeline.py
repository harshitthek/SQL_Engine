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
        "intersect": {"select": [False, []], "from": {"table_units": [], "conds": []}, "where": [], "groupBy": [], "having": [], "orderBy": [], "limit": None, "intersect": None, "un