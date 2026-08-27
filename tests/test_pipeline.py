"""Unit tests for pipeline."""
from data.processing.complexity import eval_hardness

def test_complexity_classifier():
    assert eval_hardness({"select": [False, [[3, [0, [0, 0, False], None]]]]}) == "simple"
